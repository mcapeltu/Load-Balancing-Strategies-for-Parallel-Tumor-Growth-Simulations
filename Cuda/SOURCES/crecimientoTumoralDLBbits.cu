#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <curand.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include "random.hpp"

using namespace std;
using namespace std::chrono;
using Random = effolkronium::random_static;


//Probabilidad de reproducción identica o normal
static const double PS = 0.1;
//Tiempo al día, cálculo de migración
static const double T = (1/24.0);

//Nº máximo de reproducciones de una célula
static const double ROMAX = 10.0;
//Probabilidad máxima de muerte
static const double ALPHAMAX = 0.01;
//Calculo de migración
static const double CCT = 24.0;
//Calculo de Reproducción
static const double MU = 100.0;

//Tamaño del grid
static const int LONG = 1024;
//Nº de días
static const int NUM_DIAS = 150;
//Nº de pasos
static const int PASOS = 24;

//Vector con los valores de n!, para los numeros del 0 al 8
__device__ __managed__ int FACTORIAL[9];

//Clase que representan los elementos de la rejilla que usamos para simular el crecimiento tumoral
class Celula {
	
	public:
	//Nº de reproducciones
	double ro;
	// Probabilidades de esta célula de morir, migrar o reproducirse respectivamente
	double P_migra;
	double P_repro;	
	//Numero de vecinos cancerigenos
	int n_vecinos;
	//Referencias a los vecinos cancerigenos o NULL en otro caso
	char bitvec;
	//Indica si es cancerigena o no
	bool cancer = false;

	
	/* Inicializa la célula con los valores que se le pasa por referencia, se establecen los vecinos a 0 y se calcula las probabilidades de morir, migrar o reproducirse
	@param c: Booleano que indica si la célula es cancerígena
	@param cc: Parametro usado para el calculo de la probabilidad de migración
	@param r: Numero de reproducciones de la celula
	@param m: Parametro usado para el cálculo de la probabilidad de reproducción
	@param a: Parametro usado para el cálculo de la probabilidad de morir
	*/
	__host__ __device__ Celula (bool c, double r){
		cancer = c;
		ro = r;
		//n_vecinos = 0;
		
		//vecinos[0] = NULL;
		//vecinos[1] = NULL;
		//vecinos[2] = NULL;
		//vecinos[3] = NULL;
		//vecinos[4] = NULL;
		//vecinos[5] = NULL;
		//vecinos[6] = NULL;
		//vecinos[7] = NULL;
		
		calculoProbabilidades();
	}
	
	/* Inicializa la célula copiando los valores de la célula pasada como parámetro
	@param cell: Célula de la que se va a hacer la copia
	*/
	__host__ __device__ Celula (const Celula &cell){
		cancer = cell.cancer;
		ro = cell.ro;
		n_vecinos = cell.n_vecinos;
		P_migra = cell.P_migra;
		P_repro = cell.P_repro;
		
	}
	
	/* Inicializa la célula como no cancerígena y con todos los valores a 0
	*/
	__host__ __device__ Celula (){
		cancer = false;
		ro = 0.0;
		n_vecinos = 0;
		bitvec &= (0<<0);
		calculoProbabilidades();
	}
	
	/*Definición del operador de asignación, copia los valores de los parametros de la celula a la derecha del operador
	@param cell: Célula de la que se va a hacer la copia
	*/
	Celula& operator=( const Celula* cell){
		cancer = cell->cancer;
		ro = cell->ro;
		n_vecinos = 0;
		P_migra = cell->P_migra;
		P_repro = cell->P_repro;
		return *this;
	}
	
	/* Función que asigna los valores de cancer, cct, ro, mu y alpha en conjunto y luego recalcula las probabilidades
	@param c:  Nuevo valor de cancer
	@param cc: Nuevo valor de cct
	@param r:  Nuevo valor de ro
	@param m:  Nuevo valor de mu
	@param a:  Nuevo valor de alpha
	*/
	__device__ void assign( bool c, double r){
		cancer = c;
		ro = r;
		
		calculoProbabilidades();
	}
	
	/* Usa los valores de cct, mu y alpha para calcular las probabilidades de morir, migrar y reproducirse
	*/	
	__host__ __device__ void calculoProbabilidades(){
		double pd = 0.0; 
		if( cancer ){
			pd = (24.0/CCT)*T;
		} 
		double migrar = (1-pd)*(MU*T);
		if( ro <= 0){
			pd = 0.0;
		}
		double p_total;
		if( ro > 10 ){
			p_total = migrar + pd;
		} else {
			p_total = ALPHAMAX + migrar + pd;
		}
		//Con la suma de los valores de pd, migrar y alpha se calculan las probabilidades de reproducirse, migrar y morir respectivamente
		P_migra = migrar/p_total;
		P_repro = pd/p_total;
	}
	
	/* Reduce el valor de ro en uno y, si ha llegado a 0, se recalculan las probabilidades
	*/
	__device__ void decreaseRo(){

//		if(ro>0) ro --;	
	
		ro --;
		if(ro == 0)
			calculoProbabilidades();
	}
	
	__device__ Celula* vecino( int vec, int rand, Celula *grid, int estado ){
		if( vec == 0 ){
			return &grid[estado*LONG*LONG + rand - LONG - 1];
		} else if ( vec == 1 ){
			return &grid[estado*LONG*LONG + rand - LONG ];
		} else if ( vec == 2 ){
			return &grid[estado*LONG*LONG + rand - LONG + 1];
		} else if ( vec == 3 ){
			return &grid[estado*LONG*LONG + rand + 1];
		} else if ( vec == 4 ){
			return &grid[estado*LONG*LONG + rand + LONG + 1];
		} else if ( vec == 5 ){
			return &grid[estado*LONG*LONG + rand + LONG ];
		} else if ( vec == 6 ){
			return &grid[estado*LONG*LONG + rand + LONG - 1];
		} else if ( vec == 7 ){
			return &grid[estado*LONG*LONG + rand - 1];
		}
		return NULL;
	}
	
	__device__ double getSum(){
		return P_repro + P_migra;
	}
};


//Clase de Coordenadas, par de interos, para representar coordenadas de una matriz
struct Coordenadas{
	int x;
	int y;
	
	Coordenadas& operator=( const Coordenadas* coor){
		x = coor->x;
		y = coor->y;
		return *this;
	}
};

/*_________________________________________________________
	CUDA Declarations
  _________________________________________________________	
*/
	//Coordenadas de la primera Célula Madre 
	__device__ __managed__ Coordenadas Madre;
	
/* Kernel para inicializar el generador de números aleatorios
	@param *state: Vector de direcciones donde guardar el estado del generador para cada hilo
*/
__global__ void setup_kernel(curandState *state){	
	int id = blockIdx.x * LONG + threadIdx.x;
	// Each thread gets same seed, a different sequence number, no offset
	curand_init(1, id, 0, &state[id]);
}

/* Kernel que actualiza los vectores de vecinos y el número de vecinos de cada célula
	@param *grid:  Rejilla donde se simula el crecimiento tumoral
	@param estado: Entero 0 o 1 que nos indica en cual de los dos grid trabajamos
*/
__global__ void actualizarVecinos( Celula *grid, int estado ){
	int x = threadIdx.x;
	int y = blockIdx.x;
		
	//Reiniciamos los valores a 0 y NULL
	grid[estado*LONG*LONG + y*LONG + x].n_vecinos = 0;
	
	//Desde la esquina superior izquierda, recorriendo las 8 posiciones circundantes en sentido horario, se comprueba que no hay células cancerígenas.
	if (  x-1 >= 0  && y-1 >= 0 && grid[estado*LONG*LONG + (y-1)*LONG + x-1].cancer ){
       		grid[estado*LONG*LONG + y*LONG + x].bitvec |= (1 << 0);
       		grid[estado*LONG*LONG + y*LONG + x].n_vecinos++;
       		
	} else {
		grid[estado*LONG*LONG + y*LONG + x].bitvec &= ~(1 << 0);	
	}
	
	if ( y-1 >= 0 && grid[estado*LONG*LONG + (y-1)*LONG + x].cancer ){
       		grid[estado*LONG*LONG + y*LONG + x].bitvec |= (1 << 1);
       		grid[estado*LONG*LONG + y*LONG + x].n_vecinos++;
       		
	} else {
		grid[estado*LONG*LONG + y*LONG + x].bitvec &= ~(1 << 1);	
	}
	
	if ( y-1 >= 0 && x+1 < LONG && grid[estado*LONG*LONG + (y-1)*LONG + x+1].cancer ){
       		grid[estado*LONG*LONG + y*LONG + x].bitvec |= (1 << 2);
       		grid[estado*LONG*LONG + y*LONG + x].n_vecinos++;
       		
	} else {
		grid[estado*LONG*LONG + y*LONG + x].bitvec &= ~(1 << 2);
	}
	
	if ( x+1 < LONG && grid[estado*LONG*LONG + y*LONG + x+1].cancer ){
       		grid[estado*LONG*LONG + y*LONG + x].bitvec |= (1 << 3);
       		grid[estado*LONG*LONG + y*LONG + x].n_vecinos++;
       		
	} else {
		grid[estado*LONG*LONG + y*LONG + x].bitvec &= ~(1 << 3);	
	}
	
	if ( x+1 < LONG && y+1 < LONG && grid[estado*LONG*LONG + (y+1)*LONG + x+1].cancer ){
       		grid[estado*LONG*LONG + y*LONG + x].bitvec |= (1 << 4);
       		grid[estado*LONG*LONG + y*LONG + x].n_vecinos++;
       		
	} else {
		grid[estado*LONG*LONG + y*LONG + x].bitvec &= ~(1 << 4);	
	}
	
	if ( y+1 < LONG && grid[estado*LONG*LONG + (y+1)*LONG + x].cancer ){
       		grid[estado*LONG*LONG + y*LONG + x].bitvec |= (1 << 5);
       		grid[estado*LONG*LONG + y*LONG + x].n_vecinos++;
       		
	} else {
		grid[estado*LONG*LONG + y*LONG + x].bitvec &= ~(1 << 5);
	}
	
	if ( x-1 >= 0  && y+1 < LONG && grid[estado*LONG*LONG + (y+1)*LONG + x-1].cancer ){
       		grid[estado*LONG*LONG + y*LONG + x].bitvec |= (1 << 6);
       		grid[estado*LONG*LONG + y*LONG + x].n_vecinos++;
       		
	} else {
		grid[estado*LONG*LONG + y*LONG + x].bitvec &= ~(1 << 6);	
	}
	
	if ( x-1 >= 0  && grid[estado*LONG*LONG + y*LONG + x-1].cancer ){
       		grid[estado*LONG*LONG + y*LONG + x].bitvec |= (1 << 7);
       		grid[estado*LONG*LONG + y*LONG + x].n_vecinos++;
	} else {
		grid[estado*LONG*LONG + y*LONG + x].bitvec &= ~(1 << 7);	
	}
}


/* Función que añade la probabilidad de que uno de los vecinos no ocurra, y se llama a si misma para añadir el resto de vecinos, partiendo que la célula actual no es cancerígena
	@param elegido: Número del vecino que tiene probabilidad positiva
	@param inicio: Número del vecino desde el que empezar a iterar
	@param vecinos: Número total de vecinos
	@param nivel: Número de elementos múltiplicados en este nivel
	@probabilidad_v: Vector con las probabilidades de migrar y reproducirse de los vecinos
	@return: Probabilidad de que los vecinos a partir del inicio, y sin contar el elegido, no migren ni se reproduzcan en una célula vecina
*/
__device__ double combinaciones(int elegido, int inicio, int vecinos, int nivel, double * probabilidad_v){
	double p = 0.0;
	for( int j = inicio; j < vecinos; j++){
		if( j != elegido )
			p += (1-probabilidad_v[j])*( (FACTORIAL[vecinos - 1 - nivel]*FACTORIAL[nivel]) + combinaciones(elegido, j+1, vecinos, nivel+1, probabilidad_v) );
	}
	return p;
}

/* Función que añade la probabilidad de que uno de los vecinos no ocurra, y se llama a si misma para añadir el resto de vecinos, partiendo que la célula actual es cancerígena
	@param elegido: Número del vecino que tiene probabilidad positiva
	@param inicio: Número del vecino desde el que empezar a iterar
	@param vecinos: Número total de vecinos
	@param nivel: Número de elementos múltiplicados en este nivel
	@probabilidad_v: Vector con las probabilidades de migrar y reproducirse de los vecinos
	@return: Probabilidad de que los vecinos a partir del inicio, y sin contar el elegido, no migren ni se reproduzcan en una célula vecina
*/
__device__ double combinacionesCancer(int elegido, int inicio, int vecinos, int nivel, double * probabilidad_v){
	double p = 0.0;
	for( int j = inicio; j < vecinos; j++){
		if( j != elegido )
			p += (1-probabilidad_v[j])*( (FACTORIAL[vecinos - nivel]*FACTORIAL[nivel]) + combinacionesCancer(elegido, j+1, vecinos, nivel+1, probabilidad_v) );
	}
	return p;
}

/* Cálculo de la probabilidad de que los vecinos de la celula migren o se reproduzcan donde la célula se encuentra
	@param cell: Célula en la posición que estamos calculando
	@param probabilidad_f: Vector donde se van a guardar la probabilidad de cada vecino de generar una célula cancerígena
	@param cancer: Indicador de si la célula actual es cancerígena o no
	@return: Vector con los valores de las probabilidades de los vecinos de migrar o reproducirse en el sitio de la célula
*/
__device__ void calculoProbabilidadVecinos(Celula &cell, double * probabilidad_f, int cancer, int rand, Celula *grid, int estado ){
	
	double probabilidad_v[8]; //Vector donde se van a guardar las probabilidades de los vecinos
	 //Vector final con las probabilidades múltiplicadas
	int vec = 0;
	
	// Para cada vecino, si no es nulo, se cálcula la probabilidad propia de generar una célula cancerígena en la actual
	for( int j = 0 ; j < 8; j++){
		if( cell.bitvec & (1 << j) ){ 
			probabilidad_v[vec] = cell.vecino(j, rand, grid, estado)->getSum() / (8.0 - cell.vecino(j, rand, grid, estado)->n_vecinos + cancer);
			vec++;
			
		}
	}
	// Para cada vecino tumoral, se añade la probabilidad de que se genere una célula cancerígena en la actual.
	for( int i = 0; i < cell.n_vecinos; i++){
		if(cancer == 0){
			probabilidad_f[i] = probabilidad_v[i]*(FACTORIAL[cell.n_vecinos-1] + combinaciones( i, 0, cell.n_vecinos, 1, probabilidad_v) ) / (double)FACTORIAL[cell.n_vecinos];
		} else if(cancer == 1) {
			probabilidad_f[i] = probabilidad_v[i]*(FACTORIAL[cell.n_vecinos] + combinacionesCancer( i, 0, cell.n_vecinos, 1, probabilidad_v) ) / (double)FACTORIAL[cell.n_vecinos+1];
		}
	}
	
}

/* Función de transición de cada célula, utilizando las células adyacentes se cálcula el valor en la siguiente iteración
	@param *grid:  Rejilla donde se está simulando el crecimiento tumoral
	@param estado: Entero 0 o 1 que nos indica en cual de los dos grid trabajamos
	@param *state: Vector de direcciones donde guardar el estado del generador para cada hilo  
*/
__global__ void funcionTransicion( Celula *grid, int estado, curandState *state){
	//Variables auxiliares
	double p_auxiliar, p_acumulada, p_obtenida;
	double p_vecinos[8];
	volatile int vecino;
	//Coordenadas actuales
	volatile int x = threadIdx.x;
	volatile int y = blockIdx.x;
	
	volatile int rand;
	//Coordenadas traducidas para usar en la rejilla
	volatile int cellActual;
	volatile int cellIterSig;
	
	for( int rep = 0; rep < 64; rep++){
	/*	rand = rep + y*64 + x * LONG;
		cellActual = estado*LONG*LONG + rand;
		cellIterSig = ((estado+1)%2)*LONG*LONG + rand;
	 
		rand = (y*64 + rep)*LONG + x;
		cellActual = estado*LONG*LONG + rand;
		cellIterSig = ((estado+1)%2)*LONG*LONG + rand;
		
		rand = (y + rep * 16)*LONG + x;
		cellActual = estado*LONG*LONG + rand;
		cellIterSig = ((estado+1)%2)*LONG*LONG + rand;
	*/
		rand = x*LONG + (y + rep * 16);
		cellActual = estado*LONG*LONG + rand;
		cellIterSig = ((estado+1)%2)*LONG*LONG + rand;
	
	if( grid[cellActual].cancer ){ //Si la célula es cancerígena
		if( grid[cellActual].n_vecinos > 0 ){ //Si tiene vecinos cancerígenos
			if( grid[cellActual].n_vecinos == 8){ // P=1 Quiescencia, se mantiene igual
				grid[cellIterSig].assign(true, grid[cellActual].ro);
			} else {
				//Se cálcula las probabilidades de cada vecino
				p_auxiliar = 0.0;
				calculoProbabilidadVecinos(grid[cellActual], p_vecinos, 1, rand, grid, estado);	
				for( int vec = 0; vec < grid[cellActual].n_vecinos; vec ++){
					p_auxiliar += p_vecinos[vec]; }
				
				//Se calcula la probabilidad total de la posición de que haya una célula en la iteración siguiente
				p_acumulada = grid[cellActual].P_repro + (1-grid[cellActual].P_repro)*p_auxiliar;
				p_obtenida = curand_uniform(&state[rand]);
				
				if( p_obtenida < grid[cellActual].P_repro ){ //Si la célula actual se reproduce
					grid[cellIterSig].assign(true, grid[cellActual].ro);
				} else if( p_obtenida < p_acumulada ){ //Si otra célula ocupa este sitio
					
					p_auxiliar = grid[cellActual].P_repro;
					for(int vec = 0; vec < grid[cellActual].n_vecinos; vec++){ //Vemos que vecino ocupa este sitio
						p_auxiliar += (1-grid[cellActual].P_repro)*p_vecinos[vec];
						if( p_obtenida < p_auxiliar ){
							vecino = vec;
							vec = grid[cellActual].n_vecinos;
						}
					} //Copiamos la célula 
					for(int indice_vecinos = 0; indice_vecinos < 8; indice_vecinos ++){
						if( (grid[cellActual].bitvec & (1 << indice_vecinos)) && vecino == 0){
							grid[cellIterSig].assign(grid[cellActual].vecino(indice_vecinos, rand, grid, estado)->cancer, grid[cellActual].vecino(indice_vecinos, rand, grid, estado)->ro);
							
							if( grid[cellActual].vecino(indice_vecinos, rand, grid, estado)->ro > 10 && curand_uniform(&state[rand]) > PS){
								grid[cellIterSig].assign(true, ROMAX);
							}
							indice_vecinos = 8;
							
						}else if( (grid[cellActual].bitvec & (1 << indice_vecinos)) )
							vecino --;
					}
				}else{
					grid[cellIterSig].assign(false, 0.0);
				}
			}
		} else { // Tiene cancer y no tiene vecinas cancerígenas
			//Se cálcula las probabilidades de la célula de cada acción según sus parametros
			p_obtenida = curand_uniform(&state[rand]);
			
			if( p_obtenida < grid[cellActual].P_repro ){
				//grid[cellActual].decreaseRo();
				grid[cellIterSig].assign(true, grid[cellActual].ro);
			}else{
				grid[cellIterSig].assign(false, 0.0);
			}
				
		}
	} else { // Si es célula no cancerígena
		if( grid[cellActual].n_vecinos > 0 ){ //Si tiene celulas cancerígenas alrededor
			//Se cálcula las probabilidades de cada vecino
			p_auxiliar = 0.0;
			calculoProbabilidadVecinos(grid[cellActual], p_vecinos, 0, rand, grid, estado);	
			for( int vec = 0; vec < grid[cellActual].n_vecinos; vec ++){
				p_auxiliar += p_vecinos[vec];  }
			//Se calcula la probabilidad total de la posición de que haya una célula en la iteración siguiente
			p_acumulada = p_auxiliar;
			p_obtenida = curand_uniform(&state[rand]);
			if( p_obtenida < p_acumulada ){//Si otra célula ocupa este sitio
				p_auxiliar = 0.0;
				for(int vec = 0; vec < grid[cellActual].n_vecinos; vec++){//Vemos que vecino ocupa este sitio
					p_auxiliar += p_vecinos[vec];
					if( p_obtenida < p_auxiliar ){
						vecino = vec;
						vec = grid[cellActual].n_vecinos;
					}
				} //Copiamos la célula 
				for(int indice_vecinos = 0; indice_vecinos < 8; indice_vecinos ++){
					if( (grid[cellActual].bitvec & (1 << indice_vecinos)) && vecino == 0){
						grid[cellIterSig].assign(grid[cellActual].vecino(indice_vecinos, rand, grid, estado)->cancer, grid[cellActual].vecino(indice_vecinos, rand, grid, estado)->ro);
						
						if( grid[cellActual].vecino(indice_vecinos, rand, grid, estado)->ro > 10 && curand_uniform(&state[rand]) > PS){
							grid[cellIterSig].assign(true, ROMAX);
						}
						indice_vecinos = 8;
						
					}else if( (grid[cellActual].bitvec & (1 << indice_vecinos)) )
						vecino --;
				}
			}else{
				grid[cellIterSig].assign(false, 0.0);
			}
		} else { // P = 0; Sin cancer cercano, se mantiene igual
			grid[cellIterSig].assign(false, 0.0);
		}
	}
	}
}

/*Kernel encargado de comprobar que la célula madre de la iteración pasada desaparece por el cálculo probabilístico
	@param grid: Grid de células donde se simula el crecimiento del tumor
	@param estado: Indicador de cual de las dos rejillas es la actual
*/
__global__ void comprobarMadre( Celula *grid, int estado ){
	int x = Madre.x;
	int y = Madre.y;
	if( x*y >= 0 && x < LONG && y < LONG && grid[estado*LONG*LONG + y*LONG + x].ro < 11 ){
		if( y-1 >= 0 && x-1 >= 0 && grid[estado*LONG*LONG + (y-1)*LONG + x-1].ro > 10){
			Madre.x = x-1;
			Madre.y = y-1;
		}else if( y-1 >= 0 && grid[estado*LONG*LONG + (y-1)*LONG + x].ro > 10){
			
			Madre.y = y-1;
		}else if( y-1 >= 0 && x+1 < LONG && grid[estado*LONG*LONG + (y-1)*LONG + x+1].ro > 10){
			Madre.x = x+1;
			Madre.y = y-1;
		}else if( x+1 < LONG && grid[estado*LONG*LONG + y*LONG + x+1].ro > 10){
			Madre.x = x+1;
			
		}else if( y+1 < LONG && x+1 < LONG && grid[estado*LONG*LONG + (y+1)*LONG + x+1].ro > 10){
			Madre.x = x+1;
			Madre.y = y+1;
		}else if( y+1 < LONG && grid[estado*LONG*LONG + (y+1)*LONG + x].ro > 10){
			
			Madre.y = y+1;
		}else if( y+1 < LONG && x-1 >= 0 && grid[estado*LONG*LONG + (y+1)*LONG + x-1].ro > 10){
			Madre.x = x-1;
			Madre.y = y+1;
		}else if( x-1 >= 0 && grid[estado*LONG*LONG + y*LONG + x-1].ro > 10){
			Madre.x = x-1;
			
		}else{
			grid[estado*LONG*LONG + y*LONG + x].assign(true, 100000.0);
		}		
	}
}


/* Simulación durante NUM_DIAS días, con PASOS iteracioens por día del crecimiento de un tumor en un grid.
*/
int simulacion_cancer( ){
	// Inicializamos el grid de Células
	Celula ***h_grid;
    
    	h_grid = (Celula ***) malloc((size_t)(2*sizeof(Celula **)));	
    	h_grid[0] = (Celula **) malloc((size_t)(2*LONG*sizeof(Celula *)));
    
    	h_grid[1] = h_grid[0] + LONG;
    
    	h_grid[0][0] = (Celula *) malloc((size_t)(2*LONG*LONG*sizeof(Celula))); 
    	for(int i = 1; i < LONG ; i++){
    		h_grid[0][i] = h_grid[0][i-1] + LONG;
    	}
    
    	h_grid[1][0] = h_grid[0][LONG-1] + LONG;
    	for(int i = 1; i < LONG ; i++){
    		h_grid[1][i] = h_grid[1][i-1] + LONG;
    	}	
    
    	for( int i = 0; i < LONG; i++){
    		for( int j = 0; j < LONG; j++){
    			h_grid[0][i][j] = new Celula();
    			h_grid[1][i][j] = new Celula();
    		}   
    	}
		
	//Introducimos en la mitad del grid una célula madre	
	Celula cell(true, 100000.0);
	
	Madre.x = (LONG - 1)/2;
	Madre.y = (LONG - 1)/2;
	h_grid[0][Madre.y][Madre.x] = cell;
	
	//Copiamos el grid en la GPU
    	Celula *d_grid;
    	cudaMalloc(&d_grid, 2*LONG*LONG*sizeof(Celula));
    	cudaMemcpy(d_grid, h_grid[0][0], 2*LONG*LONG*sizeof(Celula), cudaMemcpyHostToDevice);
    	
	int indice_rejilla = 0;
	int sudo_cont=0;
	
	ofstream myfile;
	string file;
	
	//const dim3 threadsPerBlock(16, 16);
	//const dim3 blockCount (64, 64);
	const unsigned int totalThreads = LONG*LONG;
	curandState *devStates;
	cudaMalloc((void **)&devStates, totalThreads * sizeof(curandState));
	
	setup_kernel<<<LONG, LONG>>>(devStates);
	
	auto new_extra = high_resolution_clock::now(); //Declaramos los valores que vamos a usar para el calculo de tiempo
	auto fin_extra = high_resolution_clock::now();
	auto extra = duration_cast<chrono::milliseconds>(fin_extra - fin_extra);
	int time = 0;
	
	
	
	//Durante 150 días
	for( int dia = 1; dia <= NUM_DIAS; dia++){
		//En cada día 24 pasos
		for (int paso = 0; paso < PASOS; paso++){
			//Actualizamos el indice de rejillas
			indice_rejilla = paso % 2;
			
			actualizarVecinos <<< LONG, LONG>>> (d_grid, indice_rejilla);
			
			funcionTransicion <<< 16, LONG>>> (d_grid, indice_rejilla, devStates);
			
			comprobarMadre <<< 1, 1 >>>( d_grid, (indice_rejilla+1)%2);
			
		}
		cudaDeviceSynchronize();
		new_extra = high_resolution_clock::now();
		
		if (dia % 5 == 0 || dia == 1){
			sudo_cont = 0;
			cudaMemcpy(h_grid[0][0], d_grid, 2*LONG*LONG*sizeof(Celula), cudaMemcpyDeviceToHost);
			file = "TABLES/" + to_string(dia) + "dia.txt";
						
			myfile.open( file );
	  		if (myfile.is_open()){
	  			myfile << LONG << "\n";
	  			for( int filas = 0; filas < LONG; filas++ ){
	  				for ( int columnas = 0; columnas < LONG; columnas++){
		  				if( h_grid[(indice_rejilla+1)%2][filas][columnas].cancer ){
			  				myfile << filas << " " << columnas << " " << h_grid[(indice_rejilla+1)%2][filas][columnas].ro << "\n";
			  				sudo_cont ++;
			  			}
		  			}
	  			}
	    			myfile.close();
	  		} else cout << "Unable to open file TABLE";
		  	
			cout << "Día: " << dia << endl;
			cout << "Numero de celulas: "<< sudo_cont << endl;  
	  	}
	  	
  		file = "TIMES/tiempos.txt";
		myfile.open( file, ios::app );
  		if (myfile.is_open()){
			myfile << sudo_cont << " " << duration_cast<milliseconds>(new_extra - fin_extra).count() << "\n";
		  	myfile.close();
  		} else cout << "Unable to open file TIME";
	  		
	  	
	  	fin_extra = high_resolution_clock::now();
	  	extra = duration_cast<milliseconds>(fin_extra - new_extra);
		time += extra.count();
		
	}
	cudaFree(devStates);
	cudaFree(d_grid);
	return time;
}



int main(){ 
	FACTORIAL[0] = 1;
	for(int i = 1; i < 9; i++){
		FACTORIAL[i] = FACTORIAL[i-1] * i;
	}	
	//Inicializamos 
	Random::seed(1);
    	
	//Simulamos el grid cálculando el tiempo que tarda
	auto start = high_resolution_clock::now(); //Declaramos los valores que vamos a usar para el calculo de tiempo
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);
	int tiempo = 0;
	
	start = high_resolution_clock::now();
	//
	tiempo -= simulacion_cancer();
	//
	stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(stop - start);
	tiempo += duration.count();
	
	cout << "Tiempo: " << tiempo << endl;
	return 0;
}

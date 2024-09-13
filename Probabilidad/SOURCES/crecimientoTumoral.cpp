#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
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

//Tamaño del grid
static const int LONG = 1024;
//Nº de días
static const int NUM_DIAS = 150;
//Nº de pasos
static const int PASOS = 24;
//Vector con los valores de n!, para los numeros del 0 al 8
static vector< int > FACTORIAL(9);

//Clase que representan los elementos de la rejilla que usamos para simular el crecimiento tumoral
class Celula {
	
	public:
	//Indica si es cancerigena o no
	bool cancer = false;
	//Cálculo de migración
	double cct;
	//Nº de reproducciones
	double ro;
	//Cálculo de Reproducción
	double mu;
	//Prababilidad de muerte
	double alpha;
	//Numero de vecinos cancerigenos
	int n_vecinos;
	//Referencias a los vecinos cancerigenos o NULL en otro caso
	Celula * vecinos[8];
	
	// Probabilidades de esta célula de morir, migrar o reproducirse respectivamente
	double P_morir;
	double P_migra;
	double P_repro;	
	
	/* Inicializa la célula con los valores que se le pasa por referencia, se establecen los vecinos a 0 y se calcula las probabilidades de morir, migrar o reproducirse
	@param c: Booleano que indica si la célula es cancerígena
	@param cc: Parametro usado para el calculo de la probabilidad de migración
	@param r: Numero de reproducciones de la celula
	@param m: Parametro usado para el cálculo de la probabilidad de reproducción
	@param a: Parametro usado para el cálculo de la probabilidad de morir
	*/
	Celula (bool c, double cc, double r, double m, double a){
		cancer = c;
		cct = cc;
		ro = r;
		mu = m;
		alpha = a;
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
	Celula (const Celula &cell){
		cancer = cell.cancer;
		cct = cell.cct;
		ro = cell.ro;
		mu = cell.mu;
		alpha = cell.alpha;
		n_vecinos = cell.n_vecinos;
		P_morir = cell.P_morir;
		P_migra = cell.P_migra;
		P_repro = cell.P_repro;
		
		vecinos[0] = NULL;
		vecinos[1] = NULL;
		vecinos[2] = NULL;
		vecinos[3] = NULL;
		vecinos[4] = NULL;
		vecinos[5] = NULL;
		vecinos[6] = NULL;
		vecinos[7] = NULL;
	}
	
	/* Inicializa la célula como no cancerígena y con todos los valores a 0
	*/
	Celula (){
		cancer = false;
		ro = 0.0;
		n_vecinos = 0;
		P_morir = 0;
		P_migra = 0;
		P_repro = 0;
		n_vecinos = 0;
		calculoProbabilidades();
		vecinos[0] = NULL;
		vecinos[1] = NULL;
		vecinos[2] = NULL;
		vecinos[3] = NULL;
		vecinos[4] = NULL;
		vecinos[5] = NULL;
		vecinos[6] = NULL;
		vecinos[7] = NULL;
	}
	
	//Definición del operador de asignación, copia los valores de los parametros de la celula a la derecha del operador
	Celula& operator=( const Celula* cell){
		cancer = cell->cancer;
		cct = cell->cct;
		ro = cell->ro;
		mu = cell->mu;
		alpha = cell->alpha;
		n_vecinos = 0;
		P_morir = cell->P_morir;
		P_migra = cell->P_migra;
		P_repro = cell->P_repro;
		vecinos[0] = NULL;
		vecinos[1] = NULL;
		vecinos[2] = NULL;
		vecinos[3] = NULL;
		vecinos[4] = NULL;
		vecinos[5] = NULL;
		vecinos[6] = NULL;
		vecinos[7] = NULL;
		return *this;
	}
	
	/* Función que asigna los valores de cancer, cct, ro, mu y alpha en conjunto y luego recalcula las probabilidades
	@param c:  Nuevo valor de cancer
	@param cc: Nuevo valor de cct
	@param r:  Nuevo valor de ro
	@param m:  Nuevo valor de mu
	@param a:  Nuevo valor de alpha
	*/
	void assign( bool c, double cc, double r, double m, double a ){
		cancer = c;
		cct = cc;
		ro = r;
		mu = m;
		alpha = a;
		
		calculoProbabilidades();
	}
	
	/* Usa los valores de cct, mu y alpha para calcular las probabilidades de morir, migrar y reproducirse
	*/	
	void calculoProbabilidades(){
		double pd = 0.0; 
		if( cct > 0 ){
			pd = (24.0/cct)*T;
		} 
		double migrar = (1-pd)*(mu*T);
		if( ro <= 0){
			pd = 0.0;
		}
		//Con la suma de los valores de pd, migrar y alpha se calculan las probabilidades de reproducirse, migrar y morir respectivamente
		double p_total = alpha + migrar + pd;
		P_morir = alpha/p_total;
		P_migra = migrar/p_total;
		P_repro = pd/p_total;
	}
	
	/* Reduce el valor de ro en uno y, si ha llegado a 0, se recalculan las probabilidades
	*/
	void decreaseRo(){
		ro --;
		if(ro == 0)
			calculoProbabilidades();
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

/* Cálcula cuales de las 8 casillas adyacentes a una dada están libres, es decir, sin células cancerigenas.
@param i: Valor x de las coordenadas de la casilla
@param j: Valor y de las coordenadas de la casilla
@param rejilla: Grid de células que se está simulando
*/	
void actualizarVecinos( int x, int y, vector< vector <Celula> > &rejilla){
	rejilla[x][y].n_vecinos = 0;
	//Desde la esquina superior izquierda, recorriendo las 8 posiciones circundantes en sentido horario, se comprueba que no hay células cancerígenas.
	if (  x-1 >= 0  && y-1 >= 0 && rejilla[x-1][y-1].cancer ){
       		rejilla[x][y].vecinos[0] = &rejilla[x-1][y-1];
       		rejilla[x][y].n_vecinos++;
       		
	} else {
		rejilla[x][y].vecinos[0] = NULL;	
	}
	
	if ( y-1 >= 0 && rejilla[x][y-1].cancer ){
       		rejilla[x][y].vecinos[1] = &rejilla[x][y-1];
       		rejilla[x][y].n_vecinos++;
       		
	} else {
		rejilla[x][y].vecinos[1] = NULL;	
	}
	
	if ( y-1 >= 0 && x+1 < LONG && rejilla[x+1][y-1].cancer ){
       		rejilla[x][y].vecinos[2] = &rejilla[x+1][y-1];
       		rejilla[x][y].n_vecinos++;
       		
	} else {
		rejilla[x][y].vecinos[2] = NULL;	
	}
	
	if ( x+1 < LONG && rejilla[x+1][y].cancer ){
       		rejilla[x][y].vecinos[3] = &rejilla[x+1][y];
       		rejilla[x][y].n_vecinos++;
       		
	} else {
		rejilla[x][y].vecinos[3] = NULL;	
	}
	
	if ( x+1 < LONG && y+1 < LONG && rejilla[x+1][y+1].cancer ){
       		rejilla[x][y].vecinos[4] = &rejilla[x+1][y+1];
       		rejilla[x][y].n_vecinos++;
       		
	} else {
		rejilla[x][y].vecinos[4] = NULL;	
	}
	
	if ( y+1 < LONG && rejilla[x][y+1].cancer ){
       		rejilla[x][y].vecinos[5] = &rejilla[x][y+1];
       		rejilla[x][y].n_vecinos++;
       		
	} else {
		rejilla[x][y].vecinos[5] = NULL;	
	}
	
	if ( x-1 >= 0  && y+1 < LONG && rejilla[x-1][y+1].cancer ){
       		rejilla[x][y].vecinos[6] = &rejilla[x-1][y+1];
       		rejilla[x][y].n_vecinos++;
       		
	} else {
		rejilla[x][y].vecinos[6] = NULL;	
	}
	
	if ( x-1 >= 0  && rejilla[x-1][y].cancer ){
       		rejilla[x][y].vecinos[7] = &rejilla[x-1][y];
       		rejilla[x][y].n_vecinos++;
	} else {
		rejilla[x][y].vecinos[7] = NULL;	
	}
}

/* Función que añade la probabilidad de que uno de los vecinos no ocurra, y se llama a si misma para añadir el resto de vecinos
	@param elegido: Número del vecino que tiene probabilidad positiva
	@param inicio: Número del vecino desde el que empezar a iterar
	@param vecinos: Número total de vecinos
	@param nivel: Número de elementos múltiplicados en este nivel
	@probabilidad_v: Vector con las probabilidades de migrar y reproducirse de los vecinos
	@return: Probabilidad de que los vecinos a partir del inicio, y sin contar el elegido, no migren ni se reproduzcan en una célula vecina
*/
double combinaciones(int elegido, int inicio, int vecinos, int nivel, vector< double > &probabilidad_v){
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
double combinacionesCancer(int elegido, int inicio, int vecinos, int nivel, vector< double > &probabilidad_v){
	double p = 0.0;
	for( int j = inicio; j < vecinos; j++){
		if( j != elegido )
			p += (1-probabilidad_v[j])*( (FACTORIAL[vecinos - nivel]*FACTORIAL[nivel]) + combinacionesCancer(elegido, j+1, vecinos, nivel+1, probabilidad_v) );
	}
	return p;
}

/* Cálculo de la probabilidad de que los vecinos de la celula migren o se reproduzcan donde la célula se encuentra
	@param cell: Célula en la posición que estamos calculando
	@return: Vector con los valores de las probabilidades de los vecinos de migrar o reproducirse en el sitio de la célula
*/
vector <double> calculoProbabilidadVecinos(Celula &cell, int cancer){
	
	vector< double > probabilidad_v; //Vector donde se van a guardar las probabilidades de los vecinos
	vector< double > probabilidad_f; //Vector final con las probabilidades múltiplicadas
	
	for( int i = 0 ; i < 8; i++){
		if( cell.vecinos[i] != NULL ){ 
			probabilidad_v.push_back( (cell.vecinos[i]->P_migra + cell.vecinos[i]->P_repro) / (8.0 - cell.vecinos[i]->n_vecinos + cancer) );
		}
	}
		
	for( int i = 0; i < cell.n_vecinos; i++){
		if(cancer == 0){
			probabilidad_f.push_back( probabilidad_v[i]*(FACTORIAL[cell.n_vecinos-1] + combinaciones( i, 0, cell.n_vecinos, 1, probabilidad_v) ) / (double)FACTORIAL[cell.n_vecinos] );
		} else if (cancer == 1){
			probabilidad_f.push_back( probabilidad_v[i]*(FACTORIAL[cell.n_vecinos] + combinacionesCancer( i, 0, cell.n_vecinos, 1, probabilidad_v) ) / (double)FACTORIAL[cell.n_vecinos+1] );
		}
	}
	
	return probabilidad_f;
}

/* Simulación durante 50 días, con 24 pasos por día del crecimiento de un tumor en un grid.
	@param matriz: Vector con todas las posibles coordenadas que tiene el grid
	@param rejilla: Grid de células donde se va a simular el crecimiento del tumor
*/
int simulacion_cancer(vector < vector< vector <Celula> > > &rejillas){
	int min_x = ((LONG - 1)/2) - 1, max_x = ((LONG - 1)/2) + 1, min_y = ((LONG - 1)/2) - 1, max_y = ((LONG - 1)/2) + 1;
	int indice_rejilla = 0;
	
	// Declaración de variables auxiliares
	vector <Coordenadas> celulas_madre;
	vector <Coordenadas> futuras;
	Coordenadas coor, Madre;
	Madre.x = (LONG - 1)/2;
	Madre.y = (LONG - 1)/2;
	celulas_madre.push_back( coor );
	
	int i, j, indice_libre, vecino, contador = 1, pasos = 24;
	Celula cell, nueva_cell;
	vector<int> cas_libres;
	vector<double> p_vecinos;
	double alpha, pd, migrar, p_obtenida, p_acumulada, p_auxiliar;
	double p_migrar, p_morir, p_repro;
	ofstream myfile;
	string file;
	vector <Coordenadas>::iterator it;
	
	auto new_extra = high_resolution_clock::now(); //Declaramos los valores que vamos a usar para el calculo de tiempo
	auto fin_extra = high_resolution_clock::now();
	auto extra = duration_cast<chrono::milliseconds>(fin_extra - fin_extra);
	int time = 0;
	
	//Durante 50 días
	for( int dia = 1; dia <= NUM_DIAS; dia++){
		//En cada día 24 pasos
		for (int paso = 0; paso < pasos; paso++){
			//Actualizamos el indice de rejillas
			indice_rejilla = paso%2;
			
			for(int x = min_x; x <= max_x ; x++){ // Se actualizan los vecinos de cada célula
				for(int y = min_y; y <= max_y; y++){
					actualizarVecinos(x, y, rejillas[indice_rejilla]);
				}
			}
			
			for(int x = min_x; x <= max_x ; x++){ // Cálculo de función de transición de las células normales
				for(int y = min_y; y <= max_y; y++){	
					//Vemos que casilla vamos a acceder
					cell = rejillas[indice_rejilla][x][y];
					
					if( cell.cancer ){ //Si la célula es cancerígena
						if( cell.n_vecinos > 0 ){ //Si tiene casillas libres alrededor
							if( cell.n_vecinos == 8){ // P=1 Quiescencia, se mantiene igual
								rejillas[(indice_rejilla+1)%2][x][y] = cell;
							} else {
								//Se cálcula las probabilidades de cada vecino
								p_auxiliar = 0.0;
								p_vecinos = calculoProbabilidadVecinos(cell, 1);	
								for( int vec = 0; vec < p_vecinos.size(); vec ++){
									p_auxiliar += p_vecinos[vec]; }
								
								//Se calcula la probabilidad total de la posición de que haya una célula en la iteración siguiente
								p_acumulada = cell.P_repro + (1-cell.P_repro)*p_auxiliar;
								p_obtenida = Random::get(0.0, 1.0);
								
								if( p_obtenida < cell.P_repro ){ //Si la célula actual se reproduce
									rejillas[(indice_rejilla+1)%2][x][y] = cell;
								} else if( p_obtenida < p_acumulada ){ //Si otra célula ocupa este sitio
									p_auxiliar = cell.P_repro;
									for(int vec = 0; vec < p_vecinos.size(); vec++){ //Vemos que vecino ocupa este sitio
										p_auxiliar += (1-cell.P_repro)*p_vecinos[vec];
										if( p_obtenida < p_auxiliar ){
											vecino = vec;
											vec = p_vecinos.size();
										}
									} //Copiamos la célula 
									for(int indice_vecinos = 0; indice_vecinos < 8; indice_vecinos ++){
										if( cell.vecinos[indice_vecinos] != NULL && vecino == 0){
											rejillas[(indice_rejilla+1)%2][x][y] = *cell.vecinos[indice_vecinos];
											
											if( cell.vecinos[indice_vecinos]->ro > 10 && Random::get(0.0, 1.0) > PS ){
												rejillas[(indice_rejilla+1)%2][x][y] = new Celula(true, cell.vecinos[indice_vecinos]->cct, ROMAX, cell.vecinos[indice_vecinos]->mu, ALPHAMAX);
											}
											indice_vecinos = 8;
											
										}else if(cell.vecinos[indice_vecinos] != NULL)
											vecino --;
									}
								}else{
									rejillas[(indice_rejilla+1)%2][x][y] = new Celula();
									contador --;
								}
							}
						} else { // Tiene cancer y no tiene vecinas cancerígenas
							//Se cálcula las probabilidades de la célula de cada acción según sus parametros
							p_obtenida = Random::get(0.0, 1.0);
							
							if( p_obtenida < cell.P_repro ){
								rejillas[(indice_rejilla+1)%2][x][y] = cell;
							}else{
								rejillas[(indice_rejilla+1)%2][x][y] = new Celula();
								contador --;
							}
								
						}
					} else { // Si es célula no cancerígena		
						if( cell.n_vecinos > 0 ){ //Si tiene celulas cancerígenas alrededor
							//Se cálcula las probabilidades de cada vecino
							p_auxiliar = 0.0;
							p_vecinos = calculoProbabilidadVecinos(cell,0);	
							for( int vec = 0; vec < p_vecinos.size(); vec ++){
								p_auxiliar += p_vecinos[vec];  }
							//Se calcula la probabilidad total de la posición de que haya una célula en la iteración siguiente
							p_acumulada = p_auxiliar;
							p_obtenida = Random::get(0.0, 1.0);
							if( p_obtenida < p_acumulada ){//Si otra célula ocupa este sitio
								p_auxiliar = 0.0;
								for(int vec = 0; vec < p_vecinos.size(); vec++){//Vemos que vecino ocupa este sitio
									p_auxiliar += p_vecinos[vec];
									if( p_obtenida < p_auxiliar ){
										vecino = vec;
										vec = p_vecinos.size();
									}
								} //Copiamos la célula 
								for(int indice_vecinos = 0; indice_vecinos < 8; indice_vecinos ++){
									if( cell.vecinos[indice_vecinos] != NULL && vecino == 0){
										rejillas[(indice_rejilla+1)%2][x][y] = new Celula(*cell.vecinos[indice_vecinos]);
										if( cell.vecinos[indice_vecinos]->ro > 10 && Random::get(0.0, 1.0 )){
											rejillas[(indice_rejilla+1)%2][x][y] = new Celula(true, cell.vecinos[indice_vecinos]->cct, ROMAX, cell.vecinos[indice_vecinos]->mu, ALPHAMAX);
										}
										indice_vecinos = 8;					
										
									}else if(cell.vecinos[indice_vecinos] != NULL)
										vecino --;
								}
								contador ++;
								if( x == min_x && min_x != 0){
									min_x--;
								}else if( x == max_x && max_x != LONG-1){
									max_x++;
								}else if( y == min_y && min_y != 0){
									min_y--;
								}else if( y == max_y && max_y != LONG-1)
									max_y++;
							}else{
								rejillas[(indice_rejilla+1)%2][x][y] = cell;
							}
						} else { // P = 0; Sin cancer cercano, se mantiene igual
							rejillas[(indice_rejilla+1)%2][x][y] = cell;	
						}
					}
				}
			}
			
			coor.x = Madre.x;
			coor.y = Madre.y;
			if( rejillas[(indice_rejilla+1)%2][coor.x][coor.y].ro < 11 ){
				if( coor.y-1 >= 0 && coor.x-1 >= 0 && rejillas[(indice_rejilla+1)%2][coor.x-1][coor.y-1].ro > 10){
					Madre.x = coor.x-1;
					Madre.y = coor.y-1;
				}else if( coor.y-1 >= 0 && rejillas[(indice_rejilla+1)%2][coor.x][coor.y-1].ro > 10){
					
					Madre.y = coor.y-1;
				}else if( coor.y-1 >= 0 && coor.x+1 < LONG && rejillas[(indice_rejilla+1)%2][coor.x+1][coor.y-1].ro > 10){
					Madre.x = coor.x+1;
					Madre.y = coor.y-1;
				}else if( coor.x+1 < LONG && rejillas[(indice_rejilla+1)%2][coor.x+1][coor.y].ro > 10){
					Madre.x = coor.x+1;
					
				}else if( coor.y+1 < LONG && coor.x+1 < LONG && rejillas[(indice_rejilla+1)%2][coor.x+1][coor.y+1].ro > 10){
					Madre.x = coor.x+1;
					Madre.y = coor.y+1;
				}else if( coor.y+1 < LONG && rejillas[(indice_rejilla+1)%2][coor.x][coor.y+1].ro > 10){
					
					Madre.y = coor.y+1;
				}else if( coor.y+1 < LONG && coor.x-1 >= 0 && rejillas[(indice_rejilla+1)%2][coor.x-1][coor.y+1].ro > 10){
					Madre.x = coor.x-1;
					Madre.y = coor.y+1;
				}else if( coor.x-1 >= 0 && rejillas[(indice_rejilla+1)%2][coor.x-1][coor.y].ro > 10){
					Madre.x = coor.x-1;
					
				}else{
					if( rejillas[(indice_rejilla+1)%2][coor.x][coor.y].ro == 0 )
						contador++;
						
					rejillas[(indice_rejilla+1)%2][coor.x][coor.y].assign(true, 24.0, 100000.0, 100.0, 0.0);
				}		
			}
			
		}
		
		new_extra = high_resolution_clock::now();
		//Cada día indicamos cuantas células hay
		cout << "Día: " << dia << endl;
		cout << "Numero de celulas: "<< contador << ", de las cuales madre: " << celulas_madre.size() << endl;
		
		if (dia % 5 == 0 || dia == 1){
			file = "TABLES/" + to_string(dia) + "dia.txt";
						
			myfile.open( file );
	  		if (myfile.is_open()){
	  			myfile << LONG << "\n";
	  			for( int filas = 0; filas < LONG; filas++ ){
	  				for ( int columnas = 0; columnas < LONG; columnas++){
		  				if( rejillas[indice_rejilla][filas][columnas].cancer ){
			  				myfile << filas << " " << columnas << " " << rejillas[indice_rejilla][filas][columnas].ro << "\n";
			  			}
		  			}
	  			}
	    			myfile.close();
	  		} else cout << "Unable to open file";
	  	}	  	
	  	
	  	fin_extra = high_resolution_clock::now();
	  	extra = duration_cast<milliseconds>(fin_extra - new_extra);
		time += extra.count();
		
	}
	return time;
}

int calc_factorial( int n ){
	return (n == 1 || n == 0) ? 1 : calc_factorial(n - 1) * n;
}

int main(){
	FACTORIAL[0] = 1;
	for(int i = 1; i < FACTORIAL.size(); i++ )
		FACTORIAL[i] = FACTORIAL[i-1] * i;

	//Inicializamos 
	Random::seed(1);
	// Inicializamos el grid de Células
	vector < vector < vector < Celula > > > rejillas;
	rejillas.push_back( vector < vector <Celula> > (LONG) );
	rejillas.push_back( vector < vector <Celula> > (LONG) );
	for( int i = 0; i < LONG; i++){
		rejillas[0][i] = vector <Celula> (LONG);
		rejillas[1][i] = vector <Celula> (LONG);
	}
		
	//Introducimos en la mitad del grid una célula madre	
	Celula cell(true, 24.0 , 1000.0 , 100.0, 0.0);
	
	rejillas[0][(LONG - 1)/2][(LONG - 1)/2] = cell;

	
	//Simulamos el grid cálculando el tiempo que tarda
	auto start = high_resolution_clock::now(); //Declaramos los valores que vamos a usar para el calculo de tiempo
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);
	int tiempo = 0;
	
	start = high_resolution_clock::now();
	//
	tiempo -= simulacion_cancer(rejillas);
	//
	stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(stop - start);
	tiempo += duration.count();
	
	cout << "Tiempo: " << tiempo << endl;
	return 0;
}

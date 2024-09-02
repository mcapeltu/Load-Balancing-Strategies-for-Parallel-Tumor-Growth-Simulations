#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#include "random.hpp"

using namespace std;
using namespace std::chrono;
using Random = effolkronium::random_static;

// Clase Celula
struct Celula {
	//Indicar si es cancerigena o no
	bool cancer = false;
	//Cálculo de migración
	double cct;
	//Nº de reproducciones
	int ro;
	//Cálculo de Reproducción
	double mu;
	//Prababilidad de muerte
	double alpha;
	
	//Constructor con parametros, para cada uno de ellos
	Celula (bool c, double cc, int r, double m, double a){
		cancer = c;
		cct = cc;
		ro = r;
		mu = m;
		alpha = a;
	}
	
	//Constructor copia, genera una instancia nueva con los mismos parametros
	Celula (const Celula &cell){
		cancer = cell.cancer;
		cct = cell.cct;
		ro = cell.ro;
		mu = cell.mu;
		alpha = cell.alpha;
	}
	
	//Constructor por defecto, genera celula sin cancer, el resto de parametros sin inicializar
	Celula (){
		cancer = false;
	}
	
	//Definición del operador de asignación, copia los valores de los parametros de la celula a la derecha del operador
	Celula& operator=( const Celula* cell){
		cancer = cell->cancer;
		cct = cell->cct;
		ro = cell->ro;
		mu = cell->mu;
		alpha = cell->alpha;
		return *this;
	}
	
};

//Clase de Coordenadas, par de interos, para representar coordenadas de una matriz
struct Coordenadas{
	int x;
	int y;
	
	//Definición del operador de asignación, copia los valores x e y de la coordenada a la derecha del operador
	Coordenadas& operator=( const Coordenadas* coor){
		x = coor->x;
		y = coor->y;
		return *this;
	}
};

//Probabilidad de reproducción identica o normal
static double PS = 0.1;
//Tiempo al día, cálculo de migración
static double T = (1/24.0);

//Nº máximo de reproducciones de una célula
static int ROMAX = 10;
//Probabilidad máxima de muerte
static double ALPHAMAX = 0.01;

//Tamaño del grid
static int LONG = 751;
//Nº de días
static int NUM_DIAS = 171;

/* Introduce el valor de la celula, en el sitio correspondiente, alrededor de las coordenadas que se pasan
	como parametro. Al final de la función, hay una nueva celula en una casilla adyacente a las coordenadas
	que se han pasado
	@param i: Valor x de las coordenadas
	@param j: Valor y de las coordenadas
	@param casilla: Indicativo de en cuál de las 8 casillas adyacentes se ha de introducir la nueva célula, este valor debe ser válido
	@param celula: Nueva celula que se va a introducir en el grid
	@param rejilla: Grid de células que se está simulando
*/
Coordenadas introducir_en_casilla( int i, int j, int casilla, Celula celula, vector< vector <Celula> > &rejilla){
	Coordenadas coor;
	//Se comprueba cúal de los posibles 8 casillas es la que se ha pasado, y se introduce la célula en las coordenadas correspondientes
	if (casilla == 0){
        	rejilla[i - 1][j - 1] = celula;
        	coor.x = i-1;
        	coor.y = j-1;
    	}else if (casilla == 1){
        	rejilla[i][j - 1] = celula;
        	coor.x = i;
        	coor.y = j-1;
    	}
    	else if (casilla == 2){
        	rejilla[i + 1][j - 1] = celula;
        	coor.x = i+1;
        	coor.y = j-1;
    	}
    	else if (casilla == 3){
        	rejilla[i + 1][j] = celula;
        	coor.x = i+1;
        	coor.y = j;
    	}
    	else if (casilla == 4){
        	rejilla[i + 1][j + 1] = celula;
        	coor.x = i+1;
        	coor.y = j+1;
    	}
    	else if (casilla == 5){
        	rejilla[i][j + 1] = celula;
        	coor.x = i;
        	coor.y = j+1;
    	}
    	else if (casilla == 6){
        	rejilla[i - 1][j + 1] = celula;
        	coor.x = i-1;
        	coor.y = j+1;
    	}
    	else if (casilla == 7){
        	rejilla[i - 1][j] = celula;
        	coor.x = i-1;
        	coor.y = j;
    	}
    	return coor;
}

/* Calcula dada unas coordenadas cuales de las 8 casillas adyacentes están libres, es decir, sin células cancerigenas.
	@param i: Valor x de las coordenadas de la casilla
	@param j: Valor y de las coordenadas de la casilla
	@param rejilla: Grid de células que se está simulando
	@return libres: Vector con las posiciones alrededor de la casilla indicada, donde no hay células cancerígenas, indicadas con números del 0 al 7.
*/
vector<int> casillas_libres( int i, int j, vector< vector <Celula> > &rejilla){
	vector<int> libres(0); //Vector donde se guardan las posiciones
	//Desde la esquina superior izquierda, recorriendo las 8 posiciones circundantes en sentido horario, se comprueba que no hay células cancerígenas.
	if ( i-1 >= 0 && j-1 >= 0 && i-1 < LONG && j-1 < LONG && !(rejilla[i-1][j-1].cancer) )
       		libres.push_back(0);
	if ( i >= 0 && j-1 >= 0 && i < LONG && j-1 < LONG && !(rejilla[i][j-1].cancer) )
       		libres.push_back(1);
	if ( i+1 >= 0 && j-1 >= 0 && i+1 < LONG && j-1 < LONG && !(rejilla[i+1][j-1].cancer) )
       		libres.push_back(2);
	if ( i+1 >= 0 && j >= 0 && i+1 < LONG && j < LONG && !(rejilla[i+1][j].cancer) )
       		libres.push_back(3);
	if ( i+1 >= 0 && j+1 >= 0 && i+1 < LONG && j+1 < LONG && !(rejilla[i+1][j+1].cancer) )
       		libres.push_back(4);
	if ( i >= 0 && j+1 >= 0 && i < LONG && j+1 < LONG && !(rejilla[i][j+1].cancer) )
       		libres.push_back(5);
	if ( i-1 >= 0 && j+1 >= 0 && i-1 < LONG && j+1 < LONG && !(rejilla[i-1][j+1].cancer) )
       		libres.push_back(6);
	if ( i-1 >= 0 && j >= 0 && i-1 < LONG && j < LONG && !(rejilla[i-1][j].cancer) )
       		libres.push_back(7);
	
	return libres;
}

/* Simulación durante NUM_DIAS días, con 24 pasos por día del crecimiento de un tumor en un grid.
	@param matriz: Matriz divida en regiones, representadas como 3 filas, que recoge las coordenadas donde hay celulas cancerígenas
	@param futuras: Matriz dividida en filas, representando las filas de la rejilla, donde se recogen las coordenadas de la siguiente iteración
	@param rejilla: Grid de células donde se va a simular el crecimiento del tumor
*/
int simulacion_cancer(vector < vector <Coordenadas> > &matriz, vector < vector <Coordenadas> > &futuras, vector< vector <Celula> > &rejilla){
	// Declaración de variables auxiliares
	Coordenadas casilla_elegida, aux;
	int i, j, contador_cells, pasos = 24, tid, indice_futuras, indice_libre, region_cambiada;
	Celula cell, nueva_cell;
	vector<int> cas_libres;
	double alpha, pd, migrar, p_obtenida, p_total;
	ofstream myfile;
	string file;
	vector < vector <Coordenadas>::iterator > vec_iteradores( LONG );
	bool pasar;
	vector <int> distribucion(matriz.size());
	vector <double> cel_procesadas(omp_get_max_threads());
	
	//La distribución inicial de los threads, empieza por 3 filas por thread, 1 por región, para todos los threads menos los extremos
	for( int i = 1; i < (matriz.size()/3)-1; i++){
		distribucion[3*i] = 1;
		distribucion[3*i+1] = 1;
		distribucion[3*i+2] = 1;
	}
	
	//Los threads con los extremos del grid, empiezan con 2 filas, para el bot y top seam, y se reparten el resto de filas 
	distribucion[0] = 1;
	distribucion[1] = (LONG - matriz.size() + 2)/2;
	distribucion[2] = 1;
	
	distribucion[matriz.size()-3] = 1;
	distribucion[matriz.size()-2] = ( (LONG - matriz.size() + 2)/2 == 0) ? (LONG - matriz.size() + 2)/2 : (LONG - matriz.size() + 2)/2 + 1;
	distribucion[matriz.size()-1] = 1;
	
	
	auto new_extra = high_resolution_clock::now(); //Declaramos los valores que vamos a usar para el calculo de tiempo
	auto fin_extra = high_resolution_clock::now();
	auto extra = duration_cast<chrono::milliseconds>(fin_extra - fin_extra);
	int time = 0;
	
	auto new_time = high_resolution_clock::now(); //Declaramos los valores que vamos a usar para el calculo de tiempo
	auto fin_time = high_resolution_clock::now();
	int sudo_cont = 0;
	
	//Durante NUM_DIAS días
	for( int dia = 1; dia < NUM_DIAS; dia++){
		//En cada día 24 pasos
		for (int paso = 0; paso < pasos; paso++){
			new_time = high_resolution_clock::now();			
			#pragma omp parallel default( none ) shared( cel_procesadas, indice_futuras, cout, dia, paso, matriz, futuras, rejilla, contador_cells, pasos, myfile, file, new_extra, fin_extra, extra, time, vec_iteradores, T, PS, ALPHAMAX, ROMAX, pasar, distribucion ) private(region_cambiada, indice_libre, casilla_elegida, aux, i, j, tid, cell, nueva_cell, cas_libres, alpha, pd, migrar, p_obtenida, p_total)				
			{ 
				//Ponemos los iteradores al principio de los vectores de Futuras
				#pragma omp for schedule (guided)
				for( int index = 0; index < vec_iteradores.size(); index++){
					vec_iteradores[index] = futuras[index].begin();
				}
				
				//Cada thread guarda su ID
				tid = omp_get_thread_num();
				
				//Se inicia a 0 el contador de celulas por thread
				cel_procesadas[tid] = 0.0;
				
				//Para cada una de las regiones: Top seam, Center y Bot Seam
				for( int regiones = 0; regiones < 3; regiones++ ){
					region_cambiada = (regiones + paso)%3;
					//Aleatorizamos su acceso
					Random::shuffle(matriz[ tid*3 + region_cambiada ]);
					//Para cada elemento
					for( int indice = 0; indice < matriz[ tid*3 + region_cambiada ].size(); indice++){
						//Vemos que casilla vamos a acceder
						casilla_elegida = matriz[ tid*3 + region_cambiada ][indice];
						i = casilla_elegida.x; //Guardamos las coordenadas
						j = casilla_elegida.y;
						cell = rejilla[i][j]; //Y guardamos la célula en esa coordenada
						cel_procesadas[tid] += 1.0; //Contamos las celulas procesadas por hilo
						
						cas_libres = casillas_libres( i, j, rejilla ); //Cálculamos las casillas libres alrededor
						
						if( cas_libres.size() > 0 ){ //Si tiene casillas libres alrededor
							//Se cálcula las probabilidades de la célula de cada acción según sus parametros
							alpha = cell.alpha;	
							pd = (24.0/cell.cct)*T; 
							migrar = (1-pd)*(cell.mu*T);
							if( cell.ro <= 0){
								pd = 0.0;
							}
							//Se escoge qué acción va a hacer la célula
							p_total = alpha + migrar + pd;
							p_obtenida = Random::get(0.0, p_total);
							
							if( p_obtenida < alpha ){ //Si muere
								rejilla[i][j] = new Celula(); //Se reemplaza en el grid por una celula no cancerígena
							
							}else if( p_obtenida < (alpha + migrar) ){ //Si migra
								indice_libre = Random::get( 0, int(cas_libres.size()-1)); //Aleatorizamos el acceso a las casilla adyacentes
								
								nueva_cell = new Celula(cell); //Se genera una copia de la célula
								
								rejilla[i][j] = new Celula(); //Eliminamos la célula del lugar actual en el que está
								
								aux = introducir_en_casilla( i, j, cas_libres[indice_libre], nueva_cell, rejilla); //Introducimos la copia en la casilla adyacente
								(*vec_iteradores[aux.x]) = aux; 
								vec_iteradores[aux.x] ++;
								
									
							}else{ // Si se reproduce
								indice_libre = Random::get( 0, int(cas_libres.size()-1)); //Aleatorizamos el acceso a las casilla adyacentes
								
								if( cell.alpha == 0.0 ){ //Si no puede morir (Célula madre)
									p_obtenida = Random::get(0.0, 1.0); //Cálculamos si sera copia exacta o hija
									if( p_obtenida < PS ){ //Si es copia exacta
										nueva_cell = new Celula(cell); //Hacemos la copia
										aux = introducir_en_casilla( i, j, cas_libres[indice_libre], nueva_cell, rejilla); //Introducimos la copia al lado
										(*vec_iteradores[aux.x]) = aux; 
										vec_iteradores[aux.x] ++;
										
									}else{ //Si es hija
										nueva_cell = new Celula(true, cell.cct, ROMAX, cell.mu, ALPHAMAX); //Creamos una hija con nuevos parametros
										aux = introducir_en_casilla( i, j, cas_libres[indice_libre], nueva_cell, rejilla); //Introducimos la hija al lado
										(*vec_iteradores[aux.x]) = aux; 
										vec_iteradores[aux.x] ++;
									}
								} else { //Si no es célula madre
									// ESTO ESTÁ ASÍ EN EL ORIGINAL Y NO SE SI DEBERÍA ESTAR AL REVES
									cell.ro = cell.ro-1; //Reducimos el número de reproducciones de la célula
									nueva_cell = new Celula(cell); //Hacemos una copia
									aux = introducir_en_casilla( i, j, cas_libres[indice_libre], nueva_cell, rejilla); //Introducimos la hija al lado
									(*vec_iteradores[aux.x]) = aux; 
									vec_iteradores[aux.x] ++;
									
								}
								(*vec_iteradores[i]) = casilla_elegida;
								vec_iteradores[i] ++;
							}
						} else {
							(*vec_iteradores[i]) = casilla_elegida;
							vec_iteradores[i] ++;
						}
					} //Todos los hilos esperan al resto para acabar la region
					#pragma omp barrier
				}
			}
			
			pasar = false;
			//Para cada hilo menos el último
			for( int hilo = 0; hilo < matriz.size()/3-1; hilo++){
				if(!pasar){ //Si no hemos hecho un cambio antes
					//Si el hilo actual tiene suficientes filas, y el hilo actual ha procesado más de un % más de células
					if( distribucion[3*hilo+1] > 1 && cel_procesadas[hilo]/cel_procesadas[hilo+1] > 1.05 ){
						//El hilo siguiente gana una fila para procesar, y el hilo actual la pierde
						distribucion[3*hilo+1] --;
						distribucion[3*(hilo+1)+1] ++;
						pasar = true; //Se pasa la siguiente ronda
					//Si el hilo siguiente tiene suficientes filas, y el hilo actual ha procesado más de un % menos de células
					} else if( distribucion[3*(hilo+1)+1] > 1 && cel_procesadas[hilo]/cel_procesadas[hilo+1] < 0.95 ){
						//El hilo siguiente pierde una fila para procesar, y el hilo actual la gana
						distribucion[3*(hilo+1)+1] --;
						distribucion[3*hilo+1] ++;
						pasar = true;
					}
				} else	//Si has pasado antes, no pasas la siguiente iteración
					pasar = false;
			}
			
			indice_futuras = 0;
			for( int indice_matriz = 0; indice_matriz < matriz.size()/3; indice_matriz ++){
				matriz[3*indice_matriz].assign( futuras[indice_futuras].begin(), vec_iteradores[indice_futuras]);
				indice_futuras ++;
				matriz[3*indice_matriz + 1].clear();
				for( int a = 0; a < distribucion[3*indice_matriz+1]; a++){
					matriz[3*indice_matriz+1].insert(matriz[3*indice_matriz+1].end(), futuras[indice_futuras].begin(), vec_iteradores[indice_futuras]);
					indice_futuras ++;
				}
				matriz[3*indice_matriz+2].assign( futuras[indice_futuras].begin(), vec_iteradores[indice_futuras]);
				indice_futuras ++;
			}
			
			fin_time = high_resolution_clock::now();
			file = "TIMES/tiempos" + to_string(dia) + "dia.txt";
			sudo_cont = 0;
			myfile.open( file, ios::app );
	  		if (myfile.is_open()){
				for( int filas = 0; filas < rejilla.size(); filas++ ){
	  				for ( int columnas = 0; columnas < rejilla[filas].size(); columnas++){
		  				if( rejilla[filas][columnas].cancer ){
			  				sudo_cont ++;
			  				//filas ++;
			  			}
		  			}
	  			}
  				myfile << sudo_cont << " " << duration_cast<milliseconds>(fin_time - new_time).count() << "\n";
	    			myfile.close();
	  		} else cout << "Unable to open file TIME dia " << dia << endl ;
			
		}
		new_extra = high_resolution_clock::now();		
		//Cada día indicamos cuantas células hay
		cout << "Día: " << dia << endl;
		contador_cells = 0;
		for( int cont = 0; cont < matriz.size(); cont++){
			contador_cells += matriz[cont].size();
		}
		cout << "Numero de celulas: "<< contador_cells << endl;
		
		if (dia % 5 == 0){
			
			file = "TABLES/" + to_string(dia) + "dia.txt";
						
			myfile.open( file );
	  		if (myfile.is_open()){
	  			myfile << LONG << "\n";
	  			for( int filas = 0; filas < rejilla.size(); filas++ ){
	  				for ( int columnas = 0; columnas < rejilla[filas].size(); columnas++){
		  				if( rejilla[filas][columnas].cancer )
			  				myfile << filas << " " << columnas << " " << rejilla[filas][columnas].ro << "\n";
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


int main(){
	//Inicializamos el vector con las coordenadas
	vector < vector <Coordenadas> > matriz ( omp_get_max_threads()*3 ); // (LONG*LONG);
	vector < vector <Coordenadas> > futuras( LONG );
	Coordenadas coor;

	// Inicializamos el grid de Células
	vector < vector <Celula> > rejilla(LONG);
	for( int i = 0; i < LONG; i++){
		rejilla[i] = vector <Celula> (LONG);
		futuras[i] = vector <Coordenadas> (LONG);
	}
		
	//Introducimos en la mitad del grid una célula madre	
	Celula cell(true, 24.0 , 100 , 100.0, 0.0);
	
	rejilla[(LONG - 1)/2][(LONG - 1)/2] = cell;
	
	coor.x = (LONG - 1)/2;
	coor.y = (LONG - 1)/2;
	matriz[matriz.size()/2].push_back(coor); //Esto esta regular
	
	//Simulamos el grid cálculando el tiempo que tarda
	auto start = high_resolution_clock::now(); //Declaramos los valores que vamos a usar para el calculo de tiempo
	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>(stop - start);
	int tiempo = 0;
	
	start = high_resolution_clock::now();
	tiempo -= simulacion_cancer(matriz, futuras, rejilla);
	stop = high_resolution_clock::now();
	duration = duration_cast<milliseconds>(stop - start);
	tiempo += duration.count();
	
	cout << "Tiempo: " << tiempo << endl;
	
	return 0;
}


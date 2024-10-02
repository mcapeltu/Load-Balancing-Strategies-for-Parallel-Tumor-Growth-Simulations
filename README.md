# tfg-luisrodriguez
Código de programas sobre crecimiento tumoral simulado.

Se empezó con una primmera implementación muy simple (y lenta) del algoritmo que aparece en el trabajo "A High-Performance Cellular Automaton Model of Tumor Growth with Dynamically Growing Domains" de Jan Poleszczuk y Heiko Enderling (2014).

El ciclo de vida de una célula tumoral se puede encontrar con claridad en el artículo "Paradoxical Dependencies of Tumor Dormancy and Progression on Basic Cell Kinetics
de Heiko Enderling (2019), pag.8815.

El pseudocódigo del algoritmo en el artículo "Dynamic Load Balancing Strategy for Parallel Tumor Growth Simulations" de Salguero, Tomeu-Hardasmal y Capel (2019) :

Algorihtm ACT
Input:
nGen, nCells, cells[x][y],
W(s_N(r)->z_1 ), W'(s_N(v)->z_1 ), W''(s_N(v)->1).
Output: Tumor growth in cells simulation.
1. Assign initial tumor-seed in cells[x][y];
2. For(i=0; i<nGen; i++)
3. For(x=0; j<nCells; j++)
4. For(y=0; y<nCells; y++)
5. rr<-random( );
6. If(rr >= W(s_N(r)->z_1 ))
7. cells[x][y]=0;
8. goto(line 2);
9. If(rr < W'(s_N(v)->1))
10. PH++;
11. If(PH >= NP)
12. Pi<-random(); i=1, 2, 3, 4.
13. If(proliferation()) GoTo (line 4);
14. ElseGoTo (line 2);
15. Else
16. rrm<-random();
17. If(rrm < W''(s_N(v)->1))
18. Mi<-random(); i=1, 2, 3, 4.
19. If(migration()) GoTo (line 4)
20. ElseGoTo (line 2);
21. ReupdateCell Positions;
22. GoTo (line 1);

Todo esto se encuentra en el fichero crecimiento_tumoral.ipynb, junto con una implementación más eficiente del algoritmo.

Tras estos primeros pasos, hice la implementación del código en C++, donde pude mejorar todavía más los tiempos y eficiencia del código, esta implementación se encuentra en la carpeta 'Estandar'. Finalizado este paso, continué con la implementación de la estrategia de paralelización del código y equilibrio dinámico descritas en el artículo 'Dynamic Load Balancing Strategy for Parallel Tumor Growth Simulations' mencionado anteriormente. Este código se encuentra en la carpeta 'Paralelización'.

Para finalizar con mi trabajo pasé a la implementación de un modelo inspirado en los trabajos previos pero que me permitiera una paralelización en hardware de GPU eficiente. Para ello uso un modelo de AC con cálculo probabilístico que me permite asignarle una hebra a cada célula de la rejilla donde se simula el crecimiento tumoral, aprovechando al máximo el nivel de paralelización que me ofrece CUDA. Una primera implementación se encuentra en la carpeta 'Probabilidad', todavía en C++, y la implementación final en CUDA se encuentra en la carpeta con el mismo nombre.

En todas las carpetas hay una Makefile, donde la instrucción 'make run' ejecutará el programa principal y representará gráficamente el tumor en la carpeta 'Images'
La carpeta de Cuda contiene diferentes programas en Source, por lo que hay diferentes comandos para compilar cada uno de ellos, estos son: runDLB runbits, runDLBbits


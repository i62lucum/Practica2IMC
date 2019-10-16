//============================================================================
// Introducción a los Modelos Computacionales
// Name        : practica1.cpp
// Author      : Pedro A. Gutiérrez
// Version     :
// Copyright   : Universidad de Córdoba
//============================================================================


#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <iostream>
#include <ctime>    // Para cojer la hora time()
#include <cstdlib>  // Para establecer la semilla srand() y generar números aleatorios rand()
#include <string.h>
#include <math.h>
#include "imc/PerceptronMulticapa.h"
#include "imc/util.h"

using namespace imc;

using std::cout;
using std::endl;
using std::ofstream;
using std::cerr;

int main(int argc, char **argv) {
    // Procesar los argumentos de la línea de comandos
    bool Tflag = 0, wflag = 0, pflag = 0,tflag=0, oflag=0,sflag=0;
    char *Tvalue = NULL, *wvalue = NULL,*tvalue=NULL;
    int c,iter=1000,nOcultas=1,nNeuronas=5,decr=1,f=0;
    double eta=0.7,mu=1,val=0.0;
    opterr = 0;

    // a: opción que requiere un argumento
    // a:: el argumento requerido es opcional
    while ((c = getopt(argc, argv, "t:T:i:l:h:e:m:v:d:of:sw:p")) != -1)
    {
        // Se han añadido los parámetros necesarios para usar el modo opcional de predicción (kaggle).
        // Añadir el resto de parámetros que sean necesarios para la parte básica de las prácticas.
        switch(c){
        	case 't':
        		tflag=true;
        		tvalue=optarg;
        		break;
            case 'T':
                Tflag = true;
                Tvalue = optarg;
                break;
            case 'i':
            	iter=atoi(optarg);
            	break;
            case 'l':
                nOcultas=atoi(optarg);
            	break;
            case 'h':
                nNeuronas=atoi(optarg);
            	break;
            case 'e':
            	eta=atof(optarg);
            	break;
            case 'm':
            	mu=atof(optarg);
            	break;
            case 'v':
            	val=atof(optarg);
            	break;
            case 'd':
            	decr=atoi(optarg);
            	break;
            case 'o':
            	oflag=true;
            	break;
            case 'f':
            	f=atoi(optarg);
            	break;
            case 's':
            	sflag=true;
            	break;
            case 'w':
                wflag = true;
                wvalue = optarg;
                break;
            case 'p':
                pflag = true;
                break;

            case '?':
                if (optopt == 'T' || optopt == 'w' || optopt == 'p' || optopt=='t' || optopt=='i')
                    fprintf (stderr, "La opción -%c requiere un argumento.\n", optopt);
                else if (isprint (optopt))
                    fprintf (stderr, "Opción desconocida `-%c'.\n", optopt);
                else
                    fprintf (stderr,
                             "Caracter de opción desconocido `\\x%x'.\n",
                             optopt);
                return EXIT_FAILURE;
            default:
                return EXIT_FAILURE;
        }
    }


    if(iter<1){
    	fprintf (stderr, "El comando -i indica las iteraciones, han de ser superior a 0\n");
    	return EXIT_FAILURE;
    }
    if(nOcultas<1){
    	fprintf (stderr, "El comando -l indica las capas ocultas, han de ser superior a 0\n");
    	return EXIT_FAILURE;
    }
    if(nNeuronas<1){
    	fprintf (stderr, "El comando -l indica las neuronas en capa oculta, han de ser superior a 0\n");
    	return EXIT_FAILURE;
    }
    if(eta<0.0 || eta>1.0){
    	fprintf (stderr, "El comando -e indica la tasa de aprendizaje, ha de estar entre 0 y 1\n");
    	return EXIT_FAILURE;
    }
    if(mu<0.0 || mu>1.0){
    	fprintf (stderr, "El comando -m indica el momento, ha de estar entre 0 y 1\n");
    	return EXIT_FAILURE;
    }
    if(val<0.0 || val>1.0){
    	fprintf (stderr, "El comando -v indica el porcentaje de validación, ha de estar entre 0 y 1\n");
    	return EXIT_FAILURE;
    }
    if(decr<1){
    	fprintf (stderr, "El comando -d indica el decremento por capa, ha ser 1 o superior\n");
    	return EXIT_FAILURE;
    }
    if(f!=0 && f!=1){
    	fprintf (stderr, "El comando -f indica la función de la última capa, 0 sigmoide, 1 softmax\n");
    	return EXIT_FAILURE;
    }


    if (!pflag) {
        if (!tflag){
        	fprintf (stderr, "El programa necesita la opción -t que indica el nombre del fichero con los datos de train.\n");
        	return EXIT_FAILURE;
        }
        ////////////////////////////////////////
        // MODO DE ENTRENAMIENTO Y EVALUACIÓN //
        ///////////////////////////////////////

    	// Objeto perceptrón multicapa
    	PerceptronMulticapa mlp;
    	Datos * pDatosTrain,*pDatosTest;

        // Parámetros del mlp.
    	mlp.dEta=eta;
    	mlp.dMu=mu;
    	mlp.dDecremento=decr;
    	mlp.dValidacion=val;
    	mlp.bOnline=oflag;

    	// Lectura de datos de entrenamiento y test
    	pDatosTrain=mlp.leerDatos(tvalue);
    	if(Tflag==false){
    		pDatosTest=mlp.leerDatos(tvalue);
    	}
    	else{
    		pDatosTest=mlp.leerDatos(Tvalue);
    	}
    	int capas=nOcultas;
    	int iteraciones=iter;


        // Inicializar vector topología
        int *topologia = new int[capas+2];
        topologia[0] = pDatosTrain->nNumEntradas;
        for(int i=1; i<(capas+2-1); i++)
        	topologia[i] = nNeuronas;
        topologia[capas+2-1] = pDatosTrain->nNumSalidas;
        //Inicializar vector de tipo de función
        int *tipo = new int[capas+2];
        for(int i=0; i<(capas+2-1); i++)
        	tipo[i] = 0;
        sflag?tipo[capas+2-1] =1:tipo[capas+2-1] =0;
        // Inicializar red con vector de topología
        mlp.inicializar(capas+2,topologia,tipo);

    	//Estructuras para realizar validación.
		int *aleatorios;
		Datos * pDatosValidacion=NULL;
		Datos * pDatosTrainAux;
		if(val>0.0 && val<1.0){
			pDatosValidacion=new Datos;
			pDatosTrainAux=new Datos;
			pDatosValidacion->nNumPatrones=val * pDatosTrain->nNumPatrones;

			pDatosTrainAux->nNumPatrones=pDatosTrain->nNumPatrones - pDatosValidacion->nNumPatrones;
			pDatosValidacion->nNumEntradas=pDatosTrainAux->nNumEntradas=pDatosTrain->nNumEntradas;
			pDatosValidacion->nNumSalidas=pDatosTrainAux->nNumSalidas=pDatosTrain->nNumSalidas;

			pDatosValidacion->entradas=new double*[pDatosValidacion->nNumPatrones];
			pDatosValidacion->salidas=new double*[pDatosValidacion->nNumPatrones];

			pDatosTrainAux->entradas=new double*[pDatosTrainAux->nNumPatrones];
			pDatosTrainAux->salidas=new double*[pDatosTrainAux->nNumPatrones];
			//Se crea un vector lleno de 0 con el tamaño del numero de patrones total.
			int aux[pDatosTrain->nNumPatrones]{};
			//Se crea otro vector con valores aleatorios no repetidos que representa los patrones seleccionados para validacion.
			aleatorios=util::vectorAleatoriosEnterosSinRepeticion(0,pDatosTrain->nNumPatrones-1,pDatosValidacion->nNumPatrones);
			//Las posiciones que sean del conjunto de validacion se marcaran como 1 y el resto sera entrenamiento.
			//Para ello se recorre el vector aleatorios y se accede a la posicion correspondiente del vector aux que representa los patrones.

			for(int i=0;i<pDatosValidacion->nNumPatrones;i++){
				aux[aleatorios[i]]=1;
			}

			int countTrain=0, countVal=0;
			for(int i=0;i<pDatosTrain->nNumPatrones;i++){
				if(aux[i]==0){
					pDatosTrainAux->entradas[countTrain]=pDatosTrain->entradas[i];
					pDatosTrainAux->salidas[countTrain]=pDatosTrain->salidas[i];
					countTrain++;
				}
				else{
					pDatosValidacion->entradas[countVal]=pDatosTrain->entradas[i];
					pDatosValidacion->salidas[countVal]=pDatosTrain->salidas[i];
					countVal++;
				}
			}

			//Se libera la memoria de los datos de train ahora almacenados en pDatosTrainAux y pDatosValidacion
			delete pDatosTrain->entradas;
			delete pDatosTrain->salidas;
			delete pDatosTrain;

			//Los nuevos datos de train se vuelven a almacenar pDatosTrain para asegurar la consistencia del codigo.
			pDatosTrain=pDatosTrainAux;
		}


		ofstream file("salida.txt");
		file <<"iter"<<" "<<"trainError"<<" "<<"validationError"<<" "<<"testError"<<endl;

        // Semilla de los números aleatorios
        int semillas[] = {1,2,3,4,5};
        double *errores = new double[5];
        double *erroresTrain = new double[5];
        double *erroresValidacion = new double[5];
        double *ccrs = new double[5];
        double *ccrsTrain = new double[5];
        double mejorErrorTest = 1.0;

        for(int i=0; i<5; i++){
        	cout << "**********" << endl;
        	cout << "SEMILLA " << semillas[i] << endl;
        	cout << "**********" << endl;
    		srand(semillas[i]);


    		mlp.ejecutarAlgoritmo(pDatosTrain,pDatosTest,pDatosValidacion,iteraciones,&(erroresTrain[i]),&(errores[i]),&(erroresValidacion[i]),&(ccrsTrain[i]),&(ccrs[i]),f,file);
    		cout << "Finalizamos => CCR de test final: " << ccrs[i] << endl;

            // (Opcional - Kaggle) Guardamos los pesos cada vez que encontremos un modelo mejor.
            if(wflag && errores[i] <= mejorErrorTest)
            {
                mlp.guardarPesos(wvalue);
                mejorErrorTest = errores[i];
            }


        }


        double mediaError = 0, desviacionTipicaError = 0;
        double mediaErrorTrain = 0, desviacionTipicaErrorTrain = 0;
        double mediaErrorValidacion = 0, desviacionTipicaErrorValidacion = 0;
        double mediaCCR = 0, desviacionTipicaCCR = 0;
        double mediaCCRTrain = 0, desviacionTipicaCCRTrain = 0;

        // Calcular medias y desviaciones típicas de entrenamiento y test

        for(int i =0; i<5;i++){
        	mediaErrorTrain+=erroresTrain[i];
        	mediaError+=errores[i];
        	mediaCCR+=ccrs[i];
        	mediaCCRTrain+=ccrsTrain[i];
        	if(val>0.0 && val<1.0)
        		mediaErrorValidacion+=erroresValidacion[i];
        }
        mediaErrorTrain/=5;
        mediaError/=5;
        mediaCCR/=5;
        mediaCCRTrain/=5;
        if(val>0.0 && val<1.0)
        	mediaErrorValidacion/=5;

        for(int i=0; i<5;i++){
        	desviacionTipicaErrorTrain+= pow(erroresTrain[i]-mediaErrorTrain,2);
			desviacionTipicaError+= pow(errores[i]-mediaError,2);
			desviacionTipicaCCR+= pow(ccrs[i]-mediaCCR,2);
			desviacionTipicaCCRTrain+= pow(ccrsTrain[i]-mediaCCRTrain,2);
			if(val>0.0 && val<1.0)
				desviacionTipicaErrorValidacion+= pow(erroresValidacion[i]-mediaErrorValidacion,2);
        }
        desviacionTipicaErrorTrain/=5;
        desviacionTipicaError/=5;
        desviacionTipicaCCR/=5;
        desviacionTipicaCCRTrain/=5;
        if(val>0.0 && val<1.0)
        	desviacionTipicaErrorValidacion/=5;

        desviacionTipicaErrorTrain=sqrt(desviacionTipicaErrorTrain);
        desviacionTipicaError=sqrt(desviacionTipicaError);
        desviacionTipicaCCR=sqrt(desviacionTipicaCCR);
        desviacionTipicaCCRTrain=sqrt(desviacionTipicaCCRTrain);
        if(val>0.0 && val<1.0)
        	desviacionTipicaErrorValidacion=sqrt(desviacionTipicaErrorValidacion);


        cout << "HEMOS TERMINADO TODAS LAS SEMILLAS" << endl;

    	cout << "INFORME FINAL" << endl;
    	cout << "*************" << endl;
        cout << "Error de entrenamiento (Media +- DT): " << mediaErrorTrain << " +- " << desviacionTipicaErrorTrain << endl;
        cout << "Error de test (Media +- DT): " << mediaError << " +- " << desviacionTipicaError << endl;
        if(val>0.0 && val<1.0)
               	cout << "Error de validacion (Media +- DT): " << mediaErrorValidacion << " +- " << desviacionTipicaErrorValidacion<<endl;
        cout << "CCR de entrenamiento (Media +- DT): " << mediaCCRTrain << " +- " << desviacionTipicaCCRTrain << endl;
        cout << "CCR de test (Media +- DT): " << mediaCCR << " +- " << desviacionTipicaCCR << endl;
    	return EXIT_SUCCESS;
    } else {

        /////////////////////////////////
        // MODO DE PREDICCIÓN (KAGGLE) //
        ////////////////////////////////

        // Desde aquí hasta el final del fichero no es necesario modificar nada.
        
        // Objeto perceptrón multicapa
        PerceptronMulticapa mlp;

        // Inicializar red con vector de topología
        if(!wflag || !mlp.cargarPesos(wvalue))
        {
            cerr << "Error al cargar los pesos. No se puede continuar." << endl;
            exit(-1);
        }

        // Lectura de datos de entrenamiento y test: llamar a mlp.leerDatos(...)
        Datos *pDatosTest;
        pDatosTest = mlp.leerDatos(Tvalue);
        if(pDatosTest == NULL)
        {
            cerr << "El conjunto de datos de test no es válido. No se puede continuar." << endl;
            exit(-1);
        }

        mlp.predecir(pDatosTest);

        return EXIT_SUCCESS;

    }
}


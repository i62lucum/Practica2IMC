/*********************************************************************
 * File  : PerceptronMulticapa.cpp
 * Date  : 2018
 *********************************************************************/

#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <cstdlib>  // Para establecer la semilla srand() y generar números aleatorios rand()
#include <limits>
#include <math.h>

#include "PerceptronMulticapa.h"
#include "util.h"

using namespace imc;
using namespace std;
using namespace util;


// ------------------------------
// CONSTRUCTOR: Dar valor por defecto a todos los parámetros (dEta, dMu, dValidacion y dDecremento)
PerceptronMulticapa::PerceptronMulticapa(){
	this->bOnline=true;
	this->nNumPatronesTrain=0;
	this->nNumCapas=0;
	this->pCapas=NULL;
	this->dEta=0.0;
	this->dDecremento=0.0;
	this->dValidacion=0.0;
	this->dMu=0.0;
}

// ------------------------------
// Reservar memoria para las estructuras de datos
// nl tiene el numero de capas y npl es un vector que contiene el número de neuronas por cada una de las capas
// tipo contiene el tipo de cada capa (0 => sigmoide, 1 => softmax)
// Rellenar vector Capa* pCapas
int PerceptronMulticapa::inicializar(int nl, int npl[], int tipo[]) {
	this->nNumCapas=nl;
	this->pCapas=new Capa[nl];
	//Bucle que reserva las neuronas de cada capa
	for(int h=0;h<nl;h++){
		this->pCapas[h].nNumNeuronas=npl[h];
		this->pCapas[h].nNumNeuronas=tipo[h];
		this->pCapas[h].pNeuronas=new Neurona[npl[h]];
		//Bucle que reserva los arrays de pesos de cada neurona.
		for(int j=0; j<npl[h];j++){
			//Si la capa es la primera, no tiene ninguna entrada y por lo tanto ningun vector de pesos.
			if(h==0){
				this->pCapas[h].pNeuronas[j].deltaW=NULL;
				this->pCapas[h].pNeuronas[j].ultimoDeltaW=NULL;
				this->pCapas[h].pNeuronas[j].w=NULL;
				this->pCapas[h].pNeuronas[j].wCopia=NULL;
			}
			else{
				this->pCapas[h].pNeuronas[j].deltaW=new double[npl[h-1]+1]();
				this->pCapas[h].pNeuronas[j].ultimoDeltaW=new double[npl[h-1]+1]();
				this->pCapas[h].pNeuronas[j].w=new double[npl[h-1]+1];
				this->pCapas[h].pNeuronas[j].wCopia=new double[npl[h-1]+1];
			}
		}
	}

	return 1;
}


// ------------------------------
// DESTRUCTOR: liberar memoria
PerceptronMulticapa::~PerceptronMulticapa() {
	liberarMemoria();
}


// ------------------------------
// Liberar memoria para las estructuras de datos
void PerceptronMulticapa::liberarMemoria() {

	for(int h=1; h<this->nNumCapas;h++){
		for(int j=0; j<this->pCapas[h].nNumNeuronas; j++){
			delete this->pCapas[h].pNeuronas[j].deltaW;
			delete this->pCapas[h].pNeuronas[j].ultimoDeltaW;
			delete this->pCapas[h].pNeuronas[j].w;
			delete this->pCapas[h].pNeuronas[j].wCopia;
		}
		delete this->pCapas[h].pNeuronas;
	}
	delete this->pCapas[0].pNeuronas;
	delete this->pCapas;
}

// ------------------------------
// Rellenar todos los pesos (w) aleatoriamente entre -1 y 1
void PerceptronMulticapa::pesosAleatorios() {
	for(int h=1; h<this->nNumCapas;h++){
		for(int j=0; j<this->pCapas[h].nNumNeuronas; j++){
			for(int i=0; i<=this->pCapas[h-1].nNumNeuronas;i++){
				double nRand=((double)rand()/RAND_MAX)*2-1;
				this->pCapas[h].pNeuronas[j].w[i]=nRand;
			}
		}
	}

}

// ------------------------------
// Alimentar las neuronas de entrada de la red con un patrón pasado como argumento
void PerceptronMulticapa::alimentarEntradas(double* input) {
	for(int i=0;i<this->pCapas[0].nNumNeuronas;i++){
		this->pCapas[0].pNeuronas[i].x=input[i];
	}
}

// ------------------------------
// Recoger los valores predichos por la red (out de la capa de salida) y almacenarlos en el vector pasado como argumento
void PerceptronMulticapa::recogerSalidas(double* output){
	for(int j=0;j<this->pCapas[this->nNumCapas-1].nNumNeuronas;j++){
		output[j]=this->pCapas[this->nNumCapas-1].pNeuronas[j].x;
	}
}

// ------------------------------
// Hacer una copia de todos los pesos (copiar w en copiaW)
void PerceptronMulticapa::copiarPesos() {
	for(int h=1; h<this->nNumCapas;h++){
		for(int j=0; j<this->pCapas[h].nNumNeuronas; j++){
			for(int i=0; i<=this->pCapas[h-1].nNumNeuronas;i++){
				this->pCapas[h].pNeuronas[j].wCopia[i]=this->pCapas[h].pNeuronas[j].w[i];
			}
		}
	}
}

// ------------------------------
// Restaurar una copia de todos los pesos (copiar copiaW en w)
void PerceptronMulticapa::restaurarPesos() {
	for(int h=1; h<this->nNumCapas;h++){
		for(int j=0; j<this->pCapas[h].nNumNeuronas; j++){
			for(int i=0; i<=this->pCapas[h-1].nNumNeuronas;i++){
				this->pCapas[h].pNeuronas[j].w[i]=this->pCapas[h].pNeuronas[j].wCopia[i];
			}
		}
	}
}

// ------------------------------
// Calcular y propagar las salidas de las neuronas, desde la primera capa hasta la última
void PerceptronMulticapa::propagarEntradas() {
	//TODO Establecer la ultima capa como softmax
	double acum;
	for(int h=1; h<this->nNumCapas;h++){
		for(int j=0; j<this->pCapas[h].nNumNeuronas; j++){
			acum=0.0;
			for(int i=0; i<=this->pCapas[h-1].nNumNeuronas;i++){
				//Si z es el ultimo elemento, es el sesgo.
				if(i==this->pCapas[h-1].nNumNeuronas){
					acum+=this->pCapas[h].pNeuronas[j].w[i];
				}
				//Si no, es el peso z asociaciado a la neurona z de la anterior capa
				else{
					acum+=this->pCapas[h].pNeuronas[j].w[i]*this->pCapas[h-1].pNeuronas[i].x;
				}
			}
			//Se aplica sigmoide
			if(this->pCapas[h].tipo==0){
				this->pCapas[h].pNeuronas[j].x=1/(1+exp(-acum));
			}
			//Se aplica softmax.
			else{

			}

		}
	}
}

// ------------------------------
// Calcular el error de salida del out de la capa de salida con respecto a un vector objetivo y devolverlo
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
double PerceptronMulticapa::calcularErrorSalida(double* target, int funcionError) {
	//Se obtiene el número de neuronas de la capa de salida.
	int nCapaSalida=this->pCapas[this->nNumCapas-1].nNumNeuronas;

	//Vector que recoge la salida.
	double *salida=new double[nCapaSalida];
	this->recogerSalidas(salida);

	double error=0,aux;

	//TODO Entropía cruzada
	if(funcionError==1){

	}
	//MSE
	else{
		for(int j=0;j<nCapaSalida;j++){
			aux=salida[j]-target[j];
			error+=(aux*aux);
		}
		error/=nCapaSalida;
	}

	return error;
}


// ------------------------------
// Retropropagar el error de salida con respecto a un vector pasado como argumento, desde la última capa hasta la primera
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
void PerceptronMulticapa::retropropagarError(double* objetivo, int funcionError) {


	//Se obtiene el número de neuronas de la capa de salida.
	int nCapaSalida=this->pCapas[this->nNumCapas-1].nNumNeuronas;
	double error,salida,aux;
	//TODO Entropía cruzada
	if(funcionError==1){

	}
	//MSE
	else{
		//Se obtienen las derivadas a de la capa de salida.
		for(int j=0; j<nCapaSalida;j++){
			salida=this->pCapas[this->nNumCapas-1].pNeuronas[j].x;
			error=objetivo[j]-salida;
			//Derivada de la salida con respecto al error.
			aux=-error*salida*(1-salida);
			this->pCapas[this->nNumCapas-1].pNeuronas[j].dX=aux;
		}
	}


	//Se recorren las capas de la salida a la entrada, empezando por la anteior a la salida.
	for(int h=this->nNumCapas-2;h>=1;h--){
		//Se recorren las neuronas de la capa i.

		for(int j=0; j<this->pCapas[h].nNumNeuronas;j++){
			aux=0.0;
			//Se obtiene el sumatorio los cambios asociados a los pesos de esa neurona,
			//de los enlaces conectados a la siguiente capa h+1.
			for(int i=0; i<this->pCapas[h+1].nNumNeuronas;i++){
				aux+=this->pCapas[h+1].pNeuronas[i].dX*this->pCapas[h+1].pNeuronas[i].w[j];
			}
			salida=this->pCapas[h].pNeuronas[j].x;
			aux=aux*salida*(1-salida);
			this->pCapas[h].pNeuronas[j].dX=aux;
		}
	}

}

// ------------------------------
// Acumular los cambios producidos por un patrón en deltaW
void PerceptronMulticapa::acumularCambio() {

	double derivadaSalida,salida;
	for(int h=1;h<this->nNumCapas;h++){
		for(int j=0;j<this->pCapas[h].nNumNeuronas;j++){
			//Se obtiene la derivada de la salida de la neurona j de la capa actual.
			derivadaSalida=this->pCapas[h].pNeuronas[j].dX;
			for(int i=0;i<this->pCapas[h-1].nNumNeuronas;i++){
				//Se obtiene la salida de la neurona i de la capa anterior
				salida=this->pCapas[h-1].pNeuronas[i].x;
				//Se acumula el cambio.
				this->pCapas[h].pNeuronas[j].deltaW[i]+=(salida*derivadaSalida);
			}
			//Se acumulan los cambios en el sesgo.
			this->pCapas[h].pNeuronas[j].deltaW[this->pCapas[h-1].nNumNeuronas]+=derivadaSalida*1;
		}
	}
}

// ------------------------------
// Actualizar los pesos de la red, desde la primera capa hasta la última
void PerceptronMulticapa::ajustarPesos() {

	//Se utilizan las siguientes variables para facilitar la comprensión del código
	double *peso,*dPeso,*dPesoAnterior,etah;
	for(int h=1;h<this->nNumCapas;h++){
		//Se decrementa eta en función de lo lejos que este de la capa de salida
		etah=this->dEta*pow(this->dDecremento,-(this->nNumCapas-1-h));
		for(int j=0;j<this->pCapas[h].nNumNeuronas;j++){

			for(int i=0;i<this->pCapas[h-1].nNumNeuronas;i++){
				peso=&(this->pCapas[h].pNeuronas[j].w[i]);
				dPeso=&(this->pCapas[h].pNeuronas[j].deltaW[i]);
				dPesoAnterior=&(this->pCapas[h].pNeuronas[j].ultimoDeltaW[i]);
				//Se aplican los cambios al peso del arco.
				*peso=(*peso)-(etah*(*dPeso))-this->dMu*(etah*(*dPesoAnterior));

				//Se establece el incremento a cero y se almacena el ultimo
				*dPesoAnterior=*dPeso;
				*dPeso=0.0;

			}
			//Se aplican los cambios en el arco del sesgo.
			peso=&(this->pCapas[h].pNeuronas[j].w[this->pCapas[h-1].nNumNeuronas]);
			dPeso=&(this->pCapas[h].pNeuronas[j].deltaW[this->pCapas[h-1].nNumNeuronas]);
			dPesoAnterior=&(this->pCapas[h].pNeuronas[j].ultimoDeltaW[this->pCapas[h-1].nNumNeuronas]);
			*peso=(*peso)-(etah*(*dPeso))-this->dMu*(etah*(*dPesoAnterior));

			//Se establece el incremento a cero y se almacena el ultimo
			*dPesoAnterior=*dPeso;
			*dPeso=0.0;
		}
	}
}

// ------------------------------
// Imprimir la red, es decir, todas las matrices de pesos
void PerceptronMulticapa::imprimirRed() {
	for(int h =1; h<this->nNumCapas;h++){
		std::cout<<"Capa "<<h<<std::endl<<"-----"<<std::endl;
		for(int j=0;j<this->pCapas[h].nNumNeuronas;j++){
			std::cout<<"Neurona "<<j<<": ";
			for(int i=0;i<=this->pCapas[h-1].nNumNeuronas;i++){
				std::cout<<this->pCapas[h].pNeuronas[j].w[i]<<"  ";
			}
			std::cout<<std::endl;
		}
		std::cout<<std::endl;
	}
}

// ------------------------------
// Simular la red: propragar las entradas hacia delante, computar el error, retropropagar el error y ajustar los pesos
// entrada es el vector de entradas del patrón, objetivo es el vector de salidas deseadas del patrón.
// El paso de ajustar pesos solo deberá hacerse si el algoritmo es on-line
// Si no lo es, el ajuste de pesos hay que hacerlo en la función "entrenar"
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
void PerceptronMulticapa::simularRed(double* entrada, double* objetivo, int funcionError) {

	this->alimentarEntradas(entrada);

	this->propagarEntradas();

	this->retropropagarError(objetivo,funcionError);
	this->acumularCambio();

	this->ajustarPesos();
}

// ------------------------------
// Leer una matriz de datos a partir de un nombre de fichero y devolverla
Datos* PerceptronMulticapa::leerDatos(const char *archivo) {
	Datos *matriz=new Datos;
	ifstream file(archivo);

	file>>matriz->nNumEntradas;
	file>>matriz->nNumSalidas;
	file>>matriz->nNumPatrones;
	matriz->entradas=new double*[matriz->nNumPatrones];
	matriz->salidas=new double*[matriz->nNumPatrones];
	for(int i=0;i<matriz->nNumPatrones;i++){
		matriz->entradas[i]=new double[matriz->nNumEntradas];
		matriz->salidas[i]=new double[matriz->nNumSalidas];
	}

	for(int i=0;i<matriz->nNumPatrones;i++){
		for(int j=0;j<matriz->nNumEntradas;j++){
			file>>matriz->entradas[i][j];
		}
		for(int j=0;j<matriz->nNumSalidas;j++){
			file>>matriz->salidas[i][j];
		}
	}
	file.close();
	return matriz;
}


// ------------------------------
// Entrenar la red para un determinado fichero de datos (pasar una vez por todos los patrones)
void PerceptronMulticapa::entrenar(Datos* pDatosTrain, int funcionError) {
	int i;
	for(i=0; i<pDatosTrain->nNumPatrones; i++){
		simularRed(pDatosTrain->entradas[i], pDatosTrain->salidas[i],funcionError);
	}
}

// ------------------------------
// Probar la red con un conjunto de datos y devolver el error cometido
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
double PerceptronMulticapa::test(Datos* pDatosTest, int funcionError) {
	//TODO
	double MSE=0.0;
	int i;

	for(i=0; i<pDatosTest->nNumPatrones; i++){
		this->alimentarEntradas(pDatosTest->entradas[i]);
		this->propagarEntradas();
		MSE+=this->calcularErrorSalida(pDatosTest->salidas[i],funcionError);

	}
	MSE/=pDatosTest->nNumPatrones;
	return MSE;
	return 0.0;
}

// OPCIONAL - KAGGLE
// Imprime las salidas predichas para un conjunto de datos.
// Utiliza el formato de Kaggle: dos columnas (Id y Predicted)
void PerceptronMulticapa::predecir(Datos* pDatosTest)
{
	int i;
	int j;
	int numSalidas = pCapas[nNumCapas-1].nNumNeuronas;
	double * salidas = new double[numSalidas];
	
	cout << "Id,Predicted" << endl;
	
	for (i=0; i<pDatosTest->nNumPatrones; i++){

		alimentarEntradas(pDatosTest->entradas[i]);
		propagarEntradas();
		recogerSalidas(salidas);

		int maxIndex = 0;
		for (j = 0; j < numSalidas; j++)
			if (salidas[j] >= salidas[maxIndex])
				maxIndex = j;
		
		cout << i << "," << maxIndex << endl;

	}
}

// ------------------------------
// Probar la red con un conjunto de datos y devolver el CCR
double PerceptronMulticapa::testClassification(Datos* pDatosTest) {

	return 0.0;
}

// ------------------------------
// Ejecutar el algoritmo de entrenamiento durante un número de iteraciones, utilizando pDatosTrain
// Una vez terminado, probar como funciona la red en pDatosTest
// Tanto el error MSE de entrenamiento como el error MSE de test debe calcularse y almacenarse en errorTrain y errorTest
// funcionError=1 => EntropiaCruzada // funcionError=0 => MSE
void PerceptronMulticapa::ejecutarAlgoritmo(Datos * pDatosTrain, Datos * pDatosTest, int maxiter, double *errorTrain, double *errorTest, double *ccrTrain, double *ccrTest, int funcionError)
{
	int countTrain = 0;

	// Inicialización de pesos
	pesosAleatorios();

	double minTrainError = 0;
	int numSinMejorar = 0;
	double testError = 0;
	nNumPatronesTrain = pDatosTrain->nNumPatrones;

	Datos * pDatosValidacion = NULL;
	double validationError = 0, previousValidationError = 0;
	int numSinMejorarValidacion = 0;

	// Generar datos de validación
	if(dValidacion > 0 && dValidacion < 1){

	}

	// Aprendizaje del algoritmo
	do {

		entrenar(pDatosTrain,funcionError);

		double trainError = test(pDatosTrain,funcionError);
		if(countTrain==0 || trainError < minTrainError){
			minTrainError = trainError;
			copiarPesos();
			numSinMejorar = 0;
		}
		else if( (trainError-minTrainError) < 0.00001)
			numSinMejorar = 0;
		else
			numSinMejorar++;

		if(numSinMejorar==50){
			cout << "Salida porque no mejora el entrenamiento!!"<< endl;
			restaurarPesos();
			countTrain = maxiter;
		}

		testError = test(pDatosTest,funcionError);
		countTrain++;

		// Comprobar condiciones de parada de validación y forzar

		cout << "Iteración " << countTrain << "\t Error de entrenamiento: " << trainError << "\t Error de test: " << testError << "\t Error de validacion: " << validationError << endl;

	} while ( countTrain<maxiter );

	if ( (numSinMejorarValidacion!=50) && (numSinMejorar!=50))
		restaurarPesos();

	cout << "PESOS DE LA RED" << endl;
	cout << "===============" << endl;
	imprimirRed();

	cout << "Salida Esperada Vs Salida Obtenida (test)" << endl;
	cout << "=========================================" << endl;
	for(int i=0; i<pDatosTest->nNumPatrones; i++){
		double* prediccion = new double[pDatosTest->nNumSalidas];

		// Cargamos las entradas y propagamos el valor
		alimentarEntradas(pDatosTest->entradas[i]);
		propagarEntradas();
		recogerSalidas(prediccion);
		for(int j=0; j<pDatosTest->nNumSalidas; j++)
			cout << pDatosTest->salidas[i][j] << " -- " << prediccion[j] << " \\\\ " ;
		cout << endl;
		delete[] prediccion;

	}

	*errorTest=test(pDatosTest,funcionError);;
	*errorTrain=minTrainError;
	*ccrTest = testClassification(pDatosTest);
	*ccrTrain = testClassification(pDatosTrain);

}

// OPCIONAL - KAGGLE
//Guardar los pesos del modelo en un fichero de texto.
bool PerceptronMulticapa::guardarPesos(const char * archivo)
{
	// Objeto de escritura de fichero
	ofstream f(archivo);

	if(!f.is_open())
		return false;

	// Escribir el numero de capas, el numero de neuronas en cada capa y el tipo de capa en la primera linea.
	f << nNumCapas;

	for(int i = 0; i < nNumCapas; i++)
	{
		f << " " << pCapas[i].nNumNeuronas;
		f << " " << pCapas[i].tipo;
	}
	f << endl;

	// Escribir los pesos de cada capa
	for(int i = 1; i < nNumCapas; i++)
		for(int j = 0; j < pCapas[i].nNumNeuronas; j++)
			for(int k = 0; k < pCapas[i-1].nNumNeuronas + 1; k++)
				f << pCapas[i].pNeuronas[j].w[k] << " ";

	f.close();

	return true;

}


// OPCIONAL - KAGGLE
//Cargar los pesos del modelo desde un fichero de texto.
bool PerceptronMulticapa::cargarPesos(const char * archivo)
{
	// Objeto de lectura de fichero
	ifstream f(archivo);

	if(!f.is_open())
		return false;

	// Número de capas y de neuronas por capa.
	int nl;
	int *npl;
	int *tipos;

	// Leer número de capas.
	f >> nl;

	npl = new int[nl];
	tipos = new int[nl];

	// Leer número de neuronas en cada capa y tipo de capa.
	for(int i = 0; i < nl; i++)
	{
		f >> npl[i];
		f >> tipos[i];
	}

	// Inicializar vectores y demás valores.
	inicializar(nl, npl, tipos);

	// Leer pesos.
	for(int i = 1; i < nNumCapas; i++)
		for(int j = 0; j < pCapas[i].nNumNeuronas; j++)
			for(int k = 0; k < pCapas[i-1].nNumNeuronas + 1; k++)
				f >> pCapas[i].pNeuronas[j].w[k];

	f.close();
	delete[] npl;

	return true;
}

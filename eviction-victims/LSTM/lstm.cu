/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
//		Author:    Chethan Palangotu Keshava
//		LinkedIn:  https://www.linkedin.com/in/chethankeshava/
//		File:      CUDA implementation of LSTM model
//		Objective: Testing the performance of GPU architecture modifications done 
//			   to GPGPU-SIM. The simulator is built on old CUDA version (4.0)
//			   and hence no libraries are used for computations, with each
//			   computation done manually
///////////////////////////////////////////////////////////////////////////////////////// 

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <string>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <assert.h>
using namespace std;

#define input_0 0.98
#define input_1 0.88

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);
void NeuralNetwork();

unsigned g_verbose;
unsigned NUM;

/////////////////////////////////////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{
	int i, commandline_error;
	commandline_error = 0;
	g_verbose = 0;
	if (argc >= 2) {
		NUM = atoi(argv[1]);
		for (i=2; i < argc;i++) {
			if (argv[i][0] == '-') {
				switch (argv[i][1]) {
				case 'v': g_verbose = 1;
					break;
				default: commandline_error=1;
				}
			}
			else commandline_error=1;
		}
	} else commandline_error=1;

	if (commandline_error || !NUM) {
		printf("Usage: ./LSTM <NUM> [-v]\n");
		printf("where NUM is the number of images to process in parallel (up to 10000 for the t10k-images-idx3-ubyte database file) and -v is used to display approximately what each image looks like.\n");
		return 1;
	}


	NeuralNetwork();
    //CUT_EXIT(argc, argv);
}

void InitHostMem(double *w_i, double *u_i, double *b_i, double *w_f, double *u_f, double *b_f, double *w_c, double *u_c, double *b_c, double *w_o, double *u_o, double *b_o, double *weight, double *bias)
{
	// Input Gate Weights and Bias
	FILE * pFile1 = fopen ("data/w_i.txt","rb");
	if (pFile1 != NULL)
	{
		//printf("File Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile1);
		//printf("Reading Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			w_i[i] = temp_num;
			i++;
			index++;
			if(i==100)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile1);
	}	
	
	if (!pFile1)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile2 = fopen ("data/b_i.txt","rb");
	if (pFile2 != NULL)
	{
		//printf("File Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile2);
		//printf("Reading Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			b_i[i] = temp_num;
			i++;
			index++;
			if(i==100)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile2);
	}	
	//cout<<"Input gate reading completed, "<<b_i[99]<<endl;
	if (!pFile2)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	// Forget gate Weights and Bias
	FILE * pFile3 = fopen ("data/w_f.txt","rb");
	if (pFile3 != NULL)
	{
		//printf("File Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile3);
		//printf("Reading Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			w_f[i] = temp_num;
			i++;
			index++;
			if(i==100)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile3);
	}	
	
	if (!pFile3)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile4 = fopen ("data/b_f.txt","rb");
	if (pFile4 != NULL)
	{
		//printf("File Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile4);
		//printf("Reading Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			b_f[i] = temp_num;
			i++;
			index++;
			if(i==100)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile4);
	}	
	//cout<<"Forget gate reading completed, "<<b_f[99]<<endl;
	if (!pFile4)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	// Cell gate Weights and Bias
	FILE * pFile5 = fopen ("data/w_c.txt","rb");
	if (pFile5 != NULL)
	{
		//printf("File Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile5);
		//printf("Reading Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			w_c[i] = temp_num;
			i++;
			index++;
			if(i==100)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile5);
	}	
	
	if (!pFile5)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile6 = fopen ("data/b_c.txt","rb");
	if (pFile6 != NULL)
	{
		//printf("File Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile6);
		//printf("Reading Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			b_c[i] = temp_num;
			i++;
			index++;
			if(i==100)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile6);
	}	
	//cout<<"Forget gate reading completed, "<<b_c[99]<<endl;
	if (!pFile6)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	// Output gate Weights and Bias
	FILE * pFile7 = fopen ("data/w_o.txt","rb");
	if (pFile7 != NULL)
	{
		//printf("File Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile7);
		//printf("Reading Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			w_o[i] = temp_num;
			i++;
			index++;
			if(i==100)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile7);
	}	
	
	if (!pFile7)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile8 = fopen ("data/b_o.txt","rb");
	if (pFile8 != NULL)
	{
		//printf("File Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile8);
		//printf("Reading Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			b_o[i] = temp_num;
			i++;
			index++;
			if(i==100)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile8);
	}	
	//cout<<"Output gate reading completed, "<<b_o[99]<<endl;
	if (!pFile8)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile9 = fopen ("data/W.txt","rb");
	if (pFile9 != NULL)
	{
		//printf("File Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile9);
		//printf("Reading Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			weight[i] = temp_num;
			i++;
			index++;
			if(i==101)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile9);
	}
	*bias = weight[100];	
	//cout<<"Fully Connected layer reading completed, "<<*bias<<endl;
	if (!pFile9)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	//Recurrent weights of output gate
	FILE * pFile10 = fopen ("data/u_o.txt","rb");
	if (pFile10 != NULL)
	{
		//printf("File Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile10);
		//printf("Reading Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			u_o[i] = temp_num;
			i++;
			index++;
			if(i==10000)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile10);
	}	
	
	if (!pFile10)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	//Recurrent weights of input gate
	FILE * pFile11 = fopen ("data/u_i.txt","rb");
	if (pFile11 != NULL)
	{
		//printf("File Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile11);
		//printf("Reading Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			u_i[i] = temp_num;
			i++;
			index++;
			if(i==10000)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile11);
	}	
	
	if (!pFile11)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	//Recurrent weights of cell gate
	FILE * pFile12 = fopen ("data/u_c.txt","rb");
	if (pFile12 != NULL)
	{
		//printf("File Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile12);
		//printf("Reading Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			u_c[i] = temp_num;
			i++;
			index++;
			if(i==10000)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile12);
	}	
	
	if (!pFile12)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	//Recurrent weights of forget gate
	FILE * pFile13 = fopen ("data/u_f.txt","rb");
	if (pFile13 != NULL)
	{
		//printf("File Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile13);
		//printf("Reading Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			u_f[i] = temp_num;
			i++;
			index++;
			if(i==10000)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile13);
	}	
	
	if (!pFile13)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
}

__device__ double hard_sigmoid(double x)
{
	if(x<-2.5)
		return 0;
	else
	{
		if(x>2.5)
			return 1;
		else
			return (0.2*x + 0.5);
	}
}

__global__ void ExecuteLSTM(double *w_i, double *u_i, double *b_i, double *w_f, double *u_f, double *b_f, double *w_c, double *u_c, double *b_c, double *w_o, double *u_o, double *b_o, double *weight, double *bias, double *LSTM_results)
{
	int x = threadIdx.x;
	__shared__ double u[100];
	double c = 0;
	double at, it, ft, ot, statet, output_state;
	double i[2];
	double temp_ua = 0, temp_ui = 0, temp_uf = 0, temp_uo = 0;
	//int index = threadIdx.x*2;
	i[0] = input_0;
	i[1] = input_1;
	at = tanh((i[0]*w_c[x]) + b_c[x]);
	it = hard_sigmoid((i[0]*w_i[x]) + b_i[x]);
	ft = hard_sigmoid((i[0]*w_f[x]) + b_f[x]);
	ot = hard_sigmoid((i[0]*w_o[x]) + b_o[x]);
	statet = at*it + ft*c;
	c = statet;
	output_state = ot*tanh(statet);
	u[x] = output_state;
	__syncthreads();
	if(x==0)
	{
		double result = *bias;
		for(int i=0; i<100; i++)
		{
			result += weight[i]*u[i];
		}
		//printf("The result for i=0 is %f\n",result);
		LSTM_results[0] = result;
	}
	__syncthreads();
	for(int i=0; i<100; i++)
	{
		temp_ua += u[i] * u_c[i*100 + x];
		temp_ui += u[i] * u_i[i*100 + x];
		temp_uf += u[i] * u_f[i*100 + x];
		temp_uo += u[i] * u_o[i*100 + x]; 	
	}
	at = tanh((i[1]*w_c[x]) + temp_ua + b_c[x]);
	it = hard_sigmoid((i[1]*w_i[x]) + temp_ui + b_i[x]);
	ft = hard_sigmoid((i[1]*w_f[x]) + temp_uf + b_f[x]);
	ot = hard_sigmoid((i[1]*w_o[x]) + temp_uo + b_o[x]);
	statet = at*it + ft*c;
	output_state = ot*tanh(statet);
	u[x] = output_state;
	__syncthreads();
	if(x==0)
	{
		double result = *bias;
		for(int i=0; i<100; i++)
		{
			result += weight[i]*u[i];
		}
		//printf("The result for i=1 is %f\n",result);
		LSTM_results[1] = result;
	}
}

void NeuralNetwork()
{
	cudaError_t err;
	//cudaEvent_t start, stop;
	double *w_i = (double*) malloc (100 * NUM * sizeof(double));
	double *u_i = (double*) malloc (10000 * NUM * sizeof(double));
	double *b_i = (double*) malloc (100 * NUM * sizeof(double));
	double *w_f = (double*) malloc (100 * NUM * sizeof(double));
	double *u_f = (double*) malloc (10000 * NUM * sizeof(double));
	double *b_f = (double*) malloc (100 * NUM * sizeof(double));
	double *w_c = (double*) malloc (100 * NUM * sizeof(double));
	double *u_c = (double*) malloc (10000 * NUM * sizeof(double));
	double *b_c = (double*) malloc (100 * NUM * sizeof(double));
	double *w_o = (double*) malloc (100 * NUM * sizeof(double));
	double *u_o = (double*) malloc (10000 * NUM * sizeof(double));
	double *b_o = (double*) malloc (100 * NUM * sizeof(double));
	double *weights = (double*) malloc (101 * NUM * sizeof(double));
	double *bias = (double*) malloc (sizeof(double));
	double *result = (double*) malloc (2 * NUM * sizeof(double));
	InitHostMem(w_i, u_i, b_i, w_f, u_f, b_f, w_c, u_c, b_c, w_o, u_o, b_o, weights, bias);
	//cout<<*bias<<endl;
	double *w_i_device;
	err = cudaMalloc((void**) &w_i_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *u_i_device;
	err = cudaMalloc((void**) &u_i_device, 10000* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *b_i_device;
	err = cudaMalloc((void**) &b_i_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *w_f_device;
	err = cudaMalloc((void**) &w_f_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *u_f_device;
	err = cudaMalloc((void**) &u_f_device, 10000* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *b_f_device;
	err = cudaMalloc((void**) &b_f_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *w_c_device;
	err = cudaMalloc((void**) &w_c_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *u_c_device;
	err = cudaMalloc((void**) &u_c_device, 10000* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *b_c_device;
	err = cudaMalloc((void**) &b_c_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *w_o_device;
	err = cudaMalloc((void**) &w_o_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *u_o_device;
	err = cudaMalloc((void**) &u_o_device, 10000* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *b_o_device;
	err = cudaMalloc((void**) &b_o_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *weights_device;
	err = cudaMalloc((void**) &weights_device, 100* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *bias_device;
	err = cudaMalloc((void**) &bias_device, 1* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *LSTM_results;
	err = cudaMalloc((void**) &LSTM_results, 2* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	//printf("Malloc completed\n");
	//Start Memory Copy
	err = cudaMemcpy(w_i_device, w_i, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data 1(error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(u_i_device, u_i, sizeof(double)*10000*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data 2(error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(b_i_device, b_i, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data 3(error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(w_f_device, w_f, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data 4(error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(u_f_device, u_f, sizeof(double)*10000*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data 5(error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(b_f_device, b_f, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data 6(error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(w_c_device, w_c, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data 7(error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(u_c_device, u_c, sizeof(double)*10000*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data 8(error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(b_c_device, b_c, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data 9(error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(w_o_device, w_o, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(u_o_device, u_o, sizeof(double)*10000*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(b_o_device, b_o, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(weights_device, weights, sizeof(double)*100*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(bias_device, bias, sizeof(double)*1*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device bias data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	//printf("Memcpy completed\n");
	dim3 n_threads(100,1,1);
	dim3 n_blocks(1,1,1);
	ExecuteLSTM<<<n_blocks,n_threads>>>(w_i_device, u_i_device, b_i_device, w_f_device, u_f_device, b_f_device, w_c_device, u_c_device, b_c_device, w_o_device, u_o_device, b_o_device, weights_device, bias_device, LSTM_results);
	//cudaThreadSynchronize();
	
	cudaStreamSynchronize(0);
	err = cudaMemcpy(result, LSTM_results, sizeof(double)*2*NUM, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy result data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	printf("The result for i=0 is %f and i=1 is %f\n",result[0], result[1]);

        /*cudaFree(w_i_device);
        cudaFree(u_i_device);
        cudaFree(b_i_device);
        cudaFree(w_f_device);
        cudaFree(u_f_device);
        cudaFree(b_f_device);
        cudaFree(w_c_device);
        cudaFree(u_c_device);
        cudaFree(b_c_device);
        cudaFree(w_o_device);
        cudaFree(u_o_device);
        cudaFree(b_o_device);
        cudaFree(weights_device);
        cudaFree(bias_device);
        cudaFree(LSTM_results);*/

	free(w_i);
	free(u_i);
	free(b_i);
	free(w_f);
	free(u_f);
	free(b_f);
	free(w_c);
	free(u_c);
	free(b_c);
	free(w_o);
	free(u_o);
	free(b_o);
	free(weights);
	free(bias);

}

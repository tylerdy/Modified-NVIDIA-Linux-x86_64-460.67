/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
//		Author:    Chethan Palangotu Keshava
//		LinkedIn:  https://www.linkedin.com/in/chethankeshava/
//		File:      C/C++ implementation of CifarNet
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


extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);
void NeuralNetwork();

unsigned g_verbose;
unsigned NUM;

////////////////////////////////////////////////////////////////////////////////
// Program main
////////////////////////////////////////////////////////////////////////////////
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
		printf("Usage: ./NN <NUM> [-v]\n");
		printf("where NUM is the number of images to process in parallel (up to 10000 for the t10k-images-idx3-ubyte database file) and -v is used to display approximately what each image looks like.\n");
		return 1;
	}


	NeuralNetwork();
    //CUT_EXIT(argc, argv);
}

void InitHostMem(double *Layer1_Weights_CPU, double *Layer2_Weights_CPU, double *Layer3_Weights_CPU, double *Layer4_Weights_CPU, double *Layer5_Weights_CPU)
{
	// initial layer 1 weight
	FILE * pFile1 = fopen ("data/conv1.txt","rb");
	if (pFile1 != NULL)
	{
		printf("File Opened\n");
		char s[300000] = "";
		fread(s,sizeof(s),1,pFile1);
		printf("Done2\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			Layer1_Weights_CPU[i] = temp_num;
			i++;
			index++;
			if(i==2400)
			{
				printf("Breaking\n");
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
	//Layer 2 Weights
	FILE * pFile2 = fopen ("data/conv2.txt","rb");
	if (pFile2 != NULL)
	{
		printf("File 2 Opened\n");
		char s[3000000] = "";
		fread(s,sizeof(s),1,pFile2);
		printf("Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			Layer2_Weights_CPU[i] = temp_num;
			i++;
			index++;
			if(i==25600)
			{
				printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile2);
	}	
	
	if (!pFile2)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	//Layer 3 Weights
	FILE * pFile3 = fopen ("data/conv3.txt","rb");
	if (pFile3 != NULL)
	{
		printf("File 3 Opened\n");
		char s[6000000] = "";
		fread(s,sizeof(s),1,pFile2);
		printf("Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			Layer3_Weights_CPU[i] = temp_num;
			i++;
			index++;
			if(i==51200)
			{
				printf("Breaking\n");
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
	//Layer 4 Weights
	FILE * pFile4 = fopen ("data/ip1.txt","rb");
	if (pFile4 != NULL)
	{
		printf("File 4 Opened\n");
		char s[8000000] = "";
		fread(s,sizeof(s),1,pFile2);
		printf("Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			Layer4_Weights_CPU[i] = temp_num;
			i++;
			index++;
			if(i==65536)
			{
				printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile4);
	}	
	
	if (!pFile4)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	//Layer 5 Weights
	FILE * pFile5 = fopen ("data/ip2.txt","rb");
	if (pFile5 != NULL)
	{
		printf("File 5 Opened\n");
		char s[80000] = "";
		fread(s,sizeof(s),1,pFile2);
		printf("Done\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			Layer5_Weights_CPU[i] = temp_num;
			i++;
			index++;
			if(i==576)
			{
				printf("Breaking\n");
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
}

void LoadInput(int *Data_Layer_CPU)
{
	FILE * pFile1 = fopen ("data/speed-limit-35.txt","rb");
	if (pFile1 != NULL)
	{
		printf("File Opened\n");
		char s[300000] = "";
		fread(s,sizeof(s),1,pFile1);
		printf("Done2\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			int temp_num = atof(temp_string);	
			Data_Layer_CPU[i] = temp_num;
			i++;
			index++;
			if(i==(32*32*3))
			{
				printf("Breaking input\n");
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
}

void ConvertInput(int *Data_Layer_CPU_R, int *Data_Layer_CPU_G, int *Data_Layer_CPU_B, int *Data_Layer_CPU)
{
	for(int i=0; i<32*32*3; i+=3)
	{
		Data_Layer_CPU_R[i/3] = Data_Layer_CPU[i];
		Data_Layer_CPU_G[i/3] = Data_Layer_CPU[i+1];
		Data_Layer_CPU_B[i/3] = Data_Layer_CPU[i+2];
	}
}
void ExecuteFirstLayer(double *Layer1_Weights_CPU, int *Data_Layer_CPU_R, int *Data_Layer_CPU_G, int *Data_Layer_CPU_B, double ***Layer1_Features)
{
	printf("First Layer Execution\n");
	int f=0;
	{
		double maskR[25], maskG[25], maskB[25];
		int imageR[25], imageG[25], imageB[25];
		for(int i=0; i<25; i++)
		{
			maskR[i] = Layer1_Weights_CPU[i+f*75];
			maskG[i] = Layer1_Weights_CPU[i+25+f*75];
			maskB[i] = Layer1_Weights_CPU[i+50+f*75];
		}
		int x=0;
		{
			int y=0;
			{
				double result = 0;
				for(int i = x-2; i<=x+2; i++)
				{
    					for(int j=y-2; j<=y+2; j++)
    					{
						int x_index = i-x+2;
						int y_index = j-y+2;
						int m = (y_index)+(x_index)*5;
         					if(i<0 || j<0)
						{
             				 		imageR[(y_index)+(x_index)*5] = 0;
							imageG[(y_index)+(x_index)*5] = 0;
							imageB[(y_index)+(x_index)*5] = 0;
						}
         					else if(j>31 || i>31)
						{
              						imageR[(y_index)+(x_index)*5] = 0;
							imageG[(y_index)+(x_index)*5] = 0;
							imageB[(y_index)+(x_index)*5] = 0;
						}
         					else
						{
               						imageR[(y_index)+(x_index)*5] = Data_Layer_CPU_R[(y_index-2) + x*32 + y + (x_index-2)*32];
               						imageG[(y_index)+(x_index)*5] = Data_Layer_CPU_G[(y_index-2) + x*32 + y + (x_index-2)*32];
               						imageB[(y_index)+(x_index)*5] = Data_Layer_CPU_B[(y_index-2) + x*32 + y + (x_index-2)*32];
							result += Data_Layer_CPU_R[(y_index-2) + x*32 + y + (x_index-2)*32]*Layer1_Weights_CPU[m+f*75] + Data_Layer_CPU_G[(y_index-2) + x*32 + y + (x_index-2)*32]*Layer1_Weights_CPU[m+25+f*75] + Data_Layer_CPU_B[(y_index-2) + x*32 + y + (x_index-2)*32]*Layer1_Weights_CPU[m+50+f*75];
							printf("%d %d\n", ((y_index-2) + x*32 + y + (x_index-2)*32), m); 		
						}
					}
				}  
				Layer1_Features[f][x][y] = result;
				printf("%f ", result);
			}
		}
	}
}

void ExecuteSecondLayer(double *Layer2_Weights_CPU, double ***Layer2_Features, double ***Layer2_pool_GPU)
{
	printf("Second Layer Execution\n");
	for(int f=0; f<32; f++)
	{
		double mask[32][25];
		double input[32][25];
		double Features[32][16][16];
		for(int n=0; n<32; n++)
		{
			for(int i=0; i<25; i++)
			{
				mask[n][i] = Layer2_Weights_CPU[i+f*25*32+n*25];
			}
		}
		for(int n=0; n<32; n++)
		{
			for(int x=0; x<16; x++)
			{
				for(int y=0; y<16; y++)
				{
					for(int i = x-2; i<=x+2; i++)
					{
    						for(int j=y-2; j<=y+2; j++)
    						{
							int x_index = i-x+2;
							int y_index = j-y+2;
         						if(i<0 || j<0)
							{
             					 		input[n][(y_index)+(x_index)*5] = 0;
							}
         						else if(j>15 || i>15)
							{
              							input[n][(y_index)+(x_index)*5] = 0;
							}
         						else
							{
               							input[n][(y_index)+(x_index)*5] = Layer2_pool_GPU[n][x_index+x-2][y_index+y-2];			
							}
						}
					}
					double result = 0;
					for(int i=0; i<25; i++)
					{
						result+= input[n][i]*mask[n][i]; 
					}   
					Features[n][x][y] = result;
				}
			}
		}
		for(int n=0; n<32; n++)
		{
			for(int x=0; x<16; x++)
			{
				for(int y=0; y<16; y++)
				{
					Layer2_Features[f][x][y]+= Features[n][x][y];
				}
			}
		}
	}
	for(int f=0; f<32; f++)
	{
		for(int x=0; x<16; x++)
		{
			for(int y=0; y<16; y++)
			{
				if(Layer2_Features[f][x][y] < 0)
					Layer2_Features[f][x][y] = 0;
			}	
		}
	}
}

void ExecuteThirdLayer(double *Layer3_Weights_CPU, double ***Layer3_Features, double ***Layer3_pool_GPU)
{
	printf("Third Layer Execution\n");
	for(int f=0; f<64; f++)
	{
		double mask[32][25];
		double input[32][25];
		double Features[64][8][8];
		for(int n=0; n<32; n++)
		{
			for(int i=0; i<25; i++)
			{
				mask[n][i] = Layer3_Weights_CPU[i+f*25*32+n*25];
			}
		}
		for(int n=0; n<32; n++)
		{
			for(int x=0; x<8; x++)
			{
				for(int y=0; y<8; y++)
				{
					for(int i = x-2; i<=x+2; i++)
					{
    						for(int j=y-2; j<=y+2; j++)
    						{
							int x_index = i-x+2;
							int y_index = j-y+2;
         						if(i<0 || j<0)
							{
             					 		input[n][(y_index)+(x_index)*5] = 0;
							}
         						else if(j>7 || i>7)
							{
              							input[n][(y_index)+(x_index)*5] = 0;
							}
         						else
							{
               							input[n][(y_index)+(x_index)*5] = Layer3_pool_GPU[n][x_index+x-2][y_index+y-2];			
							}
						}
					}
					double result = 0;
					for(int i=0; i<25; i++)
					{
						result+= input[n][i]*mask[n][i]; 
					}   
					Features[n][x][y] = result;
				}
			}
		}
		for(int n=0; n<32; n++)
		{
			for(int x=0; x<8; x++)
			{
				for(int y=0; y<8; y++)
				{
					Layer3_Features[f][x][y]+= Features[n][x][y];
				}
			}
		}
	}
	for(int f=0; f<64; f++)
	{
		for(int x=0; x<8; x++)
		{
			for(int y=0; y<8; y++)
			{
				if(Layer3_Features[f][x][y] < 0)
					Layer3_Features[f][x][y] = 0;
			}	
		}
	}
}

void ExecuteFourthLayer(double *Layer4_Weights_CPU, double *Layer4_Features, double ***Pool3_Layer_Features)
{
	printf("Fourth Layer Execution\n");
	for(int n=0;n<64; n++)
	{
		double result = 0;
		for(int f=0; f<64; f++)
		{
			for(int x=0; x<4; x++)
			{
				for(int y=0; y<4; y++)
				{
					result+= Pool3_Layer_Features[f][x][y] * Layer4_Weights_CPU[y+(x*4)+(f*4*4)+(n*4*4*64)];
				}
			}
		}
		Layer4_Features[n] = result;
		result = 0;
	}
}

void ExecuteFifthLayer(double *Layer5_Weights_CPU, double *Layer5_Features, double *Layer4_Features)
{
	printf("Fifth Layer Execution\n");
	for(int n=0;n<9; n++)
	{
		double result = 0;
		for(int f=0; f<64; f++)
		{
			result+= Layer4_Features[f] * Layer5_Weights_CPU[f+n*64];
		}
		Layer5_Features[n] = result;
		printf("%.8f ",result);
		result = 0;
	}
	printf("\n");
}

void pooling1(double ***Layer2_Neurons_GPU,double ***Layer2_pool_GPU,int out,int out_fr,int out_fc,int kernel,int stride_width,int in_fr,int in_fc)
{
    printf("pooling Activation layer \n");
    double max = 0.0;
    {
        for(int output =0;output < out ;output++)
        {
            for(int row =1; row <= 31 ;row+=2)
            { 
                for(int col =1; col <= 31 ;col+=2)
                {
                    for(int i = row-1; i <= row+1; i++)
                    {   
			if(i>31) break;        
                        for(int j = col-1; j <= col+1; j++)
                        {
			    if(j>31) break;
                            if(max < ((Layer2_Neurons_GPU[output][i][j])))
                                max =   ((Layer2_Neurons_GPU[output][i][j])) ;

                        }
                    }
		    if(max<0)
			max = 0;
                    Layer2_pool_GPU[output][(row-1)/2][(col-1)/2] = max;    
                    max = 0.0;   
                }
            }
        }
    }
}

void pooling2(double ***Layer2_Neurons_GPU,double ***Layer2_pool_GPU,int out,int out_fr,int out_fc,int kernel,int stride_width,int in_fr,int in_fc)
{
    printf("pooling 2 layer \n");
    double avg = 0.0;
    int count = 0;
    {
        for(int output =0;output < out ;output++)
        {
            for(int row =1; row <= 16 ;row+=2)
            { 
                for(int col =1; col <= 16 ;col+=2)
                {
                    for(int i = row-1; i <= row+1; i++)
                    {   
			if(i>15) break;        
                        for(int j = col-1; j <= col+1; j++)
                        {
			    if(j>15) break;
                            avg+= ((Layer2_Neurons_GPU[output][i][j]));
			    count++;

                        }
                    }
                    Layer2_pool_GPU[output][(row-1)/2][(col-1)/2] = avg/count;     
                    avg = 0.0;   
		    count=0;
                }
            }
        }
    }
}

void pooling3(double ***Layer3_Neurons_GPU,double ***Layer3_pool_GPU,int out,int out_fr,int out_fc,int kernel,int stride_width,int in_fr,int in_fc)
{
    printf("pooling 3 layer \n");
    double avg = 0.0;
    int count = 0;
    {
        for(int output =0;output < out ;output++)
        {
            for(int row =1; row <= 8 ;row+=2)
            { 
                for(int col =1; col <= 8 ;col+=2)
                {
                    for(int i = row-1; i <= row+1; i++)
                    {   
			if(i>7) break;        
                        for(int j = col-1; j <= col+1; j++)
                        {
			    if(j>7) break;
                            avg+= ((Layer3_Neurons_GPU[output][i][j]));
			    count++;

                        }
                    }
                    Layer3_pool_GPU[output][(row-1)/2][(col-1)/2] = avg/count;     
                    avg = 0.0;   
		    count=0;
                }
            }
        }
    }
}

void NeuralNetwork()
{
	double *Layer1_Weights_CPU = (double*) malloc (3*32*32* NUM * sizeof(double));
	double *Layer2_Weights_CPU = (double*) malloc (5*5*32*32* NUM * sizeof(double));
	double *Layer3_Weights_CPU = (double*) malloc (5*5*32*64* NUM * sizeof(double));
	double *Layer4_Weights_CPU = (double*) malloc (64*4*4*64* NUM * sizeof(double));
	double *Layer5_Weights_CPU = (double*) malloc (64*9* NUM * sizeof(double));
	int *Data_Layer_CPU_R = (int*) malloc (32*32*NUM*sizeof(int));
	int *Data_Layer_CPU_G = (int*) malloc (32*32*NUM*sizeof(int));
	int *Data_Layer_CPU_B = (int*) malloc (32*32*NUM*sizeof(int));
	int *Data_Layer_CPU = (int*) malloc (3*32*32*NUM*sizeof(int));
	InitHostMem(Layer1_Weights_CPU, Layer2_Weights_CPU, Layer3_Weights_CPU, Layer4_Weights_CPU, Layer5_Weights_CPU);
	LoadInput(Data_Layer_CPU);
	ConvertInput(Data_Layer_CPU_R, Data_Layer_CPU_G, Data_Layer_CPU_B, Data_Layer_CPU);
	double ***Layer1_Features;
	Layer1_Features = (double***)malloc(32*sizeof(double **));
	assert(Layer1_Features!= NULL);
	for(int i=0; i<32; i++)
	{
		Layer1_Features[i] = (double**)malloc(32*sizeof(double *));
		assert(Layer1_Features[i] != NULL);
		for(int j=0; j<32; j++)
		{
			Layer1_Features[i][j] = (double*)malloc(32*sizeof(double));
		}
	}
	ExecuteFirstLayer(Layer1_Weights_CPU, Data_Layer_CPU_R, Data_Layer_CPU_G, Data_Layer_CPU_B, Layer1_Features);
	double ***Pool_Layer_Features;
	Pool_Layer_Features = (double***)malloc(32*sizeof(double **));
	assert(Pool_Layer_Features!= NULL);
	for(int i=0; i<32; i++)
	{
		Pool_Layer_Features[i] = (double**)malloc(16*sizeof(double *));
		assert(Pool_Layer_Features[i] != NULL);
		for(int j=0; j<16; j++)
		{
			Pool_Layer_Features[i][j] = (double*)malloc(16*sizeof(double));
		}
	}
	pooling1(Layer1_Features, Pool_Layer_Features, 32, 16, 16, 5, 2, 32, 32);
	double ***Layer2_Features;
	Layer2_Features = (double***)malloc(32*sizeof(double **));
	assert(Layer2_Features!= NULL);
	for(int i=0; i<32; i++)
	{
		Layer2_Features[i] = (double**)malloc(16*sizeof(double *));
		assert(Layer2_Features[i] != NULL);
		for(int j=0; j<16; j++)
		{
			Layer2_Features[i][j] = (double*)malloc(16*sizeof(double));
		}
	}
	ExecuteSecondLayer(Layer2_Weights_CPU, Layer2_Features, Pool_Layer_Features);
	double ***Pool2_Layer_Features;
	Pool2_Layer_Features = (double***)malloc(32*sizeof(double **));	
	assert(Pool2_Layer_Features!= NULL);
	for(int i=0; i<32; i++)
	{
		Pool2_Layer_Features[i] = (double**)malloc(8*sizeof(double *));
		assert(Pool2_Layer_Features[i] != NULL);
		for(int j=0; j<8; j++)
		{
			Pool2_Layer_Features[i][j] = (double*)malloc(8*sizeof(double));
		}
	}
	pooling2(Layer2_Features, Pool2_Layer_Features, 32, 8, 8, 5, 2, 16, 16);
	double ***Layer3_Features;
	Layer3_Features = (double***)malloc(64*sizeof(double **));
	assert(Layer3_Features!= NULL);
	for(int i=0; i<64; i++)
	{
		Layer3_Features[i] = (double**)malloc(8*sizeof(double *));
		assert(Layer3_Features[i] != NULL);
		for(int j=0; j<8; j++)
		{
			Layer3_Features[i][j] = (double*)malloc(8*sizeof(double));
		}
	}
	ExecuteThirdLayer(Layer3_Weights_CPU, Layer3_Features, Pool2_Layer_Features);
	double ***Pool3_Layer_Features;
	Pool3_Layer_Features = (double***)malloc(64*sizeof(double **));	
	assert(Pool3_Layer_Features!= NULL);
	for(int i=0; i<64; i++)
	{
		Pool3_Layer_Features[i] = (double**)malloc(4*sizeof(double *));
		assert(Pool3_Layer_Features[i] != NULL);
		for(int j=0; j<4; j++)
		{
			Pool3_Layer_Features[i][j] = (double*)malloc(4*sizeof(double));
		}
	}
	pooling3(Layer3_Features, Pool3_Layer_Features, 64, 4, 4, 5, 2, 8, 8);
	double *Layer4_Features;
	Layer4_Features = (double*)malloc(64*sizeof(double));
	ExecuteFourthLayer(Layer4_Weights_CPU, Layer4_Features, Pool3_Layer_Features);	
	double *Layer5_Features;
	Layer5_Features = (double*)malloc(9*sizeof(double));
	ExecuteFifthLayer(Layer5_Weights_CPU, Layer5_Features, Layer4_Features);	

}



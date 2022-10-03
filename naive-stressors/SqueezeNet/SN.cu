/////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////
//		Author:    Chethan Palangotu Keshava
//		LinkedIn:  https://www.linkedin.com/in/chethankeshava/
//		File:      CUDA implementation of SqueezeNet
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

  /* Wrapper function for reading the global nanosecond-precision timer
 */

static __device__ __inline__ unsigned long long int gclock64() {

    unsigned long long int rv;

    asm volatile ( "mov.u64 %0, %%globaltimer;" : "=l"(rv) );

    return rv;

}

extern "C"
void computeGold(float*, const float*, const float*, unsigned int, unsigned int, unsigned int);
void NeuralNetwork(unsigned long long);

unsigned g_verbose;
unsigned NUM;

/////////////////////////////////////////////////////////////////////////////////////////
// Program main
/////////////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char** argv)
{

	unsigned long long run_time = 0;
    int i=0;
    while (i < argc) {
        if (strcmp (argv[i], "-t") == 0) { // run time in seconds
            if (++i >= argc) printf("bad params");
            run_time = atoi(argv[i]) * 1000000000ULL;  //seconds to nanoseconds
        }
        i++;
    }

	int commandline_error;
	commandline_error = 0;
	g_verbose = 0;
	NUM=1;
	/*if (argc >= 2) {
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
	}*/


	NeuralNetwork(run_time);
    //CUT_EXIT(argc, argv);
}

/////////////////////////////////////////////////////////////////////////////////////////
// Read the input image file, which is a txt file with R, G and B values
/////////////////////////////////////////////////////////////////////////////////////////
/*void LoadInput(int *Data_Layer_CPU)
{
	FILE * pFile1 = fopen ("../../data-SN//cat.txt","rb");
	if (pFile1 != NULL)
	{
		printf("File Opened\n");
		char s[3000000] = "";
		fread(s,sizeof(s),1,pFile1);
		printf("Done2\n");
		printf("%s", s);
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			int temp_num = atof(temp_string);	
			Data_Layer_CPU[i] = temp_num;
			if(temp_num == 0)
			{
				printf("Index: %ld", i);
				break;
			}
			i++;
			index++;
			//printf("%d ", temp_num);
			if(i==(227*227*3))
			{
				printf("Breaking input\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		fclose (pFile1);
	}	
	//printf("%d \n",Data_Layer_CPU[227*227*3-2]);
	if (!pFile1)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
}*/

void LoadInput(int *Data_Layer_CPU)
{
	FILE *fp = fopen ("../../data-SN//cat.txt","rb");
	size_t len;
        char delim[1];
        delim[0] = '\n';
	int count = 0;
	char *token;
        char *line = NULL;
	if (fp != NULL)
	{
		//printf("File Found\n");
		{
			while ((getline(&line, &len, fp)) != -1)
			{
                                token = strtok(line,delim);
                                Data_Layer_CPU[count] = atof(token);
				count++;		
			}
			//printf("Read completed, Count :: %d\n",count);		
		}
		fclose(fp);
	}
	else
	{
		printf(" File NOt FOUND\n");
	}
	
}


/////////////////////////////////////////////////////////////////////////////////////////
// Split the RGB array to separate R, G amd B channel arrays
/////////////////////////////////////////////////////////////////////////////////////////
void ConvertInput(int *Data_Layer_CPU_R, int *Data_Layer_CPU_G, int *Data_Layer_CPU_B, int *Data_Layer_CPU)
{
	for(int i=0; i<227*227*3; i+=3)
	{
		Data_Layer_CPU_R[i/3] = Data_Layer_CPU[i];
		Data_Layer_CPU_G[i/3] = Data_Layer_CPU[i+1];
		Data_Layer_CPU_B[i/3] = Data_Layer_CPU[i+2];
		//printf("%d ",Data_Layer_CPU_R[i/3]);
	}
}


/////////////////////////////////////////////////////////////////////////////////////////
// Read all the weights from the weight files for all layers to the intialised host memory
/////////////////////////////////////////////////////////////////////////////////////////
void InitHostMem(double *Layer1_Weights_CPU, double *fire2squeeze1x1_Weights_CPU, double *fire2expand1x1_Weights_CPU, double *fire2expand3x3_Weights_CPU, double *fire3squeeze1x1_Weights_CPU, double *fire3expand1x1_Weights_CPU, double *fire3expand3x3_Weights_CPU, double *fire4squeeze1x1_Weights_CPU, double *fire4expand1x1_Weights_CPU, double *fire4expand3x3_Weights_CPU, double *fire5squeeze1x1_Weights_CPU, double *fire5expand1x1_Weights_CPU, double *fire5expand3x3_Weights_CPU, double *fire6squeeze1x1_Weights_CPU, double *fire6expand1x1_Weights_CPU, double *fire6expand3x3_Weights_CPU, double *fire7squeeze1x1_Weights_CPU, double *fire7expand1x1_Weights_CPU, double *fire7expand3x3_Weights_CPU, double *fire8squeeze1x1_Weights_CPU, double *fire8expand1x1_Weights_CPU, double *fire8expand3x3_Weights_CPU, double *fire9squeeze1x1_Weights_CPU, double *fire9expand1x1_Weights_CPU, double *fire9expand3x3_Weights_CPU, double *Layer10_Weights_CPU)
{
	// initial layer 1 weight
	FILE * pFile1 = fopen ("../../data-SN//conv1_s.txt","rb");
	if (pFile1 != NULL)
	{
		//printf("File Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile1);
		//printf("Done2\n");
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
			if(i==14112)
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
	FILE * pFile2 = fopen ("../../data-SN//fire2_squeeze1x1.txt","rb");
	if (pFile2 != NULL)
	{
		//printf("File2 Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile2);
		//printf("Done2\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			fire2squeeze1x1_Weights_CPU[i] = temp_num;
			i++;
			index++;
			if(i==1536)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//fclose (pFile2);
	}	
	
	if (!pFile2)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile3 = fopen ("../../data-SN//fire2_expand1x1.txt","rb");
	if (pFile3 != NULL)
	{
		//printf("File3 Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile3);
		//printf("Done2\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			fire2expand1x1_Weights_CPU[i] = temp_num;
			//printf("%.8f ",temp_num);
			i++;
			index++;
			if(i==1024)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//fclose (pFile3);
	}	
	
	if (!pFile3)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile4 = fopen ("../../data-SN//fire2_expand3x3.txt","rb");
	if (pFile4 != NULL)
	{
		//printf("File4 Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile4);
		//printf("Done4 %s\n",s);
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			fire2expand3x3_Weights_CPU[i] = temp_num;
			//printf("%.8f ",temp_num);
			i++;
			index++;
			if(i==9216)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//printf("w: %.8f\n",fire2expand3x3_Weights_CPU[0]);
		//fclose (pFile3);
	}	
	
	if (!pFile4)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile5 = fopen ("../../data-SN//fire3_squeeze1x1.txt","rb");
	if (pFile5 != NULL)
	{
		//printf("File5 Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile5);
		//printf("Done2\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			fire3squeeze1x1_Weights_CPU[i] = temp_num;
			i++;
			index++;
			if(i==2048)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//fclose (pFile2);
	}	
	
	if (!pFile5)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile6 = fopen ("../../data-SN//fire3_expand1x1.txt","rb");
	if (pFile6 != NULL)
	{
		//printf("File6 Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile6);
		//printf("Done2\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			fire3expand1x1_Weights_CPU[i] = temp_num;
			//printf("%.8f ",temp_num);
			i++;
			index++;
			if(i==1024)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//fclose (pFile3);
	}	
	
	if (!pFile6)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile7 = fopen ("../../data-SN//fire3_expand3x3.txt","rb");
	if (pFile7 != NULL)
	{
		//printf("File7 Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile7);
		//printf("Done4 %s\n",s);
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			fire3expand3x3_Weights_CPU[i] = temp_num;
			//printf("%.8f ",temp_num);
			i++;
			index++;
			if(i==9216)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//printf("w: %.8f\n",fire3expand3x3_Weights_CPU[0]);
		//fclose (pFile3);
	}	
	
	if (!pFile7)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile8 = fopen ("../../data-SN//fire4_squeeze1x1.txt","rb");
	if (pFile8 != NULL)
	{
		//printf("File8 Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile8);
		//printf("Done2\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			fire4squeeze1x1_Weights_CPU[i] = temp_num;
			i++;
			index++;
			if(i==4096)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//fclose (pFile2);
	}	
	
	if (!pFile8)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile9 = fopen ("../../data-SN//fire4_expand1x1.txt","rb");
	if (pFile9 != NULL)
	{
		//printf("File9 Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile9);
		//printf("Done2\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			fire4expand1x1_Weights_CPU[i] = temp_num;
			//printf("%.8f ",temp_num);
			i++;
			index++;
			if(i==4096)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//fclose (pFile3);
	}	
	
	if (!pFile9)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile10 = fopen ("../../data-SN//fire4_expand3x3.txt","rb");
	if (pFile10 != NULL)
	{
		//printf("File10 Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile10);
		//printf("Done4 %s\n",s);
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			fire4expand3x3_Weights_CPU[i] = temp_num;
			//printf("%.8f ",temp_num);
			i++;
			index++;
			if(i==36864)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//printf("w: %.8f\n",fire4expand3x3_Weights_CPU[0]);
		//fclose (pFile3);
	}	
	
	if (!pFile10)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile11 = fopen ("../../data-SN//fire5_squeeze1x1.txt","rb");
	if (pFile11 != NULL)
	{
		//printf("File11 Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile11);
		//printf("Done2\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			fire5squeeze1x1_Weights_CPU[i] = temp_num;
			i++;
			index++;
			if(i==8192)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//fclose (pFile2);
	}	
	
	if (!pFile11)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile12 = fopen ("../../data-SN//fire5_expand1x1.txt","rb");
	if (pFile12 != NULL)
	{
		//printf("File12 Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile12);
		//printf("Done2\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			fire5expand1x1_Weights_CPU[i] = temp_num;
			//printf("%.8f ",temp_num);
			i++;
			index++;
			if(i==4096)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//fclose (pFile3);
	}	
	
	if (!pFile12)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile13 = fopen ("../../data-SN//fire5_expand3x3.txt","rb");
	if (pFile13 != NULL)
	{
		//printf("File13 Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile13);
		//printf("Done4 %s\n",s);
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			fire5expand3x3_Weights_CPU[i] = temp_num;
			//printf("%.8f ",temp_num);
			i++;
			index++;
			if(i==36864)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//printf("w: %.8f\n",fire5expand3x3_Weights_CPU[0]);
		//fclose (pFile3);
	}	
	
	if (!pFile13)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile14 = fopen ("../../data-SN//fire6_squeeze1x1.txt","rb");
	if (pFile14 != NULL)
	{
		//printf("File14 Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile14);
		//printf("Done2\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			fire6squeeze1x1_Weights_CPU[i] = temp_num;
			i++;
			index++;
			if(i==12288)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//fclose (pFile2);
	}	
	
	if (!pFile14)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile15 = fopen ("../../data-SN//fire6_expand1x1.txt","rb");
	if (pFile15 != NULL)
	{
		//printf("File15 Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile15);
		//printf("Done2\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			fire6expand1x1_Weights_CPU[i] = temp_num;
			//printf("%.8f ",temp_num);
			i++;
			index++;
			if(i==9216)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//fclose (pFile3);
	}	
	
	if (!pFile15)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile16 = fopen ("../../data-SN//fire6_expand3x3.txt","rb");
	if (pFile16 != NULL)
	{
		//printf("File16 Opened\n");
		char s[3000000] = "";
		fread(s,sizeof(s),1,pFile16);
		//printf("Done4 %s\n",s);
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			fire6expand3x3_Weights_CPU[i] = temp_num;
			//printf("%.8f ",temp_num);
			i++;
			index++;
			if(i==82944)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//printf("%ld w: %.8f\n",i,fire6expand3x3_Weights_CPU[0]);
		//fclose (pFile3);
	}	
	
	if (!pFile16)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile17 = fopen ("../../data-SN//fire7_squeeze1x1.txt","rb");
	if (pFile17 != NULL)
	{
		//printf("File17 Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile17);
		//printf("Done2\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			fire7squeeze1x1_Weights_CPU[i] = temp_num;
			i++;
			index++;
			if(i==18432)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//fclose (pFile2);
	}	
	
	if (!pFile17)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile18 = fopen ("../../data-SN//fire7_expand1x1.txt","rb");
	if (pFile18 != NULL)
	{
		//printf("File18 Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile18);
		//printf("Done2\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			fire7expand1x1_Weights_CPU[i] = temp_num;
			//printf("%.8f ",temp_num);
			i++;
			index++;
			if(i==9216)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//fclose (pFile3);
	}	
	
	if (!pFile18)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile19 = fopen ("../../data-SN//fire7_expand3x3.txt","rb");
	if (pFile19 != NULL)
	{
		//printf("File19 Opened\n");
		char s[3000000] = "";
		fread(s,sizeof(s),1,pFile19);
		//printf("Done4 %s\n",s);
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			fire7expand3x3_Weights_CPU[i] = temp_num;
			//printf("%.8f ",temp_num);
			i++;
			index++;
			if(i==82944)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//printf("w: %.8f\n",fire7expand3x3_Weights_CPU[0]);
		//fclose (pFile3);
	}	
	
	if (!pFile19)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile20 = fopen ("../../data-SN//fire8_squeeze1x1.txt","rb");
	if (pFile20 != NULL)
	{
		//printf("File18 Opened\n");
		char s[3000000] = "";
		fread(s,sizeof(s),1,pFile20);
		//printf("Done2\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			fire8squeeze1x1_Weights_CPU[i] = temp_num;
			i++;
			index++;
			if(i==24640)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//fclose (pFile2);
	}	
	
	if (!pFile20)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	/*FILE * pFile20 = fopen ("../../data-SN//fire8_squeeze1x1.txt","rb");
	if (pFile20 != NULL)
	{
		printf("File20 Opened\n");
		char s[2000000] = "";
		fread(s,sizeof(s),1,pFile20);
		printf("Done2\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			fire8squeeze1x1_Weights_CPU[i] = temp_num;
			i++;
			index++;
			if(i==24640)
			{
				printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		printf("w: %.8f\n",fire8squeeze1x1_Weights_CPU[24575]);
		//fclose (pFile2);
	}	
	
	if (!pFile20)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}*/
	FILE * pFile21 = fopen ("../../data-SN//fire8_expand1x1.txt","rb");
	if (pFile21 != NULL)
	{
		//printf("File21 Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile21);
		//printf("Done2\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			fire8expand1x1_Weights_CPU[i] = temp_num;
			//printf("%.8f ",temp_num);
			i++;
			index++;
			if(i==16384)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//printf("w: %ld\n",i);
		//fclose (pFile3);
	}	
	
	if (!pFile21)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile22 = fopen ("../../data-SN//fire8_expand3x3.txt","rb");
	if (pFile22 != NULL)
	{
		//printf("File22 Opened\n");
		char s[6000000] = "";
		fread(s,sizeof(s),1,pFile22);
		//printf("Done4 %s\n",s);
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			fire8expand3x3_Weights_CPU[i] = temp_num;
			//printf("%.8f ",temp_num);
			i++;
			index++;
			if(i==147456)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//printf("w8: %.8f\n",fire8expand3x3_Weights_CPU[74206]);
		//fclose (pFile3);
	}	
	
	if (!pFile22)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile23 = fopen ("../../data-SN//fire9_squeeze1x1.txt","rb");
	if (pFile23 != NULL)
	{
		//printf("File23 Opened\n");
		char s[3000000] = "";
		fread(s,sizeof(s),1,pFile23);
		//printf("Done2\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			fire9squeeze1x1_Weights_CPU[i] = temp_num;
			i++;
			index++;
			if(i==32768)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//fclose (pFile2);
	}	
	
	if (!pFile23)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile24 = fopen ("../../data-SN//fire9_expand1x1.txt","rb");
	if (pFile24 != NULL)
	{
		//printf("File24 Opened\n");
		char s[1000000] = "";
		fread(s,sizeof(s),1,pFile24);
		//printf("Done2\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			fire9expand1x1_Weights_CPU[i] = temp_num;
			//printf("%.8f ",temp_num);
			i++;
			index++;
			if(i==16384)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//printf("w: %ld\n",i);
		//fclose (pFile3);
	}	
	
	if (!pFile24)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}
	FILE * pFile25 = fopen ("../../data-SN//fire9_expand3x3.txt","rb");
	if (pFile25 != NULL)
	{
		//printf("File25 Opened\n");
		char s[6000000] = "";
		fread(s,sizeof(s),1,pFile25);
		//printf("Done4 %s\n",s);
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			fire9expand3x3_Weights_CPU[i] = temp_num;
			//printf("%.8f ",temp_num);
			i++;
			index++;
			if(i==147456)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//printf("w: %.8f\n",fire9expand3x3_Weights_CPU[147455]);
		//fclose (pFile3);
	}	
	
	if (!pFile25)
	{
		printf("FAIL! INPUT WEIGHTS NOT FOUND!\n");
		exit(1);
	}

	FILE * pFile26 = fopen ("../../data-SN//conv10_s_p1.txt","rb");
	if (pFile26 != NULL)
	{
		//printf("File 26 Opened\n");
		char s[8200000] = "";
		//char *s;
		//double *Layer10_Weights_CPU = (double*) malloc (512000* NUM * sizeof(double));
		fread(s,sizeof(s),1,pFile26);
		//printf("Done2\n");
		long int index = 0, i = 0;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			Layer10_Weights_CPU[i] = temp_num;
			i++;
			index++;
			if(i==256000)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//printf("Elements read p1: %ld\n",i);
		fclose (pFile26);
	}
	FILE * pFile27 = fopen ("../../data-SN//conv10_s_p2.txt","rb");
	if (pFile27 != NULL)
	{
		//printf("File 27 Opened\n");
		char s[8200000] = "";
		//char *s;
		//double *Layer10_Weights_CPU = (double*) malloc (512000* NUM * sizeof(double));
		fread(s,sizeof(s),1,pFile27);
		//printf("Done2\n");
		long int index = 0, i = 256000;
		char delim[2];
		delim[0] = '\n';
    		delim[1] = 0;
		char* temp_string = strtok(s, delim);
		while(temp_string != NULL)
		{ 	
			double temp_num = atof(temp_string);
			Layer10_Weights_CPU[i] = temp_num;
			i++;
			index++;
			if(i==512000)
			{
				//printf("Breaking\n");
				break;
			}
			temp_string = strtok(NULL, delim);
		}
		//printf("Elements read: %ld, Last element: %.8f\n",i,Layer10_Weights_CPU[i-1]);
		fclose (pFile27);
	}
	
	/*FILE *pFile26 = fopen ("../../data-SN//conv10_s.txt","rb");
	size_t len;
        char delim[1];
        delim[0] = '\n';
	int count = 0;
	char *token;
        char *line = NULL;
	if (pFile26 != NULL)
	{
		printf("File Found\n");
		{
			while (((getline(&line, &len, pFile26)) != -1) || count<51000)
			{
                                token = strtok(line,delim);
                                Layer10_Weights_CPU[count] = atof(token);
				count++;		
			}
			printf("Read completed, Count :: %d\n",count);		
		}
		fclose(pFile26);
	}
	else
	{
		printf(" File NOt FOUND\n");
	}*/
}
#define num_sm 8
/////////////////////////////////////////////////////////////////////////////////////////
// Device function to execute first convolutional layer
/////////////////////////////////////////////////////////////////////////////////////////
__global__ void ExecuteFirstLayer(double *Layer1_Weights_CPU, int *Data_Layer_CPU_R, int *Data_Layer_CPU_G, int *Data_Layer_CPU_B, double *Layer1_Features, unsigned long long run_time)
{

	int smid;
    asm("mov.u32 %0, %smid;" : "=r"(smid));
    if(smid >= num_sm) {
        return;
    }

	extern __shared__ unsigned long long clock_begin;   //clock value kernel marks as its start time
    extern __shared__ unsigned long long clock_now;     //clock value current instant

    
    clock_begin = gclock64();
    clock_now = clock_begin;
    while ((clock_now - clock_begin) < (run_time+100)) {


	int x = (threadIdx.x)*2 + 3;
	int y = (blockIdx.x)*2 + 3;
	//int f=0;
	for(int f=0; f<96; f++)
	{
				double result = 0;
				for(int i = x-3; i<=x+3; i++)
				{
    					for(int j=y-3; j<=y+3; j++)
    					{
						int x_index = i-x+3;
						int y_index = j-y+3;
						int m = (y_index)+(x_index)*7;
         					if(i<0 || j<0)
						{
							result+= 0;
							//printf("Error %d\n",m);						
						}
         					else if(j>226 || i>226)
						{
							result+= 0;
							//printf("Error %d\n",m);
						}
         					else
						{
							double temp = Data_Layer_CPU_R[(y_index-3) + x*227 + y + (x_index-3)*227]*Layer1_Weights_CPU[m+f*147] + Data_Layer_CPU_G[(y_index-3) + x*227 + y + (x_index-3)*227]*Layer1_Weights_CPU[m+49+f*147] + Data_Layer_CPU_B[(y_index-3) + x*227 + y + (x_index-3)*227]*Layer1_Weights_CPU[m+98+f*147];			
							result+= temp;
							//printf("%.8f %d	",temp, m);
						}
					}
				} 
				if(result < 0)
					result = 0;
				Layer1_Features[f*111*111+((x-3)/2)*111+((y-3)/2)] = result;
				//printf("%0.8f ",result);
	}
	//if(x==4 && y==4)
	//	printf("%.8f ",Layer1_Features[1]);
	clock_now=gclock64();
	}
}

__global__ void pooling1(double *Layer2_Neurons_GPU,double *Layer2_pool_GPU,int out,int out_fr,int out_fc,int kernel,int stride_width,int in_fr,int in_fc)
{
    int row = threadIdx.x;
    int col = blockIdx.x;
    double max = 0.0;
    {
        for(int output =0;output < 96 ;output++)
        {
            if(row%2 != 0)
            { 
                if(col%2 != 0)
                {
                    for(int i = row-1; i <= row+1; i++)
                    {   
			if(i>110) break;        
                        for(int j = col-1; j <= col+1; j++)
                        {
			    if(j>110) break;
                            if(max < ((Layer2_Neurons_GPU[output*111*111+i*111+j])))
                                max =   ((Layer2_Neurons_GPU[output*111*111+i*111+j])) ;
 
                        }
                    }
                    Layer2_pool_GPU[output*55*55+((row-1)/2)*55+(col-1)/2] = max;
                    max = 0.0;   
                }
            }
        }
    }
    __syncthreads();
    //if(row == 1 && col == 1)
	//printf("Max pool o/p: %.8f\n", Layer2_pool_GPU[290399]);
}

__global__ void Executefire2squeeze1x1(double *fire2squeeze1x1_Weights_GPU, double *fire2squeeze1x1_Features, double *Layer2_pool_GPU)
{
	double Features = 0;
	int x = threadIdx.x;
	int y = blockIdx.x;
	for(int f=0; f<16; f++)
	{
		Features = 0;
		for(int n=0; n<96; n++)
		{
			//double result = 0;
			//printf("(%d, %d) ", f,n);
               		Features+= Layer2_pool_GPU[n*55*55 + x*55 + y]*fire2squeeze1x1_Weights_GPU[f*96+n];
		}
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		fire2squeeze1x1_Features[f*55*55 + x*55 + y] = Features;
		//printf("%.8f ",Features);
	}
	__syncthreads();
	//if(x == 0 && y == 0)
		//printf("Fire2 Squeeze o/p: %.8f", fire2squeeze1x1_Features[15149]);
}

__global__ void Executefire2expand1x1(double *fire2expand1x1_Weights_GPU, double *fire2_Features, double *fire2squeeze1x1_Features)
{
	double Features = 0;
	int x = threadIdx.x;
	int y = blockIdx.x;
	for(int f=0; f<64; f++)
	{
		Features = 0;
		for(int n=0; n<16; n++)
		{
			double result = 0;
               		result = fire2squeeze1x1_Features[n*55*55 + x*55 + y]*fire2expand1x1_Weights_GPU[f*16+n];
			//if(x==0 && y==0 && f==0)
				//printf("(%.8f,%.8f) ", result, fire2expand1x1_Weights_GPU[f*16+n]);
			Features+= result;
		}
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		fire2_Features[f*55*55 + x*55 + y] = Features;
	}
	__syncthreads();
	//if(x == 0 && y == 0)
		//printf("\n%.8f \n", fire2_Features[187736]);
}

__global__ void Executefire2expand3x3(double *fire2expand3x3_Weights_GPU, double *fire2_Features, double *fire2squeeze1x1_Features)
{
	double Features = 0;
	int x = threadIdx.x;
	int y = blockIdx.x;
	for(int f=0; f<64; f++)
	{
		Features = 0;
		for(int n=0; n<16; n++)
		{	double result = 0;
				for(int i = x-1; i<=x+1; i++)
				{
    					for(int j=y-1; j<=y+1; j++)
    					{
						int x_index = i-x+1;
						int y_index = j-y+1;
						int m = (y_index)+(x_index)*3;
         					if(i<0 || j<0)
						{
							result+=0;
						}
         					else if(j>54 || i>54)
						{
							result+=0;	
						}
         					else
						{
               						result+= fire2squeeze1x1_Features[n*55*55 + i*55 + j]*fire2expand3x3_Weights_GPU[m+f*9*16+n*9];			
						}
					}
				} 
				Features += result;
		}
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		fire2_Features[f*55*55 + x*55 + y] = Features;
	}
}

__global__ void Executefire3squeeze1x1(double *fire3squeeze1x1_Weights_GPU, double *fire3squeeze1x1_Features, double *fire2_Features)
{
	double Features = 0;
	int x = threadIdx.x;
	int y = blockIdx.x;
	for(int f=0; f<16; f++)
	{
		Features = 0;
		for(int n=0; n<128; n++)
		{
			//double result = 0;
			//printf("(%d, %d) ", f,n);
               		Features+= fire2_Features[n*55*55 + x*55 + y]*fire3squeeze1x1_Weights_GPU[f*128+n];
		}
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		fire3squeeze1x1_Features[f*55*55 + x*55 + y] = Features;
		//printf("%.8f ",Features);
	}
	__syncthreads();
	//if(x == 0 && y == 0)
		//printf("Fire3 Squeeze o/p: %.8f", fire3squeeze1x1_Features[48399]);
}

__global__ void Executefire3expand1x1(double *fire3expand1x1_Weights_GPU, double *fire3_Features, double *fire3squeeze1x1_Features)
{
	double Features = 0;
	int x = threadIdx.x;
	int y = blockIdx.x;
	for(int f=0; f<64; f++)
	{
		Features = 0;
		for(int n=0; n<16; n++)
		{
			double result = 0;
               		result = fire3squeeze1x1_Features[n*55*55 + x*55 + y]*fire3expand1x1_Weights_GPU[f*16+n];
			//if(x==0 && y==0 && f==0)
				//printf("(%.8f,%.8f) ", result, fire2expand1x1_Weights_GPU[f*16+n]);
			Features+= result;
		}
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		fire3_Features[f*55*55 + x*55 + y] = Features;
	}
	__syncthreads();
	//if(x == 0 && y == 0)
		//printf("\nFire 3 expand: %.8f \n", fire3_Features[164425]);
}

__global__ void Executefire3expand3x3(double *fire3expand3x3_Weights_GPU, double *fire3_Features, double *fire3squeeze1x1_Features)
{
	double Features = 0;
	int x = threadIdx.x;
	int y = blockIdx.x;
	for(int f=0; f<64; f++)
	{
		Features = 0;
		for(int n=0; n<16; n++)
		{	double result = 0;
				for(int i = x-1; i<=x+1; i++)
				{
    					for(int j=y-1; j<=y+1; j++)
    					{
						int x_index = i-x+1;
						int y_index = j-y+1;
						int m = (y_index)+(x_index)*3;
         					if(i<0 || j<0)
						{
							result+=0;
						}
         					else if(j>54 || i>54)
						{
							result+=0;	
						}
         					else
						{
               						result+= fire3squeeze1x1_Features[n*55*55 + i*55 + j]*fire3expand3x3_Weights_GPU[m+f*9*16+n*9];			
						}
					}
				} 
				Features += result;
		}
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		fire3_Features[f*55*55 + x*55 + y] = Features;
	}
	__syncthreads();
	//if(x == 0 && y == 0)
		//printf("Fire 3 expand3x3: %.8f \n", fire3_Features[6270]);
}

__global__ void Executefire4squeeze1x1(double *fire4squeeze1x1_Weights_GPU, double *fire4squeeze1x1_Features, double *fire3_Features)
{
	double Features = 0;
	int x = threadIdx.x;
	int y = blockIdx.x;
	for(int f=0; f<32; f++)
	{
		Features = 0;
		for(int n=0; n<128; n++)
		{
			//double result = 0;
			//printf("(%d, %d) ", f,n);
               		Features+= fire3_Features[n*55*55 + x*55 + y]*fire4squeeze1x1_Weights_GPU[f*128+n];
		}
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		fire4squeeze1x1_Features[f*55*55 + x*55 + y] = Features;
		//printf("%.8f ",Features);
	}
	__syncthreads();
	//if(x == 0 && y == 0)
		//printf("Fire4 Squeeze o/p: %.8f", fire4squeeze1x1_Features[1067]);
}

__global__ void Executefire4expand1x1(double *fire4expand1x1_Weights_GPU, double *fire4_Features, double *fire4squeeze1x1_Features)
{
	double Features = 0;
	int x = threadIdx.x;
	int y = blockIdx.x;
	for(int f=0; f<128; f++)
	{
		Features = 0;
		for(int n=0; n<32; n++)
		{
			double result = 0;
               		result = fire4squeeze1x1_Features[n*55*55 + x*55 + y]*fire4expand1x1_Weights_GPU[f*32+n];
			//if(x==0 && y==0 && f==0)
				//printf("(%.8f,%.8f) ", result, fire2expand1x1_Weights_GPU[f*16+n]);
			Features+= result;
		}
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		fire4_Features[f*55*55 + x*55 + y] = Features;
	}
	__syncthreads();
	//if(x == 0 && y == 0)
		//printf("\nFire 4 expand: %.8f \n", fire4_Features[164425]);
}

__global__ void Executefire4expand3x3(double *fire4expand3x3_Weights_GPU, double *fire4_Features, double *fire4squeeze1x1_Features)
{
	double Features = 0;
	int x = threadIdx.x;
	int y = blockIdx.x;
	for(int f=0; f<128; f++)
	{
		Features = 0;
		for(int n=0; n<32; n++)
		{	double result = 0;
				for(int i = x-1; i<=x+1; i++)
				{
    					for(int j=y-1; j<=y+1; j++)
    					{
						int x_index = i-x+1;
						int y_index = j-y+1;
						int m = (y_index)+(x_index)*3;
         					if(i<0 || j<0)
						{
							result+=0;
						}
         					else if(j>54 || i>54)
						{
							result+=0;	
						}
         					else
						{
               						result+= fire4squeeze1x1_Features[n*55*55 + i*55 + j]*fire4expand3x3_Weights_GPU[m+f*9*32+n*9];			
						}
					}
				} 
				Features += result;
		}
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		fire4_Features[f*55*55 + x*55 + y] = Features;
	}
	__syncthreads();
	//if(x == 0 && y == 0)
		//printf("Fire 4 expand3x3: %.8f \n", fire4_Features[191520]);
}

__global__ void pooling4(double *Layer4_Neurons_GPU,double *Layer4_pool_GPU,int out,int out_fr,int out_fc,int kernel,int stride_width,int in_fr,int in_fc)
{
    int row = threadIdx.x;
    int col = blockIdx.x;
    double max = 0.0;
    {
        for(int output =0;output < 256 ;output++)
        {
            if(row%2 != 0)
            { 
                if(col%2 != 0)
                {
                    for(int i = row-1; i <= row+1; i++)
                    {   
			if(i>54) break;        
                        for(int j = col-1; j <= col+1; j++)
                        {
			    if(j>54) break;
                            if(max < ((Layer4_Neurons_GPU[output*55*55+i*55+j])))
                                max =   ((Layer4_Neurons_GPU[output*55*55+i*55+j])) ;
 
                        }
                    }
                    Layer4_pool_GPU[output*27*27+((row-1)/2)*27+(col-1)/2] = max;
                    max = 0.0;   
                }
            }
        }
    }
    __syncthreads();
   // if(row == 1 && col == 1)
	//printf("Max pool 4 o/p: %.8f\n", Layer4_pool_GPU[90399]);
}

__global__ void Executefire5squeeze1x1(double *fire5squeeze1x1_Weights_GPU, double *fire5squeeze1x1_Features, double *fire4_Features)
{
	double Features = 0;
	int x = threadIdx.x;
	int y = blockIdx.x;
	for(int f=0; f<32; f++)
	{
		Features = 0;
		for(int n=0; n<256; n++)
		{
			//double result = 0;
			//printf("(%d, %d) ", f,n);
               		Features+= fire4_Features[n*27*27 + x*27 + y]*fire5squeeze1x1_Weights_GPU[f*256+n];
		}
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		fire5squeeze1x1_Features[f*27*27 + x*27 + y] = Features;
		//printf("%.8f ",Features);
	}
	__syncthreads();
	//if(x == 0 && y == 0)
		//printf("Fire5 Squeeze o/p: %.8f", fire5squeeze1x1_Features[1067]);
}

__global__ void Executefire5expand1x1(double *fire5expand1x1_Weights_GPU, double *fire5_Features, double *fire5squeeze1x1_Features)
{
	double Features = 0;
	int x = threadIdx.x;
	int y = blockIdx.x;
	for(int f=0; f<128; f++)
	{
		Features = 0;
		for(int n=0; n<32; n++)
		{
			double result = 0;
               		result = fire5squeeze1x1_Features[n*27*27 + x*27 + y]*fire5expand1x1_Weights_GPU[f*32+n];
			//if(x==0 && y==0 && f==0)
				//printf("(%.8f,%.8f) ", result, fire2expand1x1_Weights_GPU[f*16+n]);
			Features+= result;
		}
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		fire5_Features[f*27*27 + x*27+ y] = Features;
	}
	__syncthreads();
	//if(x == 0 && y == 0)
		//printf("\nFire 5 expand: %.8f \n", fire5_Features[1]);
}

__global__ void Executefire5expand3x3(double *fire5expand3x3_Weights_GPU, double *fire5_Features, double *fire5squeeze1x1_Features)
{
	double Features = 0;
	int x = threadIdx.x;
	int y = blockIdx.x;
	for(int f=0; f<128; f++)
	{
		Features = 0;
		for(int n=0; n<32; n++)
		{	double result = 0;
				for(int i = x-1; i<=x+1; i++)
				{
    					for(int j=y-1; j<=y+1; j++)
    					{
						int x_index = i-x+1;
						int y_index = j-y+1;
						int m = (y_index)+(x_index)*3;
         					if(i<0 || j<0)
						{
							result+=0;
						}
         					else if(j>26 || i>26)
						{
							result+=0;	
						}
         					else
						{
               						result+= fire5squeeze1x1_Features[n*27*27 + i*27 + j]*fire5expand3x3_Weights_GPU[m+f*9*32+n*9];			
						}
					}
				} 
				Features += result;
		}
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		fire5_Features[f*27*27 + x*27 + y] = Features;
	}
	__syncthreads();
	//if(x == 0 && y == 0)
		//printf("Fire 5 expand3x3: %.8f \n", fire5_Features[6]);
}

__global__ void Executefire6squeeze1x1(double *fire6squeeze1x1_Weights_GPU, double *fire6squeeze1x1_Features, double *fire5_Features)
{
	double Features = 0;
	int x = threadIdx.x;
	int y = blockIdx.x;
	for(int f=0; f<48; f++)
	{
		Features = 0;
		for(int n=0; n<256; n++)
		{
			//double result = 0;
			//printf("(%d, %d) ", f,n);
               		Features+= fire5_Features[n*27*27 + x*27 + y]*fire6squeeze1x1_Weights_GPU[f*256+n];
		}
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		fire6squeeze1x1_Features[f*27*27 + x*27 + y] = Features;
		//printf("%.8f ",Features);
	}
	__syncthreads();
	//if(x == 0 && y == 0)
		//printf("Fire6 Squeeze o/p: %.8f", fire6squeeze1x1_Features[1065]);
}

__global__ void Executefire6expand1x1(double *fire6expand1x1_Weights_GPU, double *fire6_Features, double *fire6squeeze1x1_Features)
{
	double Features = 0;
	int x = threadIdx.x;
	int y = blockIdx.x;
	for(int f=0; f<192; f++)
	{
		Features = 0;
		for(int n=0; n<48; n++)
		{
			double result = 0;
               		result = fire6squeeze1x1_Features[n*27*27 + x*27 + y]*fire6expand1x1_Weights_GPU[f*48+n];
			//if(x==0 && y==0 && f==0)
				//printf("(%.8f,%.8f) ", result, fire2expand1x1_Weights_GPU[f*16+n]);
			Features+= result;
		}
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		fire6_Features[f*27*27 + x*27+ y] = Features;
	}
	__syncthreads();
	//if(x == 0 && y == 0)
		//printf("\nFire 6 expand: %.8f \n", fire6_Features[4414]);
}

__global__ void Executefire6expand3x3(double *fire6expand3x3_Weights_GPU, double *fire6_Features, double *fire6squeeze1x1_Features)
{
	double Features = 0;
	int x = threadIdx.x;
	int y = blockIdx.x;
	for(int f=0; f<192; f++)
	{
		Features = 0;
		for(int n=0; n<48; n++)
		{	double result = 0;
				for(int i = x-1; i<=x+1; i++)
				{
    					for(int j=y-1; j<=y+1; j++)
    					{
						int x_index = i-x+1;
						int y_index = j-y+1;
						int m = (y_index)+(x_index)*3;
         					if(i<0 || j<0)
						{
							result+=0;
						}
         					else if(j>26 || i>26)
						{
							result+=0;	
						}
         					else
						{
               						result+= fire6squeeze1x1_Features[n*27*27 + i*27 + j]*fire6expand3x3_Weights_GPU[m+f*9*48+n*9];			
						}
					}
				} 
				Features += result;
		}
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		fire6_Features[f*27*27 + x*27 + y] = Features;
	}
	__syncthreads();
	//if(x == 0 && y == 0)
		//printf("Fire 6 expand3x3: %.8f \n", fire6_Features[21]);
}

__global__ void Executefire7squeeze1x1(double *fire7squeeze1x1_Weights_GPU, double *fire7squeeze1x1_Features, double *fire6_Features)
{
	double Features = 0;
	int x = threadIdx.x;
	int y = blockIdx.x;
	for(int f=0; f<48; f++)
	{
		Features = 0;
		for(int n=0; n<384; n++)
		{
			//double result = 0;
			//printf("(%d, %d) ", f,n);
               		Features+= fire6_Features[n*27*27 + x*27 + y]*fire7squeeze1x1_Weights_GPU[f*384+n];
		}
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		fire7squeeze1x1_Features[f*27*27 + x*27 + y] = Features;
		//printf("%.8f ",Features);
	}
	__syncthreads();
	//if(x == 0 && y == 0)
		//printf("Fire7 Squeeze o/p: %.8f", fire7squeeze1x1_Features[1065]);
}

__global__ void Executefire7expand1x1(double *fire7expand1x1_Weights_GPU, double *fire7_Features, double *fire7squeeze1x1_Features)
{
	double Features = 0;
	int x = threadIdx.x;
	int y = blockIdx.x;
	for(int f=0; f<192; f++)
	{
		Features = 0;
		for(int n=0; n<48; n++)
		{
			double result = 0;
               		result = fire7squeeze1x1_Features[n*27*27 + x*27 + y]*fire7expand1x1_Weights_GPU[f*48+n];
			//if(x==0 && y==0 && f==0)
				//printf("(%.8f,%.8f) ", result, fire2expand1x1_Weights_GPU[f*16+n]);
			Features+= result;
		}
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		fire7_Features[f*27*27 + x*27+ y] = Features;
	}
	__syncthreads();
	//if(x == 0 && y == 0)
		//printf("\nFire 7 expand: %.8f \n", fire7_Features[4414]);
}

__global__ void Executefire7expand3x3(double *fire7expand3x3_Weights_GPU, double *fire7_Features, double *fire7squeeze1x1_Features)
{
	double Features = 0;
	int x = threadIdx.x;
	int y = blockIdx.x;
	for(int f=0; f<192; f++)
	{
		Features = 0;
		for(int n=0; n<48; n++)
		{	double result = 0;
				for(int i = x-1; i<=x+1; i++)
				{
    					for(int j=y-1; j<=y+1; j++)
    					{
						int x_index = i-x+1;
						int y_index = j-y+1;
						int m = (y_index)+(x_index)*3;
         					if(i<0 || j<0)
						{
							result+=0;
						}
         					else if(j>26 || i>26)
						{
							result+=0;	
						}
         					else
						{
               						result+= fire7squeeze1x1_Features[n*27*27 + i*27 + j]*fire7expand3x3_Weights_GPU[m+f*9*48+n*9];			
						}
					}
				} 
				Features += result;
		}
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		fire7_Features[f*27*27 + x*27 + y] = Features;
	}
	__syncthreads();
	//if(x == 0 && y == 0)
		//printf("Fire 7 expand3x3: %.8f \n", fire7_Features[59884]);
}

/*__global__ void Executefire8squeeze1x1(double *fire8squeeze1x1_Weights_GPU, double *fire8squeeze1x1_Features, double *fire7_Features)
{
	double Features = 0;
	int x = threadIdx.x;
	int y = blockIdx.x;
	for(int f=0; f<64; f++)
	{
		Features = 0;
		for(int n=0; n<384; n++)
		{
			//double result = 0;
			//printf("(%d, %d) ", f,n);
               		Features+= fire7_Features[n*27*27 + x*27 + y]*fire8squeeze1x1_Weights_GPU[f*384+n];
		}
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		fire8squeeze1x1_Features[f*27*27 + x*27 + y] = Features;
		//printf("%.8f ",Features);
	}
	__syncthreads();
	if(x == 0 && y == 0)
		printf("Fire 8 Squeeze o/p: %.8f", fire8squeeze1x1_Features[8916]);
}*/

__global__ void Executefire8squeeze1x1(double *fire8squeeze1x1_Weights_GPU, double *fire8squeeze1x1_Features, double *fire7_Features)
{
	double Features = 0;
	int x = threadIdx.x;
	int y = blockIdx.x;
	//bool flag = 0;
	//if(x == 0 && y==1)
	//	flag = 1;
	for(int f=0; f<64; f++)
	{
		Features = 0;
		for(int n=0; n<384; n++)
		{
			//double result = 0;
			//printf("(%d, %d) ", f,n);
               		Features+= fire7_Features[n*27*27 + x*27 + y]*fire8squeeze1x1_Weights_GPU[f*384+n];
			//if(f==0 && flag)
				//printf("%.8f * %.8f = %.8f\n", fire7_Features[n*27*27 + x*27 + y], fire8squeeze1x1_Weights_GPU[f*384+n], fire7_Features[n*27*27 + x*27 + y]*fire8squeeze1x1_Weights_GPU[f*384+n]);
		}
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		fire8squeeze1x1_Features[f*27*27 + x*27 + y] = Features + fire8squeeze1x1_Weights_GPU[24576 + f];
		//printf("%.8f ",Features);
	}
	__syncthreads();
	//if(x == 0 && y == 0)
		//printf("Fire8 Squeeze o/p: %.8f", fire8squeeze1x1_Features[1]);
}

__global__ void Executefire8expand1x1(double *fire8expand1x1_Weights_GPU, double *fire8_Features, double *fire8squeeze1x1_Features)
{
	double Features = 0;
	int x = threadIdx.x;
	int y = blockIdx.x;
	for(int f=0; f<256; f++)
	{
		Features = 0;
		for(int n=0; n<64; n++)
		{
			double result = 0;
               		result = fire8squeeze1x1_Features[n*27*27 + x*27 + y]*fire8expand1x1_Weights_GPU[f*64+n];
			//if(x==0 && y==0 && f==0)
				//printf("(%.8f,%.8f) ", result, fire2expand1x1_Weights_GPU[f*16+n]);
			Features+= result;
		}
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		fire8_Features[f*27*27 + x*27+ y] = Features;
	}
	__syncthreads();
	//if(x == 0 && y == 0)
		//printf("\nFire 8 expand: %.8f \n", fire8_Features[35330]);
}

__global__ void Executefire8expand3x3(double *fire8expand3x3_Weights_GPU, double *fire8_Features, double *fire8squeeze1x1_Features)
{
	double Features = 0;
	int x = threadIdx.x;
	int y = blockIdx.x;
	for(int f=0; f<256; f++)
	{
		Features = 0;
		for(int n=0; n<64; n++)
		{	double result = 0;
				for(int i = x-1; i<=x+1; i++)
				{
    					for(int j=y-1; j<=y+1; j++)
    					{
						int x_index = i-x+1;
						int y_index = j-y+1;
						int m = (y_index)+(x_index)*3;
         					if(i<0 || j<0)
						{
							result+=0;
						}
         					else if(j>26 || i>26)
						{
							result+=0;	
						}
         					else
						{
               						result+= fire8squeeze1x1_Features[n*27*27 + i*27 + j]*fire8expand3x3_Weights_GPU[m+f*9*64+n*9];			
						}
					}
				} 
				Features += result;
		}
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		fire8_Features[f*27*27 + x*27 + y] = Features;
	}
	__syncthreads();
	//if(x == 0 && y == 0)
		//printf("Fire 8 expand3x3: %.8f \n", fire8_Features[11866]);
}

__global__ void pooling8(double *Layer8_Neurons_GPU,double *Layer8_pool_GPU,int out,int out_fr,int out_fc,int kernel,int stride_width,int in_fr,int in_fc)
{
    int row = threadIdx.x;
    int col = blockIdx.x;
    double max = 0.0;
    {
        for(int output =0;output < 512 ;output++)
        {
            if(row%2 != 0)
            { 
                if(col%2 != 0)
                {
                    for(int i = row-1; i <= row+1; i++)
                    {   
			if(i>26) break;        
                        for(int j = col-1; j <= col+1; j++)
                        {
			    if(j>26) break;
                            if(max < ((Layer8_Neurons_GPU[output*27*27+i*27+j])))
                                max =   ((Layer8_Neurons_GPU[output*27*27+i*27+j])) ;
 
                        }
                    }
                    Layer8_pool_GPU[output*13*13+((row-1)/2)*13+(col-1)/2] = max;
                    max = 0.0;   
                }
            }
        }
    }
    __syncthreads();
    //if(row == 1 && col == 1)
	//printf("Max pool 8 o/p: %.8f\n", Layer8_pool_GPU[5115]);
}

__global__ void Executefire9squeeze1x1(double *fire9squeeze1x1_Weights_GPU, double *fire9squeeze1x1_Features, double *fire8_Features)
{
	double Features = 0;
	int x = threadIdx.x;
	int y = blockIdx.x;
	//bool flag = 0;
	//if(x == 0 && y==1)
	//	flag = 1;
	for(int f=0; f<64; f++)
	{
		Features = 0;
		for(int n=0; n<512; n++)
		{
			//double result = 0;
			//printf("(%d, %d) ", f,n);
               		Features+= fire8_Features[n*13*13 + x*13 + y]*fire9squeeze1x1_Weights_GPU[f*512+n];
			//if(f==0 && flag)
				//printf("%.8f * %.8f = %.8f\n", fire7_Features[n*27*27 + x*27 + y], fire8squeeze1x1_Weights_GPU[f*384+n], fire7_Features[n*27*27 + x*27 + y]*fire8squeeze1x1_Weights_GPU[f*384+n]);
		}
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		fire9squeeze1x1_Features[f*13*13 + x*13 + y] = Features;// + fire8squeeze1x1_Weights_GPU[24576 + f];
		//printf("%.8f ",Features);
	}
	__syncthreads();
	//if(x == 0 && y == 0)
		//printf("Fire9 Squeeze o/p: %.8f", fire9squeeze1x1_Features[4]);
}

__global__ void Executefire9expand1x1(double *fire9expand1x1_Weights_GPU, double *fire9_Features, double *fire9squeeze1x1_Features)
{
	double Features = 0;
	int x = threadIdx.x;
	int y = blockIdx.x;
	for(int f=0; f<256; f++)
	{
		Features = 0;
		for(int n=0; n<64; n++)
		{
			double result = 0;
               		result = fire9squeeze1x1_Features[n*13*13 + x*13 + y]*fire9expand1x1_Weights_GPU[f*64+n];
			//if(x==0 && y==0 && f==0)
				//printf("(%.8f,%.8f) ", result, fire2expand1x1_Weights_GPU[f*16+n]);
			Features+= result;
		}
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		fire9_Features[f*13*13 + x*13+ y] = Features;
	}
	__syncthreads();
	//if(x == 0 && y == 0)
		//printf("\nFire 9 expand: %.8f \n", fire9_Features[3191]);
}

__global__ void Executefire9expand3x3(double *fire9expand3x3_Weights_GPU, double *fire9_Features, double *fire9squeeze1x1_Features)
{
	double Features = 0;
	int x = threadIdx.x;
	int y = blockIdx.x;
	for(int f=0; f<256; f++)
	{
		Features = 0;
		for(int n=0; n<64; n++)
		{	double result = 0;
				for(int i = x-1; i<=x+1; i++)
				{
    					for(int j=y-1; j<=y+1; j++)
    					{
						int x_index = i-x+1;
						int y_index = j-y+1;
						int m = (y_index)+(x_index)*3;
         					if(i<0 || j<0)
						{
							result+=0;
						}
         					else if(j>12 || i>12)
						{
							result+=0;	
						}
         					else
						{
               						result+= fire9squeeze1x1_Features[n*13*13 + i*13 + j]*fire9expand3x3_Weights_GPU[m+f*9*64+n*9];			
						}
					}
				} 
				Features += result;
		}
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		fire9_Features[f*13*13 + x*13 + y] = Features;
	}
	__syncthreads();
	//if(x == 0 && y == 0)
		//printf("Fire 9 expand3x3: %.8f \n", fire9_Features[368]);
}

__global__ void ExecuteTenthLayer(double *Layer10_Weights_GPU, double *fire9_Features, double *Layer10_Features)
{
	double Features = 0;
	int x = threadIdx.x;
	int y = blockIdx.x;
	//bool flag = 0;
	//if(x == 1 && y==1)
	//	flag = 1;
	if(x!=0 && x!=14 && y!=0 && y!=14)
	{
	for(int f=0; f<1000; f++)
	{
		Features = 0;
		for(int n=0; n<512; n++)
		{
			//double result = 0;
			//printf("(%d, %d) ", f,n);
               		Features+= fire9_Features[n*13*13 + (x-1)*13 + y-1]*Layer10_Weights_GPU[f*512+n];
			//if(f==0 && flag)
			//	printf("%.8f * %.8f = %.8f\n", fire9_Features[n*15*15 + x*15 + y], Layer10_Weights_GPU[f*512+n], fire9_Features[n*15*15 + x*15 + y]*Layer10_Weights_GPU[f*512+n]);
		}
		//ReLU activation function computation
		if(Features<0)
			Features = 0;
		Layer10_Features[f*15*15 + x*15 + y] = Features;// + fire8squeeze1x1_Weights_GPU[24576 + f];
		//printf("%.8f ",Features);
	}
	}
	__syncthreads();
	//if(x == 1 && y == 1)
		//printf("Layer10 o/p: %.8f, %d\n", Layer10_Features[39]);
}

__global__ void global_pooling(double *Layer10_Features,double *output_GPU)
{
    int tid = threadIdx.x;
    double avg = 0.0;
    for(int i = 0; i < 15; i++)
    {           
        for(int j = 0; j <= 15; j++)
        {
		avg+= Layer10_Features[tid*225 + i*15 + j];
        }   
    }
    output_GPU[tid] = avg/225;
    __syncthreads();
    //if(tid==0)
	//printf("Global pool o/p: %.8f\n", output_GPU[1]);
}

int predict_class(double *output_CPU)
{
	double max = 0;
	int predicted_class = 0;
	for(int i=0; i<1000; i++)
	{
		if(output_CPU[i] > max)
		{
			max = output_CPU[i];
			predicted_class = i;
		}
	}
	return predicted_class;
}

void NeuralNetwork(unsigned long long run_time)
{
	cudaError_t err;
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	int deviceCount;                                                         
	cudaGetDeviceCount(&deviceCount);                
	if (deviceCount == 0) {                                                  
		fprintf(stderr, "There is no device.\n");                            
		exit(EXIT_FAILURE);                                                  
	}                                                                    
	int dev;                                                                 
	for (dev = 0; dev < deviceCount; ++dev) {                                
		cudaDeviceProp deviceProp;                                           
		cudaGetDeviceProperties(&deviceProp, dev);   
		if (deviceProp.major >= 1)                                           
			break;                                                           
	}                                                                        
	if (dev == deviceCount) {                                                
		fprintf(stderr, "There is no device supporting CUDA.\n");            
		exit(EXIT_FAILURE);                                                  
	}                                                                        
	else                                                                     
		cudaSetDevice(dev);
	//Allocation of host memory for weights
	double *Layer1_Weights_CPU = (double*) malloc (3*96*49* NUM * sizeof(double));
	double *fire2squeeze1x1_Weights_CPU = (double*) malloc (1536* NUM * sizeof(double));
	double *fire2expand1x1_Weights_CPU = (double*) malloc (1024* NUM * sizeof(double));
	double *fire2expand3x3_Weights_CPU = (double*) malloc (64*9*16* NUM * sizeof(double));
	double *fire3squeeze1x1_Weights_CPU = (double*) malloc (2048* NUM * sizeof(double));
	double *fire3expand1x1_Weights_CPU = (double*) malloc (1024* NUM * sizeof(double));
	double *fire3expand3x3_Weights_CPU = (double*) malloc (64*9*16* NUM * sizeof(double));
	double *fire4squeeze1x1_Weights_CPU = (double*) malloc (4096* NUM * sizeof(double));
	double *fire4expand1x1_Weights_CPU = (double*) malloc (4096* NUM * sizeof(double));
	double *fire4expand3x3_Weights_CPU = (double*) malloc (36864* NUM * sizeof(double));
	double *fire5squeeze1x1_Weights_CPU = (double*) malloc (8192* NUM * sizeof(double));
	double *fire5expand1x1_Weights_CPU = (double*) malloc (4096* NUM * sizeof(double));
	double *fire5expand3x3_Weights_CPU = (double*) malloc (36864* NUM * sizeof(double));
	double *fire6squeeze1x1_Weights_CPU = (double*) malloc (12288* NUM * sizeof(double));
	double *fire6expand1x1_Weights_CPU = (double*) malloc (9216* NUM * sizeof(double));
	double *fire6expand3x3_Weights_CPU = (double*) malloc (82944* NUM * sizeof(double));
	double *fire7squeeze1x1_Weights_CPU = (double*) malloc (18432* NUM * sizeof(double));
	double *fire7expand1x1_Weights_CPU = (double*) malloc (9216* NUM * sizeof(double));
	double *fire7expand3x3_Weights_CPU = (double*) malloc (82944* NUM * sizeof(double));
	double *fire8squeeze1x1_Weights_CPU = (double*) malloc (24640* NUM * sizeof(double));
	double *fire8expand1x1_Weights_CPU = (double*) malloc (16384* NUM * sizeof(double));
	double *fire8expand3x3_Weights_CPU = (double*) malloc (147456* NUM * sizeof(double));
	double *fire9squeeze1x1_Weights_CPU = (double*) malloc (32768* NUM * sizeof(double));
	double *fire9expand1x1_Weights_CPU = (double*) malloc (16384* NUM * sizeof(double));
	double *fire9expand3x3_Weights_CPU = (double*) malloc (147456* NUM * sizeof(double));
	double *Layer10_Weights_CPU = (double*) malloc (512000* NUM * sizeof(double));
	
	//Allocation of host memory for input data
	int *Data_Layer_CPU_R = (int*) malloc (227*227*NUM*sizeof(int));
	int *Data_Layer_CPU_G = (int*) malloc (227*227*NUM*sizeof(int));
	int *Data_Layer_CPU_B = (int*) malloc (227*227*NUM*sizeof(int));
	//Allocation of device memory for input data
	int *Data_Layer_GPU_R;
	int *Data_Layer_GPU_G;
	int *Data_Layer_GPU_B;
	int *Data_Layer_CPU = (int*) malloc (3*227*227*NUM*sizeof(int));
	InitHostMem(Layer1_Weights_CPU, fire2squeeze1x1_Weights_CPU, fire2expand1x1_Weights_CPU, fire2expand3x3_Weights_CPU, fire3squeeze1x1_Weights_CPU, fire3expand1x1_Weights_CPU, fire3expand3x3_Weights_CPU, fire4squeeze1x1_Weights_CPU, fire4expand1x1_Weights_CPU, fire4expand3x3_Weights_CPU, fire5squeeze1x1_Weights_CPU, fire5expand1x1_Weights_CPU, fire5expand3x3_Weights_CPU, fire6squeeze1x1_Weights_CPU, fire6expand1x1_Weights_CPU, fire6expand3x3_Weights_CPU, fire7squeeze1x1_Weights_CPU, fire7expand1x1_Weights_CPU, fire7expand3x3_Weights_CPU, fire8squeeze1x1_Weights_CPU, fire8expand1x1_Weights_CPU, fire8expand3x3_Weights_CPU, fire9squeeze1x1_Weights_CPU, fire9expand1x1_Weights_CPU, fire9expand3x3_Weights_CPU, Layer10_Weights_CPU);
	//printf("\nWeight: %.8f\n", fire2expand1x1_Weights_CPU[0]);
	LoadInput(Data_Layer_CPU);
	ConvertInput(Data_Layer_CPU_R, Data_Layer_CPU_G, Data_Layer_CPU_B, Data_Layer_CPU);
	double *Layer1_Features;
	double *Layer1_Weights_GPU;
	cudaEventRecord(start);
	err = cudaMalloc((void**) &Layer1_Features, 111*111*96* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMalloc((void**) &Layer1_Weights_GPU, 14112* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMalloc((void**) &Data_Layer_GPU_R, 227*227* NUM * sizeof(int));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMalloc((void**) &Data_Layer_GPU_G, 227*227* NUM * sizeof(int));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMalloc((void**) &Data_Layer_GPU_B, 227*227* NUM * sizeof(int));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	//printf("Malloc completed\n");
	cudaMemcpy(Layer1_Weights_GPU,Layer1_Weights_CPU, sizeof(double)*14112*NUM, cudaMemcpyHostToDevice);
	cudaMemcpy(Data_Layer_GPU_R,Data_Layer_CPU_R, 227*227* NUM * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(Data_Layer_GPU_G,Data_Layer_CPU_G, 227*227* NUM * sizeof(int), cudaMemcpyHostToDevice);
	err = cudaMemcpy(Data_Layer_GPU_B,Data_Layer_CPU_B, 227*227* NUM * sizeof(int), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	//printf("Memcpy completed\n");
	dim3 n_threads(111,1,1);
	dim3 n_blocks(111,1,1); 
	cudaThreadSynchronize();
	//Execute First Layer
	ExecuteFirstLayer<<<n_blocks,n_threads>>>(Layer1_Weights_GPU, Data_Layer_GPU_R, Data_Layer_GPU_G, Data_Layer_GPU_B, Layer1_Features, run_time);
	cudaStreamSynchronize(0);
	
	/*err = cudaGetLastError();
        if (err != cudaSuccess)
        {
        	fprintf(stderr, "1st LayerKernel execution failed (error code %s)\n", cudaGetErrorString(err));
       		exit(EXIT_FAILURE);
        }
	cudaThreadSynchronize();
	double *Layer1_features_CPU = (double*) malloc (111*111*96* NUM * sizeof(double));
	err = cudaMemcpy(Layer1_features_CPU, Layer1_Features, 111*111*96*NUM*sizeof(double), cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaThreadSynchronize();
	//printf("Completed Conv1, %.8f\n",Layer1_features_CPU[1182815]);
	double *Pool_Layer_Features;
	err = cudaMalloc((void**) &Pool_Layer_Features, 290400* NUM * sizeof(double));
	dim3 n_threads_pool(111,1,1);
	dim3 n_blocks_pool(111,1,1);
	pooling1<<<n_blocks_pool,n_threads_pool>>>(Layer1_Features, Pool_Layer_Features, 96, 55, 55, 3, 2, 111, 111);
	cudaThreadSynchronize();
	//Fire 2///////////////////////////////////////////////////////////////////////////////////////////////////////////
	double *fire2squeeze1x1_Features;
	double *fire2squeeze1x1_Weights_GPU;
	err = cudaMalloc((void**) &fire2squeeze1x1_Features, 55*55*16* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMalloc((void**) &fire2squeeze1x1_Weights_GPU, 1536* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(fire2squeeze1x1_Weights_GPU,fire2squeeze1x1_Weights_CPU, sizeof(double)*1536*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaThreadSynchronize();
	dim3 n_threads1(55,1,1);
	dim3 n_blocks1(55,1,1);
	Executefire2squeeze1x1<<<n_blocks1,n_threads1>>>(fire2squeeze1x1_Weights_GPU, fire2squeeze1x1_Features, Pool_Layer_Features);
	cudaThreadSynchronize();
	double *fire2_Features;
	double *fire2expand1x1_Weights_GPU;
	err = cudaMalloc((void**) &fire2_Features, 55*55*128* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMalloc((void**) &fire2expand1x1_Weights_GPU, 1024* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(fire2expand1x1_Weights_GPU,fire2expand1x1_Weights_CPU, sizeof(double)*1024*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaThreadSynchronize();
	Executefire2expand1x1<<<n_blocks1,n_threads1>>>(fire2expand1x1_Weights_GPU, fire2_Features, fire2squeeze1x1_Features);
	cudaThreadSynchronize();
	double *fire2expand3x3_Weights_GPU;
	err = cudaMalloc((void**) &fire2expand3x3_Weights_GPU, 9216* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(fire2expand3x3_Weights_GPU,fire2expand3x3_Weights_CPU, sizeof(double)*9216*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaThreadSynchronize();
	Executefire2expand3x3<<<n_blocks1,n_threads1>>>(fire2expand3x3_Weights_GPU, fire2_Features+(55*55*64), fire2squeeze1x1_Features);
	cudaThreadSynchronize();
	//Fire 3  //////////////////////////////////////////////////////////////////////////////////////////////////////////
	double *fire3squeeze1x1_Features;
	double *fire3squeeze1x1_Weights_GPU;
	err = cudaMalloc((void**) &fire3squeeze1x1_Features, 55*55*16* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMalloc((void**) &fire3squeeze1x1_Weights_GPU, 2048* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(fire3squeeze1x1_Weights_GPU,fire3squeeze1x1_Weights_CPU, sizeof(double)*2048*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaThreadSynchronize();
	Executefire3squeeze1x1<<<n_blocks1,n_threads1>>>(fire3squeeze1x1_Weights_GPU, fire3squeeze1x1_Features, fire2_Features);
	cudaThreadSynchronize();
	double *fire3_Features;
	double *fire3expand1x1_Weights_GPU;
	err = cudaMalloc((void**) &fire3_Features, 55*55*128* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMalloc((void**) &fire3expand1x1_Weights_GPU, 1024* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(fire3expand1x1_Weights_GPU,fire3expand1x1_Weights_CPU, sizeof(double)*1024*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaThreadSynchronize();
	Executefire3expand1x1<<<n_blocks1,n_threads1>>>(fire3expand1x1_Weights_GPU, fire3_Features, fire3squeeze1x1_Features);
	cudaThreadSynchronize();
	double *fire3expand3x3_Weights_GPU;
	err = cudaMalloc((void**) &fire3expand3x3_Weights_GPU, 9216* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(fire3expand3x3_Weights_GPU,fire3expand3x3_Weights_CPU, sizeof(double)*9216*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaThreadSynchronize();
	Executefire3expand3x3<<<n_blocks1,n_threads1>>>(fire3expand3x3_Weights_GPU, fire3_Features+(55*55*64), fire3squeeze1x1_Features);
	cudaThreadSynchronize();
	//Fire 4  //////////////////////////////////////////////////////////////////////////////////////////////////////////
	double *fire4squeeze1x1_Features;
	double *fire4squeeze1x1_Weights_GPU;
	err = cudaMalloc((void**) &fire4squeeze1x1_Features, 55*55*32* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMalloc((void**) &fire4squeeze1x1_Weights_GPU, 4096* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(fire4squeeze1x1_Weights_GPU,fire4squeeze1x1_Weights_CPU, sizeof(double)*4096*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaThreadSynchronize();
	Executefire4squeeze1x1<<<n_blocks1,n_threads1>>>(fire4squeeze1x1_Weights_GPU, fire4squeeze1x1_Features, fire3_Features);
	cudaThreadSynchronize();
	double *fire4_Features;
	double *fire4expand1x1_Weights_GPU;
	err = cudaMalloc((void**) &fire4_Features, 55*55*256* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMalloc((void**) &fire4expand1x1_Weights_GPU, 4096* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(fire4expand1x1_Weights_GPU,fire4expand1x1_Weights_CPU, sizeof(double)*4096*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaThreadSynchronize();
	Executefire4expand1x1<<<n_blocks1,n_threads1>>>(fire4expand1x1_Weights_GPU, fire4_Features, fire4squeeze1x1_Features);
	cudaThreadSynchronize();
	double *fire4expand3x3_Weights_GPU;
	err = cudaMalloc((void**) &fire4expand3x3_Weights_GPU, 36864* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(fire4expand3x3_Weights_GPU,fire4expand3x3_Weights_CPU, sizeof(double)*36864*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaThreadSynchronize();
	Executefire4expand3x3<<<n_blocks1,n_threads1>>>(fire4expand3x3_Weights_GPU, fire4_Features+(55*55*128), fire4squeeze1x1_Features);
	cudaThreadSynchronize();
	// Pool 4 ///////////////////////////////////////////////////////////////////////////////////////////
	double *Pool_Layer4_Features;
	err = cudaMalloc((void**) &Pool_Layer4_Features, 186624* NUM * sizeof(double));
	pooling4<<<n_blocks1,n_threads1>>>(fire4_Features, Pool_Layer4_Features, 256, 27, 27, 3, 2, 55, 55);
	cudaThreadSynchronize();
	//Fire 5  //////////////////////////////////////////////////////////////////////////////////////////////////////////
	dim3 n_threads2(27,1,1);
	dim3 n_blocks2(27,1,1);
	double *fire5squeeze1x1_Features;
	double *fire5squeeze1x1_Weights_GPU;
	err = cudaMalloc((void**) &fire5squeeze1x1_Features, 27*27*32* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMalloc((void**) &fire5squeeze1x1_Weights_GPU, 8192* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(fire5squeeze1x1_Weights_GPU,fire5squeeze1x1_Weights_CPU, sizeof(double)*8192*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaThreadSynchronize();
	Executefire5squeeze1x1<<<n_blocks2,n_threads2>>>(fire5squeeze1x1_Weights_GPU, fire5squeeze1x1_Features, Pool_Layer4_Features);
	cudaThreadSynchronize();
	double *fire5_Features;
	double *fire5expand1x1_Weights_GPU;
	err = cudaMalloc((void**) &fire5_Features, 27*27*256* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMalloc((void**) &fire5expand1x1_Weights_GPU, 4096* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(fire5expand1x1_Weights_GPU,fire5expand1x1_Weights_CPU, sizeof(double)*4096*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaThreadSynchronize();
	Executefire5expand1x1<<<n_blocks2,n_threads2>>>(fire5expand1x1_Weights_GPU, fire5_Features, fire5squeeze1x1_Features);
	cudaThreadSynchronize();
	double *fire5expand3x3_Weights_GPU;
	err = cudaMalloc((void**) &fire5expand3x3_Weights_GPU, 36864* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(fire5expand3x3_Weights_GPU,fire5expand3x3_Weights_CPU, sizeof(double)*36864*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaThreadSynchronize();
	Executefire5expand3x3<<<n_blocks2,n_threads2>>>(fire5expand3x3_Weights_GPU, fire5_Features+(27*27*128), fire5squeeze1x1_Features);
	cudaThreadSynchronize();
	//Fire 6  //////////////////////////////////////////////////////////////////////////////////////////////////////////
	double *fire6squeeze1x1_Features;
	double *fire6squeeze1x1_Weights_GPU;
	err = cudaMalloc((void**) &fire6squeeze1x1_Features, 27*27*48* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMalloc((void**) &fire6squeeze1x1_Weights_GPU, 12288* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(fire6squeeze1x1_Weights_GPU,fire6squeeze1x1_Weights_CPU, sizeof(double)*12288*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaThreadSynchronize();
	Executefire6squeeze1x1<<<n_blocks2,n_threads2>>>(fire6squeeze1x1_Weights_GPU, fire6squeeze1x1_Features, fire5_Features);
	cudaThreadSynchronize();
	double *fire6_Features;
	double *fire6expand1x1_Weights_GPU;
	err = cudaMalloc((void**) &fire6_Features, 27*27*384* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMalloc((void**) &fire6expand1x1_Weights_GPU, 9216* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(fire6expand1x1_Weights_GPU,fire6expand1x1_Weights_CPU, sizeof(double)*9216*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaThreadSynchronize();
	Executefire6expand1x1<<<n_blocks2,n_threads2>>>(fire6expand1x1_Weights_GPU, fire6_Features, fire6squeeze1x1_Features);
	cudaThreadSynchronize();
	double *fire6expand3x3_Weights_GPU;
	err = cudaMalloc((void**) &fire6expand3x3_Weights_GPU, 82944* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(fire6expand3x3_Weights_GPU,fire6expand3x3_Weights_CPU, sizeof(double)*82944*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaThreadSynchronize();
	Executefire6expand3x3<<<n_blocks2,n_threads2>>>(fire6expand3x3_Weights_GPU, fire6_Features+(27*27*192), fire6squeeze1x1_Features);
	cudaThreadSynchronize();
	//Fire 7 //////////////////////////////////////////////////////////////////////////////////////////////////////////
	double *fire7squeeze1x1_Features;
	double *fire7squeeze1x1_Weights_GPU;
	err = cudaMalloc((void**) &fire7squeeze1x1_Features, 27*27*48* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMalloc((void**) &fire7squeeze1x1_Weights_GPU, 18432* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(fire7squeeze1x1_Weights_GPU,fire7squeeze1x1_Weights_CPU, sizeof(double)*18432*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaThreadSynchronize();
	Executefire7squeeze1x1<<<n_blocks2,n_threads2>>>(fire7squeeze1x1_Weights_GPU, fire7squeeze1x1_Features, fire6_Features);
	cudaThreadSynchronize();
	double *fire7_Features;
	double *fire7expand1x1_Weights_GPU;
	err = cudaMalloc((void**) &fire7_Features, 27*27*384* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMalloc((void**) &fire7expand1x1_Weights_GPU, 9216* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(fire7expand1x1_Weights_GPU,fire7expand1x1_Weights_CPU, sizeof(double)*9216*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaThreadSynchronize();
	Executefire7expand1x1<<<n_blocks2,n_threads2>>>(fire7expand1x1_Weights_GPU, fire7_Features, fire7squeeze1x1_Features);
	cudaThreadSynchronize();
	double *fire7expand3x3_Weights_GPU;
	err = cudaMalloc((void**) &fire7expand3x3_Weights_GPU, 82944* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(fire7expand3x3_Weights_GPU,fire7expand3x3_Weights_CPU, sizeof(double)*82944*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaThreadSynchronize();
	Executefire7expand3x3<<<n_blocks2,n_threads2>>>(fire7expand3x3_Weights_GPU, fire7_Features+(27*27*192), fire7squeeze1x1_Features);
	cudaThreadSynchronize();
	//Fire 8 //////////////////////////////////////////////////////////////////////////////////////////////////////////
	double *fire8squeeze1x1_Features;
	double *fire8squeeze1x1_Features_CPU = (double*) malloc (27*27*64* NUM * sizeof(double));
	double *fire8squeeze1x1_Weights_GPU;
	err = cudaMalloc((void**) &fire8squeeze1x1_Features, 27*27*64* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMalloc((void**) &fire8squeeze1x1_Weights_GPU, 24640* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(fire8squeeze1x1_Weights_GPU,fire8squeeze1x1_Weights_CPU, sizeof(double)*24640*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaThreadSynchronize();
	Executefire8squeeze1x1<<<n_blocks2,n_threads2>>>(fire8squeeze1x1_Weights_GPU, fire8squeeze1x1_Features, fire7_Features);
	cudaThreadSynchronize();
	err = cudaMemcpy(fire8squeeze1x1_Features_CPU,fire8squeeze1x1_Features, sizeof(double)*27*27*64*NUM, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	//printf("\nCPU Fire8 Squeeze: %.8f",fire8squeeze1x1_Features_CPU[32811]);
	double *fire8_Features;
	double *fire8expand1x1_Weights_GPU;
	err = cudaMalloc((void**) &fire8_Features, 27*27*512* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMalloc((void**) &fire8expand1x1_Weights_GPU, 16384* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(fire8expand1x1_Weights_GPU,fire8expand1x1_Weights_CPU, sizeof(double)*16384*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaThreadSynchronize();
	Executefire8expand1x1<<<n_blocks2,n_threads2>>>(fire8expand1x1_Weights_GPU, fire8_Features, fire8squeeze1x1_Features);
	cudaThreadSynchronize();
	double *fire8expand3x3_Weights_GPU;
	err = cudaMalloc((void**) &fire8expand3x3_Weights_GPU, 147456* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(fire8expand3x3_Weights_GPU,fire8expand3x3_Weights_CPU, sizeof(double)*147456*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	double *fire8squeeze1x1_Features_CPU_2 = (double*) malloc (27*27*64* NUM * sizeof(double));
	cudaThreadSynchronize();
	err = cudaMemcpy(fire8squeeze1x1_Features_CPU_2,fire8squeeze1x1_Features, sizeof(double)*27*27*64*NUM, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	int i;
	for(i=0;i<27*27*64*NUM;i++) {
			printf("%f,",fire8squeeze1x1_Features_CPU_2[i]);
	}
	cudaThreadSynchronize();
	Executefire8expand3x3<<<n_blocks2,n_threads2>>>(fire8expand3x3_Weights_GPU, fire8_Features+(27*27*256), fire8squeeze1x1_Features);
	cudaThreadSynchronize();
	// Pool 8 ///////////////////////////////////////////////////////////////////////////////////////////
	double *Pool_Layer8_Features;
	err = cudaMalloc((void**) &Pool_Layer8_Features, 86528* NUM * sizeof(double));
	pooling8<<<n_blocks2,n_threads2>>>(fire8_Features, Pool_Layer8_Features, 512, 13, 13, 3, 2, 27, 27);
	cudaThreadSynchronize();
	err = cudaGetLastError();
        if (err != cudaSuccess)
        {
        	fprintf(stderr, "Pool 8 execution failed (error code %s)\n", cudaGetErrorString(err));
       		exit(EXIT_FAILURE);
        }
	//printf("Done\n");
	//Fire 9 //////////////////////////////////////////////////////////////////////////////////////////////////////////
	dim3 n_threads3(13,1,1);
	dim3 n_blocks3(13,1,1);
	double *fire9squeeze1x1_Features;
	double *fire9squeeze1x1_Features_CPU = (double*) malloc (13*13*64* NUM * sizeof(double));
	double *fire9squeeze1x1_Weights_GPU;
	err = cudaMalloc((void**) &fire9squeeze1x1_Features, 13*13*64* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMalloc((void**) &fire9squeeze1x1_Weights_GPU, 32768* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(fire9squeeze1x1_Weights_GPU,fire9squeeze1x1_Weights_CPU, sizeof(double)*32768*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaThreadSynchronize();
	Executefire9squeeze1x1<<<n_blocks3,n_threads3>>>(fire9squeeze1x1_Weights_GPU, fire9squeeze1x1_Features, Pool_Layer8_Features);
	cudaThreadSynchronize();
	err = cudaMemcpy(fire9squeeze1x1_Features_CPU,fire9squeeze1x1_Features, sizeof(double)*13*13*64*NUM, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	//printf("\nCPU Fire9 Squeeze: %.8f",fire9squeeze1x1_Features_CPU[3281]);
	double *fire9_Features;
	double *fire9expand1x1_Weights_GPU;
	err = cudaMalloc((void**) &fire9_Features, 13*13*512* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMalloc((void**) &fire9expand1x1_Weights_GPU, 16384* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(fire9expand1x1_Weights_GPU,fire9expand1x1_Weights_CPU, sizeof(double)*16384*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaThreadSynchronize();
	Executefire9expand1x1<<<n_blocks3,n_threads3>>>(fire9expand1x1_Weights_GPU, fire9_Features, fire9squeeze1x1_Features);
	cudaThreadSynchronize();
	double *fire9expand3x3_Weights_GPU;
	err = cudaMalloc((void**) &fire9expand3x3_Weights_GPU, 147456* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(fire9expand3x3_Weights_GPU,fire9expand3x3_Weights_CPU, sizeof(double)*147456*NUM, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	cudaThreadSynchronize();
	Executefire9expand3x3<<<n_blocks3,n_threads3>>>(fire9expand3x3_Weights_GPU, fire9_Features+(13*13*256), fire9squeeze1x1_Features);
	cudaThreadSynchronize();
	//Execute 10th layer Layer/////////////////////////////////////////////////////////////////////////
	dim3 n_threads4(15,1,1);
	dim3 n_blocks4(15,1,1);
	double *Layer10_Weights_GPU;
	double *Layer10_Features;
	err = cudaMalloc((void**) &Layer10_Weights_GPU, 512000* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMalloc((void**) &Layer10_Features, 1000*15*15* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	err = cudaMemcpy(Layer10_Weights_GPU,Layer10_Weights_CPU, 512000* NUM * sizeof(double), cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to copy device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	//printf("Memcpy completed\n");
	cudaThreadSynchronize();
	ExecuteTenthLayer<<<n_blocks4,n_threads4>>>(Layer10_Weights_GPU, fire9_Features, Layer10_Features);
	
	err = cudaGetLastError();
        if (err != cudaSuccess)
        {
        	fprintf(stderr, "10th Layer Kernel execution failed (error code %s)\n", cudaGetErrorString(err));
       		exit(EXIT_FAILURE);
        }
	cudaThreadSynchronize();
	//Global Avg pooling ///////////////////////////////////////////////////////////////////////////////////
	dim3 n_threads5(1000,1,1);
	dim3 n_blocks5(1,1,1);
	double *output_GPU;
	err = cudaMalloc((void**) &output_GPU, 1000* NUM * sizeof(double));
	if (err != cudaSuccess)
        {
        	fprintf(stderr, "Failed to allocate device data (error code %s)!\n", cudaGetErrorString(err));
        	exit(EXIT_FAILURE);
        }
	global_pooling<<<n_blocks5,n_threads5>>>(Layer10_Features, output_GPU);
	cudaThreadSynchronize();
	double *output_CPU = (double*) malloc (1000* NUM * sizeof(double));
	err = cudaMemcpy(output_CPU,output_GPU, sizeof(double)*1000*NUM, cudaMemcpyDeviceToHost);
	cudaThreadSynchronize();
	int predicted_class;
	predicted_class = predict_class(output_CPU);
	cudaStreamSynchronize(0);
	printf("Prediction: %d\n", predicted_class);

        //Free functions
        //cudafree(Layer1_Features);
        //cudafree(Layer1_Weights_GPU);
        //cudafree(Data_Layer_GPU_R);
        //cudafree(Data_Layer_GPU_G);
        //cudafree(Data_Layer_GPU_B);
        //cudafree(Pool_Layer_Features);
        //cudafree(fire2squeeze1x1_Features);
        //cudafree(fire2squeeze1x1_Weights_GPU);
        //cudafree(fire2_Features);
        //cudafree(fire2expand1x1_Weights_GPU);
        //cudafree(fire2expand3x3_Weights_GPU);
        //cudafree(fire3squeeze1x1_Features);
        //cudafree(fire3squeeze1x1_Weights_GPU);
        //cudafree(fire3_Features);
        //cudafree(fire3expand1x1_Weights_GPU);
        //cudafree(fire3expand3x3_Weights_GPU);
        //cudafree(fire4squeeze1x1_Features);
        //cudafree(fire4squeeze1x1_Weights_GPU);
        //cudafree(fire4_Features);
        //cudafree(fire4expand1x1_Weights_GPU);
        //cudafree(fire4expand3x3_Weights_GPU);
        //cudafree(Pool_Layer4_Features);
        //cudafree(fire5squeeze1x1_Features);
        //cudafree(fire5squeeze1x1_Weights_GPU);
        //cudafree(fire5_Features);
        //cudafree(fire5expand1x1_Weights_GPU);
        //cudafree(fire5expand3x3_Weights_GPU);
        //cudafree(fire6squeeze1x1_Features);
        //cudafree(fire6squeeze1x1_Weights_GPU);
        //cudafree(fire6_Features);
        //cudafree(fire6expand1x1_Weights_GPU);
        //cudafree(fire6expand3x3_Weights_GPU);
        //cudafree(fire7squeeze1x1_Features);
        //cudafree(fire7squeeze1x1_Weights_GPU);
        //cudafree(fire7_Features);
        //cudafree(fire7expand1x1_Weights_GPU);
        //cudafree(fire7expand3x3_Weights_GPU);
        //cudafree(fire8squeeze1x1_Features);
        //cudafree(fire8squeeze1x1_Weights_GPU);
        //cudafree(fire8_Features);
        //cudafree(fire8expand1x1_Weights_GPU);
        //cudafree(fire8expand3x3_Weights_GPU);
        //cudafree(Pool_Layer8_Features);
        //cudafree(fire9squeeze1x1_Features);
        //cudafree(fire9squeeze1x1_Weights_GPU);
        //cudafree(fire9_Features);
        //cudafree(fire9expand1x1_Weights_GPU);
        //cudafree(fire9expand3x3_Weights_GPU);
        //cudafree(Layer10_Weights_GPU);
        //cudafree(Layer10_Features);
        //cudafree(output_GPU);
*/
}
	


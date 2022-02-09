/**********************************************************************
* FILENAME :        resnet_host.cu            
* 
* DESCRIPTION :
*       Host side implementation of ResNet network
*
* NOTES :
*       This file includes CUDA memory allocations and CUDA
*       memory copies to host.
*       Invokes kernel from host.
*       Reads inputs and weight from files
* 
* AUTHOR :    Aajna Karki 
*             https://www.linkedin.com/in/aajna/
*********************************************************************/
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
// includes, project
//#include <cutil.h>

#include "../test.hpp"
#include <helper_cuda.h>  

#define INPUT_SIZE 224*224*3
//#define CPU



#ifdef DEBUG_PRINTF
#define DEBUGPRINT(a) printf a
#else
#define DEBUGPRINT(a) (void)0
#endif

//#define NUM 10
// includes, kernels
#include "rn_kernel.cu"
////////////////////////////////////////////////////////////////////////////////
// declaration, forward
extern "C"
void NeuralNetwork();
unsigned g_verbose;
unsigned NUM;
void extract_weights(char *pFileName,float *layer_weights,bool bias)
{
	FILE * pFile1 = fopen (pFileName,"rb");
	char delim[2];
	if(bias == true)
		delim[0] = ' ';
	else
		delim[0] = '\n';
	delim[1] = 0;
	char *token;
	int count = 0;
	char *line = NULL;
	size_t len = 0;
	if (!(pFile1 != NULL && layer_weights != NULL))
        {
            DEBUGPRINT(("File Not Found\n"));
	    return;
	}
	if (pFile1 != NULL && (bias == false))
	{
		DEBUGPRINT((" File FOUND %s\n",pFileName));
		{
			
			//fread(weights,sizeof(weights),1,pFile1);
			//token = strtok(weights,delim);
			//while(token != NULL)
			while (getline(&line, &len, pFile1) != -1)
			{
				token = strtok(line,delim);
				float temp_num = atof(token);
			        layer_weights[count] = temp_num;	
				//DEBUGPRINT("%.8f\t",temp_num); 
				count++; 
			//	token = strtok(NULL,delim);
			}
		}
		DEBUGPRINT(("Final Count : %d\n",count));
		fclose(pFile1);
	}
	if (pFile1 != NULL && (bias == true))
	{
		DEBUGPRINT((" File FOUND %s\n",pFileName));
		{

			char weights[94590] = "";
			fread(weights,sizeof(weights),1,pFile1);
			token = strtok(weights,delim);
			while(token != NULL)
			{
				float temp_num = atof(token);
			        layer_weights[count] = temp_num;	
				//DEBUGPRINT("%.8f\t",temp_num); 
				count++; 
				token = strtok(NULL,delim);
			}
		}
		DEBUGPRINT(("Final Count : %d\n",count));
		fclose(pFile1);
	}

}
int main(int argc, char** argv)
{
	NeuralNetwork();
}
void Fill_weights(float *Layer1_Weights_CPU,float *Layer2_Weights_CPU,float *Layer3_Weights_CPU,float *Layer4_Weights_CPU,float *Layer5_Weights_CPU,float *Layer6_Weights_CPU,float *Layer7_Weights_CPU,float *Layer8_Weights_CPU)
{
	extract_weights("data/conv1.txt",Layer1_Weights_CPU,false);
	extract_weights("alexnet/conv2.txt",Layer2_Weights_CPU,false);
	extract_weights("alexnet/conv3.txt",Layer3_Weights_CPU,false);
	extract_weights("alexnet/conv4.txt",Layer4_Weights_CPU,false);
	extract_weights("alexnet/conv5.txt",Layer5_Weights_CPU,false);
	extract_weights("alexnet/fc6.txt",Layer6_Weights_CPU,false);
	extract_weights("alexnet/fc7.txt",Layer7_Weights_CPU,false);
	extract_weights("alexnet/fc8.txt",Layer8_Weights_CPU,false);
	DEBUGPRINT(("Extracted Weights and Bias successfully\n"));
}
void Fill_bias(float *bias_1,float *bias_2,float *bias_3,float *bias_4,float *bias_5,float *bias_6,float *bias_7,float *bias_8)
{
	extract_weights("alexnet/bias.txt",bias_1,true);
	extract_weights("alexnet/bias2.txt",bias_2,true);
	extract_weights("alexnet/bias3.txt",bias_3,true);
	extract_weights("alexnet/bias4.txt",bias_4,true);
	extract_weights("alexnet/bias5.txt",bias_5,true);
	extract_weights("alexnet/bias6.txt",bias_6,true);
	extract_weights("alexnet/bias7.txt",bias_7,true);
	extract_weights("alexnet/bias8.txt",bias_8,true);
}
void freeMemory(float *ptr)
{
       DEBUGPRINT(("Free Memory\n"));
       if(ptr != NULL)
       {
	       free(ptr);
	       ptr = NULL;
       }
}
#ifdef CPU
void executeLayer2(float *Layer2_pool_CPU,float *Layer2c_Neurons_CPU)
{
        float mean[256],var[256],scale[256],bias[256];
	/* res2a_branch1 */
	float *Layer2a_Weights_CPU = (float *)malloc(sizeof(float) *(64*64)); //4096
	float *Layer2b_Weights_CPU = (float *)malloc(sizeof(float) *(64*64*3*3)); //16384*3
	float *Layer2c_Weights_CPU = (float *)malloc(sizeof(float) *(256*64)); //16384

	float *Layer2a_Neurons_CPU = (float *)malloc(sizeof(float) *(64*56*56));
	float *Layer2b_Neurons_CPU = (float *)malloc(sizeof(float) *(64*56*56));
	float *Layer_ELtwise_Neurons = (float *)malloc(sizeof(float) *(256*56*56));

	extract_weights("data/res2a_branch1.txt",Layer2c_Weights_CPU,false);     
	execute3Dconvolution(NULL,Layer2_pool_CPU,Layer2c_Weights_CPU,Layer_ELtwise_Neurons,256,56,56,1,1,0,64,56);
	/* Batch Norm */
	extract_weights("data/bn2ab1_mean.txt",mean,false);
	extract_weights("data/bn2ab1_var.txt",var,false);
        executeBnNormLayer(Layer_ELtwise_Neurons,mean,var,256,56*56);
        /* Scale */ 
	extract_weights("data/sc2ab1_scale.txt",scale,false);
	extract_weights("data/sc2ab1_bias.txt",bias,false);	
	executeScaleLayer(Layer_ELtwise_Neurons,scale,bias,256,56*56);

	/* res2a_branch2a*/
	extract_weights("data/res2a_branch2a.txt",Layer2a_Weights_CPU,false);     //4096 
	execute3Dconvolution(NULL,Layer2_pool_CPU,Layer2a_Weights_CPU,Layer2a_Neurons_CPU,64,56,56,1,1,0,64,56);
	/* Batch Norm */
	extract_weights("data/bn2a2a_mean.txt",mean,false);
	extract_weights("data/bn2a2a_var.txt",var,false);
        executeBnNormLayer(Layer2a_Neurons_CPU,mean,var,64,56*56);
        /* Scale */ 
	extract_weights("data/sc2a2a_scale.txt",scale,false);
	extract_weights("data/sc2a2a_bias.txt",bias,false);	
	executeScaleLayer(Layer2a_Neurons_CPU,scale,bias,64,56*56);
	/* ReLU */	
	executeReLULayer(Layer2a_Neurons_CPU,(64*56*56));

	/* res2a_branch2b*/
	extract_weights("data/res2a_branch2b.txt",Layer2b_Weights_CPU,false);     //36864 
	execute3Dconvolution(NULL,Layer2a_Neurons_CPU,Layer2b_Weights_CPU,Layer2b_Neurons_CPU,64,56,56,1,3,1,64,56);
	/* Batch Norm */
	extract_weights("data/bn2a2b_mean.txt",mean,false);
	extract_weights("data/bn2a2b_var.txt",var,false);
        executeBnNormLayer(Layer2b_Neurons_CPU,mean,var,64,56*56);
        /* Scale */ 
	extract_weights("data/sc2a2b_scale.txt",scale,false);
	extract_weights("data/sc2a2b_bias.txt",bias,false);	
	executeScaleLayer(Layer2b_Neurons_CPU,scale,bias,64,56*56);
	/* ReLU */	
	executeReLULayer(Layer2b_Neurons_CPU,(64*56*56));


	/* res2a_branch2c*/
	extract_weights("data/res2a_branch2c.txt",Layer2c_Weights_CPU,false);     //16384 
	execute3Dconvolution(NULL,Layer2b_Neurons_CPU,Layer2c_Weights_CPU,Layer2c_Neurons_CPU,256,56,56,1,1,0,64,56);
	/* Batch Norm */
	extract_weights("data/bn2a2c_mean.txt",mean,false);
	extract_weights("data/bn2a2c_var.txt",var,false);
        executeBnNormLayer(Layer2c_Neurons_CPU,mean,var,256,56*56);
        /* Scale */ 
	extract_weights("data/sc2a2c_scale.txt",scale,false);
	extract_weights("data/sc2a2c_bias.txt",bias,false);	
	executeScaleLayer(Layer2c_Neurons_CPU,scale,bias,256,56*56);
	/* Eltwise */
        executeEltWiseLayer(Layer_ELtwise_Neurons,Layer2c_Neurons_CPU,Layer_ELtwise_Neurons,256*56*56);
	/* ReLU */	
	executeReLULayer(Layer_ELtwise_Neurons,(256*56*56));
	
	/* res2b_branch2a*/
	extract_weights("data/res2b_branch2a.txt",Layer2c_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer_ELtwise_Neurons,Layer2c_Weights_CPU,Layer2a_Neurons_CPU,64,56,56,1,1,0,256,56);
	/* Batch Norm */
	extract_weights("data/bn2b2a_mean.txt",mean,false);
	extract_weights("data/bn2b2a_var.txt",var,false);
        executeBnNormLayer(Layer2a_Neurons_CPU,mean,var,64,56*56);
        /* Scale */ 
	extract_weights("data/sc2b2a_scale.txt",scale,false);
	extract_weights("data/sc2b2a_bias.txt",bias,false);	
	executeScaleLayer(Layer2a_Neurons_CPU,scale,bias,64,56*56);
	/* ReLU */	
	executeReLULayer(Layer2a_Neurons_CPU,(64*56*56));
	
	/* res2b_branch2b*/
	extract_weights("data/res2b_branch2b.txt",Layer2b_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer2a_Neurons_CPU,Layer2b_Weights_CPU,Layer2b_Neurons_CPU,64,56,56,1,3,1,64,56);
	/* Batch Norm */
	extract_weights("data/bn2b2b_mean.txt",mean,false);
	extract_weights("data/bn2b2b_var.txt",var,false);
        executeBnNormLayer(Layer2b_Neurons_CPU,mean,var,64,56*56);
        /* Scale */ 
	extract_weights("data/sc2b2b_scale.txt",scale,false);
	extract_weights("data/sc2b2b_bias.txt",bias,false);	
	executeScaleLayer(Layer2b_Neurons_CPU,scale,bias,64,56*56);
	/* ReLU */	
	executeReLULayer(Layer2b_Neurons_CPU,(64*56*56));

	/* res2b_branch2c*/
	extract_weights("data/res2b_branch2c.txt",Layer2c_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer2b_Neurons_CPU,Layer2c_Weights_CPU,Layer2c_Neurons_CPU,256,56,56,1,1,0,64,56);
	/* Batch Norm */
	extract_weights("data/bn2b2c_mean.txt",mean,false);
	extract_weights("data/bn2b2c_var.txt",var,false);
        executeBnNormLayer(Layer2c_Neurons_CPU,mean,var,256,56*56);
        /* Scale */ 
	extract_weights("data/sc2b2c_scale.txt",scale,false);
	extract_weights("data/sc2b2c_bias.txt",bias,false);	
	executeScaleLayer(Layer2c_Neurons_CPU,scale,bias,256,56*56);
	/* Eltwise */
        executeEltWiseLayer(Layer_ELtwise_Neurons,Layer2c_Neurons_CPU,Layer_ELtwise_Neurons,256*56*56);
	/* ReLU */	
	executeReLULayer(Layer_ELtwise_Neurons,(256*56*56));
	/* res2c_branch2a*/
	extract_weights("data/res2c_branch2a.txt",Layer2c_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer_ELtwise_Neurons,Layer2c_Weights_CPU,Layer2a_Neurons_CPU,64,56,56,1,1,0,256,56);
	/* Batch Norm */
	extract_weights("data/bn2c2a_mean.txt",mean,false);
	extract_weights("data/bn2c2a_var.txt",var,false);
        executeBnNormLayer(Layer2a_Neurons_CPU,mean,var,64,56*56);
        /* Scale */ 
	extract_weights("data/sc2c2a_scale.txt",scale,false);
	extract_weights("data/sc2c2a_bias.txt",bias,false);	
	executeScaleLayer(Layer2a_Neurons_CPU,scale,bias,64,56*56);
	/* ReLU */	
	executeReLULayer(Layer2a_Neurons_CPU,(64*56*56));

	/* res2c_branch2b*/
	extract_weights("data/res2c_branch2b.txt",Layer2b_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer2a_Neurons_CPU,Layer2b_Weights_CPU,Layer2b_Neurons_CPU,64,56,56,1,3,1,64,56);
	/* Batch Norm */
	extract_weights("data/bn2c2b_mean.txt",mean,false);
	extract_weights("data/bn2c2b_var.txt",var,false);
        executeBnNormLayer(Layer2b_Neurons_CPU,mean,var,64,56*56);
        /* Scale */ 
	extract_weights("data/sc2c2b_scale.txt",scale,false);
	extract_weights("data/sc2c2b_bias.txt",bias,false);	
	executeScaleLayer(Layer2b_Neurons_CPU,scale,bias,64,56*56);
	/* ReLU */	
	executeReLULayer(Layer2b_Neurons_CPU,(64*56*56));

	/* res2c_branch2c*/
	extract_weights("data/res2c_branch2c.txt",Layer2c_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer2b_Neurons_CPU,Layer2c_Weights_CPU,Layer2c_Neurons_CPU,256,56,56,1,1,0,64,56);
	/* Batch Norm */
	extract_weights("data/bn2c2c_mean.txt",mean,false);
	extract_weights("data/bn2c2c_var.txt",var,false);
        executeBnNormLayer(Layer2c_Neurons_CPU,mean,var,256,56*56);
        /* Scale */ 
	extract_weights("data/sc2c2c_scale.txt",scale,false);
	extract_weights("data/sc2c2c_bias.txt",bias,false);	
	executeScaleLayer(Layer2c_Neurons_CPU,scale,bias,256,56*56);
	/* Eltwise */
        executeEltWiseLayer(Layer_ELtwise_Neurons,Layer2c_Neurons_CPU,Layer2c_Neurons_CPU,256*56*56);
	/* ReLU */	
	executeReLULayer(Layer2c_Neurons_CPU,(256*56*56));
	free(Layer2c_Weights_CPU);
	free(Layer2b_Weights_CPU);
	free(Layer2a_Neurons_CPU);
	free(Layer2b_Neurons_CPU);
        DEBUGPRINT(("Layer 2 computations completed\n"));
}

void executeLayer3(float *Layer2c_Neurons_CPU,float *Layer3c_Neurons_CPU)
{
        DEBUGPRINT(("Layer 3 allocate memory\n"));
        float mean[512],var[512],scale[512],bias[512];
	float *Layer3c_Weights_CPU = (float *)malloc(sizeof(float) *(512*256));
	float *Layer3c1_Weights_CPU = (float *)malloc(sizeof(float) *(512*128));
	float *Layer3a_Weights_CPU = (float *)malloc(sizeof(float) *(128*256));
	float *Layer3b_Weights_CPU = (float *)malloc(sizeof(float) *(128*128*3*3));
	float *Layer3b1_Neurons_CPU = (float *)malloc(sizeof(float) *(512*28*28));
	float *Layer3a_Neurons_CPU = (float *)malloc(sizeof(float) *(128*28*28));
	float *Layer3b_Neurons_CPU = (float *)malloc(sizeof(float) *(128*28*28));
	float *Layer3c2_Neurons_CPU = (float *)malloc(sizeof(float) *(512*28*28));
	float *Layer_ELtwise_Neurons = (float *)malloc(sizeof(float) *(512*28*28));
	
	extract_weights("data/res3a_branch1.txt",Layer3c_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer2c_Neurons_CPU,Layer3c_Weights_CPU,Layer3b1_Neurons_CPU,512,28,28,2,1,0,256,56);
	/* Batch Norm */
	extract_weights("data/bn3ab1_mean.txt",mean,false);
	extract_weights("data/bn3ab1_var.txt",var,false);
        executeBnNormLayer(Layer3b1_Neurons_CPU,mean,var,512,28*28);
        /* Scale */ 
	extract_weights("data/sc3ab1_scale.txt",scale,false);
	extract_weights("data/sc3ab1_bias.txt",bias,false);	
	executeScaleLayer(Layer3b1_Neurons_CPU,scale,bias,512,28*28);

	/* res3a_branch2a*/
	extract_weights("data/res3a_branch2a.txt",Layer3a_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer2c_Neurons_CPU,Layer3a_Weights_CPU,Layer3a_Neurons_CPU,128,28,28,2,1,0,256,56);
	/* Batch Norm */
	extract_weights("data/bn3a2a_mean.txt",mean,false);
	extract_weights("data/bn3a2a_var.txt",var,false);
        executeBnNormLayer(Layer3a_Neurons_CPU,mean,var,128,28*28);
        /* Scale */ 
	extract_weights("data/sc3a2a_scale.txt",scale,false);
	extract_weights("data/sc3a2a_bias.txt",bias,false);	
	executeScaleLayer(Layer3a_Neurons_CPU,scale,bias,128,28*28);
	/* ReLU */	
	executeReLULayer(Layer3a_Neurons_CPU,(128*28*28));
         
	/* res3a_branch2b*/
	extract_weights("data/res3a_branch2b.txt",Layer3b_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer3a_Neurons_CPU,Layer3b_Weights_CPU,Layer3b_Neurons_CPU,128,28,28,1,3,1,128,28);
	/* Batch Norm */
	extract_weights("data/bn3a2b_mean.txt",mean,false);
	extract_weights("data/bn3a2b_var.txt",var,false);
        executeBnNormLayer(Layer3b_Neurons_CPU,mean,var,128,28*28);
        /* Scale */ 
	extract_weights("data/sc3a2b_scale.txt",scale,false);
	extract_weights("data/sc3a2b_bias.txt",bias,false);	
	executeScaleLayer(Layer3b_Neurons_CPU,scale,bias,128,28*28);
	/* ReLU */	
	executeReLULayer(Layer3b_Neurons_CPU,(128*28*28));

	/* res3a_branch2c*/
	extract_weights("data/res3a_branch2c.txt",Layer3c_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer3b_Neurons_CPU,Layer3c_Weights_CPU,Layer3c_Neurons_CPU,512,28,28,1,1,0,128,28);
	/* Batch Norm */
	extract_weights("data/bn3a2c_mean.txt",mean,false);
	extract_weights("data/bn3a2c_var.txt",var,false);
        executeBnNormLayer(Layer3c_Neurons_CPU,mean,var,512,28*28);
        /* Scale */ 
	extract_weights("data/sc3a2c_scale.txt",scale,false);
	extract_weights("data/sc3a2c_bias.txt",bias,false);	
	executeScaleLayer(Layer3c_Neurons_CPU,scale,bias,512,28*28);
	/* Eltwise */
        executeEltWiseLayer(Layer3b1_Neurons_CPU,Layer3c_Neurons_CPU,Layer_ELtwise_Neurons,512*28*28);
	/* ReLU */	
	executeReLULayer(Layer_ELtwise_Neurons,(512*28*28));

	/* res3b_branch2a*/
	extract_weights("data/res3b_branch2a.txt",Layer3c1_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer_ELtwise_Neurons,Layer3c1_Weights_CPU,Layer3a_Neurons_CPU,128,28,28,1,1,0,512,28);
	/* Batch Norm */
	extract_weights("data/bn3b2a_mean.txt",mean,false);
	extract_weights("data/bn3b2a_var.txt",var,false);
        executeBnNormLayer(Layer3a_Neurons_CPU,mean,var,128,28*28);
        /* Scale */ 
	extract_weights("data/sc3b2a_scale.txt",scale,false);
	extract_weights("data/sc3b2a_bias.txt",bias,false);	
	executeScaleLayer(Layer3a_Neurons_CPU,scale,bias,128,28*28);
	/* ReLU */	
	executeReLULayer(Layer3a_Neurons_CPU,(128*28*28));

	/* res3b_branch2b*/
	extract_weights("data/res3b_branch2b.txt",Layer3b_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer3a_Neurons_CPU,Layer3b_Weights_CPU,Layer3b_Neurons_CPU,128,28,28,1,3,1,128,28);
	/* Batch Norm */
	extract_weights("data/bn3b2b_mean.txt",mean,false);
	extract_weights("data/bn3b2b_var.txt",var,false);
        executeBnNormLayer(Layer3b_Neurons_CPU,mean,var,128,28*28);
        /* Scale */ 
	extract_weights("data/sc3b2b_scale.txt",scale,false);
	extract_weights("data/sc3b2b_bias.txt",bias,false);	
	executeScaleLayer(Layer3b_Neurons_CPU,scale,bias,128,28*28);
	/* ReLU */	
	executeReLULayer(Layer3b_Neurons_CPU,(128*28*28));

	/* res3b_branch2c*/
	extract_weights("data/res3b_branch2c.txt",Layer3c_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer3b_Neurons_CPU,Layer3c_Weights_CPU,Layer3c2_Neurons_CPU,512,28,28,1,1,0,128,28);
	/* Batch Norm */
	extract_weights("data/bn3b2c_mean.txt",mean,false);
	extract_weights("data/bn3b2c_var.txt",var,false);
        executeBnNormLayer(Layer3c2_Neurons_CPU,mean,var,512,28*28);
        /* Scale */ 
	extract_weights("data/sc3b2c_scale.txt",scale,false);
	extract_weights("data/sc3b2c_bias.txt",bias,false);	
	executeScaleLayer(Layer3c2_Neurons_CPU,scale,bias,512,28*28);
	/* Eltwise */
        executeEltWiseLayer(Layer_ELtwise_Neurons,Layer3c2_Neurons_CPU,Layer_ELtwise_Neurons,512*28*28);
	/* ReLU */	
	executeReLULayer(Layer_ELtwise_Neurons,(512*28*28));

	/* res3c_branch2a*/
	extract_weights("data/res3c_branch2a.txt",Layer3c1_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer_ELtwise_Neurons,Layer3c1_Weights_CPU,Layer3a_Neurons_CPU,128,28,28,1,1,0,512,28);
	/* Batch Norm */
	extract_weights("data/bn3c2a_mean.txt",mean,false);
	extract_weights("data/bn3c2a_var.txt",var,false);
        executeBnNormLayer(Layer3a_Neurons_CPU,mean,var,128,28*28);
        /* Scale */ 
	extract_weights("data/sc3c2a_scale.txt",scale,false);
	extract_weights("data/sc3c2a_bias.txt",bias,false);	
	executeScaleLayer(Layer3a_Neurons_CPU,scale,bias,128,28*28);
	/* ReLU */	
	executeReLULayer(Layer3a_Neurons_CPU,(128*28*28));

	/* res3c_branch2b*/
	extract_weights("data/res3c_branch2b.txt",Layer3b_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer3a_Neurons_CPU,Layer3b_Weights_CPU,Layer3b_Neurons_CPU,128,28,28,1,3,1,128,28);
	/* Batch Norm */
	extract_weights("data/bn3c2b_mean.txt",mean,false);
	extract_weights("data/bn3c2b_var.txt",var,false);
        executeBnNormLayer(Layer3b_Neurons_CPU,mean,var,128,28*28);
        /* Scale */ 
	extract_weights("data/sc3c2b_scale.txt",scale,false);
	extract_weights("data/sc3c2b_bias.txt",bias,false);	
	executeScaleLayer(Layer3b_Neurons_CPU,scale,bias,128,28*28);
	/* ReLU */	
	executeReLULayer(Layer3b_Neurons_CPU,(128*28*28));
	
	/* res3c_branch2c*/
	extract_weights("data/res3c_branch2c.txt",Layer3c_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer3b_Neurons_CPU,Layer3c_Weights_CPU,Layer3c_Neurons_CPU,512,28,28,1,1,0,128,28);
	/* Batch Norm */
	extract_weights("data/bn3c2c_mean.txt",mean,false);
	extract_weights("data/bn3c2c_var.txt",var,false);
        executeBnNormLayer(Layer3c_Neurons_CPU,mean,var,512,28*28);
        /* Scale */ 
	extract_weights("data/sc3c2c_scale.txt",scale,false);
	extract_weights("data/sc3c2c_bias.txt",bias,false);	
	executeScaleLayer(Layer3c_Neurons_CPU,scale,bias,512,28*28);
	/* Eltwise */
        executeEltWiseLayer(Layer_ELtwise_Neurons,Layer3c_Neurons_CPU,Layer_ELtwise_Neurons,512*28*28);
	/* ReLU */	
	executeReLULayer(Layer_ELtwise_Neurons,(512*28*28));
        
	/* res3d_branch2a*/
	extract_weights("data/res3d_branch2a.txt",Layer3c1_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer_ELtwise_Neurons,Layer3c1_Weights_CPU,Layer3a_Neurons_CPU,128,28,28,1,1,0,512,28);
	/* Batch Norm */
	extract_weights("data/bn3d2a_mean.txt",mean,false);
	extract_weights("data/bn3d2a_var.txt",var,false);
        executeBnNormLayer(Layer3a_Neurons_CPU,mean,var,128,28*28);
        /* Scale */ 
	extract_weights("data/sc3d2a_scale.txt",scale,false);
	extract_weights("data/sc3d2a_bias.txt",bias,false);	
	executeScaleLayer(Layer3a_Neurons_CPU,scale,bias,128,28*28);
	/* ReLU */	
	executeReLULayer(Layer3a_Neurons_CPU,(128*28*28));

	/* res3d_branch2b*/
	extract_weights("data/res3d_branch2b.txt",Layer3b_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer3a_Neurons_CPU,Layer3b_Weights_CPU,Layer3b_Neurons_CPU,128,28,28,1,3,1,128,28);
	/* Batch Norm */
	extract_weights("data/bn3d2b_mean.txt",mean,false);
	extract_weights("data/bn3d2b_var.txt",var,false);
        executeBnNormLayer(Layer3b_Neurons_CPU,mean,var,128,28*28);
        /* Scale */ 
	extract_weights("data/sc3d2b_scale.txt",scale,false);
	extract_weights("data/sc3d2b_bias.txt",bias,false);	
	executeScaleLayer(Layer3b_Neurons_CPU,scale,bias,128,28*28);
	/* ReLU */	
	executeReLULayer(Layer3b_Neurons_CPU,(128*28*28));
	
	/* res3d_branch2c*/
	extract_weights("data/res3d_branch2c.txt",Layer3c_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer3b_Neurons_CPU,Layer3c_Weights_CPU,Layer3c_Neurons_CPU,512,28,28,1,1,0,128,28);
	/* Batch Norm */
	extract_weights("data/bn3d2c_mean.txt",mean,false);
	extract_weights("data/bn3d2c_var.txt",var,false);
        executeBnNormLayer(Layer3c_Neurons_CPU,mean,var,512,28*28);
        /* Scale */ 
	extract_weights("data/sc3d2c_scale.txt",scale,false);
	extract_weights("data/sc3d2c_bias.txt",bias,false);	
	executeScaleLayer(Layer3c_Neurons_CPU,scale,bias,512,28*28);
	/* Eltwise */
        executeEltWiseLayer(Layer_ELtwise_Neurons,Layer3c_Neurons_CPU,Layer3c_Neurons_CPU,512*28*28);
	/* ReLU */	
	executeReLULayer(Layer3c_Neurons_CPU,(512*28*28));
	freeMemory(Layer3a_Weights_CPU);
	freeMemory(Layer3b_Weights_CPU);
	freeMemory(Layer3c_Weights_CPU);
	freeMemory(Layer3c1_Weights_CPU);
	freeMemory(Layer3a_Neurons_CPU);
	freeMemory(Layer3b_Neurons_CPU);
	freeMemory(Layer3b1_Neurons_CPU);
	freeMemory(Layer3c2_Neurons_CPU);
        DEBUGPRINT(("layer 3 computations completed\n"));
}
void executeLayer4(float *Layer3c_Neurons_CPU,float *Layer4c_Neurons_CPU)
{
        float mean[1024],var[1024],scale[1024],bias[1024];
	float *Layer4a_Weights_CPU = (float *)malloc(sizeof(float) *(256*512));
	float *Layer4a1_Weights_CPU = (float *)malloc(sizeof(float) *(256*1024));
	float *Layer4b_Weights_CPU = (float *)malloc(sizeof(float) *(256*256*3*3));
	float *Layer4c_Weights_CPU = (float *)malloc(sizeof(float) *(1024*512));
	float *Layer4b1_Neurons_CPU = (float *)malloc(sizeof(float) *(1024*14*14));
	float *Layer4a_Neurons_CPU = (float *)malloc(sizeof(float) *(256*14*14));
	float *Layer4b_Neurons_CPU = (float *)malloc(sizeof(float) *(256*14*14));
	float *Layer_ELtwise_Neurons = (float *)malloc(sizeof(float) *(1024*14*14));
	
	extract_weights("data/res4a_branch1.txt",Layer4c_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer3c_Neurons_CPU,Layer4c_Weights_CPU,Layer4b1_Neurons_CPU,1024,14,14,2,1,0,512,28);
	/* Batch Norm */
	extract_weights("data/bn4ab1_mean.txt",mean,false);
	extract_weights("data/bn4ab1_var.txt",var,false);
        executeBnNormLayer(Layer4b1_Neurons_CPU,mean,var,1024,14*14);
        /* Scale */ 
	extract_weights("data/sc4ab1_scale.txt",scale,false);
	extract_weights("data/sc4ab1_bias.txt",bias,false);	
	executeScaleLayer(Layer4b1_Neurons_CPU,scale,bias,1024,14*14);
          
	/* res4a_branch2a*/
	extract_weights("data/res4a_branch2a.txt",Layer4a_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer3c_Neurons_CPU,Layer4a_Weights_CPU,Layer4a_Neurons_CPU,256,14,14,2,1,0,512,28);
	/* Batch Norm */
	extract_weights("data/bn4a2a_mean.txt",mean,false);
	extract_weights("data/bn4a2a_var.txt",var,false);
        executeBnNormLayer(Layer4a_Neurons_CPU,mean,var,256,14*14);
        /* Scale */ 
	extract_weights("data/sc4a2a_scale.txt",scale,false);
	extract_weights("data/sc4a2a_bias.txt",bias,false);	
	executeScaleLayer(Layer4a_Neurons_CPU,scale,bias,256,14*14);
	/* ReLU */	
	executeReLULayer(Layer4a_Neurons_CPU,(256*14*14));
	
	/* res4a_branch2b*/
	extract_weights("data/res4a_branch2b.txt",Layer4b_Weights_CPU,false);
	execute3Dconvolution(NULL,Layer4a_Neurons_CPU,Layer4b_Weights_CPU,Layer4b_Neurons_CPU,256,14,14,1,3,1,256,14);
	/* Batch Norm */
	extract_weights("data/bn4a2b_mean.txt",mean,false);
	extract_weights("data/bn4a2b_var.txt",var,false);
        executeBnNormLayer(Layer4b_Neurons_CPU,mean,var,256,14*14);
        /* Scale */ 
	extract_weights("data/sc4a2b_scale.txt",scale,false);
	extract_weights("data/sc4a2b_bias.txt",bias,false);	
	executeScaleLayer(Layer4b_Neurons_CPU,scale,bias,256,14*14);
	/* ReLU */	
	executeReLULayer(Layer4b_Neurons_CPU,(256*14*14));

	/* res4a_branch2c*/
	extract_weights("data/res4a_branch2c.txt",Layer4c_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer4b_Neurons_CPU,Layer4c_Weights_CPU,Layer4c_Neurons_CPU,1024,14,14,1,1,0,256,14);
	/* Batch Norm */
	extract_weights("data/bn4a2c_mean.txt",mean,false);
	extract_weights("data/bn4a2c_var.txt",var,false);
        executeBnNormLayer(Layer4c_Neurons_CPU,mean,var,1024,14*14);
        /* Scale */ 
	extract_weights("data/sc4a2c_scale.txt",scale,false);
	extract_weights("data/sc4a2c_bias.txt",bias,false);	
	executeScaleLayer(Layer4c_Neurons_CPU,scale,bias,1024,14*14);
	/* Eltwise */
        executeEltWiseLayer(Layer4b1_Neurons_CPU,Layer4c_Neurons_CPU,Layer_ELtwise_Neurons,1024*14*14);
	/* ReLU */	
	executeReLULayer(Layer_ELtwise_Neurons,(1024*14*14));

	/* res4b_branch2a*/
	extract_weights("data/res4b_branch2a.txt",Layer4a1_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer_ELtwise_Neurons,Layer4a1_Weights_CPU,Layer4a_Neurons_CPU,256,14,14,1,1,0,1024,14);
	/* Batch Norm */
	extract_weights("data/bn4b2a_mean.txt",mean,false);
	extract_weights("data/bn4b2a_var.txt",var,false);
        executeBnNormLayer(Layer4a_Neurons_CPU,mean,var,256,14*14);
        /* Scale */ 
	extract_weights("data/sc4b2a_scale.txt",scale,false);
	extract_weights("data/sc4b2a_bias.txt",bias,false);	
	executeScaleLayer(Layer4a_Neurons_CPU,scale,bias,256,14*14);
	/* ReLU */	
	executeReLULayer(Layer4a_Neurons_CPU,(256*14*14));
         
	/* res4b_branch2b*/
	extract_weights("data/res4b_branch2b.txt",Layer4b_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer4a_Neurons_CPU,Layer4b_Weights_CPU,Layer4b_Neurons_CPU,256,14,14,1,3,1,256,14);
	/* Batch Norm */
	extract_weights("data/bn4b2b_mean.txt",mean,false);
	extract_weights("data/bn4b2b_var.txt",var,false);
        executeBnNormLayer(Layer4b_Neurons_CPU,mean,var,256,14*14);
        /* Scale */ 
	extract_weights("data/sc4b2b_scale.txt",scale,false);
	extract_weights("data/sc4b2b_bias.txt",bias,false);	
	executeScaleLayer(Layer4b_Neurons_CPU,scale,bias,256,14*14);
	/* ReLU */	
	executeReLULayer(Layer4b_Neurons_CPU,(256*14*14));

	/* res4b_branch2c*/
	extract_weights("data/res4b_branch2c.txt",Layer4c_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer4b_Neurons_CPU,Layer4c_Weights_CPU,Layer4c_Neurons_CPU,1024,14,14,1,1,0,256,14);
	/* Batch Norm */
	extract_weights("data/bn4b2c_mean.txt",mean,false);
	extract_weights("data/bn4b2c_var.txt",var,false);
        executeBnNormLayer(Layer4c_Neurons_CPU,mean,var,1024,14*14);
        /* Scale */ 
	extract_weights("data/sc4b2c_scale.txt",scale,false);
	extract_weights("data/sc4b2c_bias.txt",bias,false);	
	executeScaleLayer(Layer4c_Neurons_CPU,scale,bias,1024,14*14);
	/* Eltwise */
        executeEltWiseLayer(Layer_ELtwise_Neurons,Layer4c_Neurons_CPU,Layer_ELtwise_Neurons,1024*14*14);
	/* ReLU */	
	executeReLULayer(Layer_ELtwise_Neurons,(1024*14*14));
	
	/* res4c_branch2a*/
	extract_weights("data/res4c_branch2a.txt",Layer4a1_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer_ELtwise_Neurons,Layer4a1_Weights_CPU,Layer4a_Neurons_CPU,256,14,14,1,1,0,1024,14);
	/* Batch Norm */
	extract_weights("data/bn4c2a_mean.txt",mean,false);
	extract_weights("data/bn4c2a_var.txt",var,false);
        executeBnNormLayer(Layer4a_Neurons_CPU,mean,var,256,14*14);
        /* Scale */ 
	extract_weights("data/sc4c2a_scale.txt",scale,false);
	extract_weights("data/sc4c2a_bias.txt",bias,false);	
	executeScaleLayer(Layer4a_Neurons_CPU,scale,bias,256,14*14);
	/* ReLU */	
	executeReLULayer(Layer4a_Neurons_CPU,(256*14*14));
         
	/* res4c_branch2b*/
	extract_weights("data/res4c_branch2b.txt",Layer4b_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer4a_Neurons_CPU,Layer4b_Weights_CPU,Layer4b_Neurons_CPU,256,14,14,1,3,1,256,14);
	/* Batch Norm */
	extract_weights("data/bn4c2b_mean.txt",mean,false);
	extract_weights("data/bn4c2b_var.txt",var,false);
        executeBnNormLayer(Layer4b_Neurons_CPU,mean,var,256,14*14);
        /* Scale */ 
	extract_weights("data/sc4c2b_scale.txt",scale,false);
	extract_weights("data/sc4c2b_bias.txt",bias,false);	
	executeScaleLayer(Layer4b_Neurons_CPU,scale,bias,256,14*14);
	/* ReLU */	
	executeReLULayer(Layer4b_Neurons_CPU,(256*14*14));

	/* res4c_branch2c*/
	extract_weights("data/res4c_branch2c.txt",Layer4c_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer4b_Neurons_CPU,Layer4c_Weights_CPU,Layer4c_Neurons_CPU,1024,14,14,1,1,0,256,14);
	/* Batch Norm */
	extract_weights("data/bn4c2c_mean.txt",mean,false);
	extract_weights("data/bn4c2c_var.txt",var,false);
        executeBnNormLayer(Layer4c_Neurons_CPU,mean,var,1024,14*14);
        /* Scale */ 
	extract_weights("data/sc4c2c_scale.txt",scale,false);
	extract_weights("data/sc4c2c_bias.txt",bias,false);	
	executeScaleLayer(Layer4c_Neurons_CPU,scale,bias,1024,14*14);
	/* Eltwise */
        executeEltWiseLayer(Layer_ELtwise_Neurons,Layer4c_Neurons_CPU,Layer_ELtwise_Neurons,1024*14*14);
	/* ReLU */	
	executeReLULayer(Layer_ELtwise_Neurons,(1024*14*14));

	/* res4d_branch2a*/
	extract_weights("data/res4d_branch2a.txt",Layer4a1_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer_ELtwise_Neurons,Layer4a1_Weights_CPU,Layer4a_Neurons_CPU,256,14,14,1,1,0,1024,14);
	/* Batch Norm */
	extract_weights("data/bn4d2a_mean.txt",mean,false);
	extract_weights("data/bn4d2a_var.txt",var,false);
        executeBnNormLayer(Layer4a_Neurons_CPU,mean,var,256,14*14);
        /* Scale */ 
	extract_weights("data/sc4d2a_scale.txt",scale,false);
	extract_weights("data/sc4d2a_bias.txt",bias,false);	
	executeScaleLayer(Layer4a_Neurons_CPU,scale,bias,256,14*14);
	/* ReLU */	
	executeReLULayer(Layer4a_Neurons_CPU,(256*14*14));
         
	/* res4d_branch2b*/
	extract_weights("data/res4d_branch2b.txt",Layer4b_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer4a_Neurons_CPU,Layer4b_Weights_CPU,Layer4b_Neurons_CPU,256,14,14,1,3,1,256,14);
	/* Batch Norm */
	extract_weights("data/bn4d2b_mean.txt",mean,false);
	extract_weights("data/bn4d2b_var.txt",var,false);
        executeBnNormLayer(Layer4b_Neurons_CPU,mean,var,256,14*14);
        /* Scale */ 
	extract_weights("data/sc4d2b_scale.txt",scale,false);
	extract_weights("data/sc4d2b_bias.txt",bias,false);	
	executeScaleLayer(Layer4b_Neurons_CPU,scale,bias,256,14*14);
	/* ReLU */	
	executeReLULayer(Layer4b_Neurons_CPU,(256*14*14));

	/* res4d_branch2c*/
	extract_weights("data/res4d_branch2c.txt",Layer4c_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer4b_Neurons_CPU,Layer4c_Weights_CPU,Layer4c_Neurons_CPU,1024,14,14,1,1,0,256,14);
	/* Batch Norm */
	extract_weights("data/bn4d2c_mean.txt",mean,false);
	extract_weights("data/bn4d2c_var.txt",var,false);
        executeBnNormLayer(Layer4c_Neurons_CPU,mean,var,1024,14*14);
        /* Scale */ 
	extract_weights("data/sc4d2c_scale.txt",scale,false);
	extract_weights("data/sc4d2c_bias.txt",bias,false);	
	executeScaleLayer(Layer4c_Neurons_CPU,scale,bias,1024,14*14);
	/* Eltwise */
        executeEltWiseLayer(Layer_ELtwise_Neurons,Layer4c_Neurons_CPU,Layer_ELtwise_Neurons,1024*14*14);
	/* ReLU */	
	executeReLULayer(Layer_ELtwise_Neurons,(1024*14*14));
	
	/* res4e_branch2a*/
	extract_weights("data/res4e_branch2a.txt",Layer4a1_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer_ELtwise_Neurons,Layer4a1_Weights_CPU,Layer4a_Neurons_CPU,256,14,14,1,1,0,1024,14);
	/* Batch Norm */
	extract_weights("data/bn4e2a_mean.txt",mean,false);
	extract_weights("data/bn4e2a_var.txt",var,false);
        executeBnNormLayer(Layer4a_Neurons_CPU,mean,var,256,14*14);
        /* Scale */ 
	extract_weights("data/sc4e2a_scale.txt",scale,false);
	extract_weights("data/sc4e2a_bias.txt",bias,false);	
	executeScaleLayer(Layer4a_Neurons_CPU,scale,bias,256,14*14);
	/* ReLU */	
	executeReLULayer(Layer4a_Neurons_CPU,(256*14*14));

	/* res4e_branch2b*/
	extract_weights("data/res4e_branch2b.txt",Layer4b_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer4a_Neurons_CPU,Layer4b_Weights_CPU,Layer4b_Neurons_CPU,256,14,14,1,3,1,256,14);
	/* Batch Norm */
	extract_weights("data/bn4e2b_mean.txt",mean,false);
	extract_weights("data/bn4e2b_var.txt",var,false);
        executeBnNormLayer(Layer4b_Neurons_CPU,mean,var,256,14*14);
        /* Scale */ 
	extract_weights("data/sc4e2b_scale.txt",scale,false);
	extract_weights("data/sc4e2b_bias.txt",bias,false);	
	executeScaleLayer(Layer4b_Neurons_CPU,scale,bias,256,14*14);
	/* ReLU */	
	executeReLULayer(Layer4b_Neurons_CPU,(256*14*14));

	/* res4e_branch2c*/
	extract_weights("data/res4e_branch2c.txt",Layer4c_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer4b_Neurons_CPU,Layer4c_Weights_CPU,Layer4c_Neurons_CPU,1024,14,14,1,1,0,256,14);
	/* Batch Norm */
	extract_weights("data/bn4e2c_mean.txt",mean,false);
	extract_weights("data/bn4e2c_var.txt",var,false);
        executeBnNormLayer(Layer4c_Neurons_CPU,mean,var,1024,14*14);
        /* Scale */ 
	extract_weights("data/sc4e2c_scale.txt",scale,false);
	extract_weights("data/sc4e2c_bias.txt",bias,false);	
	executeScaleLayer(Layer4c_Neurons_CPU,scale,bias,1024,14*14);
	/* Eltwise */
        executeEltWiseLayer(Layer_ELtwise_Neurons,Layer4c_Neurons_CPU,Layer_ELtwise_Neurons,1024*14*14);
	/* ReLU */	
	executeReLULayer(Layer_ELtwise_Neurons,(1024*14*14));
	
	/* res4f_branch2a*/
	extract_weights("data/res4f_branch2a.txt",Layer4a1_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer_ELtwise_Neurons,Layer4a1_Weights_CPU,Layer4a_Neurons_CPU,256,14,14,1,1,0,1024,14);
	/* Batch Norm */
	extract_weights("data/bn4f2a_mean.txt",mean,false);
	extract_weights("data/bn4f2a_var.txt",var,false);
        executeBnNormLayer(Layer4a_Neurons_CPU,mean,var,256,14*14);
        /* Scale */ 
	extract_weights("data/sc4f2a_scale.txt",scale,false);
	extract_weights("data/sc4f2a_bias.txt",bias,false);	
	executeScaleLayer(Layer4a_Neurons_CPU,scale,bias,256,14*14);
	/* ReLU */	
	executeReLULayer(Layer4a_Neurons_CPU,(256*14*14));

	/* res4f_branch2b*/
	extract_weights("data/res4f_branch2b.txt",Layer4b_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer4a_Neurons_CPU,Layer4b_Weights_CPU,Layer4b_Neurons_CPU,256,14,14,1,3,1,256,14);
	/* Batch Norm */
	extract_weights("data/bn4f2b_mean.txt",mean,false);
	extract_weights("data/bn4f2b_var.txt",var,false);
        executeBnNormLayer(Layer4b_Neurons_CPU,mean,var,256,14*14);
        /* Scale */ 
	extract_weights("data/sc4f2b_scale.txt",scale,false);
	extract_weights("data/sc4f2b_bias.txt",bias,false);	
	executeScaleLayer(Layer4b_Neurons_CPU,scale,bias,256,14*14);
	/* ReLU */	
	executeReLULayer(Layer4b_Neurons_CPU,(256*14*14));

	/* res4f_branch2c*/
	extract_weights("data/res4f_branch2c.txt",Layer4c_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer4b_Neurons_CPU,Layer4c_Weights_CPU,Layer4c_Neurons_CPU,1024,14,14,1,1,0,256,14);
	/* Batch Norm */
	extract_weights("data/bn4f2c_mean.txt",mean,false);
	extract_weights("data/bn4f2c_var.txt",var,false);
        executeBnNormLayer(Layer4c_Neurons_CPU,mean,var,1024,14*14);
        /* Scale */ 
	extract_weights("data/sc4f2c_scale.txt",scale,false);
	extract_weights("data/sc4f2c_bias.txt",bias,false);	
	executeScaleLayer(Layer4c_Neurons_CPU,scale,bias,1024,14*14);
	/* Eltwise */
        executeEltWiseLayer(Layer_ELtwise_Neurons,Layer4c_Neurons_CPU,Layer4c_Neurons_CPU,1024*14*14);
	/* ReLU */	
	executeReLULayer(Layer4c_Neurons_CPU,(1024*14*14));
	freeMemory(Layer4a_Weights_CPU);
	freeMemory(Layer4b_Weights_CPU);
	freeMemory(Layer4a1_Weights_CPU);
	freeMemory(Layer4c_Weights_CPU);
	freeMemory(Layer4b1_Neurons_CPU);
	freeMemory(Layer4a_Neurons_CPU);
	freeMemory(Layer4b_Neurons_CPU);

}
void executeLayer5(float *Layer4c_Neurons_CPU,float *Layer5c_Neurons_CPU)
{
        float mean[2048],var[2048],scale[2048],bias[2048];
	/* res5a_branch1 */
	float *Layer5a_Weights_CPU = (float *)malloc(sizeof(float) *(512*1024));
	float *Layer5b_Weights_CPU = (float *)malloc(sizeof(float) *(512*512*3*3));
	float *Layer5c_Weights_CPU = (float *)malloc(sizeof(float) *(2048*1024));
	float *Layer5a1_Weights_CPU = (float *)malloc(sizeof(float) *(2048*512));
	float *Layer5b1_Neurons_CPU = (float *)malloc(sizeof(float) *(2048*7*7));
	float *Layer5a_Neurons_CPU = (float *)malloc(sizeof(float) *(512*7*7));
	float *Layer5b_Neurons_CPU = (float *)malloc(sizeof(float) *(512*7*7));
	float *Layer_ELtwise_Neurons = (float *)malloc(sizeof(float) *(2048*7*7));
	
	extract_weights("data/res5a_branch1.txt",Layer5c_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer4c_Neurons_CPU,Layer5c_Weights_CPU,Layer5b1_Neurons_CPU,2048,7,7,2,1,0,1024,14);
	/* Batch Norm */
	extract_weights("data/bn5ab1_mean.txt",mean,false);
	extract_weights("data/bn5ab1_var.txt",var,false);
        executeBnNormLayer(Layer5b1_Neurons_CPU,mean,var,2048,7*7);
        /* Scale */ 
	extract_weights("data/sc5ab1_scale.txt",scale,false);
	extract_weights("data/sc5ab1_bias.txt",bias,false);	
	executeScaleLayer(Layer5b1_Neurons_CPU,scale,bias,2048,7*7);
	
	/* res5a_branch2a*/
	extract_weights("data/res5a_branch2a.txt",Layer5a_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer4c_Neurons_CPU,Layer5a_Weights_CPU,Layer5a_Neurons_CPU,512,7,7,2,1,0,1024,14);
	/* Batch Norm */
	extract_weights("data/bn5a2a_mean.txt",mean,false);
	extract_weights("data/bn5a2a_var.txt",var,false);
        executeBnNormLayer(Layer5a_Neurons_CPU,mean,var,512,7*7);
        /* Scale */ 
	extract_weights("data/sc5a2a_scale.txt",scale,false);
	extract_weights("data/sc5a2a_bias.txt",bias,false);	
	executeScaleLayer(Layer5a_Neurons_CPU,scale,bias,512,7*7);
	/* ReLU */	
	executeReLULayer(Layer5a_Neurons_CPU,(512*7*7));
         
	/* res5a_branch2b*/
	extract_weights("data/res5a_branch2b.txt",Layer5b_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer5a_Neurons_CPU,Layer5b_Weights_CPU,Layer5b_Neurons_CPU,512,7,7,1,3,1,512,7);
	/* Batch Norm */
	extract_weights("data/bn5a2b_mean.txt",mean,false);
	extract_weights("data/bn5a2b_var.txt",var,false);
        executeBnNormLayer(Layer5b_Neurons_CPU,mean,var,512,7*7);
        /* Scale */ 
	extract_weights("data/sc5a2b_scale.txt",scale,false);
	extract_weights("data/sc5a2b_bias.txt",bias,false);	
	executeScaleLayer(Layer5b_Neurons_CPU,scale,bias,512,7*7);
	/* ReLU */	
	executeReLULayer(Layer5b_Neurons_CPU,(512*7*7));

	/* res5a_branch2c*/
	extract_weights("data/res5a_branch2c.txt",Layer5c_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer5b_Neurons_CPU,Layer5c_Weights_CPU,Layer5c_Neurons_CPU,2048,7,7,1,1,0,512,7);
	/* Batch Norm */
	extract_weights("data/bn5a2c_mean.txt",mean,false);
	extract_weights("data/bn5a2c_var.txt",var,false);
        executeBnNormLayer(Layer5c_Neurons_CPU,mean,var,2048,7*7);
        /* Scale */ 
	extract_weights("data/sc5a2c_scale.txt",scale,false);
	extract_weights("data/sc5a2c_bias.txt",bias,false);	
	executeScaleLayer(Layer5c_Neurons_CPU,scale,bias,2048,7*7);
	/* Eltwise */
        executeEltWiseLayer(Layer5b1_Neurons_CPU,Layer5c_Neurons_CPU,Layer_ELtwise_Neurons,2048*7*7);
	/* ReLU */	
	executeReLULayer(Layer_ELtwise_Neurons,(2048*7*7));

	/* res5b_branch2a*/
	extract_weights("data/res5b_branch2a.txt",Layer5a1_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer_ELtwise_Neurons,Layer5a1_Weights_CPU,Layer5a_Neurons_CPU,512,7,7,1,1,0,2048,7);
	/* Batch Norm */
	extract_weights("data/bn5b2a_mean.txt",mean,false);
	extract_weights("data/bn5b2a_var.txt",var,false);
        executeBnNormLayer(Layer5a_Neurons_CPU,mean,var,512,7*7);
        /* Scale */ 
	extract_weights("data/sc5b2a_scale.txt",scale,false);
	extract_weights("data/sc5b2a_bias.txt",bias,false);	
	executeScaleLayer(Layer5a_Neurons_CPU,scale,bias,512,7*7);
	/* ReLU */	
	executeReLULayer(Layer5a_Neurons_CPU,(512*7*7));

	/* res5b_branch2b*/
	extract_weights("data/res5b_branch2b.txt",Layer5b_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer5a_Neurons_CPU,Layer5b_Weights_CPU,Layer5b_Neurons_CPU,512,7,7,1,3,1,512,7);
	/* Batch Norm */
	extract_weights("data/bn5b2b_mean.txt",mean,false);
	extract_weights("data/bn5b2b_var.txt",var,false);
        executeBnNormLayer(Layer5b_Neurons_CPU,mean,var,512,7*7);
        /* Scale */ 
	extract_weights("data/sc5b2b_scale.txt",scale,false);
	extract_weights("data/sc5b2b_bias.txt",bias,false);	
	executeScaleLayer(Layer5b_Neurons_CPU,scale,bias,512,7*7);
	/* ReLU */	
	executeReLULayer(Layer5b_Neurons_CPU,(512*7*7));

	/* res5b_branch2c*/
	extract_weights("data/res5b_branch2c.txt",Layer5c_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer5b_Neurons_CPU,Layer5c_Weights_CPU,Layer5c_Neurons_CPU,2048,7,7,1,1,0,512,7);
	/* Batch Norm */
	extract_weights("data/bn5b2c_mean.txt",mean,false);
	extract_weights("data/bn5b2c_var.txt",var,false);
        executeBnNormLayer(Layer5c_Neurons_CPU,mean,var,2048,7*7);
        /* Scale */
	extract_weights("data/sc5b2c_scale.txt",scale,false);
	extract_weights("data/sc5b2c_bias.txt",bias,false);	
	executeScaleLayer(Layer5c_Neurons_CPU,scale,bias,2048,7*7);
	/* Eltwise */
        executeEltWiseLayer(Layer_ELtwise_Neurons,Layer5c_Neurons_CPU,Layer_ELtwise_Neurons,2048*7*7);
	/* ReLU */	
	executeReLULayer(Layer_ELtwise_Neurons,(2048*7*7));

	/* res5c_branch2a*/
	extract_weights("data/res5c_branch2a.txt",Layer5a1_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer_ELtwise_Neurons,Layer5a1_Weights_CPU,Layer5a_Neurons_CPU,512,7,7,1,1,0,2048,7);
	/* Batch Norm */
	extract_weights("data/bn5c2a_mean.txt",mean,false);
	extract_weights("data/bn5c2a_var.txt",var,false);
        executeBnNormLayer(Layer5a_Neurons_CPU,mean,var,512,7*7);
        /* Scale */ 
	extract_weights("data/sc5c2a_scale.txt",scale,false);
	extract_weights("data/sc5c2a_bias.txt",bias,false);	
	executeScaleLayer(Layer5a_Neurons_CPU,scale,bias,512,7*7);
	/* ReLU */	
	executeReLULayer(Layer5a_Neurons_CPU,(512*7*7));

	/* res5c_branch2b*/
	extract_weights("data/res5c_branch2b.txt",Layer5b_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer5a_Neurons_CPU,Layer5b_Weights_CPU,Layer5b_Neurons_CPU,512,7,7,1,3,1,512,7);
	/* Batch Norm */
	extract_weights("data/bn5c2b_mean.txt",mean,false);
	extract_weights("data/bn5c2b_var.txt",var,false);
        executeBnNormLayer(Layer5b_Neurons_CPU,mean,var,512,7*7);
        /* Scale */ 
	extract_weights("data/sc5c2b_scale.txt",scale,false);
	extract_weights("data/sc5c2b_bias.txt",bias,false);	
	executeScaleLayer(Layer5b_Neurons_CPU,scale,bias,512,7*7);
	/* ReLU */	
	executeReLULayer(Layer5b_Neurons_CPU,(512*7*7));

	/* res5c_branch2c*/
	extract_weights("data/res5c_branch2c.txt",Layer5c_Weights_CPU,false);      
	execute3Dconvolution(NULL,Layer5b_Neurons_CPU,Layer5c_Weights_CPU,Layer5c_Neurons_CPU,2048,7,7,1,1,0,512,7);
	/* Batch Norm */
	extract_weights("data/bn5c2c_mean.txt",mean,false);
	extract_weights("data/bn5c2c_var.txt",var,false);
        executeBnNormLayer(Layer5c_Neurons_CPU,mean,var,2048,7*7);
        /* Scale */ 
	extract_weights("data/sc5c2c_scale.txt",scale,false);
	extract_weights("data/sc5c2c_bias.txt",bias,false);	
	executeScaleLayer(Layer5c_Neurons_CPU,scale,bias,2048,7*7);
	/* Eltwise */
        executeEltWiseLayer(Layer_ELtwise_Neurons,Layer5c_Neurons_CPU,Layer5c_Neurons_CPU,2048*7*7);
	/* ReLU */	
	executeReLULayer(Layer5c_Neurons_CPU,(2048*7*7));
	//extract_weights("data/res5c_out.txt",Layer5c_Neurons_CPU,false);      

	freeMemory(Layer5a_Weights_CPU);
	freeMemory(Layer5a1_Weights_CPU);
	freeMemory(Layer5b_Weights_CPU);
	freeMemory(Layer5c_Weights_CPU);
	freeMemory(Layer5a_Neurons_CPU);
	freeMemory(Layer5b_Neurons_CPU);
	freeMemory(Layer5b1_Neurons_CPU);

}
#else
void Cuda_allocateMemory(float **ptr,int size_in_bytes)
{
        DEBUGPRINT(("CUDA : Allocate memory\n"));
	cudaError_t err = cudaSuccess;
	err = cudaMalloc((void**)ptr,size_in_bytes);
	if (err != cudaSuccess)
                   DEBUGPRINT(("CUDA Allocate MEmory Failed\n")); 

}

void Cuda_memcpyfromHostToDevice(float *ptr_gpu,float *ptr_cpu,int size_in_bytes)
{
        DEBUGPRINT(("CUDA : Allocate memcpy host to device\n"));
	cudaError_t err = cudaSuccess;
	err = cudaMemcpy(ptr_gpu,ptr_cpu,size_in_bytes,cudaMemcpyHostToDevice);// FILL HERE
	if (err != cudaSuccess)
	{
                   DEBUGPRINT(("CUDA Memcpy Failed\n")); 
	}

}

void Cuda_memcpyfromDeviceToHost(float *ptr_cpu,float *ptr_gpu,int size_in_bytes)
{
        DEBUGPRINT(("CUDA : Allocate memcpy device to host\n"));
	cudaError_t err = cudaSuccess;
	err = cudaMemcpy(ptr_cpu,ptr_gpu,size_in_bytes,cudaMemcpyDeviceToHost);// FILL HERE
	if (err != cudaSuccess)
	{
                   DEBUGPRINT(("CUDA Memcpy Failed\n")); 
	}

}
void executeLayer2_CUDA(float *Layer2_pool_GPU,float *Layer2c_Neurons_GPU)
{
        float mean[256],var[256],scale[256],bias[256];
	/* res2a_branch1 */
	float *Layer2a_Weights_CPU = (float *)malloc(sizeof(float) *(64*64)); //4096
	float *Layer2b_Weights_CPU = (float *)malloc(sizeof(float) *(64*64*3*3)); //16384*3
	float *Layer2c_Weights_CPU = (float *)malloc(sizeof(float) *(256*64)); //16384

	float *Layer2a_Weights_GPU ;//= (float *)malloc(sizeof(float) *(64*64)); //4096
	float *Layer2b_Weights_GPU ;//= (float *)malloc(sizeof(float) *(64*64*3*3)); //16384*3
	float *Layer2c_Weights_GPU ;//= (float *)malloc(sizeof(float) *(256*64)); //16384

	float *Layer2a_Neurons_GPU ;//= (float *)malloc(sizeof(float) *(64*56*56));
	float *Layer2b_Neurons_GPU;// = (float *)malloc(sizeof(float) *(64*56*56));
	float *Layer_ELtwise_Neurons_GPU;// = (float *)malloc(sizeof(float) *(256*56*56));

	float *mean_GPU, *var_GPU, *scale_GPU, *bias_GPU ;
	
	Cuda_allocateMemory(&mean_GPU,sizeof(float) * 256);
	Cuda_allocateMemory(&var_GPU,sizeof(float) * 256);
	Cuda_allocateMemory(&scale_GPU,sizeof(float) * 256);
	Cuda_allocateMemory(&bias_GPU,sizeof(float) * 256);

	Cuda_allocateMemory(&Layer2a_Weights_GPU,sizeof(float) * 64*64);
	Cuda_allocateMemory(&Layer2b_Weights_GPU,sizeof(float) * 64*64*3*3);
	Cuda_allocateMemory(&Layer2c_Weights_GPU,sizeof(float) * 256*64);
	Cuda_allocateMemory(&Layer2a_Neurons_GPU,sizeof(float) * 64*56*56);
	Cuda_allocateMemory(&Layer2b_Neurons_GPU,sizeof(float) * 64*56*56);
	Cuda_allocateMemory(&Layer_ELtwise_Neurons_GPU,sizeof(float) * (256*56*56));

        dim3 numBlocks(256,1,1);
        dim3 numThreads(32,32);
        dim3 numBlocks_res(64,1,1);
        dim3 numThreads_res(32,32);
	
	double val = 54.0/32.0;	
	int tfactor = ceil(val);
	DEBUGPRINT((" Split Factor :: %d\n", tfactor));
	
	extract_weights("data/res2a_branch1.txt",Layer2c_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer2c_Weights_GPU,Layer2c_Weights_CPU, sizeof(float)*256*64);
	execute3DconvolutionCuda_split<<<numBlocks,numThreads>>>(NULL,Layer2_pool_GPU,Layer2c_Weights_GPU,Layer_ELtwise_Neurons_GPU,256,56,56,1,1,0,64,56,tfactor);
	/* Batch Norm */
	extract_weights("data/bn2ab1_mean.txt",mean,false);
	extract_weights("data/bn2ab1_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*256);
        executeBnNormLayerCUDA_split<<<numBlocks,numThreads>>>(Layer_ELtwise_Neurons_GPU,mean_GPU,var_GPU,256,56,tfactor);
        /* Scale */ 
	extract_weights("data/sc2ab1_scale.txt",scale,false);
	extract_weights("data/sc2ab1_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*256);
	executeScaleLayerCUDA_split<<<numBlocks,numThreads>>>(Layer_ELtwise_Neurons_GPU,scale_GPU,bias_GPU,256,56,tfactor);

	/* res2a_branch2a*/
	extract_weights("data/res2a_branch2a.txt",Layer2a_Weights_CPU,false);     //4096 
	Cuda_memcpyfromHostToDevice(Layer2a_Weights_GPU,Layer2a_Weights_CPU, sizeof(float)*64*64);
	execute3DconvolutionCuda_split<<<numBlocks_res,numThreads_res>>>(NULL,Layer2_pool_GPU,Layer2a_Weights_GPU,Layer2a_Neurons_GPU,64,56,56,1,1,0,64,56,tfactor);
	/* Batch Norm */
	extract_weights("data/bn2a2a_mean.txt",mean,false);
	extract_weights("data/bn2a2a_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*64);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*64);
        executeBnNormLayerCUDA_split<<<numBlocks_res,numThreads_res>>>(Layer2a_Neurons_GPU,mean_GPU,var_GPU,64,56,tfactor);
        /* Scale */ 
	extract_weights("data/sc2a2a_scale.txt",scale,false);
	extract_weights("data/sc2a2a_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*64);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*64);
	executeScaleLayerCUDA_split<<<numBlocks_res,numThreads_res>>>(Layer2a_Neurons_GPU,scale_GPU,bias_GPU,64,56,tfactor);
	/* ReLU */	
	executeReLULayerCUDA_split<<<numBlocks_res,numThreads_res>>>(Layer2a_Neurons_GPU,56,tfactor);
         
	/* res2a_branch2b*/
	extract_weights("data/res2a_branch2b.txt",Layer2b_Weights_CPU,false);     //36864 
	Cuda_memcpyfromHostToDevice(Layer2b_Weights_GPU,Layer2b_Weights_CPU, sizeof(float)*64*64*3*3);
	execute3DconvolutionCuda_split<<<numBlocks_res,numThreads_res>>>(NULL,Layer2a_Neurons_GPU,Layer2b_Weights_GPU,Layer2b_Neurons_GPU,64,56,56,1,3,1,64,56,tfactor);
	/* Batch Norm */
	extract_weights("data/bn2a2b_mean.txt",mean,false);
	extract_weights("data/bn2a2b_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*64);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*64);
        executeBnNormLayerCUDA_split<<<numBlocks_res,numThreads_res>>>(Layer2b_Neurons_GPU,mean_GPU,var_GPU,64,56,tfactor);
        /* Scale */ 
	extract_weights("data/sc2a2b_scale.txt",scale,false);
	extract_weights("data/sc2a2b_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*64);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*64);
	executeScaleLayerCUDA_split<<<numBlocks_res,numThreads_res>>>(Layer2b_Neurons_GPU,scale_GPU,bias_GPU,64,56,tfactor);
	/* ReLU */	
	executeReLULayerCUDA_split<<<numBlocks_res,numThreads_res>>>(Layer2b_Neurons_GPU,56,tfactor);

	/* res2a_branch2c*/
	extract_weights("data/res2a_branch2c.txt",Layer2c_Weights_CPU,false);     //16384 
	Cuda_memcpyfromHostToDevice(Layer2c_Weights_GPU,Layer2c_Weights_CPU, sizeof(float)*256*64);
	execute3DconvolutionCuda_split<<<numBlocks,numThreads>>>(NULL,Layer2b_Neurons_GPU,Layer2c_Weights_GPU,Layer2c_Neurons_GPU,256,56,56,1,1,0,64,56,tfactor);
	/* Batch Norm */
	extract_weights("data/bn2a2c_mean.txt",mean,false);
	extract_weights("data/bn2a2c_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*256);
        executeBnNormLayerCUDA_split<<<numBlocks,numThreads>>>(Layer2c_Neurons_GPU,mean_GPU,var_GPU,256,56,tfactor);
        /* Scale */ 
	extract_weights("data/sc2a2c_scale.txt",scale,false);
	extract_weights("data/sc2a2c_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*256);
	executeScaleLayerCUDA_split<<<numBlocks,numThreads>>>(Layer2c_Neurons_GPU,scale_GPU,bias_GPU,256,56,tfactor);
	/* Eltwise */
        executeEltWiseLayerCUDA_split<<<numBlocks,numThreads>>>(Layer_ELtwise_Neurons_GPU,Layer2c_Neurons_GPU,Layer_ELtwise_Neurons_GPU,56,tfactor);
	/* ReLU */	
	executeReLULayerCUDA_split<<<numBlocks,numThreads>>>(Layer_ELtwise_Neurons_GPU,56,tfactor);

	/* res2b_branch2a*/
	extract_weights("data/res2b_branch2a.txt",Layer2c_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer2c_Weights_GPU,Layer2c_Weights_CPU, sizeof(float)*256*64);
	execute3DconvolutionCuda_split<<<numBlocks_res,numThreads_res>>>(NULL,Layer_ELtwise_Neurons_GPU,Layer2c_Weights_GPU,Layer2a_Neurons_GPU,64,56,56,1,1,0,256,56,tfactor);
	/* Batch Norm */
	extract_weights("data/bn2b2a_mean.txt",mean,false);
	extract_weights("data/bn2b2a_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*64);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*64);
        executeBnNormLayerCUDA_split<<<numBlocks_res,numThreads_res>>>(Layer2a_Neurons_GPU,mean_GPU,var_GPU,64,56,tfactor);
        /* Scale */ 
	extract_weights("data/sc2b2a_scale.txt",scale,false);
	extract_weights("data/sc2b2a_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*64);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*64);
	executeScaleLayerCUDA_split<<<numBlocks_res,numThreads_res>>>(Layer2a_Neurons_GPU,scale_GPU,bias_GPU,64,56,tfactor);
	/* ReLU */	
	executeReLULayerCUDA_split<<<numBlocks_res,numThreads_res>>>(Layer2a_Neurons_GPU,56,tfactor);

	/* res2b_branch2b*/
	extract_weights("data/res2b_branch2b.txt",Layer2b_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer2b_Weights_GPU,Layer2b_Weights_CPU, sizeof(float)*64*64*3*3);
	execute3DconvolutionCuda_split<<<numBlocks_res,numThreads_res>>>(NULL,Layer2a_Neurons_GPU,Layer2b_Weights_GPU,Layer2b_Neurons_GPU,64,56,56,1,3,1,64,56,tfactor);
	/* Batch Norm */
	extract_weights("data/bn2b2b_mean.txt",mean,false);
	extract_weights("data/bn2b2b_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*64);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*64);
        executeBnNormLayerCUDA_split<<<numBlocks_res,numThreads_res>>>(Layer2b_Neurons_GPU,mean_GPU,var_GPU,64,56,tfactor);
        /* Scale */ 
	extract_weights("data/sc2b2b_scale.txt",scale,false);
	extract_weights("data/sc2b2b_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*64);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*64);
	executeScaleLayerCUDA_split<<<numBlocks_res,numThreads_res>>>(Layer2b_Neurons_GPU,scale_GPU,bias_GPU,64,56,tfactor);
	/* ReLU */	
	executeReLULayerCUDA_split<<<numBlocks_res,numThreads_res>>>(Layer2b_Neurons_GPU,56,tfactor);

	/* res2b_branch2c*/
	extract_weights("data/res2b_branch2c.txt",Layer2c_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer2c_Weights_GPU,Layer2c_Weights_CPU, sizeof(float)*256*64);
	execute3DconvolutionCuda_split<<<numBlocks,numThreads>>>(NULL,Layer2b_Neurons_GPU,Layer2c_Weights_GPU,Layer2c_Neurons_GPU,256,56,56,1,1,0,64,56,tfactor);
	/* Batch Norm */
	extract_weights("data/bn2b2c_mean.txt",mean,false);
	extract_weights("data/bn2b2c_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*256);
        executeBnNormLayerCUDA_split<<<numBlocks,numThreads>>>(Layer2c_Neurons_GPU,mean_GPU,var_GPU,256,56,tfactor);
        /* Scale */ 
	extract_weights("data/sc2b2c_scale.txt",scale,false);
	extract_weights("data/sc2b2c_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*256);
	executeScaleLayerCUDA_split<<<numBlocks,numThreads>>>(Layer2c_Neurons_GPU,scale_GPU,bias_GPU,256,56,tfactor);
	/* Eltwise */
        executeEltWiseLayerCUDA_split<<<numBlocks,numThreads>>>(Layer_ELtwise_Neurons_GPU,Layer2c_Neurons_GPU,Layer_ELtwise_Neurons_GPU,56,tfactor);
	/* ReLU */	
	executeReLULayerCUDA_split<<<numBlocks,numThreads>>>(Layer_ELtwise_Neurons_GPU,56,tfactor);

	/* res2c_branch2a*/
	extract_weights("data/res2c_branch2a.txt",Layer2c_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer2c_Weights_GPU,Layer2c_Weights_CPU, sizeof(float)*256*64);
	execute3DconvolutionCuda_split<<<numBlocks_res,numThreads_res>>>(NULL,Layer_ELtwise_Neurons_GPU,Layer2c_Weights_GPU,Layer2a_Neurons_GPU,64,56,56,1,1,0,256,56,tfactor);
	/* Batch Norm */
	extract_weights("data/bn2c2a_mean.txt",mean,false);
	extract_weights("data/bn2c2a_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*64);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*64);
        executeBnNormLayerCUDA_split<<<numBlocks_res,numThreads_res>>>(Layer2a_Neurons_GPU,mean_GPU,var_GPU,64,56,tfactor);
        /* Scale */ 
	extract_weights("data/sc2c2a_scale.txt",scale,false);
	extract_weights("data/sc2c2a_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*64);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*64);
	executeScaleLayerCUDA_split<<<numBlocks_res,numThreads_res>>>(Layer2a_Neurons_GPU,scale_GPU,bias_GPU,64,56,tfactor);
	/* ReLU */	
	executeReLULayerCUDA_split<<<numBlocks_res,numThreads_res>>>(Layer2a_Neurons_GPU,56,tfactor);

	/* res2c_branch2b*/
	extract_weights("data/res2c_branch2b.txt",Layer2b_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer2b_Weights_GPU,Layer2b_Weights_CPU, sizeof(float)*64*64*3*3);
	execute3DconvolutionCuda_split<<<numBlocks_res,numThreads_res>>>(NULL,Layer2a_Neurons_GPU,Layer2b_Weights_GPU,Layer2b_Neurons_GPU,64,56,56,1,3,1,64,56,tfactor);
	/* Batch Norm */
	extract_weights("data/bn2c2b_mean.txt",mean,false);
	extract_weights("data/bn2c2b_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*64);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*64);
        executeBnNormLayerCUDA_split<<<numBlocks_res,numThreads_res>>>(Layer2b_Neurons_GPU,mean_GPU,var_GPU,64,56,tfactor);
        /* Scale */ 
	extract_weights("data/sc2c2b_scale.txt",scale,false);
	extract_weights("data/sc2c2b_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*64);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*64);
	executeScaleLayerCUDA_split<<<numBlocks_res,numThreads_res>>>(Layer2b_Neurons_GPU,scale_GPU,bias_GPU,64,56,tfactor);
	/* ReLU */	
	executeReLULayerCUDA_split<<<numBlocks_res,numThreads_res>>>(Layer2b_Neurons_GPU,56,tfactor);

	/* res2c_branch2c*/
	extract_weights("data/res2c_branch2c.txt",Layer2c_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer2c_Weights_GPU,Layer2c_Weights_CPU, sizeof(float)*256*64);
	execute3DconvolutionCuda_split<<<numBlocks,numThreads>>>(NULL,Layer2b_Neurons_GPU,Layer2c_Weights_GPU,Layer2c_Neurons_GPU,256,56,56,1,1,0,64,56,tfactor);
	/* Batch Norm */
	extract_weights("data/bn2c2c_mean.txt",mean,false);
	extract_weights("data/bn2c2c_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*256);
        executeBnNormLayerCUDA_split<<<numBlocks,numThreads>>>(Layer2c_Neurons_GPU,mean_GPU,var_GPU,256,56,tfactor);
        /* Scale */ 
	extract_weights("data/sc2c2c_scale.txt",scale,false);
	extract_weights("data/sc2c2c_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*256);
	executeScaleLayerCUDA_split<<<numBlocks,numThreads>>>(Layer2c_Neurons_GPU,scale_GPU,bias_GPU,256,56,tfactor);
	/* Eltwise */
        executeEltWiseLayerCUDA_split<<<numBlocks,numThreads>>>(Layer_ELtwise_Neurons_GPU,Layer2c_Neurons_GPU,Layer2c_Neurons_GPU,56,tfactor);
	/* ReLU */	
	executeReLULayerCUDA_split<<<numBlocks,numThreads>>>(Layer2c_Neurons_GPU,56,tfactor);


	cudaFree(mean_GPU);
	cudaFree(var_GPU);
	cudaFree(scale_GPU);
	cudaFree(bias_GPU);
	cudaFree(Layer2a_Weights_GPU);
	cudaFree(Layer2b_Weights_GPU);
	cudaFree(Layer2c_Weights_GPU);
	cudaFree(Layer2a_Neurons_GPU);
	cudaFree(Layer2b_Neurons_GPU);
	cudaFree(Layer_ELtwise_Neurons_GPU);
	
	free(Layer2a_Weights_CPU);
	free(Layer2c_Weights_CPU);
	free(Layer2b_Weights_CPU);
        DEBUGPRINT(("Layer 2 computations completed\n"));
}
void executeLayer3_CUDA(float *Layer2c_Neurons_GPU,float *Layer3c_Neurons_GPU)
{
        DEBUGPRINT(("Layer 3 allocate memory\n"));
        float mean[512],var[512],scale[512],bias[512];
	float *Layer3c_Weights_CPU = (float *)malloc(sizeof(float) *(512*256));
	float *Layer3c1_Weights_CPU = (float *)malloc(sizeof(float) *(512*128));
	float *Layer3a_Weights_CPU = (float *)malloc(sizeof(float) *(128*256));
	float *Layer3b_Weights_CPU = (float *)malloc(sizeof(float) *(128*128*3*3));
	
	float *Layer3c_Weights_GPU ;//= (float *)malloc(sizeof(float) *(512*256));
	float *Layer3c1_Weights_GPU ;//= (float *)malloc(sizeof(float) *(512*128));
	float *Layer3a_Weights_GPU ;//= (float *)malloc(sizeof(float) *(128*256));
	float *Layer3b_Weights_GPU ;//= (float *)malloc(sizeof(float) *(128*128*3*3));
	float *Layer3b1_Neurons_GPU;// = (float *)malloc(sizeof(float) *(512*28*28));
	float *Layer3a_Neurons_GPU ;//= (float *)malloc(sizeof(float) *(128*28*28));
	float *Layer3b_Neurons_GPU;// = (float *)malloc(sizeof(float) *(128*28*28));
	float *Layer3c2_Neurons_GPU ;//= (float *)malloc(sizeof(float) *(512*28*28));
	float *Layer_ELtwise_Neurons_GPU ;// (float *)malloc(sizeof(float) *(512*28*28));


	float *mean_GPU, *var_GPU, *scale_GPU, *bias_GPU ;
	
	Cuda_allocateMemory(&mean_GPU,sizeof(float) * 512);
	Cuda_allocateMemory(&var_GPU,sizeof(float) * 512);
	Cuda_allocateMemory(&scale_GPU,sizeof(float) * 512);
	Cuda_allocateMemory(&bias_GPU,sizeof(float) * 512);

	Cuda_allocateMemory(&Layer3c_Weights_GPU,sizeof(float) * 256 * 512 );
	Cuda_allocateMemory(&Layer3c1_Weights_GPU,sizeof(float) * 512*128);
	Cuda_allocateMemory(&Layer3a_Weights_GPU,sizeof(float) * 128*256);
	Cuda_allocateMemory(&Layer3b_Weights_GPU,sizeof(float) * 128*128*3*3);
	Cuda_allocateMemory(&Layer3b1_Neurons_GPU,sizeof(float) * 512*28*28);
	Cuda_allocateMemory(&Layer3a_Neurons_GPU,sizeof(float) * 128*28*28);
	Cuda_allocateMemory(&Layer3b_Neurons_GPU,sizeof(float) * 128*28*28);
	Cuda_allocateMemory(&Layer3c2_Neurons_GPU,sizeof(float) * (512*28*28));
	Cuda_allocateMemory(&Layer_ELtwise_Neurons_GPU,sizeof(float) * (512*28*28));

        dim3 numBlocks(512,1,1);
        dim3 numThreads(28,28);
        dim3 numBlocks_res(128,1,1);
        dim3 numThreads_res(28,28);
	
	extract_weights("data/res3a_branch1.txt",Layer3c_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer3c_Weights_GPU,Layer3c_Weights_CPU, sizeof(float)*512*256);
	execute3DconvolutionCuda<<<numBlocks, numThreads>>>(NULL,Layer2c_Neurons_GPU,Layer3c_Weights_GPU,Layer3b1_Neurons_GPU,512,28,28,2,1,0,256,56);
	/* Batch Norm */
	extract_weights("data/bn3ab1_mean.txt",mean,false);
	extract_weights("data/bn3ab1_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*512);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*512);
        executeBnNormLayerCUDA<<<numBlocks, numThreads>>>(Layer3b1_Neurons_GPU,mean_GPU,var_GPU,512,28*28);
        /* Scale */ 
	extract_weights("data/sc3ab1_scale.txt",scale,false);
	extract_weights("data/sc3ab1_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*512);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*512);
	executeScaleLayerCUDA<<<numBlocks, numThreads>>>(Layer3b1_Neurons_GPU,scale_GPU,bias_GPU,512,28*28);

	/* res3a_branch2a*/
	extract_weights("data/res3a_branch2a.txt",Layer3a_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer3a_Weights_GPU,Layer3a_Weights_CPU, sizeof(float)*128*256);
	execute3DconvolutionCuda<<<numBlocks_res, numThreads_res>>>(NULL,Layer2c_Neurons_GPU,Layer3a_Weights_GPU,Layer3a_Neurons_GPU,128,28,28,2,1,0,256,56);
	/* Batch Norm */
	extract_weights("data/bn3a2a_mean.txt",mean,false);
	extract_weights("data/bn3a2a_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*128);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*128);
        executeBnNormLayerCUDA<<<numBlocks_res, numThreads_res>>>(Layer3a_Neurons_GPU,mean_GPU,var_GPU,128,28*28);
        /* Scale */ 
	extract_weights("data/sc3a2a_scale.txt",scale,false);
	extract_weights("data/sc3a2a_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*128);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*128);
	executeScaleLayerCUDA<<<numBlocks_res, numThreads_res>>>(Layer3a_Neurons_GPU,scale_GPU,bias_GPU,128,28*28);
	/* ReLU */	
	executeReLULayerCUDA<<<numBlocks_res, numThreads_res>>>(Layer3a_Neurons_GPU,(128*28*28));
         
	/* res3a_branch2b*/
	extract_weights("data/res3a_branch2b.txt",Layer3b_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer3b_Weights_GPU,Layer3b_Weights_CPU, sizeof(float)*128*128*3*3);
	execute3DconvolutionCuda<<<numBlocks_res, numThreads_res>>>(NULL,Layer3a_Neurons_GPU,Layer3b_Weights_GPU,Layer3b_Neurons_GPU,128,28,28,1,3,1,128,28);
	/* Batch Norm */
	extract_weights("data/bn3a2b_mean.txt",mean,false);
	extract_weights("data/bn3a2b_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*128);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*128);
        executeBnNormLayerCUDA<<<numBlocks_res, numThreads_res>>>(Layer3b_Neurons_GPU,mean_GPU,var_GPU,128,28*28);
        /* Scale */ 
	extract_weights("data/sc3a2b_scale.txt",scale,false);
	extract_weights("data/sc3a2b_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*128);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*128);
	executeScaleLayerCUDA<<<numBlocks_res, numThreads_res>>>(Layer3b_Neurons_GPU,scale_GPU,bias_GPU,128,28*28);
	/* ReLU */	
	executeReLULayerCUDA<<<numBlocks_res, numThreads_res>>>(Layer3b_Neurons_GPU,(128*28*28));

	/* res3a_branch2c*/
	extract_weights("data/res3a_branch2c.txt",Layer3c_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer3c_Weights_GPU,Layer3c_Weights_CPU, sizeof(float)*512*256);
	execute3DconvolutionCuda<<<numBlocks, numThreads>>>(NULL,Layer3b_Neurons_GPU,Layer3c_Weights_GPU,Layer3c_Neurons_GPU,512,28,28,1,1,0,128,28);
	/* Batch Norm */
	extract_weights("data/bn3a2c_mean.txt",mean,false);
	extract_weights("data/bn3a2c_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*512);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*512);
        executeBnNormLayerCUDA<<<numBlocks, numThreads>>>(Layer3c_Neurons_GPU,mean_GPU,var_GPU,512,28*28);
        /* Scale */ 
	extract_weights("data/sc3a2c_scale.txt",scale,false);
	extract_weights("data/sc3a2c_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*512);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*512);
	executeScaleLayerCUDA<<<numBlocks, numThreads>>>(Layer3c_Neurons_GPU,scale_GPU,bias_GPU,512,28*28);
	/* Eltwise */
        executeEltWiseLayerCUDA<<<numBlocks, numThreads>>>(Layer3b1_Neurons_GPU,Layer3c_Neurons_GPU,Layer_ELtwise_Neurons_GPU,512*28*28);
	/* ReLU */	
	executeReLULayerCUDA<<<numBlocks, numThreads>>>(Layer_ELtwise_Neurons_GPU,(512*28*28));

	/* res3b_branch2a*/
	extract_weights("data/res3b_branch2a.txt",Layer3c1_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer3c1_Weights_GPU,Layer3c1_Weights_CPU, sizeof(float)*512*128);
	execute3DconvolutionCuda<<<numBlocks_res, numThreads_res>>>(NULL,Layer_ELtwise_Neurons_GPU,Layer3c1_Weights_GPU,Layer3a_Neurons_GPU,128,28,28,1,1,0,512,28);
	/* Batch Norm */
	extract_weights("data/bn3b2a_mean.txt",mean,false);
	extract_weights("data/bn3b2a_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*128);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*128);
        executeBnNormLayerCUDA<<<numBlocks_res, numThreads_res>>>(Layer3a_Neurons_GPU,mean_GPU,var_GPU,128,28*28);
        /* Scale */ 
	extract_weights("data/sc3b2a_scale.txt",scale,false);
	extract_weights("data/sc3b2a_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*128);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*128);
	executeScaleLayerCUDA<<<numBlocks_res, numThreads_res>>>(Layer3a_Neurons_GPU,scale_GPU,bias_GPU,128,28*28);
	/* ReLU */	
	executeReLULayerCUDA<<<numBlocks_res, numThreads_res>>>(Layer3a_Neurons_GPU,(128*28*28));

	/* res3b_branch2b*/
	extract_weights("data/res3b_branch2b.txt",Layer3b_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer3b_Weights_GPU,Layer3b_Weights_CPU, sizeof(float)*128*128*3*3);
	execute3DconvolutionCuda<<<numBlocks_res, numThreads_res>>>(NULL,Layer3a_Neurons_GPU,Layer3b_Weights_GPU,Layer3b_Neurons_GPU,128,28,28,1,3,1,128,28);
	/* Batch Norm */
	extract_weights("data/bn3b2b_mean.txt",mean,false);
	extract_weights("data/bn3b2b_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*128);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*128);
        executeBnNormLayerCUDA<<<numBlocks_res, numThreads_res>>>(Layer3b_Neurons_GPU,mean_GPU,var_GPU,128,28*28);
        /* Scale */ 
	extract_weights("data/sc3b2b_scale.txt",scale,false);
	extract_weights("data/sc3b2b_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*128);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*128);
	executeScaleLayerCUDA<<<numBlocks_res, numThreads_res>>>(Layer3b_Neurons_GPU,scale_GPU,bias_GPU,128,28*28);
	/* ReLU */	
	executeReLULayerCUDA<<<numBlocks_res, numThreads_res>>>(Layer3b_Neurons_GPU,(128*28*28));

	/* res3b_branch2c*/
	extract_weights("data/res3b_branch2c.txt",Layer3c_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer3c_Weights_GPU,Layer3c_Weights_CPU, sizeof(float)*512*256);
	execute3DconvolutionCuda<<<numBlocks, numThreads>>>(NULL,Layer3b_Neurons_GPU,Layer3c_Weights_GPU,Layer3c2_Neurons_GPU,512,28,28,1,1,0,128,28);
	/* Batch Norm */
	extract_weights("data/bn3b2c_mean.txt",mean,false);
	extract_weights("data/bn3b2c_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*512);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*512);
        executeBnNormLayerCUDA<<<numBlocks, numThreads>>>(Layer3c2_Neurons_GPU,mean_GPU,var_GPU,512,28*28);
        /* Scale */ 
	extract_weights("data/sc3b2c_scale.txt",scale,false);
	extract_weights("data/sc3b2c_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*512);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*512);
	executeScaleLayerCUDA<<<numBlocks, numThreads>>>(Layer3c2_Neurons_GPU,scale_GPU,bias_GPU,512,28*28);
	/* Eltwise */
        executeEltWiseLayerCUDA<<<numBlocks, numThreads>>>(Layer_ELtwise_Neurons_GPU,Layer3c2_Neurons_GPU,Layer_ELtwise_Neurons_GPU,512*28*28);
	/* ReLU */	
	executeReLULayerCUDA<<<numBlocks, numThreads>>>(Layer_ELtwise_Neurons_GPU,(512*28*28));

	/* res3c_branch2a*/
	extract_weights("data/res3c_branch2a.txt",Layer3c1_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer3c1_Weights_GPU,Layer3c1_Weights_CPU, sizeof(float)*512*128);
	execute3DconvolutionCuda<<<numBlocks_res, numThreads_res>>>(NULL,Layer_ELtwise_Neurons_GPU,Layer3c1_Weights_GPU,Layer3a_Neurons_GPU,128,28,28,1,1,0,512,28);
	/* Batch Norm */
	extract_weights("data/bn3c2a_mean.txt",mean,false);
	extract_weights("data/bn3c2a_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*128);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*128);
        executeBnNormLayerCUDA<<<numBlocks_res, numThreads_res>>>(Layer3a_Neurons_GPU,mean_GPU,var_GPU,128,28*28);
        /* Scale */ 
	extract_weights("data/sc3c2a_scale.txt",scale,false);
	extract_weights("data/sc3c2a_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*128);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*128);
	executeScaleLayerCUDA<<<numBlocks_res, numThreads_res>>>(Layer3a_Neurons_GPU,scale_GPU,bias_GPU,128,28*28);
	/* ReLU */	
	executeReLULayerCUDA<<<numBlocks_res, numThreads_res>>>(Layer3a_Neurons_GPU,(128*28*28));

	/* res3c_branch2b*/
	extract_weights("data/res3c_branch2b.txt",Layer3b_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer3b_Weights_GPU,Layer3b_Weights_CPU, sizeof(float)*128*128*3*3);
	execute3DconvolutionCuda<<<numBlocks_res, numThreads_res>>>(NULL,Layer3a_Neurons_GPU,Layer3b_Weights_GPU,Layer3b_Neurons_GPU,128,28,28,1,3,1,128,28);
	/* Batch Norm */
	extract_weights("data/bn3c2b_mean.txt",mean,false);
	extract_weights("data/bn3c2b_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*128);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*128);
        executeBnNormLayerCUDA<<<numBlocks_res, numThreads_res>>>(Layer3b_Neurons_GPU,mean_GPU,var_GPU,128,28*28);
        /* Scale */ 
	extract_weights("data/sc3c2b_scale.txt",scale,false);
	extract_weights("data/sc3c2b_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*128);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*128);
	executeScaleLayerCUDA<<<numBlocks_res, numThreads_res>>>(Layer3b_Neurons_GPU,scale_GPU,bias_GPU,128,28*28);
	/* ReLU */	
	executeReLULayerCUDA<<<numBlocks_res, numThreads_res>>>(Layer3b_Neurons_GPU,(128*28*28));
	
	/* res3c_branch2c*/
	extract_weights("data/res3c_branch2c.txt",Layer3c_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer3c_Weights_GPU,Layer3c_Weights_CPU, sizeof(float)*512*256);
	execute3DconvolutionCuda<<<numBlocks, numThreads>>>(NULL,Layer3b_Neurons_GPU,Layer3c_Weights_GPU,Layer3c_Neurons_GPU,512,28,28,1,1,0,128,28);
	/* Batch Norm */
	extract_weights("data/bn3c2c_mean.txt",mean,false);
	extract_weights("data/bn3c2c_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*512);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*512);
        executeBnNormLayerCUDA<<<numBlocks, numThreads>>>(Layer3c_Neurons_GPU,mean_GPU,var_GPU,512,28*28);
        /* Scale */ 
	extract_weights("data/sc3c2c_scale.txt",scale,false);
	extract_weights("data/sc3c2c_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*512);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*512);
	executeScaleLayerCUDA<<<numBlocks, numThreads>>>(Layer3c_Neurons_GPU,scale_GPU,bias_GPU,512,28*28);
	/* Eltwise */
        executeEltWiseLayerCUDA<<<numBlocks, numThreads>>>(Layer_ELtwise_Neurons_GPU,Layer3c_Neurons_GPU,Layer_ELtwise_Neurons_GPU,512*28*28);
	/* ReLU */	
	executeReLULayerCUDA<<<numBlocks, numThreads>>>(Layer_ELtwise_Neurons_GPU,(512*28*28));
        
	/* res3d_branch2a*/
	extract_weights("data/res3d_branch2a.txt",Layer3c1_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer3c1_Weights_GPU,Layer3c1_Weights_CPU, sizeof(float)*512*128);
	execute3DconvolutionCuda<<<numBlocks_res, numThreads_res>>>(NULL,Layer_ELtwise_Neurons_GPU,Layer3c1_Weights_GPU,Layer3a_Neurons_GPU,128,28,28,1,1,0,512,28);
	/* Batch Norm */
	extract_weights("data/bn3d2a_mean.txt",mean,false);
	extract_weights("data/bn3d2a_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*128);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*128);
        executeBnNormLayerCUDA<<<numBlocks_res, numThreads_res>>>(Layer3a_Neurons_GPU,mean_GPU,var_GPU,128,28*28);
        /* Scale */ 
	extract_weights("data/sc3d2a_scale.txt",scale,false);
	extract_weights("data/sc3d2a_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*128);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*128);
	executeScaleLayerCUDA<<<numBlocks_res, numThreads_res>>>(Layer3a_Neurons_GPU,scale_GPU,bias_GPU,128,28*28);
	/* ReLU */	
	executeReLULayerCUDA<<<numBlocks_res, numThreads_res>>>(Layer3a_Neurons_GPU,(128*28*28));

	/* res3d_branch2b*/
	extract_weights("data/res3d_branch2b.txt",Layer3b_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer3b_Weights_GPU,Layer3b_Weights_CPU, sizeof(float)*128*128*3*3);
	execute3DconvolutionCuda<<<numBlocks_res, numThreads_res>>>(NULL,Layer3a_Neurons_GPU,Layer3b_Weights_GPU,Layer3b_Neurons_GPU,128,28,28,1,3,1,128,28);
	/* Batch Norm */
	extract_weights("data/bn3d2b_mean.txt",mean,false);
	extract_weights("data/bn3d2b_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*128);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*128);
        executeBnNormLayerCUDA<<<numBlocks_res, numThreads_res>>>(Layer3b_Neurons_GPU,mean_GPU,var_GPU,128,28*28);
        /* Scale */ 
	extract_weights("data/sc3d2b_scale.txt",scale,false);
	extract_weights("data/sc3d2b_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*128);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*128);
	executeScaleLayerCUDA<<<numBlocks_res,numThreads_res>>>(Layer3b_Neurons_GPU,scale_GPU,bias_GPU,128,28*28);
	/* ReLU */	
	executeReLULayerCUDA<<<numBlocks_res, numThreads_res>>>(Layer3b_Neurons_GPU,(128*28*28));
	
	/* res3d_branch2c*/
	extract_weights("data/res3d_branch2c.txt",Layer3c_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer3c_Weights_GPU,Layer3c_Weights_CPU, sizeof(float)*512*256);
	execute3DconvolutionCuda<<<numBlocks, numThreads>>>(NULL,Layer3b_Neurons_GPU,Layer3c_Weights_GPU,Layer3c_Neurons_GPU,512,28,28,1,1,0,128,28);
	/* Batch Norm */
	extract_weights("data/bn3d2c_mean.txt",mean,false);
	extract_weights("data/bn3d2c_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*512);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*512);
        executeBnNormLayerCUDA<<<numBlocks, numThreads>>>(Layer3c_Neurons_GPU,mean_GPU,var_GPU,512,28*28);
        /* Scale */ 
	extract_weights("data/sc3d2c_scale.txt",scale,false);
	extract_weights("data/sc3d2c_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*512);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*512);
	executeScaleLayerCUDA<<<numBlocks, numThreads>>>(Layer3c_Neurons_GPU,scale_GPU,bias_GPU,512,28*28);
	/* Eltwise */
        executeEltWiseLayerCUDA<<<numBlocks, numThreads>>>(Layer_ELtwise_Neurons_GPU,Layer3c_Neurons_GPU,Layer3c_Neurons_GPU,512*28*28);
	/* ReLU */	
	executeReLULayerCUDA<<<numBlocks, numThreads>>>(Layer3c_Neurons_GPU,(512*28*28));

	
	cudaFree(mean_GPU);
	cudaFree(var_GPU);
	cudaFree(scale_GPU);
	cudaFree(bias_GPU);
	cudaFree(Layer3c_Weights_GPU);
	cudaFree(Layer3c1_Weights_GPU);
	cudaFree(Layer3a_Weights_GPU);
	cudaFree(Layer3b_Weights_GPU);
	cudaFree(Layer3b1_Neurons_GPU);
	cudaFree(Layer3a_Neurons_GPU);
	cudaFree(Layer3b_Neurons_GPU);
	cudaFree(Layer3c2_Neurons_GPU);
	cudaFree(Layer_ELtwise_Neurons_GPU);

	freeMemory(Layer3a_Weights_CPU);
	freeMemory(Layer3b_Weights_CPU);
	freeMemory(Layer3c_Weights_CPU);
	freeMemory(Layer3c1_Weights_CPU);
        DEBUGPRINT(("layer 3 computations completed\n"));
}

void executeLayer4_CUDA(float *Layer3c_Neurons_GPU,float *Layer4c_Neurons_GPU)
{
        float mean[1024],var[1024],scale[1024],bias[1024];
	float *Layer4a_Weights_CPU = (float *)malloc(sizeof(float) *(256*512));
	float *Layer4a1_Weights_CPU = (float *)malloc(sizeof(float) *(256*1024));
	float *Layer4b_Weights_CPU = (float *)malloc(sizeof(float) *(256*256*3*3));
	float *Layer4c_Weights_CPU = (float *)malloc(sizeof(float) *(1024*512));


	float *Layer4a_Weights_GPU ;//= (float *)malloc(sizeof(float) *(256*512));
	float *Layer4a1_Weights_GPU ;//= (float *)malloc(sizeof(float) *(256*1024));
	float *Layer4b_Weights_GPU ;//= (float *)malloc(sizeof(float) *(256*256*3*3));
	float *Layer4c_Weights_GPU ;//= (float *)malloc(sizeof(float) *(1024*512));
	float *Layer4b1_Neurons_GPU ;//= (float *)malloc(sizeof(float) *(1024*14*14));
	float *Layer4a_Neurons_GPU ;//= (float *)malloc(sizeof(float) *(256*14*14));
	float *Layer4b_Neurons_GPU ;//= (float *)malloc(sizeof(float) *(256*14*14));
	float *Layer_ELtwise_Neurons_GPU ;//= (float *)malloc(sizeof(float) *(1024*14*14));
	
	float *mean_GPU, *var_GPU, *scale_GPU, *bias_GPU ;
	
	Cuda_allocateMemory(&mean_GPU,sizeof(float) * 1024);
	Cuda_allocateMemory(&var_GPU,sizeof(float) * 1024);
	Cuda_allocateMemory(&scale_GPU,sizeof(float) * 1024);
	Cuda_allocateMemory(&bias_GPU,sizeof(float) * 1024);

	Cuda_allocateMemory(&Layer4a_Weights_GPU,sizeof(float) * 256 * 512 );
	Cuda_allocateMemory(&Layer4a1_Weights_GPU,sizeof(float) * 256*1024);
	Cuda_allocateMemory(&Layer4b_Weights_GPU,sizeof(float) * 256*256*3*3);
	Cuda_allocateMemory(&Layer4c_Weights_GPU,sizeof(float) * 1024*512);
	Cuda_allocateMemory(&Layer4b1_Neurons_GPU,sizeof(float) * 1024*14*14);
	Cuda_allocateMemory(&Layer4a_Neurons_GPU,sizeof(float) * 256*14*14);
	Cuda_allocateMemory(&Layer4b_Neurons_GPU,sizeof(float) * (256*14*14));
	Cuda_allocateMemory(&Layer_ELtwise_Neurons_GPU,sizeof(float) * (1024*14*14));

        dim3 numBlocks(1024,1,1);
        dim3 numThreads(14,14);
        dim3 numBlocks_res(256,1,1);
        dim3 numThreads_res(14,14);
	
	extract_weights("data/res4a_branch1.txt",Layer4c_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer4c_Weights_GPU,Layer4c_Weights_CPU, sizeof(float)*1024*512);
	
	execute3DconvolutionCuda <<< numBlocks, numThreads >>>(NULL,Layer3c_Neurons_GPU,Layer4c_Weights_GPU,Layer4b1_Neurons_GPU,1024,14,14,2,1,0,512,28);
	/* Batch Norm */
	extract_weights("data/bn4ab1_mean.txt",mean,false);
	extract_weights("data/bn4ab1_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*1024);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*1024);
        executeBnNormLayerCUDA<<< numBlocks, numThreads >>>(Layer4b1_Neurons_GPU,mean_GPU,var_GPU,1024,14*14);
        /* Scale */ 
	extract_weights("data/sc4ab1_scale.txt",scale,false);
	extract_weights("data/sc4ab1_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*1024);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*1024);
	executeScaleLayerCUDA<<< numBlocks, numThreads >>>(Layer4b1_Neurons_GPU,scale_GPU,bias_GPU,1024,14*14);
          
	/* res4a_branch2a*/
	extract_weights("data/res4a_branch2a.txt",Layer4a_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer4a_Weights_GPU,Layer4a_Weights_CPU, sizeof(float)*256*512);
	execute3DconvolutionCuda<<< numBlocks_res, numThreads_res >>>(NULL,Layer3c_Neurons_GPU,Layer4a_Weights_GPU,Layer4a_Neurons_GPU,256,14,14,2,1,0,512,28);
	/* Batch Norm */
	extract_weights("data/bn4a2a_mean.txt",mean,false);
	extract_weights("data/bn4a2a_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*256);
        executeBnNormLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4a_Neurons_GPU,mean_GPU,var_GPU,256,14*14);
        /* Scale */ 
	extract_weights("data/sc4a2a_scale.txt",scale,false);
	extract_weights("data/sc4a2a_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*256);
	executeScaleLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4a_Neurons_GPU,scale_GPU,bias_GPU,256,14*14);
	/* ReLU */	
	executeReLULayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4a_Neurons_GPU,(256*14*14));
	
	/* res4a_branch2b*/
	extract_weights("data/res4a_branch2b.txt",Layer4b_Weights_CPU,false);
	Cuda_memcpyfromHostToDevice(Layer4b_Weights_GPU,Layer4b_Weights_CPU, sizeof(float)*(256*256*3*3));
	execute3DconvolutionCuda<<< numBlocks_res, numThreads_res >>>(NULL,Layer4a_Neurons_GPU,Layer4b_Weights_GPU,Layer4b_Neurons_GPU,256,14,14,1,3,1,256,14);
	/* Batch Norm */
	extract_weights("data/bn4a2b_mean.txt",mean,false);
	extract_weights("data/bn4a2b_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*256);
        executeBnNormLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4b_Neurons_GPU,mean_GPU,var_GPU,256,14*14);
        /* Scale */ 
	extract_weights("data/sc4a2b_scale.txt",scale,false);
	extract_weights("data/sc4a2b_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*256);
	executeScaleLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4b_Neurons_GPU,scale_GPU,bias_GPU,256,14*14);
	/* ReLU */	
	executeReLULayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4b_Neurons_GPU,(256*14*14));

	/* res4a_branch2c*/
	extract_weights("data/res4a_branch2c.txt",Layer4c_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer4c_Weights_GPU,Layer4c_Weights_CPU, sizeof(float)*1024*512);
	execute3DconvolutionCuda<<< numBlocks, numThreads >>>(NULL,Layer4b_Neurons_GPU,Layer4c_Weights_GPU,Layer4c_Neurons_GPU,1024,14,14,1,1,0,256,14);
	/* Batch Norm */
	extract_weights("data/bn4a2c_mean.txt",mean,false);
	extract_weights("data/bn4a2c_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*1024);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*1024);
        executeBnNormLayerCUDA<<< numBlocks, numThreads >>>(Layer4c_Neurons_GPU,mean_GPU,var_GPU,1024,14*14);
        /* Scale */ 
	extract_weights("data/sc4a2c_scale.txt",scale,false);
	extract_weights("data/sc4a2c_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*1024);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*1024);
	executeScaleLayerCUDA<<< numBlocks, numThreads >>>(Layer4c_Neurons_GPU,scale_GPU,bias_GPU,1024,14*14);
	/* Eltwise */
        executeEltWiseLayerCUDA<<< numBlocks, numThreads >>>(Layer4b1_Neurons_GPU,Layer4c_Neurons_GPU,Layer_ELtwise_Neurons_GPU,1024*14*14);
	/* ReLU */	
	executeReLULayerCUDA<<< numBlocks, numThreads >>>(Layer_ELtwise_Neurons_GPU,(1024*14*14));

	/* res4b_branch2a*/
	extract_weights("data/res4b_branch2a.txt",Layer4a1_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer4a1_Weights_GPU,Layer4a1_Weights_CPU, sizeof(float)*1024*256);
	execute3DconvolutionCuda<<< numBlocks_res, numThreads_res >>>(NULL,Layer_ELtwise_Neurons_GPU,Layer4a1_Weights_GPU,Layer4a_Neurons_GPU,256,14,14,1,1,0,1024,14);
	/* Batch Norm */
	extract_weights("data/bn4b2a_mean.txt",mean,false);
	extract_weights("data/bn4b2a_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*256);
        executeBnNormLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4a_Neurons_GPU,mean_GPU,var_GPU,256,14*14);
        /* Scale */ 
	extract_weights("data/sc4b2a_scale.txt",scale,false);
	extract_weights("data/sc4b2a_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*256);
	executeScaleLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4a_Neurons_GPU,scale_GPU,bias_GPU,256,14*14);
	/* ReLU */	
	executeReLULayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4a_Neurons_GPU,(256*14*14));
         
	/* res4b_branch2b*/
	extract_weights("data/res4b_branch2b.txt",Layer4b_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer4b_Weights_GPU,Layer4b_Weights_CPU, sizeof(float)*(256*256*3*3));
	execute3DconvolutionCuda<<< numBlocks_res, numThreads_res >>>(NULL,Layer4a_Neurons_GPU,Layer4b_Weights_GPU,Layer4b_Neurons_GPU,256,14,14,1,3,1,256,14);
	/* Batch Norm */
	extract_weights("data/bn4b2b_mean.txt",mean,false);
	extract_weights("data/bn4b2b_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*256);
        executeBnNormLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4b_Neurons_GPU,mean_GPU,var_GPU,256,14*14);
        /* Scale */ 
	extract_weights("data/sc4b2b_scale.txt",scale,false);
	extract_weights("data/sc4b2b_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*256);
	executeScaleLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4b_Neurons_GPU,scale_GPU,bias_GPU,256,14*14);
	/* ReLU */	
	executeReLULayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4b_Neurons_GPU,(256*14*14));

	/* res4b_branch2c*/
	extract_weights("data/res4b_branch2c.txt",Layer4c_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer4c_Weights_GPU,Layer4c_Weights_CPU, sizeof(float)*1024*512);
	execute3DconvolutionCuda<<< numBlocks, numThreads >>>(NULL,Layer4b_Neurons_GPU,Layer4c_Weights_GPU,Layer4c_Neurons_GPU,1024,14,14,1,1,0,256,14);
	/* Batch Norm */
	extract_weights("data/bn4b2c_mean.txt",mean,false);
	extract_weights("data/bn4b2c_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*1024);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*1024);
        executeBnNormLayerCUDA<<< numBlocks, numThreads >>>(Layer4c_Neurons_GPU,mean_GPU,var_GPU,1024,14*14);
        /* Scale */ 
	extract_weights("data/sc4b2c_scale.txt",scale,false);
	extract_weights("data/sc4b2c_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*1024);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*1024);
	executeScaleLayerCUDA<<< numBlocks, numThreads >>>(Layer4c_Neurons_GPU,scale_GPU,bias_GPU,1024,14*14);
	/* Eltwise */
        executeEltWiseLayerCUDA<<< numBlocks, numThreads >>>(Layer_ELtwise_Neurons_GPU,Layer4c_Neurons_GPU,Layer_ELtwise_Neurons_GPU,1024*14*14);
	/* ReLU */	
	executeReLULayerCUDA<<< numBlocks, numThreads >>>(Layer_ELtwise_Neurons_GPU,(1024*14*14));
	
	/* res4c_branch2a*/
	extract_weights("data/res4c_branch2a.txt",Layer4a1_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer4a1_Weights_GPU,Layer4a1_Weights_CPU, sizeof(float)*1024*256);
	execute3DconvolutionCuda<<< numBlocks_res, numThreads_res >>>(NULL,Layer_ELtwise_Neurons_GPU,Layer4a1_Weights_GPU,Layer4a_Neurons_GPU,256,14,14,1,1,0,1024,14);
	/* Batch Norm */
	extract_weights("data/bn4c2a_mean.txt",mean,false);
	extract_weights("data/bn4c2a_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*256);
        executeBnNormLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4a_Neurons_GPU,mean_GPU,var_GPU,256,14*14);
        /* Scale */ 
	extract_weights("data/sc4c2a_scale.txt",scale,false);
	extract_weights("data/sc4c2a_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*256);
	executeScaleLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4a_Neurons_GPU,scale_GPU,bias_GPU,256,14*14);
	/* ReLU */	
	executeReLULayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4a_Neurons_GPU,(256*14*14));
         
	/* res4c_branch2b*/
	extract_weights("data/res4c_branch2b.txt",Layer4b_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer4b_Weights_GPU,Layer4b_Weights_CPU, sizeof(float)*(256*256*3*3));
	execute3DconvolutionCuda<<< numBlocks_res, numThreads_res >>>(NULL,Layer4a_Neurons_GPU,Layer4b_Weights_GPU,Layer4b_Neurons_GPU,256,14,14,1,3,1,256,14);
	/* Batch Norm */
	extract_weights("data/bn4c2b_mean.txt",mean,false);
	extract_weights("data/bn4c2b_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*256);
        executeBnNormLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4b_Neurons_GPU,mean_GPU,var_GPU,256,14*14);
        /* Scale */ 
	extract_weights("data/sc4c2b_scale.txt",scale,false);
	extract_weights("data/sc4c2b_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*256);
	executeScaleLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4b_Neurons_GPU,scale_GPU,bias_GPU,256,14*14);
	/* ReLU */	
	executeReLULayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4b_Neurons_GPU,(256*14*14));

	/* res4c_branch2c*/
	extract_weights("data/res4c_branch2c.txt",Layer4c_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer4c_Weights_GPU,Layer4c_Weights_CPU, sizeof(float)*1024*512);
	execute3DconvolutionCuda<<< numBlocks, numThreads >>>(NULL,Layer4b_Neurons_GPU,Layer4c_Weights_GPU,Layer4c_Neurons_GPU,1024,14,14,1,1,0,256,14);
	/* Batch Norm */
	extract_weights("data/bn4c2c_mean.txt",mean,false);
	extract_weights("data/bn4c2c_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*1024);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*1024);
        executeBnNormLayerCUDA<<< numBlocks, numThreads >>>(Layer4c_Neurons_GPU,mean_GPU,var_GPU,1024,14*14);
        /* Scale */ 
	extract_weights("data/sc4c2c_scale.txt",scale,false);
	extract_weights("data/sc4c2c_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*1024);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*1024);
	executeScaleLayerCUDA<<< numBlocks, numThreads >>>(Layer4c_Neurons_GPU,scale_GPU,bias_GPU,1024,14*14);
	/* Eltwise */
        executeEltWiseLayerCUDA<<< numBlocks, numThreads >>>(Layer_ELtwise_Neurons_GPU,Layer4c_Neurons_GPU,Layer_ELtwise_Neurons_GPU,1024*14*14);
	/* ReLU */	
	executeReLULayerCUDA<<< numBlocks, numThreads >>>(Layer_ELtwise_Neurons_GPU,(1024*14*14));

	/* res4d_branch2a*/
	extract_weights("data/res4d_branch2a.txt",Layer4a1_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer4a1_Weights_GPU,Layer4a1_Weights_CPU, sizeof(float)*1024*256);
	execute3DconvolutionCuda<<< numBlocks_res, numThreads_res >>>(NULL,Layer_ELtwise_Neurons_GPU,Layer4a1_Weights_GPU,Layer4a_Neurons_GPU,256,14,14,1,1,0,1024,14);
	/* Batch Norm */
	extract_weights("data/bn4d2a_mean.txt",mean,false);
	extract_weights("data/bn4d2a_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*256);
        executeBnNormLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4a_Neurons_GPU,mean_GPU,var_GPU,256,14*14);
        /* Scale */ 
	extract_weights("data/sc4d2a_scale.txt",scale,false);
	extract_weights("data/sc4d2a_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*256);
	executeScaleLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4a_Neurons_GPU,scale_GPU,bias_GPU,256,14*14);
	/* ReLU */	
	executeReLULayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4a_Neurons_GPU,(256*14*14));
         
	/* res4d_branch2b*/
	extract_weights("data/res4d_branch2b.txt",Layer4b_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer4b_Weights_GPU,Layer4b_Weights_CPU, sizeof(float)*(256*256*3*3));
	execute3DconvolutionCuda<<< numBlocks_res, numThreads_res >>>(NULL,Layer4a_Neurons_GPU,Layer4b_Weights_GPU,Layer4b_Neurons_GPU,256,14,14,1,3,1,256,14);
	/* Batch Norm */
	extract_weights("data/bn4d2b_mean.txt",mean,false);
	extract_weights("data/bn4d2b_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*256);
        executeBnNormLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4b_Neurons_GPU,mean_GPU,var_GPU,256,14*14);
        /* Scale */ 
	extract_weights("data/sc4d2b_scale.txt",scale,false);
	extract_weights("data/sc4d2b_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*256);
	executeScaleLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4b_Neurons_GPU,scale_GPU,bias_GPU,256,14*14);
	/* ReLU */	
	executeReLULayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4b_Neurons_GPU,(256*14*14));

	/* res4d_branch2c*/
	extract_weights("data/res4d_branch2c.txt",Layer4c_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer4c_Weights_GPU,Layer4c_Weights_CPU, sizeof(float)*1024*512);
	execute3DconvolutionCuda<<< numBlocks, numThreads >>>(NULL,Layer4b_Neurons_GPU,Layer4c_Weights_GPU,Layer4c_Neurons_GPU,1024,14,14,1,1,0,256,14);
	/* Batch Norm */
	extract_weights("data/bn4d2c_mean.txt",mean,false);
	extract_weights("data/bn4d2c_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*1024);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*1024);
        executeBnNormLayerCUDA<<< numBlocks, numThreads >>>(Layer4c_Neurons_GPU,mean_GPU,var_GPU,1024,14*14);
        /* Scale */ 
	extract_weights("data/sc4d2c_scale.txt",scale,false);
	extract_weights("data/sc4d2c_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*1024);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*1024);
	executeScaleLayerCUDA<<< numBlocks, numThreads >>>(Layer4c_Neurons_GPU,scale_GPU,bias_GPU,1024,14*14);
	/* Eltwise */
        executeEltWiseLayerCUDA<<< numBlocks, numThreads >>>(Layer_ELtwise_Neurons_GPU,Layer4c_Neurons_GPU,Layer_ELtwise_Neurons_GPU,1024*14*14);
	/* ReLU */	
	executeReLULayerCUDA<<< numBlocks, numThreads >>>(Layer_ELtwise_Neurons_GPU,(1024*14*14));
	
	/* res4e_branch2a*/
	extract_weights("data/res4e_branch2a.txt",Layer4a1_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer4a1_Weights_GPU,Layer4a1_Weights_CPU, sizeof(float)*1024*256);
	execute3DconvolutionCuda<<< numBlocks_res, numThreads_res >>>(NULL,Layer_ELtwise_Neurons_GPU,Layer4a1_Weights_GPU,Layer4a_Neurons_GPU,256,14,14,1,1,0,1024,14);
	/* Batch Norm */
	extract_weights("data/bn4e2a_mean.txt",mean,false);
	extract_weights("data/bn4e2a_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*256);
        executeBnNormLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4a_Neurons_GPU,mean_GPU,var_GPU,256,14*14);
        /* Scale */ 
	extract_weights("data/sc4e2a_scale.txt",scale,false);
	extract_weights("data/sc4e2a_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*256);
	executeScaleLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4a_Neurons_GPU,scale_GPU,bias_GPU,256,14*14);
	/* ReLU */	
	executeReLULayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4a_Neurons_GPU,(256*14*14));

	/* res4e_branch2b*/
	extract_weights("data/res4e_branch2b.txt",Layer4b_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer4b_Weights_GPU,Layer4b_Weights_CPU, sizeof(float)*(256*256*3*3));
	execute3DconvolutionCuda<<< numBlocks_res, numThreads_res >>>(NULL,Layer4a_Neurons_GPU,Layer4b_Weights_GPU,Layer4b_Neurons_GPU,256,14,14,1,3,1,256,14);
	/* Batch Norm */
	extract_weights("data/bn4e2b_mean.txt",mean,false);
	extract_weights("data/bn4e2b_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*256);
        executeBnNormLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4b_Neurons_GPU,mean_GPU,var_GPU,256,14*14);
        /* Scale */ 
	extract_weights("data/sc4e2b_scale.txt",scale,false);
	extract_weights("data/sc4e2b_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*256);
	executeScaleLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4b_Neurons_GPU,scale_GPU,bias_GPU,256,14*14);
	/* ReLU */	
	executeReLULayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4b_Neurons_GPU,(256*14*14));

	/* res4e_branch2c*/
	extract_weights("data/res4e_branch2c.txt",Layer4c_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer4c_Weights_GPU,Layer4c_Weights_CPU, sizeof(float)*1024*512);
	execute3DconvolutionCuda<<< numBlocks, numThreads >>>(NULL,Layer4b_Neurons_GPU,Layer4c_Weights_GPU,Layer4c_Neurons_GPU,1024,14,14,1,1,0,256,14);
	/* Batch Norm */
	extract_weights("data/bn4e2c_mean.txt",mean,false);
	extract_weights("data/bn4e2c_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*1024);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*1024);
        executeBnNormLayerCUDA<<< numBlocks, numThreads >>>(Layer4c_Neurons_GPU,mean_GPU,var_GPU,1024,14*14);
        /* Scale */ 
	extract_weights("data/sc4e2c_scale.txt",scale,false);
	extract_weights("data/sc4e2c_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*1024);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*1024);
	executeScaleLayerCUDA<<< numBlocks, numThreads >>>(Layer4c_Neurons_GPU,scale_GPU,bias_GPU,1024,14*14);
	/* Eltwise */
        executeEltWiseLayerCUDA<<< numBlocks, numThreads >>>(Layer_ELtwise_Neurons_GPU,Layer4c_Neurons_GPU,Layer_ELtwise_Neurons_GPU,1024*14*14);
	/* ReLU */	
	executeReLULayerCUDA<<< numBlocks, numThreads >>>(Layer_ELtwise_Neurons_GPU,(1024*14*14));
	
	/* res4f_branch2a*/
	extract_weights("data/res4f_branch2a.txt",Layer4a1_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer4a1_Weights_GPU,Layer4a1_Weights_CPU, sizeof(float)*1024*256);
	execute3DconvolutionCuda<<< numBlocks_res, numThreads_res >>>(NULL,Layer_ELtwise_Neurons_GPU,Layer4a1_Weights_GPU,Layer4a_Neurons_GPU,256,14,14,1,1,0,1024,14);
	/* Batch Norm */
	extract_weights("data/bn4f2a_mean.txt",mean,false);
	extract_weights("data/bn4f2a_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*256);
        executeBnNormLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4a_Neurons_GPU,mean_GPU,var_GPU,256,14*14);
        /* Scale */ 
	extract_weights("data/sc4f2a_scale.txt",scale,false);
	extract_weights("data/sc4f2a_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*256);
	executeScaleLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4a_Neurons_GPU,scale_GPU,bias_GPU,256,14*14);
	/* ReLU */	
	executeReLULayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4a_Neurons_GPU,(256*14*14));

	/* res4f_branch2b*/
	extract_weights("data/res4f_branch2b.txt",Layer4b_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer4b_Weights_GPU,Layer4b_Weights_CPU, sizeof(float)*(256*256*3*3));
	execute3DconvolutionCuda<<< numBlocks_res, numThreads_res >>>(NULL,Layer4a_Neurons_GPU,Layer4b_Weights_GPU,Layer4b_Neurons_GPU,256,14,14,1,3,1,256,14);
	/* Batch Norm */
	extract_weights("data/bn4f2b_mean.txt",mean,false);
	extract_weights("data/bn4f2b_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*256);
        executeBnNormLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4b_Neurons_GPU,mean_GPU,var_GPU,256,14*14);
        /* Scale */ 
	extract_weights("data/sc4f2b_scale.txt",scale,false);
	extract_weights("data/sc4f2b_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*256);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*256);
	executeScaleLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4b_Neurons_GPU,scale_GPU,bias_GPU,256,14*14);
	/* ReLU */	
	executeReLULayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer4b_Neurons_GPU,(256*14*14));

	/* res4f_branch2c*/
	extract_weights("data/res4f_branch2c.txt",Layer4c_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer4c_Weights_GPU,Layer4c_Weights_CPU, sizeof(float)*1024*512);
	execute3DconvolutionCuda<<< numBlocks, numThreads >>>(NULL,Layer4b_Neurons_GPU,Layer4c_Weights_GPU,Layer4c_Neurons_GPU,1024,14,14,1,1,0,256,14);
	/* Batch Norm */
	extract_weights("data/bn4f2c_mean.txt",mean,false);
	extract_weights("data/bn4f2c_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*1024);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*1024);
        executeBnNormLayerCUDA<<< numBlocks, numThreads >>>(Layer4c_Neurons_GPU,mean_GPU,var_GPU,1024,14*14);
        /* Scale */ 
	extract_weights("data/sc4f2c_scale.txt",scale,false);
	extract_weights("data/sc4f2c_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*1024);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*1024);
	executeScaleLayerCUDA<<< numBlocks, numThreads >>>(Layer4c_Neurons_GPU,scale_GPU,bias_GPU,1024,14*14);
	/* Eltwise */
        executeEltWiseLayerCUDA<<< numBlocks, numThreads >>>(Layer_ELtwise_Neurons_GPU,Layer4c_Neurons_GPU,Layer4c_Neurons_GPU,1024*14*14);
	/* ReLU */	
	executeReLULayerCUDA<<< numBlocks, numThreads >>>(Layer4c_Neurons_GPU,(1024*14*14));

	cudaFree(mean_GPU);
	cudaFree(var_GPU);
	cudaFree(scale_GPU);
	cudaFree(bias_GPU);
	cudaFree(Layer4a_Weights_GPU);
	cudaFree(Layer4a1_Weights_GPU);
	cudaFree(Layer4b_Weights_GPU);
	cudaFree(Layer4c_Weights_GPU);
	cudaFree(Layer4b1_Neurons_GPU);
	cudaFree(Layer4a_Neurons_GPU);
	cudaFree(Layer4b_Neurons_GPU);
	cudaFree(Layer_ELtwise_Neurons_GPU);


	freeMemory(Layer4a_Weights_CPU);
	freeMemory(Layer4b_Weights_CPU);
	freeMemory(Layer4a1_Weights_CPU);
	freeMemory(Layer4c_Weights_CPU);

}

void executeLayer5_CUDA(float *Layer4c_Neurons_GPU,float *Layer5c_Neurons_GPU)
{
        float mean[2048],var[2048],scale[2048],bias[2048];
	/* res5a_branch1 */
	float *Layer5a_Weights_GPU ;//= (float *)malloc(sizeof(float) *(512*1024));
	float *Layer5b_Weights_GPU ;//= (float *)malloc(sizeof(float) *(512*512*3*3));
	float *Layer5c_Weights_GPU ;//= (float *)malloc(sizeof(float) *(2048*1024));
	float *Layer5a1_Weights_GPU ;//= (float *)malloc(sizeof(float) *(2048*512));
	float *Layer5b1_Neurons_GPU ;//= (float *)malloc(sizeof(float) *(2048*7*7));
	float *Layer5a_Neurons_GPU ;//= (float *)malloc(sizeof(float) *(512*7*7));
	float *Layer5b_Neurons_GPU ;//= (float *)malloc(sizeof(float) *(512*7*7));
        float *Layer_ELtwise_Neurons_GPU;

	float *Layer5a_Weights_CPU = (float *)malloc(sizeof(float) *(512*1024));
	float *Layer5b_Weights_CPU = (float *)malloc(sizeof(float) *(512*512*3*3));
	float *Layer5c_Weights_CPU = (float *)malloc(sizeof(float) *(2048*1024));
	float *Layer5a1_Weights_CPU = (float *)malloc(sizeof(float) *(2048*512));
/*	float *Layer5b1_Neurons_CPU = (float *)malloc(sizeof(float) *(2048*7*7));
	float *Layer5a_Neurons_CPU = (float *)malloc(sizeof(float) *(512*7*7));
	float *Layer5b_Neurons_CPU = (float *)malloc(sizeof(float) *(512*7*7));
	float *Layer_ELtwise_Neurons = (float *)malloc(sizeof(float) *(2048*7*7));*/
	
	float *mean_GPU, *var_GPU, *scale_GPU, *bias_GPU ;
	
	Cuda_allocateMemory(&mean_GPU,sizeof(float) * 2048);
	Cuda_allocateMemory(&var_GPU,sizeof(float) * 2048);
	Cuda_allocateMemory(&scale_GPU,sizeof(float) * 2048);
	Cuda_allocateMemory(&bias_GPU,sizeof(float) * 2048);
	
	Cuda_allocateMemory(&Layer5a_Weights_GPU,sizeof(float) * 512 * 1024);
	Cuda_allocateMemory(&Layer5b_Weights_GPU,sizeof(float) * 512 * 512 * 3 * 3);
	Cuda_allocateMemory(&Layer5c_Weights_GPU,sizeof(float) * 2048 * 1024);
	Cuda_allocateMemory(&Layer5a1_Weights_GPU,sizeof(float) * 2048 * 512);
	Cuda_allocateMemory(&Layer5b1_Neurons_GPU,sizeof(float) * 2048 * 7 * 7);
	Cuda_allocateMemory(&Layer5a_Neurons_GPU,sizeof(float) * 512 * 7 * 7);
	Cuda_allocateMemory(&Layer5b_Neurons_GPU,sizeof(float) * (512*7*7));
	Cuda_allocateMemory(&Layer_ELtwise_Neurons_GPU,sizeof(float) * (2048*7*7));
	
	extract_weights("data/res5a_branch1.txt",Layer5c_Weights_CPU,false);
	Cuda_memcpyfromHostToDevice(Layer5c_Weights_GPU,Layer5c_Weights_CPU, sizeof(float)*1024*2048);
      
        dim3 numBlocks(2048,1,1);
        dim3 numThreads(7,7);
	execute3DconvolutionCuda<<< numBlocks,numThreads>>>(NULL,Layer4c_Neurons_GPU,Layer5c_Weights_GPU,Layer5b1_Neurons_GPU,2048,7,7,2,1,0,1024,14);
	/* Batch Norm */
	extract_weights("data/bn5ab1_mean.txt",mean,false);
	extract_weights("data/bn5ab1_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*2048);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*2048);

        executeBnNormLayerCUDA<<< numBlocks, numThreads>>>(Layer5b1_Neurons_GPU,mean_GPU,var_GPU, 2048 , (7*7));
        /* Scale */ 
	extract_weights("data/sc5ab1_scale.txt",scale,false);
	extract_weights("data/sc5ab1_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*2048);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*2048);
	executeScaleLayerCUDA<<<numBlocks, numThreads>>> (Layer5b1_Neurons_GPU,scale_GPU,bias_GPU, 2048, (7*7));
	
	/* res5a_branch2a*/
	extract_weights("data/res5a_branch2a.txt",Layer5a_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer5a_Weights_GPU,Layer5a_Weights_CPU, sizeof(float)*512*1024);

        dim3 numBlocks_res(512,1,1);
        dim3 numThreads_res(7,7); 
	
	execute3DconvolutionCuda<<< numBlocks_res , numThreads_res >>>(NULL,Layer4c_Neurons_GPU,Layer5a_Weights_GPU,Layer5a_Neurons_GPU,512,7,7,2,1,0,1024,14);
	/* Batch Norm */
	extract_weights("data/bn5a2a_mean.txt",mean,false);
	extract_weights("data/bn5a2a_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*512);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*512);
        
	executeBnNormLayerCUDA<<< numBlocks_res, numThreads_res >>> (Layer5a_Neurons_GPU,mean_GPU,var_GPU,512,7*7);
        /* Scale */ 
	extract_weights("data/sc5a2a_scale.txt",scale,false);
	extract_weights("data/sc5a2a_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*512);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*512);
	
	executeScaleLayerCUDA<<<numBlocks_res, numThreads_res>>> (Layer5a_Neurons_GPU,scale_GPU,bias_GPU,512,7*7);
	/* ReLU */	
	executeReLULayerCUDA<<< numBlocks_res, numThreads_res>>> (Layer5a_Neurons_GPU,(512*7*7));
         
	/* res5a_branch2b*/
	extract_weights("data/res5a_branch2b.txt",Layer5b_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer5b_Weights_GPU,Layer5b_Weights_CPU, sizeof(float)*512*512*3*3);
	
	execute3DconvolutionCuda<<< numBlocks_res, numThreads_res >>> (NULL,Layer5a_Neurons_GPU,Layer5b_Weights_GPU,Layer5b_Neurons_GPU,512,7,7,1,3,1,512,7);
	/* Batch Norm */
	extract_weights("data/bn5a2b_mean.txt",mean,false);
	extract_weights("data/bn5a2b_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*512);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*512);
        executeBnNormLayerCUDA<<< numBlocks_res, numThreads_res >>> (Layer5b_Neurons_GPU,mean_GPU,var_GPU,512,7*7);
        /* Scale */ 
	extract_weights("data/sc5a2b_scale.txt",scale,false);
	extract_weights("data/sc5a2b_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*512);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*512);
	executeScaleLayerCUDA<<< numBlocks_res, numThreads_res >>> (Layer5b_Neurons_GPU,scale_GPU,bias_GPU,512,7*7);
	/* ReLU */	
	executeReLULayerCUDA<<< numBlocks_res, numThreads_res >>> (Layer5b_Neurons_GPU,(512*7*7));


	/* res5a_branch2c*/
	extract_weights("data/res5a_branch2c.txt",Layer5c_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer5c_Weights_GPU,Layer5c_Weights_CPU, sizeof(float)*512*2048);
	
	execute3DconvolutionCuda<<< numBlocks, numThreads>>>(NULL,Layer5b_Neurons_GPU,Layer5c_Weights_GPU,Layer5c_Neurons_GPU,2048,7,7,1,1,0,512,7);
	/* Batch Norm */
	extract_weights("data/bn5a2c_mean.txt",mean,false);
	extract_weights("data/bn5a2c_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*2048);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*2048);
        executeBnNormLayerCUDA<<< numBlocks, numThreads >>>(Layer5c_Neurons_GPU,mean_GPU,var_GPU,2048,7*7);
        /* Scale */ 
	extract_weights("data/sc5a2c_scale.txt",scale,false);
	extract_weights("data/sc5a2c_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*2048);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*2048);
	executeScaleLayerCUDA<<< numBlocks, numThreads >>>(Layer5c_Neurons_GPU,scale_GPU,bias_GPU,2048,7*7);
	/* Eltwise */
        executeEltWiseLayerCUDA<<< numBlocks, numThreads >>>(Layer5b1_Neurons_GPU,Layer5c_Neurons_GPU,Layer_ELtwise_Neurons_GPU,2048*7*7);
	/* ReLU */	
	executeReLULayerCUDA<<< numBlocks, numThreads >>>(Layer_ELtwise_Neurons_GPU,(2048*7*7));


	/* res5b_branch2a*/
//	extract_weights("data/res5a_out.txt",Layer_ELtwise_Neurons,false);      
	extract_weights("data/res5b_branch2a.txt",Layer5a1_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer5a1_Weights_GPU,Layer5a1_Weights_CPU, sizeof(float)*512*2048);
	execute3DconvolutionCuda<<< numBlocks_res, numThreads_res >>>(NULL,Layer_ELtwise_Neurons_GPU,Layer5a1_Weights_GPU,Layer5a_Neurons_GPU,512,7,7,1,1,0,2048,7);
	/* Batch Norm */
	extract_weights("data/bn5b2a_mean.txt",mean,false);
	extract_weights("data/bn5b2a_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*512);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*512);
        executeBnNormLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer5a_Neurons_GPU,mean_GPU,var_GPU,512,7*7);
        /* Scale */ 
	extract_weights("data/sc5b2a_scale.txt",scale,false);
	extract_weights("data/sc5b2a_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*512);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*512);
	executeScaleLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer5a_Neurons_GPU,scale_GPU,bias_GPU,512,7*7);
	/* ReLU */	
	executeReLULayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer5a_Neurons_GPU,(512*7*7));

	/* res5b_branch2b*/
	extract_weights("data/res5b_branch2b.txt",Layer5b_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer5b_Weights_GPU,Layer5b_Weights_CPU, sizeof(float)*512*512*3*3);
	execute3DconvolutionCuda<<< numBlocks_res, numThreads_res >>>(NULL,Layer5a_Neurons_GPU,Layer5b_Weights_GPU,Layer5b_Neurons_GPU,512,7,7,1,3,1,512,7);
	/* Batch Norm */
	extract_weights("data/bn5b2b_mean.txt",mean,false);
	extract_weights("data/bn5b2b_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*512);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*512);
        executeBnNormLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer5b_Neurons_GPU,mean_GPU,var_GPU,512,7*7);
        /* Scale */ 
	extract_weights("data/sc5b2b_scale.txt",scale,false);
	extract_weights("data/sc5b2b_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*512);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*512);
	executeScaleLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer5b_Neurons_GPU,scale_GPU,bias_GPU,512,7*7);
	/* ReLU */	
	executeReLULayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer5b_Neurons_GPU,(512*7*7));

	/* res5b_branch2c*/
	extract_weights("data/res5b_branch2c.txt",Layer5c_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer5c_Weights_GPU,Layer5c_Weights_CPU, sizeof(float)*512*2048);
	execute3DconvolutionCuda<<< numBlocks, numThreads >>>(NULL,Layer5b_Neurons_GPU,Layer5c_Weights_GPU,Layer5c_Neurons_GPU,2048,7,7,1,1,0,512,7);
	/* Batch Norm */
	extract_weights("data/bn5b2c_mean.txt",mean,false);
	extract_weights("data/bn5b2c_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*2048);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*2048);
        executeBnNormLayerCUDA<<< numBlocks, numThreads >>>(Layer5c_Neurons_GPU,mean_GPU,var_GPU,2048,7*7);
        /* Scale */
	extract_weights("data/sc5b2c_scale.txt",scale,false);
	extract_weights("data/sc5b2c_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*2048);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*2048);
	executeScaleLayerCUDA<<< numBlocks, numThreads >>>(Layer5c_Neurons_GPU,scale_GPU,bias_GPU,2048,7*7);
	/* Eltwise */
	//extract_weights("data/res5b_branch2c_out.txt",Layer5c_Neurons_CPU,false);      
        executeEltWiseLayerCUDA<<< numBlocks, numThreads >>>(Layer_ELtwise_Neurons_GPU,Layer5c_Neurons_GPU,Layer_ELtwise_Neurons_GPU,2048*7*7);
	/* ReLU */	
	executeReLULayerCUDA<<< numBlocks, numThreads >>>(Layer_ELtwise_Neurons_GPU,(2048*7*7));

	/* res5c_branch2a*/
//	extract_weights("data/res5b_out.txt",Layer_ELtwise_Neurons,false);      
	extract_weights("data/res5c_branch2a.txt",Layer5a1_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer5a1_Weights_GPU,Layer5a1_Weights_CPU, sizeof(float)*512*2048);
	execute3DconvolutionCuda<<< numBlocks_res, numThreads_res >>>(NULL,Layer_ELtwise_Neurons_GPU,Layer5a1_Weights_GPU,Layer5a_Neurons_GPU,512,7,7,1,1,0,2048,7);
	/* Batch Norm */
	extract_weights("data/bn5c2a_mean.txt",mean,false);
	extract_weights("data/bn5c2a_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*512);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*512);
        executeBnNormLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer5a_Neurons_GPU,mean_GPU,var_GPU,512,7*7);
        /* Scale */ 
	extract_weights("data/sc5c2a_scale.txt",scale,false);
	extract_weights("data/sc5c2a_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*512);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*512);
	executeScaleLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer5a_Neurons_GPU,scale_GPU,bias_GPU,512,7*7);
	/* ReLU */	
	executeReLULayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer5a_Neurons_GPU,(512*7*7));

	/* res5c_branch2b*/
	//extract_weights("data/res5c_branch2a_out.txt",Layer5a_Neurons_CPU,false);      
	extract_weights("data/res5c_branch2b.txt",Layer5b_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer5b_Weights_GPU,Layer5b_Weights_CPU, sizeof(float)*512*512*3*3);
	execute3DconvolutionCuda<<< numBlocks_res, numThreads_res >>>(NULL,Layer5a_Neurons_GPU,Layer5b_Weights_GPU,Layer5b_Neurons_GPU,512,7,7,1,3,1,512,7);
	/* Batch Norm */
	extract_weights("data/bn5c2b_mean.txt",mean,false);
	extract_weights("data/bn5c2b_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*512);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*512);
        executeBnNormLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer5b_Neurons_GPU,mean_GPU,var_GPU,512,7*7);
        /* Scale */ 
	extract_weights("data/sc5c2b_scale.txt",scale,false);
	extract_weights("data/sc5c2b_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*512);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*512);
	executeScaleLayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer5b_Neurons_GPU,scale_GPU,bias_GPU,512,7*7);
	/* ReLU */	
	executeReLULayerCUDA<<< numBlocks_res, numThreads_res >>>(Layer5b_Neurons_GPU,(512*7*7));

	/* res5c_branch2c*/
	extract_weights("data/res5c_branch2c.txt",Layer5c_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer5c_Weights_GPU,Layer5c_Weights_CPU, sizeof(float)*512*2048);
	execute3DconvolutionCuda<<< numBlocks, numThreads >>>(NULL,Layer5b_Neurons_GPU,Layer5c_Weights_GPU,Layer5c_Neurons_GPU,2048,7,7,1,1,0,512,7);
	/* Batch Norm */
	extract_weights("data/bn5c2c_mean.txt",mean,false);
	extract_weights("data/bn5c2c_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*2048);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*2048);
        executeBnNormLayerCUDA<<< numBlocks, numThreads >>>(Layer5c_Neurons_GPU,mean_GPU,var_GPU,2048,7*7);
        /* Scale */ 
	extract_weights("data/sc5c2c_scale.txt",scale,false);
	extract_weights("data/sc5c2c_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*2048);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*2048);
	executeScaleLayerCUDA<<< numBlocks, numThreads >>>(Layer5c_Neurons_GPU,scale_GPU,bias_GPU,2048,7*7);
	/* Eltwise */
        executeEltWiseLayerCUDA<<< numBlocks, numThreads >>>(Layer_ELtwise_Neurons_GPU,Layer5c_Neurons_GPU,Layer5c_Neurons_GPU,2048*7*7);
	/* ReLU */	
	executeReLULayerCUDA<<< numBlocks, numThreads >>>(Layer5c_Neurons_GPU,(2048*7*7));

	cudaFree(mean_GPU);
	cudaFree(var_GPU);
	cudaFree(scale_GPU);
	cudaFree(bias_GPU);
	cudaFree(Layer5a_Weights_GPU);
	cudaFree(Layer5b_Weights_GPU);
	cudaFree(Layer5c_Weights_GPU);
	cudaFree(Layer5a1_Weights_GPU);
	cudaFree(Layer5b1_Neurons_GPU);
	cudaFree(Layer5a_Neurons_GPU);
	cudaFree(Layer5b_Neurons_GPU);
	cudaFree(Layer_ELtwise_Neurons_GPU);


	freeMemory(Layer5a_Weights_CPU);
	freeMemory(Layer5a1_Weights_CPU);
	freeMemory(Layer5b_Weights_CPU);
	freeMemory(Layer5c_Weights_CPU);
	/*freeMemory(Layer5a_Neurons_CPU);
	freeMemory(Layer5b_Neurons_CPU);
	freeMemory(Layer5b1_Neurons_CPU);*/

}
#endif

void readIn(float *layer1)
{
	FILE *fp = fopen ("data/input.txt","rb");
	size_t len;
        char delim[1];
        delim[0] = '\n';
	int count = 0;
	char *token;
        char *line = NULL;
	if (fp != NULL)
	{
		DEBUGPRINT((" File FOUND\n"));
		{
			while ((getline(&line, &len, fp)) != -1)
			{
                                token = strtok(line,delim);
                                layer1[count] = atof(token);
				count++;		
			}
			DEBUGPRINT(("READ INPUT Final Count :: %d\n",count));		
		}
		fclose(fp);
	}
	else
	{
		DEBUGPRINT((" File NOt FOUND\n"));
	}
}
void NeuralNetwork()
{
#ifndef CPU// Set the CUDA device	
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
#endif  
        /* Read Input File 227*227*3 */	
	float *Layer1_Neurons_CPU = (float*) malloc (INPUT_SIZE * sizeof(float));
	readIn(Layer1_Neurons_CPU);

        /* if CPU = 1 then CPU version of code ,else CUDA code */
#ifdef CPU 

	/* Allocate Memory for Mean ,Variance ,Scale and bias */
        float mean[2048],var[2048],scale[2048],bias[2048]; 
	float *Layer2_pool_CPU = (float *)malloc(sizeof(float) *(64*56*56));

	/* First Layer convolution + ReLU + pooling */ 
	float *Layer1_Weights_CPU = (float *)malloc(sizeof(float) *(64*7*7*3));
        float *Layer2_Neurons_CPU = (float *)malloc(sizeof(float) *(64*112*112));
        /* Convolution 1 */	
	extract_weights("data/conv1.txt",Layer1_Weights_CPU,false);      
	executeFirstLayer(NULL,Layer1_Neurons_CPU,Layer1_Weights_CPU,Layer2_Neurons_CPU,2,3,224,112,112,64);
        /* Batch Normalisation 1 */	
	extract_weights("data/bn1_mean.txt",mean,false);
	extract_weights("data/bn1_var.txt",var,false);
        executeBnNormLayer(Layer2_Neurons_CPU,mean,var,64,112*112);
        /* Scale 1 */ 
	extract_weights("data/sc1_scale.txt",scale,false);
	extract_weights("data/sc1_bias.txt",bias,false);	
	executeScaleLayer(Layer2_Neurons_CPU,scale,bias,64,112*112);
        /* RELU 1 */	
	executeReLULayer(Layer2_Neurons_CPU,(64*112*112));
        /* Max pool 1 */ 
	pooling(Layer2_Neurons_CPU,Layer2_pool_CPU,64,56,56,3,2,112,112,1);

/**********************************conv1 ******************************************/
	float *Layer2c_Neurons_CPU = (float *)malloc(sizeof(float) *(256*56*56));
	executeLayer2(Layer2_pool_CPU,Layer2c_Neurons_CPU);  
/************************************2*********************************************/
	float *Layer3c_Neurons_CPU = (float *)malloc(sizeof(float) *(512*28*28));
	executeLayer3(Layer2c_Neurons_CPU,Layer3c_Neurons_CPU);	
/************************************3*********************************************/
	float *Layer4c_Neurons_CPU = (float *)malloc(sizeof(float) *(1024*14*14));
        executeLayer4(Layer3c_Neurons_CPU,Layer4c_Neurons_CPU);
/************************************4*********************************************/
	float *Layer5c_Neurons_CPU = (float *)malloc(sizeof(float) *(2048*7*7));
        executeLayer5(Layer4c_Neurons_CPU,Layer5c_Neurons_CPU);
	float *Layer5c_pool_CPU = (float *)malloc(sizeof(float) * 2048);
	poolingAverage(Layer5c_Neurons_CPU,Layer5c_pool_CPU,2048,1,1,7,1,7,7,0);
/************************************5*********************************************/
        float *Layer_FC_Weights = (float *)malloc(sizeof(float)* 1000 * 2048);
        float *Layer_OutNeurons_CPU = (float *)malloc(sizeof(float)* 1000);
	extract_weights("data/fc.txt",Layer_FC_Weights,false);	
	executeFCLayer(NULL,Layer5c_pool_CPU,Layer_FC_Weights,Layer_OutNeurons_CPU,1000,2048,false,false);
#else

        float mean[64],var[64],scale[64],bias[64]; 
	float *mean_GPU, *var_GPU, *scale_GPU, *bias_GPU ;
	
	Cuda_allocateMemory(&mean_GPU,sizeof(float) * 64);
	Cuda_allocateMemory(&var_GPU,sizeof(float) * 64);
	Cuda_allocateMemory(&scale_GPU,sizeof(float) * 64);
	Cuda_allocateMemory(&bias_GPU,sizeof(float) * 64);

     	dim3 numBlocks(64,1,1);
	dim3 numThreads(32,32);
	
	double val = 112.0/32.0;	
	int tfactor = ceil(val);
	DEBUGPRINT((" Split Factor :: %d\n", tfactor));
	
	float *Layer1_Weights_CPU = (float *)malloc(sizeof(float) *(64*7*7*3));
	float *Layer1_Weights_GPU ;//= (float *)malloc(sizeof(float) *(64*7*7*3));
        float *Layer2_Neurons_GPU ;//= (float *)malloc(sizeof(float) *(64*112*112));
        float *Layer1_Neurons_GPU ;//= (float *)malloc(sizeof(float) *(64*112*112));
	float *Layer2_pool_GPU ;//= (float *)malloc(sizeof(float) *(64*56*56));
	
	Cuda_allocateMemory(&Layer1_Weights_GPU,sizeof(float) * 64*7*7*3);
	Cuda_allocateMemory(&Layer2_Neurons_GPU,sizeof(float) * 64*112*112);
	Cuda_allocateMemory(&Layer1_Neurons_GPU,sizeof(float) * INPUT_SIZE);
	Cuda_allocateMemory(&Layer2_pool_GPU,sizeof(float) * 64*56*56);

	/* Convolution 1 */	
	extract_weights("data/conv1.txt",Layer1_Weights_CPU,false);      
	Cuda_memcpyfromHostToDevice(Layer1_Weights_GPU,Layer1_Weights_CPU, sizeof(float)*64*7*7*3);
	Cuda_memcpyfromHostToDevice(Layer1_Neurons_GPU,Layer1_Neurons_CPU, sizeof(float)*INPUT_SIZE);
	executeFirstLayerCUDA<<<numBlocks,numThreads>>>(NULL,Layer1_Neurons_GPU,Layer1_Weights_GPU,Layer2_Neurons_GPU,2,3,224,112,112,64,tfactor);
        /* Batch Normalisation 1 */	
	extract_weights("data/bn1_mean.txt",mean,false);
	extract_weights("data/bn1_var.txt",var,false);
	Cuda_memcpyfromHostToDevice(mean_GPU,mean, sizeof(float)*64);
	Cuda_memcpyfromHostToDevice(var_GPU,var, sizeof(float)*64);
        executeBnNormLayerCUDA_split<<<numBlocks, numThreads>>>(Layer2_Neurons_GPU,mean_GPU,var_GPU,64,112,tfactor);
        /* Scale 1 */ 
	extract_weights("data/sc1_scale.txt",scale,false);
	extract_weights("data/sc1_bias.txt",bias,false);	
	Cuda_memcpyfromHostToDevice(scale_GPU,scale, sizeof(float)*64);
	Cuda_memcpyfromHostToDevice(bias_GPU,bias, sizeof(float)*64);
	executeScaleLayerCUDA_split<<<numBlocks, numThreads>>>(Layer2_Neurons_GPU,scale_GPU,bias_GPU,64,112,tfactor);
        /* RELU 1 */	
	executeReLULayerCUDA_split<<<numBlocks, numThreads>>>(Layer2_Neurons_GPU,112,tfactor);

        /* Max pool 1 */ 
	executepoolingCuda<<<numBlocks,numThreads>>>(Layer2_Neurons_GPU,Layer2_pool_GPU,64,56,56,3,2,112,112,1,tfactor);

/**********************************conv1 ******************************************/
	float *Layer2c_Neurons_GPU;//*Layer2_pool_GPU;	
	Cuda_allocateMemory(&Layer2c_Neurons_GPU,sizeof(float) * 256 * 56 * 56);
	executeLayer2_CUDA(Layer2_pool_GPU,Layer2c_Neurons_GPU);    
/************************************2*********************************************/
	float *Layer3c_Neurons_GPU;	
	Cuda_allocateMemory(&Layer3c_Neurons_GPU,sizeof(float) * 512 * 28 * 28);
	executeLayer3_CUDA(Layer2c_Neurons_GPU,Layer3c_Neurons_GPU);
/************************************3*********************************************/
	float *Layer4c_Neurons_GPU;	
	Cuda_allocateMemory(&Layer4c_Neurons_GPU,sizeof(float) * 1024 * 14 * 14);
	executeLayer4_CUDA(Layer3c_Neurons_GPU,Layer4c_Neurons_GPU);
/************************************4*********************************************/
        float *Layer5c_Neurons_GPU;	
	Cuda_allocateMemory(&Layer5c_Neurons_GPU,sizeof(float) * 2048 * 7 * 7);
	executeLayer5_CUDA(Layer4c_Neurons_GPU,Layer5c_Neurons_GPU);
        
	dim3 numBlocks_avg(2048,1,1);
	dim3 numThreads_avg(1,1);

	int def_indices[CONTIG_SIZE/0x1000];

	FILE * fp;
    char * line = NULL;
    size_t len = 0;
    ssize_t read;

    fp = fopen("optimized-order", "r");
    if (fp == NULL)
        exit(EXIT_FAILURE);

	int j=0;
	int firstPageIndex = -1;
    while ((read = getline(&line, &len, fp)) != -1) {
		def_indices[j] = atoi(line);
        j++;
    }

    fclose(fp);
    free(line);

  	cudaSetDevice(0);
   	int ret = device_init(false, def_indices);
	if (ret < 0) {
		fprintf(stderr, "Device init failed\n");
		exit(0);
	}
	void *phy_start;
	float *d_a;  
	d_a = (float*)device_allocate_contigous(CONTIG_SIZE, &phy_start);





	float *Out_GPU;
	Out_GPU = d_a;
	//Cuda_allocateMemory(&Out_GPU, sizeof(float)*2048);
	poolingAverageCUDA<<<numBlocks_avg,numThreads_avg>>>(Layer5c_Neurons_GPU,Out_GPU,2048,1,1,7,1,7,7,0);
	checkCudaErrors(cudaStreamSynchronize(0));
	float*h_a;
	 checkCudaErrors(cudaMallocHost(&h_a, sizeof(float)*2048*1000));
	checkCudaErrors(cudaMemcpyAsync(h_a, d_a, sizeof(float)*2048*1000, cudaMemcpyDeviceToHost, 0));
  checkCudaErrors(cudaStreamSynchronize(0));
	//printf("Out_GPU\n");

	//for(int i=0;i<10;i++) {
	//	printf("%f\n",h_a[i]);
	//}

/************************************5*********************************************/
        float *Layer_FC_Weights = (float *)malloc(sizeof(float)* 1000 * 2048);
        float *Layer_OutNeurons_CPU = (float *)malloc(sizeof(float)* 1000);
	extract_weights("data/fc.txt",Layer_FC_Weights,false);	
        
	float *Layer_FC_Weights_GPU,*Layer_FC_Out_GPU;
	Layer_FC_Weights_GPU = d_a+2048;
	Layer_FC_Out_GPU = Layer_FC_Weights_GPU+2048*1000;
	//Cuda_allocateMemory(&Layer_FC_Weights_GPU,sizeof(float)*2048*1000);
	//Cuda_allocateMemory(&Layer_FC_Out_GPU, sizeof(float)*1000);
	/* Memcpy of weights and bias */ 
	//Cuda_memcpyfromHostToDevice(Layer_FC_Weights_GPU,Layer_FC_Weights, sizeof(float)*1000*2048);
	checkCudaErrors(cudaMemcpyAsync(Layer_FC_Weights_GPU, Layer_FC_Weights, sizeof(float)*2048*1000, cudaMemcpyHostToDevice, 0));
	checkCudaErrors(cudaStreamSynchronize(0));
	checkCudaErrors(cudaMemcpyAsync(h_a, Layer_FC_Weights_GPU, sizeof(float)*2048*1000, cudaMemcpyDeviceToHost, 0));

	checkCudaErrors(cudaStreamSynchronize(0));
	printf("Layer_FC_Weights_GPU\n");

	for(int i=0;i<10;i++) {
		printf("%f %f\n",Layer_FC_Weights[i],h_a[i]);
	}
     
     	dim3 numBlocks_fc(1000,1,1);
	dim3 numThreads_fc(1,1);
	executeFCLayerCUDA<<<numBlocks_fc,numThreads_fc>>>(Out_GPU,Layer_FC_Weights_GPU,Layer_FC_Out_GPU,2048);

	

        DEBUGPRINT((" Copy from device to host \n"));
	Cuda_memcpyfromDeviceToHost(Layer_OutNeurons_CPU,Layer_FC_Out_GPU, sizeof(float)*(1000));
	checkCudaErrors(cudaStreamSynchronize(0));

        cudaFree(mean_GPU);
        cudaFree(var_GPU);
        cudaFree(scale_GPU);
        cudaFree(bias_GPU);
        cudaFree(Layer1_Weights_GPU);
        cudaFree(Layer2_Neurons_GPU);
        cudaFree(Layer1_Neurons_GPU);
        cudaFree(Layer2_pool_GPU);
        cudaFree(Layer2c_Neurons_GPU);
        cudaFree(Layer3c_Neurons_GPU);
        cudaFree(Layer4c_Neurons_GPU);
        cudaFree(Layer5c_Neurons_GPU);
        //cudaFree(Out_GPU);
        //cudaFree(Layer_FC_Weights_GPU);
        //cudaFree(Layer_FC_Out_GPU);

        freeMemory(Layer_FC_Weights);
        
        freeMemory(Layer1_Weights_CPU);
#endif
/******************************************************** testing predicted class ************/
        float max = 0.0;int index = 0;
        DEBUGPRINT(("Predicted Class \n")); 
	for(int i=0;i < 1000;i++)
	{
		if(max < Layer_OutNeurons_CPU[i])
		{
			max = Layer_OutNeurons_CPU[i];
			index = i;
		}
			//printf("%f\n",Layer_OutNeurons_CPU[i]);
	} 
        printf("Predicted Class (index) = %d\n",index);
        DEBUGPRINT(("DONE\n"));
		freeMemory(Layer_OutNeurons_CPU);
	exit(0);
}


			

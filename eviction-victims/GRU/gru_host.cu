/**********************************************************************
* FILENAME :        gru_host.cu            
* 
* DESCRIPTION :
*       Host side implementation of GRU network
*
* NOTES :
*       This file includes CUDA memory allocations and CUDA
*       memory copies to host.
*       Invokes kernel from host.
*       Reads inputs and weight from files
*       Invoking kernel with the state obtained from previous cell
* 
* AUTHOR :    Aajna Karki 
*             https://www.linkedin.com/in/aajna/
*********************************************************************/
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
// includes, project
//#include <cutil.h>

#define INPUT_SIZE 224*224*3
//#define CPU

//#define DEBUG_PRINTF

#ifdef DEBUG_PRINTF
#define DEBUGPRINT(a) printf(a)
#else
#define DEBUGPRINT(a) (void)0
#endif

//#define NUM 10
// includes, kernels
#include "gru_kernel.cu"

void extract_weights(const char *pFileName,float *layer_weights)
{
	char delim[2];
	delim[0] = '\n';
	delim[1] = 0;
	char *token;
	int count = 0;
	char *line = NULL;
	size_t len = 0;
	{
		FILE * pFile1 = fopen (pFileName,"rb");
		if (pFile1 != NULL)
		{
                        printf("%s\n",pFileName);
			printf(" File FOUND %s\n",pFileName);
			{
				while (getline(&line, &len, pFile1) != -1)
				{
					token = strtok(line,delim);
					float temp_num = atof(token);
					layer_weights[count] = temp_num;	
					count++;
                    //printf("%d\n",count); 
				}
			}
			printf("Final Count : %d\n",count);
			fclose(pFile1);
		}
	}

}
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


int main()
{
    int input_dim = 1;
    int num_units = 100;

    int size_wGates = (input_dim + num_units) * ( 2 * num_units) ; 
    int size_bGates =  2 * num_units ; 
    int size_wHidden = (input_dim + num_units) * ( num_units) ; 
    int size_bHidden = num_units; 

    float prodsum = 0.0;
    printf(" Gates,bias, candidate_gates,candidate_bias, o_weights :: %d %d %d %d %d\n",size_wGates,size_bGates,size_wHidden,size_bHidden,num_units);

//Keras
    size_wGates = (input_dim * num_units); 
    size_bGates =  num_units;
    size_wHidden = num_units * num_units;
    size_bHidden = num_units; 

    float kernel_h_w[input_dim * num_units],kernel_r_w[ input_dim * num_units],kernel_z_w[ input_dim * num_units];	
    float bias_h[num_units],bias_z[num_units],bias_r[num_units];	
    float rec_h_w[num_units * num_units],rec_z_w[ num_units * num_units],rec_r_w[ num_units * num_units];
    float dense_kernel[num_units],dense_bias[1];	

	extract_weights("data/kernel_h.txt",kernel_h_w);
	extract_weights("data/kernel_r.txt",kernel_r_w);
	extract_weights("data/kernel_z.txt",kernel_z_w);
	extract_weights("data/bias_h.txt",bias_h);
	extract_weights("data/bias_r.txt",bias_r);
	extract_weights("data/bias_z.txt",bias_z);
	extract_weights("data/recurrent_kernel_h.txt",rec_h_w);
	extract_weights("data/recurrent_kernel_r.txt",rec_r_w);
	extract_weights("data/recurrent_kernel_z.txt",rec_z_w);
	extract_weights("data/dense_kernel.txt",dense_kernel);
	extract_weights("data/dense_bias.txt",dense_bias);

    float CPU_states[num_units];
    float x[input_dim];
	x[0] = 0.98;
	prodsum = 0;
	for(int i=0; i < num_units;i++)
	{
		CPU_states[i] = 0;
	}

    float *GPU_kernel_h_w,*GPU_kernel_r_w,*GPU_kernel_z_w;	
    float *GPU_bias_h,*GPU_bias_z,*GPU_bias_r;	
    float *GPU_rec_h_w,*GPU_rec_z_w,*GPU_rec_r_w;	
    /* Allocate Memory */
	Cuda_allocateMemory(&GPU_kernel_h_w,sizeof(float) * size_wGates);
	Cuda_allocateMemory(&GPU_bias_h,sizeof(float) * size_bGates);
	Cuda_allocateMemory(&GPU_rec_h_w,sizeof(float) * size_wHidden);
	Cuda_allocateMemory(&GPU_kernel_r_w,sizeof(float) * size_wGates);
	Cuda_allocateMemory(&GPU_bias_r,sizeof(float) * size_bGates);
	Cuda_allocateMemory(&GPU_rec_r_w,sizeof(float) * size_wHidden);
	Cuda_allocateMemory(&GPU_kernel_z_w,sizeof(float) * size_wGates);
	Cuda_allocateMemory(&GPU_bias_z,sizeof(float) * size_bGates);
	Cuda_allocateMemory(&GPU_rec_z_w,sizeof(float) * size_wHidden);
    /* Copy weights and bias*/
	Cuda_memcpyfromHostToDevice(GPU_kernel_h_w,kernel_h_w,sizeof(float) * size_wGates);
	Cuda_memcpyfromHostToDevice(GPU_bias_h,bias_h,sizeof(float) * size_bGates);
	Cuda_memcpyfromHostToDevice(GPU_rec_h_w,rec_h_w,sizeof(float) * size_wHidden);
	Cuda_memcpyfromHostToDevice(GPU_kernel_r_w,kernel_r_w,sizeof(float) * size_wGates);
	Cuda_memcpyfromHostToDevice(GPU_bias_r,bias_r,sizeof(float) * size_bGates);
	Cuda_memcpyfromHostToDevice(GPU_rec_r_w,rec_r_w,sizeof(float) * size_wHidden);
	Cuda_memcpyfromHostToDevice(GPU_kernel_z_w,kernel_z_w,sizeof(float) * size_wGates);
	Cuda_memcpyfromHostToDevice(GPU_bias_z,bias_z,sizeof(float) * size_bGates);
	Cuda_memcpyfromHostToDevice(GPU_rec_z_w,rec_z_w,sizeof(float) * size_wHidden);
	
	dim3 numBlocks(1,1,1);
    dim3 numThreads(10, 10);
	

    float *GPU_x;
    int out_loop = 0;
    /* Initialise hidden states */
	Cuda_allocateMemory(&GPU_x,sizeof(float) * input_dim);
	Cuda_memcpyfromHostToDevice(GPU_x,x,sizeof(float) * input_dim);

        
    float *GPU_states;
	Cuda_allocateMemory(&GPU_states,sizeof(float) * num_units);
    printf("Invoke GPU kernel\n");
    /* Initial state is zero */
	GPU_forward_pass_gru<<<numBlocks, numThreads>>>(GPU_x, GPU_states ,GPU_kernel_h_w,GPU_kernel_r_w,GPU_kernel_z_w,GPU_bias_h,GPU_bias_r,GPU_bias_z,GPU_rec_h_w,GPU_rec_r_w,GPU_rec_z_w);
	cudaThreadSynchronize();
	Cuda_memcpyfromDeviceToHost(CPU_states,GPU_states,sizeof(float) * num_units);
    prodsum = 0; 
    for(out_loop = 0; out_loop < num_units; out_loop++)
	{
        prodsum += (CPU_states[out_loop] * dense_kernel[out_loop]);
	}
    printf("PROCUDT (1) %f\n",prodsum + dense_bias[0]);
        
    /* Predict based on previous states */
          
	x[0] = 0.88;

	Cuda_memcpyfromHostToDevice(GPU_x,x,sizeof(float) * input_dim);
	GPU_forward_pass_gru<<<numBlocks, numThreads>>>(GPU_x, GPU_states ,GPU_kernel_h_w,GPU_kernel_r_w,GPU_kernel_z_w,GPU_bias_h,GPU_bias_r,GPU_bias_z,GPU_rec_h_w,GPU_rec_r_w,GPU_rec_z_w);
	cudaThreadSynchronize();
	Cuda_memcpyfromDeviceToHost(CPU_states,GPU_states,sizeof(float) * num_units);
    prodsum = 0; 
    for(out_loop = 0; out_loop < num_units; out_loop++)
	{
        prodsum += (CPU_states[out_loop] * dense_kernel[out_loop]);
	}
	cudaStreamSynchronize(0);
    printf("PROCUDT (2) %f\n",prodsum + dense_bias[0]);

    /* Free Memory */
	//cudaFree(GPU_x);
	//cudaFree(GPU_states);
	//cudaFree(GPU_kernel_h_w);
	//cudaFree(GPU_bias_h);
	//cudaFree(GPU_rec_h_w);
	//cudaFree(GPU_kernel_r_w);
	//cudaFree(GPU_bias_r);
	//cudaFree(GPU_rec_r_w);
	//cudaFree(GPU_kernel_z_w);
	//cudaFree(GPU_bias_z);
	//cudaFree(GPU_rec_z_w);
	//
}

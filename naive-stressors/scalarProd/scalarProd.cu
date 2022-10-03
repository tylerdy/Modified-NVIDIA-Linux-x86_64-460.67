/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

/*
 * This sample calculates scalar products of a
 * given set of input vector pairs
 */



#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>

#include <helper_functions.h>
#include <helper_cuda.h>



///////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on CPU
///////////////////////////////////////////////////////////////////////////////
extern "C"
void scalarProdCPU(
    float *h_C,
    float *h_A,
    float *h_B,
    int vectorN,
    int elementN
);



///////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on GPU
///////////////////////////////////////////////////////////////////////////////
#include "scalarProd_kernel.cuh"



////////////////////////////////////////////////////////////////////////////////
// Helper function, returning uniformly distributed
// random float in [low, high] range
////////////////////////////////////////////////////////////////////////////////
float RandFloat(float low, float high)
{
    float t = (float)rand() / (float)RAND_MAX;
    return (1.0f - t) * low + t * high;
}



///////////////////////////////////////////////////////////////////////////////
// Data configuration
///////////////////////////////////////////////////////////////////////////////

//Total number of input vector pairs; arbitrary
const int VECTOR_N = 256;
//Number of elements per vector; arbitrary,
//but strongly preferred to be a multiple of warp size
//to meet memory coalescing constraints
const int ELEMENT_N = 4096;
//Total number of data elements
const int    DATA_N = VECTOR_N * ELEMENT_N;

const int   DATA_SZ = DATA_N * sizeof(float);
const int RESULT_SZ = VECTOR_N  * sizeof(float);

// const int   DATA_SZ = DATA_N * sizeof(int);
// const int RESULT_SZ = VECTOR_N  * sizeof(int);



///////////////////////////////////////////////////////////////////////////////
// Main program
///////////////////////////////////////////////////////////////////////////////
int main(int argc, char **argv)
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


    float *h_A, *h_B, *h_C_CPU, *h_C_GPU;
    float *d_A, *d_B, *d_C, *saddr;

    // int *h_A, *h_B, *h_C_CPU, *h_C_GPU;
    // int *d_A, *d_B, *d_C;
    unsigned int *d_result,*h_result, *d_trash;
    // double delta, ref, sum_delta, sum_ref, L1norm;
    // StopWatchInterface *hTimer = NULL;

    // printf("%s Starting...\n\n", argv[0]);

    // use command-line specified CUDA device, otherwise use device with highest Gflops/s
    findCudaDevice(argc, (const char **)argv);

    // sdkCreateTimer(&hTimer);

    // printf("Initializing data...\n");
    // printf("...allocating CPU memory.\n");
    h_A     = (float *)malloc(DATA_SZ);
    h_B     = (float *)malloc(DATA_SZ);
    h_C_CPU = (float *)malloc(RESULT_SZ);
    h_C_GPU = (float *)malloc(RESULT_SZ);
    h_result = (unsigned int *)malloc(256*128*sizeof(unsigned int));

    
    //cudaStream_t stream;
    //checkCudaErrors(cudaStreamCreate(&stream));
    
    // printf("...allocating GPU memory.\n");
    checkCudaErrors(cudaMalloc((void **)&saddr, DATA_SZ*2+RESULT_SZ));
    d_A = saddr;
    d_B = (float*)((void *)d_A + DATA_SZ);
    d_C = (float*)((void *)d_B + DATA_SZ);
    // checkCudaErrors(cudaMalloc((void **)&d_A, DATA_SZ));
    // checkCudaErrors(cudaMalloc((void **)&d_B, DATA_SZ));
    // checkCudaErrors(cudaMalloc((void **)&d_C, RESULT_SZ));


    // checkCudaErrors(cudaMalloc((void **)&d_result, 256*128*sizeof(unsigned int)));
    // checkCudaErrors(cudaMalloc((void **)&d_trash, 4*sizeof(unsigned int)));
    // printf("...generating input data in CPU mem.\n");
    srand(123);

    //Generating input data on CPU
    for (i = 0; i < DATA_N; i++)
    {
        h_A[i] = RandFloat(0.0f, 1.0f);
        h_B[i] = RandFloat(0.0f, 1.0f);
    }

    // printf("...copying input data to GPU mem.\n");
    //Copy options data to GPU memory for further processing
    checkCudaErrors(cudaMemcpyAsync(d_A, h_A, DATA_SZ, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpyAsync(d_B, h_B, DATA_SZ, cudaMemcpyHostToDevice));
    // printf("Data init done.\n");


    // printf("Executing GPU kernel...\n");
    // checkCudaErrors(cudaStreamSynchronize(stream));
    // sdkResetTimer(&hTimer);
    // sdkStartTimer(&hTimer);
    for(int j = 0; j <1; j++){
        // printf("%d ", j);
        
    scalarProdGPU<<<80, 256>>>(d_C, d_A, d_B, VECTOR_N, ELEMENT_N, run_time);
    // getLastCudaError("scalarProdGPU() execution failed\n");
    //checkCudaErrors(cudaStreamSynchronize(stream));
    }
    // checkCudaErrors(cudaMemcpyAsync(h_result, d_result, 256*128*sizeof(unsigned int), cudaMemcpyDeviceToHost, stream));
    // checkCudaErrors(cudaStreamSynchronize(stream));
    // unsigned int rsum = 0;
    // for(int i = 0; i < 256*128; i++){
    //     printf("%d\n", h_result[i]);
    //     rsum += h_result[i];
    // }

    // printf("num is %d, rate is %f\n",rsum, rsum / (128*256.0*64));
    // sdkStopTimer(&hTimer);
    // printf("GPU time: %f msecs.\n", sdkGetTimerValue(&hTimer));

    // printf("Reading back GPU result...\n");
    // //Read back GPU results to compare them to CPU results
    // checkCudaErrors(cudaMemcpy(h_C_GPU, d_C, RESULT_SZ, cudaMemcpyDeviceToHost));


    // printf("Checking GPU results...\n");
    // printf("..running CPU scalar product calculation\n"); 
    // scalarProdCPU(h_C_CPU, h_A, h_B, VECTOR_N, ELEMENT_N);

    // printf("...comparing the results\n");
    // //Calculate max absolute difference and L1 distance
    // //between CPU and GPU results
    // sum_delta = 0;
    // sum_ref   = 0;

    // for (i = 0; i < VECTOR_N; i++)
    // {
    //     delta = fabs(h_C_GPU[i] - h_C_CPU[i]);
    //     ref   = h_C_CPU[i];
    //     sum_delta += delta;
    //     sum_ref   += ref;
    // }

    // L1norm = sum_delta / sum_ref;

    // printf("Shutting down...\n");
    // checkCudaErrors(cudaFree(d_C));
    // checkCudaErrors(cudaFree(d_B));
    // checkCudaErrors(cudaFree(d_A));
    //free(h_C_GPU);
    //free(h_C_CPU);
    // free(h_B);
    //free(h_A);
    // sdkDeleteTimer(&hTimer);

    // printf("L1 error: %E\n", L1norm);
    // printf((L1norm < 1e-6) ? "Test passed\n" : "Test failed!\n");
    // exit(L1norm < 1e-6 ? EXIT_SUCCESS : EXIT_FAILURE);
    cudaStreamSynchronize(0);
    //printf("finished (skipped test)\n");
}

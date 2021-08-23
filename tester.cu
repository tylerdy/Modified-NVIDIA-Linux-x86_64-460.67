#define MAX_SPACES 20           // max number of cache-size spaces with pointer chasing
#define NUM_SPACES 1            // number of spaces for this instance
#define NUM_PASSES 1 		// number of read passes over each space
#define MAX_WARP_LOG 16384 
#define TX2_CACHE_LINE 128     // cache line 128 bytes, 32 words
#define TX2_CACHE_SIZE  2097152 // bytes of 1080 cache
#define NUM_BLOCKS  4      // fixed number of blocks
#define NUM_WARPS   1       // fixed number of warps per block
#define SAMPLES 1
#include <stdio.h>
#include <cstdint>

#include <test.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <unistd.h>
#include <sched.h>
#include <errno.h>
#include <sys/syscall.h>
#include "test.hpp"

// CUDA API and helper functions
// For the CUDA runtime routines (prefixed with "cuda")
#include <cuda_runtime.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples
#include <helper_cuda.h>       // helper for checking cuda initialization and error checking
#include <helper_string.h>     // helper functions for string parsing


#include "app_kernel.cuh" 
 
#define min(a,b) ((a) <= (b) ? (a) : (b))
#define max(a,b) ((a) >= (b) ? (a) : (b))

/* to get usage information, invoke program with -h switch */
void Usage(const char *s) 
{
  fprintf (stderr,"\nUsage: %s\n", s);
  fprintf (stderr,"    [-t run_seconds (run time in seconds)] \n"); 
  fprintf (stderr,"\n");
  exit(-1);
}

#define MIN_PAGES 16          // defines range from random number of pages allocated
#define MAX_PAGES 96          // to create free blocks in non-contiguous memory 

struct random_data buf;       // to use the random_r() function
char r_state[64];

/* Function to generate a random number between MIN_- and MAX_PAGES
 */
 

int get_pages(struct random_data *buf) 
{
 int32_t random;
 random_r(buf, &random);
 return (MIN_PAGES + random%((MAX_PAGES - MIN_PAGES) + 1));
}

double elapsed_ns(struct timespec *te, struct timespec *ts)
{
  double elapsed;

  elapsed = (((double)te->tv_sec)*1e9 + (double)te->tv_nsec) -
            (((double)ts->tv_sec)*1e9 + (double)ts->tv_nsec);
  return(elapsed);
}

int main(int argc, char *argv[])
{
  int ret;
  void *virt_start;
  void *phy_start;
  cudaStream_t my_stream;

  // Device memory pointers
  unsigned int *contig_start;
  unsigned int *d_data, *device_p;   //cache-size device space to hold pointer-chasing array
  unsigned int *d_skip;   //device space skipped to create non-continuous areas
  unsigned int **d_ptrs;  //list of device spaces passed to kernel
  
  unsigned int *d_flush;  //cache-size device space for inital cache flush
  unsigned int *d_result;  //device memory array to hold logged values

  unsigned int *h_data;   //cache-size host memory to initialize pointer chasing
  unsigned int *h_ptrs[MAX_SPACES];  //list of allocated device memory spaces
  unsigned int *h_result;  //host memory array to hold logged values
  
    
  // Kernel execution parameters
  dim3 Threads;
  dim3 Blocks;

  // parameters for program
  //default run time
  int run_seconds = 1;

  //number of bytes in TX2 L2 cache
  int bytesize = TX2_CACHE_SIZE;

  //number of words in a cache line = array elements per line
  int line_elements = TX2_CACHE_LINE / sizeof(unsigned int);

  //number of lines in TX2 cache = number of array elements
  int element_count = bytesize / TX2_CACHE_LINE;

  int array_count = TX2_CACHE_SIZE / sizeof(unsigned int);

  //size to hold logged values
  int wrp_log;

  int skip_space1, skip_space2;
  unsigned long long checkAlign;

  unsigned long long run_time;   //time in nanoseconds to run kernel

  int ptr, nextptr;

  int i;
  int log_idx, shared_space, j;   //UNCOMMENT FOR LOGGING 
  
  pid_t my_pid = getpid();  


  // initialize for generating random numbers
  initstate_r((unsigned int)my_pid, r_state, sizeof(r_state), &buf);
  
  cudaSetDevice(0); //only one on TX2
  // ret = device_init(true);
  // if (ret < 0)
  //       fprintf(stderr, "Device init failed\n");
  cudaStreamCreate(&my_stream);
   checkCudaErrors(cudaMalloc((void **) &d_flush, bytesize));  
  // allocate list of device memory spaces 
  checkCudaErrors(cudaMalloc((void **) &d_ptrs, sizeof(h_ptrs))); 
  // contig_start = (unsigned int*)device_allocate_contigous(bytesize, &phy_start);
  // return 0;
  // d_flush = contig_start;
  
  //printf("%016x\n", device_p);
  checkCudaErrors(cudaMalloc((void **) &d_data, bytesize));
  // d_data = (unsigned int *)(((void *)contig_start) + bytesize);
  // printf("%016x\n%016x\n",contig_start, d_data);
  // return 0;
  // d_data = &()
  //printf("%016x\n", d_data);
  //checkCudaErrors(cudaStreamSynchronize(my_stream));
  h_ptrs[0] = d_data; 

  checkCudaErrors(cudaMemcpy(d_ptrs, h_ptrs, sizeof(h_ptrs), cudaMemcpyHostToDevice));
  
  //allocate device memory to hold log copied from shared memory
  checkCudaErrors(cudaMalloc((void **) &d_result, bytesize));
  
  // allocate host memory to hold logs copied from device memory
  checkCudaErrors(cudaMallocHost(&h_result, bytesize));

  // allocate host memory to create pointer-chasing list (copied to device memory)
  checkCudaErrors(cudaMallocHost(&h_data, bytesize));
  int wrpcnts = NUM_BLOCKS * NUM_WARPS;
  // int stride = array_count / wrpcnts / mult;
  // // return 0;
  // ptr = 0;
  //  for (int i = 0; i < element_count; i++) {
  //   //  index values separated by number of elements per line (32)
  //   // printf("%d\n", i);
  //    nextptr = i * line_elements;
  //    h_data[ptr] = nextptr;
  //   //  h_data[ptr] = 0;
  //   //  printf("[%d] = %d\n", ptr, nextptr);
  //    ptr = nextptr;
  //  }
  // //  return 0;
  //  h_data[nextptr] = 0;  //last points to first
  for(i = 0; i < NUM_BLOCKS; i++){
    h_data[i*32] = i << 1;
  }
  Threads = dim3(32, 1, 1);
  Blocks = dim3(NUM_BLOCKS, 1, 1);
  
  for (i = 0; i < NUM_SPACES; i++) {
     checkCudaErrors(cudaMemcpyAsync(h_ptrs[i],  h_data, bytesize, cudaMemcpyHostToDevice, my_stream)); 
     checkCudaErrors(cudaStreamSynchronize(my_stream));
  }
  
  flushKernel<<<Blocks, Threads, 0, my_stream>>>(d_result, bytesize, d_flush, 0);  
  checkCudaErrors(cudaStreamSynchronize(my_stream));
  
  testerKernel<<<Blocks, Threads, 0, my_stream>>>(d_data, d_result, 0,2048);
  checkCudaErrors(cudaStreamSynchronize(my_stream));
  checkCudaErrors(cudaMemcpyAsync(h_result, d_result, sizeof(unsigned int) * NUM_BLOCKS, cudaMemcpyDeviceToHost, my_stream));
  checkCudaErrors(cudaStreamSynchronize(my_stream));

  for(i = 0; i < NUM_BLOCKS; i++){
    printf("%d\n", h_result[i]);
  }
  checkCudaErrors(cudaMemcpyAsync(h_data, h_ptrs[0], bytesize, cudaMemcpyDeviceToHost, my_stream));
  checkCudaErrors(cudaStreamSynchronize(my_stream));
}
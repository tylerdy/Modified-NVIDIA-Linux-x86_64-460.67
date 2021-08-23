#define MAX_SPACES 20           // max number of cache-size spaces with pointer chasing
#define NUM_SPACES 1            // number of spaces for this instance
#define NUM_PASSES 1 		// number of read passes over each space
#define MAX_WARP_LOG 16384 
#define TX2_CACHE_LINE 128     // cache line 128 bytes, 32 words
#define TX2_CACHE_SIZE  2097152 // bytes of 1080 cache
#define NUM_BLOCKS  4      // fixed number of blocks
#define NUM_WARPS   2       // fixed number of warps per block
#define SAMPLES 1
#include <stdio.h>
#include <cstdint>
#include<cuda_profiler_api.h>
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
  unsigned int *d_data, *device_p;   //cache-size device space to hold pointer-chasing array
  unsigned int *d_skip;   //device space skipped to create non-continuous areas
  unsigned int **d_ptrs;  //list of device spaces passed to kernel
  
  unsigned int *d_flush, *d_flush2, *h_flush;  //cache-size device space for inital cache flush
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

// Parse the command line 
// only parameter is -t for run time in seconds
  i = 1;
  while (i < argc) {
    if (strcmp (argv[i], "-t") == 0) {
      if (++i >= argc) Usage (argv[0]);
      run_seconds = atoi(argv[i]);
    }
    else 
      Usage (argv[0]);
    i++;
  }

  // initialize for generating random numbers
  initstate_r((unsigned int)my_pid, r_state, sizeof(r_state), &buf);
  
  cudaSetDevice(0); //only one on TX2
  // ret = device_init(true);
  // if (ret < 0)
        // fprintf(stderr, "Device init failed\n");
  cudaStreamCreate(&my_stream);
  // unsigned int *contig_start = (unsigned int*)device_allocate_contigous(bytesize*4, &phy_start);
  // allocate list of device memory spaces 
   checkCudaErrors(cudaMalloc((void **) &d_flush, bytesize));  
   checkCudaErrors(cudaMallocHost((void **) &h_flush, bytesize));  
  //  checkCudaErrors(cudaMalloc((void **) &d_flush2, bytesize));  
  // d_flush = contig_start;
  // d_flush2 = contig_start + (bytesize >> 2);
  checkCudaErrors(cudaMalloc((void **) &d_ptrs, sizeof(h_ptrs))); 
  // d_data = (unsigned int*)device_allocate_contigous(bytesize, &phy_start);
  // printf("%016x\n", device_p);
  // d_data = contig_start + (bytesize >> 2)*4;// + (bytesize >> 3);
  checkCudaErrors(cudaMalloc((void **) &d_data, bytesize*4));
  // d_data = &(d_flush[bytesize >> 2]);
  //d_data = d_data + (bytesize >> 2);
  // d_data = &()
  printf("%016x\n%016x\n%016x\n", d_flush,d_flush2, d_data);
  //checkCudaErrors(cudaStreamSynchronize(my_stream));

  h_ptrs[0] = d_data; 
  

  // fprintf(stdout, "virt %p phys %p\n", h_ptrs[0], phy_start);
  // for (i = 0; i < NUM_SPACES; i++) {
  //    //make random size holes in device memory to encourage non-contiguous allocations
  //    skip_space1 = 4096 * get_pages(&buf);  //hole will be this size
  //    skip_space2 = 4096 * get_pages(&buf);  //next unavailable pages         
  //    checkCudaErrors(cudaMalloc((void **) &d_skip, skip_space1));
  //    checkCudaErrors(cudaMalloc((void **) &d_data, skip_space2));
  //    checkCudaErrors(cudaFree(d_skip));   //hole goes here
     
  //    //allocate device memory space equal to L2 cache size
  //    //aligned on page boundary 
  //    checkCudaErrors(cudaMalloc((void **) &d_data, bytesize));
  //    checkAlign = (unsigned long long)d_data & 0x000000000000007f;
  //    if (checkAlign != (unsigned long long)0) {
  //       printf("Failed Aligned Page, Size %d Ptr %p Check %llu\n", bytesize, d_data, checkAlign);
  //       exit(-1);
  //    }
  //    h_ptrs[i] = d_data;  //save pointer to allocated device space in host list
  // }

  checkCudaErrors(cudaMemcpy(d_ptrs, h_ptrs, sizeof(h_ptrs), cudaMemcpyHostToDevice));
  

  // space needed to log times for reading each element in each device space
  wrp_log = NUM_SPACES * element_count * sizeof(unsigned long long);

  //allocate device memory to hold log copied from shared memory
  checkCudaErrors(cudaMalloc((void **) &d_result, bytesize));
  
  // allocate host memory to hold logs copied from device memory
  checkCudaErrors(cudaMallocHost(&h_result, bytesize));

  // allocate host memory to create pointer-chasing list (copied to device memory)
  checkCudaErrors(cudaMallocHost(&h_data, bytesize));
  int mult = 2;
  int wrpcnts = NUM_BLOCKS * NUM_WARPS;
  int stride = array_count / wrpcnts / mult;
  
  // for(i = 0; i < array_count; i++){
  //   h_data[i] = 0;
  // }
  // for (i = 0; i < wrpcnts * mult; i++){
  //   h_data[i * stride] = i * stride;
    // printf("%d\n", i*stride);
  // }


  //  ptr = 0;
  //  for (int i = 0; i < array_count; i++) {
  //    h_data[i] = i+1;
  //  }
  //  h_data[array_count - 1] = 0;  //last points to first
  
  //  for (int i = 0; i < 16384; i++) {
    //  printf("%d\n", h_data[i]);
  //  }
  //  return 0;


  ptr = 0;
   for (int i = 0; i < element_count; i++) {
    //  index values separated by number of elements per line (32)
     nextptr = i * line_elements;
     h_flush[ptr] = nextptr;
     h_data[ptr] = nextptr;
    //  h_data[ptr] = 0;
    //  printf("[%d] = %d\n", ptr, nextptr);
     ptr = nextptr;
   }
   h_data[nextptr] = 0;  //last points to first
// return 0;
  Threads = dim3(32, NUM_WARPS, 1);
  Blocks = dim3(NUM_BLOCKS, 1, 1);
  checkCudaErrors(cudaMemcpyAsync(d_flush,  h_flush, bytesize, cudaMemcpyHostToDevice, my_stream)); 
  checkCudaErrors(cudaStreamSynchronize(my_stream));
  // copy pointer-chasing array in host memory to device memory spaces
  for (i = 0; i < NUM_SPACES; i++) {
     checkCudaErrors(cudaMemcpyAsync(h_ptrs[i],  h_data, bytesize, cudaMemcpyHostToDevice, my_stream)); 
     checkCudaErrors(cudaStreamSynchronize(my_stream));
  }
  
  // run_time = run_seconds * 1000000000ULL;  //seconds to nanoseconds
  // shared_space = MAX_WARP_LOG * sizeof(unsigned short) + (1<<10); //32KB per block/SM

  //memoryKernel<<<Blocks, Threads, shared_space, my_stream>>>(d_ptrs, d_result, bytesize, run_time, 0, d_flush);  
  // int   num_visits = (bytesize / sizeof(unsigned int)) / (NUM_BLOCKS * NUM_WARPS);
  int   num_visits = (bytesize / 128) / (NUM_BLOCKS * NUM_WARPS);
  
  cudaProfilerStart();
  // for(i = 0; i < 8; i++){
  // flushKernel<<<Blocks, Threads, 0, my_stream>>>(d_result, bytesize, d_flush,0);  
  // checkCudaErrors(cudaStreamSynchronize(my_stream));
  // }

  
  // flushKernel<<<Blocks, Threads, 0, my_stream>>>(d_result, bytesize, d_flush2, 0);  
  // checkCudaErrors(cudaStreamSynchronize(my_stream));
  
  
  for(int samp = 0; samp < SAMPLES; samp++){
    appMemoryKernel<<<Blocks, Threads, 0, my_stream>>>(d_data, d_result, bytesize, 2, 0, d_flush, num_visits);  
    checkCudaErrors(cudaStreamSynchronize(my_stream));
  }
   cudaProfilerStop();
  // testSM<<<Blocks, Threads, 0, my_stream>>>(d_result, 0);
  // checkCudaErrors(cudaStreamSynchronize(my_stream));
  // return 0;
  // copy any logged data back to host memory
  checkCudaErrors(cudaMemcpyAsync(h_result, d_result, wrp_log, cudaMemcpyDeviceToHost, my_stream));
  checkCudaErrors(cudaStreamSynchronize(my_stream));
  // // for(i = 0; i < NUM_BLOCKS; i++){
  // //   printf("%d\n", h_result[i]);
  // // }

  // // copy any side information stored in device space zero back to host memory
  checkCudaErrors(cudaMemcpyAsync(h_data, h_ptrs[0], bytesize, cudaMemcpyDeviceToHost, my_stream));
  checkCudaErrors(cudaStreamSynchronize(my_stream));

  // printf("elapsed time: %d\n", h_result[3]);
  // int min =  10000;
  int cnt  = 0;
  int total = 0;
  // for (i = 0; i < NUM_SPACES; i++) {
  //   for (j = 0; j < 16384; j++) {
  //     int tmp = h_result[j];
  //     // log_idx = (i * element_count);
  //     printf("%d\n", tmp); 
  //     //if(min > tmp && tmp >0) min = tmp;
      
  //     if(tmp != 0){
  //       total++;
  //       if(tmp < 350) cnt++;
  //     }
  //   }	  
  // // // }
  // printf("count is %d, total is %d, %f is the rate\n",cnt,total, cnt / (total + 0.0));
  // printf("%d out of %d\n", cnt, element_count);
  // for(i = 0; i < 8; i++)
  //   printf("%d\n", h_data[i+4]);
   // cudaDeviceReset();
}
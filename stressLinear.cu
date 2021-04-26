/* GPU L2 cache-eviction program using sequential pointer-chasing.  
 * The program launches a kernel that reads 32-bit unsigned integers
 * from an array in device memory that is equal in size to the GPU cache
 * (in this case the 512 KB L2 cache of the TX2).  This array contains
 * 131,072 elements of 4 bytes each aligned on word boundaries.
 * The TX2 cache line size is 128 bytes (32 integers) so a read of the
 * array element that maps to the first 4 bytes of a cache line is
 * sufficient to fetch the entire line into the cache on a cache miss.
 * In this case, a line containing the "victim's" data will be evicted
 * from the cache.
 *
 * Before launching the kernel, a device-memory space of 512 KB is
 * allocated and initialized for pointer-chasing execution.
 * Each array element that is read holds the array index of the next
 * element to read.  The element that maps to the first 4 bytes of a
 * cache line will be at array indices of 0, 32, 64, 96, ..., 131,040.
 * These array elements are initialized with index values needed to
 * fetch all lines in the cache (4096).
 *
 * This design implicitly assumes that contiguous array elements in
 * device memory will map to unique lines in the L2 cache and
 * reads at the indexes in the pointer-chasing list will reference
 * all lines of the L2 cache.  Unfortunately this assumption does
 * not hold when contiguous device memory maps to non-contiguous
 * physical memory or a hash function is used to map physical memory
 * to cache sets.  To illustrate these effects, this program
 * allocates 10 spaces of 512 KB in device memory, initializes them
 * for pointer chasing, and passes a list of spaces to the kernel.
 * The kernel cycles continuously through the list of spaces
 * executing the pointer-chasing list in each.  For use in real
 * cache-eviction experiments, the 10 spaces should be allocated
 * by attempting to allocate contiguous spaces, not using
 * this program's attempt to explicitly create non-contiguous
 * allocations (see comments below for the allocation code).
 *
 * A goal of the cache-eviction program is to rapidly and
 * continuously evict all the vicim's data from the cache.
 * The GPU kernel simply loops reading an element of the array and
 * using the value read as the index of the element read on the next
 * loop iteration. To speed the process of evicting cache lines, the
 * kernel uses the thread parallelism provided by the GPU. The TX2
 * scheduler dispatches sets of 32 threads, called warps, and all threads
 * execute the same instruction concurrently. When a warp is executing
 * the pointer-chasing loop, all 32 threads will issue a load instruction
 * concurrently to read the same next array element. On a cache miss, the
 * warp will stall while waiting for the data to be fetched from DRAM.
 *
 * Warp stalls can be overlapped with warp executions when each warp
 * references a different section of the device-memory array.  The kernel
 * is organized as two blocks of 128 threads (4 warps) each.  Each of the
 * 8 warps performs pointer-chasing in a unique partition of the array
 * having one-eigth of the elements.  The sequential pointer list for
 * each partition contains only the set of index values for that
 * partition.
 *
 * The CUDA main program and kernel are written so the latency for 
 * reading each element can be recorded once for each device-memory
 * space.  The recorded values are initially stored in shared memory
 * allocated for the blocks and copied to device memory only when
 * all reads are finished.  Shared memory is much faster than device
 * memory.  Using shared memory also avoids L2 cache writes interspersed
 * with the pointer-chasing reads.
 *
 * An example of the intended invocation of this program to run for
 * 30 seconds and output recorded times is:
 *
 * ./stressLinear -t 30 >log
 *  (Note: -t 0 is valid and can be used to record data only for the
 *   first pass for each space to get cold cache timings.)
 *
 * Written by Don Smith, Department of Computer Science,
 * University of North Carolina at Chapel Hill
 * 2020
 */

// system service include files
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

#define MAX_SPACES 20           // max number of cache-size spaces with pointer chasing
#define NUM_SPACES 1            // number of spaces for this instance
#define NUM_PASSES 2		// number of read passes over each space
#define MAX_WARP_LOG 16384 
#define TX2_CACHE_LINE 128     // cache line 128 bytes, 32 words
#define TX2_CACHE_SIZE  2097152 // bytes of 1080 cache
#define NUM_BLOCKS  2        // fixed number of blocks
#define NUM_WARPS   4         // fixed number of warps per block
#include "stress_kernel.hpp" 
 
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


////////////////////////////////////////////////////////////////////////////////
// Program main
// Handle all requirements for pre- and post-processing for kernel
////////////////////////////////////////////////////////////////////////////////
int
main(int argc, char **argv)
{
  // Stream for the GPU Operations
  cudaStream_t my_stream;

  // Device memory pointers
  unsigned int *d_data;   //cache-size device space to hold pointer-chasing array
  unsigned int *d_skip;   //device space skipped to create non-continuous areas
  unsigned int **d_ptrs;  //list of device spaces passed to kernel
  
  unsigned int *d_flush;  //cache-size device space for inital cache flush
  unsigned short *d_result;  //device memory array to hold logged values

  unsigned int *h_data;   //cache-size host memory to initialize pointer chasing
  unsigned int *h_ptrs[MAX_SPACES];  //list of allocated device memory spaces
  unsigned short *h_result;  //host memory array to hold logged values
  
    
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
  int ret;
  ret = device_init(true);
  if (ret < 0)
        fprintf(stderr, "Device init failed\n");
      
  // force initialization of GPU and CUDA runtime
  cudaFree(0);
  // create a user-defined stream
  cudaStreamCreate(&my_stream);

  // allocate list of device memory spaces 
  checkCudaErrors(cudaMalloc((void **) &d_ptrs, sizeof(h_ptrs))); 

/* Allocate multiple device memory spaces.  Before requesting a real allocation,
 * two random size blocks of pages are allocated in device space.  The first is
 * made available by freeing it, and it is followed by the second which remains
 * unavailable.  The subsequent request for device space may (or may not) be
 * partially satisfied with pages from the "hole" created in device space or
 * completely satisfied with contiguous pages available in device space.
 *
 * WARNING: this code was written to try creating the effects of non-contiguous
 * allocations of device space for pointer-chasing. This should be changed to
 * simply allocate from available device space when running cache-evictions
 * for benchmarks to take advantage of contiguous allocation when available
 */

  for (i = 0; i < NUM_SPACES; i++) {
     //make random size holes in device memory to encourage non-contiguous allocations
     skip_space1 = 4096 * get_pages(&buf);  //hole will be this size
     skip_space2 = 4096 * get_pages(&buf);  //next unavailable pages         
     checkCudaErrors(cudaMalloc((void **) &d_skip, skip_space1));
     checkCudaErrors(cudaMalloc((void **) &d_data, skip_space2));
     checkCudaErrors(cudaFree(d_skip));   //hole goes here
     
     //allocate device memory space equal to L2 cache size
     //aligned on page boundary 
     checkCudaErrors(cudaMalloc((void **) &d_data, bytesize));
     checkAlign = (unsigned long long)d_data & 0x000000000000007f;
     if (checkAlign != (unsigned long long)0) {
        printf("Failed Aligned Page, Size %d Ptr %p Check %llu\n", bytesize, d_data, checkAlign);
        exit(-1);
     }
     h_ptrs[i] = d_data;  //save pointer to allocated device space in host list
  }
  // copy host list to device list passed to kernel
  checkCudaErrors(cudaMemcpy(d_ptrs, h_ptrs, sizeof(h_ptrs), cudaMemcpyHostToDevice));
  
  // allocate another 512 KB space for initial cache flush by kernel
  checkCudaErrors(cudaMalloc((void **) &d_flush, bytesize));  

  // space needed to log times for reading each element in each device space
  wrp_log = NUM_SPACES * element_count * sizeof(unsigned short);

  //allocate device memory to hold log copied from shared memory
  checkCudaErrors(cudaMalloc((void **) &d_result, wrp_log));
  
  // allocate host memory to hold logs copied from device memory
  checkCudaErrors(cudaMallocHost(&h_result, wrp_log));

  // allocate host memory to create pointer-chasing list (copied to device memory)
  checkCudaErrors(cudaMallocHost(&h_data, bytesize));

/* Store the sequential index values in the host memory array that will
 * be copied to the device memory arrays.  Array index values will be
 * separated by the number of 32-bit integers in a cache line (32) and
 * are stored at array locations 0, 32, 64, 96, ...., 131040. 
 * Each stored index value gives the array index for the next location
 * to be stored (0->32->64->96,......).
 */
 
   ptr = 0;
   for (int i = 0; i < element_count; i++) {
     // index values separated by number of elements per line (32)
     nextptr = i * line_elements;
     h_data[ptr] = nextptr;
     ptr = nextptr;
   }
   h_data[ptr] = 0;  //last points to first

#ifdef CHECK_CHASE
  element_count = bytesize / sizeof(unsigned int);
  for (i = 0; i < element_count; i++)
     printf("hdata[%d] = %u\n", i, h_data[i]); 
     
#endif

  Threads = dim3(32, NUM_WARPS, 1);
  Blocks = dim3(NUM_BLOCKS, 1, 1);

  // copy pointer-chasing array in host memory to device memory spaces
  for (i = 0; i < NUM_SPACES; i++) {
     checkCudaErrors(cudaMemcpyAsync(h_ptrs[i],  h_data, bytesize, cudaMemcpyHostToDevice, my_stream)); 
     checkCudaErrors(cudaStreamSynchronize(my_stream));
  }

  run_time = run_seconds * 1000000000ULL;  //seconds to nanoseconds
/*
 * For access time logging uncomment the following and comment the kernel
 * launch below that does not allocate shared memory
 */
  shared_space = MAX_WARP_LOG * sizeof(unsigned short); //32KB per block/SM
  memoryKernel<<<Blocks, Threads, shared_space, my_stream>>>(d_ptrs, d_result, bytesize, run_time, 0, d_flush);  

  //memoryKernel<<<Blocks, Threads, 0, my_stream>>>(d_ptrs, d_result, bytesize, run_time, 0, d_flush);
  checkCudaErrors(cudaStreamSynchronize(my_stream));

  // copy any logged data back to host memory
  checkCudaErrors(cudaMemcpyAsync(h_result, d_result, wrp_log, cudaMemcpyDeviceToHost, my_stream));
  checkCudaErrors(cudaStreamSynchronize(my_stream));

  // copy any side information stored in device space zero back to host memory
  checkCudaErrors(cudaMemcpyAsync(h_data, h_ptrs[0], bytesize, cudaMemcpyDeviceToHost, my_stream));
  checkCudaErrors(cudaStreamSynchronize(my_stream));
/*
 * For access time logging, comment the ifdef/endif statements for printing
 * logged data on stdout (best to redirect stdout to a file)
 */
 
//#ifdef DO_LOG
  int cnt  =0;
  for (i = 0; i < NUM_SPACES; i++) {
    for (j = 0; j < element_count; j++) {
      log_idx = (i * element_count);
      printf("%hu\n", h_result[log_idx + j]); 
      if(h_result[log_idx+j] < 300)
          cnt++;
      
      // printf("%hu\n", h_result[log_idx + j]); 
    }	
  }
  printf("%d out of %d\n", cnt, element_count);
//#endif

  // cudaDeviceReset causes the driver to clean up all state. While
  // not mandatory in normal operation, it is good practice.  It is also
  // needed to ensure correct operation when the application is being
  // profiled. Calling cudaDeviceReset causes all profile data to be
  // flushed before the application exits
  cudaDeviceReset();
}


#define MAX_SPACES 20           // max number of cache-size spaces with pointer chasing
#define NUM_SPACES 1            // number of spaces for this instance
#define NUM_PASSES 2 		// number of read passes over each space
#define MAX_WARP_LOG 16384 
#define TX2_CACHE_LINE 128     // cache line 128 bytes, 32 words
#define TX2_CACHE_SIZE  2097152 // bytes of 1080 cache
//#define NUM_SMS 16//8 // does this have to be multiple of 4?. without any changes, for thd level accesses this must be at most 8
//#define NUM_BLOCKS NUM_SMS//+20    // fixed number of blocks 

#include <stdio.h>
#include <cstdint>
#include<chrono>
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
#include <cuda_fp16.h> 

// CUDA API and helper functions
// For the CUDA runtime routines (prefixed with "cuda")
#include <cuda_runtime.h>
#include <helper_functions.h>  // helper for shared that are common to CUDA Samples
#include <helper_cuda.h>       // helper for checking cuda initialization and error checking
#include <helper_string.h>     // helper functions for string parsing
#include "stress_kernel.cuh" 


 
#define min(a,b) ((a) <= (b) ? (a) : (b))
#define max(a,b) ((a) >= (b) ? (a) : (b))

/* to get usage information, invoke program with -h switch */
void Usage(const char *s) 
{
  fprintf (stderr,"\nUsage(outdated): %s\n", s);
  fprintf (stderr,"    [-t run_seconds (run time in seconds) -b num_mem_blocks (blocks to iterate on) -s num_mem_sms (sms to run eviction kernel on) -w warp_accesses (every thread in a warp accesses same array element) -c baseline_only] \nnum_blocks must equal num_mem_sms or 2*num_mem_sms. num_mem_blocks must be a power of 2 (for now, TBD) and must be at most 16 unless warp_accesses mode is specified."); 
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
  cudaStream_t mem_stream,mem_stream2,comp_stream;

  // Device memory pointers
  unsigned int *d_data, *device_p;   //cache-size device space to hold pointer-chasing array
  unsigned int *d_skip;   //device space skipped to create non-continuous areas
  unsigned int **d_ptrs, **d_ptrs2;  //list of device spaces passed to kernel
  
  unsigned int *d_flush;  //cache-size device space for inital cache flush
  int *d_result, *d_result2;  //device memory array to hold logged values
  unsigned long long *d_result_time;  //device memory array to hold logged values

  unsigned int *h_data;   //cache-size host memory to initialize pointer chasing
  unsigned int *h_ptrs[MAX_SPACES];  //list of allocated device memory spaces
  int *h_result;  //host memory array to hold logged values
  unsigned long long *h_result_time;  //host memory array to hold logged values
  
    
  // Kernel execution parameters
  dim3 Threads;
  dim3 Blocks;

  // parameters for program
  //default run time
  int run_seconds = 10;

  //number of bytes in TX2 L2 cache
  int bytesize = TX2_CACHE_SIZE;


  //size to hold logged values
  int wrp_log;

  int skip_space1, skip_space2;
  unsigned long long checkAlign;

  unsigned long long run_time;   //time in nanoseconds to run kernel

  int ptr, nextptr;

  int i;
  int log_idx, shared_space, j;   //UNCOMMENT FOR LOGGING 
  
  pid_t my_pid = getpid();  

  bool identical_thread_accesses = false; // if true, each thread in warp access the same element, if false, each accesses a different element
  bool baseline = false; // if true, kernel will be baseline (compute for memory accesses, suspended for compute accesses)
  int num_mem_sms = 0; // number of SMs to limit memory eviction blocks to
  int num_comp_warps = 0; // number of warps per SM the victim will be running on which will be reserved for compute (or L1) interference
  bool l1 = false;

  int comp_instr = 1;
  int stride_bytes = TX2_CACHE_LINE;  

  int rw_mode = 3;

  bool smk_hybrid = false;
  int smk_comp_warps = 0;

  // numbers of enemy warps (out of 32 per SM) to devote to each type of instruction for mixed experiments
  int mem_r_warps = -1;
  int mem_w_warps = -1;
  int comp_fp_warps = -1;
  int comp_int_warps = -1;
 
// Parse the command line 
  i = 1;
  while (i < argc) {
    if (strcmp (argv[i], "-t") == 0) { // run time in seconds
      if (++i >= argc) Usage (argv[0]);
      run_seconds = atoi(argv[i]);
    } else if (strcmp (argv[i], "-w") == 0) { // identical accesses by threads in the same warp?
      identical_thread_accesses = true;
    } else if (strcmp (argv[i], "-m") == 0) { // sms to dedicate to L2 cache interference
      if (++i >= argc) Usage (argv[0]);
      num_mem_sms = atoi(argv[i]);
    } else if (strcmp (argv[i], "-l") == 0) { // memory interference, but on L1 cache (pass -c not -m)
      l1 = true; // automatically sets num_mem_blocks to 20 and (NOT ANYMORE) [identical_thread_accesses to true]
      // rw_mode = 2; // different default for l1
    } else if (strcmp (argv[i], "-b") == 0) { // baseline?
      baseline = true;
    } else if (strcmp (argv[i], "-c") == 0) { // warps per victim sm to dedicate to compute (or L1 cache) interference
      if (++i >= argc) Usage (argv[0]);
      num_comp_warps = atoi(argv[i]);
    } else if (strcmp (argv[i], "-i") == 0) { // instruction choice for compute
      if (++i >= argc) Usage (argv[0]);
      /*
        FP INT
        0  6   16 bit ADD
        1  7   16 bit MULT
        2  8   32 bit ADD
        3  9   32 bit MULT
        4  10  64 bit ADD
        5  11  64 bit MULT

        12     32 bit bitwise AND
        13     32 bit INT bit reverse
        14     32 bit FP reciprocal
        15     count of leading zeros
        16     pop count

        */
      comp_instr = atoi(argv[i]);
    } else if (strcmp (argv[i], "-a") == 0) { // array size for memory interference 
      if (++i >= argc) Usage (argv[0]);
      bytesize = atoi(argv[i]);
    } else if (strcmp (argv[i], "-s") == 0) { // stride size for memory interference
      if (++i >= argc) Usage (argv[0]);
      stride_bytes = atoi(argv[i]);
    } else if (strcmp (argv[i], "-r") == 0) { // memory instruction type (R/W)
      if (++i >= argc) Usage (argv[0]);
      rw_mode = atoi(argv[i]);
      /*
      0 = R
      1 = W
      2 = R,W
      3 = W,R
      */
    } else if (strcmp (argv[i], "--smk-hybrid-comp-warps") == 0) { // stride size for memory interference
      if (++i >= argc) Usage (argv[0]);
      smk_hybrid = true;
      smk_comp_warps = atoi(argv[i]);
      if(smk_comp_warps%16!=0 || smk_comp_warps>32) Usage (argv[0]);
    } else if (strcmp (argv[i], "--mem-r") == 0) { 
      if (++i >= argc) Usage (argv[0]);
      mem_r_warps = atoi(argv[i]);
    } else if (strcmp (argv[i], "--mem-w") == 0) { 
      if (++i >= argc) Usage (argv[0]);
      mem_w_warps = atoi(argv[i]);
    } else if (strcmp (argv[i], "--comp-fp") == 0) { 
      if (++i >= argc) Usage (argv[0]);
      comp_fp_warps = atoi(argv[i]);
    } else if (strcmp (argv[i], "--comp-int") == 0) {
      if (++i >= argc) Usage (argv[0]);
      comp_int_warps = atoi(argv[i]);
    } 
    else 
      Usage (argv[0]);
    i++;
  } 

  int num_mem_blocks = num_mem_sms*2; // num blocks to execute and not kill (not necessarily the number to launch)
  
  //number of words in a cache line = array elements per line
  int line_elements = stride_bytes / sizeof(unsigned int);

  //number of lines in TX2 cache = number of array elements
  int element_count = bytesize / stride_bytes;


  if(l1) {
    /*
    for L1 experiments should use same number of warps across all experiments (so that we leave same number for victim)
    must use all threads in warp same access mode

    array size pilot (128 byte stride, 32 warps): 
    16KB: 4 accesses/warp
    32KB: 8
    48KB: 12
    64KB: 16
    80KB: 20
    96KB: 24
    112KB: 28
    128KB: 32

    stride length pilot (48KB array size, 32 warps):
    8B: 192 accesses/warp
    16B: 96
    32B: 48
    64B: 24
    128B: 12
    256B: 6
    512B: 3

    */

    if(num_mem_sms>0) Usage(argv[0]); // hybrid with L2 and L1 cache interference not yet supported since L1 interference uses the L2 setup
    
    num_mem_blocks = 20;
    // identical_thread_accesses = true;    

    if(!smk_hybrid && (num_comp_warps<1 || num_comp_warps>32)) Usage(argv[0]);
  } else {
    /* L2
    array size pilot (128 byte stride):
    1MB: 4 SMs
    2MB: 8 SMs
    3MB: 12 SMs
    4MB: 16 SMs
    5MB
    6MB
    7MB
    8MB
    use 4 SMs

    stride length pilot (2MB array size):
    8 bytes stride
    16 bytes stride
    32 bytes stride: 32 SMs
    64 bytes stride: 16 SMs
    128 bytes stride: 8 SMs
    256 bytes stride: 4 SMs
    use 4 SMs

    */
    if(num_mem_sms!=num_mem_blocks && 2*num_mem_sms!=num_mem_blocks) Usage(argv[0]);
    if((num_mem_sms & (num_mem_sms - 1)) != 0 || (!identical_thread_accesses && num_mem_sms>16)) Usage(argv[0]);
    if(2*num_mem_sms!=num_mem_blocks && num_comp_warps>0) Usage(argv[0]); // only allow victim SM compute interference when SMs are either compute or victim
    if(num_comp_warps%16!=0) Usage(argv[0]); // only allow multiples of 16 warps per SM for victim SM compute interference
    if (num_mem_sms>16) Usage(argv[0]);
    //if(num_comp_warps==0 && comp_instr>=0  ) Usage(argv[0]);
    if(num_comp_warps>0 && comp_instr<0  ) Usage(argv[0]);
    int total_warps = 32*num_mem_blocks;
    float wrp_max, thd_max, access_max;
    wrp_max = (float (bytesize / stride_bytes)) / (total_warps); // if block size is 32 and 2 warps per SM, wrp_max=num cache lines per warp we're executing
    thd_max = (float (bytesize / stride_bytes)) / (total_warps*32);
    if((identical_thread_accesses && wrp_max<1) || (!identical_thread_accesses && thd_max<1)) Usage(argv[0]);
  }    

  if((mem_r_warps>-1 || mem_w_warps>-1 || comp_fp_warps>-1 || comp_int_warps>-1) && (mem_r_warps+mem_w_warps+comp_fp_warps+comp_int_warps!=32 || !smk_hybrid))  Usage(argv[0]);
  if(((mem_r_warps>0 && (comp_fp_warps>0 || comp_int_warps>0)) || (mem_w_warps>0 && (comp_fp_warps>0 || comp_int_warps>0))) && (mem_r_warps+mem_w_warps!=16 || comp_fp_warps+comp_int_warps!=16)) Usage(argv[0]);
  
  


  // initialize for generating random numbers
  initstate_r((unsigned int)my_pid, r_state, sizeof(r_state), &buf);
  
  cudaSetDevice(0); //only one on TX2
   ret = device_init(true);
   if (ret < 0)
         fprintf(stderr, "Device init failed\n");
  cudaStreamCreate(&mem_stream);
  cudaStreamCreate(&mem_stream2);
  cudaStreamCreate(&comp_stream);
  // allocate list of device memory spaces 
  checkCudaErrors(cudaMalloc((void **) &d_ptrs, sizeof(h_ptrs)));
  checkCudaErrors(cudaMalloc((void **) &d_ptrs2, sizeof(h_ptrs)));
   d_data = (unsigned int*)device_allocate_contigous(CONTIG_SIZE, &phy_start);
  //printf("%016x\n", device_p);
  // checkCudaErrors(cudaMalloc((void **) &d_data, bytesize));
  //printf("%016x\n", d_data);
  //checkCudaErrors(cudaStreamSynchronize(mem_stream));
 
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
  checkCudaErrors(cudaMemcpy(d_ptrs2, h_ptrs, sizeof(h_ptrs), cudaMemcpyHostToDevice));
  
  // allocate another 512 KB space for initial cache flush by kernel
  checkCudaErrors(cudaMalloc((void **) &d_flush, bytesize));  

  // space needed to log times for reading each element in each device space
  wrp_log = NUM_SPACES * element_count * sizeof(unsigned long);

  //allocate device memory to hold log copied from shared memory
  checkCudaErrors(cudaMalloc((void **) &d_result, wrp_log));
  checkCudaErrors(cudaMalloc((void **) &d_result2, wrp_log));
  checkCudaErrors(cudaMalloc((void **) &d_result_time, wrp_log));
  
  // allocate host memory to hold logs copied from device memory
  checkCudaErrors(cudaMallocHost(&h_result, wrp_log));
  checkCudaErrors(cudaMallocHost(&h_result_time, wrp_log));

  // allocate host memory to create pointer-chasing list (copied to device memory)
  checkCudaErrors(cudaMallocHost(&h_data, bytesize));

  /*ptr = 0;
   for (int i = 0; i < element_count; i++) {
     // index values separated by number of elements per line (32)
     nextptr = i * line_elements;
     h_data[ptr] = nextptr;
    //  printf("[%d] = %d\n", ptr, nextptr);
     ptr = nextptr;
   }
   h_data[ptr] = 0;  //last points to first*/
// return 0;
  
  // Threads = dim3(32, 1, 1);
  // Blocks = dim3(20, 1, 1);

  // copy pointer-chasing array in host memory to device memory spaces
  for (i = 0; i < NUM_SPACES; i++) {
     //checkCudaErrors(cudaMemcpyAsync(h_ptrs[i],  h_data, bytesize, cudaMemcpyHostToDevice, mem_stream));  
     //checkCudaErrors(cudaStreamSynchronize(mem_stream));
  }
  run_time = run_seconds * 1000000000ULL;  //seconds to nanoseconds
  shared_space = MAX_WARP_LOG * sizeof(unsigned short) + (1<<10); //32KB per block/SM
  // printf("eviction begun...\n");  
  //memoryKernel<<<Blocks, Threads, shared_space, mem_stream>>>(d_ptrs, d_result, bytesize, run_time, 0, d_flush);  
  // for (auto start = std::chrono::steady_clock::now(), now = start; now < start + std::chrono::seconds{run_seconds}; now = std::chrono::steady_clock::now()) 
  // for(int iter = 0; iter < 10; iter++)
  // {
    // memoryKernel<<<Blocks, Threads, 0, mem_stream>>>(d_ptrs, d_result, bytesize, run_time, 0, d_flush);  
    // launchSM<<<Blocks, Threads, 0, mem_stream>>>(d_result, 0);  
    if(smk_hybrid) {
      // WE'RE NOT DOING BASELINES ANYMORE
      switch(smk_comp_warps) {
        case 0: // 0 comp warps 32 mem warps
          Threads = dim3(32, 32, 1);   
          Blocks = dim3(20, 1, 1);
          // bytesize=983040 thread-level-accesses=false(bc l1) compute=false mem_blocks=20 sms=20 stride=128 l1=true rw_mode=1
          memoryKernelSingleSM<<<Blocks, Threads, 0, mem_stream>>>(d_ptrs, d_result, 983040, run_time, 0, d_flush, false, false, 20, 20, 128, true, 1, mem_r_warps);
          break; 

        case 16: // 16 comp warps 16 mem warps
          Threads = dim3(32, 4, 1); // 80 blocks of 4 warps bc hybrid, so can do all granularities
          Blocks = dim3(80, 1, 1);
          // bytesize=983040 thread-level-accesses=false(bc l1) compute=false mem_blocks=20 sms=20 stride=128 l1=true rw_mode=1
          memoryKernelSingleSM<<<Blocks, Threads, 0, mem_stream>>>(d_ptrs, d_result, 983040, run_time, 0, d_flush, false, false, 80, 20, 128, true, 1, mem_r_warps/4);
          usleep(100); // 100us

          Threads = dim3(32, 4, 1);
          Blocks = dim3(80, 1, 1); // 80 blocks of 4 warps bc hybrid, so can do all granularities
          // comp_instr=0 (FP16 ADD)
          computeKernel<<<Blocks, Threads, 0, comp_stream>>>(d_ptrs2, d_result2, 983040, run_time, 0,    8,    2, 1048*1048, comp_fp_warps/4);
          break;

        case 32:  // 32 comp warps 0 mem warps
          Threads = dim3(32, 16, 1);
          Blocks = dim3(20*2, 1, 1); // we do 40 blocks of 16 warps rather than 20 of 32 bc this is how we do compute
          // comp_instr=0 (FP16 ADD)
          computeKernel<<<Blocks, Threads, 0, comp_stream>>>(d_ptrs2, d_result2, 983040, run_time, 0,    8,    2, 1048*1048, comp_fp_warps/2);    
          break;  
      }
      checkCudaErrors(cudaStreamSynchronize(mem_stream));   
      checkCudaErrors(cudaStreamSynchronize(comp_stream));
      return;
    }
    /*
    if(smk_hybrid) { 
      // L1
      Threads = dim3(32, 32-smk_comp_warps, 1);   
      Blocks = dim3(20, 1, 1);
      // bytesize !identical_thread_accesses (bc l1) mem_blocks=20 sms=20 stride=128 l1 rw_mode
      memoryKernelSingleSM<<<Blocks, Threads, 0, mem_stream>>>(d_ptrs, d_result, bytesize, run_time, 0, d_flush, !identical_thread_accesses, baseline, 20, 20, 128, l1, rw_mode);
      usleep(100); // 100us

      if(!baseline) {
        // launch compute blocks
        Threads = dim3(32, smk_comp_warps, 1); 
        Blocks = dim3(20*smk_comp_warps/16, 1, 1);   
        // comp_instr=0 (FP16 ADD)
        computeKernel<<<Blocks, Threads, 0, comp_stream>>>(d_ptrs2, d_result2, bytesize, run_time, 0, 0, 2, 1048*1048); 
      } else {
        // launch baseline suspension blocks
        Threads = dim3(32, 16, 1); // 16 warps instead of 32 (NUM_WARPS_PER_BL) because we will only allow multiples of 16 warps
        Blocks = dim3(20*smk_comp_warps/16, 1, 1);  
        occupySM<<<Blocks, Threads, 0, comp_stream>>>(d_result2, run_time);  
      }
      checkCudaErrors(cudaStreamSynchronize(mem_stream));   
      checkCudaErrors(cudaStreamSynchronize(mem_stream2));  
      checkCudaErrors(cudaStreamSynchronize(comp_stream));
      return;
    }
    */

    if(num_mem_blocks>0) { // l1 included in this
      // launch memory (L2 or L1) blocks (or baseline compute blocks)
      Threads = dim3(32, 32, 1);
      Blocks = dim3(num_mem_blocks==num_mem_sms?num_mem_blocks:num_mem_sms+20, 1, 1);
      if(l1) {
        Threads = dim3(32, num_comp_warps, 1);   
        Blocks = dim3(20, 1, 1);    
      }
      // should not do this if baseline && l1 - in that case need to launch occupation 
      memoryKernelSingleSM<<<Blocks, Threads, 0, mem_stream>>>(d_ptrs, d_result, bytesize, run_time, 0, d_flush, !identical_thread_accesses, baseline, num_mem_blocks, num_mem_sms, stride_bytes, l1, rw_mode, -1);    
    }
    usleep(100); // 100us
    if(num_comp_warps>0 && !l1) {
      if(!baseline) {
        // launch compute blocks
        Threads = dim3(32, 16, 1); // 16 warps instead of 32 (NUM_WARPS_PER_BL) because we will only allow multiples of 16 warps
        Blocks = dim3((20-num_mem_sms)*num_comp_warps/16, 1, 1);   
        computeKernel<<<Blocks, Threads, 0, comp_stream>>>(d_ptrs2, d_result2, bytesize, run_time, 0, comp_instr, 2, 1048*1048, -1); 
      } else {
        // launch baseline suspension blocks
        Threads = dim3(32, 16, 1); // 16 warps instead of 32 (NUM_WARPS_PER_BL) because we will only allow multiples of 16 warps
        Blocks = dim3((20-num_mem_sms)*num_comp_warps/16, 1, 1);
        occupySM<<<Blocks, Threads, 0, comp_stream>>>(d_result2, run_time);  
      }
    }
    checkCudaErrors(cudaStreamSynchronize(mem_stream));   
    checkCudaErrors(cudaStreamSynchronize(mem_stream2));  
    checkCudaErrors(cudaStreamSynchronize(comp_stream));
    




  // }
  //testKernel<<<Blocks, Threads, 0, mem_stream>>>(d_ptrs, d_result);
  
  // return 0;
  // copy any logged data back to host memory
  // checkCudaErrors(cudaMemcpyAsync(h_result, d_result, wrp_log, cudaMemcpyDeviceToHost, mem_stream));
  // checkCudaErrors(cudaStreamSynchronize(mem_stream));

  // checkCudaErrors(cudaMemcpyAsync(h_result_time, d_result_time, wrp_log, cudaMemcpyDeviceToHost, mem_stream));
  // checkCudaErrors(cudaStreamSynchronize(mem_stream));

  // // copy any side information stored in device space zero back to host memory
  // checkCudaErrors(cudaMemcpyAsync(h_data, h_ptrs[0], bytesize, cudaMemcpyDeviceToHost, mem_stream));
  // checkCudaErrors(cudaStreamSynchronize(mem_stream));
  // for(i= 0; i < 20;i++){
  //   int base = 0;
  //   if(h_result_time[i] < h_result_time[base])
  //     printf("%d: %llu\n",h_result_time[20+i], h_result_time[base] - h_result_time[i]);
  //   else
  //     printf("%d: %llu\n",h_result_time[20+i], h_result_time[i] - h_result_time[base]);
  // }
  // for(i = 0; i < 50; i++){
  //   printf("%d\n", h_result[i]);
  // }
  // return 0;
  // int min =  10000;
  // int cnt  = 0;

  // for (i = 0; i < NUM_SPACES; i++) {
  //   for (j = 0; j < element_count; j++) {
  //     unsigned long tmp = h_result[j];
  //     // log_idx = (i * element_count);
  //     printf("%lu\n", tmp); 
  //     //if(min > tmp && tmp >0) min = tmp;
  //     if(tmp < 400) cnt++;
  //   }	
  // }
  // printf("%d out of %d\n", cnt, element_count);
  // printf("%llu\n", h_result[3]);
  //printf("min: %d\n", min);
   // cudaDeviceReset();
}

#include <helper_kernels.cuh>

__global__ void
appMemoryKernel(unsigned int **k_ptrs, unsigned int *k_result, int bytesize, int samples, int myZero, unsigned int *c_flush, int num_visits)  //c_flush is size of k_data (bytesize) and used to flush cache initially
{
    //WARNING: All data arrays and numbers of blocks/warps powers of 2

  // uncomment next statement to allocate a shared memory array for logging read times
 //  __shared__ unsigned short blk_log[MAX_WARP_LOG + 1];
  
  unsigned int *k_data;  //pointer to current array
  // Use built-in variables to compute block, thread, and warp numbers
  int gbl_blk,lcl_thd,lcl_wrp;
  gbl_blk = (blockIdx.y * gridDim.x) + blockIdx.x;
  lcl_thd = (threadIdx.y * blockDim.x) + threadIdx.x;
  lcl_wrp = lcl_thd / 32;

  if(lcl_thd==0 && gbl_blk==0) printf("starting app.\n");

  int i,j, k;

  int wrp_count;
  
  int wrp_max; 
  wrp_max = (bytesize / TX2_CACHE_LINE) / (NUM_BLOCKS_APP * NUM_WARPS_APP);

  unsigned long long cycles_before, cycles_after, before, after;
  unsigned short cycles_add;
  int wrp_log;
  wrp_log =  ((gbl_blk * NUM_WARPS_APP) + lcl_wrp) * wrp_max;

  int ptr = 0;  // holds the index of the next array element to be read
  //unsigned int ptr_start;
  //ptr_start = ((gbl_blk * NUM_WARPS_APP) + lcl_wrp) * (wrp_max * 32) + ((ptr) * myZero * wrp_count);
  unsigned int r_sum;  //a nonsense variable used to help defeat optimization
  r_sum = 0;
  int flush_max = bytesize / sizeof(unsigned int);
  //flush existing data from the cache by references to c_flush
 //  for (i = 0; i < flush_max; i++)
    //   r_sum = r_sum + c_flush[i];
      // r_sum = r_sum + __ldcg(&(c_flush[i]));
  
  //record the kernel start and current times in nanoseconds
 //  cycles_after = clock64();
  __syncthreads();
 
 // while ((clock_now - clock_begin) < (run_time+100)) {
  
     // loop over all the device memory spaces
    
       k_data = k_ptrs[0];  //get pointer to next device space
       
        // loop over each space for the number of passes specified
        for (i = 0; i < samples; i++) {
      // compute the local warp number and the start index in its array partition
            // ptr = ((gbl_blk * NUM_WARPS_APP) + lcl_wrp) * (wrp_max * 32) + (ptr * myZero * wrp_count);
            ptr = ((gbl_blk * NUM_WARPS_APP) + lcl_wrp) * (wrp_max * 32) + (ptr * myZero * wrp_count);
            int base = ptr;
            //ptr = ptr_start;
            __syncthreads();
            before = clock64();
            // the local warp loops while chasing the pointers in its partition
      // uncomment the lines inside the loop to record read access times in shared memory
       //wrp_count = 0;
// #pragma unroll 1
      for (wrp_count = 0; wrp_count < wrp_max; wrp_count++) {
              //  cycles_before = clock64();
              //  ptr = __ldcv(&(k_data[ptr]));
              ptr = k_data[ptr];
              // k_result[base + wrp_count] = ptr;
             //  k_result[base + wrp_count] = ptr;
               r_sum += ptr;
              //  cycles_after = clock64();
              //  blk_log[wrp_log + wrp_count] = (unsigned short) (cycles_after - cycles_before);
       }
       
             __syncthreads();
             after = clock64();
/*
* For access time logging, comment the ifdef/endif statements to copy times
* from shared memory to device memory log
*/
//#ifdef DO_LOG	       
       //  }
          //    __syncthreads();
             
          //    int log_idx;
          //    log_idx = k * MAX_WARP_LOG;
          //   for (j = 0; j < wrp_max; j++)
              //   k_result[log_idx + wrp_log + j] = (unsigned short)(blk_log[wrp_log + j]);
              //   k_result[log_idx + wrp_log + j] = (unsigned short)cycles_add;
//#endif		  
  } //end loop for passes through a device space
  //   } // end loop over all spaces	  
  //    __syncthreads();
     //clock_now = gclock64();
 // } //end outer loop for run time

  // __syncthreads();
  ptr = ptr + r_sum;    

  // k_data[0] = wrp_count;
  k_result[1] = ptr; // Make sure the compiler believes ptr is a result
                   // and does not eliminate references as optimization
  k_result[2] = r_sum;
  k_result[3] = after-before;

  if(lcl_thd==0 && gbl_blk==0) printf("finishing app.\n");
}
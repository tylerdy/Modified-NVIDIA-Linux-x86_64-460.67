#include <kernel_helpers.cuh>

__global__ void
victim_memoryKernel(unsigned int *k_ptrs[MAX_SPACES], unsigned short *k_result, int bytesize, unsigned long long run_time, int myZero, unsigned int *c_flush)  //c_flush is size of k_data (bytesize) and used to flush cache initially
//WARNING: All data arrays and numbers of blocks/warps powers of 2
{
    
    // uncomment next statement to allocate a shared memory array for logging read times
    __shared__ unsigned short blk_log[MAX_WARP_LOG + 1];
    
    unsigned int *k_data;  //pointer to current array
    // int *k_data;  //pointer to current array
    // k_data = k_ptrs[0];
    // k_data[1] = k_data[32];
    // return;
    // Use built-in variables to compute block, thread, and warp numbers
    int gbl_blk,lcl_thd,lcl_wrp;
    gbl_blk = (blockIdx.y * gridDim.x) + blockIdx.x;
    lcl_thd = (threadIdx.y * blockDim.x) + threadIdx.x;
    lcl_wrp = lcl_thd / 32;
    
    if(lcl_thd==0 && gbl_blk==0) {
        //printf("VICTIM KERNEL. %d passes. Runtime is %d. %d/%d blocks/warps. Cache Operator Victim is %s.\n", NUM_PASSES, run_time, NUM_BLOCKS, NUM_WARPS, "");

        if(NUM_PASSES<2 || run_time>0) {
            printf("Warning: possibly bad parameter.\n");
        }
    }

    int i,j, k;

    int wrp_count;
    
    // the number of array elements in each warp's non-overlapping
    // partition of the array
    int wrp_max;
    wrp_max = (bytesize / TX2_CACHE_LINE) / (NUM_BLOCKS * NUM_WARPS);

    /* Uncomment the following for logging read access times
     * Logging must be coordinated with the launching CUDA program
     */
    unsigned long long cycles_before, cycles_after, before_pass, after_pass;
    unsigned short cycles_add;
    int wrp_log;
    wrp_log =  ((gbl_blk * NUM_WARPS) + lcl_wrp) * wrp_max;

    int ptr = 0;  // holds the index of the next array element to be read
    //unsigned int ptr_start;
    //ptr_start = ((gbl_blk * NUM_WARPS) + lcl_wrp) * (wrp_max * 32) + ((ptr) * myZero * wrp_count);
    unsigned int r_sum;  //a nonsense variable used to help defeat optimization
    r_sum = 0;
    //extern __shared__ unsigned long long clock_begin;   //clock value kernel marks as its start time
    //extern __shared__ unsigned long long clock_now;     //clock value current instant
    //ptr = __ldcg(&(k_ptrs[0][myZero]));
    cycles_before = clock64();
    // r_sum += ptr;
    cycles_after = clock64();
    cycles_add = (unsigned short) (cycles_after - cycles_before);
    //cycles_add = 0;

    
    //record the kernel start and current times in nanoseconds
    cycles_after = clock64();
    __syncthreads();
    //clock_begin = gclock64();
    //clock_now = clock_begin;

    //Main loop runs while the time elapsed since the start is less
    //than the run_time parameter 
   // while ((clock_now - clock_begin) < (run_time+100)) {
    
       // loop over all the device memory spaces
       for (k = 0; k < NUM_SPACES; k++) {
         k_data = k_ptrs[0];  //get pointer to next device space

          // loop over each space for the number of passes specified
          for (i = 0; i < NUM_PASSES; i++) {
	      // compute the local warp number and the start index in its array partition
              ptr = ((gbl_blk * NUM_WARPS) + lcl_wrp) * (wrp_max * 32) + (ptr * myZero * wrp_count);
              //ptr = ptr_start;
              __syncthreads();
              if(i==1 && gbl_blk==0 && lcl_thd==0) before_pass = clock64();

              // the local warp loops while chasing the pointers in its partition
	      // uncomment the lines inside the loop to record read access times in shared memory
         //wrp_count = 0;
// #pragma unroll 1
        for (wrp_count = 0; wrp_count < wrp_max; wrp_count++) {
                //cycles_before = clock64();
                #ifdef CACHE_OP
                ptr = __ldcv(&(k_data[ptr]));
                #else // CACHE_OP
                ptr = k_data[ptr];
                #endif // CACHE_OP
                r_sum += ptr;
                //cycles_after = clock64();
                //blk_log[wrp_log + wrp_count] = (unsigned short) (cycles_after - cycles_before);
         }
               __syncthreads();
               if(i==NUM_PASSES-1 && gbl_blk==0 && lcl_thd==0) after_pass = clock64();
/*
 * For access time logging, comment the ifdef/endif statements to copy times
 * from shared memory to device memory log
 */
//#ifdef DO_LOG	       
          }
               __syncthreads();
               
//#endif		  
	//   } //end loop for passes through a device space
    //   } // end loop over all spaces	  
       __syncthreads();
       //clock_now = gclock64();
   } //end outer loop for run time

   //int log_idx;
    //log_idx = 0 * MAX_WARP_LOG;
    //for (j = 0; j < wrp_max; j++)
    //    k_result[log_idx + wrp_log + j] = (unsigned short)(blk_log[wrp_log + j]);
    //    k_result[log_idx + wrp_log + j] = (unsigned short)cycles_add;

    //__syncthreads();
    ptr = ptr + r_sum;    

    k_data[0] = wrp_count;
    k_data[1] = ptr; // Make sure the compiler believes ptr is a result
                     // and does not eliminate references as optimization
    k_data[2] = r_sum;
    if(gbl_blk==0 && lcl_thd==0) k_data[3] = after_pass-before_pass;
}

/* Wrapper function for reading the global nanosecond-precision timer
 */

static __device__ __inline__ unsigned long long int gclock64() {

    unsigned long long int rv;

    asm volatile ( "mov.u64 %0, %%globaltimer;" : "=l"(rv) );

    return rv;

}
/* Wrapper function for reading the local SM cycle counter
 */

static __device__ __inline__ unsigned long long int cycles64() {

    unsigned long long int rv;

    asm volatile ( "mov.u64 %0, %%clock64;" : "=l"(rv) );

    return rv;

}

/* The kernel for sequential pointer chasing in device memory arrays.
 * The kernel expects to be launched with two blocks and 128 threads
 * (4 warps) per block. Each warp performs pointer chasing on a
 * non-overlapping portion of an array which was initialzed in the
 * launching CUDA program.
 *
 * The parameters of the kernel are:
 *   k_ptrs, an array of pointers to initialized arrays in device memory
 *   K_result, an array of device memory locations for the kernel to
 *       store logged read access times
 *   bytesize, the size in bytes of the pointer-chasing arrays
 *   run_time, the time in nanoseconds the kernel will run
 *   myZero, always set to 0 by the launch code -- used to avoid
 *       compiler optimizations that eliminate statements
 *   c_flush, an array allocated in device memory and used by the kernel
 *       to perform an initial flush of the cache content
 *
 * The kernel (and the launching CUDA program) can be customized to
 * log read access times for elements of the arrays.  Logging must be
 * coordinated between the kernel and the CUDA program.
 */

__global__ void
memoryKernel(unsigned int *k_ptrs[MAX_SPACES], unsigned short *k_result, int bytesize, unsigned long long run_time, int myZero, unsigned int *c_flush)  //c_flush is size of k_data (bytesize) and used to flush cache initially
{
    //WARNING: All data arrays and numbers of blocks/warps powers of 2

    // uncomment next statement to allocate a shared memory array for logging read times
    __shared__ unsigned short blk_log[MAX_WARP_LOG + 1];

    unsigned int *k_data;  //pointer to current array
    
    // Use built-in variables to compute block, thread, and warp numbers
    int gbl_blk,lcl_thd,lcl_wrp;
    gbl_blk = (blockIdx.y * gridDim.x) + blockIdx.x;
    lcl_thd = (threadIdx.y * blockDim.x) + threadIdx.x;
    lcl_wrp = lcl_thd / 32;

    int i,j, k;

    int wrp_count;
    
    // the number of array elements in each warp's non-overlapping
    // partition of the array
    int wrp_max;
    wrp_max = (bytesize / TX2_CACHE_LINE) / (NUM_BLOCKS * NUM_WARPS);

    /* Uncomment the following for logging read access times
     * Logging must be coordinated with the launching CUDA program
     */
    unsigned long long cycles_before, cycles_after, cycles_add;
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
    r_sum += ptr;
    cycles_after = clock64();
    cycles_add = (unsigned short) (cycles_after - cycles_before);
    //cycles_add = 0;
    int flush_max = bytesize / sizeof(unsigned int);
    //flush existing data from the cache by references to c_flush
    for (i = 0; i < flush_max; i++)
         r_sum = r_sum + c_flush[i];
    
    //record the kernel start and current times in nanoseconds
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

              // the local warp loops while chasing the pointers in its partition
	      // uncomment the lines inside the loop to record read access times in shared memory
         //wrp_count = 0;
#pragma unroll 1
        for (wrp_count = 0; wrp_count < wrp_max; wrp_count++) {
                 cycles_before = clock64();
                 ptr = __ldcg(&(k_data[ptr]));
                 r_sum += ptr;
                 cycles_after = clock64();
                 blk_log[wrp_log + wrp_count] = (unsigned short) (cycles_after - cycles_before);
         }
               __syncthreads();
/*
 * For access time logging, comment the ifdef/endif statements to copy times
 * from shared memory to device memory log
 */
//#ifdef DO_LOG	       
          }
               __syncthreads();
               int log_idx;
               log_idx = k * MAX_WARP_LOG;
               for (j = 0; j < wrp_max; j++)
                    k_result[log_idx + wrp_log + j] = blk_log[wrp_log + j] - cycles_add;
                  //k_result[log_idx + wrp_log + j] = cycles_add;
//#endif		  
	//   } //end loop for passes through a device space
      } // end loop over all spaces	  
       __syncthreads();
       //clock_now = gclock64();
   //} //end outer loop for run time

    //__syncthreads();
    ptr = ptr + r_sum;    

    k_data[0] = wrp_count;
    k_data[1] = ptr; // Make sure the compiler believes ptr is a result
                     // and does not eliminate references as optimization
    k_data[2] = r_sum;
}

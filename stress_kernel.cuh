
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
launchSM(int *k_result, int myZero){
    // extern __shared__ unsigned long long clock_begin;   
    // extern __shared__ unsigned long long clock_now;  
    int blockid;

    __shared__ unsigned short blk_log[MAX_WARP_LOG];
    blockid = blockIdx.x;

    int smid;
    asm("mov.u32 %0, %smid;" : "=r"(smid));
    if(smid != 0) return;
    // k_result[blockid] = smid;
    int gbl_blk,lcl_thd,lcl_wrp;
    lcl_thd = (threadIdx.y * blockDim.x) + threadIdx.x;
    lcl_wrp = lcl_thd / 32;
    // if(smid != 0) return;
    int wrp_count, myid;
    myid = blockIdx.x / 20;
    // if(myid != 0) return;
    int wrp_max;
    wrp_max = (TX2_CACHE_SIZE / TX2_CACHE_LINE) / (64);
    int wrp_log;
    wrp_log =  ((myid * 32) + lcl_wrp) * wrp_max;
    int local_wrp_log;
    local_wrp_log =  (lcl_wrp) * wrp_max;

    // clock_begin = gclock64();
    // clock_now = clock_begin;
    unsigned long long cycles_before, cycles_after, before, after;
    int ptr, sum;
    sum = 0;
    // ptr = ((myid * 32) + lcl_wrp) * (wrp_max * 32) + (ptr * myZero * wrp_count);
// #pragma unroll 1
    for (wrp_count = 0; wrp_count < wrp_max; wrp_count++) {
        cycles_before = clock64();
            ptr = wrp_count * myZero;
            sum += ptr;
            cycles_after =   clock64();
            blk_log[wrp_log + wrp_count] = (unsigned short) (cycles_after - cycles_before);

    }
    __syncthreads();
    // int cnt = 0;
    for (int j = 0; j < wrp_max; j++){
        k_result[wrp_log + j] = blk_log[wrp_log + j];
    }
    k_result[0] = sum;
    k_result[1] = ptr;
        // k_result[wrp_log + j] += blk_log[wrp_log + j];
        // __syncthreads();
        // k_result[0] += cnt;
        // atomicAdd(&(k_result[0]), cnt);
        // k_result[wrp_log + j] +=  local_wrp_log + j;
    // __syncthreads();
        // k_result[wrp_log + j] = blockid;
    // for(int j = 0; j < 40; j++){
    // k_result[blockid] = (blockid + 1)*100+smid;
    // }
}

__global__ void
testKernel(unsigned long long *k_result){
    k_result[blockIdx.x] = clock64();
    int smid;
    asm("mov.u32 %0, %smid;" : "=r"(smid));
    k_result[20+blockIdx.x] = smid;
}

__global__ void
memoryKernelSingleSM(unsigned int *k_ptrs[MAX_SPACES], int *k_result, int bytesize, unsigned long long run_time, int myZero, unsigned int *c_flush)  //c_flush is size of k_data (bytesize) and used to flush cache initially
{
    int num_sm = 16;
    int smid;
    asm("mov.u32 %0, %smid;" : "=r"(smid));
    if(smid >= num_sm) return;
    // __shared__ unsigned short blk_log[MAX_WARP_LOG];
    unsigned int *k_data;  
    // Use built-in variables to compute block, thread, and warp numbers
    int gbl_blk,lcl_thd,lcl_wrp;
    lcl_thd = (threadIdx.y * blockDim.x) + threadIdx.x;
    lcl_wrp = lcl_thd / 32;

    int i,j, k;
    k = 0;
    int wrp_count, myid;
    
    // myid = blockIdx.x / 20;
    myid = blockIdx.x;
    if(myid >= 20) myid -= (20-num_sm);
    int wrp_max;
    wrp_max = (bytesize / TX2_CACHE_LINE) / (64*num_sm);

    /* Uncomment the following for logging read access times
     * Logging must be coordinated with the launching CUDA program
     */
    unsigned long long cycles_before, cycles_after, before, after;
    unsigned short cycles_add;
    int wrp_log;
    wrp_log =  ((myid * 32) + lcl_wrp) * wrp_max;
    int local_wrp_log;
    local_wrp_log =  (lcl_wrp) * wrp_max;
    int ptr = 0;  // holds the index of the next array element to be read
    unsigned int r_sum;  //a nonsense variable used to help defeat optimization
    r_sum = 0;
    extern __shared__ unsigned long long clock_begin;   //clock value kernel marks as its start time
    extern __shared__ unsigned long long clock_now;     //clock value current instant
    //record the kernel start and current times in nanoseconds
    __syncthreads();
    clock_begin = gclock64();
    clock_now = clock_begin;

    k_data = k_ptrs[0]; 
        
   while ((clock_now - clock_begin) < (run_time+100)) {
    // while(true){
    // for (k = 0; k < NUM_SPACES; k++) {
    //   for (i  = 0; i < NUM_PASSES; i++) {
        ptr = ((myid * 32) + lcl_wrp) * (wrp_max * 32) + (ptr * myZero * wrp_count);
        // int base = ptr;
// #pragma unroll 1
        for (wrp_count = 0; wrp_count < wrp_max; wrp_count++) {
                //  cycles_before = clock64();
                //  ptr = __ldcv(&(k_data[ptr]));
                // ptr = k_data[ptr];
                ptr = wrp_count;   
                 r_sum += ptr;  
                //  cycles_after =   clock64();/
                
                // k_result[wrp_log + wrp_count] = ptr;
                // blk_log[wrp_log + wrp_count] = ptr;
                // blk_log[local_wrp_log + wrp_count] = wrp_count;
                   
            // }
        }
            //    __syncthreads();
            //    after = clock64();
               clock_now=gclock64();
   }
        // int log_idx;
    // __syncthreads();           
            //   for (int j = 0; j < wrp_max; j++){
                //    k_result[wrp_log + j] = blk_log[wrp_log + j];
            //   }

    __syncthreads();
    ptr = ptr + r_sum;       
    k_result[1] = wrp_count;
    k_result[2] = ptr; // Make sure the compiler believes ptr is a result
                     // and does not eliminate references as optimization
    k_result[3] = r_sum;
    // k_result[4] = after-before;
}


__global__ void
memoryKernel(unsigned int *k_ptrs[MAX_SPACES], int *k_result, int bytesize, unsigned long long run_time, int myZero, unsigned int *c_flush)  //c_flush is size of k_data (bytesize) and used to flush cache initially
{
    //WARNING: All data arrays and numbers of blocks/warps powers of 2

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

    int i,j, k;
    k = 0;
    int wrp_count;
    
    // the number of array elements in each warp's non-overlapping
    // partition of the array
    int wrp_max;
    wrp_max = (bytesize / TX2_CACHE_LINE) / (NUM_BLOCKS * NUM_WARPS);

    /* Uncomment the following for logging read access times
     * Logging must be coordinated with the launching CUDA program
     */
    unsigned long long cycles_before, cycles_after, before, after;
    unsigned short cycles_add;
    int wrp_log;
    wrp_log =  ((gbl_blk * NUM_WARPS) + lcl_wrp) * wrp_max;

    int ptr = 0;  // holds the index of the next array element to be read
    //unsigned int ptr_start;
    //ptr_start = ((gbl_blk * NUM_WARPS) + lcl_wrp) * (wrp_max * 32) + ((ptr) * myZero * wrp_count);
    unsigned int r_sum;  //a nonsense variable used to help defeat optimization
    r_sum = 0;
    extern __shared__ unsigned long long clock_begin;   //clock value kernel marks as its start time
    extern __shared__ unsigned long long clock_now;     //clock value current instant
    //ptr = __ldcg(&(k_ptrs[0][myZero]));
    // cycles_before = clock64();
    // r_sum += ptr;
    // cycles_after = clock64();
    // cycles_add = (unsigned short) (cycles_after - cycles_before);
    //cycles_add = 0;
    int flush_max = bytesize / sizeof(unsigned int);
    //flush existing data from the cache by references to c_flush
    // for (i = 0; i < flush_max; i++)
    //     r_sum = r_sum + c_flush[i];
        // r_sum = r_sum + __ldcg(&(c_flush[i]));
    
    //record the kernel start and current times in nanoseconds
    // cycles_after = clock64();
    __syncthreads();
    clock_begin = gclock64();
    clock_now = clock_begin;

    //Main loop runs while the time elapsed since the start is less
    //than the run_time parameter 
    k_data = k_ptrs[0]; 
    // for(int ppp=0; ppp<256;ppp++){
        
    while ((clock_now - clock_begin) < (run_time+100)) {
    // while(true){
       // loop over all the device memory spaces
    //    for (k = 0; k < NUM_SPACES; k++) {
          //get pointer to next device space
         
          // loop over each space for the number of passes specified
        //   for (i  = 0; i < NUM_PASSES; i++) {
	      // compute the local warp number and the start index in its array partition
              ptr = ((gbl_blk * NUM_WARPS) + lcl_wrp) * (wrp_max * 32) + (ptr * myZero * wrp_count);
              //ptr = ptr_start;
            //   __syncthreads();
            //   before = clock64();
              // the local warp loops while chasing the pointers in its partition
	      // uncomment the lines inside the loop to record read access times in shared memory
         //wrp_count = 0;
// #pragma unroll 1
        for (wrp_count = 0; wrp_count < wrp_max; wrp_count++) {
                //  cycles_before = clock64();
                //  ptr = __ldcv(&(k_data[ptr]));
                ptr = k_data[ptr];
                // ptr = wrp_count;
                 r_sum += ptr;
                //  cycles_after = clock64();
                //  blk_log[wrp_log + wrp_count] = (unsigned short) (cycles_after - cycles_before);
         }
        //   }
            //    __syncthreads();
            //    after = clock64();
               clock_now=gclock64();
//    }
/*
 * For access time logging, comment the ifdef/endif statements to copy times
 * from shared memory to device memory log
 */
//#ifdef DO_LOG	       
        
            //    __syncthreads();
               
            //    int log_idx;
            //    log_idx = 0 * MAX_WARP_LOG;
            //   for (j = 0; j < wrp_max; j++)
            //       k_result[log_idx + wrp_log + j] = (unsigned short)(blk_log[wrp_log + j]);
                //   k_result[log_idx + wrp_log + j] = (unsigned short)cycles_add;
//#endif		  
	//   } //end loop for passes through a device space
    //   } // end loop over all spaces	  
    //    __syncthreads();
       //clock_now = gclock64();
    } //end outer loop for run time
    // }
    __syncthreads();
    ptr = ptr + r_sum;    
    // k_data[0] = wrp_count;    
    k_result[0] = wrp_count;
    k_result[1] = ptr; // Make sure the compiler believes ptr is a result
                     // and does not eliminate references as optimization
    k_result[2] = r_sum;
    k_result[3] = after-before;
    // k_data[0] = wrp_count;
    // k_data[1] = ptr; // Make sure the compiler believes ptr is a result
    //                  // and does not eliminate references as optimization
    // k_data[2] = r_sum;
    // k_data[3] = after-before;
}

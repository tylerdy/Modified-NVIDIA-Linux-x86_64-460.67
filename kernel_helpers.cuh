#ifndef HELPERS

#define HELPERS

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

__global__ void
flushKernel(unsigned short *k_result,int bytesize, unsigned int *c_flush) {
   int flush_max = bytesize / sizeof(unsigned int);
   unsigned int r_sum; 
   for (int i = 0; i < flush_max; i++)
        r_sum = r_sum + c_flush[i];
   k_result[0] = r_sum;
   
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
testKernel(unsigned int *k_ptr[MAX_SPACES], unsigned short *k_result){
  int gbl_blk,lcl_thd,lcl_wrp;
  gbl_blk = (blockIdx.y * gridDim.x) + blockIdx.x;
  lcl_thd = (threadIdx.y * blockDim.x) + threadIdx.x;
  lcl_wrp = lcl_thd / 32;

  int wrp_count;

  int wrp_max;
  wrp_max =(TX2_CACHE_SIZE / TX2_CACHE_LINE) /(NUM_BLOCKS * NUM_WARPS);

  int wrp_log;
  wrp_log =  ((gbl_blk * NUM_WARPS) + lcl_wrp) * wrp_max;
  unsigned int *k_data = k_ptr[0];
  for (int j = 0; j < wrp_max; j++)
      k_result[wrp_log + j] = k_data[wrp_log + j];
}

#endif
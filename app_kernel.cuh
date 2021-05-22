__global__ void
appMemoryKernel(unsigned int *k_data, unsigned long long *k_result, int bytesize, int samples, int myZero, unsigned int *c_flush)  //c_flush is size of k_data (bytesize) and used to flush cache initially
{
   //WARNING: All data arrays and numbers of blocks/warps powers of 2
   //Shared space is 32KB per block which holds 4096 long long ints
   //Log to global memory each time the warp has made 4096 entries (or at
   //end of cache fill if fewer).

    extern __shared__ unsigned long long blk_log[];

    // Use built-in variables to compute block, thread, and warp numbers
    int gbl_blk = (blockIdx.y * gridDim.x) + blockIdx.x;
    int Nwarps = (blockDim.x * blockDim.y) / 32;
    int lcl_thd = (threadIdx.y * blockDim.x) + threadIdx.x;
    int lcl_wrp = lcl_thd / 32;
	
    int log_wrp = 0;
    int max_log = 4096;  //dump recorded data when shared memory is full of values
    int limit = bytesize / sizeof(unsigned int);  //maximum number of elements in array
    
    int i, j;
    int ptr;
    int counter;
    unsigned int r_sum = 0;

    unsigned long long cycles_before;  // record cycle counts just before loop begins
    
    int flush_max = bytesize / sizeof(unsigned int);
    
    //clear memory at k_data from the cache by references to c_flush
    for (i = 0; i < flush_max; i++)
       r_sum = r_sum + c_flush[i];

    ptr = 0;  //initialize to first array element
    // main loop runs until all measurement samples have been recorded
    for (counter = 0; counter < samples; counter++) {
       __syncthreads();
       cycles_before = cycles64();   // record cycle count before loop

       // loop over all elements in the array
       for (i = 0; i < limit; i++) {
          ptr = k_data[ptr];    // read array
          // compute next arbitrary index
          ptr = (ptr * myZero) + ((cycles64() + gclock64()) & 0X1FFFF);
       } // end loop over all array elements
       // record the elapsed number of cycles during the loop
       blk_log[counter % max_log] = cycles64() - cycles_before; 
       // re-initialize to first array element (and prevent compiler variable elimination)
       ptr = (ptr * myZero * limit);
       
       // if shared memory is full, copy contents to device memory
       if ((counter % max_log) < (max_log - 1))
       	  continue;
       __syncthreads();
       for (j = 0; j < max_log; j++)
          k_result[log_wrp + j] = blk_log[j];
       log_wrp = log_wrp + max_log;
       
    } // end loop for samples
    // if there are still values in shared memory, copy to device memory
    if ((counter % max_log) < max_log) {
       for (j = 0; j < (counter % max_log); j++)
         k_result[log_wrp + j] = blk_log[j];
    }
    ptr = ptr + r_sum;    
    __syncthreads();
    k_data[0] = limit;
    k_data[1] = counter;
    k_data[2] = r_sum;
    // make ptr an output variable to prevent compiler variable elimination
    k_data[(gbl_blk * Nwarps) + lcl_wrp + 3] = ptr;
}
//#define COMP_ACCESSES 1048*1048

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
    int wrp_count, bl_id;
    bl_id = blockIdx.x / 20;
    // if(bl_id != 0) return;
    int wrp_max;
    wrp_max = (TX2_CACHE_SIZE / TX2_CACHE_LINE) / (64);
    int wrp_log;
    wrp_log =  ((bl_id * 32) + lcl_wrp) * wrp_max;
    int local_wrp_log;
    local_wrp_log =  (lcl_wrp) * wrp_max;

    // clock_begin = gclock64();
    // clock_now = clock_begin;
    unsigned long long cycles_before, cycles_after, before, after;
    int ptr, sum;
    sum = 0;
    // ptr = ((bl_id * 32) + lcl_wrp) * (wrp_max * 32) + (ptr * myZero * wrp_count);
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
occupySM(int *k_result, unsigned long long run_time){
    int smid;
    asm("mov.u32 %0, %smid;" : "=r"(smid));
    if((threadIdx.y * blockDim.x) + threadIdx.x==0)printf("16-warp suspension block on %d\n",smid);
    
    int lcl_thd,lcl_wrp;
    lcl_thd = (threadIdx.y * blockDim.x) + threadIdx.x; 
    lcl_wrp = lcl_thd / 32;
    extern __shared__ unsigned long long clock_begin;   //clock value kernel marks as its start time
    extern __shared__ unsigned long long clock_now;     //clock value current instant
    if(lcl_wrp==0) {
        
        clock_begin = gclock64();
        clock_now = clock_begin;
        while ((clock_now - clock_begin) < (run_time+100)) {
            clock_now=gclock64();
        }
        __syncthreads();
    } else {
        __syncthreads();
    }
    clock_now=gclock64();
    k_result[blockIdx.x] = clock_now;
}

__global__ void
memoryKernelSingleSM(unsigned int *k_ptrs[MAX_SPACES], int *k_result, int bytesize, unsigned long long run_time, int myZero, unsigned int *c_flush, bool thread_accesses, bool compute, int num_blocks, int num_sm, int stride_bytes, bool l1, int rw_mode, int r_per_block)  //c_flush is size of k_data (bytesize) and used to flush cache initially
{
    // gridDim.x is number of blocks launched (disregard bc we may kill some - use num_blocks instead)
    // blockDim.x is always 32
    // blockDim.y is number of warps per block (32 for L2, variable for L1)
    // threadIdx.y is local (within block) warp ID
    // threadIdx.x is thread ID within warp


    unsigned int *k_data; 
    k_data = k_ptrs[0]; 

    // return on non enemy dedicated blocks (if L2 and 2 blocks per SM)
    int smid;
    asm("mov.u32 %0, %smid;" : "=r"(smid));
    if(!l1 && num_blocks>num_sm && smid >= num_sm) { // if l1 we may have multiple blocks per SM but dont want to kill any
        if((threadIdx.y * blockDim.x) + threadIdx.x==0)printf("block returned on smid %d\n",smid);
        return;
    }


    int bl_id = blockIdx.x;
    if(bl_id >= 20 && !l1) bl_id -= (20-num_sm); // adjust if any were killed so that all remaining block ids are unique and consecutive

    int lcl_thd = (threadIdx.y * blockDim.x) + threadIdx.x; 
    if(lcl_thd>=blockDim.x*blockDim.y) printf("BAD THD ID %d",lcl_thd);
    // if(bl_id==0 && lcl_thd==blockDim.x*blockDim.y-1) printf("checked lcl_thd\n");

    

    int accessing_threads = num_blocks*( blockDim.x * blockDim.y); // number of threads in all blocks
    if(!thread_accesses) {
        accessing_threads/=32; // if all threads in warp make identical accesses, this is now number of warps
        lcl_thd/=32; // indexing by warp instead of by thread
        if(lcl_thd>blockDim.y) printf("BAD WRP ID %d\n",lcl_thd);
        // if(bl_id==0 && lcl_thd==blockDim.y-1) printf("checked lcl_thd (warp)\n");
    }

    // r_per_block = -1 means don't override instrucion type
    if(r_per_block>=0 && bl_id==0 && threadIdx.x==0 && threadIdx.y ==0) printf("%d blocks x %d warps/block of memory. %d warps/block are reads, totalling %d warps per SM.\n", num_blocks, blockDim.y, r_per_block, r_per_block*num_blocks/num_sm);

    if(r_per_block>=0) {
        if(threadIdx.y<r_per_block) rw_mode = 0;
        else rw_mode = 1;
    }

    int ELEMSIZE = 4;
    int strides = bytesize/stride_bytes;
    int elems = bytesize/ELEMSIZE;
    int elems_per_stride = stride_bytes/ELEMSIZE;
    int elems_per_thread = elems/accessing_threads;
    int strides_per_thread = strides/accessing_threads;
    if(elems_per_stride<=0) printf("BAD STRIDE SIZE\n");
    if(elems%num_blocks!=0) printf("elements don't divide into blocks evenly\n");
    if(elems%accessing_threads!=0) printf("elements don't divide into threads evenly\n");
    if(elems<accessing_threads) printf("not enough elements for threads\n");

    int bl_start_ind = bl_id * (elems/num_blocks); // (elems/num_blocks) is elems per block
    int thd_start_ind = bl_start_ind + lcl_thd * elems_per_thread; 

    // print using actual lcl_thd (bc of identical accesses case)
    // if((threadIdx.y * blockDim.x) + threadIdx.x==0) printf("bl %d from [%d to %d) /%d. lcl_thd0 covering %dB /%dB by %d strides of %dB.\n",bl_id, bl_start_ind*ELEMSIZE,(bl_id+1)*(elems/num_blocks)*ELEMSIZE, bytesize, strides_per_thread*elems_per_stride*ELEMSIZE, bytesize/accessing_threads, strides_per_thread, elems_per_stride*ELEMSIZE);

    unsigned int r_sum;  //a nonsense variable used to help defeat optimization
    r_sum = 0;
    extern __shared__ unsigned long long clock_begin;   //clock value kernel marks as its start time
    extern __shared__ unsigned long long clock_now;     //clock value current instant
    //record the kernel start and current times in nanoseconds
    __syncthreads();
    clock_begin = gclock64();
    clock_now = clock_begin;

    /*
    0 = R
    1 = W
    2 = R,W
    3 = W,R
    */
    
    int i,j;    
   while ((clock_now - clock_begin) < (run_time+100)) {
        //ptr = ((bl_id * 32) + lcl_wrp) * (wrp_max * 32) + (ptr * myZero * access_count);
        if(compute) {
            for (i=0; i< ((myZero + 1) << 11); i++) {
                thd_start_ind = i; 
                r_sum += thd_start_ind;
            }
        } else if(rw_mode==0) {
            #pragma unroll
            for (i=0; i<strides_per_thread; i++) {
                r_sum += k_data[thd_start_ind + i * elems_per_stride];
            }
        } else if(rw_mode==1) {
            #pragma unroll
            for (i=0; i<strides_per_thread; i++) {
                k_data[thd_start_ind + i * elems_per_stride] = 7;
            }
        } else if(rw_mode==2) {
            #pragma unroll
            for (i=0; i<strides_per_thread; i++) {
                r_sum += k_data[thd_start_ind + i * elems_per_stride];
                k_data[thd_start_ind + i * elems_per_stride] = 7;
            }
        } else if(rw_mode==3) {
            #pragma unroll
            for (i=0; i<strides_per_thread; i++) {
                k_data[thd_start_ind + i * elems_per_stride] = 7;
                r_sum += k_data[thd_start_ind + i * elems_per_stride + myZero];
            }
        }

        clock_now=gclock64();
   }
        
    __syncthreads();
    k_result[1] = strides_per_thread;
    k_result[2] = thd_start_ind; // Make sure the compiler believes ptr is a result
                     // and does not eliminate references as optimization
    k_result[3] = r_sum;
}

__global__ void
computeKernel(unsigned int *k_ptrs[MAX_SPACES], int *k_result, int bytesize, unsigned long long run_time, int myZero, int comp_instr, int op_val, int COMP_ACCESSES, int fp_per_block)  //c_flush is size of k_data (bytesize) and used to flush cache initially
{
    int smid;
    asm("mov.u32 %0, %smid;" : "=r"(smid));
    // if((threadIdx.y * blockDim.x) + threadIdx.x==0)printf("16?-warp compute block on %d\n",smid);

    // __shared__ unsigned short blk_log[MAX_WARP_LOG];
    unsigned int *k_data;  


    int access_count, bl_id;
    
    bl_id = blockIdx.x;
    
    int total_warps = gridDim.x*blockDim.y;

    // if((threadIdx.y * blockDim.x) + threadIdx.x==0)printf("bl_id %d  (COMPUTE) \n",bl_id);

    // fp_per_block = -1 means don't override instrucion type
    int lcl_thd = (threadIdx.y * blockDim.x) + threadIdx.x; 
    if(fp_per_block>=0 && bl_id==0 && threadIdx.x==0 && threadIdx.y ==0) printf("%d blocks x %d warps/block of memory. %d warps/block are fp.\n", gridDim.x, blockDim.y, fp_per_block);

    if(fp_per_block>=0) {
        if(threadIdx.y<fp_per_block) comp_instr = 2;
        else comp_instr = 8;
    }

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

    __half a = __float2half(0.0f), a_op_add = __float2half(1.1f), a_op_mult = __float2half(1.3f); // 16 bit FP
    float b = 0, b_op_add = 1.1f, b_op_mult=1.3f; // 32 bit FP
    double c = 0, c_op_add = 1.1, c_op_mult=1.3; // 64 bit FP
    int16_t d = 0, d_op_add = (int16_t)(op_val), d_op_mult = int16_t(op_val); // 16 bit INT
    int32_t e = 0, e_op_add = op_val, e_op_mult = op_val; // 32 bit INT
    int64_t f = 0, f_op_add = (int64_t)(op_val), f_op_mult = (int64_t)(op_val); // 64 bit INT

    uint32_t g = 0; // 32 bit uINT
    
        
   while ((clock_now - clock_begin) < (run_time+100)) {

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

        17     16 bit FP DIV

        */

        switch(comp_instr) {
            case 0:
                #pragma unroll 32
                for (access_count = 0; access_count <COMP_ACCESSES; access_count++) { 
                    a+=a_op_add;
                }
                break;

            case 1:
                a = __float2half(1.0f);
                #pragma unroll 32
                for (access_count = 0; access_count <COMP_ACCESSES; access_count++) { 
                    a*=a_op_mult;
                }
                break;

            case 2:
                #pragma unroll 32
                for (access_count = 0; access_count <COMP_ACCESSES; access_count++) { 
                    b+=b_op_add;
                }
                break;

            case 3:
                b = 1.0f;
                #pragma unroll 32
                for (access_count = 0; access_count <COMP_ACCESSES; access_count++) { 
                    b*=b_op_mult;
                }
                break;

            case 4:
                #pragma unroll 32
                for (access_count = 0; access_count <COMP_ACCESSES; access_count++) { 
                    c+=c_op_add;
                }
                break;

            case 5:
                c = 1.0f;
                #pragma unroll 32
                for (access_count = 0; access_count <COMP_ACCESSES; access_count++) { 
                    c*=c_op_mult;
                }
                break;
            
            case 6:
                #pragma unroll 32
                for (access_count = 0; access_count <COMP_ACCESSES; access_count++) { 
                   //d+=d_op_add;
                   asm("add.s16 %0, %0, %1;" : "+h"(d) : "h"(d_op_add));
                }
                break;

            case 7:
                d = d_op_mult;
                #pragma unroll 32
                for (access_count = 0; access_count <COMP_ACCESSES; access_count++) { 
                    //d*=d_op_mult;
                   asm("mul.lo.s16 %0, %0, %1;" : "+h"(d) : "h"(d_op_mult));
                }
                break;

            case 8:
                #pragma unroll 32
                for (access_count = 0; access_count <COMP_ACCESSES; access_count++) { 
                   asm("add.s32 %0, %0, %1;" : "+r"(e) : "r"(e_op_add));
                   // e+=e_op_add;
                }
                break;

            case 9:
                e = e_op_mult;
                #pragma unroll 32
                for (access_count = 0; access_count <COMP_ACCESSES; access_count++) { 
                    e*=e_op_mult;
                }
                break;
            
            case 10:
                #pragma unroll 32
                for (access_count = 0; access_count <COMP_ACCESSES; access_count++) { 
                    //f+=f_op_add;
                   asm("add.s64 %0, %0, %1;" : "+l"(f) : "l"(f_op_add));
                }
                break;

            case 11:
                f = f_op_mult;
                #pragma unroll 32
                for (access_count = 0; access_count <COMP_ACCESSES; access_count++) { 
                    f*=f_op_mult; 
                }
                break;

            case 12:
                e = 931383626;
                #pragma unroll
                for (access_count = 0; access_count <COMP_ACCESSES; access_count++) { 
                    e&=3412319977;
                }
                break;

            case 13:
                g = 931383626;
                #pragma unroll
                for (access_count = 0; access_count <COMP_ACCESSES; access_count++) { 
                    g = __brev(g); // bitwise reverse 
                }
                break;

            case 14:
                b = 23.0f;
                #pragma unroll 32
                for (access_count = 0; access_count <COMP_ACCESSES; access_count++) { 
                    b = __frcp_ru (b); // fp reciprocal (1/x) round up mode
                }
                break;

            case 15:
                e = 0;
                #pragma unroll
                for (access_count = 0; access_count <COMP_ACCESSES; access_count++) { 
                    e = __clz(e); // consecutive high-order zero bits
                }
                break;

            case 16:
                g = 931383626;
                #pragma unroll
                for (access_count = 0; access_count <COMP_ACCESSES; access_count++) { 
                    g = __popc(g); // pop count
                }
                break;

            case 17:
                a = 1;
                #pragma unroll
                for (access_count = 0; access_count <COMP_ACCESSES; access_count++) { 
                    a/=.97;
                }
                break;

            default:
                printf("BAD COMPUTE INSTR TYPE VALUE\n");
                return;
      }
        clock_now=gclock64();
   }

   r_sum = (float)a+b+c+d+e+f+g;

    __syncthreads();
    ptr = ptr + r_sum;       
    k_result[1] = access_count;
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
    return;
    wrp_max = -1; // to compile since we got rid of NUM_BLOCKS
    //wrp_max = (bytesize / TX2_CACHE_LINE) / (NUM_BLOCKS * NUM_WARPS_PER_BL);

    /* Uncomment the following for logging read access times
     * Logging must be coordinated with the launching CUDA program
     */
    unsigned long long cycles_before, cycles_after, before, after;
    unsigned short cycles_add;
    int wrp_log;
    wrp_log =  ((gbl_blk * 32) + lcl_wrp) * wrp_max;

    int ptr = 0;  // holds the index of the next array element to be read
    //unsigned int ptr_start;
    //ptr_start = ((gbl_blk * NUM_WARPS_PER_BL) + lcl_wrp) * (wrp_max * 32) + ((ptr) * myZero * wrp_count);
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
    int flush_max = bytesize / 4;
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
              ptr = ((gbl_blk * 32) + lcl_wrp) * (wrp_max * 32) + (ptr * myZero * wrp_count);
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

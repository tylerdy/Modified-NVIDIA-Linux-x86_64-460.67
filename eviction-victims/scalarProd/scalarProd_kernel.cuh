/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <cooperative_groups.h>

namespace cg = cooperative_groups;

///////////////////////////////////////////////////////////////////////////////
// On G80-class hardware 24-bit multiplication takes 4 clocks per warp
// (the same as for floating point  multiplication and addition),
// whereas full 32-bit multiplication takes 16 clocks per warp.
// So if integer multiplication operands are  guaranteed to fit into 24 bits
// (always lie within [-8M, 8M - 1] range in signed case),
// explicit 24-bit multiplication is preferred for performance.
///////////////////////////////////////////////////////////////////////////////
#define IMUL(a, b) __mul24(a, b)


static __device__ __inline__ unsigned long long int cycles64() {

    unsigned long long int rv;

    asm volatile ( "mov.u64 %0, %%clock64;" : "=l"(rv) );

    return rv;

}


///////////////////////////////////////////////////////////////////////////////
// Calculate scalar products of VectorN vectors of ElementN elements on GPU
// Parameters restrictions:
// 1) ElementN is strongly preferred to be a multiple of warp size to
//    meet alignment constraints of memory coalescing.
// 2) ACCUM_N must be a power of two.
///////////////////////////////////////////////////////////////////////////////
#define ACCUM_N 1024
__global__ void scalarProdGPU(
    float *d_C,
    float *d_A,
    float *d_B,
    // int *d_C,
    // int *d_A,
    // int *d_B,
    int vectorN,
    int elementN,
    int myZero,
    unsigned int *result,
    unsigned int* trash
)
{
    // Handle to thread block group
    cg::thread_block cta = cg::this_thread_block();
    //Accumulators cache
    __shared__ float accumResult[ACCUM_N];
    // __shared__ int accumResult[ACCUM_N];
    unsigned int cnt, rsum;
    float ra, rb;
    // int ra, rb;
    unsigned long long cycles_before, cycles_after,diff;

    cnt = 0;
    rsum = 0;
    ////////////////////////////////////////////////////////////////////////////
    // Cycle through every pair of vectors,
    // taking into account that vector counts can be different
    // from total number of thread blocks
    ////////////////////////////////////////////////////////////////////////////
    for (int vec = blockIdx.x; vec < vectorN; vec += gridDim.x)
    {
        int vec = blockIdx.x;
        int vectorBase = IMUL(elementN, vec);
        int vectorEnd  = vectorBase + elementN;

        ////////////////////////////////////////////////////////////////////////
        // Each accumulator cycles through vectors with
        // stride equal to number of total number of accumulators ACCUM_N
        // At this stage ACCUM_N is only preferred be a multiple of warp size
        // to meet memory coalescing alignment constraints.
        ////////////////////////////////////////////////////////////////////////
        for (int iAccum = threadIdx.x; iAccum < ACCUM_N; iAccum += blockDim.x)
        {
            float sum = 0;
            int iAccum = threadIdx.x;
// #pragma unroll 1
            for (int pos = vectorBase + iAccum; pos < vectorEnd; pos += ACCUM_N){
                int pos = vectorBase+threadIdx.x;
                // cycles_before = clock64();
                ra = d_A[pos];
                // rsum += ra * myZero;
                // cycles_after = clock64();
                // diff = cycles_after-cycles_before;
                // rsum += diff;
                // if(diff > 350) cnt++;
                // if(diff > 0) cnt=diff;
                // cnt++;
                
                // cycles_before = clock64();
                rb = d_B[pos];
                // rsum += rb * myZero+ra;
                // cycles_after = clock64();
                // diff = cycles_after-cycles_before;
                // rsum += diff;
                // if(diff > 350) cnt++;
                // if(diff >0) cnt=diff;
                // cnt++;
                sum += ra * rb;
                // sum += d_A[pos] * d_B[pos];

            }
                // sum +=  __ldcv(&(d_A[pos])) *  __ldcv(&(d_B[pos]));

            accumResult[iAccum] = sum;
        }    

        ////////////////////////////////////////////////////////////////////////
        // Perform tree-like reduction of accumulators' results.
        // ACCUM_N has to be power of two at this stage
        ////////////////////////////////////////////////////////////////////////
        for (int stride = ACCUM_N / 2; stride > 0; stride >>= 1)
        {
            cg::sync(cta);

            for (int iAccum = threadIdx.x; iAccum < stride; iAccum += blockDim.x)
                accumResult[iAccum] += accumResult[stride + iAccum];
        }

        cg::sync(cta);

        if (threadIdx.x == 0) d_C[vec] = accumResult[0];
        // if (threadIdx.x == 0)  __stwt(&(d_C[vec]), accumResult[0]);
    
    }
    // result[blockIdx.x * blockDim.x + threadIdx.x] = cnt;
    // trash[0] = rsum * myZero;
}

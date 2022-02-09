#include <helper_cuda.h>       // helper for checking cuda initialization and error checking
 #include "test.hpp"

#define CACHE_SIZE 2097152 // bytes of 1080 cache

#define HITS false
#define UNIQUE_THREADS false // if true, each thread in a warp accesses a different element

#define ITERS (HITS?128:4)
#define CACHE_MULT (HITS?1:32)



#define CACHE_LINE_SIZE 128


__device__ uint get_smid(void) {

  uint ret;

  asm("mov.u32 %0, %smid;" : "=r"(ret) );

  return ret;

}

__global__ void clock_block(clock_t *d_o, clock_t clock_count)
{

    clock_t start_clock = clock();
    if(get_smid()==19) {
      if(blockIdx.x*blockDim.x+threadIdx.x==0)printf("%d blocks\n",gridDim.x); // TBD this is required for some reason
      clock_count=100000000; // ~50ms
    }
    clock_t clock_offset = 0;
    while (clock_offset < clock_count)
    {
        clock_offset = clock() - start_clock;
    }
     d_o[0] = clock_offset;
}


__global__ void
simpleKernel(float *a, int size, float* res) {
  //if(threadIdx.x==0)printf("launched on SM %d\n",get_smid());
  if(get_smid()!=19) {
    if(threadIdx.x==0)printf("launched on SM %d\n",get_smid());
    return;
  }
  int threads = blockDim.x * gridDim.x / (UNIQUE_THREADS?1:32);
  int els_per_thread = size/CACHE_LINE_SIZE/threads;
  int inds_per_el = CACHE_LINE_SIZE/sizeof(float);
  int id = blockIdx.x*blockDim.x+threadIdx.x / (UNIQUE_THREADS?1:32);
  float sum=0;
  for(int iter=0; iter<ITERS; iter++) {
    for(int i=0;i<els_per_thread;i++) { 
      int ind = i*threads*inds_per_el+id*inds_per_el;
      sum+=a[ind];
    }
  }
  if((int)sum!=els_per_thread*ITERS) *res = sum;
  //if(blockIdx.x*blockDim.x+threadIdx.x==0) printf("sum %f expected %d (%d)\n",sum, els_per_thread*ITERS,(int)sum==els_per_thread*ITERS);
}

int main(int argc, char *argv[])
{ 
  if(argc<2) {
    printf("Enter number of warps.");
    return 0;
  }
  int warps = atoi(argv[1]);
  if(warps>64) {
    printf("Max is 64 warps (1 SM).");
    return 0;
  }
  int blocks = (warps-1)/32+1;
  int warps_per_block = warps/blocks;
  printf("%d blocks. %d warps per block.\n",blocks,warps_per_block);

  int bytesize = CACHE_SIZE * CACHE_MULT;
  float *h_a, *d_a;
  checkCudaErrors(cudaMallocHost(&h_a, bytesize));
  checkCudaErrors(cudaMalloc((void **) &d_a, bytesize));  

  for(int i=0;i<bytesize/sizeof(float);i+=CACHE_LINE_SIZE/sizeof(float)) {
    h_a[i] = 1;
  }

  checkCudaErrors(cudaMemcpy(d_a, h_a, bytesize, cudaMemcpyHostToDevice));

  float *d_res;
  checkCudaErrors(cudaMalloc(&d_res, sizeof(float)));

  cudaStream_t stream[41];
  for (int i = 0; i < 41; i ++)
  {
    checkCudaErrors(cudaStreamCreate(&stream[i]));
  }
  clock_t *d_o;
  checkCudaErrors(cudaMalloc(&d_o, 40*sizeof(clock_t)));
  for(int i=0;i<31;i++) {
    clock_block<<<i<9?2:1,32*32, 0, stream[i]>>>(&d_o[i],1000000000); // ~500ms
  }

  simpleKernel<<<blocks, warps_per_block*32,0,stream[40]>>>(d_a, bytesize, d_res);
  checkCudaErrors(cudaStreamSynchronize(0));
}


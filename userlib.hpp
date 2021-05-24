#include <stdio.h>

#include <cuda.h>

#include <cuda.h>
#include <cuda_runtime.h>


/* Assertion for CUDA functions */
#define gpuErrAssert(ans) gpuAssert((ans), __FILE__, __LINE__, true)
#define gpuErrCheck(ans) gpuAssert((ans), __FILE__, __LINE__, false)

inline int gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUcheck: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort)
          exit(code);
      else
          return -1;
   }
   return 0;
}

/* Forward declarations */
typedef struct allocator allocator_t;

/* Function declarations */
allocator_t *allocator_init(void *buf, size_t size);
void *allocator_alloc(allocator_t *ctx, void* offset);
void allocator_free(allocator_t *ctx);
void allocator_deinit(allocator_t *ctx);
#include <cuda_runtime_api.h>

#define CONTIG_SIZE                  2097152 // TBD this is also set manually in driver code

void *device_allocate_contigous(size_t contiguous_size, void **phy_start_p);
int device_init();

int fgpu_memory_set_contig_info(int device, size_t length, cudaStream_t stream);
int fgpu_memory_allocate(void **p);
int fgpu_memory_free(void *p);
void *fgpu_memory_get_phy_address(void *addr);
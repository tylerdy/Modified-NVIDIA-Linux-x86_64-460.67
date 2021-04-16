#ifndef __TEST_HPP__
#define __TEST_HPP__

#include <cuda_runtime_api.h>

#define CONTIG_SIZE                  4297064448 // TBD this is also set manually in driver code (uvm_pmm_gpu.h)

void *device_allocate_contigous(size_t contiguous_size, void **phy_start_p);
int device_init(bool init_reverse_engineer);

int fgpu_memory_set_contig_info(int device, size_t length, cudaStream_t stream);
int fgpu_memory_allocate(void **p);
int fgpu_memory_free(void *p);
void *fgpu_memory_get_phy_address(void *addr);

#endif /* __TEST_HPP__ */
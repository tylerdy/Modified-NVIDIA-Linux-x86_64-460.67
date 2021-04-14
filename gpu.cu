#include <test.hpp>

/* Stream used for all operations. NULL stream is not used */
cudaStream_t contig_stream;

/* Allocates contiguous memory and returns the starting physical address */
/* For this to work, Nvidia driver must be configured properly. */
void *device_allocate_contigous(size_t contiguous_size, void **phy_start_p)
{
    size_t size = contiguous_size;
    void *gpu_mem;
    void *phy_start;
    int ret;
    
    ret = fgpu_memory_allocate(&gpu_mem);
    if (ret < 0)
        return NULL;
   
    phy_start = fgpu_memory_get_phy_address(gpu_mem);
    if (!phy_start)
        return NULL;

    *phy_start_p = phy_start;

    return gpu_mem;
}

int device_init()
{
    cudaFree(0);
  return fgpu_memory_set_contig_info(0, CONTIG_SIZE, 0); // use default stream for now
}
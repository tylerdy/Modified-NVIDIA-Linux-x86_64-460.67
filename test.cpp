#include <stdio.h>
#include <cstdint>

#include <test.hpp>

int main(int argc, char *argv[])
{
  int ret;
  void *virt_start;
  void *phy_start;

  ret = device_init(false);
  if (ret < 0)
        fprintf(stderr, "Device init failed\n");

  virt_start = device_allocate_contigous(CONTIG_SIZE, &phy_start);

  fprintf(stdout, "virt %p phys %p\n", virt_start, phy_start);

  uintptr_t start = (uintptr_t) virt_start;
  uintptr_t next = start + CONTIG_SIZE/2;

  fprintf(stdout, "first chunk addr val %d, second chunk addr val %d\n",*(int*)start, *(int*)next);
}
#include <stdio.h>
#include <cstdint>

#include <test.hpp>

int main(int argc, char *argv[])
{
  int ret;
  void *virt_start;
  void *phy_start;

  ret = device_init(true);
  if (ret < 0)
        fprintf(stderr, "Device init failed\n");

  return 0;

  virt_start = device_allocate_contigous(CONTIG_SIZE, &phy_start);

  fprintf(stdout, "virt %p phys %p\n", virt_start, phy_start);

  return ret;
}
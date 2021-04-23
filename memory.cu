#include <assert.h>
#include <dlfcn.h>
#include <errno.h>
#include <iostream>
#include <inttypes.h>
#include <linux/ioctl.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <unistd.h>

/* CUDA/NVML */
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nvml.h>
#include <driver_types.h>

/* NVIDIA driver */
//#include <uvm_minimal_init.h>
//#include <nvCpuUuid.h>

#include <userlib.hpp>

#include <uvm_ioctl.h>

#define NVIDIA_UVM_DEVICE_PATH  "/dev/nvidia-uvm" // tbd fix
/* TODO: This path can be changed via environment variable */
#define NVIDIA_MPS_CONTROL_PATH "/tmp/nvidia-mps/control"

/* Ioctl codes */
#define IOCTL_SET_PROCESS_CONTIG_INFO    _IOC(0, 0, UVM_SET_PROCESS_CONTIG_INFO, 0)
#define IOCTL_GET_PROCESS_CONTIG_INFO    _IOC(0, 0, UVM_GET_PROCESS_CONTIG_INFO, 0)

/* UVM device fd */
static int g_uvm_fd = -1;

typedef int (*orig_open_f_type)(const char *pathname, int flags, int mode);
orig_open_f_type g_orig_open;

typedef int (*orig_connect_f_type)(int sockfd, const struct sockaddr *addr,
                   socklen_t addrlen);
orig_connect_f_type g_orig_connect;

pthread_once_t g_pre_init_once = PTHREAD_ONCE_INIT;
pthread_once_t g_post_init_once = PTHREAD_ONCE_INIT;
bool g_init_failed;

/* All information needed for tracking memory */
struct {
    bool is_initialized;

    /* Start physical address of allocation */
    void *base_phy_addr;

    /* Actual memory available for coloring */
    size_t reserved_len;

    /* Actual memory allocation */
    void *base_addr;

    allocator_t *allocator;

} g_memory_ctx;

/* Does the most neccesary initialization */
static void pre_initialization(void)
{
    g_orig_open = (orig_open_f_type)dlsym(RTLD_NEXT,"open");
    if (!g_orig_open) {
        g_init_failed = true;
        return;
    }

    g_orig_connect = (orig_connect_f_type)dlsym(RTLD_NEXT,"connect");
    if (!g_orig_connect) {
        g_init_failed = true;
        return;
    }
}

static void post_initialization(void)
{
    nvmlReturn_t ncode;

    ncode = nvmlInit();
    if (ncode != NVML_SUCCESS) {
        g_init_failed = true;
        return;
    }
}

/* Does the initialization atmost once */
static int init(bool do_post_init)
{
    int ret;

    ret = pthread_once(&g_pre_init_once, pre_initialization);
    if (ret < 0)
        return ret;

    if (g_init_failed) {
        fprintf(stderr, "FGPU:Initialization failed\n");
        return -EINVAL;
    }
    
    if (!do_post_init)
        return 0;

    ret = pthread_once(&g_post_init_once, post_initialization);
    if (ret < 0)
        return ret;
    
    if (g_init_failed) {
        fprintf(stderr, "FGPU:Initialization failed\n");
        return -EINVAL;
    }
    fprintf(stdout,"returning 0\n");
    return 0;
}

/* Retrieve the device UUID from the CUDA device handle */
static int get_device_UUID(int device, NvProcessorUuid *uuid)
{
    nvmlReturn_t ncode;
    cudaError_t ccode;
    char pciID[32];
    nvmlDevice_t handle;
    char buf[100];
    char hex[3];
    char *nbuf;
    int cindex, hindex, uindex, needed_bytes;
    char c;
    int len;
    std::string prefix = "GPU";
    const char *gpu_prefix = prefix.c_str();
    int gpu_prefix_len = strlen(gpu_prefix);

    /* Get PCI ID from the device handle and then use NVML library to get UUID */
    ccode = cudaDeviceGetPCIBusId(pciID, sizeof(pciID), device);
    if (ccode != cudaSuccess) {
        fprintf(stderr, "FGPU:Couldn't find PCI Bus ID\n");
        return -EINVAL;
    }

    ncode = nvmlDeviceGetHandleByPciBusId(pciID, &handle);
    if (ncode != NVML_SUCCESS){
        fprintf(stderr, "FGPU:Couldn't get Device Handle\n");
        return -EINVAL;
    }

     
    ncode = nvmlDeviceGetUUID(handle, buf, sizeof(buf));
    if (ncode != NVML_SUCCESS){
        fprintf(stderr, "FGPU:Couldn't find device UUID\n");
        return -EINVAL;
    }

    if (strncmp(buf, gpu_prefix, gpu_prefix_len != 0))
        return 0;

    nbuf = buf + gpu_prefix_len;

    /*
     * UUID has characters and hexadecimal numbers. 
     * We are only interested in hexadecimal numbers.
     * Each hexadecimal numbers is equal to 1 byte.
     */
    needed_bytes = sizeof(NvProcessorUuid);
    len = strlen(nbuf);

    for (cindex = 0, hindex = 0, uindex = 0; cindex < len; cindex++) {
        c = nbuf[cindex];
        if ((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') || (c >= '0' && c <= '9')) {
            hex[hindex] = c;
            hindex++;
            if (hindex == 2) {
                hex[2] = '\0';
                uuid->uuid[uindex] = (uint8_t)strtol(hex, NULL, 16);
                uindex++;
                hindex = 0;
                if (uindex > needed_bytes) {
                    fprintf(stderr, "FGPU:Invalid device UUID\n");
                    return -EINVAL;
                }  
            }
        }
    }

    if (uindex != needed_bytes) {
        fprintf(stderr, "FGPU:Invalid device UUID\n");
        return -EINVAL;
    }

    return 0;
}

extern "C" {

/* Trap open() calls (interested in UVM device opened by CUDA) */
int open(const char *pathname, int flags, int mode)
{
    int ret;

    ret = init(false);
    if (ret < 0)
        return ret;
    
    ret = g_orig_open(pathname,flags, mode);

    if (g_uvm_fd < 0 && 
            strncmp(pathname, NVIDIA_UVM_DEVICE_PATH, strlen(NVIDIA_UVM_DEVICE_PATH)) == 0) {
        g_uvm_fd = ret;
    }
    return ret;
}

/* Trap connect() calls (interested in connection to MPS) */
int connect(int sockfd, const struct sockaddr *addr,
                   socklen_t addrlen)
{
    int ret;

    ret = init(false);
    if (ret < 0)
        return ret;
    
    ret = g_orig_connect(sockfd, addr, addrlen);

    if (ret >= 0 && g_uvm_fd < 0 && addr && addr->sa_family == AF_LOCAL && 
            strncmp(addr->sa_data, NVIDIA_MPS_CONTROL_PATH, strlen(NVIDIA_MPS_CONTROL_PATH)) == 0) {
        g_uvm_fd = sockfd;
    }
    return ret;
}

} /* extern "C" */

static int set_process_contig_info(int device, size_t req_length,
  cudaStream_t stream)
{
UVM_SET_PROCESS_CONTIG_INFO_PARAMS params;
size_t actual_length = req_length; // TBD change to param
int ret;

/* Con can only be set once */
if (g_memory_ctx.is_initialized) {
  fprintf(stderr, "FGPU:Process color already set\n");
  return -EINVAL;
}

ret = get_device_UUID(device, &params.destinationUuid);
if (ret < 0) {
    fprintf(stderr, "Failed to get device UUID\n");
    return ret;
}

params.length = actual_length;

ret = ioctl(g_uvm_fd, IOCTL_SET_PROCESS_CONTIG_INFO, &params);
if (ret < 0) {
    fprintf(stderr, "Set process contig ioctl failed\n");
    return ret;
}

if (params.rmStatus != NV_OK) {
  fprintf(stderr, "FGPU:Couldn't set process color property\n");
  return -EINVAL;
}
fprintf(stdout, "about to alloc contig range\n");
ret = gpuErrCheck(cudaMallocManaged(&g_memory_ctx.base_addr, actual_length));
if (ret < 0)
  return ret;

/* Do the actual allocation on device */
ret = gpuErrCheck(cudaMemPrefetchAsync(g_memory_ctx.base_addr, actual_length,
          device, stream));
if (ret < 0) {
  cudaFree(g_memory_ctx.base_addr);
  return ret;
}

ret = gpuErrCheck(cudaStreamSynchronize(stream));
if (ret < 0) {
cudaFree(g_memory_ctx.base_addr);
  return ret;
}

g_memory_ctx.is_initialized = true;
g_memory_ctx.base_phy_addr = (void *)params.address;
fprintf(stdout, "paramsaddress is %llx, basephysaddr (void*) is %p\n", params.address, g_memory_ctx.base_phy_addr);
g_memory_ctx.reserved_len = req_length;

g_memory_ctx.allocator = allocator_init(g_memory_ctx.base_addr, 
      req_length);
if (!g_memory_ctx.allocator) {
  fprintf(stderr, "FGPU:Allocator Initialization Failed\n");
  return -EINVAL;
}
/*UVM_GET_PROCESS_CONTIG_INFO_PARAMS get_params;
get_params.destinationUuid = params.destinationUuid;

ret = ioctl(g_uvm_fd, IOCTL_GET_PROCESS_CONTIG_INFO, &get_params);
if (ret < 0)
  return ret;

fprintf(stdout, "Get contig info returned virt %llu and phys %llu. Allocated address is %llu\n",get_params.virt_start, get_params.phys_start, g_memory_ctx.allocator);*/

return 0;
}

/* Indicates the color set currently for the process and the length reserved */
int fgpu_memory_set_contig_info(int device, size_t length,
  cudaStream_t stream)
{
int ret;

ret = init(true);
if (ret < 0)
  return ret;

if (g_uvm_fd < 0) {
  fprintf(stderr, "FGPU:Initialization not done\n");
  return -EBADF;
}

return set_process_contig_info(device, length, stream);
}

void fgpu_memory_deinit(void)
{
    if (!g_memory_ctx.is_initialized)
        return;

    if (g_memory_ctx.allocator)
        allocator_deinit(g_memory_ctx.allocator);

    cudaFree(g_memory_ctx.base_addr);

    g_memory_ctx.is_initialized = false;
}

int fgpu_memory_allocate(void **p)
{
    void *ret_addr;

    if (!g_memory_ctx.is_initialized) {
        fprintf(stderr, "FGPU:Initialization not done\n");
        return -EBADF;
    }


    ret_addr = allocator_alloc(g_memory_ctx.allocator);
    if (!ret_addr) {
        fprintf(stderr, "FGPU:Can't allocate device memory\n");
        return -ENOMEM;
    }

    *p = ret_addr;
    
    return 0;
}

int fgpu_memory_free()
{
    if (!g_memory_ctx.is_initialized) {
        fprintf(stderr, "FGPU:Initialization not done\n");
        return -EBADF;
    }

    allocator_free(g_memory_ctx.allocator);

    return 0;
}

/* Useful for only reverse engineering */
void *fgpu_memory_get_phy_address(void *addr)
{
    if (!g_memory_ctx.base_phy_addr)
        return NULL;

    return (void *)((uintptr_t)g_memory_ctx.base_phy_addr + 
            (uintptr_t)addr - (uintptr_t)g_memory_ctx.base_addr);
}
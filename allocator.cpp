/* 
 * This is a custom allocator to manage memory of a given buffer
 * without read/writing on the buffer.
 * Currently it is a naive implementation using 'next-fit' logic (single threaded)
 */
#include <assert.h>
#include <inttypes.h>
#include <stdio.h>

#include <userlib.hpp>

/* Represents the context saved for an allocator */
typedef struct allocator {
    void *start_address;
    size_t size;
    bool free;
} allocator_t;

/* 
 * Initializes an allocator context. 
 * Buf points to the start address.
 * Size denotes total size of buffer.
 * Alignment denotes the alignment of all subsequent allocations.
 * Returns 0 on success, otherwise failure
 */
allocator_t *allocator_init(void *buf, size_t size)
{
    uintptr_t start_address, round_address;
    allocator_t *ctx;

    start_address = (uintptr_t)buf;

    ctx = new allocator_t;

    ctx->size = size;
    ctx->start_address = buf;
    ctx->free = true;

    return ctx;
}

/* Allocated a buffer and returns it */
void *allocator_alloc(allocator_t *ctx, void* offset)
{
    ctx->free = false;
    void* addr = ctx->start_address + (uint64_t)offset;
    // printf("addr: %016x\n", addr);
    return addr;
}

/* Frees up a buffer */
void allocator_free(allocator_t *ctx)
{
    ctx->free = true;
}

/* Frees up the allocator */
void allocator_deinit(allocator_t *ctx)
{
    delete ctx;
}

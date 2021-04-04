/*******************************************************************************
    Copyright (c) 2016-2019 NVIDIA Corporation

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to
    deal in the Software without restriction, including without limitation the
    rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
    sell copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be
        included in all copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
    THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
    FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
    DEALINGS IN THE SOFTWARE.

*******************************************************************************/

#include "uvm_common.h"
#include "uvm_ioctl.h"
#include "uvm_linux.h"
#include "uvm_global.h"
#include "uvm_gpu.h"
#include "uvm_lock.h"
#include "uvm_va_space.h"
#include "uvm_va_range.h"
#include "uvm_va_block.h"
#include "uvm_tracker.h"
#include "uvm_api.h"
#include "uvm_channel.h"
#include "uvm_push.h"
#include "uvm_hal.h"
#include "uvm_tools.h"
#include "uvm_migrate.h"
#include "uvm_migrate_pageable.h"
#include "nv_speculation_barrier.h"

typedef enum
{
    UVM_MIGRATE_PASS_FIRST,
    UVM_MIGRATE_PASS_SECOND
} uvm_migrate_pass_t;

static int uvm_perf_migrate_cpu_preunmap_enable = 1;
module_param(uvm_perf_migrate_cpu_preunmap_enable, int, S_IRUGO);

#define UVM_PERF_MIGRATE_CPU_PREUNMAP_BLOCK_ORDER_DEFAULT 2
#define UVM_PERF_MIGRATE_CPU_PREUNMAP_BLOCK_ORDER_MAX     10
static unsigned uvm_perf_migrate_cpu_preunmap_block_order = UVM_PERF_MIGRATE_CPU_PREUNMAP_BLOCK_ORDER_DEFAULT;
module_param(uvm_perf_migrate_cpu_preunmap_block_order, uint, S_IRUGO);

// Global post-processed values of the module parameters
static bool g_uvm_perf_migrate_cpu_preunmap_enable __read_mostly;
static NvU64 g_uvm_perf_migrate_cpu_preunmap_size __read_mostly;

static bool is_migration_single_block(uvm_va_range_t *first_va_range, NvU64 base, NvU64 length)
{
    NvU64 end = base + length - 1;

    if (end > first_va_range->node.end)
        return false;

    return uvm_va_range_block_index(first_va_range, base) == uvm_va_range_block_index(first_va_range, end);
}

static NV_STATUS block_migrate_map_mapped_pages(uvm_va_block_t *va_block,
                                                uvm_va_block_retry_t *va_block_retry,
                                                uvm_va_block_context_t *va_block_context,
                                                uvm_va_block_region_t region,
                                                uvm_processor_id_t dest_id)
{
    uvm_prot_t prot;
    uvm_page_index_t page_index;
    NV_STATUS status = NV_OK;
    const uvm_page_mask_t *pages_mapped_on_destination = uvm_va_block_map_mask_get(va_block, dest_id);

    for (prot = UVM_PROT_READ_ONLY; prot <= UVM_PROT_READ_WRITE_ATOMIC; ++prot)
        va_block_context->mask_by_prot[prot - 1].count = 0;

    // Only map those pages that are not already mapped on destination
    for_each_va_block_unset_page_in_region_mask(page_index, pages_mapped_on_destination, region) {
        prot = uvm_va_block_page_compute_highest_permission(va_block, dest_id, page_index);
        UVM_ASSERT(prot != UVM_PROT_NONE);

        if (va_block_context->mask_by_prot[prot - 1].count++ == 0)
            uvm_page_mask_zero(&va_block_context->mask_by_prot[prot - 1].page_mask);

        uvm_page_mask_set(&va_block_context->mask_by_prot[prot - 1].page_mask, page_index);
    }

    for (prot = UVM_PROT_READ_ONLY; prot <= UVM_PROT_READ_WRITE_ATOMIC; ++prot) {
        if (va_block_context->mask_by_prot[prot - 1].count == 0)
            continue;

        // We pass UvmEventMapRemoteCauseInvalid since the destination processor
        // of a migration will never be mapped remotely
        status = uvm_va_block_map(va_block,
                                  va_block_context,
                                  dest_id,
                                  region,
                                  &va_block_context->mask_by_prot[prot - 1].page_mask,
                                  prot,
                                  UvmEventMapRemoteCauseInvalid,
                                  &va_block->tracker);
        if (status != NV_OK)
            break;

        // Whoever added the other mapping(s) should have already added
        // SetAccessedBy processors
    }

    return status;
}

static NV_STATUS block_migrate_map_unmapped_pages(uvm_va_block_t *va_block,
                                                  uvm_va_block_retry_t *va_block_retry,
                                                  uvm_va_block_context_t *va_block_context,
                                                  uvm_va_block_region_t region,
                                                  uvm_processor_id_t dest_id)

{
    uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
    NV_STATUS status = NV_OK;
    NV_STATUS tracker_status;

    // Save the mask of unmapped pages because it will change after the
    // first map operation
    uvm_page_mask_complement(&va_block_context->caller_page_mask, &va_block->maybe_mapped_pages);

    // Only map those pages that are not mapped anywhere else (likely due
    // to a first touch or a migration). We pass
    // UvmEventMapRemoteCauseInvalid since the destination processor of a
    // migration will never be mapped remotely.
    status = uvm_va_block_map(va_block,
                              va_block_context,
                              dest_id,
                              region,
                              &va_block_context->caller_page_mask,
                              UVM_PROT_READ_WRITE_ATOMIC,
                              UvmEventMapRemoteCauseInvalid,
                              &local_tracker);
    if (status != NV_OK)
        goto out;

    // Add mappings for AccessedBy processors
    //
    // No mappings within this call will operate on dest_id, so we don't
    // need to acquire the map operation above.
    status = uvm_va_block_add_mappings_after_migration(va_block,
                                                       va_block_context,
                                                       dest_id,
                                                       dest_id,
                                                       region,
                                                       &va_block_context->caller_page_mask,
                                                       UVM_PROT_READ_WRITE_ATOMIC,
                                                       NULL);

out:
    tracker_status = uvm_tracker_add_tracker_safe(&va_block->tracker, &local_tracker);
    uvm_tracker_deinit(&local_tracker);
    return status == NV_OK ? tracker_status : status;
}

// Pages that are not mapped anywhere can be safely mapped with RWA permission.
// The rest of pages need to individually compute the maximum permission that
// does not require a revocation.
static NV_STATUS block_migrate_add_mappings(uvm_va_block_t *va_block,
                                            uvm_va_block_retry_t *va_block_retry,
                                            uvm_va_block_context_t *va_block_context,
                                            uvm_va_block_region_t region,
                                            uvm_processor_id_t dest_id)

{
    NV_STATUS status;

    status = block_migrate_map_unmapped_pages(va_block,
                                              va_block_retry,
                                              va_block_context,
                                              region,
                                              dest_id);
    if (status != NV_OK)
        return status;

    return block_migrate_map_mapped_pages(va_block,
                                          va_block_retry,
                                          va_block_context,
                                          region,
                                          dest_id);
}

NV_STATUS uvm_va_block_migrate_locked(uvm_va_block_t *va_block,
                                      uvm_va_block_retry_t *va_block_retry,
                                      uvm_va_block_context_t *va_block_context,
                                      uvm_va_block_region_t region,
                                      uvm_processor_id_t dest_id,
                                      uvm_migrate_mode_t mode,
                                      uvm_tracker_t *out_tracker)
{
    NV_STATUS status, tracker_status = NV_OK;
    uvm_va_range_t *va_range = va_block->va_range;

    uvm_assert_mutex_locked(&va_block->lock);

    if (uvm_va_range_is_read_duplicate(va_range)) {
        status = uvm_va_block_make_resident_read_duplicate(va_block,
                                                           va_block_retry,
                                                           va_block_context,
                                                           dest_id,
                                                           region,
                                                           NULL,
                                                           NULL,
                                                           UVM_MAKE_RESIDENT_CAUSE_API_MIGRATE);
    }
    else {
        status = uvm_va_block_make_resident(va_block,
                                            va_block_retry,
                                            va_block_context,
                                            dest_id,
                                            region,
                                            NULL,
                                            NULL,
                                            UVM_MAKE_RESIDENT_CAUSE_API_MIGRATE);
    }

    if (status == NV_OK && mode == UVM_MIGRATE_MODE_MAKE_RESIDENT_AND_MAP) {
        // block_migrate_add_mappings will acquire the work from the above
        // make_resident call and update the VA block tracker.
        status = block_migrate_add_mappings(va_block, va_block_retry, va_block_context, region, dest_id);
    }

    if (out_tracker)
        tracker_status = uvm_tracker_add_tracker_safe(out_tracker, &va_block->tracker);

    return status == NV_OK ? tracker_status : status;
}

// Unmapping CPU pages on P9 systems is very costly, to the point that it
// becomes the bottleneck of UvmMigrate. We have measured up to 3x lower BW for
// migrations that need to remove CPU mappings compared to migrations that only
// create CPU mappings. The overhead can be fully attributed to the TLB
// shootdown. When a CPU page is unmapped, it needs to (1) invalidate any copy
// in the P9 cores, and (2) if ATS is enabled, issue ATSD messages over NVLINK
// to remove the corresponding entries in the GPUs' TLBs. ATSDs are not even
// required when migration managed memory since UVM ensures that there are no
// ATS entries cached in the GPU TLBs for the managed VA ranges. However, we
// don't have a way to skip them as of today.
//
// In order to minimize the overhead of CPU unmaps during UvmMigrate we try to
// call unmap_mapping_range on VA regions larger than the VA block granularity
// before the actual migration so that TLB invalidations are batched better by
// the OS. This also has an impact in the number of ATSD messages issued. This
// is because the NPU code uses MMU notifiers in order to get a callback
// (invalidate_range) when a TLB invalidation is required. Fortunately, this
// callback is not called if there is nothing to be invalidated. Therefore, if
// we issue a large unmap, subsequent unmaps within that region will not invoke
// the callback.
//
// However, due to (1), even issuing a single invalidate for the whole migrated
// range introduces a noticeable overhead (20-30%) on systems with 3xNVLINK2.
// This is only expected to get worse if CPU-GPU interconnects' BW keeps
// increasing.
//
// Thus, VA range migrations are split into groups of contiguous VA blocks, and
// trigger a single pre-unmap of the group of VA blocks in the Linux kernel
// before the VA blocks' migration starts. This way, we trigger larger (more
// efficient) TLB invalidations than when we do it one VA block a time, while
// still being able to pipeline the migration, which allows to hide most of the
// costs of (1).
//
// However, there are some cases in which the CPU has mappings to the pages
// being migrated but they don't need to be removed (which can introduce
// unnecessary CPU faults later on). Therefore, we skip the pre-unmap step
// under the following conditions:
// - Pages mapped by the CPU that are *already* in the destination.
// - Pages mapped by the CPU that are *not* in the destination but
// read-duplication is enabled in the VA range.

// This function checks if the pre-unmap optimization is required given the
// system capabilities and the destination of the migration. This is to skip
// any subsequent checks required by the optimization, which can be costly.
//
// The current logic checks that:
// - We are in the first pass of the migration (see the explanation of the
// two-pass strategy in uvm_migrate).
// - The CPU has an NVLINK interconnect to the GPUs. Otherwise, we don't
// need this optimization since we are already limited by PCIe BW.
// - If the migration spans several VA blocks, otherwise skip the preunmap to
// avoid the overhead.
static bool migration_should_do_cpu_preunmap(uvm_va_space_t *va_space,
                                             uvm_migrate_pass_t pass,
                                             bool is_single_block)

{
    if (!g_uvm_perf_migrate_cpu_preunmap_enable)
        return false;

    if (pass != UVM_MIGRATE_PASS_FIRST || is_single_block)
        return false;

    if (uvm_processor_mask_get_gpu_count(&va_space->has_nvlink[UVM_ID_CPU_VALUE]) == 0)
        return false;

    return true;
}

// This function determines if the VA range properties avoid the need to remove
// CPU mappings on UvmMigrate. Currently, it only checks whether
// read-duplication is enabled in the VA range. This is because, when migrating
// read-duplicated VA blocks, the source processor doesn't need to be unmapped
// (though it may need write access revoked).
static bool va_range_should_do_cpu_preunmap(uvm_va_range_t *va_range)
{
    return !uvm_va_range_is_read_duplicate(va_range);
}

// Function that determines if the VA block to be migrated contains pages with
// CPU mappings that don't need to be removed (see the comment above). In that
// case false is returned. Otherwise it returns true, and stores in the
// variable pointed by num_unmap_pages the number of pages that do need to
// remove their CPU mappings.
static bool va_block_should_do_cpu_preunmap(uvm_va_block_t *va_block,
                                            uvm_va_block_context_t *va_block_context,
                                            NvU64 start,
                                            NvU64 end,
                                            uvm_processor_id_t dest_id,
                                            NvU32 *num_unmap_pages)
{
    const uvm_page_mask_t *mapped_pages_cpu;
    NvU32 num_cpu_unchanged_pages = 0;
    uvm_va_block_region_t region;

    *num_unmap_pages = 0;

    if (!va_block)
        return true;

    UVM_ASSERT(va_range_should_do_cpu_preunmap(va_block->va_range));

    region = uvm_va_block_region_from_start_end(va_block, max(start, va_block->start), min(end, va_block->end));

    uvm_mutex_lock(&va_block->lock);

    mapped_pages_cpu = uvm_va_block_map_mask_get(va_block, UVM_ID_CPU);
    if (uvm_processor_mask_test(&va_block->resident, dest_id)) {
        const uvm_page_mask_t *resident_pages_dest = uvm_va_block_resident_mask_get(va_block, dest_id);
        uvm_page_mask_t *do_not_unmap_pages = &va_block_context->scratch_page_mask;

        // TODO: Bug 1877578
        //
        // We assume that if pages are mapped on the CPU and not resident on
        // the destination, the pages will change residency so the CPU must be
        // unmapped. If we implement automatic read-duplication heuristics in
        // the future, we'll also need to check if the pages are being
        // read-duplicated.
        uvm_page_mask_and(do_not_unmap_pages, mapped_pages_cpu, resident_pages_dest);

        num_cpu_unchanged_pages = uvm_page_mask_region_weight(do_not_unmap_pages, region);
    }

    *num_unmap_pages = uvm_page_mask_region_weight(mapped_pages_cpu, region) - num_cpu_unchanged_pages;

    uvm_mutex_unlock(&va_block->lock);

    return num_cpu_unchanged_pages == 0;
}

static void preunmap_multi_block(uvm_va_range_t *va_range,
                                 uvm_va_block_context_t *va_block_context,
                                 NvU64 start,
                                 NvU64 end,
                                 uvm_processor_id_t dest_id)
{
    size_t i;
    const size_t first_block_index = uvm_va_range_block_index(va_range, start);
    const size_t last_block_index = uvm_va_range_block_index(va_range, end);
    NvU32 num_unmap_pages = 0;

    UVM_ASSERT(start >= va_range->node.start);
    UVM_ASSERT(end  <= va_range->node.end);
    UVM_ASSERT(va_range->type == UVM_VA_RANGE_TYPE_MANAGED);
    uvm_assert_rwsem_locked(&va_range->va_space->lock);

    UVM_ASSERT(uvm_range_group_all_migratable(va_range->va_space, start, end));

    for (i = first_block_index; i <= last_block_index; i++) {
        NvU32 num_block_unmap_pages;

        if (!va_block_should_do_cpu_preunmap(uvm_va_range_block(va_range, i),
                                             va_block_context,
                                             start,
                                             end,
                                             dest_id,
                                             &num_block_unmap_pages)) {
            return;
        }

        num_unmap_pages += num_block_unmap_pages;
    }

    if (num_unmap_pages > 0)
        unmap_mapping_range(&va_range->va_space->mapping, start, end - start + 1, 1);
}

static NV_STATUS uvm_va_range_migrate_multi_block(uvm_va_range_t *va_range,
                                                  uvm_va_block_context_t *va_block_context,
                                                  NvU64 start,
                                                  NvU64 end,
                                                  uvm_processor_id_t dest_id,
                                                  uvm_migrate_mode_t mode,
                                                  uvm_tracker_t *out_tracker)
{
    size_t i;
    const size_t first_block_index = uvm_va_range_block_index(va_range, start);
    const size_t last_block_index = uvm_va_range_block_index(va_range, end);

    UVM_ASSERT(start >= va_range->node.start);
    UVM_ASSERT(end  <= va_range->node.end);
    UVM_ASSERT(va_range->type == UVM_VA_RANGE_TYPE_MANAGED);
    uvm_assert_rwsem_locked(&va_range->va_space->lock);

    UVM_ASSERT(uvm_range_group_all_migratable(va_range->va_space, start, end));

    // Iterate over blocks, populating them if necessary
    for (i = first_block_index; i <= last_block_index; i++) {
        uvm_va_block_retry_t va_block_retry;
        uvm_va_block_region_t region;
        uvm_va_block_t *va_block;
        NV_STATUS status = uvm_va_range_block_create(va_range, i, &va_block);

        if (status != NV_OK)
            return status;

        region = uvm_va_block_region_from_start_end(va_block,
                                                    max(start, va_block->start),
                                                    min(end, va_block->end));

        status = UVM_VA_BLOCK_LOCK_RETRY(va_block, &va_block_retry,
                                         uvm_va_block_migrate_locked(va_block,
                                                                     &va_block_retry,
                                                                     va_block_context,
                                                                     region,
                                                                     dest_id,
                                                                     mode,
                                                                     out_tracker));
        if (status != NV_OK)
            return status;
    }

    return NV_OK;
}

static NV_STATUS uvm_va_range_migrate(uvm_va_range_t *va_range,
                                      uvm_va_block_context_t *va_block_context,
                                      NvU64 start,
                                      NvU64 end,
                                      uvm_processor_id_t dest_id,
                                      uvm_migrate_mode_t mode,
                                      bool should_do_cpu_preunmap,
                                      uvm_tracker_t *out_tracker)
{
    NvU64 preunmap_range_start = start;

    should_do_cpu_preunmap = should_do_cpu_preunmap && va_range_should_do_cpu_preunmap(va_range);

    // Divide migrations into groups of contiguous VA blocks. This is to trigger
    // CPU unmaps for that region before the migration starts.
    while (preunmap_range_start < end) {
        NV_STATUS status;
        NvU64 preunmap_range_end;

        if (should_do_cpu_preunmap) {
            preunmap_range_end = UVM_ALIGN_UP(preunmap_range_start + 1, g_uvm_perf_migrate_cpu_preunmap_size);
            preunmap_range_end = min(preunmap_range_end - 1, end);

            preunmap_multi_block(va_range,
                                 va_block_context,
                                 preunmap_range_start,
                                 preunmap_range_end,
                                 dest_id);
        }
        else {
            preunmap_range_end = end;
        }

        status = uvm_va_range_migrate_multi_block(va_range,
                                                  va_block_context,
                                                  preunmap_range_start,
                                                  preunmap_range_end,
                                                  dest_id,
                                                  mode,
                                                  out_tracker);
        if (status != NV_OK)
            return status;

        preunmap_range_start = preunmap_range_end + 1;
    }

    return NV_OK;
}

static void uvm_block_iter_deinitialization(uvm_block_iter_t *iter)
{
    struct page *page;
    NvU32 i;

    if (iter->cpu_block) {
        
        if (iter->cpu_block->cpu.pages) {

            // Release all pinned pages if any
            for (i = 0; i < PAGES_PER_UVM_VA_BLOCK; i++) {
                page = iter->cpu_block->cpu.pages[i];
                if (page) {
                    put_page(page);
                    iter->cpu_block->cpu.pages[i]= NULL;
                }
            }

            kfree(iter->cpu_block->cpu.pages);
            iter->cpu_block->cpu.pages = NULL;
        }

        kfree(iter->cpu_block);
        iter->cpu_block = NULL;
    }
}

// Iterate over all managed contiguous va_blocks till "length" is covered
// Length is in terms of color mem size
static NV_STATUS uvm_block_iter_initialization(uvm_va_space_t *va_space,
                                                NvU64 start,
                                                uvm_processor_id_t id,
                                                uvm_block_iter_t *iter)
{
    NV_STATUS status = NV_OK;
    uvm_va_range_t *first_va_range;
    size_t block_index, range_end_block_index;
    struct page **page_array;

    uvm_assert_rwsem_locked(&va_space->lock);

    iter->start = start;
    iter->va_range = NULL;
    iter->cpu_block = NULL;

    first_va_range = uvm_va_space_iter_first(va_space, start, start);

    // If block not exists in uvm but lies on CPU, maybe it is backed by linux
    // pages
    // TODO: It can be that the start block is not within any range but
    // the subsequent blocks might be. We need to handle this behaviour
    if (!first_va_range & (id.val == UVM_CPU_ID)) {

        iter->cpu_block = kmalloc(sizeof(uvm_va_block_t), GFP_KERNEL);
        if (!iter->cpu_block) {
            status = NV_ERR_NO_MEMORY;
            goto err;
        }

        iter->cpu_block->is_linux_backed = true;

        page_array = kzalloc(sizeof(struct page *) * PAGES_PER_UVM_VA_BLOCK, GFP_KERNEL);
        if (!page_array) {
            status = NV_ERR_NO_MEMORY;
            goto err;
        }

        iter->cpu_block->cpu.pages = page_array;

        iter->next_block_index = start / UVM_VA_BLOCK_SIZE;
        iter->range_end_block_index = (size_t)-1;
        return NV_OK;
    }

    if (!first_va_range || first_va_range->type != UVM_VA_RANGE_TYPE_MANAGED) {
        status = NV_ERR_INVALID_ADDRESS;
        goto err;
    }

    block_index = uvm_va_range_block_index(first_va_range, max(start, first_va_range->node.start));
    range_end_block_index = uvm_va_range_block_index(first_va_range, first_va_range->node.end);

    iter->va_range = first_va_range;
    iter->next_block_index = block_index;
    iter->range_end_block_index = range_end_block_index;    

    return NV_OK;

err:
    uvm_block_iter_deinitialization(iter);
    return status;
}

static NV_STATUS uvm_block_iter_next_block(uvm_block_iter_t *iter,
                                uvm_va_block_t **out_block)
{
    NV_STATUS status = NV_OK;

    if (iter->cpu_block) {
        iter->cpu_block->start = iter->next_block_index * UVM_VA_BLOCK_SIZE;
        iter->cpu_block->end = iter->cpu_block->start + UVM_VA_BLOCK_SIZE - 1;

        *out_block = iter->cpu_block;

        goto out;
    }

    // Reached end of current range?
    if (iter->next_block_index > iter->range_end_block_index) {
        uvm_va_range_t *va_range;
	    NvU64 end = (NvU64)-1;

    	va_range = uvm_va_space_iter_next(iter->va_range, end);
        if (!va_range || va_range->type != UVM_VA_RANGE_TYPE_MANAGED) {
            return NV_ERR_INVALID_ADDRESS;
        }

        iter->next_block_index = uvm_va_range_block_index(va_range, va_range->node.start);
	    iter->range_end_block_index = uvm_va_range_block_index(va_range, va_range->node.end);
        iter->va_range = va_range;
    }

    status = uvm_va_range_block_create(iter->va_range, iter->next_block_index, out_block);
    if (status != NV_OK) {
        return status;
    }

out:
    iter->next_block_index++;
    return status;
}

void uvm_va_colored_block_region_init(NvU64 start, NvU64 length, NvU32 color,
        uvm_va_block_colored_region_t *region)
{
    region->start = start & ~(PAGE_SIZE - 1);
    region->page_offset = start & (PAGE_SIZE - 1);
    
    region->color = color;

    region->length = length;
    
    uvm_page_mask_zero(&region->page_mask);

    region->last_block_start = 0;
}

// Update a block color range for a va block
// Since this function depends on physical address, block should be locked
// before calling this function.
NV_STATUS uvm_update_va_colored_block_region(uvm_va_block_t *va_block,
                                               uvm_processor_id_t id,
                                               uvm_va_block_colored_region_t *region)
{
    NvU64 left = region->length;
    NvU64 start, end;
    uvm_page_index_t first, outer, last;
    uvm_gpu_phys_address_t phy_addr;
    uvm_global_gpu_id_t gpu_id;
    uvm_gpu_t *gpu;
    NvU64 page_start, page_end, page_size, page_offset;
    NvU32 page_color;

    // No update needed if current block same as last block
    if (region->last_block_start && region->last_block_start == va_block->start)
        return NV_OK;

    uvm_page_mask_zero(&region->page_mask);

    page_offset = region->page_offset;

    // No coloring on CPU side
    if (id.val == UVM_ID_CPU_VALUE) {
        int ret, i;
        struct page *page;

        start = max(va_block->start, region->start) + page_offset;
        end = min(va_block->end, start + region->length - 1);

        first = uvm_va_block_cpu_page_index(va_block, start);
        outer = uvm_va_block_cpu_page_index(va_block, end) + 1;

        uvm_page_mask_fill(&region->page_mask, first, outer);

        // Only linux backed pages need to be locked
        if (va_block->is_linux_backed) {
            // Release all previously pinned pages
            for (i = 0; i < PAGES_PER_UVM_VA_BLOCK; i++) {
                page = va_block->cpu.pages[i];
                if (page) {
                    put_page(page);
                    va_block->cpu.pages[i]= NULL;
                }
            }

            // Try pinning pages
            ret = NV_GET_USER_PAGES(va_block->start + first * PAGE_SIZE,
                    outer - first, true, false, &va_block->cpu.pages[first], NULL);
            if (ret < 0) {
                return NV_ERR_INVALID_ADDRESS;
            }
        }
        goto done;
    }

    // Only blocks on CPU can be linux backed
    UVM_ASSERT(!va_block->is_linux_backed);

    start = max(va_block->start, region->start);
    first = uvm_va_block_cpu_page_index(va_block, start);
    outer = uvm_va_block_cpu_page_index(va_block, va_block->end) + 1;
    last = first;

    gpu_id.val = id.val;
    gpu = uvm_gpu_get(gpu_id);

    // If physically contiguous, get the start phy address and then increment
    // Else find physical address for all the pages seperately
    if (is_block_phys_contig(va_block, id)) {
        
        uvm_page_index_t i;

        phy_addr = uvm_va_block_gpu_phys_page_address(va_block, first, gpu);

        for (i = first; i < outer && left != 0; i++, phy_addr.address += PAGE_SIZE) {
    
            page_color = gpu->parent->arch_hal->phys_addr_to_transfer_color(gpu, phy_addr.address);
            if (page_color != region->color) {
                continue;
            }

	        last = i;

            uvm_page_mask_set(&region->page_mask, i);
            page_start = max(start, va_block->start + PAGE_SIZE * i) + page_offset;
            page_end = va_block->start + PAGE_SIZE * (i + 1) - 1;
            page_size = min(left, page_end - page_start + 1);
            left -= page_size;
            page_offset = 0;
        }
    } else {
        
        uvm_page_index_t i;

        for (i = first; i < outer && left != 0; i++) {
   
            phy_addr = uvm_va_block_gpu_phys_page_address(va_block, i, gpu);

            page_color = gpu->parent->arch_hal->phys_addr_to_transfer_color(gpu, phy_addr.address);
            if (page_color != region->color) {
                continue;
            }
	
	        last = i;

            uvm_page_mask_set(&region->page_mask, i);
            page_start = max(start, va_block->start + PAGE_SIZE * i) + page_offset;
            page_end = va_block->start + PAGE_SIZE * (i + 1) - 1;
            page_size = min(left, page_end - page_start + 1);
            left -= page_size;
            page_offset = 0;
        }

    }

    outer = last + 1;

done:
    region->region.first = first;
    region->region.outer = outer;

    region->last_block_start = va_block->start;

    return NV_OK;
}

static NV_STATUS uvm_va_block_memcpy_colored_locked(uvm_va_block_t *src_va_block,
                                                    uvm_va_block_t *dest_va_block,
                                                    uvm_processor_id_t src_id,
                                                    uvm_processor_id_t dest_id,
                                                    uvm_va_block_colored_region_t *src_region,
                                                    uvm_va_block_colored_region_t *dest_region,
                                                    NvU64 *copied,
                                                    uvm_tracker_t *out_tracker)
{
    NV_STATUS status = NV_OK;
    uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
    NV_STATUS tracker_status;

    status = uvm_update_va_colored_block_region(src_va_block, src_id, src_region);
    if (status != NV_OK)
        goto out;

    status =uvm_update_va_colored_block_region(dest_va_block, dest_id, dest_region);
    if (status != NV_OK)
        goto out;

    status = block_copy_colored_pages_between(src_va_block,
                                                dest_va_block,
                                                src_id,
                                                dest_id,
                                                src_region,
                                                dest_region,
                                                copied,
                                                &local_tracker);

out:
    if (out_tracker) {
        tracker_status = uvm_tracker_add_tracker_safe(out_tracker, &local_tracker);
        uvm_tracker_deinit(&local_tracker);
    } else {
        // Add everything from the local tracker to the block's tracker.
        tracker_status = uvm_tracker_add_tracker_safe(&dest_va_block->tracker, &local_tracker);
        uvm_tracker_deinit(&local_tracker);
    }

    return status == NV_OK ? tracker_status : status;
}

static NV_STATUS uvm_va_block_memset_colored_locked(uvm_va_block_t *va_block,
                                                    uvm_processor_id_t id,
                                                    uvm_va_block_colored_region_t *region,
                                                    NvU8 value,
                                                    NvU64 *covered,
                                                    uvm_tracker_t *out_tracker)
{
    NV_STATUS status = NV_OK;
    uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
    NV_STATUS tracker_status;

    status = uvm_update_va_colored_block_region(va_block, id, region);
    if (status != NV_OK)
        goto out;

    status = block_memset_colored_pages(va_block,
                                        id,
                                        region,
                                        value,
                                        covered,
                                        &local_tracker);

out:
    if (out_tracker) {
        tracker_status = uvm_tracker_add_tracker_safe(out_tracker, &local_tracker);
        uvm_tracker_deinit(&local_tracker);
    } else {
        // Add everything from the local tracker to the block's tracker.
        tracker_status = uvm_tracker_add_tracker_safe(&va_block->tracker, &local_tracker);
        uvm_tracker_deinit(&local_tracker);
    }

    return status == NV_OK ? tracker_status : status;
}

static NV_STATUS uvm_memcpy_colored_blocks(uvm_va_space_t *va_space,
                                           NvU64 srcBase,
                                           NvU64 destBase,
                                           NvU64 length,
                                           NvU32 color,
                                           uvm_processor_id_t src_id,
                                           uvm_processor_id_t dest_id,
                                           uvm_tracker_t *out_tracker)
{
    NV_STATUS status = NV_OK;
    uvm_block_iter_t src_block_iter, dest_block_iter;
    uvm_va_block_t *src_va_block, *dest_va_block;
    NvU64 left = length;
    NvU64 copied;
    uvm_va_block_colored_region_t src_region, dest_region;

    status = uvm_block_iter_initialization(va_space, srcBase, src_id, &src_block_iter);
    if (status != NV_OK)
        return status;

    status = uvm_block_iter_initialization(va_space, destBase, dest_id, &dest_block_iter);
    if (status != NV_OK) {
        uvm_block_iter_deinitialization(&src_block_iter);
        return status;
    }

    uvm_va_colored_block_region_init(srcBase, length, color, &src_region);
    uvm_va_colored_block_region_init(destBase, length, color, &dest_region);


    while (left != 0) {

        // If current block has been done with, fetch the next block
        if (uvm_page_mask_empty(&src_region.page_mask)) {
            status = uvm_block_iter_next_block(&src_block_iter, &src_va_block);
            if (status != NV_OK)
                goto out;
        }

        if (uvm_page_mask_empty(&dest_region.page_mask)) {
            status = uvm_block_iter_next_block(&dest_block_iter, &dest_va_block);
            if (status != NV_OK)
                goto out;
        }

        status = UVM_VA_GENERIC_MULTI_BLOCK_LOCK_RETRY(src_va_block, dest_va_block,
                NULL, NULL,
                uvm_va_block_memcpy_colored_locked(src_va_block,
                    dest_va_block,
                    src_id,
                    dest_id,
                    &src_region,
                    &dest_region,
                    &copied,
                    out_tracker));

        if (status != NV_OK)
            goto out;

        left -= copied;
    }

out:
    uvm_block_iter_deinitialization(&src_block_iter);
    uvm_block_iter_deinitialization(&dest_block_iter);

    return status;
}

static NV_STATUS uvm_memset_colored_blocks(uvm_va_space_t *va_space,
                                            NvU64 base,
                                            NvU64 length,
                                            NvU8 value,
                                            NvU32 color,
                                            uvm_processor_id_t id,
                                            uvm_tracker_t *out_tracker)
{
    NV_STATUS status;
    uvm_block_iter_t block_iter;
    uvm_va_block_t *va_block;
    NvU64 left = length;
    NvU64 covered;
    uvm_va_block_colored_region_t region;

    status = uvm_block_iter_initialization(va_space, base, id, &block_iter);
    if (status != NV_OK)
        goto out;

    uvm_va_colored_block_region_init(base, length, color, &region);

    while (left != 0) {

        // If current block has been done with, fetch the next block
        if (uvm_page_mask_empty(&region.page_mask)) {
            status = uvm_block_iter_next_block(&block_iter, &va_block);
            if (status != NV_OK)
                goto out;
        }

        status = UVM_VA_BLOCK_LOCK_RETRY(va_block, NULL,
                uvm_va_block_memset_colored_locked(va_block,
                                                    id,
                                                    &region,
                                                    value,
                                                    &covered,
                                                    out_tracker));

        if (status != NV_OK)
            goto out;

        left -= covered;
    }

out:
    uvm_block_iter_deinitialization(&block_iter);

    return status;
}

static NV_STATUS uvm_migrate_ranges(uvm_va_space_t *va_space,
                                    uvm_va_block_context_t *va_block_context,
                                    uvm_va_range_t *first_va_range,
                                    NvU64 base,
                                    NvU64 length,
                                    uvm_processor_id_t dest_id,
                                    uvm_migrate_mode_t mode,
                                    bool should_do_cpu_preunmap,
                                    uvm_tracker_t *out_tracker)
{
    uvm_va_range_t *va_range, *va_range_last;
    NvU64 end = base + length - 1;
    NV_STATUS status = NV_OK;
    bool skipped_migrate = false;

    UVM_ASSERT(first_va_range == uvm_va_space_iter_first(va_space, base, base));

    va_range_last = NULL;
    uvm_for_each_va_range_in_contig_from(va_range, va_space, first_va_range, end) {
        uvm_range_group_range_iter_t iter;
        va_range_last = va_range;

        // Only managed ranges can be migrated
        if (va_range->type != UVM_VA_RANGE_TYPE_MANAGED) {
            status = NV_ERR_INVALID_ADDRESS;
            break;
        }

        // For UVM-Lite GPUs, the CUDA driver may suballocate a single va_range
        // into many range groups.  For this reason, we iterate over each va_range first
        // then through the range groups within.
        uvm_range_group_for_each_migratability_in(&iter,
                                                  va_space,
                                                  max(base, va_range->node.start),
                                                  min(end, va_range->node.end)) {
            // Skip non-migratable VA ranges
            if (!iter.migratable) {
                // Only return NV_WARN_MORE_PROCESSING_REQUIRED if the pages aren't
                // already resident at dest_id.
                if (!uvm_id_equal(va_range->preferred_location, dest_id))
                    skipped_migrate = true;
            }
            else if (uvm_processor_mask_test(&va_range->uvm_lite_gpus, dest_id) &&
                     !uvm_id_equal(dest_id, va_range->preferred_location)) {
                // Don't migrate to a non-faultable GPU that is in UVM-Lite mode,
                // unless it's the preferred location
                status = NV_ERR_INVALID_DEVICE;
                break;
            }
            else {
                status = uvm_va_range_migrate(va_range,
                                              va_block_context,
                                              iter.start,
                                              iter.end,
                                              dest_id,
                                              mode,
                                              should_do_cpu_preunmap,
                                              out_tracker);
                if (status != NV_OK)
                    break;
            }
        }
    }


    if (status != NV_OK)
        return status;

    // Check that we were able to iterate over the entire range without any gaps
    if (!va_range_last || va_range_last->node.end < end)
        return NV_ERR_INVALID_ADDRESS;

    if (skipped_migrate)
        return NV_WARN_MORE_PROCESSING_REQUIRED;

    return NV_OK;
}

static NV_STATUS uvm_migrate(uvm_va_space_t *va_space,
                             struct mm_struct *mm,
                             NvU64 base,
                             NvU64 length,
                             uvm_processor_id_t dest_id,
                             NvU32 migrate_flags,
                             uvm_tracker_t *out_tracker)
{
    NV_STATUS status = NV_OK;
    uvm_va_range_t *first_va_range = uvm_va_space_iter_first(va_space, base, base);
    uvm_va_block_context_t *va_block_context;
    bool do_mappings;
    bool do_two_passes;
    bool is_single_block;
    bool should_do_cpu_preunmap;

    uvm_assert_rwsem_locked(&va_space->lock);

    if (!first_va_range || first_va_range->type != UVM_VA_RANGE_TYPE_MANAGED)
        return NV_ERR_INVALID_ADDRESS;

    // If the GPU has its memory disabled, just skip the migration and let
    // faults take care of things.
    if (!uvm_va_space_processor_has_memory(va_space, dest_id))
        return NV_OK;

    va_block_context = uvm_va_block_context_alloc(mm);
    if (!va_block_context)
        return NV_ERR_NO_MEMORY;

    // We perform two passes (unless the migration only covers a single VA
    // block or UVM_MIGRATE_FLAG_SKIP_CPU_MAP is passed). This helps in the
    // following scenarios:
    //
    // - Migrations that add CPU mappings, since they are synchronous operations
    // that delay the migration of the next VA blocks.
    // - Concurrent migrations. This is due to our current channel selection
    // logic that doesn't prevent false dependencies between independent
    // operations. For example, removal of mappings for outgoing transfers are
    // delayed by the mappings added by incoming transfers.
    // TODO: Bug 1764953: Re-evaluate the two-pass logic when channel selection
    // is overhauled.
    //
    // The two passes are as follows:
    //
    // 1- Transfer all VA blocks (do not add mappings)
    // 2- Go block by block reexecuting the transfer (in case someone moved it
    // since the first pass), and adding the mappings.
    is_single_block = is_migration_single_block(first_va_range, base, length);
    do_mappings = UVM_ID_IS_GPU(dest_id) || !(migrate_flags & UVM_MIGRATE_FLAG_SKIP_CPU_MAP);
    do_two_passes = do_mappings && !is_single_block;

    if (do_two_passes) {
        should_do_cpu_preunmap = migration_should_do_cpu_preunmap(va_space, UVM_MIGRATE_PASS_FIRST, is_single_block);

        status = uvm_migrate_ranges(va_space,
                                    va_block_context,
                                    first_va_range,
                                    base,
                                    length,
                                    dest_id,
                                    UVM_MIGRATE_MODE_MAKE_RESIDENT,
                                    should_do_cpu_preunmap,
                                    out_tracker);
    }

    if (status == NV_OK) {
        uvm_migrate_mode_t mode = do_mappings? UVM_MIGRATE_MODE_MAKE_RESIDENT_AND_MAP:
                                               UVM_MIGRATE_MODE_MAKE_RESIDENT;
        uvm_migrate_pass_t pass = do_two_passes? UVM_MIGRATE_PASS_SECOND:
                                                 UVM_MIGRATE_PASS_FIRST;
        should_do_cpu_preunmap = migration_should_do_cpu_preunmap(va_space, pass, is_single_block);

        status = uvm_migrate_ranges(va_space,
                                    va_block_context,
                                    first_va_range,
                                    base,
                                    length,
                                    dest_id,
                                    mode,
                                    should_do_cpu_preunmap,
                                    out_tracker);
    }

    uvm_va_block_context_free(va_block_context);

    return status;
}


static NV_STATUS uvm_memcpy_colored(uvm_va_space_t *va_space,
                                    NvU64 srcBase,
                                    NvU64 destBase,
                                    NvU64 length,
                                    NvU32 color,
                                    uvm_processor_id_t src_id,
                                    uvm_processor_id_t dest_id,
                                    uvm_tracker_t *out_tracker)
{
    NV_STATUS status = NV_OK;

    uvm_assert_mmap_lock_locked(current->mm);
    uvm_assert_rwsem_locked(&va_space->lock);

    // TODO: Populate pages and map them
    status = uvm_memcpy_colored_blocks(va_space,
                                        srcBase,
                                        destBase,
                                        length,
                                        color,
                                        src_id,
                                        dest_id,
                                        out_tracker);

    if (status != NV_OK)
        return status;

    return NV_OK;
}

static NV_STATUS uvm_memset_colored(uvm_va_space_t *va_space,
                                    NvU64 base,
                                    NvU64 length,
                                    NvU8 value,
                                    NvU32 color,
                                    uvm_processor_id_t id,
                                    uvm_tracker_t *out_tracker)
{
    NV_STATUS status = NV_OK;

    uvm_assert_mmap_lock_locked(current->mm);
    uvm_assert_rwsem_locked(&va_space->lock);

    // TODO: Populate pages and map them
    status = uvm_memset_colored_blocks(va_space,
                                       base,
                                       length,
                                       value,
                                       color,
                                       id,
                                       out_tracker);

    if (status != NV_OK)
        return status;

    return NV_OK;
}

static NV_STATUS uvm_push_async_user_sem_release(uvm_gpu_t *release_from_gpu,
                                                 uvm_va_range_semaphore_pool_t *sema_va_range,
                                                 NvU64 sema_user_addr,
                                                 NvU32 payload,
                                                 uvm_tracker_t *release_after_tracker)
{
    uvm_push_t push;
    NV_STATUS status;
    uvm_gpu_address_t sema_phys_addr;

    status = uvm_mem_map_gpu_phys(sema_va_range->mem, release_from_gpu);
    if (status != NV_OK)
        return status;

    sema_phys_addr = uvm_mem_gpu_address_physical(sema_va_range->mem, release_from_gpu,
            sema_user_addr - (NvU64)(uintptr_t)sema_va_range->mem->user.addr, 4);

    status = uvm_push_begin_acquire(release_from_gpu->channel_manager,
                                    UVM_CHANNEL_TYPE_MEMOPS,
                                    release_after_tracker,
                                    &push,
                                    "Pushing semaphore release (*0x%llx = %u)",
                                    sema_user_addr,
                                    (unsigned)payload);
    if (status != NV_OK) {
        UVM_ERR_PRINT("uvm_push_begin_acquire() returned %d (%s)\n", status, nvstatusToString(status));
        return status;
    }

    release_from_gpu->parent->host_hal->membar_sys(&push);
    release_from_gpu->parent->ce_hal->memset_4(&push, sema_phys_addr, payload, 4);

    uvm_push_end(&push);

    uvm_mutex_lock(&sema_va_range->tracker_lock);
    status = uvm_tracker_add_push_safe(&sema_va_range->tracker, &push);
    uvm_tracker_remove_completed(&sema_va_range->tracker);
    uvm_mutex_unlock(&sema_va_range->tracker_lock);
    if (status != NV_OK) {
        UVM_ERR_PRINT("uvm_tracker_add_push() returned %d (%s)\n", status, nvstatusToString(status));
        return status;
    }

    return NV_OK;
}

static void uvm_release_user_sem_from_cpu(uvm_mem_t *sema_mem, NvU64 user_addr, NvU32 payload)
{
    NvU64 sema_offset = user_addr - (NvU64)(uintptr_t)sema_mem->user.addr;
    NvU64 sema_page = uvm_div_pow2_64(sema_offset, sema_mem->chunk_size);
    NvU64 sema_page_offset = sema_offset & (sema_mem->chunk_size - 1);
    void *cpu_page_virt;
    void *cpu_addr;

    // Prevent processor speculation prior to accessing user-mapped memory to
    // avoid leaking information from side-channel attacks. Under speculation, a
    // valid VA range which does not contain this semaphore could be used by the
    // caller. It's unclear but likely that the user might be able to control
    // the data at that address. Auditing all potential ways that could happen
    // is difficult and error-prone, so to be on the safe side we'll just always
    // block speculation.
    nv_speculation_barrier();

    cpu_page_virt = kmap(sema_mem->sysmem.pages[sema_page]);
    cpu_addr = (char *)cpu_page_virt + sema_page_offset;
    UVM_WRITE_ONCE(*(NvU32 *)cpu_addr, payload);
    kunmap(sema_mem->sysmem.pages[sema_page]);
}

static NV_STATUS uvm_migrate_release_user_sem(const UVM_MIGRATE_PARAMS *params,
                                              uvm_va_space_t *va_space,
                                              uvm_va_range_t *sema_va_range,
                                              uvm_gpu_t *dest_gpu,
                                              uvm_tracker_t *tracker_ptr,
                                              bool *wait_for_tracker_out)
{
    NV_STATUS status;
    uvm_mem_t *sema_mem = sema_va_range->semaphore_pool.mem;
    uvm_gpu_t *release_from = NULL;

    *wait_for_tracker_out = true;
    if (sema_va_range->semaphore_pool.owner)
        release_from = sema_va_range->semaphore_pool.owner;
    else
        release_from = dest_gpu;

    if (sema_va_range->semaphore_pool.owner == NULL && uvm_tracker_is_completed(tracker_ptr)) {
        // No GPU has the semaphore pool cached. Attempt eager release from CPU
        // if the tracker is already completed.
        *wait_for_tracker_out = false;
        uvm_release_user_sem_from_cpu(sema_mem, params->semaphoreAddress, params->semaphorePayload);
    }
    else {
        // Semaphore has to be released from a GPU because it is cached or we were unable
        // to release it from the CPU.
        if (!release_from) {
            // We did not do a CPU release, but the destination is CPU. This means the
            // tracker is not complete, and could be because accessed_by mappings are being
            // set up asynchronously, or because of the test-only flag
            // UVM_MIGRATE_FLAG_SKIP_CPU_MAP. However, this means there should be a registered
            // GPU since all CPU work is synchronous.
            release_from = uvm_va_space_find_first_gpu(va_space);
            UVM_ASSERT(release_from);
        }
        status = uvm_push_async_user_sem_release(release_from,
                                                 &sema_va_range->semaphore_pool,
                                                 params->semaphoreAddress,
                                                 params->semaphorePayload,
                                                 tracker_ptr);
        if (status != NV_OK) {
            UVM_ERR_PRINT("uvm_push_async_user_sem_release() returned %d (%s)\n",
                    status, nvstatusToString(status));
            return status;
        }
        else {
            *wait_for_tracker_out = false;
        }
    }

    return NV_OK;
}

NV_STATUS uvm_migrate_init()
{
    NV_STATUS status = uvm_migrate_pageable_init();
    if (status != NV_OK)
        return status;

    g_uvm_perf_migrate_cpu_preunmap_enable = uvm_perf_migrate_cpu_preunmap_enable != 0;

    BUILD_BUG_ON((UVM_VA_BLOCK_SIZE) & (UVM_VA_BLOCK_SIZE - 1));

    if (g_uvm_perf_migrate_cpu_preunmap_enable) {
        if (uvm_perf_migrate_cpu_preunmap_block_order <= UVM_PERF_MIGRATE_CPU_PREUNMAP_BLOCK_ORDER_MAX) {
            g_uvm_perf_migrate_cpu_preunmap_size = UVM_VA_BLOCK_SIZE << uvm_perf_migrate_cpu_preunmap_block_order;
        }
        else {
            g_uvm_perf_migrate_cpu_preunmap_size = UVM_VA_BLOCK_SIZE << UVM_PERF_MIGRATE_CPU_PREUNMAP_BLOCK_ORDER_DEFAULT;

            pr_info("Invalid value %u for uvm_perf_migrate_cpu_preunmap_block_order. Using %u instead\n",
                    uvm_perf_migrate_cpu_preunmap_block_order,
                    UVM_PERF_MIGRATE_CPU_PREUNMAP_BLOCK_ORDER_DEFAULT);
        }
    }

    return NV_OK;
}

void uvm_migrate_exit()
{
    uvm_migrate_pageable_exit();
}

NV_STATUS uvm_api_migrate(UVM_MIGRATE_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_tracker_t tracker = UVM_TRACKER_INIT();
    uvm_tracker_t *tracker_ptr = NULL;
    uvm_gpu_t *dest_gpu = NULL;
    uvm_va_range_t *sema_va_range = NULL;
    struct mm_struct *mm = current->mm;
    NV_STATUS status = NV_OK;
    NV_STATUS tracker_status = NV_OK;
    bool wait_for_tracker = true;
    const bool is_async = params->flags & UVM_MIGRATE_FLAG_ASYNC;

    // We temporarily allow 0 length in the IOCTL parameters as a signal to
    // only release the semaphore. This is because user-space is in charge of
    // migrating pageable memory in some cases.
    //
    // TODO: Bug 2419180: do not allow 0 length migrations when we fully switch
    // to migrate_vma for all types of vmas.
    if (params->length > 0 || !is_async || params->semaphoreAddress == 0) {
        if (uvm_api_range_invalid(params->base, params->length))
            return NV_ERR_INVALID_ADDRESS;
    }

    if (params->flags & ~UVM_MIGRATE_FLAGS_ALL)
        return NV_ERR_INVALID_ARGUMENT;

    if ((params->flags & UVM_MIGRATE_FLAGS_TEST_ALL) && !uvm_enable_builtin_tests) {
        UVM_INFO_PRINT("Test flag set for UVM_MIGRATE. Did you mean to insmod with uvm_enable_builtin_tests=1?\n");
        return NV_ERR_INVALID_ARGUMENT;
    }

    // mmap_lock will be needed if we have to create CPU mappings
    uvm_down_read_mmap_lock(mm);
    uvm_va_space_down_read(va_space);

    if (!is_async) {
        if (params->semaphoreAddress != 0) {
            status = NV_ERR_INVALID_ARGUMENT;
            goto done;
        }
    }
    else {
        if (params->semaphoreAddress == 0) {
            if (params->semaphorePayload != 0) {
                status = NV_ERR_INVALID_ARGUMENT;
                goto done;
            }
        }
        else {
            sema_va_range = uvm_va_range_find(va_space, params->semaphoreAddress);
            if (!IS_ALIGNED(params->semaphoreAddress, sizeof(params->semaphorePayload)) ||
                    !sema_va_range || sema_va_range->type != UVM_VA_RANGE_TYPE_SEMAPHORE_POOL) {
                status = NV_ERR_INVALID_ADDRESS;
                goto done;
            }
        }
    }

    if (uvm_uuid_is_cpu(&params->destinationUuid)) {
        dest_gpu = NULL;
    }
    else {
        if (params->flags & UVM_MIGRATE_FLAG_NO_GPU_VA_SPACE)
            dest_gpu = uvm_va_space_get_gpu_by_uuid(va_space, &params->destinationUuid);
        else
            dest_gpu = uvm_va_space_get_gpu_by_uuid_with_gpu_va_space(va_space, &params->destinationUuid);

        if (!dest_gpu) {
            status = NV_ERR_INVALID_DEVICE;
            goto done;
        }

        if (params->length > 0 && !uvm_gpu_can_address(dest_gpu, params->base + params->length - 1)) {
            status = NV_ERR_OUT_OF_RANGE;
            goto done;
        }
    }

    UVM_ASSERT(status == NV_OK);

    // If we're synchronous or if we need to release a semaphore, use a tracker.
    if (!is_async || params->semaphoreAddress)
        tracker_ptr = &tracker;

    if (params->length > 0) {
        status = uvm_api_range_type_check(va_space, mm, params->base, params->length);
        if (status == NV_OK) {
            status = uvm_migrate(va_space,
                                 mm,
                                 params->base,
                                 params->length,
                                 (dest_gpu ? dest_gpu->id : UVM_ID_CPU),
                                 params->flags,
                                 tracker_ptr);
        }
        else if (status == NV_WARN_NOTHING_TO_DO) {
            status = uvm_migrate_pageable(va_space,
                                          mm,
                                          params->base,
                                          params->length,
                                          (dest_gpu ? dest_gpu->id : UVM_ID_CPU),
                                          (int)params->cpuNumaNode,
                                          &params->userSpaceStart,
                                          &params->userSpaceLength);
        }
    }

done:
    // We only need to hold mmap_lock to create new CPU mappings, so drop it if
    // we need to wait for the tracker to finish.
    //
    // TODO: Bug 1766650: For large migrations with destination CPU, try
    //       benchmarks to see if a two-pass approach would be faster (first
    //       pass pushes all GPU work asynchronously, second pass updates CPU
    //       mappings synchronously).
    uvm_up_read_mmap_lock_out_of_order(mm);

    if (tracker_ptr) {
        if (params->semaphoreAddress && status == NV_OK) {
            // Need to do a semaphore release.
            status = uvm_migrate_release_user_sem(params, va_space, sema_va_range,
                                                  dest_gpu, tracker_ptr, &wait_for_tracker);
        }

        if (wait_for_tracker) {
            // There was an error or we are sync. Even if there was an error, we
            // need to wait for work already dispatched to complete. Waiting on
            // a tracker requires the VA space lock to prevent GPUs being unregistered
            // during the wait.
            tracker_status = uvm_tracker_wait_deinit(tracker_ptr);
        }
        else {
            uvm_tracker_deinit(tracker_ptr);
        }
    }

    uvm_va_space_up_read(va_space);

    // When the UVM driver blocks on a migration, use the opportunity to eagerly dispatch
    // the migration events once the migration is complete, instead of waiting for a later
    // event flush to process the events.
    if (wait_for_tracker)
        uvm_tools_flush_events();

    // Only clobber status if we didn't hit an earlier error
    return status == NV_OK ? tracker_status : status;
}


NV_STATUS uvm_api_memcpy_colored(UVM_MEMCPY_COLORED_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_tracker_t tracker = UVM_TRACKER_INIT();
    // NULL = CPU
    uvm_gpu_t *src_gpu = NULL;
    uvm_gpu_t *dest_gpu = NULL;
    NvU32 color = 0;
    NV_STATUS status;
    NV_STATUS tracker_status = NV_OK;

    // mmap_sem will be needed if we have to create CPU mappings
    uvm_down_read_mmap_lock(current->mm);
    uvm_va_space_down_read(va_space);

    if (uvm_uuid_is_cpu(&params->srcUuid)) {
        src_gpu = NULL;
    }
    else {
        src_gpu = uvm_va_space_get_gpu_by_uuid_with_gpu_va_space(va_space, &params->srcUuid);

        if (!src_gpu) {
            status = NV_ERR_INVALID_DEVICE;
            goto done;
        }
    }

    if (uvm_uuid_is_cpu(&params->destUuid)) {
        dest_gpu = NULL;
    }
    else {
        dest_gpu = uvm_va_space_get_gpu_by_uuid_with_gpu_va_space(va_space, &params->destUuid);

        if (!dest_gpu) {
            status = NV_ERR_INVALID_DEVICE;
            goto done;
        }
    }

    // Either atmost one src/dest lie on CPU or both lie on same GPU
    // Invalid configuration: Both lie on CPU or different GPUs
    if ((!src_gpu && !dest_gpu) || (src_gpu && dest_gpu && src_gpu->id.val != dest_gpu->id.val)) {
    	status = NV_ERR_INVALID_DEVICE;
        goto done;
    }

    // Atleast one is a GPU. Get it's color. If both on same GPU, then also only a single color exists.
    if (src_gpu) {
        status = uvm_pmm_get_current_process_color(&src_gpu->pmm, &color);
        if (status != NV_OK)
            goto done;
    } else {
        status = uvm_pmm_get_current_process_color(&dest_gpu->pmm, &color);
        if (status != NV_OK)
            goto done;
    }

    if (params->length == 0) {
        status = NV_OK;
        goto done;
    }

    uvm_processor_id_t src_gpuid, dst_gpuid;
    src_gpuid.val = UVM_CPU_ID;
    dst_gpuid.val = UVM_CPU_ID;
    // This is synchronous call, so using a tracker.
    status = uvm_memcpy_colored(va_space, params->srcBase, params->destBase, 
                                params->length, color, (src_gpu ? src_gpu->id : src_gpuid),
                                (dest_gpu ? dest_gpu->id : dst_gpuid),
                                &tracker);

done:
    // We only need to hold mmap_sem to create new CPU mappings, so drop it if
    // we need to wait for the tracker to finish.
    //
    // TODO: Bug 1766650: For large migrations with destination CPU, try
    //       benchmarks to see if a two-pass approach would be faster (first
    //       pass pushes all GPU work asynchronously, second pass updates CPU
    //       mappings synchronously).
    uvm_up_read_mmap_lock_out_of_order(current->mm);

    // There was an error or we are sync. Even if there was an error, we
    // need to wait for work already dispatched to complete. Waiting on
    // a tracker requires the VA space lock to prevent GPUs being unregistered
    // during the wait.
    tracker_status = uvm_tracker_wait_deinit(&tracker);

    uvm_va_space_up_read(va_space);

    // When the UVM driver blocks on a migration, use the opportunity to eagerly dispatch
    // the migration events once the migration is complete, instead of waiting for a later
    // event flush to process the events.
    uvm_tools_flush_events();

    // Only clobber status if we didn't hit an earlier error
    return status == NV_OK ? tracker_status : status;
}


NV_STATUS uvm_api_memset_colored(UVM_MEMSET_COLORED_PARAMS *params, struct file *filp)
{
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    uvm_tracker_t tracker = UVM_TRACKER_INIT();
    uvm_gpu_t *gpu = NULL;
    NvU32 color = 0;
    NV_STATUS status;
    NV_STATUS tracker_status = NV_OK;

    // mmap_sem will be needed if we have to create CPU mappings
    uvm_down_read_mmap_lock(current->mm);
    uvm_va_space_down_read(va_space);

    // Only GPU are supported (CPU can use memset() in userspace)
    if (uvm_uuid_is_cpu(&params->uuid)) {
        status = NV_ERR_INVALID_DEVICE;
        goto done;
    }
        
    gpu = uvm_va_space_get_gpu_by_uuid_with_gpu_va_space(va_space, &params->uuid);
    if (!gpu) {
        status = NV_ERR_INVALID_DEVICE;
        goto done;
    }

    status = uvm_pmm_get_current_process_color(&gpu->pmm, &color);
    if (status != NV_OK)
        goto done;

    if (params->length == 0) {
        status = NV_OK;
        goto done;
    }

    // This is synchronous call, so using a tracker.
    status = uvm_memset_colored(va_space, params->base, params->length, params->value,
                                color, gpu->id, &tracker);

done:
    // We only need to hold mmap_sem to create new CPU mappings, so drop it if
    // we need to wait for the tracker to finish.
    //
    // TODO: Bug 1766650: For large migrations with destination CPU, try
    //       benchmarks to see if a two-pass approach would be faster (first
    //       pass pushes all GPU work asynchronously, second pass updates CPU
    //       mappings synchronously).
    uvm_up_read_mmap_lock_out_of_order(current->mm);

    // There was an error or we are sync. Even if there was an error, we
    // need to wait for work already dispatched to complete. Waiting on
    // a tracker requires the VA space lock to prevent GPUs being unregistered
    // during the wait.
    tracker_status = uvm_tracker_wait_deinit(&tracker);

    uvm_va_space_up_read(va_space);

    // When the UVM driver blocks on a migration, use the opportunity to eagerly dispatch
    // the migration events once the migration is complete, instead of waiting for a later
    // event flush to process the events.
    uvm_tools_flush_events();

    // Only clobber status if we didn't hit an earlier error
    return status == NV_OK ? tracker_status : status;
}

NV_STATUS uvm_api_migrate_range_group(UVM_MIGRATE_RANGE_GROUP_PARAMS *params, struct file *filp)
{
    NV_STATUS status = NV_OK;
    NV_STATUS tracker_status = NV_OK;
    uvm_va_space_t *va_space = uvm_va_space_get(filp);
    struct mm_struct *mm = current->mm;
    uvm_range_group_t *range_group;
    uvm_range_group_range_t *rgr;
    uvm_processor_id_t dest_id;
    uvm_tracker_t local_tracker = UVM_TRACKER_INIT();
    NvU32 migrate_flags = 0;
    uvm_gpu_t *gpu = NULL;

    // mmap_lock will be needed if we have to create CPU mappings
    uvm_down_read_mmap_lock(mm);
    uvm_va_space_down_read(va_space);

    if (uvm_uuid_is_cpu(&params->destinationUuid)) {
        dest_id = UVM_ID_CPU;
    }
    else {
        gpu = uvm_va_space_get_gpu_by_uuid_with_gpu_va_space(va_space, &params->destinationUuid);
        if (!gpu) {
            status = NV_ERR_INVALID_DEVICE;
            goto done;
        }

        dest_id = gpu->id;
    }

    range_group = radix_tree_lookup(&va_space->range_groups, params->rangeGroupId);
    if (!range_group) {
        status = NV_ERR_OBJECT_NOT_FOUND;
        goto done;
    }

    // Migrate all VA ranges in the range group. uvm_migrate is used because it performs all
    // VA range validity checks.
    list_for_each_entry(rgr, &range_group->ranges, range_group_list_node) {
        NvU64 start = rgr->node.start;
        NvU64 end = rgr->node.end;

        if (gpu && !uvm_gpu_can_address(gpu, end))
            status = NV_ERR_OUT_OF_RANGE;
        else
            status = uvm_migrate(va_space, mm, start, end - start + 1, dest_id, migrate_flags, &local_tracker);

        if (status != NV_OK)
            goto done;
    }

done:
    // We only need to hold mmap_lock to create new CPU mappings, so drop it if
    // we need to wait for the tracker to finish.
    //
    // TODO: Bug 1766650: For large migrations with destination CPU, try
    //       benchmarks to see if a two-pass approach would be faster (first
    //       pass pushes all GPU work asynchronously, second pass updates CPU
    //       mappings synchronously).
    uvm_up_read_mmap_lock_out_of_order(mm);

    tracker_status = uvm_tracker_wait_deinit(&local_tracker);
    uvm_va_space_up_read(va_space);

    // This API is synchronous, so wait for migrations to finish
    uvm_tools_flush_events();

    return status == NV_OK? tracker_status : status;
}

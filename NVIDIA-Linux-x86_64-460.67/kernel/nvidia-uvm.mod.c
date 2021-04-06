#include <linux/build-salt.h>
#include <linux/module.h>
#include <linux/vermagic.h>
#include <linux/compiler.h>

BUILD_SALT;

MODULE_INFO(vermagic, VERMAGIC_STRING);
MODULE_INFO(name, KBUILD_MODNAME);

__visible struct module __this_module
__section(.gnu.linkonce.this_module) = {
	.name = KBUILD_MODNAME,
	.init = init_module,
#ifdef CONFIG_MODULE_UNLOAD
	.exit = cleanup_module,
#endif
	.arch = MODULE_ARCH_INIT,
};

#ifdef CONFIG_RETPOLINE
MODULE_INFO(retpoline, "Y");
#endif

static const struct modversion_info ____versions[]
__used __section(__versions) = {
	{ 0xc79d2779, "module_layout" },
	{ 0x195ce284, "alloc_pages_current" },
	{ 0x455c18f9, "nvUvmInterfaceSessionCreate" },
	{ 0x2d3385d3, "system_wq" },
	{ 0x83c66b, "kmem_cache_destroy" },
	{ 0x4e768480, "address_space_init_once" },
	{ 0x8b66e8a1, "cdev_del" },
	{ 0x8537dfba, "kmalloc_caches" },
	{ 0xeb233a45, "__kmalloc" },
	{ 0xb8c2987b, "cdev_init" },
	{ 0x8170cee, "nvUvmInterfacePmaAllocPages" },
	{ 0x1ed8b599, "__x86_indirect_thunk_r8" },
	{ 0x9d2c519d, "nvUvmInterfaceDupAllocation" },
	{ 0x53b954a2, "up_read" },
	{ 0xc590c0f9, "__put_devmap_managed_page" },
	{ 0x30aeb11d, "nvUvmInterfacePmaUnregisterEvictionCallbacks" },
	{ 0xd6ee688f, "vmalloc" },
	{ 0x9875a8e1, "single_open" },
	{ 0xaad0ae78, "__bitmap_shift_right" },
	{ 0x3184b6ef, "param_ops_int" },
	{ 0xa0fbac79, "wake_up_bit" },
	{ 0xf0ad876, "proc_symlink" },
	{ 0xcf560b60, "nvUvmInterfaceEnableAccessCntr" },
	{ 0xab68fd3b, "nvUvmInterfaceGetP2PCaps" },
	{ 0x41f08b50, "nvUvmInterfaceChannelAllocate" },
	{ 0xa4191c0b, "memset_io" },
	{ 0x263ed23b, "__x86_indirect_thunk_r12" },
	{ 0x3ec90268, "nvUvmInterfacePmaPinPages" },
	{ 0x550cc5f7, "nvUvmInterfaceRegisterUvmCallbacks" },
	{ 0x3e230885, "single_release" },
	{ 0xa62da25a, "nvUvmInterfaceAddressSpaceDestroy" },
	{ 0x18888d00, "downgrade_write" },
	{ 0xaa9c1109, "nvUvmInterfaceUnregisterGpu" },
	{ 0xc655f87d, "nvUvmInterfaceAddressSpaceCreate" },
	{ 0x78ba08fc, "nvUvmInterfaceDeviceDestroy" },
	{ 0xd106481f, "nvUvmInterfaceDupMemory" },
	{ 0xc3385e21, "nvUvmInterfaceSessionDestroy" },
	{ 0xc0a3d105, "find_next_bit" },
	{ 0xffeedf6a, "delayed_work_timer_fn" },
	{ 0xdf566a59, "__x86_indirect_thunk_r9" },
	{ 0x81b395b3, "down_interruptible" },
	{ 0xe9a8fe46, "seq_printf" },
	{ 0x56470118, "__warn_printk" },
	{ 0xc29957c3, "__x86_indirect_thunk_rcx" },
	{ 0xe678af42, "nvUvmInterfaceGetPmaObject" },
	{ 0xdbbe9dd4, "nvUvmInterfaceGetChannelResourcePtes" },
	{ 0xc6f46339, "init_timer_key" },
	{ 0x9fa7184a, "cancel_delayed_work_sync" },
	{ 0x409bcb62, "mutex_unlock" },
	{ 0xa6024e51, "nvUvmInterfaceP2pObjectCreate" },
	{ 0x6091b333, "unregister_chrdev_region" },
	{ 0x999e8297, "vfree" },
	{ 0x1c50cbef, "nvUvmInterfaceGetEccInfo" },
	{ 0x4629334c, "__preempt_count" },
	{ 0x7a2af7b4, "cpu_number" },
	{ 0x97651e6c, "vmemmap_base" },
	{ 0x922f45a6, "__bitmap_clear" },
	{ 0xfee7c6cb, "seq_read" },
	{ 0x344fd44f, "pv_ops" },
	{ 0x63d0464d, "nvUvmInterfacePmaRegisterEvictionCallbacks" },
	{ 0x8c9b38f1, "set_page_dirty" },
	{ 0x4f38cf60, "kthread_create_on_node" },
	{ 0x668b19a1, "down_read" },
	{ 0xe2d5255a, "strcmp" },
	{ 0xb3687850, "out_of_line_wait_on_bit_lock" },
	{ 0xc5e4a5d1, "cpumask_next" },
	{ 0x3f393e9e, "proc_remove" },
	{ 0xece784c2, "rb_first" },
	{ 0xd9a5ea54, "__init_waitqueue_head" },
	{ 0xb44ad4b3, "_copy_to_user" },
	{ 0x9968327e, "PDE_DATA" },
	{ 0x17de3d5, "nr_cpu_ids" },
	{ 0xd5057c22, "nvUvmInterfaceMemoryFree" },
	{ 0x6de13801, "wait_for_completion" },
	{ 0x8848086a, "param_ops_charp" },
	{ 0xa084749a, "__bitmap_or" },
	{ 0xc2bafeef, "nvUvmInterfaceFreeDupedHandle" },
	{ 0x7e526bfa, "__x86_indirect_thunk_r10" },
	{ 0x84ccfdfb, "nvUvmInterfaceDeRegisterUvmOps" },
	{ 0xfb578fc5, "memset" },
	{ 0x19f28917, "nvUvmInterfaceServiceDeviceInterruptsRM" },
	{ 0x9e683f75, "__cpu_possible_mask" },
	{ 0x3dad9978, "cancel_delayed_work" },
	{ 0xd38cd261, "__default_kernel_pte_mask" },
	{ 0xe615712b, "nvUvmInterfaceDeviceCreate" },
	{ 0x3812050a, "_raw_spin_unlock_irqrestore" },
	{ 0x4e0ecf27, "current_task" },
	{ 0xfc7e2596, "down_trylock" },
	{ 0x977f511b, "__mutex_init" },
	{ 0xc5850110, "printk" },
	{ 0x7d5a7d0b, "kthread_stop" },
	{ 0x449ad0a7, "memcmp" },
	{ 0x5c8382a1, "nvUvmInterfaceInitFaultInfo" },
	{ 0x1edb69d6, "ktime_get_raw_ts64" },
	{ 0xda4515cf, "vmap" },
	{ 0xf1e046cc, "panic" },
	{ 0x4c9d28b0, "phys_base" },
	{ 0x7cc0e213, "nvUvmInterfaceBindChannelResources" },
	{ 0xbadcd1bf, "nvUvmInterfaceGetGpuInfo" },
	{ 0xe2cf2a80, "nvUvmInterfaceReportNonReplayableFault" },
	{ 0x479c3c86, "find_next_zero_bit" },
	{ 0xe7b00dfb, "__x86_indirect_thunk_r13" },
	{ 0x97b6e30b, "nvUvmInterfaceQueryCopyEnginesCaps" },
	{ 0xa1c76e0a, "_cond_resched" },
	{ 0x4d9b652b, "rb_erase" },
	{ 0x13f29d6a, "dma_direct_map_page" },
	{ 0x593c1bac, "__x86_indirect_thunk_rbx" },
	{ 0x5f5f1453, "kmem_cache_free" },
	{ 0x2ab7989d, "mutex_lock" },
	{ 0x1199fc8a, "nvUvmInterfaceMemoryAllocFB" },
	{ 0xf1969a8e, "__usecs_to_jiffies" },
	{ 0x6626afca, "down" },
	{ 0x52ddc3d5, "nvUvmInterfaceDestroyFaultInfo" },
	{ 0x7c173634, "__bitmap_complement" },
	{ 0xce8b1878, "__x86_indirect_thunk_r14" },
	{ 0x27491fbc, "nvUvmInterfaceStopChannel" },
	{ 0xe3a53f4c, "sort" },
	{ 0xce807a25, "up_write" },
	{ 0x57bc19d2, "down_write" },
	{ 0xa2de16e5, "fput" },
	{ 0xfe487975, "init_wait_entry" },
	{ 0x4e6e4b41, "radix_tree_delete" },
	{ 0xdc7c059d, "vm_insert_page" },
	{ 0x4ea5d10, "ksize" },
	{ 0x9de96117, "__task_pid_nr_ns" },
	{ 0x9f46ced8, "__sw_hweight64" },
	{ 0xf11543ff, "find_first_zero_bit" },
	{ 0x406681dd, "cdev_add" },
	{ 0x615911d7, "__bitmap_set" },
	{ 0xc808868a, "nvUvmInterfaceDestroyAccessCntrInfo" },
	{ 0x7cd8d75e, "page_offset_base" },
	{ 0xdac220d4, "nvUvmInterfaceHasPendingNonReplayableFaults" },
	{ 0x2f379285, "find_vma" },
	{ 0x5f3ab67a, "dma_direct_unmap_page" },
	{ 0x16e297c3, "bit_wait" },
	{ 0x9f984513, "strrchr" },
	{ 0x6b27729b, "radix_tree_gang_lookup" },
	{ 0x40a9b349, "vzalloc" },
	{ 0x80299be8, "kmem_cache_alloc" },
	{ 0x13c14232, "__free_pages" },
	{ 0xb601be4c, "__x86_indirect_thunk_rdx" },
	{ 0xb67a4af7, "nvUvmInterfaceRetainChannel" },
	{ 0x69049cd2, "radix_tree_replace_slot" },
	{ 0x12a38747, "usleep_range" },
	{ 0x41efdeaf, "radix_tree_lookup_slot" },
	{ 0xb2fcb56d, "queue_delayed_work_on" },
	{ 0xdecd0b29, "__stack_chk_fail" },
	{ 0x76aefd44, "get_user_pages" },
	{ 0x9cb986f2, "vmalloc_base" },
	{ 0xadfdfcef, "__bitmap_andnot" },
	{ 0x1000e51, "schedule" },
	{ 0x1d24c881, "___ratelimit" },
	{ 0x6b2dc060, "dump_stack" },
	{ 0x8c277752, "nvUvmInterfaceGetFbInfo" },
	{ 0x657fa9f8, "nvUvmInterfaceGetNonReplayableFaults" },
	{ 0x2ea2c95c, "__x86_indirect_thunk_rax" },
	{ 0xbfdcb43a, "__x86_indirect_thunk_r11" },
	{ 0xce96a1a4, "nvUvmInterfaceP2pObjectDestroy" },
	{ 0x3129b92e, "wake_up_process" },
	{ 0xc51ebf8e, "nvUvmInterfaceRegisterGpu" },
	{ 0xbdfb6dbb, "__fentry__" },
	{ 0xcbd4898c, "fortify_panic" },
	{ 0xc3ff38c2, "down_read_trylock" },
	{ 0xf1fcc453, "nvUvmInterfaceGetExternalAllocPtes" },
	{ 0x26c2e0b5, "kmem_cache_alloc_trace" },
	{ 0x944c43f, "node_states" },
	{ 0xdbf17652, "_raw_spin_lock" },
	{ 0x51760917, "_raw_spin_lock_irqsave" },
	{ 0xba548bb9, "nvUvmInterfaceDupAddressSpace" },
	{ 0xa5526619, "rb_insert_color" },
	{ 0xb776a08b, "kmem_cache_create" },
	{ 0x9ea53d7f, "vsnprintf" },
	{ 0x3eeb2322, "__wake_up" },
	{ 0xb3f7646e, "kthread_should_stop" },
	{ 0x8c26d495, "prepare_to_wait_event" },
	{ 0xa47bc098, "proc_create_data" },
	{ 0x39b52d19, "__bitmap_and" },
	{ 0x1f433b7d, "seq_lseek" },
	{ 0xd1600093, "nvUvmInterfaceOwnPageFaultIntr" },
	{ 0x37a0cba, "kfree" },
	{ 0x94961283, "vunmap" },
	{ 0x4d7a107f, "nvUvmInterfaceMemoryCpuUnMap" },
	{ 0x7ef5c587, "nvUvmInterfaceSetPageDirectory" },
	{ 0x90dea462, "unmap_mapping_range" },
	{ 0x69acdf38, "memcpy" },
	{ 0xe13348be, "proc_mkdir_mode" },
	{ 0xbb22731a, "nvUvmInterfacePmaFreePages" },
	{ 0xcf2a6966, "up" },
	{ 0x9929cf99, "fget" },
	{ 0xf05c7b8, "__x86_indirect_thunk_r15" },
	{ 0x53569707, "this_cpu_off" },
	{ 0x74c134b9, "__sw_hweight32" },
	{ 0xb352177e, "find_first_bit" },
	{ 0x92540fbf, "finish_wait" },
	{ 0x70ad75fb, "radix_tree_lookup" },
	{ 0x63c4d61f, "__bitmap_weight" },
	{ 0x3b644591, "__bitmap_shift_left" },
	{ 0xb83fa4e7, "nvUvmInterfaceMemoryCpuMap" },
	{ 0x494e3393, "vm_get_page_prot" },
	{ 0x29361773, "complete" },
	{ 0x656e4a6e, "snprintf" },
	{ 0x7a6e5761, "nvUvmInterfaceMemoryAllocSys" },
	{ 0x37c3bf2f, "vmalloc_to_page" },
	{ 0x43e66ecb, "nvUvmInterfaceReleaseChannel" },
	{ 0xf3928810, "nvUvmInterfaceInitAccessCntrInfo" },
	{ 0x4bc0e431, "nvUvmInterfaceUnsetPageDirectory" },
	{ 0x362ef408, "_copy_from_user" },
	{ 0x6fbc6a00, "radix_tree_insert" },
	{ 0x2e4948d6, "param_ops_uint" },
	{ 0x7b4da6ff, "__init_rwsem" },
	{ 0x9e7d6bd0, "__udelay" },
	{ 0xa5cb64fd, "dma_ops" },
	{ 0x88db9f48, "__check_object_size" },
	{ 0xe3ec2f2b, "alloc_chrdev_region" },
	{ 0x1bd44824, "__put_page" },
	{ 0xb2e108d5, "nvUvmInterfaceQueryCaps" },
	{ 0xb1e12d81, "krealloc" },
	{ 0xcd40176d, "nvUvmInterfaceChannelDestroy" },
	{ 0xaf0d91a7, "nvUvmInterfaceDisableAccessCntr" },
	{ 0x587f22d7, "devmap_managed_key" },
	{ 0x8a35b432, "sme_me_mask" },
};

MODULE_INFO(depends, "nvidia");


MODULE_INFO(srcversion, "60B362366F4E73C31E9BEBB");

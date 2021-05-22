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
	{ 0x188ea314, "jiffies_to_timespec64" },
	{ 0x6bc3fbc0, "__unregister_chrdev" },
	{ 0xd29dd5a4, "acpi_bus_register_driver" },
	{ 0x83c66b, "kmem_cache_destroy" },
	{ 0x4e768480, "address_space_init_once" },
	{ 0xe4004761, "dma_direct_unmap_sg" },
	{ 0xcf6885d0, "kernel_write" },
	{ 0x85bd1608, "__request_region" },
	{ 0x8b66e8a1, "cdev_del" },
	{ 0x8537dfba, "kmalloc_caches" },
	{ 0x44b6f301, "pci_bus_type" },
	{ 0xcc9a1e28, "pci_write_config_dword" },
	{ 0xeb233a45, "__kmalloc" },
	{ 0xb8c2987b, "cdev_init" },
	{ 0x1ed8b599, "__x86_indirect_thunk_r8" },
	{ 0x53b954a2, "up_read" },
	{ 0xc590c0f9, "__put_devmap_managed_page" },
	{ 0xef82cab8, "pci_write_config_word" },
	{ 0x1c58427f, "acpi_remove_notify_handler" },
	{ 0xd6ee688f, "vmalloc" },
	{ 0xfd93ee35, "ioremap_wc" },
	{ 0x349cba85, "strchr" },
	{ 0x9875a8e1, "single_open" },
	{ 0xd732ea78, "vga_set_legacy_decoding" },
	{ 0x3184b6ef, "param_ops_int" },
	{ 0x77358855, "iomem_resource" },
	{ 0x754d539c, "strlen" },
	{ 0x735e6a81, "acpi_evaluate_integer" },
	{ 0xb8e476e2, "pci_read_config_byte" },
	{ 0x27bbf221, "disable_irq_nosync" },
	{ 0x170ddf79, "acpi_install_notify_handler" },
	{ 0x263ed23b, "__x86_indirect_thunk_r12" },
	{ 0x5fb55c0f, "set_pages_array_uc" },
	{ 0xb0be3e1e, "pci_stop_and_remove_bus_device" },
	{ 0xf039960f, "dma_set_mask" },
	{ 0x3e230885, "single_release" },
	{ 0x1ce83cd4, "node_data" },
	{ 0xdc595427, "seq_puts" },
	{ 0x445a81ce, "boot_cpu_data" },
	{ 0x37b69db4, "pci_disable_device" },
	{ 0xabf63482, "pci_disable_msix" },
	{ 0x4ad904da, "set_page_dirty_lock" },
	{ 0xb67a9621, "backlight_device_register" },
	{ 0x20000329, "simple_strtoul" },
	{ 0xc0a3d105, "find_next_bit" },
	{ 0x5de2f866, "acpi_bus_get_device" },
	{ 0xdf566a59, "__x86_indirect_thunk_r9" },
	{ 0x81b395b3, "down_interruptible" },
	{ 0xa1f9a134, "__x86_indirect_thunk_rsi" },
	{ 0xe9a8fe46, "seq_printf" },
	{ 0x56470118, "__warn_printk" },
	{ 0xb7a8281, "remove_proc_entry" },
	{ 0x13ecd86f, "dma_direct_sync_single_for_cpu" },
	{ 0xf1e7b114, "pm_vt_switch_unregister" },
	{ 0x2db10a5b, "pci_get_class" },
	{ 0xdf8c695a, "__ndelay" },
	{ 0x2db10423, "__register_chrdev" },
	{ 0xc29957c3, "__x86_indirect_thunk_rcx" },
	{ 0x3441b5de, "filp_close" },
	{ 0x87b8798d, "sg_next" },
	{ 0xd92deb6b, "acpi_evaluate_object" },
	{ 0x75170705, "pci_write_config_byte" },
	{ 0xeae3dfd6, "__const_udelay" },
	{ 0x72f67319, "pci_release_regions" },
	{ 0x9421314f, "acpi_bus_unregister_driver" },
	{ 0xa843805a, "get_unused_fd_flags" },
	{ 0xc6f46339, "init_timer_key" },
	{ 0x409bcb62, "mutex_unlock" },
	{ 0x85df9b6c, "strsep" },
	{ 0x6091b333, "unregister_chrdev_region" },
	{ 0x999e8297, "vfree" },
	{ 0x59c0860b, "dma_free_attrs" },
	{ 0x4629334c, "__preempt_count" },
	{ 0x7a2af7b4, "cpu_number" },
	{ 0x97651e6c, "vmemmap_base" },
	{ 0xfee7c6cb, "seq_read" },
	{ 0xa1573cf1, "__alloc_pages_nodemask" },
	{ 0x344fd44f, "pv_ops" },
	{ 0xb6ab5a02, "dma_set_coherent_mask" },
	{ 0x4f38cf60, "kthread_create_on_node" },
	{ 0x15ba50a6, "jiffies" },
	{ 0x228e446b, "i2c_add_adapter" },
	{ 0x9f4f2aa3, "acpi_gbl_FADT" },
	{ 0x668b19a1, "down_read" },
	{ 0xe2d5255a, "strcmp" },
	{ 0x3f393e9e, "proc_remove" },
	{ 0xece784c2, "rb_first" },
	{ 0xc631580a, "console_unlock" },
	{ 0xd9a5ea54, "__init_waitqueue_head" },
	{ 0xb44ad4b3, "_copy_to_user" },
	{ 0x9968327e, "PDE_DATA" },
	{ 0x17de3d5, "nr_cpu_ids" },
	{ 0x6de13801, "wait_for_completion" },
	{ 0xaeeaf0fe, "kernel_read" },
	{ 0x8848086a, "param_ops_charp" },
	{ 0x2ea912e9, "pci_set_master" },
	{ 0xf1d0f6bb, "pm_vt_switch_required" },
	{ 0x97934ecf, "del_timer_sync" },
	{ 0x7e526bfa, "__x86_indirect_thunk_r10" },
	{ 0x1831ea8b, "follow_pfn" },
	{ 0xfb578fc5, "memset" },
	{ 0xc9b3fe84, "vmf_insert_pfn_prot" },
	{ 0x9e683f75, "__cpu_possible_mask" },
	{ 0xd38cd261, "__default_kernel_pte_mask" },
	{ 0x97345062, "vga_tryget" },
	{ 0x11089ac7, "_ctype" },
	{ 0x2a1cde84, "pci_iounmap" },
	{ 0x3812050a, "_raw_spin_unlock_irqrestore" },
	{ 0x4e0ecf27, "current_task" },
	{ 0xfc7e2596, "down_trylock" },
	{ 0x2db3d320, "mutex_lock_interruptible" },
	{ 0x977f511b, "__mutex_init" },
	{ 0xc5850110, "printk" },
	{ 0xbcab6ee6, "sscanf" },
	{ 0x7d5a7d0b, "kthread_stop" },
	{ 0x449ad0a7, "memcmp" },
	{ 0x7023bea8, "unregister_acpi_notifier" },
	{ 0x9ec6ca96, "ktime_get_real_ts64" },
	{ 0x1edb69d6, "ktime_get_raw_ts64" },
	{ 0xda4515cf, "vmap" },
	{ 0xf1e046cc, "panic" },
	{ 0x4c9d28b0, "phys_base" },
	{ 0x531b604e, "__virt_addr_valid" },
	{ 0xe7b00dfb, "__x86_indirect_thunk_r13" },
	{ 0xaafdc258, "strcasecmp" },
	{ 0xa1c76e0a, "_cond_resched" },
	{ 0x4d9b652b, "rb_erase" },
	{ 0x9166fada, "strncpy" },
	{ 0x365acda7, "set_normalized_timespec64" },
	{ 0x5a921311, "strncmp" },
	{ 0xfbaaf01e, "console_lock" },
	{ 0x37f74190, "pci_read_config_word" },
	{ 0x13f29d6a, "dma_direct_map_page" },
	{ 0x593c1bac, "__x86_indirect_thunk_rbx" },
	{ 0xfb481954, "vprintk" },
	{ 0xc08ca96, "dma_alloc_attrs" },
	{ 0x5f5f1453, "kmem_cache_free" },
	{ 0x2ab7989d, "mutex_lock" },
	{ 0xa94a09bb, "mem_section" },
	{ 0x3faa3b9d, "pci_get_domain_bus_and_slot" },
	{ 0x18c51305, "__close_fd" },
	{ 0xf1969a8e, "__usecs_to_jiffies" },
	{ 0x1e6d26a8, "strstr" },
	{ 0x6626afca, "down" },
	{ 0xc38c83b8, "mod_timer" },
	{ 0xce8b1878, "__x86_indirect_thunk_r14" },
	{ 0x2072ee9b, "request_threaded_irq" },
	{ 0xce807a25, "up_write" },
	{ 0x57bc19d2, "down_write" },
	{ 0xa2de16e5, "fput" },
	{ 0xa0f493d9, "efi" },
	{ 0xae8a24fc, "pci_enable_msi" },
	{ 0xfe487975, "init_wait_entry" },
	{ 0x1d1036ed, "pci_clear_master" },
	{ 0xdc7c059d, "vm_insert_page" },
	{ 0x61651be, "strcat" },
	{ 0x93c2c11f, "pci_find_capability" },
	{ 0x406681dd, "cdev_add" },
	{ 0x9975dc22, "acpi_get_handle" },
	{ 0x45e1f589, "dma_direct_map_resource" },
	{ 0x7cd8d75e, "page_offset_base" },
	{ 0x6ec536e, "module_put" },
	{ 0x2f379285, "find_vma" },
	{ 0xc6cbbc89, "capable" },
	{ 0x5f3ab67a, "dma_direct_unmap_page" },
	{ 0x875bbf9e, "i2c_del_adapter" },
	{ 0xc2ad0655, "iterate_fd" },
	{ 0x80299be8, "kmem_cache_alloc" },
	{ 0x13c14232, "__free_pages" },
	{ 0xb601be4c, "__x86_indirect_thunk_rdx" },
	{ 0x618911fc, "numa_node" },
	{ 0x93a219c, "ioremap_nocache" },
	{ 0x60d2298f, "set_pages_array_wb" },
	{ 0xfc62f3db, "pci_enable_msix_range" },
	{ 0x37b8b39e, "screen_info" },
	{ 0xb665f56d, "__cachemode2pte_tbl" },
	{ 0x973fa82e, "register_acpi_notifier" },
	{ 0x6a5cb5ee, "__get_free_pages" },
	{ 0xdecd0b29, "__stack_chk_fail" },
	{ 0x76aefd44, "get_user_pages" },
	{ 0x9cb986f2, "vmalloc_base" },
	{ 0x8ddd8aad, "schedule_timeout" },
	{ 0x1000e51, "schedule" },
	{ 0x2ea2c95c, "__x86_indirect_thunk_rax" },
	{ 0xbfdcb43a, "__x86_indirect_thunk_r11" },
	{ 0x4a3b5e69, "pci_read_config_dword" },
	{ 0x577cebb0, "dev_driver_string" },
	{ 0x7f24de73, "jiffies_to_usecs" },
	{ 0x77b1976d, "wake_up_process" },
	{ 0xbdfb6dbb, "__fentry__" },
	{ 0xd6eaaea1, "full_name_hash" },
	{ 0x1035c7c2, "__release_region" },
	{ 0xcbd4898c, "fortify_panic" },
	{ 0xdad5fe87, "pci_unregister_driver" },
	{ 0xc3ff38c2, "down_read_trylock" },
	{ 0x26c2e0b5, "kmem_cache_alloc_trace" },
	{ 0x944c43f, "node_states" },
	{ 0xdbf17652, "_raw_spin_lock" },
	{ 0x51760917, "_raw_spin_lock_irqsave" },
	{ 0xa5526619, "rb_insert_color" },
	{ 0xb776a08b, "kmem_cache_create" },
	{ 0x9ea53d7f, "vsnprintf" },
	{ 0x4302d0eb, "free_pages" },
	{ 0x3eeb2322, "__wake_up" },
	{ 0xb3f7646e, "kthread_should_stop" },
	{ 0x8c26d495, "prepare_to_wait_event" },
	{ 0xa47bc098, "proc_create_data" },
	{ 0x1f433b7d, "seq_lseek" },
	{ 0xfcec0987, "enable_irq" },
	{ 0x7812c047, "__vmalloc" },
	{ 0x37a0cba, "kfree" },
	{ 0x94961283, "vunmap" },
	{ 0x2e7484fe, "dma_direct_map_sg" },
	{ 0xee9380f8, "remap_pfn_range" },
	{ 0x90dea462, "unmap_mapping_range" },
	{ 0x69acdf38, "memcpy" },
	{ 0xbea0b906, "pci_request_regions" },
	{ 0xd9146c96, "fd_install" },
	{ 0xf4bc1d4a, "pci_disable_msi" },
	{ 0x6128b5fc, "__printk_ratelimit" },
	{ 0xedc03953, "iounmap" },
	{ 0xe13348be, "proc_mkdir_mode" },
	{ 0xd71ce18e, "pcibios_resource_to_bus" },
	{ 0xcf2a6966, "up" },
	{ 0x556422b3, "ioremap_cache" },
	{ 0x9929cf99, "fget" },
	{ 0xf7a843ec, "dma_direct_sync_single_for_device" },
	{ 0xab738f17, "__pci_register_driver" },
	{ 0xf05c7b8, "__x86_indirect_thunk_r15" },
	{ 0x9aa94788, "sg_alloc_table_from_pages" },
	{ 0xb352177e, "find_first_bit" },
	{ 0x91607d95, "set_memory_wb" },
	{ 0x92540fbf, "finish_wait" },
	{ 0x63c4d61f, "__bitmap_weight" },
	{ 0x7f5b4fe4, "sg_free_table" },
	{ 0x76b844a6, "pci_dev_put" },
	{ 0x29361773, "complete" },
	{ 0x656e4a6e, "snprintf" },
	{ 0xd979a547, "__x86_indirect_thunk_rdi" },
	{ 0x874fcd1b, "pci_iomap" },
	{ 0x37c3bf2f, "vmalloc_to_page" },
	{ 0xec2b8a42, "acpi_walk_namespace" },
	{ 0x294b9ea1, "on_each_cpu" },
	{ 0x99db8eb8, "pci_enable_device" },
	{ 0x362ef408, "_copy_from_user" },
	{ 0xc278c8c, "backlight_device_unregister" },
	{ 0x1b539f5d, "is_acpi_device_node" },
	{ 0x7b4da6ff, "__init_rwsem" },
	{ 0x9e7d6bd0, "__udelay" },
	{ 0xa5cb64fd, "dma_ops" },
	{ 0x88db9f48, "__check_object_size" },
	{ 0xe3ec2f2b, "alloc_chrdev_region" },
	{ 0x1bd44824, "__put_page" },
	{ 0xddce0ebf, "try_module_get" },
	{ 0xc1514a3b, "free_irq" },
	{ 0xab65ed80, "set_memory_uc" },
	{ 0xe914e41e, "strcpy" },
	{ 0xacc27b0b, "filp_open" },
	{ 0x587f22d7, "devmap_managed_key" },
	{ 0x9305f8e6, "cpufreq_get" },
	{ 0x8a35b432, "sme_me_mask" },
};

MODULE_INFO(depends, "");

MODULE_ALIAS("pci:v000010DEd*sv*sd*bc03sc00i00*");
MODULE_ALIAS("pci:v000010DEd*sv*sd*bc03sc02i00*");

MODULE_INFO(srcversion, "C3DA452452B379AF9F05AC7");

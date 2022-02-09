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
	{ 0xb3753869, "module_layout" },
	{ 0x35216b26, "kmalloc_caches" },
	{ 0xeb233a45, "__kmalloc" },
	{ 0xf9a482f9, "msleep" },
	{ 0x1ed8b599, "__x86_indirect_thunk_r8" },
	{ 0x53b954a2, "up_read" },
	{ 0xd6ee688f, "vmalloc" },
	{ 0xf6fa2eb9, "single_open" },
	{ 0x5ab5b891, "param_ops_int" },
	{ 0x754d539c, "strlen" },
	{ 0x263ed23b, "__x86_indirect_thunk_r12" },
	{ 0x79aa04a2, "get_random_bytes" },
	{ 0x3700313a, "single_release" },
	{ 0xad9670f1, "seq_puts" },
	{ 0xdf566a59, "__x86_indirect_thunk_r9" },
	{ 0x81b395b3, "down_interruptible" },
	{ 0xa1f9a134, "__x86_indirect_thunk_rsi" },
	{ 0x56470118, "__warn_printk" },
	{ 0xc29957c3, "__x86_indirect_thunk_rcx" },
	{ 0xededc3c3, "param_ops_bool" },
	{ 0xc6f46339, "init_timer_key" },
	{ 0x999e8297, "vfree" },
	{ 0x4629334c, "__preempt_count" },
	{ 0x4e22ab94, "seq_read" },
	{ 0xe8bc695c, "kthread_create_on_node" },
	{ 0x15ba50a6, "jiffies" },
	{ 0x668b19a1, "down_read" },
	{ 0xe2d5255a, "strcmp" },
	{ 0x7240ad89, "proc_remove" },
	{ 0xd9a5ea54, "__init_waitqueue_head" },
	{ 0xb44ad4b3, "_copy_to_user" },
	{ 0xe8c07ff2, "PDE_DATA" },
	{ 0xfb6cca55, "nvidia_unregister_module" },
	{ 0x6de13801, "wait_for_completion" },
	{ 0x97934ecf, "del_timer_sync" },
	{ 0x7e526bfa, "__x86_indirect_thunk_r10" },
	{ 0xfb578fc5, "memset" },
	{ 0x3812050a, "_raw_spin_unlock_irqrestore" },
	{ 0x1b44c663, "current_task" },
	{ 0xc5850110, "printk" },
	{ 0xa5cc5d9f, "kthread_stop" },
	{ 0x449ad0a7, "memcmp" },
	{ 0x9ec6ca96, "ktime_get_real_ts64" },
	{ 0xe7b00dfb, "__x86_indirect_thunk_r13" },
	{ 0xa1c76e0a, "_cond_resched" },
	{ 0x9166fada, "strncpy" },
	{ 0x593c1bac, "__x86_indirect_thunk_rbx" },
	{ 0x6626afca, "down" },
	{ 0xc38c83b8, "mod_timer" },
	{ 0xce807a25, "up_write" },
	{ 0x57bc19d2, "down_write" },
	{ 0x92d1dfbd, "fput" },
	{ 0xb601be4c, "__x86_indirect_thunk_rdx" },
	{ 0xdecd0b29, "__stack_chk_fail" },
	{ 0x9cb986f2, "vmalloc_base" },
	{ 0x1000e51, "schedule" },
	{ 0x193c3c08, "nvidia_get_rm_ops" },
	{ 0x6b2dc060, "dump_stack" },
	{ 0x2ea2c95c, "__x86_indirect_thunk_rax" },
	{ 0xbfdcb43a, "__x86_indirect_thunk_r11" },
	{ 0x64c17a3f, "wake_up_process" },
	{ 0xbdfb6dbb, "__fentry__" },
	{ 0xc3ff38c2, "down_read_trylock" },
	{ 0xf5cb25c8, "kmem_cache_alloc_trace" },
	{ 0x51760917, "_raw_spin_lock_irqsave" },
	{ 0x9ea53d7f, "vsnprintf" },
	{ 0x3eeb2322, "__wake_up" },
	{ 0xb3f7646e, "kthread_should_stop" },
	{ 0xab62e5d0, "proc_create_data" },
	{ 0xf77b56e2, "seq_lseek" },
	{ 0x37a0cba, "kfree" },
	{ 0x69acdf38, "memcpy" },
	{ 0xbc18c721, "proc_mkdir_mode" },
	{ 0xcf2a6966, "up" },
	{ 0x45fbd08d, "fget" },
	{ 0xf05c7b8, "__x86_indirect_thunk_r15" },
	{ 0x18970505, "nvidia_register_module" },
	{ 0x29361773, "complete" },
	{ 0xb0e602eb, "memmove" },
	{ 0xcc78395e, "vmalloc_to_page" },
	{ 0xe3fffae9, "__x86_indirect_thunk_rbp" },
	{ 0x362ef408, "_copy_from_user" },
	{ 0x7b4da6ff, "__init_rwsem" },
	{ 0x9e7d6bd0, "__udelay" },
	{ 0x88db9f48, "__check_object_size" },
};

MODULE_INFO(depends, "nvidia");


MODULE_INFO(srcversion, "CDF2864BA4579DD74B9136A");

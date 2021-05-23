#define NV_ACPI_DEVICE_OPS_REMOVE_ARGUMENT_COUNT 1
#undef NV_OUTER_FLUSH_ALL_PRESENT
#undef NV_FILE_OPERATIONS_HAS_IOCTL
#define NV_FILE_HAS_INODE
#define NV_KUID_T_PRESENT
#define NV_DMA_OPS_PRESENT
#undef NV_SWIOTLB_DMA_OPS_PRESENT
#undef NV_NONCOHERENT_SWIOTLB_DMA_OPS_PRESENT
#define NV_VM_FAULT_HAS_ADDRESS
#define NV_BACKLIGHT_PROPERTIES_TYPE_PRESENT
#undef NV_VM_INSERT_PFN_PROT_PRESENT
#define NV_VMF_INSERT_PFN_PROT_PRESENT
#define NV_ADDRESS_SPACE_INIT_ONCE_PRESENT
#define NV_VM_OPS_FAULT_REMOVED_VMA_ARG
#define NV_VMBUS_CHANNEL_HAS_RING_BUFFER_PAGE
#define NV_DEVICE_DRIVER_OF_MATCH_TABLE_PRESENT
#define NV_DEVICE_OF_NODE_PRESENT
#define NV_NODE_STATES_N_MEMORY_PRESENT
#undef NV_KMEM_CACHE_HAS_KOBJ_REMOVE_WORK
#undef NV_SYSFS_SLAB_UNLINK_PRESENT
#undef NV_PROC_OPS_PRESENT
#define NV_TIMESPEC64_PRESENT
#define NV_VMALLOC_HAS_PGPROT_T_ARG
#define NV_ACPI_FADT_LOW_POWER_S0_FLAG_PRESENT
#undef NV_MM_HAS_MMAP_LOCK
#define NV_PCI_CHANNEL_STATE_PRESENT
#undef NV_ADDRESS_SPACE_HAS_RWLOCK_TREE_LOCK
#undef NV_ADDRESS_SPACE_HAS_BACKING_DEV_INFO
#undef NV_MM_CONTEXT_T_HAS_ID
#define NV_GET_USER_PAGES_REMOTE_PRESENT
#undef NV_GET_USER_PAGES_REMOTE_HAS_WRITE_AND_FORCE_ARGS
#define NV_GET_USER_PAGES_REMOTE_HAS_TSK_ARG
#define NV_GET_USER_PAGES_REMOTE_HAS_LOCKED_ARG
#undef NV_GET_USER_PAGES_HAS_WRITE_AND_FORCE_ARGS
#undef NV_GET_USER_PAGES_HAS_TASK_STRUCT
#define NV_VM_FAULT_T_IS_PRESENT
#define NV_MMU_NOTIFIER_OPS_HAS_INVALIDATE_RANGE
#undef NV_DRM_BUS_PRESENT
#undef NV_DRM_BUS_HAS_BUS_TYPE
#undef NV_DRM_BUS_HAS_GET_IRQ
#undef NV_DRM_BUS_HAS_GET_NAME
#undef NV_DRM_DRIVER_HAS_DEVICE_LIST
#define NV_DRM_DRIVER_HAS_LEGACY_DEV_LIST
#undef NV_DRM_DRIVER_HAS_SET_BUSID
#define NV_DRM_CRTC_STATE_HAS_CONNECTORS_CHANGED
#define NV_DRM_CRTC_INIT_WITH_PLANES_HAS_NAME_ARG
#define NV_DRM_ENCODER_INIT_HAS_NAME_ARG
#define NV_DRM_UNIVERSAL_PLANE_INIT_HAS_FORMAT_MODIFIERS_ARG
#define NV_DRM_UNIVERSAL_PLANE_INIT_HAS_NAME_ARG
#define NV_DRM_HELPER_MODE_FILL_FB_STRUCT_HAS_DEV_ARG
#define NV_DRM_HELPER_MODE_FILL_FB_STRUCT_HAS_CONST_MODE_CMD_ARG
#undef NV_DRM_MASTER_DROP_HAS_FROM_RELEASE_ARG
#undef NV_DRM_DRIVER_UNLOAD_HAS_INT_RETURN_TYPE
#undef NV_DRM_ATOMIC_HELPER_CRTC_DESTROY_STATE_HAS_CRTC_ARG
#define NV_DRM_MODE_OBJECT_FIND_HAS_FILE_PRIV_ARG
#define NV_DMA_BUF_OWNER_PRESENT
#define NV_DRM_CONNECTOR_LIST_ITER_PRESENT
#define NV_DRM_CONNECTOR_LIST_ITER_BEGIN_PRESENT
#define NV_DRM_ATOMIC_HELPER_SWAP_STATE_HAS_STALL_ARG
#define NV_DRM_ATOMIC_HELPER_SWAP_STATE_RETURN_INT
#undef NV_DRM_DRIVER_PRIME_FLAG_PRESENT
#define NV_DRM_GEM_OBJECT_HAS_RESV
#define NV_DRM_CRTC_STATE_HAS_ASYNC_FLIP
#undef NV_DRM_CRTC_STATE_HAS_PAGEFLIP_FLAGS
#define NV_DRM_FORMAT_MODIFIERS_PRESENT
#define NV_DRM_VMA_NODE_IS_ALLOWED_HAS_TAG_ARG
#undef NV_DRM_VMA_OFFSET_NODE_HAS_READONLY
#define NV_DRM_DISPLAY_MODE_HAS_VREFRESH types
#define NV_DRM_DRIVER_SET_MASTER_HAS_INT_RETURN_TYPE
#define NV_DRM_DRIVER_HAS_GEM_FREE_OBJECT
#undef NV_DRM_PRIME_PAGES_TO_SG_HAS_DRM_DEVICE_ARG
#define NV_DRM_DRIVER_HAS_GEM_PRIME_CALLBACKS
#undef NV_DRM_CRTC_ATOMIC_CHECK_HAS_ATOMIC_STATE_ARG
#undef NV_DRM_GEM_OBJECT_VMAP_HAS_MAP_ARG

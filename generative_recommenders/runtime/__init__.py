from generative_recommenders.runtime.device import (  # noqa: F401
    autocast_device_type,
    can_use_bf16,
    detect_accelerator,
    dist_backend_for_accelerator,
    get_device_count,
    get_device_for_rank,
    set_current_device,
)

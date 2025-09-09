from contextlib import contextmanager

import paddle


@contextmanager
def custom_sdp_kernel(enable_math: bool = False, enable_flash: bool = True, enable_mem_efficient: bool = True):
    """Customize Paddle Attention Backend Controller"""
    # Get the original module reference
    flash_module = paddle.nn.functional.flash_attention

    # Save original global variable values
    original_settings = (flash_module.g_enable_math, flash_module.g_enable_flash, flash_module.g_enable_mem_efficient)

    try:
        # Set a new global variable value
        flash_module.g_enable_math = enable_math
        flash_module.g_enable_flash = enable_flash
        flash_module.g_enable_mem_efficient = enable_mem_efficient

        yield
    finally:
        # Restore original global variable values
        flash_module.g_enable_math, flash_module.g_enable_flash, flash_module.g_enable_mem_efficient = (
            original_settings
        )

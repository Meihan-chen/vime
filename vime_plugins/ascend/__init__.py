"""Construct Ascend platform integration without importing device runtimes or patching libraries."""


def build_integration():
    from vime.utils.platform_integration import PlatformIntegration

    from .device_env import NpuDeviceEnv
    from .training import AscendMegatronIntegration
    from .weight_transfer import NpuWeightTransfer

    return PlatformIntegration(
        env=NpuDeviceEnv(),
        megatron=AscendMegatronIntegration(),
        weight_transfer=NpuWeightTransfer(),
    )

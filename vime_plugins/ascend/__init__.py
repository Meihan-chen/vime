"""Construct Ascend support without importing device runtimes or patching libraries."""


def build_support():
    from vime.utils.support import Support

    from .device_env import NpuDeviceEnv
    from .training import NpuMegatronSupport
    from .weight_transfer import NpuWeightTransfer

    return Support(NpuDeviceEnv(), NpuMegatronSupport(), NpuWeightTransfer())

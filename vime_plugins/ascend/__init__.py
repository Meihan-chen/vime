"""Lightweight Ascend components; vendor runtimes load at backend entry points."""

from .device_env import NpuDeviceEnv
from .training import AscendMegatronIntegration
from .weight_transfer import NpuWeightTransfer

env = NpuDeviceEnv()
megatron = AscendMegatronIntegration()
weight_transfer = NpuWeightTransfer()

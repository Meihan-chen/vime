"""Ascend's vLLM transfer factories and IPC identity, loaded on first use."""

from typing import Any

from vime.backends.vllm_weight_transfer import Transport, WeightTransfer


class NpuWeightTransfer(WeightTransfer):
    def backend(self, transport: Transport) -> str:
        return {"ipc": "npu_ipc", "distributed": "hccl"}[transport]

    def init_info(self, transport: Transport, **kwargs: Any) -> Any:
        import torch_npu  # noqa: F401
        from vllm.plugins import load_general_plugins

        load_general_plugins()
        if transport == "ipc":
            from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCTrainerInitInfo

            return NPUIPCTrainerInitInfo(**kwargs)
        if transport == "distributed":
            from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLTrainerInitInfo

            return HCCLTrainerInitInfo(**kwargs)
        raise ValueError(f"Unknown weight transport: {transport!r}")

    def ipc_handle_key(self) -> str:
        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import npu_generate_uuid

        return npu_generate_uuid()

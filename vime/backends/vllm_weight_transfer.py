"""vLLM transfer policy, outside the package that initializes the device runtime."""

from typing import Any, Literal

Transport = Literal["ipc", "distributed"]


class WeightTransfer:
    def backend(self, transport: Transport) -> str:
        return {"ipc": "ipc", "distributed": "nccl"}[transport]

    def init_info(self, transport: Transport, **kwargs: Any) -> Any:
        if transport == "ipc":
            from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo

            return IPCTrainerInitInfo(**kwargs)
        if transport == "distributed":
            from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo

            return NCCLTrainerInitInfo(**kwargs)
        raise ValueError(f"Unknown weight transport: {transport!r}")

    def requires_remote_cleanup(self, trainer: Any) -> bool:
        return trainer.init_info_cls.backend == self.backend("distributed")

    def ipc_handle_key(self) -> str:
        import torch

        return str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)


def current() -> WeightTransfer:
    from vime.utils.platform_integration import current_integration

    return current_integration().weight_transfer

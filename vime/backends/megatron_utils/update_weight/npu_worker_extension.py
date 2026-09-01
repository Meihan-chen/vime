"""Ascend-only vLLM worker compatibility hooks.

This module is selected by the NPU vLLM platform provider.  Vendor imports
remain inside the individual hooks so importing Vime's common weight-transfer
code does not require ``vllm_ascend``.

The hooks work around behavior in the vLLM/vLLM Ascend versions used by the
S0 baseline.  They are intentionally separate from trainer-side IPC
orchestration and can be removed independently once the corresponding vendor
fixes are available.
"""

from __future__ import annotations

import inspect

import torch


class _NPUVLLMHijack:
    """Install the temporary vLLM Ascend worker compatibility hooks."""

    @staticmethod
    def patch_npu_worker() -> None:
        from vllm_ascend.worker.worker import NPUWorker

        if getattr(NPUWorker, "_npu_worker_patched", False):
            return

        _NPUVLLMHijack.patch_one_worker(NPUWorker)
        NPUWorker._npu_worker_patched = True

    @staticmethod
    def patch_a3_moe_alltoall_expert_ids() -> None:
        """Restore the ALLTOALL expert-ID template after memory reuse."""
        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

        if get_ascend_device_type() != AscendDeviceType.A3:
            return

        from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAll2AllV

        if getattr(TokenDispatcherWithAll2AllV, "_vime_expert_ids_patched", False):
            return

        original_dispatch_preprocess = TokenDispatcherWithAll2AllV._dispatch_preprocess
        TokenDispatcherWithAll2AllV._vime_expert_ids_generation = 0

        def _patched_dispatch_preprocess(self, hidden_states, topk_ids):
            generation = TokenDispatcherWithAll2AllV._vime_expert_ids_generation
            if self.num_local_experts > 1 and getattr(self, "_vime_seen_expert_ids_generation", -1) != generation:
                expert_ids = self.expert_ids_per_ep_rank
                self.expert_ids_per_ep_rank = torch.arange(
                    self.num_experts,
                    device=expert_ids.device,
                    dtype=expert_ids.dtype,
                ).remainder(self.num_local_experts)
                self._vime_seen_expert_ids_generation = generation
            return original_dispatch_preprocess(self, hidden_states, topk_ids)

        TokenDispatcherWithAll2AllV._dispatch_preprocess = _patched_dispatch_preprocess
        TokenDispatcherWithAll2AllV._vime_expert_ids_patched = True

    @staticmethod
    def invalidate_moe_alltoall_expert_ids() -> None:
        try:
            from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAll2AllV
        except ImportError:
            return

        if getattr(TokenDispatcherWithAll2AllV, "_vime_expert_ids_patched", False):
            TokenDispatcherWithAll2AllV._vime_expert_ids_generation += 1

    @staticmethod
    def patch_one_worker(worker_cls: type) -> None:
        """Patch one worker class; exposed as a seam for focused tests."""
        original_load_model = worker_cls.load_model
        original_start_weight_update = worker_cls.start_weight_update
        original_wake_up = worker_cls.wake_up
        has_dummy_kw = "load_dummy_weights" in inspect.signature(original_load_model).parameters

        if has_dummy_kw:

            def _patched_load_model(self, *, load_dummy_weights: bool = False, _orig=original_load_model) -> None:
                _orig(self, load_dummy_weights=load_dummy_weights)
                _NPUVLLMHijack.patch_moe_weight_loader(self.model_runner.model)

        else:

            def _patched_load_model(self, _orig=original_load_model) -> None:
                _orig(self)
                _NPUVLLMHijack.patch_moe_weight_loader(self.model_runner.model)

        def _patched_start_weight_update(
            self, is_checkpoint_format: bool = True, _orig=original_start_weight_update
        ) -> None:
            _NPUVLLMHijack.patch_moe_weight_loader(self.model_runner.model)
            _orig(self, is_checkpoint_format=is_checkpoint_format)
            _NPUVLLMHijack.invalidate_moe_alltoall_expert_ids()

        def _patched_wake_up(self, tags=None, _orig=original_wake_up) -> None:
            quant_config = self.vllm_config.quant_config
            if quant_config is not None:
                _orig(self, tags=tags)
                _NPUVLLMHijack.invalidate_moe_alltoall_expert_ids()
                return

            # vLLM Ascend transposes unquantized w13_weight/w2_weight in
            # wake_up(). Keep its allocator/buffer restore, but skip that
            # branch: layerwise reload owns the final runtime layout.
            self.vllm_config.quant_config = object()
            try:
                _orig(self, tags=tags)
            finally:
                self.vllm_config.quant_config = quant_config
            _NPUVLLMHijack.invalidate_moe_alltoall_expert_ids()

        worker_cls.load_model = _patched_load_model  # type: ignore[attr-defined]
        worker_cls.start_weight_update = _patched_start_weight_update  # type: ignore[attr-defined]
        worker_cls.wake_up = _patched_wake_up  # type: ignore[attr-defined]

    @staticmethod
    def patch_moe_weight_loader(model: torch.nn.Module) -> None:
        inner_model = getattr(model, "model", None) or getattr(model, "language_model", None)
        if inner_model is None:
            return
        if not hasattr(inner_model, "layers"):
            inner_model = getattr(inner_model, "model", None)
            if inner_model is None or not hasattr(inner_model, "layers"):
                return

        for layer in inner_model.layers:
            mlp = getattr(layer, "mlp", None) or getattr(layer, "block_sparse_moe", None)
            if mlp is None:
                continue
            experts = getattr(mlp, "experts", None)
            if experts is None or not hasattr(experts, "weight_loader"):
                continue
            for name, param in mlp.named_parameters():
                if ("w13_weight" in name or "w2_weight" in name) and not hasattr(param, "weight_loader"):
                    param.weight_loader = experts.weight_loader  # type: ignore[attr-defined]

    @staticmethod
    def patch_npu_rotary_emb() -> None:
        from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

        if getattr(ApplyRotaryEmb, "_npu_rotary_patched", False):
            return

        def _npu_rotary_emb_init(
            self,
            enforce_enable: bool = False,
            is_neox_style: bool = True,
            enable_fp32_compute: bool = False,
        ) -> None:
            super(ApplyRotaryEmb, self).__init__(enforce_enable=enforce_enable)
            self.is_neox_style = is_neox_style
            self.enable_fp32_compute = enable_fp32_compute
            self.apply_rotary_emb_flash_attn = None

        ApplyRotaryEmb.__init__ = _npu_rotary_emb_init  # type: ignore[attr-defined]
        ApplyRotaryEmb._npu_rotary_patched = True


class vLLMColocateWorkerExtension:
    """NPU ``--worker-extension-cls`` entry for colocated rollout."""

    def __new__(cls, **kwargs):
        _NPUVLLMHijack.patch_a3_moe_alltoall_expert_ids()
        _NPUVLLMHijack.patch_npu_worker()
        _NPUVLLMHijack.patch_npu_rotary_emb()
        return super().__new__(cls)


class vLLMWorkerExtension:
    """NPU ``--worker-extension-cls`` entry for non-colocated rollout."""

    def __new__(cls, **kwargs):
        _NPUVLLMHijack.patch_npu_worker()
        _NPUVLLMHijack.patch_npu_rotary_emb()
        return super().__new__(cls)

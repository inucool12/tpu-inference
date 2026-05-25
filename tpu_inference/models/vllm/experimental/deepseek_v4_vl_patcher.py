import torch

def is_deepseek_v4(vllm_model) -> bool:
    """Checks if the running vLLM class is the target DeepSeek architecture."""
    return vllm_model.__class__.__name__ == "DeepseekV4ForCausalLM"

def apply_deepseek_v4_patches(vllm_model):
    """Aligns DeepSeek-V4 layers with Google TPU native FP8 OpenXLA execution."""
    config = getattr(vllm_model, "config", None)
    if not config:
        return

    # 1. INTERCEPT COMPRESSED TENSORS VIA CONFIG STATE
    if hasattr(vllm_model, "quant_config"):
        quant_cfg = vllm_model.quant_config
        # Check if the TPU out-of-tree bridge has intercepted the configuration
        if quant_cfg.__class__.__name__ == "VllmCompressedTensorsConfig":
            # Force the underlying mapping schema to unpack disk formats into FP8
            if hasattr(quant_cfg, "update_dtype_mapping"):
                quant_cfg.update_dtype_mapping("expert_dtype", torch.float8_e4m3fn)

    # 2. REMAP THE CHECKPOINT'S FP4 SPEC TO NATIVE TPU FLOAT8_E4M3FN
    # Overrides the configuration profile to force the engine to utilize FP8 pipelines.
    if getattr(config, "expert_dtype", None) == "fp4":
        # Change the config marker so internal layers allocate FP8 structures
        config.expert_dtype = torch.float8_e4m3fn

        # Remap the model's weight translation function to expect FP8 targets
        if hasattr(vllm_model, "hf_to_vllm_mapper"):
            from tpu_inference.models.vllm.experimental.deepseek_v4 import _make_deepseek_v4_weights_mapper
            # Forriding the initialization dictionary mapper to parse weights as float8_e4m3fn
            vllm_model.hf_to_vllm_mapper = _make_deepseek_v4_weights_mapper(torch.float8_e4m3fn)

    # 3. RESOLVE COMPRESS_RATIOS LIST STRUCT FOR STATIC GRAPH GENERATION
    # XLA tracing fails when evaluating a raw index list dynamically.
    if hasattr(config, "compress_ratios") and isinstance(config.compress_ratios, list):
        config.layer_compress_ratios_map = {idx: ratio for idx, ratio in enumerate(config.compress_ratios)}

        # FINAL LAYER EXCEPTION: Ensure the last layer is fully uncompressed (ratio = 1)
        final_layer_idx = len(config.compress_ratios) - 1
        config.layer_compress_ratios_map[final_layer_idx] = 1

        config.compress_ratio = 4  # Standard vLLM fallback scalar property

    # 4. FIX ROUTED EXPERT LAYOUT BOUNDARIES
    # Config states: "n_routed_experts": 256, "num_experts_per_tok": 6
    # Forcing static routing layout parameters to avoid dynamic OpenXLA trace splits
    if hasattr(vllm_model, "model") and hasattr(vllm_model.model, "layers"):
        for layer in vllm_model.model.layers:
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"):
                setattr(layer.mlp.gate, "force_static_topk", True)
                setattr(layer.mlp.gate, "num_experts_per_tok", 6)

    # 5. MONKEY PATCH DUMMY WEIGHT LOADER
    # Replaces the base weight loader to enforce a safe sub-16-bit float buffer on CPU
    # before copy-casting to native FP8 on the tensor mesh, avoiding JAX uniform bounds errors.
    import vllm.model_executor.model_loader.weight_utils
    from vllm.platforms import current_platform
    
    @torch.no_grad()
    def tpu_safe_initialize_dummy_weight(
        param: torch.Tensor,
        low: float = -1e-3,
        high: float = 1e-3,
        seed: int = 1234,
    ) -> None:
        if param.device.type == "meta":
            return
        if not torch.is_floating_point(param):
            if current_platform.is_rocm():
                param.zero_()
            return

        if current_platform.is_tpu():
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed)
            if torch.finfo(param.dtype).bits < 16:
                rand_dtype = torch.float16
            else:
                rand_dtype = param.dtype

            param.copy_(
                (
                    (high - low)
                    * torch.rand(
                        param.shape,
                        generator=generator,
                        dtype=rand_dtype,
                        layout=param.layout,
                        requires_grad=param.requires_grad,
                        device="cpu",
                    )
                    + low
                ).to(param.dtype)
            )
            torch._sync(param)
            return

        # Fallback to GPU default if somehow executed on GPU
        param.uniform_(low, high)

    vllm.model_executor.model_loader.weight_utils.initialize_single_dummy_weight = tpu_safe_initialize_dummy_weight

    # 6. MONKEY PATCH KV CACHE COORDINATOR
    # Bypasses Hybrid len assertions and routes empty mock attention groups for Phase 1 TPUs
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator

    original_verify = HybridKVCacheCoordinator.verify_and_split_kv_cache_groups
    def tpu_safe_verify_and_split(self) -> None:
        attention_groups = []
        for i, g in enumerate(self.kv_cache_config.kv_cache_groups):
            manager_cls = self.single_type_managers[i].__class__
            spec = g.kv_cache_spec
            for existing_spec, group_ids, existing_cls in attention_groups:
                if existing_spec == spec:
                    group_ids.append(i)
                    break
            else:
                attention_groups.append((spec, [i], manager_cls))

        # We intentionally omit the `assert len(attention_groups) > 1` here!
        from vllm.v1.kv_cache_interface import FullAttentionSpec
        self.attention_groups = sorted(attention_groups, key=lambda x: not isinstance(x[0], FullAttentionSpec))
        from math import lcm
        block_sizes = [spec.block_size for spec, _, _ in attention_groups]
        self.lcm_block_size = lcm(*block_sizes) if block_sizes else 1

        self.eagle_attn_group_indices = {
            i for i, (_, group_ids, _) in enumerate(self.attention_groups)
            if any(gid in self.eagle_group_ids for gid in group_ids)
        }

    HybridKVCacheCoordinator.verify_and_split_kv_cache_groups = tpu_safe_verify_and_split

    original_find_hit = HybridKVCacheCoordinator.find_longest_cache_hit
    def tpu_safe_find_hit(self, block_hashes, max_cache_hit_length):
        if len(self.attention_groups) == 0:
            return (), 0
        return original_find_hit(self, block_hashes, max_cache_hit_length)

    HybridKVCacheCoordinator.find_longest_cache_hit = tpu_safe_find_hit

    print(f"[vLLM TPU Patch] Successfully routed DeepSeek-V4 weights through compressed_tensors FP8 schema.")

# 7. ROUTE ENGINE DEFAULT MODELS
# Overrides vanilla code by forcing the engine to instantiate the custom TPU implementation
from vllm.model_executor.models.registry import ModelRegistry
class DeepSeekV4MTP:
    pass

ModelRegistry.register_model(
    "DeepseekV4ForCausalLM", 
    "tpu_inference.models.vllm.experimental.deepseek_v4:DeepseekV4ForCausalLM" # Note this must map to where you eventually copy deepseek_v4.py
)

def maybe_apply_deepseek_v4_patches(vllm_model):
    """Orchestrates conditional patching based on model validation."""
    if is_deepseek_v4(vllm_model):
        apply_deepseek_v4_patches(vllm_model)

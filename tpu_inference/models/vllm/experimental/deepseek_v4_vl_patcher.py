import torch

# 1. MONKEY PATCH DUMMY WEIGHT LOADER
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



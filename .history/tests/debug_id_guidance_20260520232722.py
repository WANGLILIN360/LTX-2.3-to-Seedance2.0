"""Quick debug script for identity guidance delta."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "ltx-core" / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "ltx-pipelines" / "src"))

# Stub missing deps
import types
for name in ["av", "av.codec", "av.container", "av.stream", "av.video", "av.audio",
             "av.filter", "av.data", "OpenImageIO", "OpenImageIO.ImageSpec", "OpenImageIO.ImageInput"]:
    sys.modules[name] = types.ModuleType(name)

import torch
from ltx_core.types import LatentState
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
from ltx_pipelines.utils.denoisers import _guided_denoise

# Build states with realistic denoise_mask
dm = torch.cat([torch.ones(1, 1, 12), torch.zeros(1, 1, 8)], dim=2)
vs = LatentState(
    latent=torch.randn(1, 128, 20),
    denoise_mask=dm,
    positions=torch.randn(1, 20, 3),
    clean_latent=torch.zeros(1, 128, 20),
    attention_mask=torch.ones(1, 1, 1, 20),
)
ns = LatentState(
    latent=torch.randn(1, 128, 12),
    denoise_mask=torch.ones(1, 1, 12),
    positions=torch.randn(1, 12, 3),
    clean_latent=torch.zeros(1, 128, 12),
    attention_mask=torch.ones(1, 1, 1, 12),
)
g = MultiModalGuider(params=MultiModalGuiderParams(cfg_scale=1.0, identity_guidance_scale=0.5))

call_idx = [0]

class MockTransformer:
    def __call__(self, video=None, audio=None, perturbations=None):
        call_idx[0] += 1
        if video is not None and video.enabled:
            B, T, D = video.latent.shape
            print(f"  Call {call_idx[0]}: video.latent shape = ({B}, {T}, {D})")
            return torch.full((B, T, D), float(call_idx[0])), None
        return None, None

# Monkey-patch _guided_denoise to print intermediate values
import ltx_pipelines.utils.denoisers as dn
orig_fn = dn._guided_denoise

def patched_fn(*args, **kwargs):
    # Just call original and add prints
    return orig_fn(*args, **kwargs)

# Instead, let's directly test the indexing logic
print("=== Testing indexing logic ===")
cond_v = torch.full((1, 128, 20), 1.0)  # from transformer output
noref_v = torch.full((1, 128, 12), 2.0)  # from noref transformer output
tm = dm.squeeze(0).squeeze(0).bool()
print(f"cond_v shape: {cond_v.shape}, noref_v shape: {noref_v.shape}")
print(f"target_mask sum: {tm.sum()}, len: {tm.shape[0]}")

id_cond_v = cond_v[..., tm]
print(f"id_cond_v shape: {id_cond_v.shape}")
id_delta_v = 0.5 * (id_cond_v - noref_v)
print(f"id_delta_v shape: {id_delta_v.shape}, unique: {id_delta_v.unique()}")

full_delta_v = torch.zeros_like(cond_v)
full_delta_v[..., tm] = id_delta_v
print(f"full_delta_v target unique: {full_delta_v[..., tm].unique()}")
print(f"full_delta_v ref unique: {full_delta_v[..., ~tm].unique()}")

result = cond_v + full_delta_v  # guider.calculate returns cond_v (cfg=1), then + delta
print(f"\nResult target unique: {result[..., tm].unique()}")
print(f"Result ref unique: {result[..., ~tm].unique()}")
print(f"Expected: target=0.5, ref=1.0")

rv, ra = _guided_denoise(
    MockTransformer(), vs, None, torch.tensor([1.0]), g, g,
    torch.randn(1, 20, 64), None,
    last_denoised_video=None, last_denoised_audio=None,
    step_index=0, noref_video_state=ns,
)

# In patchified space, target tokens are where timesteps > 0
# For our test, the batched_video has timesteps = denoise_mask * sigma
# target (denoise=1) → timesteps=1.0, ref (denoise=0) → timesteps=0.0
# After patchification, the sequence dim is dim=1 of the Modality.latent
print(f"\n=== Actual _guided_denoise result ===")
print(f"Result denoised shape: {rv.denoised.shape}")
# The denoised shape is (B, T, D) in patchified space
# We can't easily map back to original space without knowing patchification details
# Just verify the result is computed correctly by checking unique values
unique_vals = rv.denoised.unique()
print(f"Unique values in result: {unique_vals}")

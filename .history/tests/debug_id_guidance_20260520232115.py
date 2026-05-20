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

rv, ra = _guided_denoise(
    MockTransformer(), vs, None, torch.tensor([1.0]), g, g,
    torch.randn(1, 20, 64), None,
    last_denoised_video=None, last_denoised_audio=None,
    step_index=0, noref_video_state=ns,
)

tm = dm.squeeze(0).squeeze(0).bool()
print(f"\nResult denoised shape: {rv.denoised.shape}")
print(f"Target positions unique vals: {rv.denoised[..., tm].unique()}")
print(f"Ref positions unique vals: {rv.denoised[..., ~tm].unique()}")
print(f"\nExpected target: 0.5, Expected ref: 1.0")

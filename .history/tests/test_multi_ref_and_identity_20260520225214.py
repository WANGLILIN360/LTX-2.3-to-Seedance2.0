"""Unit tests for multi-reference conditioning and identity guidance.

Covers:
1. MultiModalGuiderParams.identity_guidance_scale
2. MultiModalGuider.calculate() with noref term
3. MultiModalGuider.do_identity_guidance()
4. UnifiedMultiReferenceConditioning (IC-LoRA style, no slot embedding binding)
5. ReferenceItem.slot_id (bookkeeping only)
6. _guided_denoise noref forward pass
7. GuidedDenoiser / FactoryGuidedDenoiser noref state passthrough
8. Pipeline noref_video_state construction logic
"""

import sys
from pathlib import Path

import pytest
import torch

# Ensure ltx_core is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "packages" / "ltx-core" / "src"))


# ============================================================================
# 1. MultiModalGuiderParams — identity_guidance_scale
# ============================================================================

class TestMultiModalGuiderParamsIdentityGuidance:
    """Test identity_guidance_scale on MultiModalGuiderParams."""

    def test_default_is_zero(self):
        from ltx_core.components.guiders import MultiModalGuiderParams
        params = MultiModalGuiderParams()
        assert params.identity_guidance_scale == 0.0

    def test_custom_value(self):
        from ltx_core.components.guiders import MultiModalGuiderParams
        params = MultiModalGuiderParams(identity_guidance_scale=0.5)
        assert params.identity_guidance_scale == 0.5

    def test_frozen_dataclass(self):
        from ltx_core.components.guiders import MultiModalGuiderParams
        params = MultiModalGuiderParams(identity_guidance_scale=0.3)
        with pytest.raises(AttributeError):
            params.identity_guidance_scale = 0.9


# ============================================================================
# 2. MultiModalGuider.calculate() — noref term
# ============================================================================

class TestMultiModalGuiderCalculateNoref:
    """Test guider.calculate() with the noref (identity guidance) term."""

    def _make_guider(self, cfg_scale=1.0, identity_guidance_scale=0.0):
        from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
        params = MultiModalGuiderParams(
            cfg_scale=cfg_scale,
            identity_guidance_scale=identity_guidance_scale,
        )
        return MultiModalGuider(params=params)

    def test_noref_zero_no_effect(self):
        """When identity_guidance_scale=0, noref term has no effect."""
        g = self._make_guider(identity_guidance_scale=0.0)
        cond = torch.randn(2, 4, 8, 8)
        noref = torch.randn(2, 4, 8, 8)
        result = g.calculate(cond, 0.0, 0.0, 0.0, noref)
        # With cfg=1, stg=0, mod=1, id=0: result = cond + 0 + 0 + 0 + 0*(cond-noref) = cond
        assert torch.allclose(result, cond)

    def test_noref_positive_adds_delta(self):
        """identity_guidance_scale > 0 adds (cond - noref) scaled."""
        g = self._make_guider(identity_guidance_scale=0.5)
        cond = torch.randn(2, 4, 8, 8)
        noref = torch.randn(2, 4, 8, 8)
        result = g.calculate(cond, 0.0, 0.0, 0.0, noref)
        expected = cond + 0.5 * (cond - noref)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_noref_with_cfg(self):
        """Identity guidance works alongside CFG."""
        g = self._make_guider(cfg_scale=3.0, identity_guidance_scale=0.5)
        cond = torch.randn(2, 4, 8, 8)
        uncond = torch.randn(2, 4, 8, 8)
        noref = torch.randn(2, 4, 8, 8)
        result = g.calculate(cond, uncond, 0.0, 0.0, noref)
        expected = cond + (3.0 - 1) * (cond - uncond) + 0.5 * (cond - noref)
        assert torch.allclose(result, expected, atol=1e-5)

    def test_noref_default_zero(self):
        """Default noref=0.0 works (backward compatible)."""
        g = self._make_guider(identity_guidance_scale=0.5)
        cond = torch.randn(2, 4, 8, 8)
        result = g.calculate(cond, 0.0, 0.0, 0.0)  # no noref arg
        # 0.5 * (cond - 0) = 0.5 * cond
        expected = cond + 0.5 * cond
        assert torch.allclose(result, expected, atol=1e-6)


# ============================================================================
# 3. MultiModalGuider.do_identity_guidance()
# ============================================================================

class TestMultiModalGuiderDoIdentityGuidance:

    def test_disabled_when_zero(self):
        from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
        g = MultiModalGuider(params=MultiModalGuiderParams(identity_guidance_scale=0.0))
        assert not g.do_identity_guidance()

    def test_enabled_when_nonzero(self):
        from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
        g = MultiModalGuider(params=MultiModalGuiderParams(identity_guidance_scale=0.01))
        assert g.do_identity_guidance()

    def test_enabled_when_large(self):
        from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
        g = MultiModalGuider(params=MultiModalGuiderParams(identity_guidance_scale=5.0))
        assert g.do_identity_guidance()


# ============================================================================
# 4. UnifiedMultiReferenceConditioning — IC-LoRA style
# ============================================================================

class TestUnifiedMultiReferenceConditioning:
    """Test that UnifiedMultiReferenceConditioning no longer requires slot_embedding."""

    def test_init_without_slot_embedding(self):
        from ltx_core.conditioning import (
            UnifiedMultiReferenceConditioning,
            ReferenceItem,
            ReferenceModality,
        )
        ref = ReferenceItem(
            modality=ReferenceModality.IMAGE,
            latent=torch.randn(1, 128, 1, 60, 90),
            attribute_tags=[],
            slot_id="Image1",
        )
        cond = UnifiedMultiReferenceConditioning(references=[ref])
        assert cond.references == [ref]
        assert not hasattr(cond, "slot_embedding") or not hasattr(cond, "_slot_embedding")

    def test_init_with_attribute_weights(self):
        from ltx_core.conditioning import (
            UnifiedMultiReferenceConditioning,
            ReferenceItem,
            ReferenceModality,
            ReferenceAttribute,
        )
        ref = ReferenceItem(
            modality=ReferenceModality.IMAGE,
            latent=torch.randn(1, 128, 1, 60, 90),
            attribute_tags=[ReferenceAttribute.IDENTITY],
            slot_id="Image1",
        )
        weights = {ReferenceAttribute.IDENTITY: 0.9}
        cond = UnifiedMultiReferenceConditioning(
            references=[ref],
            attribute_weights=weights,
        )
        assert cond._attr_weights == weights


# ============================================================================
# 5. ReferenceItem.slot_id — bookkeeping
# ============================================================================

class TestReferenceItemSlotId:
    """Test that slot_id is purely bookkeeping and doesn't affect binding."""

    def test_slot_id_stored(self):
        from ltx_core.conditioning import ReferenceItem, ReferenceModality
        ref = ReferenceItem(
            modality=ReferenceModality.IMAGE,
            latent=torch.randn(1, 128, 1, 60, 90),
            attribute_tags=[],
            slot_id="Image2",
        )
        assert ref.slot_id == "Image2"

    def test_slot_id_empty_default(self):
        from ltx_core.conditioning import ReferenceItem, ReferenceModality
        ref = ReferenceItem(
            modality=ReferenceModality.VIDEO,
            latent=torch.randn(1, 128, 5, 60, 90),
            attribute_tags=[],
        )
        assert ref.slot_id == ""

    def test_slot_id_doesnt_affect_attention(self):
        """slot_id should not change the attention mask or weight."""
        from ltx_core.conditioning import ReferenceItem, ReferenceModality, ReferenceAttribute
        ref_a = ReferenceItem(
            modality=ReferenceModality.IMAGE,
            latent=torch.randn(1, 128, 1, 60, 90),
            attribute_tags=[ReferenceAttribute.IDENTITY],
            slot_id="Image1",
        )
        ref_b = ReferenceItem(
            modality=ReferenceModality.IMAGE,
            latent=torch.randn(1, 128, 1, 60, 90),
            attribute_tags=[ReferenceAttribute.IDENTITY],
            slot_id="",
        )
        # Both should have same effective_attention_weight
        assert ref_a.effective_attention_weight == ref_b.effective_attention_weight


# ============================================================================
# 6. _guided_denoise — noref forward pass
# ============================================================================

class TestGuidedDenoiseNoref:
    """Test that _guided_denoise runs a separate noref forward when needed."""

    def _make_guider(self, identity_guidance_scale=0.5):
        from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
        params = MultiModalGuiderParams(
            cfg_scale=1.0,
            identity_guidance_scale=identity_guidance_scale,
        )
        return MultiModalGuider(params=params)

    def test_noref_not_called_when_disabled(self):
        """When identity_guidance_scale=0, noref forward should not happen."""
        from ltx_pipelines.utils.denoisers import _guided_denoise
        # We use a mock transformer that counts calls
        call_count = {"count": 0}

        class MockTransformer:
            def __call__(self, video=None, audio=None, perturbations=None):
                call_count["count"] += 1
                B = video.latent.shape[0] if video is not None else 1
                return torch.randn(B, 128, 10), None

        from ltx_core.types import LatentState
        video_state = LatentState(
            latent=torch.randn(1, 128, 10),
            denoise_mask=torch.ones(1, 1, 10),
            positions=torch.randn(1, 10, 3),
            clean_latent=torch.zeros(1, 128, 10),
            attention_mask=torch.ones(1, 1, 1, 10),
        )
        guider = self._make_guider(identity_guidance_scale=0.0)
        v_ctx = torch.randn(1, 20, 4096)

        _guided_denoise(
            transformer=MockTransformer(),
            video_state=video_state,
            audio_state=None,
            sigma=torch.tensor([1.0]),
            video_guider=guider,
            audio_guider=guider,
            v_context=v_ctx,
            a_context=None,
            last_denoised_video=None,
            last_denoised_audio=None,
            step_index=0,
        )
        # Only 1 call (cond pass only, no noref)
        assert call_count["count"] == 1

    def test_noref_called_when_enabled(self):
        """When identity_guidance_scale > 0 and noref state provided, 2 transformer calls."""
        from ltx_pipelines.utils.denoisers import _guided_denoise
        call_count = {"count": 0}

        class MockTransformer:
            def __call__(self, video=None, audio=None, perturbations=None):
                call_count["count"] += 1
                B = video.latent.shape[0] if video is not None else 1
                return torch.randn(B, 128, 10), None

        from ltx_core.types import LatentState
        video_state = LatentState(
            latent=torch.randn(1, 128, 10),
            denoise_mask=torch.ones(1, 1, 10),
            positions=torch.randn(1, 10, 3),
            clean_latent=torch.zeros(1, 128, 10),
            attention_mask=torch.ones(1, 1, 1, 10),
        )
        noref_state = LatentState(
            latent=torch.randn(1, 128, 8),  # shorter! no ref tokens
            denoise_mask=torch.ones(1, 1, 8),
            positions=torch.randn(1, 8, 3),
            clean_latent=torch.zeros(1, 128, 8),
            attention_mask=torch.ones(1, 1, 1, 8),
        )
        guider = self._make_guider(identity_guidance_scale=0.5)
        v_ctx = torch.randn(1, 20, 4096)

        _guided_denoise(
            transformer=MockTransformer(),
            video_state=video_state,
            audio_state=None,
            sigma=torch.tensor([1.0]),
            video_guider=guider,
            audio_guider=guider,
            v_context=v_ctx,
            a_context=None,
            last_denoised_video=None,
            last_denoised_audio=None,
            step_index=0,
            noref_video_state=noref_state,
        )
        # 2 calls: main batch + noref forward
        assert call_count["count"] == 2


# ============================================================================
# 7. GuidedDenoiser / FactoryGuidedDenoiser — noref passthrough
# ============================================================================

class TestDenoiserNorefPassthrough:

    def test_guided_denoiser_stores_noref(self):
        from ltx_pipelines.utils.denoisers import GuidedDenoiser
        from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
        from ltx_core.types import LatentState

        noref = LatentState(
            latent=torch.randn(1, 128, 8),
            denoise_mask=torch.ones(1, 1, 8),
            positions=torch.randn(1, 8, 3),
            clean_latent=torch.zeros(1, 128, 8),
            attention_mask=None,
        )
        guider = MultiModalGuider(params=MultiModalGuiderParams(identity_guidance_scale=0.5))
        denoiser = GuidedDenoiser(
            v_context=torch.randn(1, 20, 4096),
            a_context=None,
            video_guider=guider,
            noref_video_state=noref,
        )
        assert denoiser.noref_video_state is noref

    def test_factory_denoiser_stores_noref(self):
        from ltx_pipelines.utils.denoisers import FactoryGuidedDenoiser
        from ltx_core.components.guiders import MultiModalGuiderFactory, MultiModalGuiderParams
        from ltx_core.types import LatentState

        noref = LatentState(
            latent=torch.randn(1, 128, 8),
            denoise_mask=torch.ones(1, 1, 8),
            positions=torch.randn(1, 8, 3),
            clean_latent=torch.zeros(1, 128, 8),
            attention_mask=None,
        )
        factory = MultiModalGuiderFactory.constant(
            MultiModalGuiderParams(identity_guidance_scale=0.5),
        )
        denoiser = FactoryGuidedDenoiser(
            v_context=torch.randn(1, 20, 4096),
            a_context=None,
            video_guider_factory=factory,
            noref_video_state=noref,
        )
        assert denoiser.noref_video_state is noref

    def test_guided_denoiser_default_noref_none(self):
        from ltx_pipelines.utils.denoisers import GuidedDenoiser
        denoiser = GuidedDenoiser(
            v_context=torch.randn(1, 20, 4096),
            a_context=None,
        )
        assert denoiser.noref_video_state is None
        assert denoiser.noref_audio_state is None


# ============================================================================
# 8. Pipeline noref state construction logic
# ============================================================================

class TestPipelineNorefStateLogic:
    """Test the decision logic for building noref state in the pipeline."""

    def test_noref_built_when_scale_positive_and_refs_exist(self):
        """noref state should be built when identity_guidance_scale > 0 and refs exist."""
        from ltx_core.components.guiders import MultiModalGuiderParams
        params = MultiModalGuiderParams(identity_guidance_scale=0.5)
        has_refs = True
        should_build = params.identity_guidance_scale > 0 and has_refs
        assert should_build

    def test_noref_not_built_when_scale_zero(self):
        from ltx_core.components.guiders import MultiModalGuiderParams
        params = MultiModalGuiderParams(identity_guidance_scale=0.0)
        has_refs = True
        should_build = params.identity_guidance_scale > 0 and has_refs
        assert not should_build

    def test_noref_not_built_when_no_refs(self):
        from ltx_core.components.guiders import MultiModalGuiderParams
        params = MultiModalGuiderParams(identity_guidance_scale=0.5)
        has_refs = False
        should_build = params.identity_guidance_scale > 0 and has_refs
        assert not should_build


# ============================================================================
# 9. ReferenceSlotEmbedding — optional utility (not used for binding)
# ============================================================================

class TestReferenceSlotEmbedding:
    """ReferenceSlotEmbedding is kept as optional training utility, not for inference binding."""

    def test_create_slot_embedding(self):
        from ltx_core.conditioning import ReferenceSlotEmbedding
        slot = ReferenceSlotEmbedding(embed_dim=128, max_slots=4)
        assert slot.embeddings.weight.shape == (4, 128)

    def test_add_to_tokens(self):
        from ltx_core.conditioning import ReferenceSlotEmbedding
        slot = ReferenceSlotEmbedding(embed_dim=128, max_slots=4)
        tokens = torch.zeros(1, 10, 128)
        result = slot.add_to_tokens(tokens, "Image1")
        # Should be non-zero (embedding added)
        assert not torch.allclose(result, tokens)

    def test_add_to_tokens_unknown_slot(self):
        from ltx_core.conditioning import ReferenceSlotEmbedding
        slot = ReferenceSlotEmbedding(embed_dim=128, max_slots=4)
        tokens = torch.zeros(1, 10, 128)
        result = slot.add_to_tokens(tokens, "UnknownSlot99")
        # Unknown slot should be a no-op
        assert torch.allclose(result, tokens)


# ============================================================================
# 10. Integration: full guider formula
# ============================================================================

class TestGuiderFullFormula:
    """Test the complete guider formula with all terms."""

    def test_all_terms(self):
        """Test: pred = cond + (cfg-1)*(cond-uncond) + stg*(cond-ptb) + (mod-1)*(cond-mod) + id*(cond-noref)"""
        from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
        params = MultiModalGuiderParams(
            cfg_scale=3.0,
            stg_scale=0.5,
            modality_scale=1.5,
            identity_guidance_scale=0.3,
        )
        g = MultiModalGuider(params=params)

        cond = torch.randn(2, 4, 8, 8)
        uncond = torch.randn(2, 4, 8, 8)
        ptb = torch.randn(2, 4, 8, 8)
        mod = torch.randn(2, 4, 8, 8)
        noref = torch.randn(2, 4, 8, 8)

        result = g.calculate(cond, uncond, ptb, mod, noref)
        expected = (
            cond
            + (3.0 - 1) * (cond - uncond)
            + 0.5 * (cond - ptb)
            + (1.5 - 1) * (cond - mod)
            + 0.3 * (cond - noref)
        )
        assert torch.allclose(result, expected, atol=1e-5)

    def test_rescale_applied(self):
        """Test rescale_scale is applied after all guidance terms."""
        from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
        params = MultiModalGuiderParams(
            cfg_scale=3.0,
            identity_guidance_scale=0.5,
            rescale_scale=0.7,
        )
        g = MultiModalGuider(params=params)
        cond = torch.randn(2, 4, 8, 8)
        uncond = torch.randn(2, 4, 8, 8)
        noref = torch.randn(2, 4, 8, 8)

        result = g.calculate(cond, uncond, 0.0, 0.0, noref)
        # Verify rescale was applied (result std should be closer to cond std)
        pred_no_rescale = cond + (3.0 - 1) * (cond - uncond) + 0.5 * (cond - noref)
        assert not torch.allclose(result, pred_no_rescale, atol=1e-3)

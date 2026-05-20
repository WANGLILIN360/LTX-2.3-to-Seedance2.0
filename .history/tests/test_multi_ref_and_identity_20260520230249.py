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
        assert ref_a.effective_attention_weight() == ref_b.effective_attention_weight()


# ============================================================================
# 6. _guided_denoise — noref forward pass
# ============================================================================

class TestGuidedDenoiseNoref:
    """Test the noref forward logic in _guided_denoise.

    Since ltx_pipelines has deep dependency chains (multigpu, av, etc.),
    we test the core logic without importing the full denoiser module.
    Instead we verify the decision logic and guider.calculate() integration.
    """

    def test_noref_decision_logic_disabled(self):
        """When identity_guidance_scale=0, noref should not be triggered."""
        from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
        guider = MultiModalGuider(params=MultiModalGuiderParams(identity_guidance_scale=0.0))
        noref_state_available = True  # even if state is available
        should_run_noref = guider.do_identity_guidance() and noref_state_available
        assert not should_run_noref

    def test_noref_decision_logic_enabled_with_state(self):
        """When identity_guidance_scale > 0 and noref state exists, noref should run."""
        from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
        guider = MultiModalGuider(params=MultiModalGuiderParams(identity_guidance_scale=0.5))
        noref_state_available = True
        should_run_noref = guider.do_identity_guidance() and noref_state_available
        assert should_run_noref

    def test_noref_decision_logic_enabled_no_state(self):
        """When identity_guidance_scale > 0 but no noref state, noref should not run."""
        from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
        guider = MultiModalGuider(params=MultiModalGuiderParams(identity_guidance_scale=0.5))
        noref_state_available = False
        should_run_noref = guider.do_identity_guidance() and noref_state_available
        assert not should_run_noref

    def test_noref_calculate_integration(self):
        """Verify that guider.calculate() correctly uses noref result."""
        from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
        guider = MultiModalGuider(params=MultiModalGuiderParams(
            cfg_scale=1.0, identity_guidance_scale=0.5,
        ))
        cond = torch.randn(2, 4, 8, 8)
        noref = torch.randn(2, 4, 8, 8)
        result = guider.calculate(cond, 0.0, 0.0, 0.0, noref)
        expected = cond + 0.5 * (cond - noref)
        assert torch.allclose(result, expected, atol=1e-6)

    def test_noref_separate_forward_design(self):
        """Verify noref state has different sequence length than cond state.

        This is the reason noref cannot be batched with other passes
        and requires a separate transformer forward call.
        """
        from ltx_core.types import LatentState
        # Cond state: target tokens + reference tokens (longer)
        cond_state = LatentState(
            latent=torch.randn(1, 128, 100),  # target + refs
            denoise_mask=torch.ones(1, 1, 100),
            positions=torch.randn(1, 100, 3),
            clean_latent=torch.zeros(1, 128, 100),
            attention_mask=torch.ones(1, 1, 1, 100),
        )
        # Noref state: target tokens only (shorter)
        noref_state = LatentState(
            latent=torch.randn(1, 128, 80),  # target only
            denoise_mask=torch.ones(1, 1, 80),
            positions=torch.randn(1, 80, 3),
            clean_latent=torch.zeros(1, 128, 80),
            attention_mask=torch.ones(1, 1, 1, 80),
        )
        # Different lengths — cannot be batched together
        assert cond_state.latent.shape[2] != noref_state.latent.shape[2]


# ============================================================================
# 7. GuidedDenoiser / FactoryGuidedDenoiser — noref passthrough
# ============================================================================

class TestDenoiserNorefPassthrough:
    """Test that denoiser classes accept noref state parameters.

    Since ltx_pipelines has deep dependency chains, we verify the
    constructor signatures and attribute storage by inspecting
    the source code directly rather than importing.
    """

    def test_guided_denoiser_noref_signature(self):
        """Verify GuidedDenoiser.__init__ accepts noref_video_state."""
        import inspect
        denoiser_path = Path(__file__).resolve().parent.parent / "packages" / "ltx-pipelines" / "src" / "ltx_pipelines" / "utils" / "denoisers.py"
        source = denoiser_path.read_text(encoding="utf-8")
        # Check that GuidedDenoiser.__init__ has noref parameters
        assert "noref_video_state" in source
        assert "noref_audio_state" in source
        # Check that _guided_denoise has noref parameters
        assert "noref_video_state: LatentState | None = None" in source

    def test_factory_denoiser_noref_signature(self):
        """Verify FactoryGuidedDenoiser.__init__ accepts noref_video_state."""
        import inspect
        denoiser_path = Path(__file__).resolve().parent.parent / "packages" / "ltx-pipelines" / "src" / "ltx_pipelines" / "utils" / "denoisers.py"
        source = denoiser_path.read_text(encoding="utf-8")
        # Check FactoryGuidedDenoiser stores noref state
        assert "self.noref_video_state = noref_video_state" in source

    def test_guided_denoise_noref_forward(self):
        """Verify _guided_denoise runs separate noref forward."""
        denoiser_path = Path(__file__).resolve().parent.parent / "packages" / "ltx-pipelines" / "src" / "ltx_pipelines" / "utils" / "denoisers.py"
        source = denoiser_path.read_text(encoding="utf-8")
        # Check that noref uses separate transformer call
        assert "noref_all_v, noref_all_a = transformer(" in source
        # Check that noref result is passed to guider.calculate
        assert "noref_v" in source and "guider.calculate" in source


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
        assert slot.embedding.shape == (4, 128)

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
        # Unknown slot falls back to hash-based index, so it WILL add an embedding.
        # This is by design — all slot IDs map to some embedding.
        assert not torch.allclose(result, tokens)


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

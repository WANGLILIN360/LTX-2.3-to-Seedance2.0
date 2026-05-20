"""Semantic @mention prompt parser using Gemma 3.

Uses the Gemma 3 instruction-tuned model to semantically parse
@mention references in prompts, supporting both Chinese and English.
This replaces the simple keyword-matching approach with true
language understanding.

Example input (Chinese):
    "一个人 @Image1 这个人的长相和身份 走在 @Image2 的公园场景里。
     参考 @Video1 的运镜方式。 @Audio1 的背景节奏。"

Example input (English):
    "A woman @Image1 for identity and appearance walks in @Image2's park.
     Replicate @Video1's camera movement. @Audio1 for background rhythm."

Example output:
    ParsedPrompt(
        bindings=[
            ReferenceBinding(ref_id="Image1", ref_type=IMAGE, attributes=[IDENTITY, APPEARANCE]),
            ReferenceBinding(ref_id="Image2", ref_type=IMAGE, attributes=[SCENE]),
            ReferenceBinding(ref_id="Video1", ref_type=VIDEO, attributes=[MOTION, CAMERA]),
            ReferenceBinding(ref_id="Audio1", ref_type=AUDIO, attributes=[AUDIO_RHYTHM, AUDIO_MOOD]),
        ]
    )
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import torch

from ltx_core.conditioning.types.multi_reference_cond import ReferenceAttribute, ReferenceModality
from ltx_pipelines.utils.prompt_parser import (
    ParsedPrompt,
    ReferenceBinding,
    _MENTION_PATTERN,
    _TYPE_MAP,
)

logger = logging.getLogger(__name__)

# System prompt for Gemma 3 to extract @mention bindings
_EXTRACTION_SYSTEM_PROMPT = """You are a structured data extraction assistant. Your task is to parse @mention references from a prompt and extract their assigned semantic attributes.

Available reference types:
- image: A static image reference (@Image1, @Image2, ...)
- video: A video clip reference (@Video1, @Video2, ...)
- audio: An audio clip reference (@Audio1, @Audio2, ...)

Available attributes:
- identity: person's face, character, who they are
- appearance: visual look, clothing, facial features, how they look
- style: artistic style, aesthetic, visual style
- motion: movement, choreography, action, dance, how things move
- camera: camera movement, dolly, tracking, pan, zoom, crane
- scene: background, environment, setting, interior, exterior
- audio_rhythm: rhythm, beat, tempo, pacing, musical timing
- audio_mood: mood, atmosphere, music genre, emotional tone
- lip_sync: lip synchronization, lip dub, mouth movement

Rules:
1. Each @mention maps to exactly one reference type and one or more attributes
2. Attributes should be inferred from the surrounding context, not just explicit keywords
3. If no attributes are clearly specified, assign defaults based on the reference type:
   - image default: identity, appearance
   - video default: motion, camera
   - audio default: audio_rhythm, audio_mood
4. Output ONLY valid JSON, no other text

Output format:
{
  "bindings": [
    {"ref_id": "Image1", "ref_type": "image", "attributes": ["identity", "appearance"]},
    {"ref_id": "Video1", "ref_type": "video", "attributes": ["motion", "camera"]}
  ]
}"""

_EXTRACTION_USER_TEMPLATE = """Parse the @mention references in this prompt:

{prompt}"""


class SemanticPromptParser:
    """Semantic @mention parser using Gemma 3 for language understanding.

    Supports both Chinese and English prompts. Uses the instruction-tuned
    Gemma 3 model to semantically understand what each reference should
    contribute to the generation.

    Usage:
        parser = SemanticPromptParser(gemma_root="path/to/gemma-3")
        parsed = parser.parse("一个人 @Image1 这个人的长相 走在 @Image2 的公园里")
    """

    def __init__(
        self,
        gemma_root: str,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.bfloat16,
        max_new_tokens: int = 512,
    ):
        self.gemma_root = gemma_root
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = dtype
        self.max_new_tokens = max_new_tokens
        self._tokenizer = None
        self._model = None

    def _load_model(self):
        """Lazily load the Gemma 3 model and tokenizer."""
        if self._model is not None:
            return

        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading Gemma 3 model from {self.gemma_root} for semantic parsing...")

        self._tokenizer = AutoTokenizer.from_pretrained(self.gemma_root)
        self._model = AutoModelForCausalLM.from_pretrained(
            self.gemma_root,
            torch_dtype=self.dtype,
            device_map=self.device,
        )
        self._model.eval()

        logger.info("Gemma 3 model loaded for semantic parsing")

    def _unload_model(self):
        """Free GPU memory."""
        if self._model is not None:
            del self._model
            del self._tokenizer
            self._model = None
            self._tokenizer = None
            torch.cuda.empty_cache()
            logger.info("Unloaded Gemma 3 semantic parser")

    def parse(self, prompt: str) -> ParsedPrompt:
        """Parse a prompt with @mention references using Gemma 3.

        Args:
            prompt: The prompt text containing @mention references.

        Returns:
            ParsedPrompt with semantically extracted bindings.
        """
        # Quick check: if no @mentions, return empty
        if not _MENTION_PATTERN.search(prompt):
            return ParsedPrompt(text=prompt)

        self._load_model()

        # Build the extraction prompt
        messages = [
            {"role": "system", "content": _EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": _EXTRACTION_USER_TEMPLATE.format(prompt=prompt)},
        ]

        input_text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self._tokenizer(input_text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=1.0,
            )

        # Decode only the generated tokens
        generated_ids = outputs[0][inputs["input_ids"].shape[-1]:]
        response_text = self._tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Parse the JSON response
        bindings = self._parse_response(response_text, prompt)

        # Build reference lists
        ref_images = []
        ref_videos = []
        ref_audios = []
        for b in bindings:
            if b.ref_type == ReferenceModality.IMAGE and b.ref_id not in ref_images:
                ref_images.append(b.ref_id)
            elif b.ref_type == ReferenceModality.VIDEO and b.ref_id not in ref_videos:
                ref_videos.append(b.ref_id)
            elif b.ref_type == ReferenceModality.AUDIO and b.ref_id not in ref_audios:
                ref_audios.append(b.ref_id)

        return ParsedPrompt(
            text=prompt,
            bindings=bindings,
            ref_images=ref_images,
            ref_videos=ref_videos,
            ref_audios=ref_audios,
        )

    def _parse_response(
        self,
        response_text: str,
        original_prompt: str,
    ) -> list[ReferenceBinding]:
        """Parse the Gemma 3 JSON response into ReferenceBinding objects."""
        # Try to extract JSON from the response
        json_str = self._extract_json(response_text)

        if json_str is None:
            logger.warning(
                f"Failed to extract JSON from Gemma 3 response, "
                f"falling back to rule-based parsing. Response: {response_text[:200]}"
            )
            from ltx_pipelines.utils.prompt_parser import parse_prompt as rule_parse
            result = rule_parse(original_prompt)
            return result.bindings

        try:
            data = json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.warning(f"JSON decode error: {e}, falling back to rule-based parsing")
            from ltx_pipelines.utils.prompt_parser import parse_prompt as rule_parse
            result = rule_parse(original_prompt)
            return result.bindings

        bindings: list[ReferenceBinding] = []
        for item in data.get("bindings", []):
            ref_id = item.get("ref_id", "")
            ref_type_str = item.get("ref_type", "image").lower()
            attr_strs = item.get("attributes", [])

            # Parse reference type
            ref_type = _TYPE_MAP.get(ref_type_str, ReferenceModality.IMAGE)

            # Parse attributes
            attributes: list[ReferenceAttribute] = []
            for a in attr_strs:
                try:
                    attributes.append(ReferenceAttribute(a.lower()))
                except ValueError:
                    logger.warning(f"Unknown attribute '{a}', skipping")

            # Default attributes if none found
            if not attributes:
                if ref_type == ReferenceModality.IMAGE:
                    attributes = [ReferenceAttribute.IDENTITY, ReferenceAttribute.APPEARANCE]
                elif ref_type == ReferenceModality.VIDEO:
                    attributes = [ReferenceAttribute.MOTION, ReferenceAttribute.CAMERA]
                elif ref_type == ReferenceModality.AUDIO:
                    attributes = [ReferenceAttribute.AUDIO_RHYTHM, ReferenceAttribute.AUDIO_MOOD]

            # Extract index from ref_id
            index_match = re.search(r"(\d+)$", ref_id)
            index = int(index_match.group(1)) if index_match else 1

            bindings.append(ReferenceBinding(
                ref_id=ref_id,
                ref_type=ref_type,
                attributes=attributes,
                index=index,
            ))

        return bindings

    @staticmethod
    def _extract_json(text: str) -> str | None:
        """Extract JSON object from text that may contain other content."""
        # Try to find JSON between curly braces
        brace_count = 0
        start = None
        for i, c in enumerate(text):
            if c == '{':
                if brace_count == 0:
                    start = i
                brace_count += 1
            elif c == '}':
                brace_count -= 1
                if brace_count == 0 and start is not None:
                    return text[start:i + 1]

        # Try the whole text as JSON
        text_stripped = text.strip()
        if text_stripped.startswith('{') and text_stripped.endswith('}'):
            return text_stripped

        return None


def parse_prompt_semantic(
    prompt: str,
    gemma_root: str | None = None,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
) -> ParsedPrompt:
    """Convenience function for semantic prompt parsing.

    If gemma_root is not provided, falls back to rule-based parsing.
    """
    if gemma_root is None:
        from ltx_pipelines.utils.prompt_parser import parse_prompt as rule_parse
        return rule_parse(prompt)

    parser = SemanticPromptParser(
        gemma_root=gemma_root,
        device=device,
        dtype=dtype,
    )
    try:
        return parser.parse(prompt)
    finally:
        parser._unload_model()

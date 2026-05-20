"""@mention prompt parser for multi-reference binding.

Parses prompts containing @Image1, @Video1, @Audio1 style references
and extracts structured reference bindings, similar to Seedance 2.0's
@ mention system.

Example input:
    "Man @Image1 for appearance in @Image2's elevator setting.
     Replicate @Video1's camera movements. Use @Audio1 for background music."

Example output:
    ParsedPrompt(
        text="Man @Image1 for appearance in @Image2's elevator setting. ...",
        bindings=[
            ReferenceBinding(ref_id="Image1", ref_type="image", attributes=["appearance"]),
            ReferenceBinding(ref_id="Image2", ref_type="image", attributes=["scene"]),
            ReferenceBinding(ref_id="Video1", ref_type="video", attributes=["camera", "motion"]),
            ReferenceBinding(ref_id="Audio1", ref_type="audio", attributes=["audio_rhythm", "audio_mood"]),
        ]
    )
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from ltx_core.conditioning.types.multi_reference_cond import ReferenceAttribute, ReferenceModality


# Pattern to match @mentions like @Image1, @Video2, @Audio3
_MENTION_PATTERN = re.compile(r"@(Image|Video|Audio)(\d+)", re.IGNORECASE)

# Attribute keyword mappings: natural language -> ReferenceAttribute
_ATTRIBUTE_KEYWORDS: dict[str, ReferenceAttribute] = {
    # Identity / appearance
    "identity": ReferenceAttribute.IDENTITY,
    "face": ReferenceAttribute.IDENTITY,
    "character": ReferenceAttribute.IDENTITY,
    "appearance": ReferenceAttribute.APPEARANCE,
    "look": ReferenceAttribute.APPEARANCE,
    "facial": ReferenceAttribute.APPEARANCE,
    # Style
    "style": ReferenceAttribute.STYLE,
    "aesthetic": ReferenceAttribute.STYLE,
    "visual style": ReferenceAttribute.STYLE,
    # Motion
    "motion": ReferenceAttribute.MOTION,
    "movement": ReferenceAttribute.MOTION,
    "choreography": ReferenceAttribute.MOTION,
    "dance": ReferenceAttribute.MOTION,
    "action": ReferenceAttribute.MOTION,
    # Camera
    "camera": ReferenceAttribute.CAMERA,
    "dolly": ReferenceAttribute.CAMERA,
    "tracking": ReferenceAttribute.CAMERA,
    "crane": ReferenceAttribute.CAMERA,
    "pan": ReferenceAttribute.CAMERA,
    "orbit": ReferenceAttribute.CAMERA,
    "zoom": ReferenceAttribute.CAMERA,
    "hitchcock": ReferenceAttribute.CAMERA,
    # Scene
    "scene": ReferenceAttribute.SCENE,
    "background": ReferenceAttribute.SCENE,
    "environment": ReferenceAttribute.SCENE,
    "setting": ReferenceAttribute.SCENE,
    "interior": ReferenceAttribute.SCENE,
    "exterior": ReferenceAttribute.SCENE,
    # Audio
    "rhythm": ReferenceAttribute.AUDIO_RHYTHM,
    "beat": ReferenceAttribute.AUDIO_RHYTHM,
    "tempo": ReferenceAttribute.AUDIO_RHYTHM,
    "pacing": ReferenceAttribute.AUDIO_RHYTHM,
    "mood": ReferenceAttribute.AUDIO_MOOD,
    "atmosphere": ReferenceAttribute.AUDIO_MOOD,
    "music": ReferenceAttribute.AUDIO_MOOD,
    "background music": ReferenceAttribute.AUDIO_MOOD,
    # Lip sync
    "lip sync": ReferenceAttribute.LIP_SYNC,
    "lip-sync": ReferenceAttribute.LIP_SYNC,
    "lipsync": ReferenceAttribute.LIP_SYNC,
    "lip dub": ReferenceAttribute.LIP_SYNC,
}

# Type mapping from mention prefix to modality
_TYPE_MAP: dict[str, ReferenceModality] = {
    "image": ReferenceModality.IMAGE,
    "video": ReferenceModality.VIDEO,
    "audio": ReferenceModality.AUDIO,
}


@dataclass(frozen=True)
class ReferenceBinding:
    """A parsed reference binding from a prompt.

    Attributes:
        ref_id: The identifier (e.g., "Image1", "Video2").
        ref_type: The modality type (image/video/audio).
        attributes: Extracted attribute tags from the surrounding context.
        index: The numeric index from the mention (e.g., 1 from @Image1).
    """

    ref_id: str
    ref_type: ReferenceModality
    attributes: list[ReferenceAttribute]
    index: int


@dataclass
class ParsedPrompt:
    """Result of parsing a prompt with @mention references.

    Attributes:
        text: The original prompt text (unchanged).
        bindings: List of parsed reference bindings.
        ref_images: Ordered list of image reference IDs.
        ref_videos: Ordered list of video reference IDs.
        ref_audios: Ordered list of audio reference IDs.
    """

    text: str
    bindings: list[ReferenceBinding] = field(default_factory=list)
    ref_images: list[str] = field(default_factory=list)
    ref_videos: list[str] = field(default_factory=list)
    ref_audios: list[str] = field(default_factory=list)


def _extract_attributes_for_mention(
    prompt: str,
    mention_match: re.Match[str],
) -> list[ReferenceAttribute]:
    """Extract attribute tags from the context surrounding a @mention.

    Looks at the text between this mention and the next one (or end of prompt)
    for attribute keywords.
    """
    mention_end = mention_match.end()
    # Find the next mention or end of string
    next_mention = _MENTION_PATTERN.search(prompt, mention_end)
    context_end = next_mention.start() if next_mention else len(prompt)

    # Also look at a short window before the mention for "for X" patterns
    # Keep lookback short (15 chars) to avoid picking up attributes
    # from previous mentions' context
    mention_start = mention_match.start()
    lookback_start = max(0, mention_start - 15)
    before_text = prompt[lookback_start:mention_start].lower()
    after_text = prompt[mention_end:context_end].lower()

    # Combined context
    context = before_text + " " + after_text

    attributes: list[ReferenceAttribute] = []
    seen: set[ReferenceAttribute] = set()

    # Check each attribute keyword in the context using word boundaries
    for keyword, attr in _ATTRIBUTE_KEYWORDS.items():
        pattern = rf'\b{re.escape(keyword)}\b'
        if re.search(pattern, context) and attr not in seen:
            attributes.append(attr)
            seen.add(attr)

    # Default: if no attributes found, assign based on modality
    if not attributes:
        modality_str = mention_match.group(1).lower()
        if modality_str == "image":
            attributes = [ReferenceAttribute.IDENTITY, ReferenceAttribute.APPEARANCE]
        elif modality_str == "video":
            attributes = [ReferenceAttribute.MOTION, ReferenceAttribute.CAMERA]
        elif modality_str == "audio":
            attributes = [ReferenceAttribute.AUDIO_RHYTHM, ReferenceAttribute.AUDIO_MOOD]

    return attributes


def parse_prompt(prompt: str) -> ParsedPrompt:
    """Parse a prompt with @mention references.

    Extracts structured reference bindings from a prompt that uses
    the @mention syntax (e.g., @Image1, @Video2, @Audio1).

    Args:
        prompt: The prompt text containing @mention references.

    Returns:
        ParsedPrompt with extracted bindings and reference lists.
    """
    bindings: list[ReferenceBinding] = []
    ref_images: list[str] = []
    ref_videos: list[str] = []
    ref_audios: list[str] = []

    for match in _MENTION_PATTERN.finditer(prompt):
        type_str = match.group(1).lower()
        index = int(match.group(2))
        ref_id = f"{type_str.capitalize()}{index}"

        ref_type = _TYPE_MAP.get(type_str, ReferenceModality.IMAGE)
        attributes = _extract_attributes_for_mention(prompt, match)

        binding = ReferenceBinding(
            ref_id=ref_id,
            ref_type=ref_type,
            attributes=attributes,
            index=index,
        )
        bindings.append(binding)

        if ref_type == ReferenceModality.IMAGE:
            if ref_id not in ref_images:
                ref_images.append(ref_id)
        elif ref_type == ReferenceModality.VIDEO:
            if ref_id not in ref_videos:
                ref_videos.append(ref_id)
        elif ref_type == ReferenceModality.AUDIO:
            if ref_id not in ref_audios:
                ref_audios.append(ref_id)

    return ParsedPrompt(
        text=prompt,
        bindings=bindings,
        ref_images=ref_images,
        ref_videos=ref_videos,
        ref_audios=ref_audios,
    )

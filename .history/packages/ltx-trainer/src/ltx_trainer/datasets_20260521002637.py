from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
from einops import rearrange
from torch import Tensor
from torch.utils.data import Dataset

from ltx_trainer import logger

# Constants for precomputed data directories
PRECOMPUTED_DIR_NAME = ".precomputed"


class DummyDataset(Dataset):
    """Produce random latents and prompt embeddings. For minimal demonstration and benchmarking purposes"""

    def __init__(
        self,
        width: int = 1024,
        height: int = 1024,
        num_frames: int = 25,
        fps: int = 24,
        dataset_length: int = 200,
        latent_dim: int = 128,
        latent_spatial_compression_ratio: int = 32,
        latent_temporal_compression_ratio: int = 8,
        prompt_embed_dim: int = 4096,
        prompt_sequence_length: int = 256,
    ) -> None:
        if width % 32 != 0:
            raise ValueError(f"Width must be divisible by 32, got {width=}")

        if height % 32 != 0:
            raise ValueError(f"Height must be divisible by 32, got {height=}")

        if num_frames % 8 != 1:
            raise ValueError(f"Number of frames must have a remainder of 1 when divided by 8, got {num_frames=}")

        self.width = width
        self.height = height
        self.num_frames = num_frames
        self.fps = fps
        self.dataset_length = dataset_length
        self.latent_dim = latent_dim
        self.num_latent_frames = (num_frames - 1) // latent_temporal_compression_ratio + 1
        self.latent_height = height // latent_spatial_compression_ratio
        self.latent_width = width // latent_spatial_compression_ratio
        self.latent_sequence_length = self.num_latent_frames * self.latent_height * self.latent_width
        self.prompt_embed_dim = prompt_embed_dim
        self.prompt_sequence_length = prompt_sequence_length

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int) -> dict[str, dict[str, Tensor]]:
        return {
            "latent_conditions": {
                "latents": torch.randn(
                    self.latent_dim,
                    self.num_latent_frames,
                    self.latent_height,
                    self.latent_width,
                ),
                "num_frames": self.num_latent_frames,
                "height": self.latent_height,
                "width": self.latent_width,
                "fps": self.fps,
            },
            "text_conditions": {
                "video_prompt_embeds": torch.randn(
                    self.prompt_sequence_length,
                    self.prompt_embed_dim,
                ),
                "audio_prompt_embeds": torch.randn(
                    self.prompt_sequence_length,
                    self.prompt_embed_dim,
                ),
                "prompt_attention_mask": torch.ones(
                    self.prompt_sequence_length,
                    dtype=torch.bool,
                ),
            },
        }


class PrecomputedDataset(Dataset):
    def __init__(self, data_root: str, data_sources: dict[str, str] | list[str] | None = None) -> None:
        """
        Generic dataset for loading precomputed data from multiple sources.
        Args:
            data_root: Root directory containing preprocessed data
            data_sources: Either:
              - Dict mapping directory names to output keys
              - List of directory names (keys will equal values)
              - None (defaults to ["latents", "conditions"])
        Example:
            # Standard mode (list)
            dataset = PrecomputedDataset("data/", ["latents", "conditions"])
            # Standard mode (dict)
            dataset = PrecomputedDataset("data/", {"latents": "latent_conditions", "conditions": "text_conditions"})
            # IC-LoRA mode
            dataset = PrecomputedDataset("data/", ["latents", "conditions", "reference_latents"])
            # Multi-reference mode (auto-detects ref_image_latents/0/, ref_image_latents/1/, etc.)
            dataset = PrecomputedDataset("data/", ["latents", "conditions", "ref_image_latents"])
        Note:
            Latents are always returned in non-patchified format [C, F, H, W].
            Legacy patchified format [seq_len, C] is automatically converted.
            Multi-slot directories: If a source directory contains only numeric subdirectories
            (e.g. ref_image_latents/0/, ref_image_latents/1/), files from all slots are
            loaded and merged into a single dict with "latents" as a list of tensors.
        """
        super().__init__()

        self.data_root = self._setup_data_root(data_root)
        self.data_sources = self._normalize_data_sources(data_sources)
        self.source_paths, self._multi_slot_sources = self._setup_source_paths()
        self.sample_files = self._discover_samples()
        self._validate_setup()

    @staticmethod
    def _setup_data_root(data_root: str) -> Path:
        """Setup and validate the data root directory."""
        data_root = Path(data_root).expanduser().resolve()

        if not data_root.exists():
            raise FileNotFoundError(f"Data root directory does not exist: {data_root}")

        # If the given path is the dataset root, use the precomputed subdirectory
        if (data_root / PRECOMPUTED_DIR_NAME).exists():
            data_root = data_root / PRECOMPUTED_DIR_NAME

        return data_root

    @staticmethod
    def _normalize_data_sources(data_sources: dict[str, str] | list[str] | None) -> dict[str, str]:
        """Normalize data_sources input to a consistent dict format."""
        if data_sources is None:
            # Default sources
            return {"latents": "latent_conditions", "conditions": "text_conditions"}
        elif isinstance(data_sources, list):
            # Convert list to dict where keys equal values
            return {source: source for source in data_sources}
        elif isinstance(data_sources, dict):
            return data_sources.copy()
        else:
            raise TypeError(f"data_sources must be dict, list, or None, got {type(data_sources)}")

    def _setup_source_paths(self) -> tuple[dict[str, Path], dict[str, list[Path]]]:
        """Map data source names to their actual directory paths.

        Detects multi-slot directories: if a source directory contains only numeric
        subdirectories (0/, 1/, ...), it is treated as a multi-slot source. The primary
        path points to the parent directory, and slot paths are stored separately.

        Returns:
            Tuple of (source_paths, multi_slot_sources) where:
            - source_paths: maps dir_name -> primary directory path
            - multi_slot_sources: maps dir_name -> list of slot subdirectory paths
        """
        source_paths = {}
        multi_slot_sources: dict[str, list[Path]] = {}

        for dir_name in self.data_sources:
            source_path = self.data_root / dir_name

            # Check that all sources exist.
            if not source_path.exists():
                raise FileNotFoundError(f"Required {dir_name} directory does not exist: {source_path}")

            # Detect multi-slot pattern: directory contains only numeric subdirectories
            immediate_subdirs = [p for p in source_path.iterdir() if p.is_dir()]
            if immediate_subdirs and all(p.name.isdigit() for p in immediate_subdirs):
                # Multi-slot source: sort by numeric index
                slot_paths = sorted(immediate_subdirs, key=lambda p: int(p.name))
                multi_slot_sources[dir_name] = slot_paths
                # Primary path remains the parent directory
                source_paths[dir_name] = source_path
                logger.debug(f"Detected multi-slot source '{dir_name}' with {len(slot_paths)} slots")
            else:
                # Standard single-slot source
                source_paths[dir_name] = source_path

        return source_paths, multi_slot_sources

    def _discover_samples(self) -> dict[str, list[Path]]:
        """Discover all valid sample files across all data sources.
        Uses a fast two-pass approach: first globs all sources in parallel to build
        full-path sets in memory, then checks expected paths via set membership.
        This avoids O(N * num_sources) stat calls on networked filesystems while
        correctly handling path remapping (e.g. latent_X.pt -> condition_X.pt).
        For multi-slot sources, each slot is globbed independently and must all
        contain matching files for a sample to be considered valid.
        """
        if not self.data_sources:
            raise ValueError("No data sources configured")

        data_key = "latents" if "latents" in self.data_sources else next(iter(self.data_sources.keys()))
        data_path = self.source_paths[data_key]

        # Build the list of all glob targets (source + multi-slot sub-sources)
        glob_targets: list[str] = []
        for dir_name in self.data_sources:
            if dir_name in self._multi_slot_sources:
                # Add each slot as a separate glob target
                for slot_path in self._multi_slot_sources[dir_name]:
                    glob_targets.append(f"{dir_name}/{slot_path.name}")
            else:
                glob_targets.append(dir_name)

        # Pass 1: Glob all targets in parallel, build full-path sets
        def _glob_target(target_name: str) -> tuple[list[Path], set[str]]:
            source_path = self.data_root / target_name
            paths = list(source_path.glob("**/*.pt"))
            path_set = {str(p) for p in paths}
            return paths, path_set

        with ThreadPoolExecutor(max_workers=max(len(glob_targets), 1)) as executor:
            glob_results = dict(
                zip(
                    glob_targets,
                    executor.map(_glob_target, glob_targets),
                    strict=True,
                )
            )

        # Get primary source files (cached from glob, no second scan)
        data_files, _ = glob_results[data_key]
        if not data_files:
            raise ValueError(f"No data files found in {data_path}")
        data_files.sort()

        # Log source sizes
        for target_name, (paths, _) in glob_results.items():
            logger.debug(f"Source {target_name}: {len(paths)} files")

        # Build path sets for non-primary targets
        other_path_sets = {
            target_name: path_set
            for target_name, (_, path_set) in glob_results.items()
            if target_name != data_key
        }

        # Pass 2: For each primary file, check if expected paths exist in all other targets
        sample_files: dict[str, list[Path]] = {output_key: [] for output_key in self.data_sources.values()}
        # For multi-slot sources, store per-slot file lists
        multi_slot_sample_files: dict[str, dict[int, list[Path]]] = {}
        for dir_name in self._multi_slot_sources:
            multi_slot_sample_files[dir_name] = {
                int(slot.name): [] for slot in self._multi_slot_sources[dir_name]
            }

        valid_count = 0

        for data_file in data_files:
            rel_path = data_file.relative_to(data_path)

            # Check all other targets via set lookup (O(1) per target, no stat calls)
            all_exist = True
            for target_name, path_set in other_path_sets.items():
                expected = self._get_expected_file_path_for_target(target_name, data_file, rel_path)
                if str(expected) not in path_set:
                    logger.debug(f"Skipping {data_file.name}: no matching file at {expected}")
                    all_exist = False
                    break

            if all_exist:
                self._fill_sample_data_files_v2(
                    data_file, rel_path, sample_files, multi_slot_sample_files, other_path_sets,
                )
                valid_count += 1

        # Store multi-slot file lists for __getitem__
        self._multi_slot_sample_files = multi_slot_sample_files

        skipped = len(data_files) - valid_count
        if skipped > 0:
            logger.info(f"Fast index: {valid_count} valid samples from {len(data_files)} total ({skipped} skipped)")
        else:
            logger.debug(f"Fast index: {valid_count} valid samples from {len(data_files)} total")

        return sample_files

    def _get_expected_file_path(self, dir_name: str, data_file: Path, rel_path: Path) -> Path:
        """Get the expected file path for a given data source."""
        source_path = self.source_paths[dir_name]

        # For conditions, handle legacy naming where latent_X.pt maps to condition_X.pt
        if dir_name == "conditions" and data_file.name.startswith("latent_"):
            return source_path / f"condition_{data_file.stem[7:]}.pt"

        return source_path / rel_path

    def _get_expected_file_path_for_target(self, target_name: str, data_file: Path, rel_path: Path) -> Path:
        """Get the expected file path for a glob target (source or slot sub-source).

        target_name is either a plain dir_name (e.g. 'conditions') or a
        dir_name/slot_index path (e.g. 'ref_image_latents/0').
        """
        target_path = self.data_root / target_name

        # For conditions, handle legacy naming
        if target_name == "conditions" and data_file.name.startswith("latent_"):
            return target_path / f"condition_{data_file.stem[7:]}.pt"

        return target_path / rel_path

    def _fill_sample_data_files_v2(
        self,
        data_file: Path,
        rel_path: Path,
        sample_files: dict[str, list[Path]],
        multi_slot_sample_files: dict[str, dict[int, list[Path]]],
        other_path_sets: dict[str, set[str]],
    ) -> None:
        """Add a valid sample to the sample_files tracking (v2 with multi-slot support)."""
        for dir_name, output_key in self.data_sources.items():
            if dir_name in self._multi_slot_sources:
                # Multi-slot source: record per-slot relative paths
                for slot_path in self._multi_slot_sources[dir_name]:
                    slot_idx = int(slot_path.name)
                    target_name = f"{dir_name}/{slot_path.name}"
                    expected = self._get_expected_file_path_for_target(target_name, data_file, rel_path)
                    slot_rel = expected.relative_to(slot_path)
                    multi_slot_sample_files[dir_name][slot_idx].append(slot_rel)
                # For the main sample_files, add a placeholder (the actual data
                # will be assembled from slots in __getitem__)
                sample_files[output_key].append(Path("."))  # placeholder
            else:
                expected_path = self._get_expected_file_path(dir_name, data_file, rel_path)
                sample_files[output_key].append(expected_path.relative_to(self.source_paths[dir_name]))

    def _validate_setup(self) -> None:
        """Validate that the dataset setup is correct."""
        sample_counts = {key: len(files) for key, files in self.sample_files.items()}
        if not sample_counts or all(count == 0 for count in sample_counts.values()):
            raise ValueError(
                f"No valid samples found in {self.data_root} - all configured data sources "
                f"({list(self.data_sources)}) must have matching files (per-source counts: {sample_counts})"
            )

        # Verify all output keys have the same number of samples
        if len(set(sample_counts.values())) > 1:
            raise ValueError(f"Mismatched sample counts across sources: {sample_counts}")

    def __len__(self) -> int:
        # Use the first output key as reference count
        first_key = next(iter(self.sample_files.keys()))
        return len(self.sample_files[first_key])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        result = {}

        for dir_name, output_key in self.data_sources.items():
            if dir_name in self._multi_slot_sources:
                # Multi-slot source: load from all slots and merge
                slot_data_list = []
                for slot_path in self._multi_slot_sources[dir_name]:
                    slot_idx = int(slot_path.name)
                    file_rel_path = self._multi_slot_sample_files[dir_name][slot_idx][index]
                    file_path = slot_path / file_rel_path

                    try:
                        data = torch.load(file_path, map_location="cpu", weights_only=True)
                        if "latent" in dir_name.lower():
                            data = self._normalize_video_latents(data)
                        slot_data_list.append(data)
                    except Exception as e:
                        raise RuntimeError(
                            f"Failed to load {output_key} slot {slot_idx} from {file_path}: {e}"
                        ) from e

                # Merge slots into a single dict with "latents" as a list
                if slot_data_list:
                    merged = self._merge_multi_slot_data(slot_data_list)
                    result[output_key] = merged
            else:
                source_path = self.source_paths[dir_name]
                file_rel_path = self.sample_files[output_key][index]
                file_path = source_path / file_rel_path

                try:
                    data = torch.load(file_path, map_location="cpu", weights_only=True)

                    # Normalize video latent format if this is a latent source
                    if "latent" in dir_name.lower():
                        data = self._normalize_video_latents(data)

                    result[output_key] = data
                except Exception as e:
                    raise RuntimeError(f"Failed to load {output_key} from {file_path}: {e}") from e

        # Add index for debugging
        result["idx"] = index
        return result

    @staticmethod
    def _merge_multi_slot_data(slot_data_list: list[dict]) -> dict:
        """Merge data from multiple slots into a single dict.

        The "latents" key becomes a list of tensors (one per slot).
        Metadata keys (num_frames, height, width, fps) are also stored as
        lists so that each slot's dimensions are preserved independently.
        """
        if not slot_data_list:
            return {}

        merged = slot_data_list[0].copy()
        merged["latents"] = [slot["latents"] for slot in slot_data_list]

        # Preserve per-slot metadata as lists
        for key in ("num_frames", "height", "width", "fps"):
            if key in slot_data_list[0]:
                merged[key] = [slot[key] for slot in slot_data_list]

        return merged

    @staticmethod
    def _normalize_video_latents(data: dict) -> dict:
        """
        Normalize video latents to non-patchified format [C, F, H, W].
        Used for keeping backward compatibility with legacy datasets.
        """
        latents = data["latents"]

        # Check if latents are in legacy patchified format [seq_len, C]
        if latents.dim() == 2:
            # Legacy format: [seq_len, C] where seq_len = F * H * W
            num_frames = data["num_frames"]
            height = data["height"]
            width = data["width"]

            # Unpatchify: [seq_len, C] -> [C, F, H, W]
            latents = rearrange(
                latents,
                "(f h w) c -> c f h w",
                f=num_frames,
                h=height,
                w=width,
            )

            # Update the data dict with unpatchified latents
            data = data.copy()
            data["latents"] = latents

        return data

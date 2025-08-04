import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from .base_dataset import (
    AudioParams,
    BaseOrientationDataset,
    DatasetSample,
    TrainingSample,
)


@dataclass
class ZeroShotOrientationDataset(BaseOrientationDataset):
    dataset_samples: list[DatasetSample]
    audio_params: AudioParams

    @classmethod
    def from_config(
        cls,
        dataset_path: Path,
        audio_params: AudioParams,
        max_samples: Optional[int] = None,
    ) -> "ZeroShotOrientationDataset":
        dataset_samples = list(cls._find_samples(dataset_path, max_samples, audio_params.format))
        random.shuffle(dataset_samples)
        return cls(
            dataset_samples=dataset_samples,
            audio_params=audio_params,
        )

    @staticmethod
    def _find_samples(dataset_path: Path, max_samples: Optional[int], audio_format: str) -> Iterator[DatasetSample]:
        wav_parent_paths = {audio_path.parent for audio_path in dataset_path.rglob(f"*.{audio_format}")}
        for i, audio_parent_path in enumerate(tqdm(wav_parent_paths, desc="Finding samples")):
            if sample := ZeroShotOrientationDataset._get_sample(audio_parent_path, audio_format):
                yield sample
            if max_samples is not None and i >= max_samples:
                break

    def __getitem__(self, idx: int) -> TrainingSample | None:
        try:
            assert (
                len(self.dataset_samples[idx].audio_path) == 2
            ), "Audio path must be a list of exactly two Path objects for SingleShotDataset"
            assert (
                len(self.dataset_samples[idx].metadata) == 2
            ), "Metadata must be a list of exactly two dicts for SingleShotDataset"
            audio_paths = self.dataset_samples[idx].audio_path
            metadata = self.dataset_samples[idx].metadata

            wavs = [self._load_audio(audio_path) for audio_path in audio_paths]
            min_length = min(wav.shape[0] for wav in wavs)
            wavs = [wav[:min_length, :] for wav in wavs]

            label = self._get_label(self._get_orientation(metadata[0]), self._get_orientation(metadata[1]))
            return TrainingSample(
                wav=wavs,
                label=label,
                mic_position=self._get_mic_position(metadata[0]),
                src_position=self._get_src_position(metadata[0]),
            )
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return None

    @staticmethod
    def _get_sample(folder_path: Path, audio_format: str) -> DatasetSample | None:
        audio_paths = list(folder_path.glob(f"*.{audio_format}"))
        metadata_paths = [folder_path / f"{audio_path.stem}_metadata.json" for audio_path in audio_paths]
        metadatas = [ZeroShotOrientationDataset._load_json(path) for path in metadata_paths]

        if not (all(path.exists() for path in metadata_paths) and len(audio_paths) == 2):
            return

        for metadata in metadatas:
            if metadata["facing_mic"] == 1:
                audio_path_1 = audio_paths[metadatas.index(metadata)]
                metadata_1 = metadata
            elif metadata["facing_mic"] == 0:
                audio_path_2 = audio_paths[metadatas.index(metadata)]
                metadata_2 = metadata
        return DatasetSample([audio_path_1, audio_path_2], [metadata_1, metadata_2])

    @staticmethod
    def _get_label(orientation_1: float, orientation_2: float) -> float:
        return orientation_2 - orientation_1

    @staticmethod
    def collate_fn(
        batch: list[TrainingSample],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        wavs1 = [sample.wav[0] for sample in batch if sample is not None]
        wavs2 = [sample.wav[1] for sample in batch if sample is not None]
        labels = [sample.label for sample in batch if sample is not None]
        mic_positions = np.array([sample.mic_position for sample in batch if sample is not None])
        src_positions = np.array([sample.src_position for sample in batch if sample is not None])
        return (
            pad_sequence(wavs1, batch_first=True),
            pad_sequence(wavs2, batch_first=True),
            torch.tensor(labels),
            torch.tensor(mic_positions),
            torch.tensor(src_positions),
        )

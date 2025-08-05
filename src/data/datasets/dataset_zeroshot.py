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
        dataset_path: str,
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
    def _find_samples(dataset_path: str, max_samples: Optional[int], audio_format: str) -> Iterator[DatasetSample]:
        for i, audio_path in enumerate(tqdm(Path(dataset_path).rglob(f"*.{audio_format}"), desc="Finding samples")):
            metadata_path = audio_path.parent / f"{audio_path.stem}_metadata.json"
            if metadata_path.exists():
                yield DatasetSample([audio_path], [ZeroShotOrientationDataset._load_json(metadata_path)])
            if max_samples is not None and i >= max_samples:
                break

    def __getitem__(self, idx: int) -> TrainingSample | None:
        try:
            assert (
                len(self.dataset_samples[idx].audio_path) == 1
            ), "Audio path must be a single Path object for ZeroShotDataset"
            assert len(self.dataset_samples[idx].metadata) == 1, "Metadata must be a single dict for ZeroShotDataset"
            audio_path = self.dataset_samples[idx].audio_path[0]
            metadata = self.dataset_samples[idx].metadata[0]
            wav = self._load_audio(audio_path)
            orientation = self._get_orientation(metadata)
            return TrainingSample(
                wav=[wav],
                label=orientation,
                mic_position=self._get_mic_position(metadata),
                src_position=self._get_src_position(metadata),
            )
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return None

    @staticmethod
    def collate_fn(batch: list[TrainingSample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        wavs = [sample.wav[0] for sample in batch if sample is not None]
        labels = [sample.label for sample in batch if sample is not None]
        mic_positions = np.array([sample.mic_position for sample in batch if sample is not None])
        src_positions = np.array([sample.src_position for sample in batch if sample is not None])
        return (
            pad_sequence(wavs, batch_first=True),
            torch.tensor(labels),
            torch.tensor(mic_positions),
            torch.tensor(src_positions),
        )

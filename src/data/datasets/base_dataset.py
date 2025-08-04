import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import NamedTuple, TypedDict

import numpy as np
import scipy.io.wavfile as wav
import torch
from torch.utils.data import Dataset


class TrainingSample(NamedTuple):
    wav: list[torch.Tensor]
    label: float
    mic_position: np.ndarray
    src_position: np.ndarray


class DatasetSample(NamedTuple):
    audio_path: list[Path]
    metadata: list[dict]


class AudioParams(NamedTuple):
    sr: int
    nb_channels: int
    max_duration: float = 15.0
    eps: float = 1e-6
    format: str = "wav"


class BaseOrientationDataset(Dataset, ABC):
    dataset_samples: list[DatasetSample]
    audio_params: AudioParams

    def __len__(self) -> int:
        return len(self.dataset_samples)

    @abstractmethod
    def __getitem__(self, idx: int) -> TrainingSample | None: ...

    @classmethod
    @abstractmethod
    def from_config(cls, *args, **kwargs) -> "BaseOrientationDataset": ...

    @staticmethod
    @abstractmethod
    def collate_fn(batch: list[TrainingSample]): ...

    def _load_audio(self, audio_path: Path) -> torch.Tensor:
        _, audio = wav.read(audio_path)
        audio = audio[:, : self.audio_params.nb_channels] / 32768.0 + self.audio_params.eps
        return torch.as_tensor(audio).float()[: int(self.audio_params.sr * self.audio_params.max_duration), :]

    @staticmethod
    def _load_json(json_path: Path) -> dict:
        return json.loads(json_path.read_text())

    @staticmethod
    def _get_orientation(metadata: dict) -> float:
        return metadata["orientation"][0]

    @staticmethod
    def _get_mic_position(metadata: dict) -> np.ndarray:
        return np.array(metadata["mic_position"])

    @staticmethod
    def _get_src_position(metadata: dict) -> np.ndarray:
        return np.array(metadata["source_position"])

    @staticmethod
    def _orientation_to_label(orientation: float) -> int:
        # define labels for orientation, if 0-45 and 315-360, 45-135, 135-225, 225-315, then 0, 1, 2, 3
        if 0 <= orientation < 45 or 315 <= orientation <= 360:
            return 0
        elif 45 <= orientation < 135:
            return 1
        elif 135 <= orientation < 225:
            return 2
        elif 225 <= orientation < 315:
            return 3
        else:
            raise ValueError(f"Invalid orientation value {orientation}")

    @staticmethod
    def _orientation_to_xy(orientation: float) -> tuple[float, float]:
        return np.cos(np.radians(orientation)).astype(np.float32), np.sin(np.radians(orientation)).astype(np.float32)

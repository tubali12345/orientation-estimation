import json
import random
from pathlib import Path
from typing import NamedTuple, Optional

import numpy as np
import scipy.io.wavfile as wav
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
from tqdm import tqdm


class TrainingSample(NamedTuple):
    feats: torch.Tensor
    label: float
    mic_position: np.ndarray
    src_position: np.ndarray


class DatasetSample(NamedTuple):
    audio_path: Path
    metadata: dict


class AudioOrientation(Dataset):
    def __init__(
        self,
        dataset_path: str | Path,
        *,
        sr: int = 44100,
        duration: float = 10.0,
        limit: Optional[int] = None,
        eval: bool = False,
    ):
        self.dataset_path = Path(dataset_path)
        self.dataset_samples: list[DatasetSample] = []
        for i, audio_path in enumerate(tqdm(self.dataset_path.rglob("*.wav"), desc="Loading dataset samples")):
            self.dataset_samples.append(
                DatasetSample(audio_path, json.loads((audio_path.parent / "metadata.json").read_text()))
            )
            if limit is not None and i >= limit:
                break
        random.seed(42)
        random.shuffle(self.dataset_samples)
        if eval:
            self.dataset_samples = self.dataset_samples[: int(len(self.dataset_samples) * 0.1)]
        else:
            self.dataset_samples = self.dataset_samples[int(len(self.dataset_samples) * 0.1) :]
        print(f"Loaded {len(self.dataset_samples)} samples, eval: {eval}")
        self.sr = sr
        self.duration = duration

        self._nb_channels = 6
        self._eps = 1e-8

    def __len__(self) -> int:
        return len(self.dataset_samples)

    def __getitem__(self, idx: int) -> TrainingSample | None:
        try:
            feats = self._load_audio(self.dataset_samples[idx].audio_path)
            orientation = self._get_orientation(self.dataset_samples[idx].metadata)
            return TrainingSample(
                feats,
                orientation,
                self._get_mic_position(self.dataset_samples[idx].metadata),
                self._get_src_position(self.dataset_samples[idx].metadata),
            )
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return None

    def _load_audio(self, audio_path: Path) -> torch.Tensor:
        _, audio = wav.read(audio_path)
        audio = audio[:, : self._nb_channels] / 32768.0 + self._eps
        # return torch.as_tensor(audio).float()[int(self.sr * 2) : int(self.sr * 4), :]
        return torch.as_tensor(audio).float()[: int(self.sr * 15), :]

    def _get_orientation(self, metadata: dict) -> float:
        return metadata["orientation"][0]

    def _get_mic_position(self, metadata: dict) -> np.ndarray:
        return np.array(metadata["mic_position"])

    def _get_src_position(self, metadata: dict) -> np.ndarray:
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

    @staticmethod
    def collate_fn(batch: list[TrainingSample]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feats = [sample.feats for sample in batch if sample is not None]
        labels = [sample.label for sample in batch if sample is not None]
        mic_positions = [sample.mic_position for sample in batch if sample is not None]
        src_positions = [sample.src_position for sample in batch if sample is not None]
        return (
            pad_sequence(feats, batch_first=True),
            torch.tensor(labels),
            torch.tensor(mic_positions),
            torch.tensor(src_positions),
        )

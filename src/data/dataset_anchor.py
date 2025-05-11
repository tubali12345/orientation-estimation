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
    feats1: torch.Tensor
    feats2: torch.Tensor
    label: float
    mic_position: np.ndarray
    src_position: np.ndarray


class DatasetSample(NamedTuple):
    audio_path_1: Path
    audio_path_2: Path
    metadata_1: dict
    metadata_2: dict


class AudioOrientationAnchor(Dataset):
    def __init__(
        self,
        dataset_path: str | Path,
        *,
        sr: int = 44100,
        duration: float = 10.0,
        limit: Optional[int] = None,
        eval: bool = False,
        test: bool = False,
    ):
        self.dataset_path = Path(dataset_path)
        self.dataset_samples: list[DatasetSample] = []
        wav_parent_paths = {audio_path.parent for audio_path in self.dataset_path.rglob("*.wav")}
        if test:
            wav_parent_paths = {
                wav_parent_path
                for wav_parent_path in wav_parent_paths
                if wav_parent_path.parent.stem.endswith("_test")
            }
        else:
            wav_parent_paths = {
                wav_parent_path
                for wav_parent_path in wav_parent_paths
                if not wav_parent_path.parent.stem.endswith("_test")
            }
        for i, audio_parent_path in enumerate(tqdm(wav_parent_paths, desc="Loading dataset samples")):
            if sample := self._get_sample(audio_parent_path):
                self.dataset_samples.append(sample)
            if limit is not None and i >= limit:
                break
        random.seed(40)
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
            feats1 = self._load_audio(self.dataset_samples[idx].audio_path_1)
            feats2 = self._load_audio(self.dataset_samples[idx].audio_path_2)
            # cut to the shortest length to be the same size
            min_length = min(feats1.shape[0], feats2.shape[0])
            feats1 = feats1[:min_length, :]
            feats2 = feats2[:min_length, :]
            label = self._get_label(self.dataset_samples[idx].metadata_1, self.dataset_samples[idx].metadata_2)
            return TrainingSample(
                feats1,
                feats2,
                label,
                self._get_mic_position(self.dataset_samples[idx].metadata_1),
                self._get_src_position(self.dataset_samples[idx].metadata_1),
            )
        except Exception as e:
            print(f"Error loading sample {idx}: {e}")
            return None

    def _get_sample(self, folder_path: Path) -> DatasetSample | None:
        audio_paths = list(folder_path.glob("*.wav"))
        metadata_paths = [folder_path / f"{audio_path.stem}_metadata.json" for audio_path in audio_paths]
        metadatas = [json.loads(path.read_text()) for path in metadata_paths]

        if not (all(path.exists() for path in metadata_paths) and len(audio_paths) == 2):
            return

        for metadata in metadatas:
            if metadata["facing_mic"] == 1:
                audio_path_1 = audio_paths[metadatas.index(metadata)]
                metadata_1 = metadata
            elif metadata["facing_mic"] == 0:
                audio_path_2 = audio_paths[metadatas.index(metadata)]
                metadata_2 = metadata
        return DatasetSample(
            audio_path_1=audio_path_1,
            audio_path_2=audio_path_2,
            metadata_1=metadata_1,
            metadata_2=metadata_2,
        )

    def _load_audio(self, audio_path: Path) -> torch.Tensor:
        _, audio = wav.read(audio_path)
        audio = audio[:, : self._nb_channels] / 32768.0 + self._eps
        # return torch.as_tensor(audio).float()[int(self.sr * 2) : int(self.sr * 4), :]
        return torch.as_tensor(audio).float()[: int(self.sr * 15), :]

    def _get_label(self, metadata1: dict, metadata2: dict) -> float:
        return (metadata2["orientation"][0] - metadata1["orientation"][0]) % 360

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
    def collate_fn(
        batch: list[TrainingSample],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feats1 = [sample.feats1 for sample in batch if sample is not None]
        feats2 = [sample.feats2 for sample in batch if sample is not None]
        labels = [sample.label for sample in batch if sample is not None]
        mic_positions = np.array([sample.mic_position for sample in batch if sample is not None])
        src_positions = np.array([sample.src_position for sample in batch if sample is not None])
        return (
            pad_sequence(feats1, batch_first=True),
            pad_sequence(feats2, batch_first=True),
            torch.tensor(labels),
            torch.tensor(mic_positions),
            torch.tensor(src_positions),
        )

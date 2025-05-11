from pathlib import Path
from typing import Optional

import numpy as np

# import sounddevice as sd
import soundfile as sf
from scipy.signal import resample


def load_audio(
    speech_file: str | Path,
    sampling_rate: int,
    min_sample_rate: Optional[int] = None,
    verbose: bool = False,
    max_duration: Optional[float] = None,
) -> np.ndarray | None:
    speech, fs = sf.read(speech_file)
    if min_sample_rate is not None and fs < min_sample_rate:
        if verbose:
            print(
                f"Audio file {speech_file} has a sample rate of {fs} Hz, which is lower than the minimum sample rate of {min_sample_rate} Hz"
            )
        return None
    if fs != sampling_rate:
        if verbose:
            print(f"Resampling {speech_file} audio from {fs} Hz to {sampling_rate} Hz")
        speech = resample(speech, int(len(speech) * sampling_rate / fs))
    if max_duration is not None and len(speech) / fs > max_duration:
        speech = speech[: int(max_duration * fs)]
    return speech  # type: ignore


def save_audio(file_path: str | Path, audio: np.ndarray, sampling_rate: int) -> None:
    sf.write(file_path, audio, sampling_rate)


# def play_audio(audio: np.ndarray, sampling_rate: int) -> None:
#     sd.play(audio, sampling_rate)
#     sd.wait()

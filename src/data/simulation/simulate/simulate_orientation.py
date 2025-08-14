import random
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra

from ..utils.microphones import circular_mic
from .simulate_noise import (
    get_random_noise,
    render_diffuse_array_ambience_from_mono_ambience,
)

DEFAULT_FS = 44100
DEFAULT_ROOM_PARAMS = {
    "p": [5, 5, 3],
    "fs": DEFAULT_FS,
    "max_order": 20,
    "materials": pra.Material(0.5, 0.5),
}

DEFAULT_MIC_POSITION = [2.5, 2.5, 1.5]
DEFAULT_SOURCE_POSITION = [1.5, 2.5, 1.5]

DEFAULT_MIC = pra.MicrophoneArray(
    circular_mic(radius=0.045, center_position=DEFAULT_MIC_POSITION, num_mics=6), fs=DEFAULT_FS
)
DEFAULT_SOURCE_DIRECTIVITY = pra.Cardioid(
    pra.DirectionVector(azimuth=0, colatitude=90),
    gain=1.0,
)


def simulate_orientation(
    signal: np.ndarray,
    source_directivity: pra.Directivity,
    *,
    room: Optional[pra.room.Room] = None,
    mic: Optional[pra.MicrophoneArray] = None,
    source_position: Optional[List[float]] = None,
    mic_position: Optional[List[float]] = None,
    title: Optional[str] = None,
    plot: bool = True,
    noise_list: Optional[list[str]] = None,
    noise_position: Optional[List[float]] = None,
) -> np.ndarray:
    if room is None:
        room = pra.ShoeBox(**DEFAULT_ROOM_PARAMS)
    if mic is None:
        mic = DEFAULT_MIC
    if source_position is None:
        source_position = DEFAULT_SOURCE_POSITION
    room.add_microphone_array(mic)
    room.add_source(source_position, directivity=source_directivity, signal=signal)
    room.compute_rir()

    room.simulate()
    if plot:
        _plot_rir(room, title=title)
    assert room.mic_array is not None, "Room simulation failed, no microphone array found."
    assert room.mic_array.signals is not None, "Room simulation failed, no signals found."
    mic_signals = room.mic_array.signals

    if noise_list is not None:
        assert noise_position is not None, "Noise position must be provided if noise list is given."
        assert mic_position is not None, "Mic position must be provided if noise list is given."
        noise = get_noise(noise_list, signal, mic_signals, room.fs, mic_position)
        mic_signals += noise

    return mic_signals


def get_noise(
    noise_list: list[str], signal: np.ndarray, mic_signals: np.ndarray, sr: int, mic_position: list[float]
) -> np.ndarray:
    noise = get_random_noise(noise_list, max_len=len(signal), sr=sr)
    noise_multichannel = render_diffuse_array_ambience_from_mono_ambience(
        noise, sr, circular_mic(radius=0.045, center_position=mic_position, num_mics=6).T
    ).T
    noise_multichannel = _set_snr(noise_multichannel, mic_signals, (10, 20))
    noise_multichannel = _crop_or_pad_noise_multichannel(noise_multichannel, mic_signals)
    return noise_multichannel


def _crop_or_pad_noise_multichannel(noise: np.ndarray, signal: np.ndarray) -> np.ndarray:
    """Ensure that noise is the same length as the signal"""
    if noise.shape[1] > signal.shape[1]:
        return noise[:, : signal.shape[1]]
    else:
        return np.pad(noise, ((0, 0), (0, signal.shape[1] - noise.shape[1])), mode="constant")


def _set_snr(noise: np.ndarray, signal: np.ndarray, snr_db_range: tuple[float, float]) -> np.ndarray:
    signal_power = np.mean(signal**2, axis=1)
    noise_power = np.mean(noise**2, axis=1)
    avg_signal_power = np.mean(signal_power)
    avg_noise_power = np.mean(noise_power)
    current_snr_db = 10 * np.log10(avg_signal_power / avg_noise_power)
    if not (snr_db_range[0] <= current_snr_db <= snr_db_range[1]):
        target_snr_db = random.uniform(*snr_db_range)
        scaling_factor = np.sqrt(avg_signal_power / (10 ** (target_snr_db / 10) * avg_noise_power))
        noise = noise * scaling_factor
    return noise


def _plot_rir(room: pra.room.Room, title: Optional[str] = None) -> None:
    plt.figure()
    if room.rir is not None:
        for i, rir in enumerate(room.rir):
            plt.plot(rir[0], label=f"Mic {i+1}")
        plt.legend()
        plt.title(title or "Room Impulse Response")
        plt.show()

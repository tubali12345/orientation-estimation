from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pyroomacoustics as pra

from .microphones import circular_mic

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
    title: Optional[str] = None,
    plot: bool = True,
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

    mic_signals = room.simulate(return_premix=True).squeeze(0)
    if plot:
        _plot_rir(room, title=title)
    return mic_signals


def _plot_rir(room: pra.room.Room, title: Optional[str] = None) -> None:
    plt.figure()
    if room.rir is not None:
        for i, rir in enumerate(room.rir):
            plt.plot(rir[0], label=f"Mic {i+1}")
        plt.legend()
        plt.title(title or "Room Impulse Response")
        plt.show()

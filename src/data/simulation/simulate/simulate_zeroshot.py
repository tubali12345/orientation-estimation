from pathlib import Path
from typing import Optional

import numpy as np
import pyroomacoustics as pra

from src.data.directivity.directivity_data import PyData

from ..config_utils import RoomParams
from ..utils.audio_utils import load_audio, save_audio
from ..utils.microphones import circular_mic
from .simulate_orientation import simulate_orientation
from .utils import sample_speaker_position, save_metadata


def simulate_and_save(
    dataset_path: Path,
    output_dir_path: Path,
    audio_path: Path,
    measured_directivity_data: PyData,
    sr: int,
    room_params: RoomParams,
    noise_list: Optional[list] = None,
) -> None:
    try:
        file_out_dir = output_dir_path / audio_path.relative_to(dataset_path).parent / audio_path.stem
        audio_out_path = file_out_dir / f"{audio_path.stem}.wav"

        if audio_out_path.exists() and (file_out_dir / "metadata.json").exists():
            return

        audio = load_audio(audio_path, sampling_rate=sr, min_sample_rate=sr)

        if audio is None:
            return

        room_config = {
            # "p": [np.random.uniform(3, 10), np.random.uniform(3, 10), 3],
            "p": room_params["p"],
            "fs": sr,
            "max_order": room_params["max_order"],
            "materials": pra.Material(**room_params["materials"]),
        }

        min_dist_from_wall = room_params["min_dist_from_wall"]
        min_dist_from_mic = room_params["min_dist_from_mic"]
        mic_position = [
            np.random.uniform(min_dist_from_wall, room_config["p"][0] - min_dist_from_wall),
            np.random.uniform(min_dist_from_wall, room_config["p"][1] - min_dist_from_wall),
            1.5,
        ]

        source_position = [
            *sample_speaker_position(
                room_x_length=room_config["p"][0],
                room_y_length=room_config["p"][1],
                mic_position=mic_position,
                min_dist_from_wall=min_dist_from_wall,
                min_dist_from_mic=min_dist_from_mic,
            ),
            1.5,
        ]

        mic = pra.MicrophoneArray(circular_mic(radius=0.045, center_position=mic_position, num_mics=6), fs=sr)

        mic_src_angle = (
            np.degrees(np.arctan2(mic_position[1] - source_position[1], mic_position[0] - source_position[0])) % 360
        )

        relative_orientation = [np.random.uniform(0, 360)]
        absolute_orientation = [(mic_src_angle + relative_orientation[0]) % 360]

        hrtf = pra.MeasuredDirectivity(
            orientation=pra.Rotation3D(absolute_orientation, rot_order="z"),
            grid=measured_directivity_data.grid,
            impulse_responses=measured_directivity_data.impulse_responses,
            fs=measured_directivity_data.fs,
        )

        try:
            mic_signals = simulate_orientation(
                audio,
                hrtf,
                room=pra.ShoeBox(**room_config),
                mic=mic,
                source_position=source_position,
                plot=False,
                noise_list=noise_list,
            )
        except Exception as e:
            print(
                f"Failed to simulate {audio_path} with source position {source_position}, mic position {mic_position}, relative orientation {relative_orientation} with error: {e}"
            )
            return

        file_out_dir.mkdir(parents=True, exist_ok=True)
        save_audio(audio_out_path, mic_signals.T, sr)
        save_metadata(
            file_out_dir,
            audio_out_path.stem,
            {"orientation": relative_orientation, "source_position": source_position, "mic_position": mic_position},
        )
    except Exception as e:
        print(f"Failed to simulate {audio_path} with error: {e}")

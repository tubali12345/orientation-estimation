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
    audio_path1: Path,
    audio_path2: Path,
    measured_directivity_data: PyData,
    sr: int,
    room_params: RoomParams,
    noise_list: Optional[list] = None,
) -> None:
    try:
        file_out_dir = (
            output_dir_path / audio_path1.relative_to(dataset_path).parent / f"{audio_path1.stem}_{audio_path2.stem}"
        )

        if file_out_dir.exists() and len(list(file_out_dir.glob("*.wav"))) == 2:
            return

        audios = {
            audio_path1.stem: load_audio(audio_path1, sampling_rate=sr, min_sample_rate=sr, max_duration=15),
            audio_path2.stem: load_audio(audio_path2, sampling_rate=sr, min_sample_rate=sr, max_duration=15),
        }
        if any(audio is None for audio in audios.values()):
            return

        room_config = {
            # "p": [np.random.uniform(3, 12), np.random.uniform(3, 12), np.random.uniform(2, 6)],
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

        noise_position = (
            [
                np.random.uniform(min_dist_from_wall, room_config["p"][0] - min_dist_from_wall),
                np.random.uniform(min_dist_from_wall, room_config["p"][1] - min_dist_from_wall),
                1.5,
            ]
            if noise_list
            else None
        )

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

        for orientation, audio_stem in zip([[mic_src_angle], absolute_orientation], audios.keys()):
            hrtf = pra.MeasuredDirectivity(
                orientation=pra.Rotation3D(orientation, rot_order="z"),
                grid=measured_directivity_data.grid,
                impulse_responses=measured_directivity_data.impulse_responses,
                fs=measured_directivity_data.fs,
            )

            mic_signals = simulate_orientation(
                audios[audio_stem],  # type: ignore
                hrtf,
                room=pra.ShoeBox(**room_config),
                mic=mic,
                source_position=source_position,
                mic_position=mic_position,
                plot=False,
                noise_list=noise_list,
                noise_position=noise_position,
            )

            audio_out_path = file_out_dir / f"{audio_stem}.wav"

            file_out_dir.mkdir(parents=True, exist_ok=True)
            save_audio(audio_out_path, mic_signals.T, sr)
            facing_mic = 1 if orientation[0] == mic_src_angle else 0
            save_metadata(
                file_out_dir,
                audio_stem,
                {
                    "orientation": orientation,
                    "facing_mic": facing_mic,
                    "source_position": source_position,
                    "mic_position": mic_position,
                    "room_params": room_config["p"],
                },
            )
    except Exception as e:
        print(
            f"Failed to simulate {file_out_dir.stem} with source position {source_position}, mic position {mic_position}, relative orientation {relative_orientation} with error: {e}"
        )

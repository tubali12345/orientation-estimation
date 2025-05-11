import concurrent.futures as cf
import json
import math
import random
from pathlib import Path

import numpy as np
import pyroomacoustics as pra
from tqdm import tqdm

from src.data.directivity_data import MatData, PyData
from src.simulations.audio_utils import load_audio, save_audio
from src.simulations.microphones import circular_mic
from src.simulations.simulations import DEFAULT_FS, simulate_orientation


def simulate_dataset_multiprocess(
    dataset_path: str | Path,
    output_dir_path: str | Path,
    measured_directivity_file_path: str | Path,
    test: bool = False,
) -> None:
    dataset_path = Path(dataset_path)
    output_dir_path = Path(output_dir_path)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    measured_directivity_data: PyData = MatData.from_file(measured_directivity_file_path).to_pydata()

    audio_paths = list(dataset_path.rglob("*.mp3"))
    random.shuffle(audio_paths)

    with cf.ProcessPoolExecutor(max_workers=20) as executor:
        futures = [
            executor.submit(
                _simulate_and_save_pair,
                dataset_path,
                audio_path,
                measured_directivity_data,
                output_dir_path,
                test,
            )
            for audio_path in tqdm(audio_paths, desc="Submitting jobs")
        ]
        for future in tqdm(cf.as_completed(futures), total=len(futures), desc="Simulating dataset"):
            future.result()


def simulate_dataset(
    dataset_path: str | Path, output_dir_path: str | Path, measured_directivity_file_path: str | Path
) -> None:
    dataset_path = Path(dataset_path)
    output_dir_path = Path(output_dir_path)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    measured_directivity_data: PyData = MatData.from_file(measured_directivity_file_path).to_pydata()

    for audio_path in tqdm(list(dataset_path.rglob("*.mp3")), desc="Simulating dataset"):
        _simulate_and_save_pair(dataset_path, audio_path, measured_directivity_data, output_dir_path)


def _simulate_and_save(
    dataset_path: Path, audio_path: Path, measured_directivity_data: PyData, output_dir_path: Path, count: int = 0
) -> None:
    file_out_dir = output_dir_path / audio_path.relative_to(dataset_path).parent / audio_path.stem
    audio_out_path = file_out_dir / f"{audio_path.stem}.wav"

    if audio_out_path.exists() and (file_out_dir / "metadata.json").exists():
        return

    audio = load_audio(audio_path, sampling_rate=DEFAULT_FS, min_sample_rate=DEFAULT_FS)

    if audio is None:
        return

    room_params = {
        # "p": [np.random.uniform(3, 10), np.random.uniform(3, 10), 3],
        "p": [5, 5, 3],
        "fs": DEFAULT_FS,
        "max_order": 20,
        "materials": pra.Material(0.5, 0.5),
    }

    min_dist_from_wall = 0.3
    min_dist_from_mic = 0.5
    mic_position = [
        np.random.uniform(min_dist_from_wall, room_params["p"][0] - min_dist_from_wall),
        np.random.uniform(min_dist_from_wall, room_params["p"][1] - min_dist_from_wall),
        1.5,
    ]

    def sample_speaker_position():
        min_x, max_x = min_dist_from_wall, room_params["p"][0] - min_dist_from_wall
        min_y, max_y = min_dist_from_wall, room_params["p"][1] - min_dist_from_wall

        while True:
            x = np.random.uniform(min_x, max_x)
            y = np.random.uniform(min_y, max_y)

            if math.dist((x, y), mic_position[:2]) > min_dist_from_mic:
                return (x, y)

    source_position = [*sample_speaker_position(), 1.5]

    mic = pra.MicrophoneArray(circular_mic(radius=0.045, center_position=mic_position, num_mics=6), fs=DEFAULT_FS)

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
            room=pra.ShoeBox(**room_params),
            mic=mic,
            source_position=source_position,
            plot=False,
        )
    except Exception as e:
        if count < 3:
            _simulate_and_save(dataset_path, audio_path, measured_directivity_data, output_dir_path, count + 1)
        print(
            f"Failed to simulate {audio_path} {count} times, with source position {source_position}, mic position {mic_position}, relative orientation {relative_orientation} with error: {e}"
        )
        return

    file_out_dir.mkdir(parents=True, exist_ok=True)
    save_audio(audio_out_path, mic_signals.T, DEFAULT_FS)
    _save_metadata(
        file_out_dir,
        "",
        {"orientation": relative_orientation, "source_position": source_position, "mic_position": mic_position},
    )


def _simulate_and_save_pair(
    dataset_path: Path,
    audio_path: Path,
    measured_directivity_data: PyData,
    output_dir_path: Path,
    test: bool = False,
) -> None:
    try:
        file_out_dir = output_dir_path / audio_path.relative_to(dataset_path).parent / audio_path.stem
        audio_out_path = file_out_dir / f"{audio_path.stem}.wav"

        if file_out_dir.exists() and len(list(file_out_dir.glob("*.wav"))) > 0:
            return

        if test:
            file_out_dir = output_dir_path / f"{audio_path.relative_to(dataset_path).parent}_test" / audio_path.stem

            if file_out_dir.exists() and len(list(file_out_dir.glob("*.wav"))) == 2:
                return

        audio = load_audio(audio_path, sampling_rate=DEFAULT_FS, min_sample_rate=DEFAULT_FS)

        if audio is None:
            return

        room_params = {
            "p": [np.random.uniform(3, 12), np.random.uniform(3, 12), np.random.uniform(2, 6)],
            # "p": [5, 5, 3],
            "fs": DEFAULT_FS,
            "max_order": 20,
            "materials": pra.Material(0.5, 0.5),
        }

        min_dist_from_wall = 0.3
        min_dist_from_mic = 0.5
        mic_position = [
            np.random.uniform(min_dist_from_wall, room_params["p"][0] - min_dist_from_wall),
            np.random.uniform(min_dist_from_wall, room_params["p"][1] - min_dist_from_wall),
            1.5,
        ]

        def sample_speaker_position():
            min_x, max_x = min_dist_from_wall, room_params["p"][0] - min_dist_from_wall
            min_y, max_y = min_dist_from_wall, room_params["p"][1] - min_dist_from_wall

            while True:
                x = np.random.uniform(min_x, max_x)
                y = np.random.uniform(min_y, max_y)

                if math.dist((x, y), mic_position[:2]) > min_dist_from_mic:
                    return (x, y)

        source_position = [*sample_speaker_position(), 1.5]

        mic = pra.MicrophoneArray(circular_mic(radius=0.045, center_position=mic_position, num_mics=6), fs=DEFAULT_FS)

        mic_src_angle = (
            np.degrees(np.arctan2(mic_position[1] - source_position[1], mic_position[0] - source_position[0])) % 360
        )

        relative_orientation = [np.random.uniform(0, 360)]
        absolute_orientation = [(mic_src_angle + relative_orientation[0]) % 360]

        for orientation in [[mic_src_angle], absolute_orientation]:
            hrtf = pra.MeasuredDirectivity(
                orientation=pra.Rotation3D(orientation, rot_order="z"),
                grid=measured_directivity_data.grid,
                impulse_responses=measured_directivity_data.impulse_responses,
                fs=measured_directivity_data.fs,
            )

            mic_signals = simulate_orientation(
                audio,
                hrtf,
                room=pra.ShoeBox(**room_params),
                mic=mic,
                source_position=source_position,
                plot=False,
            )

            audio_out_path = file_out_dir / f"{audio_path.stem}_{round(orientation[0], 4)}.wav"

            file_out_dir.mkdir(parents=True, exist_ok=True)
            save_audio(audio_out_path, mic_signals.T, DEFAULT_FS)
            facing_mic = 1 if orientation[0] == mic_src_angle else 0
            _save_metadata(
                file_out_dir,
                audio_out_path.stem,
                {
                    "orientation": orientation,
                    "facing_mic": facing_mic,
                    "source_position": source_position,
                    "mic_position": mic_position,
                    "room_params": room_params["p"],
                },
            )
    except Exception as e:
        print(
            f"Failed to simulate {audio_path} with source position {source_position}, mic position {mic_position}, relative orientation {relative_orientation} with error: {e}"
        )


def _save_metadata(outdir_path: Path, stem: str, metadata: dict) -> None:
    metadata_path = outdir_path / f"{stem}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=4))


def read_json_metadata(metadata_path: Path) -> dict:
    return json.loads(metadata_path.read_text())


if __name__ == "__main__":
    simulate_dataset_multiprocess(
        dataset_path="/ssd2/en_commonvoice/en/clips",
        output_dir_path="/ssd2/en_commonvoice_17.0_rs_rm_rr_pair/",
        test=True,
        measured_directivity_file_path="/home/turib/thesis/3707708/AppliedAcousticsChalmers/sound-source-directivities-2020-03-12/AppliedAcousticsChalmers-sound-source-directivities-f48ec35/DirPat_singing_voice/irs_DirPat_a_long_sweep_N9_reg.mat",
    )

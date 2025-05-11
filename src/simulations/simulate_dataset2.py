import concurrent.futures as cf
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import pyroomacoustics as pra
from tqdm import tqdm

from src.data.directivity_data import MatData, PyData
from src.simulations.audio_utils import load_audio, save_audio
from src.simulations.microphones import circular_mic
from src.simulations.simulations import DEFAULT_FS, simulate_orientation


def simulate_dataset_multiprocess(
    dataset_path: str | Path,
    split: str,
    output_dir_path: str | Path,
    measured_directivity_file_path: str | Path,
    test: bool = False,
) -> None:
    dataset_path = Path(dataset_path)
    output_dir_path = Path(f"{output_dir_path}/{split}")
    # output_dir_path.mkdir(parents=True, exist_ok=True)
    metadata = pd.read_csv(f"{dataset_path}/{split}.tsv", sep="\t")
    # group by client_id
    speaker_audiopaths = metadata.groupby("client_id").agg({"path": lambda x: list(x)}).reset_index()
    speaker_audiopaths["path"] = speaker_audiopaths["path"].apply(
        lambda x: [Path(f"{dataset_path}/clips/{path}") for path in x]
    )

    # filter the paths to only include where load_audio doesnt return None
    batch_size = 100
    filtered_paths = []

    for i in tqdm(range(batch_size * 400, len(speaker_audiopaths), batch_size), desc="Filtering audio paths"):
        batch = speaker_audiopaths.iloc[i : i + batch_size].copy()
        batch["path"] = batch["path"].apply(
            lambda x: [
                audio_path
                for audio_path in x
                if audio_path.exists()
                and load_audio(audio_path, sampling_rate=DEFAULT_FS, min_sample_rate=DEFAULT_FS) is not None
            ]
        )
        filtered_paths.append(batch)

    # Combine the filtered batches back into a single DataFrame
    speaker_audiopaths = pd.concat(filtered_paths, ignore_index=True)

    # print distribution of number of audio files per speaker
    print(speaker_audiopaths["path"].apply(len).describe())
    # filter out lists with less than 2 elements
    speaker_audiopaths = speaker_audiopaths[speaker_audiopaths["path"].apply(lambda x: len(x) > 1)]

    # print distribution of number of audio files per speaker
    print(speaker_audiopaths["path"].apply(len).describe())

    audio_path_pairs = [_generate_pairs(audio_paths) for audio_paths in speaker_audiopaths["path"]]
    random.shuffle(audio_path_pairs)

    del speaker_audiopaths
    del metadata
    del filtered_paths

    measured_directivity_data: PyData = MatData.from_file(measured_directivity_file_path).to_pydata()

    with cf.ProcessPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(
                _simulate_and_save_pair,
                dataset_path,
                audio_path1,
                audio_path2,
                measured_directivity_data,
                output_dir_path,
                test=test,
            )
            for audio_paths in tqdm(audio_path_pairs, desc="Submitting jobs")
            for audio_path1, audio_path2 in audio_paths
        ]
        for future in tqdm(cf.as_completed(futures), total=len(futures), desc="Simulating dataset"):
            try:
                future.result()
            except Exception as e:
                print(f"Error in simulation: {e}")


def _generate_pairs(audio_paths: list[Path]) -> list[tuple[Path, Path]]:
    pairs = []
    # random.shuffle(audio_paths)
    for i in range(0, len(audio_paths), 2):
        if i + 1 < len(audio_paths):
            pairs.append((audio_paths[i], audio_paths[i + 1]))
        else:
            pairs.append((audio_paths[i], audio_paths[0]))
    return pairs


def _simulate_and_save_pair(
    dataset_path: Path,
    audio_path1: Path,
    audio_path2: Path,
    measured_directivity_data: PyData,
    output_dir_path: Path,
    test: bool = False,
) -> None:
    try:
        file_out_dir = (
            output_dir_path / audio_path1.relative_to(dataset_path).parent / f"{audio_path1.stem}_{audio_path2.stem}"
        )

        if file_out_dir.exists() and len(list(file_out_dir.glob("*.wav"))) == 2:
            return

        if test:
            file_out_dir = (
                output_dir_path
                / f"{audio_path1.relative_to(dataset_path).parent}_test"
                / f"{audio_path1.stem}_{audio_path2.stem}"
            )

            if file_out_dir.exists() and len(list(file_out_dir.glob("*.wav"))) == 2:
                return

        audios = {
            audio_path1.stem: load_audio(
                audio_path1, sampling_rate=DEFAULT_FS, min_sample_rate=DEFAULT_FS, max_duration=15
            ),
            audio_path2.stem: load_audio(
                audio_path2, sampling_rate=DEFAULT_FS, min_sample_rate=DEFAULT_FS, max_duration=15
            ),
        }

        if any(audio is None for audio in audios.values()):
            return

        room_params = {
            "p": [np.random.uniform(3, 12), np.random.uniform(3, 12), np.random.uniform(2, 6)],
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
                room=pra.ShoeBox(**room_params),
                mic=mic,
                source_position=source_position,
                plot=False,
            )

            audio_out_path = file_out_dir / f"{audio_stem}.wav"

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
            f"Failed to simulate {file_out_dir.stem} with source position {source_position}, mic position {mic_position}, relative orientation {relative_orientation} with error: {e}"
        )


def _save_metadata(outdir_path: Path, stem: str, metadata: dict) -> None:
    metadata_path = outdir_path / f"{stem}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=4))


def read_json_metadata(metadata_path: Path) -> dict:
    return json.loads(metadata_path.read_text())


if __name__ == "__main__":
    simulate_dataset_multiprocess(
        dataset_path="/ssd2/en_commonvoice/en",
        split="invalidated",
        test=True,
        output_dir_path="/ssd2/en_commonvoice_17.0_rs_rm_rr_ds_pair_2/",
        measured_directivity_file_path="/home/turib/thesis/3707708/AppliedAcousticsChalmers/sound-source-directivities-2020-03-12/AppliedAcousticsChalmers-sound-source-directivities-f48ec35/DirPat_singing_voice/irs_DirPat_a_long_sweep_N9_reg.mat",
    )

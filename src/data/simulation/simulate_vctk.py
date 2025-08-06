import argparse
import concurrent.futures as cf
import random
from collections import defaultdict
from pathlib import Path
from typing import Literal, Optional, Union

from tqdm import tqdm

from src.data.directivity.directivity_data import MatData, PyData
from src.data.simulation.config_utils import RoomParams, load_config
from src.data.simulation.simulate.simulate_singleshot import (
    simulate_and_save as simulate_and_save_singleshot,
)
from src.data.simulation.simulate.simulate_zeroshot import (
    simulate_and_save as simulate_and_save_zeroshot,
)


def _parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Simulate an orientation dataset from VCTK.")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to the simulation configuration jsonnet file."
    )
    args = parser.parse_args()
    return vars(args)


def simulate_vctk(
    dataset_path: Union[str, Path],
    output_dir_path: Union[str, Path],
    measured_directivity_dir_path: Union[str, Path],
    simulation_type: Literal["singleshot", "zeroshot"],
    sr: int,
    room_params: RoomParams,
    test_set_size: float = 0.1,
    max_workers: int = 16,
    noise_dir_path: Optional[Union[str, Path]] = None,
) -> None:
    dataset_path = Path(dataset_path)
    output_dir_path = Path(output_dir_path)
    measured_directivity_dir_path = Path(measured_directivity_dir_path)

    train_dir = output_dir_path / "train"
    train_dir.mkdir(parents=True, exist_ok=True)

    test_dir = output_dir_path / "test" if test_set_size else None
    if test_dir:
        test_dir.mkdir(parents=True, exist_ok=True)

    noise_list = (
        [str(p) for p in tqdm(Path(noise_dir_path).rglob("*.wav"), desc="Gathering noises")]
        if noise_dir_path
        else None
    )

    measured_directivities = [MatData.from_file(f).to_pydata() for f in measured_directivity_dir_path.rglob("*.mat")]

    all_wavs = list(dataset_path.rglob("*.wav"))
    all_speakers = {wav.parent.name for wav in all_wavs}

    test_speakers = set()
    if test_set_size:
        n_test = max(1, int(len(all_speakers) * test_set_size))
        test_speakers = set(random.sample(list(all_speakers), n_test))

    if simulation_type == "zeroshot":
        simulate_zeroshot(
            all_wavs=all_wavs,
            test_speakers=test_speakers,
            dataset_path=dataset_path,
            test_dir=test_dir,
            train_dir=train_dir,
            measured_directivities=measured_directivities,
            sr=sr,
            room_params=room_params,
            noise_list=noise_list,
            test_set_size=test_set_size,
            max_workers=max_workers,
        )
    elif simulation_type == "singleshot":
        simulate_singleshot(
            all_wavs=all_wavs,
            test_speakers=test_speakers,
            dataset_path=dataset_path,
            test_dir=test_dir,
            train_dir=train_dir,
            measured_directivities=measured_directivities,
            sr=sr,
            room_params=room_params,
            noise_list=noise_list,
            test_set_size=test_set_size,
            max_workers=max_workers,
        )
    else:
        raise ValueError(f"Invalid simulation type: {simulation_type}. Choose 'singleshot' or 'zeroshot'.")


def simulate_zeroshot(
    all_wavs: list[Path],
    test_speakers: set[str],
    dataset_path: Path,
    test_dir: Optional[Path],
    train_dir: Path,
    measured_directivities: list[PyData],
    sr: int,
    room_params: RoomParams,
    noise_list: Optional[list[str]],
    test_set_size: float,
    max_workers: int,
) -> None:
    train_audio_paths = [wav for wav in all_wavs if wav.parent.name not in test_speakers]
    test_audio_paths = [wav for wav in all_wavs if wav.parent.name in test_speakers] if test_set_size else []

    def simulate_batch(audio_paths, out_dir, desc):
        with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    simulate_and_save_zeroshot,
                    dataset_path,
                    out_dir,
                    audio_path,
                    random.choice(measured_directivities),
                    sr,
                    room_params,
                    noise_list,
                )
                for audio_path in tqdm(audio_paths, desc=f"Submitting jobs for {desc}")
            ]
            for future in tqdm(cf.as_completed(futures), total=len(futures), desc=f"Simulating {desc} dataset"):
                future.result()

    if test_audio_paths:
        simulate_batch(test_audio_paths, test_dir, "test")
    simulate_batch(train_audio_paths, train_dir, "train")


def simulate_singleshot(
    all_wavs: list[Path],
    test_speakers: set[str],
    dataset_path: Path,
    test_dir: Optional[Path],
    train_dir: Path,
    measured_directivities: list[PyData],
    sr: int,
    room_params: RoomParams,
    noise_list: Optional[list[str]],
    test_set_size: float,
    max_workers: int,
) -> None:
    train_speaker_audio_paths = defaultdict(list)
    test_speaker_audio_paths = defaultdict(list)

    for wav in all_wavs:
        if wav.parent.name in test_speakers:
            test_speaker_audio_paths[wav.parent.name].append(wav)
        else:
            train_speaker_audio_paths[wav.parent.name].append(wav)

    train_audio_path_pairs = [
        pair
        for speaker_audio_paths in train_speaker_audio_paths.values()
        for pair in _generate_pairs(speaker_audio_paths)
    ]
    test_audio_path_pairs = (
        [
            pair
            for speaker_audio_paths in test_speaker_audio_paths.values()
            for pair in _generate_pairs(speaker_audio_paths)
        ]
        if test_set_size
        else []
    )

    def simulate_batch(audio_path_pairs, out_dir, desc):
        with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    simulate_and_save_singleshot,
                    dataset_path,
                    out_dir,
                    audio_path1,
                    audio_path2,
                    random.choice(measured_directivities),
                    sr,
                    room_params,
                    noise_list,
                )
                for audio_path1, audio_path2 in tqdm(audio_path_pairs, desc=f"Submitting jobs for {desc}")
            ]
            for future in tqdm(cf.as_completed(futures), total=len(futures), desc=f"Simulating {desc} dataset"):
                future.result()

    if test_audio_path_pairs:
        simulate_batch(test_audio_path_pairs, test_dir, "test")
    simulate_batch(train_audio_path_pairs, train_dir, "train")


def _generate_pairs(audio_paths: list[Path]) -> list[tuple[Path, Path]]:
    if len(audio_paths) < 2:
        return [(audio_paths[0], audio_paths[0])]
    pairs = []
    # random.shuffle(audio_paths)
    for i in range(0, len(audio_paths), 2):
        if i + 1 < len(audio_paths):
            pairs.append((audio_paths[i], audio_paths[i + 1]))
        else:
            pairs.append((audio_paths[i], audio_paths[0]))
    return pairs


if __name__ == "__main__":
    args = _parse_args()
    simulation_config = load_config(args["config"])
    simulate_vctk(
        dataset_path=simulation_config["dataset_path"],
        output_dir_path=simulation_config["output_dir_path"],
        measured_directivity_dir_path=simulation_config["measured_directivity_dir_path"],
        simulation_type=simulation_config["simulation_type"],
        sr=simulation_config["sr"],
        room_params=simulation_config["room_params"],
        noise_dir_path=simulation_config.get("noise_dir_path"),
    )

import argparse
import concurrent.futures as cf
import random
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from data.directivity.directivity_data import MatData, PyData

from .config_utils import RoomParams, load_config
from .simulate.simulate_zeroshot import simulate_and_save


def _parse_args() -> dict:
    parser = argparse.ArgumentParser(description="Simulate an orientation dataset from CommonVoice.")
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to the simulation configuration jsonnet file."
    )
    args = parser.parse_args()
    return vars(args)


def simulate_vctk(
    dataset_path: str | Path,
    output_dir_path: str | Path,
    measured_directivity_dir_path: str | Path,
    sr: int,
    room_params: RoomParams,
    test_set_size: float = 0.1,
    max_workers: int = 16,
    noise_dir_path: Optional[str | Path] = None,
) -> None:
    dataset_path = Path(dataset_path)
    train_output_dir_path = Path(output_dir_path) / "train"
    train_output_dir_path.mkdir(parents=True, exist_ok=True)

    if test_set_size:
        test_output_dir_path = Path(output_dir_path).parent / "test"
        test_output_dir_path.mkdir(parents=True, exist_ok=True)

    measured_directivities: list[PyData] = [
        MatData.from_file(measured_directivity_file_path).to_pydata()
        for measured_directivity_file_path in list(Path(measured_directivity_dir_path).rglob("*.mat"))
    ]

    all_speakers = {wav_path.parent.name for wav_path in dataset_path.rglob("*.wav")}

    if test_set_size:
        test_speakers = set(random.sample(list(all_speakers), int(len(all_speakers) * test_set_size)))
        test_audio_paths = [
            wav_path for wav_path in dataset_path.rglob("*.wav") if wav_path.parent.name in test_speakers
        ]

    train_audio_paths = [
        wav_path for wav_path in dataset_path.rglob("*.wav") if wav_path.parent.name not in test_speakers
    ]

    noise_list = (
        [str(path) for path in tqdm(Path(noise_dir_path).rglob("*.wav"), desc="Gathering noises")]
        if noise_dir_path
        else None
    )

    if test_set_size:
        with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    simulate_and_save,
                    dataset_path,
                    test_output_dir_path,
                    audio_path,
                    random.choice(measured_directivities),
                    sr,
                    room_params,
                    noise_list,
                )
                for audio_path in tqdm(test_audio_paths, desc="Submitting jobs")
            ]
            for future in tqdm(cf.as_completed(futures), total=len(futures), desc="Simulating test dataset"):
                future.result()

    with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                simulate_and_save,
                dataset_path,
                train_output_dir_path,
                audio_path,
                random.choice(measured_directivities),
                sr,
                room_params,
                noise_list,
            )
            for audio_path in tqdm(train_audio_paths, desc="Submitting jobs")
        ]
        for future in tqdm(cf.as_completed(futures), total=len(futures), desc="Simulating train dataset"):
            future.result()


if __name__ == "__main__":
    args = _parse_args()
    simulation_config = load_config(args["config"])
    simulate_vctk(
        dataset_path=simulation_config["dataset_path"],
        output_dir_path=simulation_config["output_dir_path"],
        measured_directivity_dir_path=simulation_config["measured_directivity_dir_path"],
        sr=simulation_config["sr"],
        room_params=simulation_config["room_params"],
        noise_dir_path=simulation_config.get("noise_dir_path"),
    )

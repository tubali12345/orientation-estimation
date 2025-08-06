import argparse
import concurrent.futures as cf
import random
from pathlib import Path
from typing import Optional, Union

from tqdm import tqdm

from src.data.directivity.directivity_data import MatData, PyData
from src.data.simulation.config_utils import RoomParams, load_config
from src.data.simulation.simulate.simulate_zeroshot import simulate_and_save


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

    measured_directivities = [MatData.from_file(f).to_pydata() for f in measured_directivity_dir_path.rglob("*.mat")]

    all_wavs = list(dataset_path.rglob("*.wav"))
    all_speakers = {wav.parent.name for wav in all_wavs}

    test_speakers = set()
    if test_set_size:
        n_test = max(1, int(len(all_speakers) * test_set_size))
        test_speakers = set(random.sample(list(all_speakers), n_test))
    train_audio_paths = [wav for wav in all_wavs if wav.parent.name not in test_speakers]
    test_audio_paths = [wav for wav in all_wavs if wav.parent.name in test_speakers] if test_set_size else []

    noise_list = (
        [str(p) for p in tqdm(Path(noise_dir_path).rglob("*.wav"), desc="Gathering noises")]
        if noise_dir_path
        else None
    )

    def simulate_batch(audio_paths, out_dir, desc):
        with cf.ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(
                    simulate_and_save,
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

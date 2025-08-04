import json
from pathlib import Path
from typing import Optional, TypedDict

import _jsonnet


class RoomParams(TypedDict):
    p: list[float]
    max_order: int
    materials: dict
    min_dist_from_wall: float
    min_dist_from_mic: float


class SimulationConfig(TypedDict):
    dataset_path: str
    output_dir_path: str
    measured_directivity_dir_path: str
    noise_dir_path: Optional[str]
    sr: int
    room_params: RoomParams


def load_config(jsonnet_path: str | Path) -> SimulationConfig:
    """
    Load a JSON file and return its content as a dictionary."""
    jsonnet_path = Path(jsonnet_path)
    if not jsonnet_path.exists():
        raise FileNotFoundError(f"File {jsonnet_path} does not exist.")

    config_dict = json.loads(_jsonnet.evaluate_file(str(jsonnet_path)))
    room_params = config_dict.pop("room_params")
    return SimulationConfig(**config_dict, room_params=RoomParams(**room_params))

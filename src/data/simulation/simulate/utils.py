import json
import math
from pathlib import Path

import numpy as np


def sample_speaker_position(
    room_x_length: float,
    room_y_length: float,
    mic_position: list,
    min_dist_from_wall: float,
    min_dist_from_mic: float,
) -> tuple:
    """
    Sample a random position for the speaker within the room, ensuring it is at least a certain
    distance from the walls and the microphone."""
    min_x, max_x = min_dist_from_wall, room_x_length - min_dist_from_wall
    min_y, max_y = min_dist_from_wall, room_y_length - min_dist_from_wall

    while True:
        x = np.random.uniform(min_x, max_x)
        y = np.random.uniform(min_y, max_y)

        if math.dist((x, y), mic_position[:2]) > min_dist_from_mic:
            return (x, y)


def save_metadata(outdir_path: Path, stem: str, metadata: dict) -> None:
    metadata_path = outdir_path / f"{stem}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=4))

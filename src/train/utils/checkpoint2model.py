import argparse
from pathlib import Path

import torch


def _parse_args() -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", "-c", type=Path, required=True)
    parser.add_argument("--output_path", "-o", type=Path, required=False)
    return vars(parser.parse_args())


def checkpoint2model(checkpoint_path: Path, output_path: Path) -> None:
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model: torch.nn.Module = checkpoint["model"]
    torch.save(model.state_dict(), output_path)


def main(**params) -> None:
    if params["output_path"] is None:
        params["output_path"] = params["checkpoint_path"].with_suffix(".pth")
    checkpoint2model(**params)


if __name__ == "__main__":
    main(**_parse_args())

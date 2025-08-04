from pathlib import Path
from typing import Union


class TxtLogger:
    def __init__(self, log_path: Union[str, Path]):
        self.log_path = log_path

    def log(self, msg: str):
        with open(self.log_path, "a") as f:
            f.write(msg + "\n")

    def log_dict(self, dict_: dict):
        for k, v in dict_.items():
            self.log(f"{k}: {v}")

    def log_list(self, list_: list):
        for item in list_:
            self.log(item)

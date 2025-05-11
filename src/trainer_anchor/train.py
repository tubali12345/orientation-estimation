import json
from typing import Optional, Tuple

import _jsonnet
import torch
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
from torchsummary import summary

from src.data.dataset_anchor import AudioOrientationAnchor
from src.seldnet_anchor_model import SeldModelAnchor
from src.trainer_anchor.trainer import Frequency, Scheduler, Trainer


def load_config(config_path: str) -> dict:
    return json.loads(_jsonnet.evaluate_file(config_path))


def init_seld_model(model_config: dict, pretrained_model_path: Optional[str] = None):
    print("Initializing model...")
    model = SeldModelAnchor(model_config)
    if pretrained_model_path:
        print(f"Loading pretrained model from {pretrained_model_path}")
        state_dict = torch.load(pretrained_model_path, map_location="cpu")
        del state_dict["fnn_list.0.weight"]
        del state_dict["fnn_list.0.bias"]
        del state_dict["fnn_list.1.weight"]
        del state_dict["fnn_list.1.bias"]
        # for key in list(state_dict.keys()):
        #     if "audio" not in key:
        #         new_key = f"encoder.{key}"
        #         state_dict[new_key] = state_dict[key]
        #         del state_dict[key]
        model.load_state_dict(state_dict, strict=False)
    else:
        print("No pretrained model provided, initializing with random weights")
    summary(model, depth=2)
    return model


def init_trainer(trainer_config: dict, lr_scheduler_config: dict, model: SeldModelAnchor) -> Trainer:
    print("Initializing new trainer from config...")
    lr = lr_scheduler_config["initial_lr"]
    model = model.to(trainer_config["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.MSELoss()
    scheduler = Scheduler(
        getattr(torch.optim.lr_scheduler, lr_scheduler_config["scheduler"])(
            optimizer, **lr_scheduler_config["params"]
        ),
        Frequency[lr_scheduler_config["frequency"]],
    )
    return Trainer(model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion, **trainer_config)


def init_data_module(data_config: dict) -> Tuple[DataLoader, DataLoader]:
    print("Initializing data module...")
    train_dl = DataLoader(
        AudioOrientationAnchor(
            data_config["data_dir_path_train"],
            limit=100000,
            **data_config["dataset_params"],
        ),
        collate_fn=AudioOrientationAnchor.collate_fn,
        shuffle=True,
        **data_config["dataloader_params"],
    )
    valid_dl = DataLoader(
        AudioOrientationAnchor(
            data_config["data_dir_path_valid"],
            limit=100000,
            eval=True,
            **data_config["dataset_params"],
        ),
        collate_fn=AudioOrientationAnchor.collate_fn,
        shuffle=False,
        **data_config["dataloader_params"],
    )
    return train_dl, valid_dl


if __name__ == "__main__":
    config = load_config("/home/turib/thesis/src/config_anchor.jsonnet")

    model = init_seld_model(config["model_params"], config["pretrained_model_path"])

    trainer = init_trainer(config["trainer_params"], config["lr_scheduler_params"], model)

    train_dl, valid_dl = init_data_module(config["data_params"])

    torch.cuda.set_device(trainer.device)
    trainer.fit(train_dl, valid_dl)

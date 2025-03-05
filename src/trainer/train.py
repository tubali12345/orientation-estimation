import json
from typing import Optional, Tuple

import _jsonnet
import torch
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
from torchsummary import summary

from src.data.dataset import AudioOrientation
from src.seldnet_model import SeldModel
from src.trainer.trainer import Frequency, Scheduler, Trainer


def load_config(config_path: str) -> dict:
    return json.loads(_jsonnet.evaluate_file(config_path))


def init_seld_model(model_config: dict, pretrained_model_path: Optional[str] = None):
    print("Initializing model...")
    model = SeldModel(model_config)
    if pretrained_model_path:
        print(f"Loading pretrained model from {pretrained_model_path}")
        state_dict = torch.load(pretrained_model_path, map_location="cpu")
        del state_dict["fnn_list.1.weight"]
        del state_dict["fnn_list.1.bias"]
        del state_dict["conv_block_list.0.conv.weight"]
        model.load_state_dict(state_dict, strict=False)
    else:
        print("No pretrained model provided, initializing with random weights")
    summary(model, depth=2)
    return model


def init_trainer(trainer_config: dict, lr_scheduler_config: dict, model: SeldModel) -> Trainer:
    print("Initializing new trainer from config...")
    lr = lr_scheduler_config["initial_lr"]
    model = model.to(trainer_config["device"])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss() if model.nb_classes > 1 else torch.nn.MSELoss()
    scheduler = Scheduler(
        getattr(torch.optim.lr_scheduler, lr_scheduler_config["scheduler"])(
            optimizer, **lr_scheduler_config["params"]
        ),
        Frequency[lr_scheduler_config["frequency"]],
    )
    trainer = Trainer(model=model, optimizer=optimizer, scheduler=scheduler, criterion=criterion, **trainer_config)
    return trainer


def init_data_module(data_config: dict) -> Tuple[DataLoader, DataLoader]:
    print("Initializing data module...")
    train_dl = DataLoader(
        AudioOrientation(
            data_config["data_dir_path_train"],
            limit=100000,
            **data_config["dataset_params"],
        ),
        collate_fn=AudioOrientation.collate_fn,
        shuffle=True,
        **data_config["dataloader_params"],
    )
    valid_dl = DataLoader(
        AudioOrientation(
            data_config["data_dir_path_valid"],
            limit=100000,
            eval=True,
            **data_config["dataset_params"],
        ),
        collate_fn=AudioOrientation.collate_fn,
        shuffle=False,
        **data_config["dataloader_params"],
    )
    return train_dl, valid_dl


if __name__ == "__main__":
    config = load_config("config.jsonnet")

    model = init_seld_model(config["model_params"], config["pretrained_model_path"])

    trainer = init_trainer(config["trainer_params"], config["lr_scheduler_params"], model)

    train_dl, valid_dl = init_data_module(config["data_params"])

    torch.cuda.set_device(trainer.device)
    trainer.fit(train_dl, valid_dl)

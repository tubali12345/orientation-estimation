import json
from pathlib import Path
from typing import Optional, Tuple

import _jsonnet
import torch
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
from torchsummary import summary

from src.data.datasets.base_dataset import AudioParams
from src.data.datasets.dataset_singleshot import SingleShotOrientationDataset
from src.model.audio_processor import AudioProcessorParams
from src.model.seldnet_singleshot_model import (
    EncoderParams,
    SeldModelSingleShot,
    SingleShotModelParams,
)
from src.train.trainer.base_trainer import Frequency, Scheduler
from src.train.trainer.singleshot_trainer import SingleShotTrainer
from src.train.validator.singleshot_validator import SingleShotValidator


def load_config(config_path: str) -> dict:
    return json.loads(_jsonnet.evaluate_file(config_path))


def init_seld_model(
    model_params: SingleShotModelParams,
    audio_processor_params: AudioProcessorParams,
    pretrained_model_path: Optional[str] = None,
):
    print("Initializing model...")
    model = SeldModelSingleShot(model_params, audio_processor_params)
    if pretrained_model_path:
        print(f"Loading pretrained model from {pretrained_model_path}")
        state_dict = torch.load(pretrained_model_path, map_location="cpu")
        state_dict = (
            _delete_or_rename_unmatched_keys(state_dict) if Path(pretrained_model_path).suffix == ".h5" else state_dict
        )
        model.load_state_dict(state_dict, strict=False)
    else:
        print("No pretrained model provided, initializing with random weights")
    summary(model, depth=2)
    return model


def _delete_or_rename_unmatched_keys(state_dict: dict) -> dict:
    del state_dict["conv_block_list.0.conv.weight"]
    del state_dict["fnn_list.0.weight"]
    del state_dict["fnn_list.0.bias"]
    del state_dict["fnn_list.1.weight"]
    del state_dict["fnn_list.1.bias"]
    for key in list(state_dict.keys()):
        if "audio" not in key:
            new_key = f"encoder.{key}"
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    return state_dict


def init_trainer(trainer_config: dict, lr_scheduler_config: dict, model: SeldModelSingleShot) -> SingleShotTrainer:
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
    validator = SingleShotValidator(device=trainer_config["device"])
    return SingleShotTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        criterion=criterion,
        validator=validator,
        **trainer_config,
    )


def init_data_module(data_config: dict, audio_params: AudioParams) -> Tuple[DataLoader, DataLoader]:
    print("Initializing data module...")
    train_dl = DataLoader(
        SingleShotOrientationDataset.from_config(
            data_config["data_dir_path_train"],
            audio_params,
            max_samples=data_config.get("max_samples"),
        ),
        collate_fn=SingleShotOrientationDataset.collate_fn,
        shuffle=True,
        **data_config["dataloader_params"],
    )
    valid_dl = DataLoader(
        SingleShotOrientationDataset.from_config(
            data_config["data_dir_path_valid"],
            audio_params,
        ),
        collate_fn=SingleShotOrientationDataset.collate_fn,
        shuffle=False,
        **data_config["dataloader_params"],
    )
    return train_dl, valid_dl


if __name__ == "__main__":
    config = load_config("/home/turib/thesis_git/src/train/configs/config_singleshot.jsonnet")

    audio_params = AudioParams(**config["audio_params"])
    audio_processor_params = AudioProcessorParams(
        sample_rate=audio_params.sr, nb_channels=audio_params.nb_channels, **config["audio_processor_params"]
    )
    encoder_params = EncoderParams(
        nb_channels=audio_params.nb_channels,
        nb_mel_bins=audio_processor_params["n_mels"],
        **config["model_params"].pop("encoder_params"),
    )
    model_params = SingleShotModelParams(encoder_params=encoder_params, **config["model_params"])

    model = init_seld_model(model_params, audio_processor_params, config["pretrained_model_path"])

    trainer = init_trainer(config["trainer_params"], config["lr_scheduler_params"], model)

    train_dl, valid_dl = init_data_module(config["data_params"], audio_params)

    torch.cuda.set_device(trainer.device)
    trainer.fit(train_dl, valid_dl)

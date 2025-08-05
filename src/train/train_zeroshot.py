import json
from typing import Optional, Tuple

import _jsonnet
import torch
import torch.optim.lr_scheduler
from torch.utils.data import DataLoader
from torchsummary import summary

from src.data.datasets.base_dataset import AudioParams
from src.data.datasets.dataset_zeroshot import ZeroShotOrientationDataset
from src.model.audio_processor import AudioProcessorParams
from src.model.seldnet_zeroshot_model import SeldModelZeroShot, ZeroShotModelParams
from src.train.trainer.base_trainer import Frequency, Scheduler
from src.train.trainer.zeroshot_trainer import ZeroShotTrainer
from src.train.validator.zeroshot_validator import ZeroShotValidator


def load_config(config_path: str) -> dict:
    return json.loads(_jsonnet.evaluate_file(config_path))


def init_seld_model(
    model_params: ZeroShotModelParams,
    audio_processor_params: AudioProcessorParams,
    pretrained_model_path: Optional[str] = None,
):
    print("Initializing model...")
    model = SeldModelZeroShot(model_params, audio_processor_params)
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


def init_trainer(trainer_config: dict, lr_scheduler_config: dict, model: SeldModelZeroShot) -> ZeroShotTrainer:
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
    validator = ZeroShotValidator(device=trainer_config["device"])
    return ZeroShotTrainer(
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
        ZeroShotOrientationDataset.from_config(
            data_config["data_dir_path_train"],
            audio_params,
            max_samples=data_config.get("max_samples"),
        ),
        collate_fn=ZeroShotOrientationDataset.collate_fn,
        shuffle=True,
        **data_config["dataloader_params"],
    )
    valid_dl = DataLoader(
        ZeroShotOrientationDataset.from_config(
            data_config["data_dir_path_valid"],
            audio_params,
        ),
        collate_fn=ZeroShotOrientationDataset.collate_fn,
        shuffle=False,
        **data_config["dataloader_params"],
    )
    return train_dl, valid_dl


if __name__ == "__main__":
    config = load_config("/home/turib/thesis_git/src/train/configs/config_zeroshot.jsonnet")

    audio_params = AudioParams(**config["audio_params"])
    audio_processor_params = AudioProcessorParams(
        sample_rate=audio_params.sr, nb_channels=audio_params.nb_channels, **config["audio_processor_params"]
    )
    model_params = ZeroShotModelParams(
        nb_channels=audio_params.nb_channels, nb_mel_bins=audio_processor_params["n_mels"], **config["model_params"]
    )

    model = init_seld_model(model_params, audio_processor_params, config["pretrained_model_path"])

    trainer = init_trainer(config["trainer_params"], config["lr_scheduler_params"], model)

    train_dl, valid_dl = init_data_module(config["data_params"], audio_params)

    torch.cuda.set_device(trainer.device)
    trainer.fit(train_dl, valid_dl)

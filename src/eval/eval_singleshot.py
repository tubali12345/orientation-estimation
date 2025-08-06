import torch
from torch.utils.data import DataLoader

from src.data.datasets.base_dataset import AudioParams
from src.data.datasets.dataset_singleshot import SingleShotOrientationDataset
from src.eval.vis import Visualizer
from src.model.seldnet_singleshot_model import SeldModelSingleShot
from src.train.validator.singleshot_validator import SingleShotValidator


def _load_model_from_checkpoint(checkpoint_path: str) -> SeldModelSingleShot:
    return torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)["model"]


def init_data_module(data_path: str, audio_params: AudioParams, dataloader_params: dict) -> DataLoader:
    return DataLoader(
        SingleShotOrientationDataset.from_config(
            data_path,
            audio_params,
        ),
        collate_fn=SingleShotOrientationDataset.collate_fn,
        shuffle=False,
        **dataloader_params,
    )


def eval(model: SeldModelSingleShot, dataloader: DataLoader, output_dir: str, device: str):
    validator = SingleShotValidator(torch.device(device))
    visualizer = Visualizer(output_dir)
    model = model.to(device)

    validator(model, dataloader)

    assert validator.result_reg.y_pred is not None, "y_pred should not be None after validation"
    assert validator.result_reg.y_true is not None, "y_true should not be None after validation"
    assert (
        validator.result_reg.angular_error_per_sample is not None
    ), "angular_error_per_sample should not be None after validation"
    assert validator.result_reg.mic_position is not None, "mic_position should not be None after validation"
    assert validator.result_reg.src_position is not None, "src_position should not be None after validation"

    visualizer.plot_hist_error(validator.result_reg.angular_error_per_sample)
    visualizer.plot_scatter_true_pred(validator.result_reg.y_true, validator.result_reg.y_pred)
    visualizer.plot_scatter_dist_error(
        validator.result_reg.mic_position,
        validator.result_reg.src_position,
        validator.result_reg.angular_error_per_sample,
    )
    visualizer.plot_grid_micpos_vs_error(
        validator.result_reg.mic_position,
        validator.result_reg.angular_error_per_sample,
        room_length=5,
        room_width=5,
    )
    visualizer.plot_grid_srcpos_vs_error(
        validator.result_reg.src_position, validator.result_reg.angular_error_per_sample, room_length=5, room_width=5
    )
    visualizer.plot_degree_vs_error(validator.result_reg.y_true, validator.result_reg.angular_error_per_sample, 10)
    visualizer.plot_error_spider_chart(validator.result_reg.y_true, validator.result_reg.angular_error_per_sample, 10)


if __name__ == "__main__":
    model_path = "/home/turib/thesis_git/model_weights/vctk/singleshot_rm_rs/checkpoints/epoch_1.ckpt"
    dataset_path = "/ssd3/VCTK-Corpus_singleshot_rs_rm/test/"
    output_dir = "plots/vctk/singleshot_rm_rs"
    model = _load_model_from_checkpoint(model_path)
    data_loader = init_data_module(
        data_path=dataset_path,
        audio_params=AudioParams(sr=48000, nb_channels=6),
        dataloader_params={"batch_size": 16},
    )
    eval(model, data_loader, output_dir, "cpu")

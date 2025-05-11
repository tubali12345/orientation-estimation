import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

import src.trainer.validator as val
from seldnet_model import SeldModel
from src.data.dataset import AudioOrientation


def _load_model_from_checkpoint(checkpoint_path) -> SeldModel:
    return torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=False)["model"]


def init_data_module(data_config: dict) -> DataLoader:
    return DataLoader(
        AudioOrientation(
            data_config["data_path"],
            limit=10**5,
            eval=True,
            **data_config["dataset_params"],
        ),
        collate_fn=AudioOrientation.collate_fn,
        shuffle=True,
        **data_config["dataloader_params"],
    )


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    return np.sqrt(np.sum((a - b) ** 2))


def positions_to_distance(mic_positions: np.ndarray, src_positions: np.ndarray) -> np.ndarray:
    distances = []
    for mic_pos, src_pos in zip(mic_positions, src_positions):
        distances.append(euclidean_distance(mic_pos, src_pos))
    return np.array(distances)


def plot_hist_error(errors: np.ndarray, bins: int = 100):
    plt.figure(figsize=(8, 4))
    sns.histplot(errors, bins=bins, kde=True)
    plt.xlabel("Angular Error (degrees)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Angular Errors")
    plt.savefig("plots/angular_error_hist.png")
    plt.show()


def plot_scatter_true_pred(y_true: np.ndarray, y_pred: np.ndarray):
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([0, 360], [0, 360], "r", linestyle="--")  # Perfect predictions line
    plt.xlabel("Target Angle (degrees)")
    plt.ylabel("Predicted Angle (degrees)")
    plt.title("Predicted vs. Target Angles")
    plt.xlim(0, 360)
    plt.ylim(0, 360)
    plt.savefig("plots/true_vs_pred_scatter.png")
    plt.show()


def plot_scatter_dist_error(distances: np.ndarray, errors: np.ndarray):
    plt.figure(figsize=(8, 4))
    sns.scatterplot(x=distances, y=errors)
    plt.xlabel("Distance (m)")
    plt.ylabel("Angular Error (degrees)")
    plt.title("Angular Error vs. Distance")
    plt.savefig("plots/angular_error_vs_distance.png")
    plt.show()


def plot_grid_micpos_vs_error(
    mic_positions: np.ndarray, errors: np.ndarray, room_length: float = 5.0, room_width: float = 5.0
):
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(mic_positions[:, 0], mic_positions[:, 1], c=errors, cmap="viridis", s=10)
    plt.colorbar(scatter, label="Angular Error (degrees)")

    # Add a rectangle to represent the 5x5 room
    ax.plot([0, room_length, room_length, 0, 0], [0, 0, room_width, room_width, 0], color="black", linewidth=1.5)

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("Angular Error by Microphone Position")
    plt.savefig("plots/micpos_vs_error.png")
    plt.show()


def plot_grid_srcpos_vs_error(
    src_positions: np.ndarray, errors: np.ndarray, room_length: float = 5.0, room_width: float = 5.0
):
    fig, ax = plt.subplots(figsize=(8, 8))
    scatter = ax.scatter(src_positions[:, 0], src_positions[:, 1], c=errors, cmap="viridis", s=10)
    plt.colorbar(scatter, label="Angular Error (degrees)")

    # Add a rectangle to represent the 5x5 room
    ax.plot([0, room_length, room_length, 0, 0], [0, 0, room_width, room_width, 0], color="black", linewidth=1.5)

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")
    ax.set_title("Angular Error by Source Position")
    plt.savefig("plots/srcpos_vs_error.png")
    plt.show()


def plot_degree_vs_error(y_true: np.ndarray, errors: np.ndarray):
    # Normalize degrees to 0-359 and bin errors
    degree_bins = np.arange(360)
    error_per_degree = np.zeros(360)
    counts = np.zeros(360)

    for deg, err in zip(y_true.astype(int) % 360, errors):
        error_per_degree[deg] += err
        counts[deg] += 1

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_error = np.divide(error_per_degree, counts)
        mean_error = np.nan_to_num(mean_error)

    # Polar plot setup
    theta = np.deg2rad(degree_bins)
    radii = np.ones_like(theta)  # Constant radius

    # Normalize colors
    norm = plt.Normalize(vmin=mean_error.min(), vmax=mean_error.max())
    cmap = plt.cm.viridis
    colors = cmap(norm(mean_error))

    # Plot
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(10, 8))
    bars = ax.bar(theta, radii, width=np.deg2rad(1), bottom=0.0, color=colors, edgecolor="black")

    ax.set_yticklabels([])
    ax.set_title("Prediction Error by Head Orientation Degree", va="bottom", fontsize=14)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", pad=0.1, shrink=0.8)
    cbar.set_label("Mean Angular Error (degrees)", fontsize=12)
    plt.savefig("plots/degree_vs_error.png", bbox_inches="tight")

    plt.show()


def eval(model: SeldModel, dataloader: DataLoader, device: str):
    validator = val.SOValidator(torch.device(device))
    model = model.to(device)

    validator(model, dataloader)

    plot_hist_error(validator.result_reg.angular_error_per_sample)
    plot_scatter_true_pred(validator.result_reg.y_true, validator.result_reg.y_pred)
    plot_scatter_dist_error(
        positions_to_distance(validator.result_reg.mic_position, validator.result_reg.src_position),
        validator.result_reg.angular_error_per_sample,
    )
    plot_grid_micpos_vs_error(
        validator.result_reg.mic_position,
        validator.result_reg.angular_error_per_sample,
    )
    plot_grid_srcpos_vs_error(
        validator.result_reg.src_position,
        validator.result_reg.angular_error_per_sample,
    )
    plot_degree_vs_error(validator.result_reg.y_true, validator.result_reg.angular_error_per_sample)


if __name__ == "__main__":
    model = _load_model_from_checkpoint("/home/turib/thesis/model_weights_rp_rm_reg/checkpoints/epoch_34.ckpt")
    data_loader = init_data_module(
        {
            "data_path": "/ssd2/en_commonvoice_17.0_rs_rm",
            "dataset_params": {
                "duration": 10,
                "sr": 44100,
            },
            "dataloader_params": {"batch_size": 16},
        }
    )
    eval(model, data_loader, "cpu")

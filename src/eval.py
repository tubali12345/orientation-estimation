from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from torch.utils.data import DataLoader

import src.trainer.validator as val
from data.datasets.dataset_zeroshot import AudioOrientation
from model.seldnet_zeroshot_model import SeldModel


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
    sns.histplot(errors, bins=bins, kde=True, color="blue", edgecolor="black")

    # Compute and plot mean
    mean_error = np.mean(errors)
    plt.axvline(mean_error, color="red", linestyle="--", linewidth=2, label=f"Mean = {mean_error:.2f}°")

    # Labels and styling
    plt.xlabel("Angular Error (degrees)")
    plt.ylabel("Frequency")
    plt.title("Distribution of Angular Errors")
    plt.legend()

    plt.savefig("plots/angular_error_hist.png", bbox_inches="tight")
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


def plot_scatter_dist_error(distances: np.ndarray, errors: np.ndarray, bin_width: float = 0.5):
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=distances, y=errors, alpha=0.3, s=10, color="gray", label="Raw Data")

    # Binning
    max_dist = np.max(distances)
    bins = np.arange(0, max_dist + bin_width, bin_width)
    bin_indices = np.digitize(distances, bins)

    bin_centers = []
    mean_errors = []

    for i in range(1, len(bins)):
        bin_mask = bin_indices == i
        if np.any(bin_mask):
            bin_center = (bins[i - 1] + bins[i]) / 2
            bin_mean_error = np.mean(errors[bin_mask])
            bin_centers.append(bin_center)
            mean_errors.append(bin_mean_error)

    # Plot trend line
    sns.lineplot(x=bin_centers, y=mean_errors, color="red", linewidth=2, label="Mean Error Trend")

    # Labels and styling
    plt.xlabel("Distance (m)")
    plt.ylabel("Angular Error (degrees)")
    plt.title("Angular Error vs. Distance")
    plt.legend()
    plt.savefig("plots/angular_error_vs_distance.png", bbox_inches="tight")
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


def plot_degree_vs_error(y_true: np.ndarray, errors: np.ndarray, resolution: int = 10):
    # Normalize degrees to 0-359 and bin errors based on the resolution
    degree_bins = np.arange(0, 360, resolution)
    error_per_bin = np.zeros(len(degree_bins))
    counts = np.zeros(len(degree_bins))

    for deg, err in zip(y_true.astype(int) % 360, errors):
        bin_index = deg // resolution
        error_per_bin[bin_index] += err
        counts[bin_index] += 1

    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_error = np.divide(error_per_bin, counts)
        mean_error = np.nan_to_num(mean_error)

    # Polar plot setup
    theta = np.deg2rad(degree_bins + resolution / 2)  # Center bins
    radii = np.ones_like(theta)  # Constant radius

    # Normalize colors
    norm = plt.Normalize(vmin=mean_error.min(), vmax=mean_error.max())
    cmap = plt.cm.viridis
    colors = cmap(norm(mean_error))

    # Plot
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"}, figsize=(10, 8))
    bars = ax.bar(theta, radii, width=np.deg2rad(resolution), bottom=0.0, color=colors, edgecolor="black")

    ax.set_yticklabels([])
    ax.set_title("Prediction Error by Head Orientation Degree", va="bottom", fontsize=14)

    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical", pad=0.1, shrink=0.8)
    cbar.set_label("Mean Angular Error (degrees)", fontsize=12)
    plt.savefig("plots/degree_vs_error.png", bbox_inches="tight")

    plt.show()


def plot_error_spider_chart(y_true: np.ndarray, errors: np.ndarray, resolution: int = 10):
    # Normalize degrees and initialize bins
    degree_bins = np.arange(0, 360, resolution)
    error_per_bin = np.zeros(len(degree_bins))
    counts = np.zeros(len(degree_bins))

    for deg, err in zip(y_true.astype(int) % 360, errors):
        bin_index = deg // resolution
        error_per_bin[bin_index] += err
        counts[bin_index] += 1

    # Mean error per bin
    with np.errstate(divide="ignore", invalid="ignore"):
        mean_error = np.divide(error_per_bin, counts)
        mean_error = np.nan_to_num(mean_error)

    # Prepare radar chart data
    labels = [f"{int(angle)}°" for angle in degree_bins]
    values = mean_error.tolist()
    values.append(values[0])  # Repeat first value to close the loop
    labels.append(labels[0])  # Repeat first label

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=True)

    # Plot spider chart
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.plot(angles, values, color="r", linewidth=2)
    ax.fill(angles, values, color="r", alpha=0.25)

    # Style
    ax.set_xticks(angles)
    ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticklabels([])
    ax.set_title("Mean Angular Error by Orientation (Spider Chart)", fontsize=14, pad=20)

    plt.savefig("plots/spider_chart_error.png", bbox_inches="tight")
    plt.show()


def eval(model: SeldModel, dataloader: DataLoader, device: str):
    validator = val.SOValidator(torch.device(device))
    model = model.to(device)

    validator(model, dataloader)

    out_dir = Path("plots")
    out_dir.mkdir(parents=True, exist_ok=True)

    plot_hist_error(validator.result_reg.angular_error_per_sample)
    plot_scatter_true_pred(validator.result_reg.y_true, validator.result_reg.y_pred)
    plot_scatter_dist_error(
        positions_to_distance(validator.result_reg.mic_position, validator.result_reg.src_position),
        validator.result_reg.angular_error_per_sample,
    )
    plot_grid_micpos_vs_error(
        validator.result_reg.mic_position,
        validator.result_reg.angular_error_per_sample,
        room_length=12,
        room_width=12,
    )
    plot_grid_srcpos_vs_error(
        validator.result_reg.src_position, validator.result_reg.angular_error_per_sample, room_length=12, room_width=12
    )
    plot_degree_vs_error(validator.result_reg.y_true, validator.result_reg.angular_error_per_sample, 10)
    plot_error_spider_chart(validator.result_reg.y_true, validator.result_reg.angular_error_per_sample, 10)


if __name__ == "__main__":
    model = _load_model_from_checkpoint("/home/turib/thesis/model_weights_rs_rm_noisy/checkpoints/epoch_30.ckpt")
    data_loader = init_data_module(
        {
            "data_path": "/ssd2/en_commonvoice_17.0_rs_rm_noisy",
            "dataset_params": {
                "duration": 10,
                "sr": 44100,
            },
            "dataloader_params": {"batch_size": 16},
        }
    )
    eval(model, data_loader, "cpu")

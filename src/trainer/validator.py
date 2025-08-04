from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.datasets.dataset_zeroshot import AudioOrientation
from ..model.seldnet_zeroshot_model import SeldModel


def _add_ndarrays(a: np.ndarray | None, b: np.ndarray | None) -> np.ndarray | None:
    if a is None and b is None:
        return None
    if a is None:
        return b
    if b is None:
        return a
    return np.concatenate((a, b))


@dataclass
class ClassificationResult:
    correct: int = 0
    total: int = 0
    correct_per_class: dict[int, int] = field(default_factory=dict)
    total_per_class: dict[int, int] = field(default_factory=dict)
    y_true: Optional[np.ndarray] = None
    y_pred: Optional[np.ndarray] = None

    def __add__(self, other: "ClassificationResult") -> "ClassificationResult":
        return ClassificationResult(
            correct=self.correct + other.correct,
            total=self.total + other.total,
            correct_per_class={
                k: self.correct_per_class.get(k, 0) + other.correct_per_class.get(k, 0)
                for k in set(self.correct_per_class) | set(other.correct_per_class)
            },
            total_per_class={
                k: self.total_per_class.get(k, 0) + other.total_per_class.get(k, 0)
                for k in set(self.total_per_class) | set(other.total_per_class)
            },
            y_true=_add_ndarrays(self.y_true, other.y_true),
            y_pred=_add_ndarrays(self.y_pred, other.y_pred),
        )

    def __str__(self) -> str:
        return f"Accuracy: {self.accuracy:.2f}, Accuracy per class: {self.accuracy_per_class}"

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total != 0 else 0.0

    @property
    def accuracy_per_class(self) -> dict[int, float]:
        return {k: self.correct_per_class[k] / self.total_per_class[k] for k in self.correct_per_class}


@dataclass
class RegressionResult:
    loss: float = 0
    angular_error: float = 0
    total: int = 0
    y_pred: Optional[np.ndarray] = None
    y_true: Optional[np.ndarray] = None
    mic_position: Optional[np.ndarray] = None
    src_position: Optional[np.ndarray] = None
    angular_error_per_sample: Optional[np.ndarray] = None

    def __add__(self, other: "RegressionResult") -> "RegressionResult":
        return RegressionResult(
            loss=self.loss + other.loss,
            angular_error=self.angular_error + other.angular_error,
            total=self.total + other.total,
            y_pred=_add_ndarrays(self.y_pred, other.y_pred),
            y_true=_add_ndarrays(self.y_true, other.y_true),
            mic_position=_add_ndarrays(self.mic_position, other.mic_position),
            src_position=_add_ndarrays(self.src_position, other.src_position),
            angular_error_per_sample=_add_ndarrays(self.angular_error_per_sample, other.angular_error_per_sample),
        )

    def __str__(self) -> str:
        return f"Loss: {self.avg_loss:.4f}, Angular error: {self.avg_angular_error:.2f}"

    @property
    def avg_loss(self) -> float:
        return self.loss / self.total if self.total != 0 else 0.0

    @property
    def avg_angular_error(self) -> float:
        return self.angular_error / self.total if self.total != 0 else 0.0


class SOValidator:
    def __init__(self, device: torch.device):
        self.device = device

        self.result_class = ClassificationResult()
        self.result_reg = RegressionResult()

        self.criteria = torch.nn.MSELoss(reduction="sum")

    def __call__(self, model: SeldModel, val_loader: DataLoader) -> None:
        self.validate(model, val_loader)

    def validate(self, model: SeldModel, val_loader: DataLoader) -> None:
        self.on_validation_start()
        model.eval()

        progress_bar = tqdm(val_loader, desc="Validating")
        for batch in progress_bar:
            batch_result_class, batch_result_reg = self.validate_batch(model, batch)
            self.result_class += batch_result_class
            self.result_reg += batch_result_reg
            progress_bar.set_postfix(
                {
                    "accuracy": self.result_class.accuracy,
                    "loss": self.result_reg.avg_loss,
                    "angular_error": self.result_reg.avg_angular_error,
                }
            )

        self.on_validation_end()

    @torch.no_grad()
    def validate_batch(self, model: SeldModel, batch) -> tuple[ClassificationResult, RegressionResult]:
        feats, target, mic_positions, src_positions = batch
        feats, target = feats.to(self.device), target.to(self.device)
        feats = model.audio_processor(feats)
        output = model(feats.permute(0, 3, 1, 2))
        return self._calc_stats(output, target, mic_positions, src_positions)

    @torch.no_grad()
    def _calc_stats(
        self, output: torch.Tensor, target: torch.Tensor, mic_positions, src_positions
    ) -> tuple[ClassificationResult, RegressionResult]:
        batch_result_class = ClassificationResult()
        batch_result_reg = RegressionResult()

        batch_result_reg.mic_position = mic_positions.cpu().numpy()
        batch_result_reg.src_position = src_positions.cpu().numpy()

        def xy_to_orientation(x, y):
            return torch.tensor((torch.atan2(y, x).item() * (180 / torch.pi)) % 360).to(self.device)

        output, target = output.to(self.device), target.to(self.device)

        target_reg = torch.stack(
            [torch.tensor(AudioOrientation._orientation_to_xy(orientation.item())) for orientation in target]
        ).to(self.device)

        loss = self.criteria(output, target_reg)
        batch_result_reg.loss = loss.item()
        batch_result_reg.total = len(target)

        target_angles = torch.atan2(target_reg[:, 0], target_reg[:, 1]) * (180 / torch.pi)
        pred_angles = torch.atan2(output[:, 0], output[:, 1]) * (180 / torch.pi)
        # bring angles to 0-360 range
        target_angles = (target_angles + 360) % 360
        pred_angles = (pred_angles + 360) % 360
        batch_result_reg.y_true = target_angles.cpu().numpy()
        batch_result_reg.y_pred = pred_angles.cpu().numpy()

        angular_error = self._calc_angular_error(output, target_reg)
        batch_result_reg.angular_error_per_sample = angular_error.cpu().numpy()
        batch_result_reg.angular_error = angular_error.sum().item()

        output = torch.stack([xy_to_orientation(*xy) for xy in output])

        output_classes = torch.tensor(
            [AudioOrientation._orientation_to_label(orientation.item()) for orientation in output]
        )
        target_classes = torch.tensor(
            [AudioOrientation._orientation_to_label(orientation.item()) for orientation in target]
        )

        batch_result_class.correct = int(torch.sum(output_classes == target_classes).item())
        batch_result_class.total = len(target)
        batch_result_class.y_true = target_classes.cpu().numpy()
        batch_result_class.y_pred = output_classes.cpu().numpy()

        for i in range(4):
            batch_result_class.correct_per_class[i] = int(
                torch.sum((output_classes == i) & (target_classes == i)).item()
            )
            batch_result_class.total_per_class[i] = int(torch.sum(target_classes == i).item())

        return batch_result_class, batch_result_reg

    @staticmethod
    def _calc_angular_error(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_angles = torch.atan2(target[:, 0], target[:, 1]) * (180 / torch.pi)
        pred_angles = torch.atan2(output[:, 0], output[:, 1]) * (180 / torch.pi)

        angular_error = torch.abs(target_angles - pred_angles)
        angular_error = torch.minimum(angular_error, 360 - angular_error)

        return angular_error

    def on_validation_start(self):
        self.result_class = ClassificationResult()
        self.result_reg = RegressionResult()

    def on_validation_end(self):
        print(f"Overall classification result: {self.result_class}")
        print(f"Overall regression result: {self.result_reg}")

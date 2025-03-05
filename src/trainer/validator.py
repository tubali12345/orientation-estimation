from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..data.dataset import AudioOrientation
from ..seldnet_model import SeldModel
from .utils import TaskType


@dataclass
class ClassificationResult:
    correct: int = 0
    total: int = 0
    correct_per_class: dict[int, int] = field(default_factory=dict)
    total_per_class: dict[int, int] = field(default_factory=dict)

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
    total: int = 0

    def __add__(self, other: "RegressionResult") -> "RegressionResult":
        return RegressionResult(loss=self.loss + other.loss, total=self.total + other.total)

    def __str__(self) -> str:
        return f"Loss: {self.avg_loss:.4f}"

    @property
    def avg_loss(self) -> float:
        return self.loss / self.total if self.total != 0 else 0.0


class SOValidator:
    def __init__(self, device: torch.device, task_type: TaskType):
        self.task_type = task_type
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
            progress_bar.set_postfix({"accuracy": self.result_class.accuracy, "loss": self.result_reg.loss})

        self.on_validation_end()

    @torch.no_grad()
    def validate_batch(self, model: SeldModel, batch) -> tuple[ClassificationResult, RegressionResult]:
        feats, target = batch
        feats, target = feats.to(self.device), target.to(self.device)
        feats = model.audio_processor(feats)
        output = model(feats.permute(0, 3, 1, 2))

        if self.task_type == TaskType.classification:
            return self._calc_classification_stats(output, target, model.nb_classes)
        return self._calc_regression_stats(output, target)

    def _calc_classification_stats(
        self, output: torch.Tensor, target: torch.Tensor, nb_classes: int
    ) -> tuple[ClassificationResult, RegressionResult]:
        batch_result_class = ClassificationResult()
        batch_result_reg = RegressionResult()

        pred = output.argmax(dim=1)
        batch_result_class.correct = int(torch.sum(pred == target).item())
        batch_result_class.total = len(target)

        for i in range(nb_classes):
            batch_result_class.correct_per_class[i] = int(torch.sum((pred == i) & (target == i)).item())
            batch_result_class.total_per_class[i] = int(torch.sum(target == i).item())

        return batch_result_class, batch_result_reg

    def _calc_regression_stats(
        self, output: torch.Tensor, target: torch.Tensor
    ) -> tuple[ClassificationResult, RegressionResult]:
        batch_result_class = ClassificationResult()
        batch_result_reg = RegressionResult()

        output = output.squeeze(1)

        loss = self.criteria(output, target)
        batch_result_reg.loss = loss.item()
        batch_result_reg.total = len(target)

        output_classes = torch.tensor(
            [AudioOrientation._orientation_to_label(orientation.item()) for orientation in output]
        )
        target_classes = torch.tensor(
            [AudioOrientation._orientation_to_label(orientation.item()) for orientation in target]
        )

        batch_result_class.correct = int(torch.sum(output_classes == target_classes).item())
        batch_result_class.total = len(target)

        for i in range(4):
            batch_result_class.correct_per_class[i] = int(
                torch.sum((output_classes == i) & (target_classes == i)).item()
            )
            batch_result_class.total_per_class[i] = int(torch.sum(target_classes == i).item())

        return batch_result_class, batch_result_reg

    def on_validation_start(self):
        self.result_class = ClassificationResult()
        self.result_reg = RegressionResult()

    def on_validation_end(self):
        print(f"Overall classification result: {self.result_class}")
        if self.task_type == TaskType.regression:
            print(f"Overall regression result: {self.result_reg}")

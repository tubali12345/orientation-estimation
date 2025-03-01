from dataclasses import dataclass, field

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from DCASE2024_seld_baseline.seldnet_model import SeldModel


@dataclass
class ValidationResult:
    correct: int = 0
    total: int = 0
    correct_per_class: dict[int, int] = field(default_factory=dict)
    total_per_class: dict[int, int] = field(default_factory=dict)

    def __add__(self, other: "ValidationResult") -> "ValidationResult":
        return ValidationResult(
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


class SOValidator:
    def __init__(self, device: torch.device):
        self.device = device

        self.result = ValidationResult()

    def __call__(self, model: SeldModel, val_loader: DataLoader) -> None:
        self.validate(model, val_loader)

    def validate(self, model: SeldModel, val_loader: DataLoader) -> None:
        self.on_validation_start()
        model.eval()

        progress_bar = tqdm(val_loader, desc="Validating")
        for batch in progress_bar:
            self.result += self.validate_batch(model, batch)
            progress_bar.set_postfix({"accuracy": self.result.accuracy})

        self.on_validation_end()

    @torch.no_grad()
    def validate_batch(self, model: SeldModel, batch) -> ValidationResult:
        batch_result = ValidationResult()

        feats, target = batch
        feats, target = feats.to(self.device), target.to(self.device)
        feats = model.audio_processor(feats)
        output = model(feats.permute(0, 3, 1, 2))

        pred = output.argmax(dim=1)
        batch_result.correct = int(torch.sum(pred == target).item())
        batch_result.total = len(target)

        for i in range(model.nb_classes):
            batch_result.correct_per_class[i] = int(torch.sum((pred == i) & (target == i)).item())
            batch_result.total_per_class[i] = int(torch.sum(target == i).item())

        return batch_result

    def on_validation_start(self):
        self.result = ValidationResult()

    def on_validation_end(self):
        print(f"Overall validation result: {self.result}")

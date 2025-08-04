from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from typing import NamedTuple, Union

import numpy as np
import pytorch_lightning.loggers as pl_loggers
import torch
import torch.nn.functional as F
from tqdm import tqdm

from ..data.datasets.dataset_anchor import AudioOrientationAnchor
from ..model.seldnet_singleshot_model import SeldModelAnchor
from .txt_logger import TxtLogger
from .validator import SOValidator


class Frequency(Enum):
    epoch = auto()
    step = auto()


class Scheduler(NamedTuple):
    scheduler: torch.optim.lr_scheduler.LRScheduler
    frequency: Frequency


@dataclass
class EpochStats:
    step_in_epoch: int = 0
    running_loss: float = 0
    running_grad: float = 0


class Trainer:
    def __init__(
        self,
        *,
        model: SeldModelAnchor,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        scheduler: Scheduler,
        max_epochs: int,
        device: torch.device,
        root_dir: Path,
        load_epoch: int = 0,
        global_step: int = 0,
        logger=None,
        log_every_n_steps: int = 30,
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.max_epochs = max_epochs
        self.device = device
        self.log_every_n_steps = log_every_n_steps
        self.current_epoch = load_epoch
        self.current_lr = scheduler.scheduler.get_last_lr()[0]
        self.global_step = global_step
        self.logger = logger
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

        self.logger = pl_loggers.TensorBoardLogger(self.root_dir)
        self.txt_logger = TxtLogger(self.root_dir / "train_log.txt")

        self.epoch_stats = EpochStats()

        self.validator = SOValidator(device)

    def fit(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader) -> None:
        self.on_fit_start()
        for _ in range(self.current_epoch, self.max_epochs):
            self.train_epoch(train_loader)
            if val_loader is not None:
                self.validator(self.model, val_loader)
                self.on_validation_end()
        self.on_fit_end()

    def train_epoch(self, train_dl: torch.utils.data.DataLoader) -> None:
        self.on_epoch_start()
        for batch_idx, batch in tqdm(
            enumerate(train_dl), total=len(train_dl), desc=f"Training epoch {self.current_epoch}", smoothing=0
        ):
            try:
                self.on_train_step_start()
                loss = self.train_step(batch, batch_idx)
                self.optimize(loss)
                self.on_train_step_end()
            except Exception as e:
                print(e)
        self.on_epoch_end()

    def train_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        feats1, feats2, target, _, _ = batch
        target = torch.stack(
            [torch.tensor(AudioOrientationAnchor._orientation_to_xy(orientation.item())) for orientation in target]
        )
        feats1, feats2, target = feats1.to(self.device), feats2.to(self.device), target.to(self.device)
        feats1 = self.model.audio_processor(feats1)
        feats2 = self.model.audio_processor(feats2)
        output = self.model(feats1.permute(0, 3, 1, 2), feats2.permute(0, 3, 1, 2))
        loss = self.criterion(output, target)

        angular_error = self._calc_mean_angular_error(output, target)

        if batch_idx % self.log_every_n_steps == 0:
            self.log_dict({"train_loss": loss.item(), "angular_error": angular_error}, unit=Frequency.step)
        return loss

    @staticmethod
    def _calc_mean_angular_error(output: torch.Tensor, target: torch.Tensor) -> float:
        target_angles = torch.atan2(target[:, 0], target[:, 1]) * (180 / torch.pi)
        pred_angles = torch.atan2(output[:, 0], output[:, 1]) * (180 / torch.pi)

        angular_error = torch.abs(target_angles - pred_angles)
        angular_error = torch.minimum(angular_error, 360 - angular_error)

        return angular_error.mean().item()

    def log_dict(self, metrics: dict, unit: Frequency = Frequency.step) -> None:
        if self.logger:
            self.logger.log_metrics(metrics, step=self.current_epoch if unit == Frequency.epoch else self.global_step)
        print(", ".join([f"{k}: {v:.6f}" for k, v in metrics.items()]))

    def on_fit_start(self) -> None:
        """Set up optimizer, scheduler, loss functions, etc."""
        pass

    def on_fit_end(self) -> None:
        """Clean up after training."""
        pass

    def on_epoch_start(self):
        """Set model to training mode and reset running metrics."""
        self.txt_logger.log_dict({"Classification": self.validator.result_class})
        self.txt_logger.log_dict({"Regression": self.validator.result_reg})
        self.model.train()
        self.current_epoch += 1
        self.txt_logger.log("-" * 50 + f"\nStarting epoch {self.current_epoch}, current lr: {self.current_lr}")

        # if self.epoch_stats.step_in_epoch == 0:
        #     self.current_epoch += 1
        #     self.txt_logger.log("-" * 50 + f"\nStarting epoch {self.current_epoch}, current lr: {self.current_lr}")
        #     print(f"Starting epoch {self.current_epoch}...")
        # else:
        #     print(f"Resuming epoch {self.current_epoch} from step {self.epoch_stats.step_in_epoch}")

    def on_epoch_end(self) -> None:
        """Log metrics, save model, etc."""
        self.save_checkpoint(f"epoch_{self.current_epoch}")
        if self.scheduler.frequency == Frequency.epoch:
            self.scheduler.scheduler.step()
            self.current_lr = self.scheduler.scheduler.get_last_lr()[0]
        self.epoch_stats = EpochStats()

    def on_train_step_start(self) -> None:
        self.optimizer.zero_grad()

    def on_train_step_end(self) -> None:
        self.global_step += 1
        self.epoch_stats.step_in_epoch += 1

        if self.scheduler.frequency == Frequency.step:
            self.scheduler.scheduler.step()
            self.current_lr = self.scheduler.scheduler.get_last_lr()[0]

        # if self.global_step % self.save_every_n_steps == 0 and self.global_step != 0 and self.cons_invalid_loss == 0:
        #     self.save_partial_checkpoint("partial")

    def on_validation_end(self) -> None:
        # log validation metrics
        if self.logger:
            self.logger.log_metrics(
                {
                    "val_loss": self.validator.result_reg.avg_loss,
                    "val_angular_error": self.validator.result_reg.avg_angular_error,
                    "val_accuracy": self.validator.result_class.accuracy,
                    **{
                        f"accuracy_class_{key}": value
                        for key, value in self.validator.result_class.accuracy_per_class.items()
                    },
                },
                step=self.current_epoch,
            )
            # # create confusion matrix
            # if isinstance(self.logger, pl_loggers.TensorBoardLogger):
            #     image = self.plot_confusion_matrix_from_dict(
            #         self.validator.result_class.y_true, self.validator.result_class.y_pred
            #     )
            #     self.logger.experiment.add_image("confusion_matrix", image, global_step=self.current_epoch)

    @staticmethod
    def plot_confusion_matrix_from_dict(y_true: np.ndarray | None, y_pred: np.ndarray | None) -> torch.Tensor:
        """
        Generates a confusion matrix image from a dictionary of class:accuracy pairs.

        Args:
            class_accuracy_dict (dict): Dictionary where keys are class labels and values are accuracy scores.

        Returns:
            torch.Tensor: Image tensor (C, H, W) format ready for logging in TensorBoard.
        """
        if y_true is None or y_pred is None:
            return torch.tensor([])
        import matplotlib.pyplot as plt
        import numpy as np
        import seaborn as sns
        from sklearn.metrics import confusion_matrix

        classes = list(set(y_true) | set(y_pred))
        conf_matrix = confusion_matrix(y_true, y_pred, labels=classes)

        # Plot the confusion matrix
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(conf_matrix, annot=True, fmt=".2f", cmap="Blues", xticklabels=classes, yticklabels=classes, ax=ax)

        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix (Simulated)")

        # Convert figure to tensor
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        img_tensor = torch.tensor(img).permute(2, 0, 1)  # Convert to (C, H, W)

        plt.close(fig)  # Close the plot to free memory

        return img_tensor

    def optimize(self, loss: torch.Tensor) -> None:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
        self.optimizer.step()

        self.epoch_stats.running_loss = (
            loss.item()
            if self.epoch_stats.running_loss == 0
            else (self.epoch_stats.running_loss * 0.99 + loss.item() * 0.01)
        )

    def save_checkpoint(self, checkpoint_name: str) -> None:
        checkpoint_path = self.root_dir / "checkpoints" / f"{checkpoint_name}.ckpt"
        self._init_dir(checkpoint_path.parent)
        checkpoint = {
            "model": self.model,
            "scheduler": self.scheduler,
            "current_epoch": self.current_epoch,
            "global_step": self.global_step,
        }
        torch.save(checkpoint, checkpoint_path)

    def _init_dir(self, dir_path: Union[str, Path]) -> Path:
        directory = Path(dir_path)
        directory.mkdir(parents=True, exist_ok=True)
        return directory

    @property
    def log_dir(self):
        if self.logger:
            if not isinstance(self.logger, (pl_loggers.TensorBoardLogger, pl_loggers.CSVLogger)):
                dirpath = self.logger.save_dir
            else:
                dirpath = self.logger.log_dir
        else:
            dirpath = self.root_dir
        return dirpath

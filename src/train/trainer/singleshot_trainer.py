import torch

from src.data.datasets.base_dataset import BaseOrientationDataset

from .base_trainer import BaseTrainer, Frequency


class ZeroShotTrainer(BaseTrainer):
    def train_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        wav1, wav2, target, _, _ = batch
        target = torch.stack(
            [torch.tensor(BaseOrientationDataset._orientation_to_xy(orientation.item())) for orientation in target]
        )
        wav1, wav2, target = wav1.to(self.device), wav2.to(self.device), target.to(self.device)
        feats1 = self.model.audio_processor(wav1)
        feats2 = self.model.audio_processor(wav2)
        output = self.model(feats1.permute(0, 3, 1, 2), feats2.permute(0, 3, 1, 2))
        loss = self.criterion(output, target)

        angular_error = self._calc_mean_angular_error(output, target)

        if batch_idx % self.log_every_n_steps == 0:
            self.log_dict(
                {
                    "train_loss": loss.item(),
                    "angular_error": angular_error,
                    "learning_rate": self.scheduler.scheduler.get_last_lr()[0],
                },
                unit=Frequency.step,
            )
        return loss

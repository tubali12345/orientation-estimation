import torch

from src.model.seldnet_singleshot_model import SeldModelSingleShot

from .base_validator import BaseValidator, ClassificationResult, RegressionResult


class SingleShotValidator(BaseValidator):
    @torch.no_grad()
    def validate_batch(self, model: SeldModelSingleShot, batch) -> tuple[ClassificationResult, RegressionResult]:
        wav1, wav2, target, mic_positions, src_positions = batch
        wav1, wav2, target = wav1.to(self.device), wav2.to(self.device), target.to(self.device)
        feats1 = model.audio_processor(wav1)
        feats2 = model.audio_processor(wav2)
        output = model(feats1.permute(0, 3, 1, 2), feats2.permute(0, 3, 1, 2))
        return self._calc_stats(output, target, mic_positions, src_positions)

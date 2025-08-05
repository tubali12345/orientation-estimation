import torch

from src.model.seldnet_zeroshot_model import SeldModelZeroShot

from .base_validator import BaseValidator, ClassificationResult, RegressionResult


class ZeroShotValidator(BaseValidator):
    @torch.no_grad()
    def validate_batch(self, model: SeldModelZeroShot, batch) -> tuple[ClassificationResult, RegressionResult]:
        wav, target, mic_positions, src_positions = batch
        wav, target = wav.to(self.device), target.to(self.device)
        feats = model.audio_processor(wav)
        output = model(feats.permute(0, 3, 1, 2))
        return self._calc_stats(output, target, mic_positions, src_positions)

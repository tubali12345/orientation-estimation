import torch
import torch.nn as nn
from librosa import filters


class AudioProcessor(nn.Module):

    def __init__(self, sample_rate, window_size, window_stride, n_mels, preemphasis=0.97):
        super(AudioProcessor, self).__init__()

        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.win_len = int(sample_rate * window_size)  # 400
        self.hop = int(sample_rate * window_stride)  # 160
        self.preemphasis = preemphasis
        self.nb_channels = 6
        self.n_fft = self._next_greater_power_of_2(self.win_len)

        self.register_buffer("window", torch.hann_window(self.win_len))
        mel_basis = filters.mel(sr=sample_rate, n_fft=self.n_fft, n_mels=n_mels)
        self.register_buffer("mel_basis", torch.FloatTensor(mel_basis))

    @staticmethod
    def _next_greater_power_of_2(x):
        return 2 ** (x - 1).bit_length()

    def forward(self, padded_audio):
        # if self.preemphasis > 0:
        #     padded_audio = padded_audio[:, 1:] - self.preemphasis * padded_audio[:, :-1]
        #     audio_size.add_(-1)
        nb_feat_frames = int(padded_audio.shape[1] / float(self.hop))
        spectra = []

        for i in range(self.nb_channels):
            stft_ch = (
                torch.stft(
                    padded_audio[:, :, i],
                    n_fft=self.n_fft,
                    win_length=self.win_len,
                    hop_length=self.hop,
                    window=self.window,
                    return_complex=False,
                )
                .pow(2.0)
                .sum(-1)
            )
            spectra.append(stft_ch[:, :, :nb_feat_frames])
        spectra = torch.stack(spectra, dim=1).transpose(1, 3)

        mel_feat = torch.zeros(
            (spectra.shape[0], spectra.shape[1], self.n_mels, spectra.shape[-1]), device=spectra.device
        )
        for ch_cnt in range(spectra.shape[-1]):
            mag_spectra = spectra[:, :, :, ch_cnt].transpose(1, 2)
            mel_spectra = (
                self.mel_basis.unsqueeze(0).expand(mag_spectra.shape[0], -1, -1).bmm(mag_spectra).transpose(1, 2)
            )
            log_mel_spectra = mel_spectra.clamp(min=1e-10).log10()
            mel_feat[:, :, :, ch_cnt] = log_mel_spectra

        return mel_feat

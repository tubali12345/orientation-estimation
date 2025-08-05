"""This file is a modified copy of https://github.com/partha2409/DCASE2024_seld_baseline/blob/main/seldnet_model.py"""

from typing import TypedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .audio_processor import AudioProcessor, AudioProcessorParams


class ZeroShotModelParams(TypedDict):
    nb_classes: int
    nb_channels: int
    nb_cnn2d_filt: int
    f_pool_size: list[int]
    dropout_rate: float
    nb_rnn_layers: int
    rnn_size: int
    nb_self_attn_layers: int
    nb_heads: int
    nb_fnn_layers: int
    fnn_size: int
    nb_mel_bins: int


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride, padding=padding
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x


class SeldModelZeroShot(torch.nn.Module):
    def __init__(self, model_params: ZeroShotModelParams, audio_processor_params: AudioProcessorParams):
        super().__init__()
        self.audio_processor = AudioProcessor(audio_processor_params)
        self.nb_classes = model_params["nb_classes"]
        self.params = model_params
        self.conv_block_list = nn.ModuleList()
        if len(model_params["f_pool_size"]):
            for conv_cnt in range(len(model_params["f_pool_size"])):
                self.conv_block_list.append(
                    ConvBlock(
                        in_channels=model_params["nb_cnn2d_filt"] if conv_cnt else model_params["nb_channels"],
                        out_channels=model_params["nb_cnn2d_filt"],
                    )
                )
                self.conv_block_list.append(
                    nn.MaxPool2d((model_params["f_pool_size"][conv_cnt], model_params["f_pool_size"][conv_cnt]))
                )
                self.conv_block_list.append(nn.Dropout2d(p=model_params["dropout_rate"]))

        self.gru_input_dim = model_params["nb_cnn2d_filt"] * int(
            np.floor(model_params["nb_mel_bins"] / np.prod(model_params["f_pool_size"]))
        )
        self.gru = torch.nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=model_params["rnn_size"],
            num_layers=model_params["nb_rnn_layers"],
            batch_first=True,
            dropout=model_params["dropout_rate"],
            bidirectional=True,
        )

        self.mhsa_block_list = nn.ModuleList()
        self.layer_norm_list = nn.ModuleList()
        for _ in range(model_params["nb_self_attn_layers"]):
            self.mhsa_block_list.append(
                nn.MultiheadAttention(
                    embed_dim=self.params["rnn_size"],
                    num_heads=self.params["nb_heads"],
                    dropout=self.params["dropout_rate"],
                    batch_first=True,
                )
            )
            self.layer_norm_list.append(nn.LayerNorm(self.params["rnn_size"]))

        self.fnn_list = torch.nn.ModuleList()
        if model_params["nb_fnn_layers"]:
            for fc_cnt in range(model_params["nb_fnn_layers"]):
                self.fnn_list.append(
                    nn.Linear(
                        model_params["fnn_size"] if fc_cnt else self.params["rnn_size"],
                        model_params["fnn_size"],
                        bias=True,
                    )
                )
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fnn_list.append(
            nn.Linear(
                model_params["fnn_size"] if model_params["nb_fnn_layers"] else self.params["rnn_size"],
                self.nb_classes,
                bias=True,
            )
        )

    def forward(self, x):
        for conv_cnt in range(len(self.conv_block_list)):
            x = self.conv_block_list[conv_cnt](x)

        x = x.transpose(1, 2).contiguous()
        x = x.view(x.shape[0], x.shape[1], -1).contiguous()
        (x, _) = self.gru(x)
        x = torch.tanh(x)
        x = x[:, :, x.shape[-1] // 2 :] * x[:, :, : x.shape[-1] // 2]

        for mhsa_cnt in range(len(self.mhsa_block_list)):
            x_attn_in = x
            x, _ = self.mhsa_block_list[mhsa_cnt](x_attn_in, x_attn_in, x_attn_in)
            x = x + x_attn_in
            x = self.layer_norm_list[mhsa_cnt](x)

        x = x.transpose(1, 2).contiguous()
        x = self.max_pool(x).squeeze(-1)

        for fnn_cnt in range(len(self.fnn_list) - 1):
            x = self.fnn_list[fnn_cnt](x)
        return self.fnn_list[-1](x)

"""This file is a modified copy of https://github.com/partha2409/DCASE2024_seld_baseline/blob/main/seldnet_model.py"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.audio_processor import AudioProcessor


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


class Encoder(nn.Module):
    def __init__(self, params: dict):
        super().__init__()
        self.params = params

        self.conv_block_list = nn.ModuleList()
        if len(params["f_pool_size"]):
            for conv_cnt in range(len(params["f_pool_size"])):
                self.conv_block_list.append(
                    ConvBlock(
                        in_channels=params["nb_cnn2d_filt"] if conv_cnt else params["nb_channels"],
                        out_channels=params["nb_cnn2d_filt"],
                    )
                )
                self.conv_block_list.append(
                    nn.MaxPool2d((params["f_pool_size"][conv_cnt], params["f_pool_size"][conv_cnt]))
                )
                self.conv_block_list.append(nn.Dropout2d(p=params["dropout_rate"]))

        self.gru_input_dim = params["nb_cnn2d_filt"] * int(
            np.floor(params["nb_mel_bins"] / np.prod(params["f_pool_size"]))
        )
        self.gru = torch.nn.GRU(
            input_size=self.gru_input_dim,
            hidden_size=params["rnn_size"],
            num_layers=params["nb_rnn_layers"],
            batch_first=True,
            dropout=params["dropout_rate"],
            bidirectional=True,
        )

        self.mhsa_block_list = nn.ModuleList()
        self.layer_norm_list = nn.ModuleList()
        for _ in range(params["nb_self_attn_layers"]):
            self.mhsa_block_list.append(
                nn.MultiheadAttention(
                    embed_dim=self.params["rnn_size"],
                    num_heads=self.params["nb_heads"],
                    dropout=self.params["dropout_rate"],
                    batch_first=True,
                )
            )
            self.layer_norm_list.append(nn.LayerNorm(self.params["rnn_size"]))

        self.max_pool = nn.AdaptiveMaxPool1d(1)

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

        return x


class SeldModelAnchor(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        self.audio_processor = AudioProcessor(sample_rate=44100, window_size=0.04, window_stride=0.02, n_mels=64)
        self.nb_classes = params["nb_classes"]
        self.params = params

        self.encoder = Encoder(params)

        self.fnn_list = torch.nn.ModuleList()
        if params["nb_fnn_layers"]:
            for fc_cnt in range(params["nb_fnn_layers"]):
                self.fnn_list.append(
                    nn.Linear(
                        params["fnn_size"] if fc_cnt else self.params["rnn_size"] * 4, params["fnn_size"], bias=True
                    )
                )

        self.fnn_list.append(
            nn.Linear(
                params["fnn_size"] if params["nb_fnn_layers"] else self.params["rnn_size"] * 4,
                self.nb_classes,
                bias=True,
            )
        )

    def forward(self, x1, x2):

        x1 = self.encoder(x1)
        x2 = self.encoder(x2)

        x = torch.cat((x1, x2, x1 - x2, x1 * x2), dim=1)

        for fc_cnt in range(len(self.fnn_list)):
            x = self.fnn_list[fc_cnt](x)
            if fc_cnt < len(self.fnn_list) - 1:
                x = F.relu(x)

        return x


# class SeldModelAnchor(torch.nn.Module):
#     def __init__(self, params):
#         super().__init__()
#         self.audio_processor = AudioProcessor(sample_rate=44100, window_size=0.04, window_stride=0.02, n_mels=64)
#         self.nb_classes = params["nb_classes"]
#         self.params = params

#         self.encoder = Encoder(params)

#         self.cross_att = torch.nn.MultiheadAttention(
#             embed_dim=params["rnn_size"],
#             num_heads=1,
#             dropout=params["dropout_rate"],
#             batch_first=True,
#         )

#         self.fnn_list = torch.nn.ModuleList()
#         if params["nb_fnn_layers"]:
#             for fc_cnt in range(params["nb_fnn_layers"]):
#                 self.fnn_list.append(
#                     nn.Linear(params["fnn_size"] if fc_cnt else self.params["rnn_size"], params["fnn_size"], bias=True)
#                 )

#         self.fnn_list.append(
#             nn.Linear(
#                 params["fnn_size"] if params["nb_fnn_layers"] else self.params["rnn_size"],
#                 self.nb_classes,
#                 bias=True,
#             )
#         )

#     def forward(self, x1, x2):

#         x1 = self.encoder(x1)
#         x2 = self.encoder(x2)

#         x1 = x1.unsqueeze(1)
#         x2 = x2.unsqueeze(1)

#         x, _ = self.cross_att(x1, x2, x2)
#         x = x.squeeze(1)

#         for fc_cnt in range(len(self.fnn_list)):
#             x = self.fnn_list[fc_cnt](x)
#             if fc_cnt < len(self.fnn_list) - 1:
#                 x = F.relu(x)

#         return x

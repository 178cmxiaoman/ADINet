import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWT1DForward


class ADI(nn.Module):
    """
    Adaptive dependency interaction block.

    This block alternates temporal interaction and cross-band interaction,
    then uses adaptive band weights and gates to control the injected
    cross-band residual.
    """

    def __init__(self, input_shape, dropout=0.2, patch=12, layernorm=True):
        super(ADI, self).__init__()
        self.input_shape = input_shape
        self.patch = patch
        self.n_history = 1
        self.layernorm = layernorm

        if self.layernorm:
            self.norm = nn.BatchNorm1d(self.input_shape[0] * self.input_shape[-1])
        self.norm1 = nn.BatchNorm1d(self.n_history * patch * self.input_shape[-1])
        self.norm2 = nn.BatchNorm1d(self.patch * self.input_shape[-1])

        self.agg = nn.Linear(self.n_history * self.patch, self.patch)
        self.dropout_t = nn.Dropout(dropout)

        self.ff_dim = 2 ** math.ceil(math.log2(self.input_shape[-1]))
        self.fc_block = nn.Sequential(
            nn.Linear(self.input_shape[-1], self.ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(self.ff_dim, self.input_shape[-1]),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.gate_fc = nn.Sequential(
            nn.Linear(self.input_shape[-1], self.input_shape[-1]),
            nn.Sigmoid(),
        )
        self.beta_fc = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(self.input_shape[1], self.input_shape[1], kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        """
        Args:
            x: Tensor with shape [batch, band_count, sequence_length].

        Returns:
            Tensor with shape [batch, band_count, sequence_length].
        """
        if self.layernorm:
            x = self.norm(torch.flatten(x, 1, -1)).reshape(x.shape)

        output = torch.zeros_like(x)
        output[:, :, : self.n_history * self.patch] = x[
            :, :, : self.n_history * self.patch
        ].clone()

        for i in range(self.n_history * self.patch, self.input_shape[0], self.patch):
            hist_input = output[:, :, i - self.n_history * self.patch : i]
            hist_input = self.norm1(torch.flatten(hist_input, 1, -1)).reshape(
                hist_input.shape
            )

            temp_out = F.gelu(self.agg(hist_input))
            temp_out = self.dropout_t(temp_out)

            tmp = temp_out + x[:, :, i : i + self.patch]
            residual = tmp.clone()

            tmp = self.norm2(torch.flatten(tmp, 1, -1)).reshape(tmp.shape)
            tmp = torch.transpose(tmp, 1, 2)
            tmp_fc = self.fc_block(tmp)
            tmp_fc = torch.transpose(tmp_fc, 1, 2)

            beta = self.beta_fc(tmp_fc)
            gate = self.gate_fc(tmp_fc.mean(-1)).unsqueeze(-1)
            tmp_modulated = gate * (beta * tmp_fc)

            output[:, :, i : i + self.patch] = residual + tmp_modulated

        return output


class LowRankLinear(nn.Module):
    def __init__(self, in_features, out_features, rank):
        super(LowRankLinear, self).__init__()
        self.rank = rank
        self.linear1 = nn.Linear(in_features, rank, bias=False)
        self.linear2 = nn.Linear(rank, out_features, bias=True)

    def forward(self, x):
        return self.linear2(self.linear1(x))


class ADINet(nn.Module):
    def __init__(self, seq_len, logger):
        super(ADINet, self).__init__()
        self.seq_len = seq_len
        self.pred_len = seq_len
        self.dropout = 0.05
        logger.info("dropout: %s", self.dropout)

        self.decompose_layer = 3
        logger.info("decompose_layer: %s", self.decompose_layer)

        wave = "haar"
        logger.info("wave: %s", wave)

        mode = "symmetric"
        logger.info("mode: %s", mode)

        self.dwt = DWT1DForward(wave=wave, J=self.decompose_layer, mode=mode)

        if seq_len == 1000:
            self.linears = nn.ModuleList(
                [
                    nn.Sequential(
                        LowRankLinear(125, 256, rank=64),
                        nn.GELU(),
                        nn.Dropout(self.dropout),
                        LowRankLinear(256, 500, rank=128),
                    ),
                    nn.Sequential(
                        LowRankLinear(125, 256, rank=64),
                        nn.GELU(),
                        nn.Dropout(self.dropout),
                        LowRankLinear(256, 500, rank=128),
                    ),
                    nn.Sequential(
                        LowRankLinear(250, 512, rank=125),
                        nn.GELU(),
                        nn.Dropout(self.dropout),
                        LowRankLinear(512, 500, rank=256),
                    ),
                ]
            )
            self.fc_blocks = nn.ModuleList(
                [ADI((500, 4), dropout=self.dropout, patch=20) for _ in range(3)]
            )
            self.fc = nn.Sequential(
                nn.Linear(2000, 125),
                nn.GELU(),
                nn.Linear(125, 2),
            )
        elif seq_len == 1024:
            self.linears = nn.ModuleList(
                [
                    nn.Sequential(
                        LowRankLinear(128, 256, rank=64),
                        nn.GELU(),
                        nn.Dropout(self.dropout),
                        LowRankLinear(256, 512, rank=128),
                    ),
                    nn.Sequential(
                        LowRankLinear(128, 256, rank=64),
                        nn.GELU(),
                        nn.Dropout(self.dropout),
                        LowRankLinear(256, 512, rank=128),
                    ),
                    nn.Sequential(
                        LowRankLinear(256, 512, rank=128),
                        nn.GELU(),
                        nn.Dropout(self.dropout),
                        LowRankLinear(512, 512, rank=256),
                    ),
                ]
            )
            self.fc_blocks = nn.ModuleList(
                [ADI((512, 4), dropout=self.dropout, patch=32) for _ in range(3)]
            )
            self.fc = nn.Sequential(
                nn.Linear(2048, 128),
                nn.GELU(),
                nn.Linear(128, 2),
            )
        else:
            raise ValueError("ADINet supports seq_len 1000 or 1024.")

    def forward(self, input, batch_y=None):
        in_dwt = input.unsqueeze(1).float()

        yl, yhs = self.dwt(in_dwt)
        coefs = [yl] + yhs
        coefs_new = [coefs[0], coefs[3], coefs[2], coefs[1]]

        scaled_coefs = []
        for i in range(self.decompose_layer):
            scaled_coefs.append(self.linears[i](coefs_new[i]))
        scaled_coefs.append(coefs_new[self.decompose_layer])

        x = torch.cat(scaled_coefs, dim=1)
        for fc_block in self.fc_blocks:
            x = fc_block(x)

        if self.seq_len == 1000:
            x = x.reshape(-1, 1, 2000)
        else:
            x = x.reshape(-1, 1, 2048)

        x = self.fc(x)
        return x.squeeze(1)

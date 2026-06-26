import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = None
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), nn.BatchNorm1d(out_channels))

    def forward(self, x):
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class CNNTransformerParallelNetwork(nn.Module):
    """Parallel CNN-Transformer network for arc fault detection."""

    def __init__(self, input_length, num_classes=2, cnn_channels=(64, 256), d_model=192, n_head=4, n_layers=4, dim_feedforward=384, dropout=0.1):
        super().__init__()
        self.input_length = input_length
        self.num_classes = num_classes

        # CNN path: low-frequency current reshaped as image-like tensor (B, 3, L/3)
        self.cnn_path = nn.Sequential(
            nn.Conv1d(3, cnn_channels[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(cnn_channels[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
            ResidualBlock1D(cnn_channels[0], cnn_channels[0]),
            ResidualBlock1D(cnn_channels[0], cnn_channels[1], stride=2),
            nn.AdaptiveAvgPool1d(1),
        )

        self.cnn_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(cnn_channels[1], 256),
            nn.ReLU(inplace=True),
        )

        # Transformer path: time series sequence
        self.input_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=max(1024, input_length + 1))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="relu",
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.transformer_pool = nn.AdaptiveAvgPool1d(1)
        self.transformer_fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(d_model, 256),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        if x.dim() != 2:
            raise ValueError(f"Expected input shape (batch, length), got {tuple(x.shape)}")

        b, l = x.shape
        if l != self.input_length:
            raise ValueError(f"Expected input length {self.input_length}, got {l}")

        # CNN branch: split into 3 channels; if not divisible, pad to nearest multiple of 3
        if l % 3 != 0:
            pad_len = 3 - (l % 3)
            x_cnn = torch.nn.functional.pad(x, (0, pad_len))
        else:
            x_cnn = x
        x_cnn = x_cnn.view(b, 3, -1)
        x_cnn = self.cnn_path(x_cnn)
        x_cnn = self.cnn_fc(x_cnn)

        # Transformer branch
        x_tr = x.unsqueeze(-1).float()
        x_tr = self.input_proj(x_tr)
        x_tr = self.pos_encoder(x_tr)
        x_tr = self.transformer_encoder(x_tr)
        x_tr = x_tr.permute(0, 2, 1)
        x_tr = self.transformer_pool(x_tr)
        x_tr = self.transformer_fc(x_tr)

        x_out = torch.cat([x_cnn, x_tr], dim=1)
        return self.classifier(x_out)

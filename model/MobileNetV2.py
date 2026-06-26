import torch
import torch.nn as nn


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super().__init__(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU6(inplace=True),
        )


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        if stride not in (1, 2):
            raise ValueError(f"stride must be 1 or 2, got {stride}")

        hidden_dim = int(round(in_channels * expand_ratio))
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1))

        layers.append(ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim))
        layers.append(
            nn.Sequential(
                nn.Conv1d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        )
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2(nn.Module):
    """1D MobileNetV2 for current-signal classification."""

    def __init__(self, input_length, num_classes=2, width_mult=1.0):
        super().__init__()
        self.input_length = input_length
        self.num_classes = num_classes

        def _c(ch):
            return max(8, int(ch * width_mult))

        input_channel = _c(32)
        last_channel = _c(1280)

        self.features = [ConvBNReLU(1, input_channel, stride=2)]

        inverted_residual_setting = [
            # t, c, n, s
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        for expand_ratio, channels, repeats, stride in inverted_residual_setting:
            output_channel = _c(channels)
            for i in range(repeats):
                self.features.append(
                    InvertedResidual(
                        input_channel,
                        output_channel,
                        stride=stride if i == 0 else 1,
                        expand_ratio=expand_ratio,
                    )
                )
                input_channel = output_channel

        self.features.append(ConvBNReLU(input_channel, last_channel, kernel_size=1))
        self.features = nn.Sequential(*self.features)

        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.2),
            nn.Linear(last_channel, num_classes),
        )

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = self.pool(x).squeeze(-1)
        x = self.classifier(x)
        return x

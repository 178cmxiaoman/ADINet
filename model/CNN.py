import torch
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, input_length):
        super(CNN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=30, kernel_size=10, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=30, out_channels=30, kernel_size=8, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=30, out_channels=40, kernel_size=6, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=40, out_channels=50, kernel_size=5, stride=1, padding="same"),
            nn.ReLU(),
            nn.Conv1d(in_channels=50, out_channels=50, kernel_size=5, stride=1, padding="same"),
            nn.ReLU(),
        )

        # 计算 Flatten 后的维度
        self.flatten_dim = input_length * 50  # 由于 padding="same"，长度保持不变

        self.Dense = nn.Sequential(
            nn.Linear(self.flatten_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 2),
            # nn.Softmax(dim=1),
        )

    def forward(self, input):
        b, l = input.shape

        input = input.unsqueeze(1)  # unsqueeze(1) 表示在第1维度（通道维度）增加一个维度

        # 传入卷积层
        output = self.conv(input)

        # 展开
        output = output.view(b, -1)
        output = output[:, : self.flatten_dim]

        # 全连接层
        output = self.Dense(output)
        output = output.reshape(b, 2)
        return output

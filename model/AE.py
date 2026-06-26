import torch
import torch.nn as nn


class AE(nn.Module):
    def __init__(self, input_length):
        super(AE, self).__init__()
        self.input_length = input_length

        # Encoder
        self.conv1 = nn.Conv1d(1, 8, kernel_size=4, stride=1, padding=0)  # Layer2
        self.fc1 = nn.Linear((input_length - 3) * 8, (input_length - 3) * 8)  # Layer4
        self.fc2 = nn.Linear((input_length - 3) * 8, 128)  # Layer5
        self.fc3 = nn.Linear(128, (input_length - 3) * 8)  # Layer6

        # Decoder
        self.pad = nn.ConstantPad1d((3, 2), 0)  # 特殊填充处理  # Layer7
        self.conv2 = nn.Conv1d(8, 1, kernel_size=4, stride=1, padding=0)  # Layer8

        # 激活函数
        self.relu = nn.ReLU()

    def forward(self, x):
        # Encoder
        x = x.unsqueeze(1)  # [B, 1, L]
        x = self.conv1(x)  # [B, 8, L-3]
        x = x.flatten(1)  # [B, 8*(L-3)]
        x = self.relu(self.fc1(x))  # Layer4
        x = self.relu(self.fc2(x))  # Layer5
        x = self.relu(self.fc3(x))  # Layer6

        # Decoder
        x = x.view(-1, 8, (self.input_length - 3))  # Reshape
        x = self.pad(x)  # 填充到L
        x = self.conv2(x)  # [B, 1, L]
        x = x.flatten(1)  # Layer9

        return x

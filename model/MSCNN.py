import torch
import torch.nn as nn


class MultiScaleCNN(nn.Module):
    def __init__(self, input_length):
        super(MultiScaleCNN, self).__init__()

        # 多尺度卷积层
        # 注意：PyTorch的Conv1d不支持padding="same"，需要手动计算padding
        # padding = (kernel_size - 1) // 2 可以实现same padding效果
        self.conv1 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1), nn.BatchNorm1d(16), nn.ReLU())  # 添加Batch Normalization
        self.conv2 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2), nn.BatchNorm1d(16), nn.ReLU())  # 添加Batch Normalization
        self.conv3 = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, stride=1, padding=3), nn.BatchNorm1d(16), nn.ReLU())  # 添加Batch Normalization

        # 合并多尺度特征
        self.conv_out_channels = 16 * 3  # 每个卷积层输出 16 通道，总共 3 个卷积层

        # 全连接层
        self.flatten_dim = input_length * self.conv_out_channels
        self.fc = nn.Sequential(nn.Linear(self.flatten_dim, 1024), nn.ReLU(), nn.Dropout(0.5), nn.Linear(1024, 2))  # 添加Dropout防止过拟合

    def forward(self, x):
        b = x.shape[0]

        # 增加通道维度
        x = x.unsqueeze(1)  # (batch_size, 1, input_length)

        # 多尺度卷积
        out1 = self.conv1(x)  # (batch_size, 16, input_length)
        out2 = self.conv2(x)  # (batch_size, 16, input_length)
        out3 = self.conv3(x)  # (batch_size, 16, input_length)

        # 拼接多尺度特征
        out = torch.cat([out1, out2, out3], dim=1)  # (batch_size, 16*3, input_length)

        # 展平
        out = out.view(b, -1)

        # 全连接层
        out = self.fc(out)
        return out

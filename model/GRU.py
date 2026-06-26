import torch
import torch.nn as nn


class GRUModel(nn.Module):
    def __init__(self, input_length, hidden_size=128, num_layers=2, output_size=2):
        """
        GRU 模型
        :param input_length: 输入序列的长度
        :param hidden_size: GRU 隐藏层的维度
        :param num_layers: GRU 的层数
        :param output_size: 输出类别数
        """
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # GRU 层
        self.gru = nn.GRU(input_size=1, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)  # 输入特征维度  # 隐藏层维度  # GRU 层数  # 输入的维度为 (batch, seq_len, input_size)

        # 全连接层
        self.fc = nn.Sequential(nn.Linear(hidden_size * input_length, 1024), nn.ReLU(), nn.Linear(1024, output_size))  # 展平后传入全连接层

    def forward(self, input):
        b, l = input.shape

        # 增加通道维度，变为 (batch, seq_len, input_size)
        input = input.unsqueeze(2)  # unsqueeze(2) 表示在第2维度增加一个维度

        # 传入 GRU 层
        gru_out, _ = self.gru(input)  # gru_out: (batch, seq_len, hidden_size)

        # 展平 GRU 输出
        gru_out = gru_out.reshape(b, -1)  # (batch, seq_len * hidden_size)

        # 全连接层
        output = self.fc(gru_out)  # (batch, output_size)
        return output

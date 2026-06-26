import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class TransformerModel(nn.Module):
    def __init__(self, input_length, d_model=256, n_head=8, dim_feedforward=1024, n_layers=4, dropout=0.1):
        super(TransformerModel, self).__init__()

        self.d_model = d_model
        self.input_proj = nn.Linear(1, d_model)  # 将标量输入投影到d_model维度
        self.pos_encoder = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_head, dim_feedforward=dim_feedforward, dropout=dropout, activation="relu")
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.flatten_dim = input_length * d_model
        self.classifier = nn.Sequential(nn.Linear(self.flatten_dim, 1024), nn.ReLU(), nn.Linear(1024, 2))

    def forward(self, input):
        # input shape: (batch_size, input_length)
        b, l = input.shape

        # 添加通道维度并投影到d_model
        x = input.unsqueeze(-1).float()  # shape: (b, l, 1)
        x = self.input_proj(x)  # shape: (b, l, d_model)

        # 添加位置编码并调整维度顺序
        x = self.pos_encoder(x)  # shape: (b, l, d_model)
        x = x.permute(1, 0, 2)  # 转为 (seq_len, batch, features)

        # Transformer编码器
        x = self.transformer_encoder(x)  # 输出形状: (seq_len, batch, d_model)

        # 恢复维度顺序并展平
        x = x.permute(1, 0, 2)  # 转为 (batch, seq_len, d_model)
        x = x.reshape(b, -1)  # 展平为 (batch, seq_len*d_model)

        # 分类层
        output = self.classifier(x)
        return output

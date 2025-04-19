import math

import torch
from torch import nn


class CircularPositionalEncoding(nn.Module):
    """修复后的环形轨道位置编码"""

    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        # 修正时间位置编码维度
        time_pe = torch.zeros(max_len, d_model)
        time_pe[:, 0::2] = torch.sin(position * div_term)
        time_pe[:, 1::2] = torch.cos(position * div_term)

        # 修正轨道位置编码维度
        track_pos = torch.arange(8)
        track_div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        track_angle = position[:8] * track_div_term  # 使用与时间编码相同的div_term

        track_pe = torch.zeros(8, d_model)
        track_pe[:, 0::2] = torch.sin(track_angle)
        track_pe[:, 1::2] = torch.cos(track_angle)

        self.register_buffer('time_pe', time_pe)
        self.register_buffer('track_pe', track_pe)

    def forward(self, x: torch.Tensor, track_ids: torch.Tensor):
        """
        修正后的维度处理:
        x: (seq_len, batch_size, d_model)
        track_ids: (seq_len, batch_size)
        """
        time_encoding = self.time_pe[:x.size(0)]  # (seq_len, d_model)
        track_encoding = self.track_pe[track_ids]  # (seq_len, batch_size, d_model)

        # 确保维度匹配
        return x + time_encoding.unsqueeze(1) + track_encoding

class NoteEncoder(nn.Module):
    """音符特征编码器"""

    def __init__(self, input_dim=10, d_model=128):
        super().__init__()
        # 修改为处理二维特征（合并音符维度）
        self.feature_embed = nn.Linear(input_dim, d_model)
        self.type_embed = nn.Embedding(6, d_model)

    def forward(self, features, note_types):
        """
        修改后的维度处理:
        features: (seq_len, batch_size, input_dim)
        note_types: (seq_len, batch_size)
        返回: (seq_len, batch_size, d_model)
        """
        # 合并特征和类型嵌入
        embedded = self.feature_embed(features) + self.type_embed(note_types)
        return embedded
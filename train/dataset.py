import json

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset

from config import config


class ChartDataset(Dataset):
    """谱面数据集处理"""

    def __init__(self, json_files, labels, max_seq_len=config.max_len, scaler=None):
        self.max_seq_len = max_seq_len
        self.features = []
        self.labels = []
        self.stat_features = []

        # 修改为可传入scaler
        self.scaler = scaler if scaler else StandardScaler()
        self.fit_scaler = scaler is None  # 标记是否需要拟合

        for file, label in zip(json_files, labels):
            with open(file, 'r') as f:
                chart_data = json.load(f)

            seq_features = self.process_sequence(chart_data)
            self.features.append(seq_features)
            self.labels.append(label)

        # 修改标准化处理逻辑
        all_features = np.stack(self.features)
        original_shape = all_features.shape
        if self.fit_scaler:
            self.features = self.scaler.fit_transform(
                all_features.reshape(-1, original_shape[-1])
            ).reshape(original_shape)
        else:
            self.features = self.scaler.transform(
                all_features.reshape(-1, original_shape[-1])
            ).reshape(original_shape)

    def process_sequence(self, chart_data):
        """将原始JSON转换为时序特征矩阵"""
        time_points = sorted(chart_data, key=lambda x: x["Time"])
        seq = []

        for tp in time_points[:self.max_seq_len]:
            # 每个时间点提取特征
            feature = np.zeros(10)  # 示例特征维度

            # 轨道存在性 (0-7)
            tracks = [n["startPosition"] - 1 for n in tp["Notes"]]
            feature[:8] = np.bincount(tracks, minlength=8)

            # 特殊音符统计
            feature[8] = sum(1 for n in tp["Notes"] if n["isBreak"])
            feature[9] = sum(1 for n in tp["Notes"] if n["noteType"] == "Slide")

            seq.append(feature)

        # 填充或截断序列
        if len(seq) < self.max_seq_len:
            seq += [np.zeros(10)] * (self.max_seq_len - len(seq))
        else:
            seq = seq[:self.max_seq_len]

        if len(seq) != self.max_seq_len:
            if len(seq) < self.max_seq_len:
                seq += [np.zeros(10)] * (self.max_seq_len - len(seq))
            else:
                seq = seq[:self.max_seq_len]

        return np.array(seq)  # 保持形状为 (max_seq_len, 10)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.features[idx]),  # 谱面特征
            torch.FloatTensor(self.stat_features[idx]),  # 统计特征
            torch.FloatTensor([self.labels[idx]])  # 标签
        )
import torch
import torch.nn as nn
import math
import json
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tqdm import tqdm
import time
import os
from datetime import datetime
import joblib

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
        track_encoding = self.track_pe[track_ids] # (seq_len, batch_size, d_model)
        
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

class NoteTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=6):
        super().__init__()
        self.note_encoder = NoteEncoder(d_model=d_model)
        
        # Transformer编码层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 时空注意力池化
        self.attention_pool = nn.MultiheadAttention(d_model, nhead)
        self.output_layer = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        
        self.pos_encoder = CircularPositionalEncoding(d_model)
        
    def forward(self, features, note_types, track_ids):
        # 编码音符特征 (seq_len, batch_size, d_model)
        note_embeddings = self.note_encoder(features, note_types)
        
        # 直接使用编码后的特征
        x = self.pos_encoder(note_embeddings, track_ids)  # track_ids维度应为(seq_len, batch_size)
        
        # Transformer处理
        memory = self.transformer(x)
        
        # 时空注意力池化
        query = torch.mean(memory, dim=0, keepdim=True)
        context, _ = self.attention_pool(query, memory, memory)
        
        # 预测难度等级
        return self.output_layer(context.squeeze(0))

class ChartDataset(Dataset):
    """谱面数据集处理"""
    def __init__(self, json_files, labels, max_seq_len=1000, scaler=None):
        self.max_seq_len = max_seq_len
        self.features = []
        self.labels = []
        
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
            tracks = [n["startPosition"]-1 for n in tp["Notes"]]
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
            
        return np.array(seq)  # 保持形状为 (max_seq_len, 10)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.features[idx]),
            torch.FloatTensor([self.labels[idx]])
        )

# 训练配置
class Config:
    batch_size = 32
    d_model = 128
    nhead = 8
    num_layers = 6
    lr = 1e-4
    epochs = 50

def train():
    # 创建保存目录
    save_dir = f"checkpoints/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)
    
    # 加载数据
    chart_info = pd.read_csv("./info/chart_info.csv")
    train_files = chart_info["FilePath"].tolist()
    train_labels = [float(l.replace("+", ".5").replace("?", "")) for l in chart_info["Level"]]
    
    dataset = ChartDataset(train_files, train_labels)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=True)
    
    # 初始化模型
    model = NoteTransformer(d_model=Config.d_model, 
                          nhead=Config.nhead,
                          num_layers=Config.num_layers)
    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.lr)
    
    # 训练循环
    best_loss = float('inf')
    start_time = time.time()
    
    for epoch in range(Config.epochs):
        epoch_start = time.time()
        total_loss = 0
        model.train()
        
        # 使用进度条
        progress = tqdm(enumerate(dataloader), total=len(dataloader), 
                       desc=f'Epoch {epoch+1}/{Config.epochs}', 
                       bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
        
        for batch_idx, (features, labels) in progress:
            # 调整维度处理 (batch_size, seq_len, features)
            track_ids = torch.argmax(features[:, :, :8], dim=-1)  # (batch, seq)
            note_types = torch.argmax(features[:, :, 8:], dim=-1)
            
            optimizer.zero_grad()
            outputs = model(features.transpose(0, 1), 
                           note_types.transpose(0, 1),
                           track_ids.transpose(0, 1))
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            # 更新进度条信息
            progress.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        # 计算epoch统计信息
        avg_loss = total_loss / len(dataloader)
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time
        
        # 彩色打印训练信息
        print(f"\n\033[92mEpoch {epoch+1:03d}\033[0m | "
              f"Loss: \033[93m{avg_loss:.4f}\033[0m | "
              f"Time: {epoch_time:.1f}s | "
              f"Total: {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m")
        
        # 修改检查点保存部分
        if (epoch+1) % 5 == 0 or epoch == Config.epochs-1:
            ckpt_path = f"{save_dir}/epoch_{epoch+1:03d}_loss{avg_loss:.4f}.pth"
            torch.save({
                'epoch': epoch+1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")
        
        # 修改最佳模型保存部分
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = f"{save_dir}/best_epoch{epoch+1:03d}_loss{best_loss:.4f}.pth"
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved to {best_path}")
        
        # 修改scaler保存路径
        scaler_path = f"{save_dir}/scaler_loss{best_loss:.4f}.pkl"
        joblib.dump(dataset.scaler, scaler_path)
    
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")

if __name__ == "__main__":
    train()
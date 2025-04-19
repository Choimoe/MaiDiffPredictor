import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
import time
import os
from datetime import datetime
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from augmenter import EnhancedChartDataset, ChartAugmenter
from config import config
from transformer import LatentDifficultyTransformer

log_path = './log'

class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temp = temperature
        self.reg_loss = nn.L1Loss()

    def factor_contrast(self, z1, z2):
        z1 = F.normalize(z1, p=2, dim=-1)
        z2 = F.normalize(z2, p=2, dim=-1)
        logits = torch.mm(z1, z2.t()) / self.temp
        labels = torch.arange(z1.size(0)).to(z1.device)
        return F.cross_entropy(logits, labels)

    def forward(self, pred, targets, z, augmented_z=None):
        main_loss = self.reg_loss(pred[0], targets)

        kl_loss = -0.5 * torch.sum(1 + pred[2][1] - pred[2][0].pow(2) - pred[2][1].exp())

        if augmented_z is not None:
            contrast_loss = self.factor_contrast(z, augmented_z)
            total_loss = main_loss + 0.2 * kl_loss + 0.3 * contrast_loss
        else:
            total_loss = main_loss + 0.2 * kl_loss

        return total_loss, (main_loss.item(), kl_loss.item())

class StatFusion(nn.Module):
    """动态融合玩家表现统计信息"""

    def __init__(self, stat_dim=4, hidden_dim=64):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(stat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, seq_features, stats):
        stat_feat = self.fc(stats)

        gate = self.gate(stat_feat)

        enhanced = seq_features * (1 + gate.unsqueeze(-1))
        return enhanced

def contrastive_loss(z, aug_z, temperature=0.1):
    """
    基于余弦相似度的对比损失
    """
    z = F.normalize(z, p=2, dim=-1)
    aug_z = F.normalize(aug_z, p=2, dim=-1)

    logits = torch.mm(z, aug_z.t()) / temperature  # [batch, batch]

    labels = torch.arange(z.size(0)).to(z.device)

    return F.cross_entropy(logits, labels)


def train():
    dev_list = 'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(dev_list)
    print(f"\033[1;36mUsing device: {device}\033[0m")

    save_dir = f"checkpoints/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(save_dir, exist_ok=True)

    chart_info = pd.read_csv("./info/chart_info.csv")
    train_files = chart_info["FilePath"].tolist()
    train_labels = [float(l.replace("+", ".5").replace("?", "")) for l in chart_info["Level"]]

    dataset = EnhancedChartDataset(
        train_files,
        train_labels,
        chart_stat_path="./info/chart_stat_diff.json",
        chart_info_path="./info/chart_info.csv"
    )

    model = LatentDifficultyTransformer(d_model=config.d_model,
                          nhead=config.nhead,
                          num_layers=config.num_layers)

    if torch.cuda.device_count() > 1:
        print(f"\033[1;32mUsing {torch.cuda.device_count()} GPUs!\033[0m")
        model = nn.DataParallel(model)
    model = model.to(device)

    dataloader = DataLoader(dataset,
                          batch_size=config.batch_size,
                          shuffle=True,
                          pin_memory=True if device.type == 'cuda' else False,
                          num_workers=0)

    if device.type == 'cuda':
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = None

    criterion = nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_loss = float('inf')
    start_time = time.time()

    augmenter = ChartAugmenter()

    def generate_augmented_features(features, augmenter):
        return torch.stack([
            torch.FloatTensor(augmenter.difficulty_preserved_augment(
                f.cpu().numpy(),
                max_seq_len=dataset.max_seq_len
            )) for f in features
        ]).to(device)
    
    for epoch in range(config.epochs):
        epoch_start = time.time()
        total_loss = 0
        epoch_total_loss = 0
        model.train()

        progress = tqdm(enumerate(dataloader), total=len(dataloader), 
                       desc=f'Epoch {epoch+1}/{config.epochs}',
                       bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

        for batch_idx, (features, stats, labels) in progress:
            features = features.to(device, non_blocking=True)
            stats = stats.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            track_ids = torch.argmax(features[..., :8], dim=-1)
            note_types = torch.argmax(features[..., 8:], dim=-1)

            inputs = (
                features.transpose(0, 1),  # [seq, batch, 10]
                note_types.transpose(0, 1),  # [seq, batch]
                track_ids.transpose(0, 1),  # [seq, batch]
                stats  # [batch, 4]
            )

            optimizer.zero_grad()
            with torch.amp.autocast(device_type='cuda', enabled=(device.type == 'cuda')):
                pred_level, _, _, z = model(*inputs)

                main_loss = F.l1_loss(pred_level.squeeze(), labels.squeeze())

                if config.use_augmentation:
                    aug_features = generate_augmented_features(features, augmenter)
                    _, _, _, aug_z = model(
                        aug_features.transpose(0, 1),
                        note_types.transpose(0, 1),
                        track_ids.transpose(0, 1),
                        stats
                    )
                    contrast_loss = contrastive_loss(z, aug_z)
                    total_loss = main_loss + 0.3 * contrast_loss
                else:
                    total_loss = main_loss

            scaler.scale(total_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_total_loss += total_loss.item()
            progress.set_postfix({
                'loss': f'{total_loss.item():.4f}',
                'avg_loss': f'{epoch_total_loss / (batch_idx + 1):.4f}'
            })

        avg_loss = epoch_total_loss / len(dataloader)
        epoch_time = time.time() - epoch_start
        total_time = time.time() - start_time

        print(f"\n\033[92mEpoch {epoch+1:03d}\033[0m | "
              f"Loss: \033[93m{avg_loss:.4f}\033[0m | "
              f"Time: {epoch_time:.1f}s | "
              f"Total: {total_time//3600:.0f}h {(total_time%3600)//60:.0f}m")

        if (epoch+1) % 5 == 0 or epoch == config.epochs-1:
            ckpt_path = f"{save_dir}/epoch_{epoch+1:03d}_loss{avg_loss:.4f}.pth"
            torch.save({
                'epoch': epoch+1,
                'model_state': model.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'loss': avg_loss,
            }, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = f"{save_dir}/best_epoch{epoch+1:03d}_loss{best_loss:.4f}.pth"
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved to {best_path}")

        scaler_path = f"{save_dir}/scaler_loss{best_loss:.4f}.pkl"
        joblib.dump(dataset.scaler, scaler_path)
        interpret_dimensions(model, dataset, epoch)
    
    print(f"\nTraining complete! Best loss: {best_loss:.4f}")


def interpret_dimensions(model, dataset, epoch, n_samples=1000):
    """潜在维度解释分析"""
    device = next(model.parameters()).device  # 获取模型所在设备

    # 收集潜在因子和统计特征
    all_z = []
    all_stats = []
    for i in range(n_samples):
        data = dataset[i]
        features, stats, _ = data
        features = features.unsqueeze(0).to(device)  # [1, seq_len, 10] -> GPU
        stats = stats.unsqueeze(0).to(device)  # [1, 4] -> GPU

        # 提取轨道ID和音符类型（直接在GPU上操作）
        track_ids = torch.argmax(features[..., :8], dim=-1)  # [1, seq_len]
        note_types = torch.argmax(features[..., 8:], dim=-1)  # [1, seq_len]

        # 调整维度顺序为模型要求的[seq_len, batch=1, ...]
        inputs = (
            features.transpose(0, 1),  # [seq_len, 1, 10]
            note_types.transpose(0, 1),  # [seq_len, 1]
            track_ids.transpose(0, 1),  # [seq_len, 1]
            stats  # [1, 4]
        )

        with torch.no_grad():
            _, _, _, z = model(*inputs)
        all_z.append(z.cpu().numpy())
        all_stats.append(stats.cpu().numpy())

    # 因子分析
    from sklearn.decomposition import FactorAnalysis
    fa = FactorAnalysis(n_components=6).fit(np.concatenate(all_z))

    # 可视化因子载荷
    plt.figure(figsize=(12, 6))
    sns.heatmap(fa.components_,
                annot=True,
                cmap='coolwarm',
                yticklabels=[f'Factor{i + 1}' for i in range(6)],
                xticklabels=['fit_diff', 'std_dev', 'log_cnt', 'fc_ratio'])  # 需要定义统计特征名称
    plt.title('Factor Loadings Matrix')
    plt.tight_layout()
    os.makedirs(log_path, exist_ok=True)
    plt.savefig(f'{log_path}/factor_analysis_{epoch+1:03d}.png')
    plt.close()  # 避免内存泄漏

if __name__ == "__main__":
    train()
import torch
from torch import nn

from encoder import CircularPositionalEncoding, NoteEncoder
from config import config

class MultiHeadOutput(nn.Module):
    """多维度难度预测头"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, 64),
                nn.GELU(),
                nn.Linear(64, 1)
            ) for _ in range(num_heads)
        ])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

class NoteTransformer(nn.Module):
    def __init__(self, d_model=128, nhead=8, num_layers=6):
        super().__init__()
        self.note_encoder = NoteEncoder(d_model=d_model)

        # Transformer编码层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=512,
            batch_first=False
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

class EnhancedNoteTransformer(NoteTransformer):
    def __init__(self, d_model=128, nhead=8, num_layers=6):
        super().__init__(d_model, nhead, num_layers)

        # 统计特征融合层
        self.stat_fc = nn.Linear(4, d_model)

        # 多任务输出头
        self.dim_head = MultiHeadOutput(d_model, config.diff_dim)
        self.level_head = nn.Linear(1, 1)

        # 维度相关性学习
        self.dim_weight = nn.Parameter(torch.ones(config.diff_dim, 1))

    def forward(self, features, note_types, track_ids, stat_features):
        note_embeddings = self.note_encoder(features, note_types)
        x = self.pos_encoder(note_embeddings, track_ids)

        # 融合统计特征
        stat_emb = self.stat_fc(stat_features)
        stat_emb = stat_emb.unsqueeze(0)
        memory = self.transformer(x + stat_emb)

        # 多维度预测
        dim_scores = self.dim_head(memory.mean(dim=0))

        # 调整矩阵乘法
        weighted = torch.matmul(dim_scores, self.dim_weight)

        # 最终预测
        level_pred = self.level_head(weighted)
        return level_pred, dim_scores

class LatentDifficultyTransformer(EnhancedNoteTransformer):
    def __init__(self, d_model=128, nhead=8, num_layers=6, latent_dim=6):
        super().__init__()
        # 共享编码层
        self.note_encoder = NoteEncoder(d_model=d_model)
        self.pos_encoder = CircularPositionalEncoding(d_model)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=512),
            num_layers=num_layers
        )

        # 潜在难度因子分解
        self.factor_proj = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Linear(256, latent_dim * 2)
        )

        # 动态权重生成
        self.weight_gen = nn.Sequential(
            nn.Linear(latent_dim + 4, 128),  # 合并统计特征
            nn.ReLU(),
            nn.Linear(128, latent_dim)
        )

        # 输出层
        self.level_head = nn.Linear(latent_dim + config.diff_dim, 1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, features, note_types, track_ids, stat_features):
        # 编码层
        note_emb = self.note_encoder(features, note_types)
        x = self.pos_encoder(note_emb, track_ids)

        # 统计特征融合（EnhancedNoteTransformer逻辑）
        stat_emb = self.stat_fc(stat_features)  # [batch, d_model]
        stat_emb = stat_emb.unsqueeze(0)  # [1, batch, d_model]

        # Transformer处理（融合统计特征）
        context = self.transformer(x + stat_emb)  # [seq_len, batch, d_model]

        # 潜在因子生成（LatentDifficultyTransformer逻辑）
        pooled = context.mean(dim=0)  # [batch, d_model]
        factor_params = self.factor_proj(pooled)  # [batch, latent_dim*2]
        mu, logvar = factor_params.chunk(2, dim=-1)  # 各[batch, latent_dim]
        z = self.reparameterize(mu, logvar)  # [batch, latent_dim]

        # 多维度预测（EnhancedNoteTransformer逻辑）
        dim_scores = self.dim_head(pooled)  # [batch, config.diff_dim]

        # 动态权重生成（LatentDifficultyTransformer逻辑）
        combined = torch.cat([z, stat_features], dim=-1)  # [batch, latent_dim+4]
        dim_weights = torch.softmax(self.weight_gen(combined), dim=-1)  # [batch, latent_dim]

        # 双路径融合
        weighted_z = z * dim_weights  # 潜在因子加权
        final_feat = torch.cat([weighted_z, dim_scores], dim=-1)  # [batch, latent_dim+config.diff_dim]

        # 最终预测
        level_pred = self.level_head(final_feat)  # [batch, 1]

        return level_pred, dim_weights, dim_scores, z

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.features[idx]),  # 谱面特征
            torch.FloatTensor(self.stat_features[idx]),  # 统计特征
            torch.FloatTensor([self.labels[idx]])  # 标签
        )
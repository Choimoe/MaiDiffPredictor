import torch
import json
import numpy as np
import joblib
from model import LatentDifficultyTransformer, ChartAugmenter  # 确保model.py在相同目录
from torch.nn import DataParallel
import argparse

from config import config

INPUT_DIM = 10
STAT_DIM = 4
MAX_SEQ_LEN = config.max_len

class Predictor:
    def __init__(self, model_path, scaler_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.model = self._load_model(model_path)
        self.scaler = joblib.load(scaler_path)

    def _load_model(self, path):
        # 模型配置必须与训练时一致
        model = LatentDifficultyTransformer(
            d_model=128,
            nhead=8,
            num_layers=6,
            latent_dim=6
        )

        state_dict = torch.load(path, map_location=self.device)
        if isinstance(model, DataParallel):
            model = DataParallel(model)
        model.load_state_dict(state_dict)
        model = model.to(self.device)
        model.eval()
        return model

    def process_json(self, json_path):
        """处理输入JSON文件，生成模型需要的特征"""
        with open(json_path, 'r') as f:
            chart_data = json.load(f)

        # 生成时序特征
        time_points = sorted(chart_data, key=lambda x: x["Time"])
        seq = []
        for tp in time_points[:MAX_SEQ_LEN]:
            feature = np.zeros(INPUT_DIM)
            # 轨道存在性 (0-7)
            tracks = [n["startPosition"] - 1 for n in tp["Notes"]]
            feature[:8] = np.bincount(tracks, minlength=8)
            # 特殊音符统计
            feature[8] = sum(1 for n in tp["Notes"] if n["isBreak"])
            feature[9] = sum(1 for n in tp["Notes"] if n["noteType"] == "Slide")
            seq.append(feature)

        # 填充/截断
        if len(seq) < MAX_SEQ_LEN:
            seq += [np.zeros(INPUT_DIM)] * (MAX_SEQ_LEN - len(seq))
        else:
            seq = seq[:MAX_SEQ_LEN]

        # 标准化处理
        seq = np.array(seq)
        original_shape = seq.shape
        seq = self.scaler.transform(
            seq.reshape(-1, original_shape[-1])
        ).reshape(original_shape)

        # 生成统计特征（示例值，实际需根据业务逻辑获取）
        stat_features = np.array([14.5, 0.8, 6.2, 0.65])  # [fit_diff, std_dev, log_cnt, fc_ratio]

        return torch.FloatTensor(seq), torch.FloatTensor(stat_features)

    def predict(self, json_path):
        # 数据处理
        features, stats = self.process_json(json_path)
        features = features.unsqueeze(0).to(self.device)  # 增加batch维度
        stats = stats.unsqueeze(0).to(self.device)

        # 提取轨道和音符类型
        track_ids = torch.argmax(features[..., :8], dim=-1)  # [1, seq_len]
        note_types = torch.argmax(features[..., 8:], dim=-1)  # [1, seq_len]

        # 调整维度顺序
        inputs = (
            features.transpose(0, 1),  # [seq, 1, 10]
            note_types.transpose(0, 1),  # [seq, 1]
            track_ids.transpose(0, 1),  # [seq, 1]
            stats  # [1, 4]
        )

        # 预测
        with torch.no_grad():
            pred_level, _, dim_scores, _ = self.model(*inputs)

        # 解析结果
        return {
            "total_difficulty": round(pred_level.item(), 2),
            "dimension_scores": {
                "Technical": dim_scores[0][0].item(),
                "Stamina": dim_scores[0][1].item(),
                "Accuracy": dim_scores[0][2].item(),
                "Pattern": dim_scores[0][3].item(),
                "Speed": dim_scores[0][4].item(),
                "Complexity": dim_scores[0][5].item()
            }
        }


# python .\train\test.py --model checkpoints/20250420-045320/best_epoch019_loss0.4215.pth --scaler checkpoints/20250420-045320/scaler_loss0.4215.pkl --input .\serialized_data\11163_3.json
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="checkpoints/20250420/best_epoch.pth")
    parser.add_argument("--scaler", type=str, default="checkpoints/20250420/scaler_loss.pkl")
    parser.add_argument("--input", type=str, required=True, help="输入JSON文件路径")
    args = parser.parse_args()

    predictor = Predictor(args.model, args.scaler)
    result = predictor.predict(args.input)

    print("\n预测结果：")
    print(f"综合难度等级：{result['total_difficulty']}")
    print("维度分析：")
    for dim, score in result['dimension_scores'].items():
        print(f"- {dim}: {score:.2f}")
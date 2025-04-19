import torch
import json
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
import joblib
from train.model import NoteTransformer, ChartDataset
import argparse
import tempfile
import os
from serializer import run_csharp_program

class Predictor:
    def __init__(self, model_path, scaler_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.load_model(model_path).to(self.device)
        self.scaler = joblib.load(scaler_path)
        
    def load_model(self, model_path):
        model = NoteTransformer(
            d_model=128, 
            nhead=8,
            num_layers=6
        )
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.eval()
        return model
    
    def preprocess(self, json_path):
        dummy_labels = [0.0]
        dataset = ChartDataset([json_path], dummy_labels, scaler=self.scaler)
        return dataset[0][0].unsqueeze(0)
        
    def convert_maidata(self, maidata_path):
        """使用serializer工具转换maidata.txt"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # 运行C#转换程序
            csv_path = os.path.join(tmpdir, "dummy.csv")
            result = run_csharp_program(
                os.path.dirname(maidata_path),
                tmpdir,
                csv_path
            )
            if not result:
                raise RuntimeError("maidata转换失败")
            
            # 查找生成的JSON文件
            json_files = [f for f in os.listdir(tmpdir) if f.endswith(".json")]
            if not json_files:
                raise FileNotFoundError("未生成JSON文件")
            return os.path.join(tmpdir, json_files[0])

    def predict(self, input_path, is_maidata=False):
        if is_maidata:
            json_path = self.convert_maidata(input_path)
            return self._predict_json(json_path)
        else:
            return self._predict_json(input_path)
    
    def _predict_json(self, json_path):
        with torch.no_grad():
            features = self.preprocess(json_path).to(self.device)
            
            track_ids = torch.argmax(features[0, :, :8], dim=-1).unsqueeze(0)
            note_types = torch.argmax(features[0, :, 8:], dim=-1).unsqueeze(0)
            
            features = features.transpose(0, 1)
            track_ids = track_ids.transpose(0, 1)
            note_types = note_types.transpose(0, 1)
            
            output = self.model(features, note_types, track_ids)
            return output.item()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Chart Difficulty Predictor')
    parser.add_argument('--model', type=str, required=True, 
                       help='Path to trained model .pth file')
    parser.add_argument('--scaler', type=str, required=True,
                       help='Path to scaler .pkl file')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input chart JSON file')
    parser.add_argument('--maidata', action='store_true',
                       help='Path to input chart maidata.txt file')
    args = parser.parse_args()
    
    predictor = Predictor(args.model, args.scaler)
    difficulty = predictor.predict(args.input, args.maidata)
    print(f"Predicted difficulty: {difficulty:.2f}")
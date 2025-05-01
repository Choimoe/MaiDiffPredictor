import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from tools.inference.src.enhance_json_with_features import process_json_file
from tools.inference.src.inference import predict_difficulty

json_file_path = "serialized_data/11663_2.json"

def main():
    process_json_file(json_file_path, "chart.json")

    file_name, predicted_difficulty = predict_difficulty(json_file_path)
    print(f"文件名: {file_name}")
    print(f"预测的难度: {predicted_difficulty}")


if __name__ == "__main__":
    main()

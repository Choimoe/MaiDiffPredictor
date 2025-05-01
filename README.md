# MaiDiffPredictor - maimai DX Chart Difficulty Prediction System

## Project Overview

A deep learning system for predicting and analyzing chart difficulty in SEGA's rhythm game *maimai DX*. Combines Transformer architecture with latent factor modeling to achieve:

- Comprehensive difficulty prediction
- 6-dimensional skill analysis (Technical/Stamina/Accuracy/etc)
- Latent difficulty visualization
- Dynamic difficulty calibration

## Key Features

- **Multimodal Fusion**: Integrates temporal note patterns with player performance statistics
- **Contrastive Learning**: Enhances robustness through data augmentation
- **Interpretable Analysis**: Factor loading matrix reveals hidden difficulty dimensions
- **Dynamic Weighting**: Adaptive fusion of statistical and sequential features
- **Production Readiness**: Supports multi-GPU training and C# interoperability

## Current Limitations (Issue #1)

The current model exhibits excessive reliance on:
- Statistical metrics from diving-fish's dataset
- Chart length/duration
- Note density averages

This leads to suboptimal recognition of:
- Complex rhythm patterns
- Note arrangement complexity
- Stamina-demanding sequences
- Unconventional slide patterns

We welcome community contributions to improve pattern recognition capabilities.

## Environment Requirements

- Python 3.8+
- PyTorch 2.0+
- .NET 6.0 SDK
- Key Python packages: `scikit-learn`, `seaborn`, `tqdm`, `joblib`

## Installation

```bash
# Clone repository
git clone https://github.com/yourname/MaiDiffPredictor.git

# Install Python dependencies
# pip install -r requirements.txt

# Install .NET SDK (Ubuntu example)
wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb
sudo dpkg -i packages-microsoft-prod.deb
sudo apt-get update && sudo apt-get install dotnet-sdk-6.0
```

## Data Preparation

```bash
# Serialize chart data
python serializer.py

# Fetch latest chart stats
python .\preprocess\getDiff.py
python .\preprocess\fixSlideTime.py .\serialized_data\ -o .\serialized_data\
python .\preprocess\enhance_json.py .\serialized_data\ -o .\serialized_data\
python .\preprocess\genInfo.py
```

## Model Training

```bash
# Single GPU training
python .\train\ori_train.py
```

## Prediction with Pretrained Model

```bash
python .\train\test.py
```

## Project Structure

```
MaiDiffPredictor/
├── config/             # Project configuration
│   ├── api.json        # API configuration
│   └── train.json      # Training configuration
├── serialized_data/    # Processed chart data
├── info/               # Chart statistics
├── train/              # Core model code
│   ├── model.py        # Model definition & training
│   ├── dataset.py      # Data loading & features 
│   ├── test.py         # Prediction interface
│   └── ...
├── tools/              # C# serialization utilities
├── data/               # Chart data
├── serializer.py       # Serializer for chart data
└── preprocess/         # Data preprocessing scripts
```

## Algorithm Overview

The system processes chart data through three main stages:

1. **Feature Extraction**
   - Temporal note patterns (hold, slide, break notes)
   - Player performance statistics (clear rates, FC distribution)
   - Density fluctuations

2. **Hierarchical Processing**
   - Transformer encoder for temporal patterns
   - Statistical feature fusion layer
   - Latent factor decomposition

3. **Dynamic Prediction**
   - Multi-head difficulty dimensions
   - Adaptive weight generation
   - Final difficulty synthesis

## Sample Output

```text
Epoch [10/1000], Loss: 0.5384
Top 5 high error samples:
Song: 786 | Level: 2
  FitDiff: 12.6518 | DS: 12.7
  Loss: 10.0690 | Predicted: 9.50 | Actual: 12.68
Song: 157 | Level: 2
  FitDiff: 12.2823 | DS: 11.6
  Loss: 9.8733 | Predicted: 8.80 | Actual: 11.94
Song: 531 | Level: 1
  FitDiff: 9.1862 | DS: 9.0
  Loss: 8.9822 | Predicted: 12.09 | Actual: 9.09
Song: 212 | Level: 2
  FitDiff: 12.4363 | DS: 12.2
  Loss: 8.7804 | Predicted: 9.35 | Actual: 12.32
Song: 189 | Level: 3
  FitDiff: 12.3803 | DS: 12.7
  Loss: 8.6610 | Predicted: 9.60 | Actual: 12.54
模型已保存: trained_models\model_epoch_10.pth
最好的模型已保存: trained_models/best_model.pth
```

## License

...

---

**Note**: Ensure `maidata.txt` files comply with SEGA's secondary creation guidelines. This system is for research purposes only.
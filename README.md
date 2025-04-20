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
pip install -r requirements.txt

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
python preprocess/getDiff.py
```

## Model Training

```bash
# Single GPU training
python train/model.py
```

## Prediction with Pretrained Model

```bash
# Example 1: Predict from JSON file
python train/test.py \
  --model best_model.pth \
  --scaler scaler.pkl \
  --input chart.json
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

...

## License

...

---

**Note**: Ensure `maidata.txt` files comply with SEGA's secondary creation guidelines. This system is for research purposes only.
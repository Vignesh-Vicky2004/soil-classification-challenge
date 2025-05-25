# Soil Classification Challenge - README

## Overview
This repository contains the code and documentation for the Soil Classification Challenge. The challenge involves classifying soil images into four categories: Alluvial soil, Black Soil, Clay soil, and Red soil.

## Repository Structure
```
challenge-1/
├── data/                  # Data directory (add your dataset here)
├── docs/cards/            # Documentation and diagrams
│   └── architecture_diagram.png  # System architecture visualization
├── notebooks/             # Jupyter notebooks
│   ├── training.ipynb     # Model training pipeline
│   └── inference.ipynb    # Inference and submission generation
├── src/                   # Source code modules
├── requirements.txt       # Python dependencies
├── LICENSE                # License information
└── README.md              # This file
```

## Approach
The solution uses a hybrid approach combining:
1. **Feature Extraction**: Using CLIP ViT-H-14 pre-trained on the LAION-2B dataset
2. **Classification**: 
   - Logistic Regression trained on extracted features
   - Zero-shot classification using CLIP text-image similarity
   - Ensemble weighting (70% LR, 30% zero-shot)

## Getting Started

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Installation
```bash
pip install -r requirements.txt
```

### Usage
1. **Training**: Run the `notebooks/training.ipynb` notebook to train the model
2. **Inference**: Run the `notebooks/inference.ipynb` notebook to generate predictions

## Model Architecture
The architecture diagram in `docs/cards/architecture_diagram.png` illustrates the data flow and model components.

## Performance
The model achieves high accuracy on the validation set through the combination of supervised learning and zero-shot capabilities.

## License
This project is licensed under the terms specified in the LICENSE file.

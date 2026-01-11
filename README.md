# Machine Learning Practical Assignment - Restaurant Sales Analysis

## Project Overview

This project implements machine learning algorithms for restaurant sales prediction and product recommendation:

1. **LR #1**: Predict if a customer buying Crazy Schnitzel will also buy Crazy Sauce
2. **LR #2**: Multi-sauce recommendation system with Top-K recommendations
3. **Ranking**: Product ranking for upselling using Naive Bayes and k-NN

## Project Structure
```
├── data/
│   ├── raw/              # Place dataset here (ap_dataset.csv)
│   └── processed/        # Preprocessed features
├── src/
│   ├── data_loader.py    # Data loading utilities
│   ├── preprocessing.py  # Feature engineering
│   └── models/
│       ├── logistic_regression.py  # LR from scratch (Gradient Descent)
│       ├── evaluation.py           # Metrics (accuracy, F1, ROC-AUC)
│       └── ranking.py              # Naive Bayes & k-NN from scratch
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   ├── 02_lr_crazy_sauce.ipynb   # LR #1: Crazy Sauce prediction
│   ├── 03_lr_multi_sauce.ipynb   # LR #2: Multi-sauce recommendation
│   ├── 04_ranking_upsell.ipynb   # Expected value ranking
│   └── 05_ranking_ml.ipynb       # ML ranking (NB, k-NN) with Hit@K
├── results/              # Generated figures
└── report/
    └── report.tex        # LaTeX report
```

## Setup

### 1. Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Download dataset and place in `data/raw/`
Dataset should be named `ap_dataset.csv`

### 3. Run notebooks:
```bash
jupyter notebook notebooks/
```

Run notebooks in order: 01 → 02 → 03 → 04 → 05

## Algorithms Implemented From Scratch

### Logistic Regression
- Sigmoid activation function
- Binary cross-entropy loss
- Gradient descent optimization
- L2 regularization

### Naive Bayes (Gaussian)
- Maximum likelihood estimation
- Gaussian probability density
- Log-likelihood computation

### k-Nearest Neighbors
- Euclidean distance metric
- Uniform and weighted voting

## Evaluation Metrics
- Accuracy, Precision, Recall, F1 Score
- ROC-AUC with curve visualization
- Confusion Matrix
- Hit@K, Precision@K, MRR (for ranking)

## Dataset Statistics
- Period: 2025-09-05 → 2025-12-03
- Receipts: ~7,869
- Product lines: 28,039
- Unique products: 59
- Average cart size: 3.56 products

## Authors
- Elisa Mercas & Denis Munteanu

## License
For educational purposes only.

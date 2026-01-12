# ATP Tennis Match Prediction

A complete machine learning pipeline for predicting ATP tennis match outcomes using historical data, engineered features, and decision tree models.

## 📊 Project Overview

This project predicts tennis match winners by:

- Engineering time-aware features (Elo ratings, H2H records, rolling form, serve stats)
- Training both a from-scratch CART implementation and sklearn's DecisionTreeClassifier
- Maintaining strict chronological ordering to prevent data leakage
- Using time-based train/test splitting for realistic evaluation

## 🏗️ Repository Structure

```
Tennies_prediction/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                 # (Download ATP match CSVs here)
│   ├── interim/             # Cached intermediate data
│   └── processed/           # Final feature matrix (parquet)
├── notebook/
│   └── 01_preprocessing_feature_engineering.ipynb
├── src/
│   ├── __init__.py
│   ├── config.py            # Project configuration
│   ├── io_utils.py          # File I/O utilities
│   ├── elo.py               # Elo rating system
│   ├── scratch_tree.py      # From-scratch CART implementation
│   ├── train_sklearn.py     # Sklearn model training
│   └── compare_models.py    # Model comparison script
└── outputs/
    ├── figures/             # Plots and visualizations
    └── models/              # Trained model artifacts
```

## 📦 Installation

### 1. Clone the repository

```bash
git clone https://github.com/Siba4442/Tennis-Match-Outcome-Prediction-
cd Tennies_prediction
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

Or using `uv` (faster):

```bash
uv pip install -r requirements.txt
```

## 📥 Data Setup

Download ATP match data from [Jeff Sackmann's tennis_atp repository](https://github.com/JeffSackmann/tennis_atp):

```bash
# Clone the tennis_atp repository
git clone https://github.com/JeffSackmann/tennis_atp.git data/raw/tennis_atp
```

Or manually download the CSV files to `data/raw/tennis_atp/`.

## 🚀 Usage

### Step 1: Preprocessing & Feature Engineering

Run the Jupyter notebook to generate the processed feature matrix:

```bash
jupyter notebook notebook/01_preprocessing_feature_engineering.ipynb
```

This notebook will:

- Load raw ATP match CSVs (1992-2024)
- Engineer 100+ features (Elo, H2H, rolling stats, serve metrics)
- Create visualizations (Elo trajectories, feature distributions, correlations)
- Export `data/processed/final_features.parquet`

**Key outputs:**

- `data/processed/final_features.parquet` - Complete feature matrix with labels

### Step 2: Train Sklearn Model

Train the sklearn DecisionTreeClassifier:

```bash
python -m src.train_sklearn
```

**Outputs:**

- Model accuracy and log-loss on test set
- Saved model: `outputs/models/sklearn_tree.joblib`

### Step 3: Compare Models

Compare scratch CART vs sklearn implementation:

```bash
python -m src.compare_models
```

**Outputs:**

- Side-by-side metrics comparison
- Probability distribution plot: `outputs/figures/proba_hist_compare.png`

## 🎯 Features

### Engineered Features (100+)

- **Elo Ratings:** Overall Elo, surface-specific Elo, Elo momentum gradients
- **Head-to-Head:** Overall H2H, surface-specific H2H
- **Rolling Form:** Win rates over windows [3, 5, 10, 25, 50, 100 matches]
- **Serve Stats:** Ace %, double fault %, 1st serve %, 2nd serve win %, break point saved %
  - Rolling windows: [3, 5, 10, 20, 50, 100, 200, 300, 2000 matches]
- **Basic Features:** Ranking, ranking points, age, height, match format, surface

### Model Configuration

- **Max Depth:** 6
- **Min Samples Split:** 200
- **Min Samples Leaf:** 100
- **Split Date:** 2022-12-31 (train ≤ 2022, test > 2022)

## 📈 Results

Example performance (will vary based on data):

```
Scratch CART     | acc=0.6523 | logloss=0.6321
Sklearn Tree     | acc=0.6518 | logloss=0.6329
```

## 🔍 Key Design Decisions

### 1. **Chronological Ordering**

All rolling features are computed using only past matches to prevent data leakage.

### 2. **Player Order Randomization**

We randomize player order (PLAYER_1 vs PLAYER_2) and flip feature signs to avoid the model learning that "winner" is always PLAYER_1.

### 3. **Time-Based Splitting**

Train on matches ≤ 2022-12-31, test on 2023+. No random shuffling to simulate real-world prediction.

### 4. **Elo System**

- Base rating: 1500
- K-factor: 32 (×1.15 for best-of-5 matches)
- Separate tracking for overall and surface-specific ratings

## 📊 Visualizations

The notebook generates:

1. **Class Balance** - Confirms ~50/50 split due to randomization
2. **Feature Distributions** - Histograms of key features
3. **Correlation Heatmap** - Feature redundancy analysis
4. **Overall Elo Trajectories** - Famous players over time
5. **Surface-Specific Elo** - Performance by surface (Hard/Clay/Grass)

## 🛠️ Configuration

Edit `src/config.py` to customize:

- Data paths
- Train/test split date
- Random seed

```python
@dataclass(frozen=True)
class SplitConfig:
    SPLIT_DATE: int = 20221231  # Change this for different splits
```



## 🙏 Acknowledgments

- **Data Source:** [Jeff Sackmann's tennis_atp repository](https://github.com/JeffSackmann/tennis_atp)
- **Inspiration:** FiveThirtyEight's Elo rating system for sports

---

**Happy Predicting! 🎾**

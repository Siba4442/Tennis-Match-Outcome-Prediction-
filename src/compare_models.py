"""Compare scratch CART implementation vs sklearn DecisionTreeClassifier."""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss

from .config import Paths, SplitConfig, RandomConfig
from .io_utils import load_parquet, ensure_dir
from .scratch_tree import DecisionTreeCART




def main():
    df = load_parquet(os.path.join(Paths.DATA_PROCESSED, "final_features.parquet"))

    y = df["RESULT"].astype(int).values
    dates = df["TOURNEY_DATE"].astype(int).values
    X_df = df.drop(columns=["RESULT", "TOURNEY_DATE"])  # exclude date

    train_mask = dates <= SplitConfig.SPLIT_DATE
    test_mask = dates > SplitConfig.SPLIT_DATE

    X_train = X_df.values[train_mask]
    y_train = y[train_mask]
    X_test = X_df.values[test_mask]
    y_test = y[test_mask]

    # --- Scratch tree ---
    scratch = DecisionTreeCART(
        max_depth=6,
        min_samples_split=200,
        min_samples_leaf=100,
        random_state=RandomConfig.SEED,
    )
    scratch.fit(X_train, y_train, feature_names=list(X_df.columns))
    scratch_proba = scratch.predict_proba(X_test)[:, 1]
    scratch_pred = (scratch_proba >= 0.5).astype(int)

    # --- Sklearn tree ---
    sk = DecisionTreeClassifier(
        criterion="gini",
        max_depth=6,
        min_samples_split=200,
        min_samples_leaf=100,
        random_state=RandomConfig.SEED,
    )
    sk.fit(X_train, y_train)
    sk_proba = sk.predict_proba(X_test)[:, 1]
    sk_pred = (sk_proba >= 0.5).astype(int)

    # Metrics
    def metrics(name, proba, pred):
        acc = accuracy_score(y_test, pred)
        ll = log_loss(y_test, np.column_stack([1 - proba, proba]))
        print(f"{name:<16} | acc={acc:.4f} | logloss={ll:.4f}")
        return acc, ll

    ensure_dir(Paths.OUTPUTS_FIG)

    m1 = metrics("Scratch CART", scratch_proba, scratch_pred)
    m2 = metrics("Sklearn Tree", sk_proba, sk_pred)

    # Plot: probability histograms
    plt.figure()
    plt.hist(scratch_proba, bins=30, alpha=0.6, label="Scratch")
    plt.hist(sk_proba, bins=30, alpha=0.6, label="Sklearn")
    plt.xlabel("Predicted P(PLAYER_1 wins)")
    plt.ylabel("Count")
    plt.legend()
    plt.title("Prediction Probability Distribution")
    out = os.path.join(Paths.OUTPUTS_FIG, "proba_hist_compare.png")
    plt.savefig(out, dpi=200, bbox_inches="tight")
    print(f"Saved: {out}")




if __name__ == "__main__":
    main()
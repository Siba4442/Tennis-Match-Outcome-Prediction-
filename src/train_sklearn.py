"""Train sklearn DecisionTreeClassifier on ATP match data."""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, log_loss

from .config import Paths, SplitConfig, RandomConfig
from .io_utils import load_parquet, ensure_dir


def main():
    data_path = os.path.join(Paths.DATA_PROCESSED, "final_features.parquet")
    df = load_parquet(data_path)

    y = df["RESULT"].astype(int).values
    dates = df["TOURNEY_DATE"].astype(int).values
    X_df = df.drop(columns=["RESULT", "TOURNEY_DATE"])  # keep date out of model

    train_mask = dates <= SplitConfig.SPLIT_DATE
    test_mask = dates > SplitConfig.SPLIT_DATE

    X_train = X_df.values[train_mask]
    y_train = y[train_mask]
    X_test = X_df.values[test_mask]
    y_test = y[test_mask]

    model = DecisionTreeClassifier(
        criterion="gini",
        max_depth=6,
        min_samples_split=200,
        min_samples_leaf=100,
        random_state=RandomConfig.SEED,
    )

    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, pred)
    ll = log_loss(y_test, np.column_stack([1 - proba, proba]))

    print(f"Sklearn DecisionTree | acc={acc:.4f} | logloss={ll:.4f}")

    ensure_dir(Paths.OUTPUTS_MODELS)
    out_path = os.path.join(Paths.OUTPUTS_MODELS, "sklearn_tree.joblib")
    joblib.dump({"model": model, "features": list(X_df.columns)}, out_path)
    print(f"Saved model to: {out_path}")


if __name__ == "__main__":
    main()

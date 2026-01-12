"""From-scratch CART Decision Tree implementation (binary classification).

This module implements a decision tree classifier using the CART algorithm with:
- Gini impurity for split quality measurement
- Greedy best split selection
- Support for numeric features only
- Probability predictions for leaves

This is an educational implementation for comparing with sklearn.
"""

import numpy as np


class DecisionTreeCART:
    class Node:
        __slots__ = (
            "feature",
            "threshold",
            "left",
            "right",
            "proba",
            "pred",
            "n",
            "gini",
        )

        def __init__(
            self,
            feature=None,
            threshold=None,
            left=None,
            right=None,
            proba=None,
            pred=None,
            n=0,
            gini=None,
        ):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.proba = proba  # P(class=1) at leaf (or node)
            self.pred = pred  # predicted class (0/1)
            self.n = n  # samples at node
            self.gini = gini  # gini at node

    def __init__(
        self,
        max_depth=6,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features=None,
        random_state=42,
    ):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features  # None or int or "sqrt" or "log2"
        self.random_state = random_state
        self.rng_ = np.random.default_rng(random_state)
        self.root_ = None
        self.n_features_ = None
        self.feature_names_ = None

    # ---------- Public API ----------
    def fit(self, X, y, feature_names=None):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be 1D array with same length as X")
        if not np.all((y == 0) | (y == 1)):
            raise ValueError("This implementation expects binary labels 0/1")

        self.n_features_ = X.shape[1]
        self.feature_names_ = feature_names
        self.root_ = self._build_tree(X, y, depth=0)
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        probs = np.zeros(X.shape[0], dtype=np.float64)
        for i in range(X.shape[0]):
            node = self.root_
            while node.left is not None and node.right is not None:
                if X[i, node.feature] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            probs[i] = node.proba
        # Return 2-column proba like sklearn: P(0), P(1)
        return np.vstack([1.0 - probs, probs]).T

    def predict(self, X):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= 0.5).astype(np.int64)

    def print_tree(self, max_lines=200):
        lines = []
        self._to_lines(self.root_, depth=0, lines=lines)
        for i, line in enumerate(lines[:max_lines]):
            print(line)
        if len(lines) > max_lines:
            print(f"... ({len(lines) - max_lines} more lines)")

    # ---------- Internals ----------
    def _gini(self, y):
        # y is 0/1
        if y.size == 0:
            return 0.0
        p1 = y.mean()
        return 1.0 - (p1 * p1 + (1.0 - p1) * (1.0 - p1))

    def _best_feature_count(self):
        if self.max_features is None:
            return self.n_features_
        if isinstance(self.max_features, int):
            return max(1, min(self.n_features_, self.max_features))
        if self.max_features == "sqrt":
            return max(1, int(np.sqrt(self.n_features_)))
        if self.max_features == "log2":
            return max(1, int(np.log2(self.n_features_)))
        raise ValueError("max_features must be None, int, 'sqrt', or 'log2'")

    def _build_tree(self, X, y, depth):
        n = y.size
        g = self._gini(y)
        p1 = float(y.mean()) if n else 0.0
        pred = 1 if p1 >= 0.5 else 0

        node = self.Node(proba=p1, pred=pred, n=n, gini=g)

        # Stopping conditions
        if depth >= self.max_depth:
            return node
        if n < self.min_samples_split:
            return node
        if g == 0.0:
            return node

        # Choose candidate features (feature subsampling like RF)
        m = self._best_feature_count()
        feat_idx = np.arange(self.n_features_)
        if m < self.n_features_:
            feat_idx = self.rng_.choice(feat_idx, size=m, replace=False)

        best = self._best_split(X, y, feat_idx)
        if best is None:
            return node

        feat, thr, left_mask = best
        n_left = int(left_mask.sum())
        n_right = n - n_left
        if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
            return node

        # Create children
        node.feature = feat
        node.threshold = float(thr)
        node.left = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        node.right = self._build_tree(X[~left_mask], y[~left_mask], depth + 1)
        return node

    def _best_split(self, X, y, feat_idx):
        # Find split that minimizes weighted gini
        n = y.size
        parent_gini = self._gini(y)
        best_gain = 0.0
        best_feat = None
        best_thr = None
        best_left_mask = None

        for f in feat_idx:
            xf = X[:, f]

            # Sort by feature values
            order = np.argsort(xf, kind="mergesort")
            xf_sorted = xf[order]
            y_sorted = y[order]

            # Candidate thresholds: midpoints where value changes
            diffs = xf_sorted[1:] != xf_sorted[:-1]
            if not np.any(diffs):
                continue

            # Prefix sums for class=1 counts
            y1_prefix = np.cumsum(y_sorted)
            left_count = np.arange(1, n + 1)
            right_count = n - left_count

            left_y1 = y1_prefix
            right_y1 = y1_prefix[-1] - y1_prefix

            # We only consider split positions where feature changes
            split_positions = np.where(diffs)[0]  # split between i and i+1

            # Compute gini quickly for each candidate split
            # left indices: [0..i], right: [i+1..n-1]
            l_n = left_count[split_positions]
            r_n = right_count[split_positions]

            # Enforce min_samples_leaf early
            valid = (l_n >= self.min_samples_leaf) & (r_n >= self.min_samples_leaf)
            if not np.any(valid):
                continue
            split_positions = split_positions[valid]
            l_n = l_n[valid]
            r_n = r_n[valid]

            l_p1 = left_y1[split_positions] / l_n
            r_p1 = right_y1[split_positions] / r_n

            l_g = 1.0 - (l_p1**2 + (1.0 - l_p1) ** 2)
            r_g = 1.0 - (r_p1**2 + (1.0 - r_p1) ** 2)

            weighted_g = (l_n / n) * l_g + (r_n / n) * r_g
            gain = parent_gini - weighted_g

            j = int(np.argmax(gain))
            if gain[j] > best_gain:
                i = int(split_positions[j])
                thr = 0.5 * (xf_sorted[i] + xf_sorted[i + 1])

                # Build mask in original order
                left_mask = xf <= thr

                best_gain = float(gain[j])
                best_feat = int(f)
                best_thr = float(thr)
                best_left_mask = left_mask

        if best_feat is None:
            return None
        return best_feat, best_thr, best_left_mask

    def _to_lines(self, node, depth, lines):
        indent = "  " * depth
        if node.left is None and node.right is None:
            lines.append(
                f"{indent}Leaf(n={node.n}, gini={node.gini:.4f}, p1={node.proba:.3f}, pred={node.pred})"
            )
            return
        fname = (
            self.feature_names_[node.feature]
            if self.feature_names_ is not None
            else f"X[{node.feature}]"
        )
        lines.append(
            f"{indent}if {fname} <= {node.threshold:.6g} (n={node.n}, gini={node.gini:.4f}, p1={node.proba:.3f}):"
        )
        self._to_lines(node.left, depth + 1, lines)
        lines.append(f"{indent}else:")
        self._to_lines(node.right, depth + 1, lines)

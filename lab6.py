"""
================================================================================
22AIE213 — Lab Session 06 : Decision Trees from Scratch
================================================================================
Dataset:
    X_features.npy  →  (20001, 384)  float32  — sentence/text embeddings
    y_labels.npy    →  (20001,)      int64    — binary labels: 0 or 1

Strategy for high-dimensional embeddings:
    Raw 384-dim vectors cannot be split as categorical features directly.
    We use PCA to compress them into a small set of interpretable components,
    then bin those components to make them categorical — exactly as the lab
    asks for continuous → categorical conversion (A4).

    • A1–A5  : 3 000-sample subset  +  top-10 PCA components  (fast, correct)
    • A6     : full 20 001 samples  +  top-10 PCA components  (best accuracy)
    • A7     : full 20 001 samples  +  top-2  PCA components  (2-D plot)
================================================================================
"""

# ── Standard library ──────────────────────────────────────────────────────────
from collections import Counter          # fast frequency counting for entropy/gini

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── Scikit-learn utilities ────────────────────────────────────────────────────
from sklearn.decomposition import PCA                       # dimensionality reduction
from sklearn.tree import DecisionTreeClassifier, plot_tree  # reference DT for A6
from sklearn.model_selection import train_test_split        # train/test split
from sklearn.inspection import DecisionBoundaryDisplay      # decision boundary plot (A7)


# ════════════════════════════════════════════════════════════════════════════════
# 0.  LOAD & PREPARE DATA
# ════════════════════════════════════════════════════════════════════════════════

# Load the raw numpy arrays from disk
X_full = np.load("/mnt/user-data/uploads/X_features.npy")   # shape: (20001, 384)
y_full = np.load("/mnt/user-data/uploads/y_labels.npy")     # shape: (20001,)

print(f"Loaded  X: {X_full.shape}  |  y: {y_full.shape}")
print(f"Classes : {np.unique(y_full)}  |  Counts: {Counter(y_full.tolist())}")

# ── Sampling ──────────────────────────────────────────────────────────────────
# The custom tree (A5) iterates over every unique bin value for every feature
# at every node — O(n * d) per split.  Running it on 20 001 rows x 384 dims
# would be extremely slow, so we draw a random sample of 3 000 rows
# (still >10x the number of leaves expected at depth=5).
np.random.seed(42)                                           # reproducibility
sample_idx = np.random.choice(len(X_full), size=3000, replace=False)
X_sample   = X_full[sample_idx]    # (3000, 384)
y_sample   = y_full[sample_idx]    # (3000,)

# ── PCA dimensionality reduction (for A1–A5) ─────────────────────────────────
# We keep the top 10 principal components.  These capture the dominant axes of
# variance in the embedding space and serve as our "features" for the tree.
# Each component is a continuous scalar → will be binned in A4.
pca_feat      = PCA(n_components=10, random_state=42)
X_pca10       = pca_feat.fit_transform(X_sample)   # fit on sample → shape (3000, 10)
FEATURE_NAMES = [f"pca_{i}" for i in range(10)]    # column names for the DataFrame

# Build a tidy DataFrame so tree functions can reference columns by name
df           = pd.DataFrame(X_pca10, columns=FEATURE_NAMES)
df["target"] = y_sample           # attach labels as a column
TARGET_COL   = "target"


# ════════════════════════════════════════════════════════════════════════════════
# A1.  ENTROPY  —  H = -Σ p_i · log₂(p_i)
# ════════════════════════════════════════════════════════════════════════════════

def equal_width_binning(series: pd.Series, n_bins: int = 4) -> pd.Series:
    """
    A1 / A4 helper — Equal-Width Binning.

    Divides the value range [min, max] into `n_bins` intervals of identical
    width and assigns each value an integer bin label (0, 1, ..., n_bins-1).

    Parameters
    ----------
    series : pd.Series
        Continuous numeric column to discretise.
    n_bins : int, default 4
        Number of equal-width bins to create.

    Returns
    -------
    pd.Series
        Integer bin labels (NaN for values outside the fitted range).

    Example
    -------
    >>> equal_width_binning(pd.Series([1.0, 2.0, 3.0, 4.0]), n_bins=2)
    0    0
    1    0
    2    1
    3    1
    """
    return pd.cut(series, bins=n_bins, labels=False)


def equal_frequency_binning(series: pd.Series, n_bins: int = 4) -> pd.Series:
    """
    A4 helper — Equal-Frequency (Quantile) Binning.

    Each bin contains approximately the same number of data points, making
    this robust to skewed distributions (unlike equal-width binning).

    Parameters
    ----------
    series : pd.Series
        Continuous numeric column to discretise.
    n_bins : int, default 4
        Number of quantile-based bins.

    Returns
    -------
    pd.Series
        Integer bin labels.  `duplicates='drop'` prevents errors when
        repeated quantile edges would create empty bins.
    """
    return pd.qcut(series, q=n_bins, labels=False, duplicates="drop")


def bin_column(series: pd.Series,
               binning_type: str = "width",
               n_bins: int = 4) -> pd.Series:
    """
    A4 — Unified binning dispatcher with default parameters.

    Acts as the single entry-point for binning, mimicking Python's
    function-overloading idiom via default argument values.  Callers can
    omit `binning_type` and `n_bins` to get sensible defaults.

    Parameters
    ----------
    series       : pd.Series  — continuous column to bin
    binning_type : str        — 'width' (equal-width) or 'frequency' (quantile)
    n_bins       : int        — number of bins; defaults to 4

    Returns
    -------
    pd.Series with integer bin labels.

    Raises
    ------
    ValueError if `binning_type` is not 'width' or 'frequency'.
    """
    if binning_type == "frequency":
        return equal_frequency_binning(series, n_bins)
    elif binning_type == "width":
        return equal_width_binning(series, n_bins)
    else:
        raise ValueError(f"Unknown binning_type='{binning_type}'. "
                         f"Choose 'width' or 'frequency'.")


def calculate_entropy(labels) -> float:
    """
    A1 — Shannon Entropy  H = -Σ p_i · log₂(p_i)

    Measures the impurity / uncertainty in a set of class labels.
    • H = 0   → all labels are the same  (pure node, no uncertainty)
    • H = 1   → labels are split 50/50   (maximum uncertainty, binary case)

    The formula avoids log(0) by skipping zero-probability classes.

    Parameters
    ----------
    labels : array-like
        Class labels (integers or strings).  Works for any number of classes.

    Returns
    -------
    float  — entropy value >= 0
    """
    labels = np.array(labels)
    n      = len(labels)
    if n == 0:
        return 0.0                           # empty node → entropy undefined, treat as 0

    counts  = Counter(labels)               # {class: frequency}
    entropy = 0.0
    for count in counts.values():
        p_i = count / n                     # probability of this class
        if p_i > 0:                         # guard against log2(0) = -inf
            entropy -= p_i * np.log2(p_i)  # accumulate -p·log₂(p) for each class

    return entropy


# ── Compute and display A1 result ─────────────────────────────────────────────
entropy_val = calculate_entropy(y_sample)
print(f"\n[A1] Dataset Entropy : {entropy_val:.4f}")
# Expected ≈ 1.0 — the dataset is perfectly balanced (50% / 50%).
# H_max for binary = -0.5·log₂(0.5) - 0.5·log₂(0.5) = 1.0


# ════════════════════════════════════════════════════════════════════════════════
# A2.  GINI INDEX  —  Gini = 1 - Σ p_j²
# ════════════════════════════════════════════════════════════════════════════════

def calculate_gini(labels) -> float:
    """
    A2 — Gini Impurity  Gini = 1 - Σ p_j²

    An alternative impurity measure to entropy.  It represents the probability
    that a randomly chosen sample would be misclassified if labelled according
    to the class distribution of the node.

    • Gini = 0    → pure node (all same class)
    • Gini = 0.5  → maximally impure binary split (50/50)

    Compared to entropy, Gini is slightly faster (no log) and tends to produce
    similar tree structures in practice.

    Parameters
    ----------
    labels : array-like
        Class labels.

    Returns
    -------
    float  — gini value in [0, 0.5] for binary, [0, 1 - 1/k] for k classes.
    """
    labels = np.array(labels)
    n      = len(labels)
    if n == 0:
        return 0.0

    counts = Counter(labels)
    gini   = 1.0                  # start from 1 and subtract Σ p_j²
    for count in counts.values():
        p_j   = count / n
        gini -= p_j ** 2          # each term reduces gini toward 0

    return gini


# ── Compute and display A2 result ─────────────────────────────────────────────
gini_val = calculate_gini(y_sample)
print(f"[A2] Dataset Gini Index : {gini_val:.4f}")
# Expected = 0.5 for a balanced binary dataset  (1 - 0.25 - 0.25 = 0.5)


# ════════════════════════════════════════════════════════════════════════════════
# A3 & A4.  INFORMATION GAIN — ROOT NODE SELECTION
# ════════════════════════════════════════════════════════════════════════════════

def information_gain(data: pd.DataFrame,
                     feature_col: str,
                     target_col: str,
                     binning_type: str = "width",
                     n_bins: int = 4) -> float:
    """
    A3 — Information Gain  IG(S, A) = H(S) - Σ_v (|S_v|/|S|) · H(S_v)

    Measures how much a feature A reduces the entropy of the target S.
    A higher IG means the feature is a better candidate for splitting.

    A4 integration: if the feature column contains continuous floats it is
    automatically discretised using `bin_column()` before computing IG.
    This converts the problem to the categorical case required by A3.

    Parameters
    ----------
    data         : pd.DataFrame — the current node's subset of data
    feature_col  : str          — name of the feature column to evaluate
    target_col   : str          — name of the class-label column
    binning_type : str          — passed to bin_column ('width' or 'frequency')
    n_bins       : int          — number of bins for continuous features

    Returns
    -------
    float  — information gain >= 0
    """
    # H(S) — entropy of the full node before any split
    parent_entropy = calculate_entropy(data[target_col])

    n       = len(data)
    feature = data[feature_col].copy()

    # A4: auto-bin continuous features so we can treat them as categorical
    if feature.dtype in [np.float64, np.float32, float]:
        feature = bin_column(feature, binning_type=binning_type, n_bins=n_bins)

    # Weighted child entropy: Σ_v (|S_v|/|S|) · H(S_v)
    weighted_entropy = 0.0
    for val in feature.dropna().unique():       # one child per bin/category
        subset  = data[feature == val]          # rows belonging to this branch
        weight  = len(subset) / n              # fraction of total samples in node
        weighted_entropy += weight * calculate_entropy(subset[target_col])

    # IG = how much entropy is reduced by knowing this feature's value
    return parent_entropy - weighted_entropy


def find_root_node(data: pd.DataFrame,
                   feature_cols,
                   target_col: str):
    """
    A3 — Identify the best feature for the root node of the Decision Tree.

    Computes Information Gain for every candidate feature and returns the one
    with the highest IG — the feature that most cleanly separates the classes.

    Parameters
    ----------
    data         : pd.DataFrame — full training data
    feature_cols : list[str]    — candidate feature column names
    target_col   : str          — class-label column

    Returns
    -------
    best_feature : str   — feature column with highest IG
    gains        : dict  — {feature_name: information_gain} for all features
    """
    gains = {}
    for col in feature_cols:
        gains[col] = information_gain(data, col, target_col)
        print(f"   IG [{col:>8}]: {gains[col]:.4f}")

    best_feature = max(gains, key=gains.get)   # argmax over IG values
    return best_feature, gains


# ── Compute and display A3 result ─────────────────────────────────────────────
print("\n[A3] Information Gain per PCA feature:")
root_feat, ig_scores = find_root_node(df, FEATURE_NAMES, TARGET_COL)
print(f"\n   Best Root Node: '{root_feat}'  (IG = {ig_scores[root_feat]:.4f})")


# ════════════════════════════════════════════════════════════════════════════════
# A5.  CUSTOM DECISION TREE MODULE
# ════════════════════════════════════════════════════════════════════════════════

class DecisionTreeNode:
    """
    Represents a single node in the custom Decision Tree.

    Attributes
    ----------
    feature  : str | None
        The feature column this node splits on (None for leaf nodes).
    children : dict
        Maps each bin/category value -> child DecisionTreeNode.
        Empty dict for leaf nodes.
    label    : int | str | None
        The predicted class for leaf nodes (None for internal nodes).
    """
    def __init__(self):
        self.feature  = None    # splitting feature (internal nodes only)
        self.children = {}      # branch value → child node
        self.label    = None    # majority class prediction (leaf nodes only)


class MyDecisionTree:
    """
    A5 — Decision Tree Classifier built entirely from scratch.

    Algorithm  (ID3-style greedy top-down induction)
    ---------
    1. Compute Information Gain for every remaining feature.
    2. Split on the feature with the highest IG.
    3. Create one child branch per unique bin/category value.
    4. Recurse into each child until a stopping criterion fires.

    Stopping criteria (any one triggers a leaf):
      • All samples in the node share the same class  (pure node).
      • Tree depth has reached `max_depth`.
      • Fewer than `min_samples_split` samples remain.
      • No features are left to split on.
      • Best IG <= 0  (no split would reduce impurity).

    Continuous features are discretised at fit-time using stored bin edges
    so that identical boundaries are applied consistently at predict-time.

    Parameters
    ----------
    max_depth         : int  — maximum depth of the tree (default 5)
    min_samples_split : int  — minimum samples needed to attempt a split (default 5)
    binning_type      : str  — 'width' (equal-width) or 'frequency' (quantile)
    n_bins            : int  — number of bins for continuous features (default 4)
    """

    def __init__(self,
                 max_depth: int = 5,
                 min_samples_split: int = 5,
                 binning_type: str = "width",
                 n_bins: int = 4):
        self.max_depth         = max_depth
        self.min_samples_split = min_samples_split
        self.binning_type      = binning_type
        self.n_bins            = n_bins
        self.root              = None   # root DecisionTreeNode, set after fit()
        self._bin_maps         = {}     # {col_name: bin_edges} reused at predict time

    # ── Private helpers ───────────────────────────────────────────────────────

    def _bin(self, series: pd.Series, col: str) -> pd.Series:
        """
        Bin a continuous column, learning and caching bin edges on the first call.

        On the first call for a given column the bin edges are computed from
        training data and cached in self._bin_maps.  Subsequent calls (at
        predict time) reuse those exact edges so test values fall into the
        same bins as training values — preventing data leakage.

        Parameters
        ----------
        series : pd.Series  — column to bin (may be train or test data)
        col    : str        — column name used as cache key

        Returns
        -------
        pd.Series with integer bin labels (NaN if value is outside training range).
        """
        if col not in self._bin_maps:
            # First call: learn edges from training data and cache them
            if self.binning_type == "frequency":
                _, edges = pd.qcut(series, q=self.n_bins, retbins=True,
                                   labels=False, duplicates="drop")
            else:
                _, edges = pd.cut(series, bins=self.n_bins, retbins=True,
                                  labels=False)
            self._bin_maps[col] = edges   # persist edges for predict-time reuse

        # Apply the stored edges (include_lowest=True catches the minimum value)
        return pd.cut(series, bins=self._bin_maps[col],
                      labels=False, include_lowest=True)

    def _preprocess(self, data: pd.DataFrame, cols: list) -> pd.DataFrame:
        """
        Apply binning to all continuous feature columns in the DataFrame.

        Makes a copy so the caller's original DataFrame is never mutated.

        Parameters
        ----------
        data : pd.DataFrame  — input data (features + optionally target)
        cols : list[str]     — feature columns to inspect and possibly bin

        Returns
        -------
        pd.DataFrame with float columns replaced by integer bin labels.
        """
        d = data.copy()
        for c in cols:
            # Only discretise columns that hold continuous float values
            if d[c].dtype in [np.float64, np.float32, float]:
                d[c] = self._bin(d[c], c)
        return d

    def _build(self,
               data: pd.DataFrame,
               feature_cols: list,
               target_col: str,
               depth: int) -> DecisionTreeNode:
        """
        Recursively build the tree using greedy top-down splitting (ID3).

        At each call this function either returns a leaf node (if a stopping
        criterion fires) or creates an internal split node and recurses into
        each branch.

        Parameters
        ----------
        data         : pd.DataFrame — current node's training subset
        feature_cols : list[str]    — features still available for splitting
        target_col   : str          — class-label column name
        depth        : int          — current tree depth (root = 0)

        Returns
        -------
        DecisionTreeNode — the root of the sub-tree grown from `data`
        """
        node   = DecisionTreeNode()
        labels = data[target_col]

        # ── Stopping criteria — return a leaf immediately if any fires ────────
        if (len(labels.unique()) == 1           # all samples have the same class
                or depth >= self.max_depth       # depth budget exhausted
                or len(data) < self.min_samples_split  # too few samples to split
                or not feature_cols):            # no features remain
            node.label = labels.mode()[0]        # predict the majority class
            return node

        # ── Find the best feature to split on ────────────────────────────────
        # Compute IG for every remaining feature and pick the one with max IG
        gains = {c: information_gain(data, c, target_col) for c in feature_cols}
        best  = max(gains, key=gains.get)

        # If no feature improves purity at all, make this a leaf
        if gains[best] <= 0:
            node.label = labels.mode()[0]
            return node

        # ── Create the split node ─────────────────────────────────────────────
        node.feature = best
        # Child nodes may not reuse the feature that was just split on
        rest = [c for c in feature_cols if c != best]

        # Create one child branch for each observed bin value
        for val in data[best].dropna().unique():
            subset = data[data[best] == val]          # rows that fall in this bin
            node.children[val] = (
                self._build(subset, rest, target_col, depth + 1)  # recurse
                if len(subset) > 0
                else self._leaf(labels)   # empty branch → majority-class leaf
            )

        return node

    def _leaf(self, labels: pd.Series) -> DecisionTreeNode:
        """
        Create a leaf node that predicts the majority class from `labels`.

        Used as a safe fallback when a branch has zero samples (e.g., a bin
        value was seen at fit time but produced an empty subset).
        """
        n       = DecisionTreeNode()
        n.label = labels.mode()[0]   # most frequent class in the parent node
        return n

    # ── Public API ────────────────────────────────────────────────────────────

    def fit(self, data: pd.DataFrame, feature_cols, target_col: str):
        """
        Train the Decision Tree on the provided DataFrame.

        Steps
        -----
        1. Bin all continuous feature columns (learning and storing bin edges).
        2. Recursively build the tree from the root downward.

        Parameters
        ----------
        data         : pd.DataFrame — training data containing features + labels
        feature_cols : list[str]    — names of the feature columns to use
        target_col   : str          — name of the class-label column
        """
        # _preprocess learns bin edges from training data (fit + transform)
        proc               = self._preprocess(data, feature_cols)
        self.feature_cols_ = list(feature_cols)   # save for predict-time use
        self.target_col_   = target_col

        # Kick off recursive tree construction from depth = 0
        self.root = self._build(proc, self.feature_cols_, target_col, depth=0)
        print("\n[A5] Custom Decision Tree trained!")

    def _predict_one(self, row: pd.Series, node: DecisionTreeNode):
        """
        Traverse the tree for a single pre-processed sample row.

        At each internal node the method looks up the row's bin value for that
        node's splitting feature and follows the matching child branch.
        If the bin value was never seen during training (out-of-range at
        predict time), it falls back to the first available child.

        Parameters
        ----------
        row  : pd.Series        — one row of the pre-processed DataFrame
        node : DecisionTreeNode — current node during traversal

        Returns
        -------
        Predicted class label (int or str).
        """
        # Base case: we've reached a leaf node
        if node.label is not None:
            return node.label

        # Lookup the bin value of this feature for the current sample
        val   = row.get(node.feature)
        child = node.children.get(val)

        if child is None:
            # Unseen bin value at predict time → use the first known child
            # (graceful degradation; avoids KeyError on out-of-range inputs)
            return list(node.children.values())[0].label

        return self._predict_one(row, child)   # recurse into matched branch

    def predict(self, data: pd.DataFrame) -> pd.Series:
        """
        Predict class labels for all rows in `data`.

        Applies the bin edges learned during fit() (transform only, no refit),
        then traverses the tree for each row independently.

        Parameters
        ----------
        data : pd.DataFrame — data to predict on (must contain the same feature
                              columns used during fit)

        Returns
        -------
        pd.Series of predicted class labels aligned with data's index.
        """
        # Reuse stored bin edges — do NOT call fit_transform again
        proc = self._preprocess(data, self.feature_cols_)
        return proc.apply(lambda r: self._predict_one(r, self.root), axis=1)

    def print_tree(self, node: DecisionTreeNode = None, indent: str = ""):
        """
        Recursively print a human-readable text representation of the tree.

        Internal nodes display the splitting feature name.
        Leaf nodes display the predicted class.
        Indentation grows with depth to convey tree structure visually.

        Parameters
        ----------
        node   : DecisionTreeNode — starting node (defaults to root)
        indent : str              — current indentation prefix (grows with depth)
        """
        node = node or self.root

        if node.label is not None:
            # Leaf node — show predicted class
            print(f"{indent}-> LEAF  class = {node.label}")
            return

        # Internal node — show which feature is being split on
        print(f"{indent}[split on: {node.feature}]")
        for val, child in node.children.items():
            print(f"{indent}  |- bin = {val}")
            self.print_tree(child, indent + "  |   ")   # increase indent for depth

    def accuracy(self, data: pd.DataFrame, target_col: str) -> float:
        """
        Compute classification accuracy on a labelled DataFrame.

        Parameters
        ----------
        data       : pd.DataFrame — data with both features and true labels
        target_col : str          — column containing the ground-truth labels

        Returns
        -------
        float in [0, 1] — fraction of samples correctly classified.
        """
        preds = self.predict(data)
        # Element-wise comparison, then take the mean (True=1, False=0)
        return (preds.values == data[target_col].values).mean()


# ── Train and evaluate the custom tree ────────────────────────────────────────
my_tree = MyDecisionTree(max_depth=5, binning_type="width", n_bins=4)
my_tree.fit(df, FEATURE_NAMES, TARGET_COL)

print("\n--- Custom Tree Structure (max_depth = 5) ---")
my_tree.print_tree()

# 80/20 hold-out split for unbiased accuracy estimate
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
acc = my_tree.accuracy(test_df, TARGET_COL)
print(f"\n   Custom Tree Test Accuracy : {acc * 100:.2f}%")


# ════════════════════════════════════════════════════════════════════════════════
# A6.  VISUALIZE THE DECISION TREE  (sklearn reference implementation)
# ════════════════════════════════════════════════════════════════════════════════
print("\n[A6] Training sklearn DT on full dataset for visualization...")

# Project all 20 001 samples into the same 10-component PCA space.
# We call pca_feat.transform() (NOT fit_transform) so the component axes stay
# consistent with the PCA fitted on the 3 000-sample subset used in A1-A5.
X_pca10_full = pca_feat.transform(X_full)          # shape: (20001, 10)

# Standard 80/20 train-test split on the full projected data
X_tr, X_te, y_tr, y_te = train_test_split(
    X_pca10_full, y_full, test_size=0.2, random_state=42)

# Sklearn's DT with the same max_depth for a direct comparison with our tree
clf = DecisionTreeClassifier(max_depth=5, random_state=42)
clf.fit(X_tr, y_tr)
print(f"   Sklearn DT Test Accuracy : {clf.score(X_te, y_te) * 100:.2f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
# plot_tree() renders each node as a box showing: split feature & threshold,
# sample count, gini/entropy value, and majority class at that node.
fig, ax = plt.subplots(figsize=(24, 12))
plot_tree(
    clf,
    feature_names=FEATURE_NAMES,           # label each split condition
    class_names=["Class 0", "Class 1"],    # label leaf predictions
    filled=True,                            # colour nodes by majority class
    rounded=True,                           # rounded corners (cosmetic)
    fontsize=9,
    ax=ax
)
plt.title("A6 - Decision Tree Visualization\n"
          "(Top-10 PCA Components of 384-dim Embeddings)", fontsize=14)
plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/A6_decision_tree.png", dpi=150)
plt.close()
print("   Saved -> A6_decision_tree.png")


# ════════════════════════════════════════════════════════════════════════════════
# A7.  DECISION BOUNDARY  (2 features in 2-D vector space)
# ════════════════════════════════════════════════════════════════════════════════
print("\n[A7] Plotting Decision Boundary on top-2 PCA components...")

# A7 requires exactly 2 features so the boundary can be drawn on a 2-D plane.
# A fresh PCA with n_components=2 is fitted on the full dataset for maximum
# variance capture.  The two axes represent the directions in 384-D space
# along which the embeddings vary the most.
pca_2d = PCA(n_components=2, random_state=42)
X_2d   = pca_2d.fit_transform(X_full)   # shape: (20001, 2)

# Train/test split in the 2-D space (same seed for reproducibility)
X_tr2, X_te2, y_tr2, y_te2 = train_test_split(
    X_2d, y_full, test_size=0.2, random_state=42)

# Fit a DT classifier in the 2-D PCA space
clf_2d = DecisionTreeClassifier(max_depth=5, random_state=42)
clf_2d.fit(X_tr2, y_tr2)

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 7))

# DecisionBoundaryDisplay creates a fine meshgrid over the 2-D feature space,
# calls clf_2d.predict() on every grid point, and fills/contours the resulting
# regions — visually showing where the DT draws its axis-aligned splits.
DecisionBoundaryDisplay.from_estimator(
    clf_2d, X_tr2,
    response_method="predict",   # hard class predictions (not probabilities)
    cmap="coolwarm",             # red = class 1 region, blue = class 0 region
    alpha=0.4,                   # semi-transparent so data points show through
    ax=ax
)

# Overlay all 20 001 data points coloured by their true label
scatter = ax.scatter(
    X_2d[:, 0], X_2d[:, 1],
    c=y_full,                   # colour = true class (not predicted)
    cmap="coolwarm",
    edgecolors="k",
    s=8,                        # small markers — 20 001 points would crowd the plot
    alpha=0.6,
    linewidths=0.3
)
plt.colorbar(scatter, ax=ax, ticks=[0, 1], label="True Class")

# Label axes with the percentage of total variance each component explains
ax.set_xlabel(
    f"PCA Component 1  ({pca_2d.explained_variance_ratio_[0] * 100:.1f}% variance)")
ax.set_ylabel(
    f"PCA Component 2  ({pca_2d.explained_variance_ratio_[1] * 100:.1f}% variance)")
ax.set_title(
    "A7 - Decision Boundary in 2-D PCA Space\n"
    "(384-dim embeddings projected to 2 principal components)")

plt.tight_layout()
plt.savefig("/mnt/user-data/outputs/A7_decision_boundary.png", dpi=150)
plt.close()
print("   Saved -> A7_decision_boundary.png")

print("\nAll tasks A1-A7 completed successfully!")

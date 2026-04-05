# ============================================================
# Multilingual Fake News Detection — 11 Classifiers
# Includes: Train vs Test, CatBoost, RandomizedSearchCV
# ============================================================

# STEP 1 — Install
import subprocess
subprocess.run(["pip", "install", "scikit-learn", "xgboost",
                "catboost", "numpy", "pandas",
                "matplotlib", "seaborn", "-q"])

# STEP 2 — Imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings, time
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, f1_score,
                             precision_score, recall_score)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              AdaBoostClassifier, ExtraTreesClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from catboost import CatBoostClassifier

# ─────────────────────────────────────────────────────────
#  ✏️  CHANGE THESE 3 LINES FOR EACH EMBEDDING
# ─────────────────────────────────────────────────────────
EMBEDDINGS_FILE = "/content/embeddings.npy"   # ← your embeddings file
LABELS_FILE     = "/content/df_labels.csv"    # ← your labels file
MODEL_NAME      = "MuRIL"                     # ← model name for output files
# ─────────────────────────────────────────────────────────

# STEP 3 — Upload files
from google.colab import files as colab_files
print("📤 Upload your embeddings.npy:")
colab_files.upload()
print("📤 Upload your df_labels.csv:")
colab_files.upload()

# STEP 4 — Load data
print(f"\n📂 Loading {MODEL_NAME} embeddings...")
X  = np.load(EMBEDDINGS_FILE).astype(np.float32)
df = pd.read_csv(LABELS_FILE)
y  = df["label"].values

print(f"  Embeddings : {X.shape}")
print(f"  Fake (1)   : {int(y.sum())}")
print(f"  Real (0)   : {int((y==0).sum())}")

# STEP 5 — Train / Test split  80 / 20
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"  Train : {len(X_train)}  |  Test : {len(X_test)}")

# ============================================================
# STEP 6 — RandomizedSearchCV Hyperparameter Tuning  (A2)
# We tune the 3 most impactful classifiers to save time
# ============================================================
print(f"\n{'='*65}")
print("  🔍 A2 — RandomizedSearchCV Hyperparameter Tuning")
print(f"{'='*65}")

# — Tune Random Forest —
print("\n  [1/3] Tuning Random Forest...")
rf_params = {
    "n_estimators"      : [50, 100, 200],
    "max_depth"         : [None, 10, 20, 30],
    "min_samples_split" : [2, 5, 10],
    "min_samples_leaf"  : [1, 2, 4],
    "max_features"      : ["sqrt", "log2"],
}
rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, n_jobs=-1),
    rf_params, n_iter=10, cv=3, scoring="accuracy",
    random_state=42, n_jobs=-1, verbose=0
)
rf_search.fit(X_train, y_train)
best_rf_params = rf_search.best_params_
print(f"     Best params : {best_rf_params}")
print(f"     Best CV acc : {rf_search.best_score_*100:.2f}%")

# — Tune XGBoost —
print("\n  [2/3] Tuning XGBoost...")
xgb_params = {
    "n_estimators"  : [50, 100, 200],
    "max_depth"     : [3, 5, 7],
    "learning_rate" : [0.01, 0.05, 0.1, 0.2],
    "subsample"     : [0.7, 0.8, 1.0],
    "colsample_bytree": [0.7, 0.8, 1.0],
}
xgb_search = RandomizedSearchCV(
    XGBClassifier(random_state=42, use_label_encoder=False,
                  eval_metric="logloss", n_jobs=-1),
    xgb_params, n_iter=10, cv=3, scoring="accuracy",
    random_state=42, n_jobs=-1, verbose=0
)
xgb_search.fit(X_train, y_train)
best_xgb_params = xgb_search.best_params_
print(f"     Best params : {best_xgb_params}")
print(f"     Best CV acc : {xgb_search.best_score_*100:.2f}%")

# — Tune SVM —
print("\n  [3/3] Tuning SVM...")
svm_params = {
    "C"      : [0.1, 1, 10, 100],
    "kernel" : ["rbf", "linear"],
    "gamma"  : ["scale", "auto"],
}
svm_search = RandomizedSearchCV(
    SVC(probability=True, random_state=42),
    svm_params, n_iter=8, cv=3, scoring="accuracy",
    random_state=42, n_jobs=-1, verbose=0
)
svm_search.fit(X_train, y_train)
best_svm_params = svm_search.best_params_
print(f"     Best params : {best_svm_params}")
print(f"     Best CV acc : {svm_search.best_score_*100:.2f}%")

# Save tuning results
tuning_df = pd.DataFrame([
    {"Classifier": "Random Forest", "Best Params": str(best_rf_params),
     "CV Accuracy": round(rf_search.best_score_*100, 2)},
    {"Classifier": "XGBoost",       "Best Params": str(best_xgb_params),
     "CV Accuracy": round(xgb_search.best_score_*100, 2)},
    {"Classifier": "SVM",           "Best Params": str(best_svm_params),
     "CV Accuracy": round(svm_search.best_score_*100, 2)},
])
print(f"\n  ✅ Tuning complete!")

# ============================================================
# STEP 7 — Define All 11 Classifiers  (A3)
# Uses best params from RandomizedSearchCV where applicable
# ============================================================
classifiers = {
    "Logistic Regression"  : LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1),
    "SVM"                  : SVC(**best_svm_params, probability=True, random_state=42),
    "Random Forest"        : RandomForestClassifier(**best_rf_params, random_state=42, n_jobs=-1),
    "Decision Tree"        : DecisionTreeClassifier(random_state=42),
    "KNN"                  : KNeighborsClassifier(n_neighbors=5, n_jobs=-1),
    "Naive Bayes"          : GaussianNB(),
    "Gradient Boosting"    : GradientBoostingClassifier(n_estimators=100, random_state=42),
    "XGBoost"              : XGBClassifier(**best_xgb_params, random_state=42,
                                           use_label_encoder=False,
                                           eval_metric="logloss", n_jobs=-1),
    "CatBoost"             : CatBoostClassifier(iterations=100, random_seed=42, verbose=0),
    "AdaBoost"             : AdaBoostClassifier(n_estimators=100, random_state=42),
    "MLP (Neural Network)" : MLPClassifier(hidden_layer_sizes=(256, 128),
                                           max_iter=300, random_state=42),
}

# ============================================================
# STEP 8 — Train & Evaluate All 11 (Train + Test metrics)
# ============================================================
print(f"\n{'='*65}")
print(f"  🤖 A3 — Training 11 Classifiers on {MODEL_NAME} Embeddings")
print(f"{'='*65}")

results = []

for idx, (clf_name, clf) in enumerate(classifiers.items(), 1):
    print(f"\n  [{idx}/11] {clf_name}...")
    start = time.time()

    clf.fit(X_train, y_train)

    # Train metrics
    y_train_pred = clf.predict(X_train)
    train_acc    = accuracy_score(y_train, y_train_pred) * 100
    train_f1     = f1_score(y_train, y_train_pred, average="weighted") * 100

    # Test metrics
    y_test_pred = clf.predict(X_test)
    test_acc    = accuracy_score(y_test, y_test_pred) * 100
    test_f1     = f1_score(y_test, y_test_pred, average="weighted") * 100
    test_prec   = precision_score(y_test, y_test_pred, average="weighted") * 100
    test_rec    = recall_score(y_test, y_test_pred, average="weighted") * 100

    elapsed = time.time() - start

    results.append({
        "Classifier"      : clf_name,
        "Train Accuracy"  : round(train_acc, 2),
        "Test Accuracy"   : round(test_acc, 2),
        "Train F1"        : round(train_f1, 2),
        "Test F1"         : round(test_f1, 2),
        "Precision"       : round(test_prec, 2),
        "Recall"          : round(test_rec, 2),
        "Time (s)"        : round(elapsed, 1),
    })

    # Overfitting check
    gap = train_acc - test_acc
    flag = "⚠️  Overfitting" if gap > 5 else "✅"
    print(f"     Train Acc : {train_acc:.2f}%  |  Test Acc : {test_acc:.2f}%  {flag}")
    print(f"     Test F1   : {test_f1:.2f}%   |  Precision: {test_prec:.2f}%  |  Recall: {test_rec:.2f}%")
    print(f"     Time      : {elapsed:.1f}s")

# ============================================================
# STEP 9 — Results Table
# ============================================================
results_df = pd.DataFrame(results).sort_values("Test Accuracy", ascending=False)
results_df.index = range(1, len(results_df) + 1)

print(f"\n{'='*65}")
print(f"  📊 Final Results — {MODEL_NAME}")
print(f"{'='*65}")
print(results_df.to_string())
print(f"\n🏆 Best Classifier : {results_df.iloc[0]['Classifier']}")
print(f"   Train Accuracy  : {results_df.iloc[0]['Train Accuracy']}%")
print(f"   Test Accuracy   : {results_df.iloc[0]['Test Accuracy']}%")
print(f"   Test F1 Score   : {results_df.iloc[0]['Test F1']}%")

# ============================================================
# STEP 10 — Per-Language Accuracy (Best Classifier)
# ============================================================
print(f"\n{'='*65}")
print(f"  🌐 Per-Language Results — {results_df.iloc[0]['Classifier']}")
print(f"{'='*65}")

best_clf = list(classifiers.values())[
    list(classifiers.keys()).index(results_df.iloc[0]["Classifier"])
]

lang_results = []
for lang in ["Gujarati", "Hindi", "Marathi", "Telugu"]:
    mask = df["language"] == lang
    X_l, y_l = X[mask], y[mask]
    _, X_lt, _, y_lt = train_test_split(
        X_l, y_l, test_size=0.2, random_state=42, stratify=y_l
    )
    y_lp     = best_clf.predict(X_lt)
    lang_acc = accuracy_score(y_lt, y_lp) * 100
    lang_f1  = f1_score(y_lt, y_lp, average="weighted") * 100
    lang_results.append({"Language": lang,
                          "Accuracy": round(lang_acc, 2),
                          "F1 Score": round(lang_f1, 2)})
    print(f"  {lang:<12}  Accuracy={lang_acc:.2f}%  F1={lang_f1:.2f}%")

lang_df = pd.DataFrame(lang_results)

# ============================================================
# STEP 11 — Charts
# ============================================================

# Chart 1 — Train vs Test Accuracy
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

x     = np.arange(len(results_df))
width = 0.35
ax1   = axes[0]
bars1 = ax1.bar(x - width/2, results_df["Train Accuracy"], width,
                label="Train Accuracy", color="steelblue", alpha=0.85)
bars2 = ax1.bar(x + width/2, results_df["Test Accuracy"],  width,
                label="Test Accuracy",  color="coral",     alpha=0.85)
ax1.set_xlabel("Classifier")
ax1.set_ylabel("Accuracy (%)")
ax1.set_title(f"Train vs Test Accuracy — {MODEL_NAME}")
ax1.set_xticks(x)
ax1.set_xticklabels(results_df["Classifier"], rotation=45, ha="right", fontsize=8)
ax1.legend()
ax1.set_ylim([50, 105])
for bar in bars1:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)
for bar in bars2:
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{bar.get_height():.1f}", ha="center", va="bottom", fontsize=7)

# Chart 2 — Per-language accuracy
ax2 = axes[1]
colors_lang = ["#4CAF50", "#2196F3", "#FF9800", "#E91E63"]
bars3 = ax2.bar(lang_df["Language"], lang_df["Accuracy"],
                color=colors_lang, alpha=0.85)
ax2.set_xlabel("Language")
ax2.set_ylabel("Accuracy (%)")
ax2.set_title(f"Per-Language Accuracy — {MODEL_NAME}\n({results_df.iloc[0]['Classifier']})")
ax2.set_ylim([50, 105])
for bar, val in zip(bars3, lang_df["Accuracy"]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
             f"{val}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

plt.tight_layout()
chart_name = f"/content/results_{MODEL_NAME}.png"
plt.savefig(chart_name, dpi=150)
plt.show()

# Chart 3 — All metrics heatmap
fig2, ax3 = plt.subplots(figsize=(12, 7))
heat_data = results_df[["Classifier", "Train Accuracy", "Test Accuracy",
                         "Train F1", "Test F1", "Precision", "Recall"]].set_index("Classifier")
sns.heatmap(heat_data, annot=True, fmt=".1f", cmap="YlOrRd",
            linewidths=0.5, ax=ax3, cbar_kws={"label": "Score (%)"})
ax3.set_title(f"All Metrics Heatmap — {MODEL_NAME} Embeddings")
plt.tight_layout()
heatmap_name = f"/content/heatmap_{MODEL_NAME}.png"
plt.savefig(heatmap_name, dpi=150)
plt.show()

# ============================================================
# STEP 12 — Save All Results
# ============================================================
results_csv  = f"/content/results_{MODEL_NAME}.csv"
tuning_csv   = f"/content/tuning_{MODEL_NAME}.csv"
lang_csv     = f"/content/lang_results_{MODEL_NAME}.csv"

results_df.to_csv(results_csv, index=False)
tuning_df.to_csv(tuning_csv,   index=False)
lang_df.to_csv(lang_csv,       index=False)

print(f"\n💾 Saved:")
print(f"   {results_csv}")
print(f"   {tuning_csv}")
print(f"   {lang_csv}")
print(f"   {chart_name}")
print(f"   {heatmap_name}")

# STEP 13 — Download everything
from google.colab import files
files.download(results_csv)
files.download(tuning_csv)
files.download(lang_csv)
files.download(chart_name)
files.download(heatmap_name)

print("\n✅ All done!")
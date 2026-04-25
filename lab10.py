"""
============================================================
Lab 10 - Feature Reduction & Explainable AI
Subject   : 22AIE213
Dataset   : Multilingual Fake & Real News
            (Hindi + Marathi + Gujarati + Telugu)
============================================================
"""

# 0. Imports
import os, random, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, chi2, SequentialFeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline

import lime
import lime.lime_text
import shap

warnings.filterwarnings("ignore")
random.seed(42)
np.random.seed(42)

# 1. DATA LOADING
# Update these paths to point to your extracted folders
LANGUAGE_DIRS = {
    "Hindi":    ("./Hindi_fake_news",    "./Hindi_real_news"),
    "Marathi":  ("./Marathi_fake_news",  "./Marathi_real_news"),
    "Gujarati": ("./Gujarati_fake_news", "./Gujarati_real_news"),
    "Telugu":   ("./Telugu_fake_news",   "./Telugu_real_news"),
}
SAMPLE_PER_CLASS_PER_LANG = 500   # 500 fake + 500 real per language = 4000 total


def load_documents(directory, label, n, lang):
    """Read n random .txt files; return (texts, labels, languages)."""
    files = random.sample(sorted(Path(directory).glob("*.txt")), n)
    texts, labels, langs = [], [], []
    for fp in files:
        text = fp.read_text(encoding="utf-8", errors="ignore").strip()
        if text:
            texts.append(text)
            labels.append(label)
            langs.append(lang)
    return texts, labels, langs


def load_dataset(sample_per_class=SAMPLE_PER_CLASS_PER_LANG):
    """Load balanced multilingual corpus; return shuffled DataFrame."""
    all_texts, all_labels, all_langs = [], [], []
    for lang, (fd, rd) in LANGUAGE_DIRS.items():
        ft, fl, fln = load_documents(fd, 0, sample_per_class, lang)
        rt, rl, rln = load_documents(rd, 1, sample_per_class, lang)
        all_texts  += ft + rt
        all_labels += fl + rl
        all_langs  += fln + rln
        print(f"{lang}: {len(ft)} fake + {len(rt)} real loaded")
    df = pd.DataFrame({"text": all_texts, "label": all_labels, "language": all_langs})
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"\nTotal dataset: {len(df)} docs | Fake={sum(df.label==0)} | Real={sum(df.label==1)}")
    return df


df = load_dataset()

# 2. TF-IDF FEATURE EXTRACTION using character n-grams (language agnostic)
def extract_tfidf(df, max_features=500):
    """
    Fit character-level TF-IDF (trigrams to 5-grams).
    Using char_wb analyzer makes this work across all four scripts
    (Devanagari, Gujarati, Telugu) without needing language-specific tokenizers.
    """
    vec = TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=True,
        min_df=3,
        analyzer="char_wb",
        ngram_range=(3, 5)
    )
    X = vec.fit_transform(df["text"]).toarray().astype("float32")
    y = df["label"].values
    print(f"TF-IDF feature matrix shape: {X.shape}")
    return X, y, vec


X_full, y_full, vectorizer = extract_tfidf(df, max_features=500)
feature_names = vectorizer.get_feature_names_out()

# Train/test split — keep raw texts for LIME and SHAP
texts    = df["text"].tolist()
langs    = df["language"].tolist()
texts_tr, texts_te, y_tr, y_te, langs_tr, langs_te = train_test_split(
    texts, y_full, langs, test_size=0.20, random_state=42, stratify=y_full
)
X_tr = vectorizer.transform(texts_tr).toarray().astype("float32")
X_te = vectorizer.transform(texts_te).toarray().astype("float32")


# Shared helper to train and evaluate three classifiers
def evaluate_models(X_train, X_test, y_train, y_test, tag):
    """Train LR, Random Forest, Linear SVM; print and return results DataFrame."""
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=500, random_state=42),
        "Random Forest":       RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42),
        "Linear SVM":          LinearSVC(max_iter=2000, random_state=42),
    }
    rows = []
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        pred = clf.predict(X_test)
        rows.append({
            "Setting":   tag,
            "Model":     name,
            "Accuracy":  round(accuracy_score(y_test, pred) * 100, 2),
            "Precision": round(precision_score(y_test, pred, average="macro") * 100, 2),
            "Recall":    round(recall_score(y_test, pred, average="macro") * 100, 2),
            "F1-Score":  round(f1_score(y_test, pred, average="macro") * 100, 2),
        })
        print(f"  {name:22s}  Acc={rows[-1]['Accuracy']:6.2f}  F1={rows[-1]['F1-Score']:6.2f}")
    return pd.DataFrame(rows)


# A1. FEATURE CORRELATION HEATMAP
def plot_correlation_heatmap(X, feature_names, top_n=30):
    """
    Select top_n features by variance, compute Pearson correlation matrix,
    and display as a heatmap. High correlation between features suggests
    redundancy that PCA or feature selection can exploit.
    """
    top_idx   = np.argsort(X.var(axis=0))[-top_n:]
    X_top     = X[:, top_idx]
    names_top = [feature_names[i] for i in top_idx]
    corr      = np.corrcoef(X_top.T)
    fig, ax   = plt.subplots(figsize=(14, 12))
    sns.heatmap(corr, xticklabels=names_top, yticklabels=names_top,
                cmap="coolwarm", center=0, linewidths=0.3, ax=ax,
                cbar_kws={"label": "Pearson Correlation"})
    ax.set_title(
        f"Fig 1: Feature Correlation Heatmap - Top-{top_n} Char N-gram TF-IDF Features\n"
        "Multilingual Dataset: Hindi + Marathi + Gujarati + Telugu",
        fontsize=11
    )
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)
    plt.tight_layout()
    plt.savefig("A1_correlation_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    return corr


print("\n[A1] Feature Correlation Heatmap")
corr_matrix = plot_correlation_heatmap(X_full, feature_names, top_n=30)


# A2. PCA with 99% explained variance
def apply_pca_and_plot(X_train, X_test, threshold, label):
    """
    Fit PCA retaining `threshold` fraction of total variance.
    Plot cumulative sum of explained variance ratios to visualise how many
    components are needed to reach the threshold.
    """
    pca     = PCA(n_components=threshold, random_state=42)
    Xtr_pca = pca.fit_transform(X_train)
    Xte_pca = pca.transform(X_test)
    n       = Xtr_pca.shape[1]
    cumvar  = np.cumsum(pca.explained_variance_ratio_) * 100
    print(f"  {label}: {X_train.shape[1]} features -> {n} components "
          f"(retains {cumvar[-1]:.2f}% variance)")
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(cumvar) + 1), cumvar, linewidth=2)
    ax.axhline(threshold * 100, color="red", linestyle="--",
               label=f"{threshold*100:.0f}% threshold")
    ax.fill_between(range(1, len(cumvar) + 1), cumvar, alpha=0.10)
    ax.set_xlabel("Number of PCA Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title(f"Fig: PCA Cumulative Variance - {threshold*100:.0f}% Threshold\n"
                 "Multilingual Fake News Dataset")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{label}_cumvar.png", dpi=150)
    plt.show()
    return Xtr_pca, Xte_pca, n


print("\n[Baseline] Full TF-IDF (500 features)")
r_base = evaluate_models(X_tr, X_te, y_tr, y_te, "Baseline (500 feat)")

print("\n[A2] PCA - 99% explained variance")
Xtr99, Xte99, n99 = apply_pca_and_plot(X_tr, X_te, 0.99, "A2_PCA99")
r_99 = evaluate_models(Xtr99, Xte99, y_tr, y_te, f"PCA 99% ({n99} comp)")

# A3. PCA with 95% explained variance
print("\n[A3] PCA - 95% explained variance")
Xtr95, Xte95, n95 = apply_pca_and_plot(X_tr, X_te, 0.95, "A3_PCA95")
r_95 = evaluate_models(Xtr95, Xte95, y_tr, y_te, f"PCA 95% ({n95} comp)")


# A4. Sequential Feature Selection
def run_feature_selection(X_train, X_test, y_train, y_test, k=50, n_sfs=20):
    """
    Two-stage feature selection pipeline:
      Stage 1 - SelectKBest (chi2): fast univariate statistical filter.
               Selects k features most statistically associated with the label.
      Stage 2 - SequentialFeatureSelector (forward): greedy wrapper method.
               Iteratively adds the feature that improves CV score the most.
    Returns result DataFrames and the fitted chi2 selector.
    """
    # Stage 1: Chi2 filter
    skb     = SelectKBest(chi2, k=k)
    Xtr_skb = skb.fit_transform(X_train, y_train)
    Xte_skb = skb.transform(X_test)
    print(f"\n[A4] SelectKBest chi2 (k={k})")
    r_skb = evaluate_models(Xtr_skb, Xte_skb, y_train, y_test, f"SelectKBest chi2 (k={k})")

    # Stage 2: Sequential Forward Selection
    lr  = LogisticRegression(max_iter=300, random_state=42)
    sfs = SequentialFeatureSelector(
        lr, n_features_to_select=n_sfs, direction="forward", cv=3, n_jobs=-1
    )
    sfs.fit(Xtr_skb, y_train)
    Xtr_sfs = sfs.transform(Xtr_skb)
    Xte_sfs = sfs.transform(Xte_skb)
    print(f"\n[A4] Sequential Forward Selection ({n_sfs} features from chi2-{k} space)")
    r_sfs = evaluate_models(Xtr_sfs, Xte_sfs, y_train, y_test, f"SFS forward ({n_sfs} feat)")
    return r_skb, r_sfs, skb


r_skb, r_sfs, skb_fitted = run_feature_selection(X_tr, X_te, y_tr, y_te)


# A5. LIME Explanations
def explain_with_lime(vectorizer, texts_train, texts_test, y_train, y_test, langs_test):
    """
    LIME (Local Interpretable Model-agnostic Explanations):
    Perturbs individual news articles by randomly masking character n-gram tokens,
    then fits a local linear model to approximate the classifier behaviour.
    Shows which n-grams most influenced each individual prediction.
    One sample selected per language for comparison.
    """
    pipe = Pipeline([
        ("tfidf", vectorizer),
        ("clf",   LogisticRegression(max_iter=500, random_state=42))
    ])
    pipe.fit(texts_train, y_train)
    explainer = lime.lime_text.LimeTextExplainer(class_names=["Fake", "Real"])
    df_te = pd.DataFrame({"text": texts_test, "label": y_test, "lang": langs_test})

    for lang in ["Hindi", "Marathi", "Gujarati", "Telugu"]:
        idx    = df_te[df_te.lang == lang].index[5]
        text   = texts_test[idx]
        true_l = "Real" if y_test[idx] == 1 else "Fake"
        pred_l = "Real" if pipe.predict([text])[0] == 1 else "Fake"
        exp    = explainer.explain_instance(text, pipe.predict_proba,
                                            num_features=10, num_samples=300)
        fig = exp.as_pyplot_figure()
        fig.suptitle(f"Fig: LIME - {lang} Sample | True: {true_l} | Predicted: {pred_l}",
                     fontsize=10)
        fig.tight_layout()
        plt.savefig(f"A5_lime_{lang.lower()}.png", dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  LIME {lang}: true={true_l}, predicted={pred_l}")
    return pipe


print("\n[A5] LIME Explanations (one sample per language)")
lime_pipe = explain_with_lime(vectorizer, texts_tr, texts_te, y_tr, y_te, langs_te)


# A5. SHAP Explanations
def explain_with_shap(vectorizer, texts_train, texts_test, y_train):
    """
    SHAP (SHapley Additive exPlanations) using LinearExplainer:
    Computes exact Shapley values for a Logistic Regression model.
    Unlike LIME (local, perturbation-based), SHAP provides globally consistent
    feature attributions grounded in cooperative game theory.
    Produces a bar chart (global importance) and beeswarm plot (distribution).
    """
    X_tr_sp    = vectorizer.transform(texts_train)
    X_te_sp    = vectorizer.transform(texts_test)
    feat_names = vectorizer.get_feature_names_out()

    lr = LogisticRegression(max_iter=500, random_state=42)
    lr.fit(X_tr_sp, y_train)

    masker    = shap.maskers.Independent(X_tr_sp[:300])
    explainer = shap.LinearExplainer(lr, masker=masker)
    shap_vals = explainer.shap_values(X_te_sp[:200])
    print(f"  SHAP values shape: {shap_vals.shape}")

    # Bar chart: mean absolute SHAP per feature
    mean_shap = np.abs(shap_vals).mean(axis=0)
    top20     = np.argsort(mean_shap)[-20:][::-1]
    colors    = ["#E74C3C" if lr.coef_[0][i] < 0 else "#2ECC71" for i in top20]
    fig, ax   = plt.subplots(figsize=(11, 7))
    ax.barh([repr(feat_names[i]) for i in top20], mean_shap[top20],
            color=colors, edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title(
        "Fig: SHAP Feature Importance - Top-20 Char N-gram Features\n"
        "Green = Real News signal  |  Red = Fake News signal\n"
        "Multilingual: Hindi + Marathi + Gujarati + Telugu",
        fontsize=10
    )
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("A5_shap_bar.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Beeswarm: distribution of SHAP values per feature
    shap.summary_plot(shap_vals, X_te_sp[:200],
                      feature_names=[repr(f) for f in feat_names],
                      show=True, max_display=20)
    plt.title("Fig: SHAP Summary (Beeswarm) - Multilingual Fake News Detection", fontsize=10)
    plt.tight_layout()
    plt.savefig("A5_shap_beeswarm.png", dpi=150, bbox_inches="tight")


print("\n[A5] SHAP Explanations")
explain_with_shap(vectorizer, texts_tr, texts_te, y_tr)

# Combined Results Table
df_results = pd.concat([r_base, r_99, r_95, r_skb, r_sfs], ignore_index=True)
print("\n" + "=" * 70)
print("Table 1: Classification Performance - All Feature Settings")
print("         Multilingual Fake News: Hindi + Marathi + Gujarati + Telugu")
print("=" * 70)
print(df_results.to_string(index=False))
df_results.to_csv("results_table.csv", index=False)

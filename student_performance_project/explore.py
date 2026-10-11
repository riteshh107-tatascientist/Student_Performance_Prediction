"""Create reproducible EDA charts plus an unsupervised PCA/K-Means exercise."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from src.pipeline import load_data, ARTIFACTS

OUT = ARTIFACTS / "figures"
OUT.mkdir(parents=True, exist_ok=True)

def main():
    data = load_data()
    sns.set_theme()
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.histplot(data=data, x="G3", bins=11, kde=True, ax=ax)
    ax.set_title("Distribution of final Mathematics grades")
    fig.tight_layout(); fig.savefig(OUT / "grade_distribution.png", dpi=160); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.boxplot(data=data, x="studytime", y="G3", ax=ax)
    ax.set_title("Final grade by weekly study-time category")
    fig.tight_layout(); fig.savefig(OUT / "grade_by_studytime.png", dpi=160); plt.close(fig)

    numeric = data.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(numeric.corr(), cmap="vlag", center=0, ax=ax)
    ax.set_title("Numeric feature correlation (descriptive, not causal)")
    fig.tight_layout(); fig.savefig(OUT / "correlation_heatmap.png", dpi=160); plt.close(fig)

    # Unsupervised learning is exploratory only; clusters are not student labels.
    X = data.drop(columns=["G1", "G2", "G3"], errors="ignore")
    cat = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    num = [c for c in X.columns if c not in cat]
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]), cat),
    ])
    Z = pre.fit_transform(X)
    pca = PCA(n_components=2, random_state=42)
    reduced = pca.fit_transform(Z)
    clusters = KMeans(n_clusters=3, random_state=42, n_init=20).fit_predict(Z)
    plot_df = pd.DataFrame({"PC1": reduced[:, 0], "PC2": reduced[:, 1], "cluster": clusters.astype(str)})
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(data=plot_df, x="PC1", y="PC2", hue="cluster", alpha=.8, ax=ax)
    ax.set_title("Exploratory PCA projection colored by K-Means cluster")
    fig.tight_layout(); fig.savefig(OUT / "pca_kmeans.png", dpi=160); plt.close(fig)
    print(f"Saved EDA and PCA/K-Means plots to {OUT}")
    print(f"PCA variance explained by two components: {pca.explained_variance_ratio_.sum():.3f}")
    print("Reminder: clusters are exploratory groupings, not validated student categories.")

if __name__ == "__main__":
    main()

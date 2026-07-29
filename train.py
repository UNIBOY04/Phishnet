"""
train.py
--------
Trains and compares ML models to classify URLs as 'phishing' or 'legitimate'.

Input data: a CSV with at least two columns:
    url   -> the URL string
    label -> 1 for phishing, 0 for legitimate  (or 'phishing'/'legitimate' text)

Public datasets you can use (free):
  - PhishTank (phishing URLs, feed): https://phishtank.org/developer_info.php
  - Kaggle "Phishing Site URLs": https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls
  - UCI Phishing Websites dataset: https://archive.ics.uci.edu/dataset/327/phishing+websites

Usage:
    python train.py --data phishing_urls.csv
    python train.py --synthetic          # test the pipeline with generated data
    python train.py --data phishing_urls.csv --fetch-content   # adds live page features (slow, needs network)
"""

import argparse
import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)

from feature_extraction import extract_features_dataframe, extract_content_features


def make_synthetic_dataset(n=4000, seed=42):
    """Generates believable phishing vs. legitimate URLs to test the pipeline."""
    rng = np.random.default_rng(seed)

    legit_domains = [
        "github.com", "google.com", "wikipedia.org", "amazon.com", "nytimes.com",
        "stackoverflow.com", "microsoft.com", "apple.com", "reddit.com", "bbc.com",
    ]
    legit_paths = ["/", "/about", "/products", "/blog/post-1", "/docs/api", "/search?q=test"]

    phishy_hosts = [
        "paypal-secure-login.com", "verify-account-update.net", "192.168.44.10",
        "amaz0n-billing.support", "apple-id-confirm.xyz", "secure-bankofamerica.info",
        "login-microsoft365.tk", "bit.ly", "account-suspended-alert.com",
        "192.168.1.5", "signin-google-verify.co",
    ]
    phishy_paths = [
        "/login/verify", "/account/update?id=8891", "/webscr?cmd=login",
        "/secure/confirm-password", "//redirect/login", "/signin?next=http://evil.com",
    ]

    def make_legit(count):
        rows = []
        for _ in range(count):
            d = rng.choice(legit_domains)
            p = rng.choice(legit_paths)
            rows.append(f"https://www.{d}{p}")
        return rows

    def make_phishing(count):
        rows = []
        for _ in range(count):
            h = rng.choice(phishy_hosts)
            p = rng.choice(phishy_paths)
            scheme = rng.choice(["http", "https"])
            rows.append(f"{scheme}://{h}{p}")
        return rows

    n_phish = n // 2
    n_legit = n - n_phish
    urls = make_legit(n_legit) + make_phishing(n_phish)
    labels = [0] * n_legit + [1] * n_phish

    df = pd.DataFrame({"url": urls, "label": labels})
    return df.sample(frac=1, random_state=seed).reset_index(drop=True)


def normalize_labels(df, label_col="label"):
    df = df.copy()
    if df[label_col].dtype == object:
        df[label_col] = df[label_col].astype(str).str.lower().map(
            lambda x: 1 if x in ("phishing", "bad", "malicious", "1", "true") else 0
        )
    return df


def build_features(df, fetch_content=False):
    feat_df = extract_features_dataframe(df["url"])
    if fetch_content:
        print("Fetching live page content for each URL (this can be slow)...")
        content_rows = []
        for i, url in enumerate(df["url"]):
            content_rows.append(extract_content_features(url))
            if (i + 1) % 50 == 0:
                print(f"  fetched {i + 1}/{len(df)}")
        content_df = pd.DataFrame(content_rows)
        feat_df = pd.concat([feat_df.reset_index(drop=True), content_df.reset_index(drop=True)], axis=1)
    return feat_df


def get_models():
    return {
        "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "DecisionTree": DecisionTreeClassifier(max_depth=15, random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "SVM": SVC(kernel="rbf", probability=True, random_state=42),
    }


def evaluate_model(name, model, X_test, y_test, out_dir):
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"\n--- {name} ---")
    print(f"Accuracy : {acc:.4f}  Precision: {prec:.4f}  Recall: {rec:.4f}  F1: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["legitimate", "phishing"]))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(4, 3.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Reds",
                xticklabels=["legitimate", "phishing"], yticklabels=["legitimate", "phishing"])
    plt.title(f"Confusion Matrix - {name}")
    plt.ylabel("True")
    plt.xlabel("Predicted")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"confusion_matrix_{name}.png"), dpi=150)
    plt.close()

    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def plot_roc_curves(trained_models, X_test, y_test, out_dir):
    plt.figure(figsize=(6, 5))
    for name, model in trained_models.items():
        y_score = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.decision_function(X_test)
        fpr, tpr, _ = roc_curve(y_test, y_score)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc:.3f})")
    plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves - Phishing Detection Models")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "roc_curves.png"), dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train ML-based phishing URL detector")
    parser.add_argument("--data", type=str, help="CSV with 'url' and 'label' columns")
    parser.add_argument("--synthetic", action="store_true", help="Use generated synthetic data")
    parser.add_argument("--fetch-content", action="store_true",
                         help="Also fetch live page content features (slower, needs network)")
    parser.add_argument("--out_dir", type=str, default="artifacts")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.synthetic or not args.data:
        print("Using synthetic dataset (pass --data <path.csv> to use a real dataset).")
        df = make_synthetic_dataset(4000)
    else:
        df = pd.read_csv(args.data)

    df = normalize_labels(df)

    print(f"Loaded {len(df)} URLs ({df['label'].sum()} phishing / {(df['label']==0).sum()} legitimate)")

    feat_df = build_features(df, fetch_content=args.fetch_content)
    feature_names = list(feat_df.columns)

    scaler = StandardScaler()
    X = scaler.fit_transform(feat_df.values)
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    joblib.dump(scaler, os.path.join(args.out_dir, "scaler.pkl"))
    joblib.dump(feature_names, os.path.join(args.out_dir, "feature_names.pkl"))
    joblib.dump(args.fetch_content, os.path.join(args.out_dir, "used_content_features.pkl"))

    models = get_models()
    trained = {}
    results = []
    for name, model in models.items():
        print(f"\nTraining {name} ...")
        model.fit(X_train, y_train)
        trained[name] = model
        results.append(evaluate_model(name, model, X_test, y_test, args.out_dir))

    plot_roc_curves(trained, X_test, y_test, args.out_dir)

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    print("\n=== Model comparison (sorted by F1) ===")
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(args.out_dir, "model_comparison.csv"), index=False)

    best_name = results_df.iloc[0]["model"]
    best_model = trained[best_name]
    joblib.dump(best_model, os.path.join(args.out_dir, "best_model.pkl"))
    joblib.dump(best_name, os.path.join(args.out_dir, "best_model_name.pkl"))

    if hasattr(best_model, "feature_importances_"):
        importances = pd.Series(best_model.feature_importances_, index=feature_names)
        importances = importances.sort_values(ascending=False).head(15)
        plt.figure(figsize=(7, 5))
        importances.plot(kind="barh")
        plt.gca().invert_yaxis()
        plt.title(f"Top 15 Feature Importances - {best_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, "feature_importance.png"), dpi=150)
        plt.close()

    print(f"\nBest model: {best_name} (saved to {args.out_dir}/best_model.pkl)")


if __name__ == "__main__":
    main()
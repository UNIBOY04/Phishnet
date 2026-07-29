"""
predict.py
----------
Loads the trained phishing-detection model and classifies new URLs.

Usage:
    python predict.py --url "http://paypal-secure-login.com/verify"
    python predict.py --input urls.csv --out_dir artifacts --output predictions.csv
"""

import argparse
import os
import joblib
import pandas as pd

from feature_extraction import extract_features_dataframe, extract_content_features


def load_artifacts(out_dir="artifacts"):
    model = joblib.load(os.path.join(out_dir, "best_model.pkl"))
    model_name = joblib.load(os.path.join(out_dir, "best_model_name.pkl"))
    scaler = joblib.load(os.path.join(out_dir, "scaler.pkl"))
    feature_names = joblib.load(os.path.join(out_dir, "feature_names.pkl"))
    used_content = joblib.load(os.path.join(out_dir, "used_content_features.pkl"))
    return model, model_name, scaler, feature_names, used_content


def predict_urls(urls, out_dir="artifacts", fetch_content=None):
    model, model_name, scaler, feature_names, used_content = load_artifacts(out_dir)
    if fetch_content is None:
        fetch_content = used_content

    df = pd.DataFrame({"url": urls})
    feat_df = extract_features_dataframe(df["url"])

    if fetch_content:
        content_rows = [extract_content_features(u) for u in df["url"]]
        content_df = pd.DataFrame(content_rows)
        feat_df = pd.concat([feat_df.reset_index(drop=True), content_df.reset_index(drop=True)], axis=1)

    # Align columns with training-time feature set
    for col in feature_names:
        if col not in feat_df.columns:
            feat_df[col] = 0
    feat_df = feat_df[feature_names]

    X = scaler.transform(feat_df.values)
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1] if hasattr(model, "predict_proba") else [None] * len(preds)

    result = df.copy()
    result["prediction"] = ["phishing" if p == 1 else "legitimate" for p in preds]
    result["phishing_probability"] = probs
    return result


def main():
    parser = argparse.ArgumentParser(description="Run phishing detection on URLs")
    parser.add_argument("--url", type=str, help="A single URL to check")
    parser.add_argument("--input", type=str, help="CSV file with a 'url' column")
    parser.add_argument("--out_dir", type=str, default="artifacts")
    parser.add_argument("--output", type=str, default="predictions.csv")
    parser.add_argument("--fetch-content", action="store_true",
                         help="Force fetching live page content (overrides training-time setting)")
    args = parser.parse_args()

    if args.url:
        result = predict_urls([args.url], args.out_dir, fetch_content=args.fetch_content or None)
        row = result.iloc[0]
        print(f"\nURL: {row['url']}")
        print(f"Prediction: {row['prediction'].upper()}")
        print(f"Phishing probability: {row['phishing_probability']:.4f}")
    elif args.input:
        df = pd.read_csv(args.input)
        result = predict_urls(df["url"].tolist(), args.out_dir, fetch_content=args.fetch_content or None)
        result.to_csv(args.output, index=False)
        n_phish = (result["prediction"] == "phishing").sum()
        print(f"Processed {len(result)} URLs: {n_phish} flagged as phishing.")
        print(f"Results saved to {args.output}")
    else:
        parser.error("Provide either --url or --input")


if __name__ == "__main__":
    main()
    


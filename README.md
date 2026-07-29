# AI-Based Phishing URL Detection

A complete, working phishing detector that classifies URLs as **phishing**
or **legitimate** using engineered features (URL structure, domain
characteristics, obfuscation signals) fed into classic ML models. Good
as-is for a college/portfolio cybersecurity project — no API keys required
for the core version.

## Project structure

```
phishing_project/
├── feature_extraction.py   # turns a raw URL into ~25 numeric features
├── train.py                  # trains + compares 5 models, saves the best one
├── predict.py                # classifies a single URL or a CSV of URLs
├── requirements.txt
└── artifacts/                 # created after training: model + plots + metrics
```

## 1. Setup

```bash
pip install -r requirements.txt
```

## 2. How it works

`feature_extraction.py` converts each raw URL string into features like:

- **Length features**: URL length, host length, path length
- **Character composition**: number of dots, hyphens, `@` symbols, digits, `%` encoding
- **Structural red flags**: IP address used as host, missing HTTPS, `@` symbol
  (browser trick to hide the real destination), URL shorteners (bit.ly etc.)
- **Suspicious keywords**: "verify", "login", "secure", "confirm", "suspend", etc.
- **Entropy**: how random/gibberish the domain looks (phishing domains often
  look machine-generated, e.g. `xk29-secure-paypal.info`)
- **Brand impersonation signal**: a well-known brand name stuffed into a
  subdomain of an unrelated domain (e.g. `paypal.login-verify.ru`)

All of this works fully **offline** — no network calls needed. There's also
an optional content-based feature set (`--fetch-content`) that actually
visits the page and checks for a password field, external form submission
targets, etc. — this is slower and needs internet access, so it's opt-in.

## 3. Get data

You have three options:

**Option A — test immediately with synthetic data (no download needed):**
```bash
python train.py --synthetic
```

**Option B — use a free public dataset.** Any CSV with a `url` column and a
`label` column (`1`/`phishing` = phishing, `0`/`legitimate` = safe) works:
- Kaggle "Phishing Site URLs": https://www.kaggle.com/datasets/taruntiwarihp/phishing-site-urls
- UCI Phishing Websites dataset: https://archive.ics.uci.edu/dataset/327/phishing+websites
- PhishTank live feed (phishing examples only — pair with a list of top
  legitimate sites, e.g. Tranco or Alexa top 1M, for the "legitimate" class):
  https://phishtank.org/developer_info.php

**Option C — build your own** by combining a phishing URL list with a
legitimate URL list into one CSV:
```
url,label
http://paypal-secure-login.com/verify,1
https://www.google.com,0
```

## 4. Train

```bash
python train.py --data phishing_urls.csv
# or, with live page-content features too (slower):
python train.py --data phishing_urls.csv --fetch-content
```

This will:
- Extract all features from every URL
- Train 5 models: Random Forest, Gradient Boosting, Decision Tree,
  Logistic Regression, SVM
- Print accuracy / precision / recall / F1 for each
- Save confusion matrices, an ROC curve comparison, and a feature-importance
  chart to `artifacts/`
- Save the best model (by F1) as `artifacts/best_model.pkl`

## 5. Predict

**Single URL:**
```bash
python predict.py --url "http://paypal-secure-login-verify.com/account/update?id=8891"
```
```
URL: http://paypal-secure-login-verify.com/account/update?id=8891
Prediction: PHISHING
Phishing probability: 1.0000
```

**Batch (CSV of URLs):**
```bash
python predict.py --input new_urls.csv --output predictions.csv
```

## Guide: presenting this as a project

**Problem statement**: Phishing URLs try to imitate legitimate sites to
steal credentials. Manually maintained blocklists (like Google Safe
Browsing) can't catch brand-new phishing domains fast enough — an ML
classifier that scores a URL's *structure* can flag suspicious ones even
before they're reported.

**Why feature engineering over raw text**: Feeding the raw URL string into
a model directly (e.g. character-level CNN/LSTM) is a valid alternative,
but hand-crafted features are more interpretable for a project
report/viva — you can point at *why* a URL was flagged, e.g. "IP address
as hostname" or "high domain entropy," which matters a lot to
non-technical reviewers.

**Suggested report structure**:
1. Problem & motivation (phishing statistics, real-world impact)
2. Dataset description (source, class balance)
3. Feature engineering (list the ~25 features, explain 4–5 in depth with
   examples)
4. Model comparison (use `artifacts/model_comparison.csv` and
   `roc_curves.png`)
5. Feature importance analysis (`artifacts/feature_importance.png` —
   discuss which signals mattered most)
6. Limitations & future work (see below)

## Limitations / good "future work" talking points

- **Adversarial evasion**: attackers can craft URLs that look legitimate by
  lexical features alone (e.g. compromising a real domain). Pair this with
  content-based or reputation-based signals (WHOIS domain age, SSL cert
  issuer, hosting ASN) for a stronger system.
- **Concept drift**: phishing tactics evolve; the model should be retrained
  periodically on fresh data.
- **False positives**: legitimate sites using URL shorteners or many query
  parameters can be flagged — tune the classification threshold
  (`phishing_probability > 0.5` by default) based on your
  precision/recall priorities.
- **Extending to email phishing**: apply similar ideas (suspicious
  keywords, sender domain mismatch, urgency language) to raw email text
  using NLP (TF-IDF + classifier, or a fine-tuned transformer).

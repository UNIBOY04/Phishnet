"""
app.py
------
Flask web frontend for the phishing URL detector.
"""

from flask import Flask, render_template, request, jsonify
from predict import predict_urls
from feature_extraction import extract_url_features

app = Flask(__name__)

OUT_DIR = "artifacts"


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


def build_reasons(features):
    """Turns raw lexical features into short, human-readable red/green flags
    so the UI can explain *why* a URL was scored the way it was."""
    reasons = []

    if features["has_ip_address"]:
        reasons.append({"flag": "bad", "text": "Uses a raw IP address instead of a domain name"})
    if not features["has_https"]:
        reasons.append({"flag": "bad", "text": "No HTTPS encryption"})
    if features["has_at_symbol"]:
        reasons.append({"flag": "bad", "text": "Contains an \u201c@\u201d symbol, which can hide the real destination"})
    if features["has_double_slash_redirect"]:
        reasons.append({"flag": "bad", "text": "Path contains a \u201c//\u201d redirect pattern"})
    if features["is_shortened"]:
        reasons.append({"flag": "warn", "text": "Uses a URL-shortening service, which can mask the true link"})
    if features["suspicious_word_count"] > 0:
        n = features["suspicious_word_count"]
        reasons.append({"flag": "bad", "text": f"Contains {n} suspicious keyword{'s' if n != 1 else ''} (e.g. \u201cverify\u201d, \u201clogin\u201d, \u201csecure\u201d)"})
    if features["brand_in_subdomain"]:
        reasons.append({"flag": "bad", "text": "A well-known brand name appears in the subdomain, a common impersonation trick"})
    if features["host_entropy"] > 3.6:
        reasons.append({"flag": "warn", "text": "Domain name looks randomly generated"})
    if features["subdomain_count"] >= 3:
        reasons.append({"flag": "warn", "text": "Unusually many subdomains"})
    if features["num_hyphens"] >= 3:
        reasons.append({"flag": "warn", "text": "Domain contains several hyphens, common in look-alike domains"})

    if not reasons:
        reasons.append({"flag": "good", "text": "No obvious red flags found in the URL structure"})
    elif features["has_https"] and not features["has_ip_address"]:
        reasons.append({"flag": "good", "text": "Uses HTTPS and a normal hostname"})

    return reasons


@app.route("/api/check", methods=["POST"])
def api_check():
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()

    if not url:
        return jsonify({"ok": False, "error": "Please enter a URL."})

    try:
        df_result = predict_urls([url], out_dir=OUT_DIR)
        row = df_result.iloc[0]
        probability_pct = round(float(row["phishing_probability"]) * 100)
        reasons = build_reasons(extract_url_features(url))
        return jsonify({
            "ok": True,
            "url": row["url"],
            "prediction": row["prediction"],
            "probability": probability_pct,
            "reasons": reasons,
        })
    except FileNotFoundError:
        return jsonify({
            "ok": False,
            "error": (
                "No trained model found. Run 'python train.py --synthetic' "
                "(or with real data) first to create the artifacts/ folder."
            ),
        })
    except Exception as e:
        return jsonify({"ok": False, "error": f"Something went wrong: {e}"})


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 5000))
    debug_mode = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host="0.0.0.0", port=port, debug=debug_mode)

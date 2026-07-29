"""
feature_extraction.py
----------------------
Extracts lexical, structural, and (optionally) page-content features from
a URL for phishing detection. No external API keys needed for the core
lexical features — those work fully offline. Content-based features
(--fetch-content in train/predict) do a live HTTP request and are optional.

Feature groups:
  1. Lexical/URL-string features   -> always available, offline
  2. Domain/host features           -> always available, offline
  3. Page content features          -> optional, requires network access
"""

import re
import math
import ipaddress
from urllib.parse import urlparse

import pandas as pd

SUSPICIOUS_WORDS = [
    "login", "verify", "update", "secure", "account", "banking", "confirm",
    "signin", "sign-in", "webscr", "ebayisapi", "paypal", "password",
    "suspend", "urgent", "click", "billing", "invoice", "gift", "bonus",
]

SHORTENING_SERVICES = [
    "bit.ly", "goo.gl", "tinyurl.com", "t.co", "ow.ly", "is.gd", "buff.ly",
    "adf.ly", "bl.ink", "cutt.ly", "rebrand.ly", "shorte.st",
]


def shannon_entropy(s):
    """Higher entropy = more random-looking string (common in phishing/DGA domains)."""
    if not s:
        return 0.0
    probs = [s.count(c) / len(s) for c in set(s)]
    return -sum(p * math.log2(p) for p in probs)


def is_ip_address(host):
    try:
        ipaddress.ip_address(host)
        return True
    except ValueError:
        return False


def extract_url_features(url):
    """Extracts a dict of numeric/binary features for a single URL string."""
    url = str(url).strip()
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url_for_parse = "http://" + url
    else:
        url_for_parse = url

    parsed = urlparse(url_for_parse)
    host = parsed.netloc.split(":")[0].lower()
    path = parsed.path or ""
    query = parsed.query or ""
    full = url.lower()

    subdomain_count = max(host.count(".") - 1, 0) if host else 0
    domain_parts = host.split(".") if host else []

    features = {
        # --- basic length features ---
        "url_length": len(url),
        "host_length": len(host),
        "path_length": len(path),

        # --- character composition ---
        "num_dots": url.count("."),
        "num_hyphens": url.count("-"),
        "num_underscores": url.count("_"),
        "num_slashes": url.count("/"),
        "num_question_marks": url.count("?"),
        "num_equal_signs": url.count("="),
        "num_at_signs": url.count("@"),
        "num_ampersands": url.count("&"),
        "num_digits": sum(c.isdigit() for c in url),
        "num_percent": url.count("%"),

        # --- structural red flags ---
        "has_ip_address": int(is_ip_address(host)),
        "has_https": int(parsed.scheme == "https"),
        "has_at_symbol": int("@" in url),
        "has_double_slash_redirect": int("//" in path),
        "subdomain_count": subdomain_count,
        "is_shortened": int(any(s in host for s in SHORTENING_SERVICES)),

        # --- suspicious keyword presence ---
        "suspicious_word_count": sum(w in full for w in SUSPICIOUS_WORDS),
        "brand_in_subdomain": int(
            len(domain_parts) > 2 and any(
                w in ".".join(domain_parts[:-2]) for w in
                ["paypal", "apple", "amazon", "microsoft", "google", "bank"]
            )
        ),

        # --- randomness / obfuscation signals ---
        "host_entropy": shannon_entropy(host),
        "path_entropy": shannon_entropy(path),

        # --- port / query oddities ---
        "has_port": int(parsed.port is not None),
        "num_query_params": query.count("=") if query else 0,
    }
    return features


def extract_features_dataframe(urls):
    """Takes an iterable of URL strings, returns a feature DataFrame."""
    rows = [extract_url_features(u) for u in urls]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Optional: live page-content features (requires network access)
# ---------------------------------------------------------------------------

def extract_content_features(url, timeout=5):
    """
    Fetches the URL and extracts simple page-content signals:
    - presence of a password input field
    - number of external links vs internal links
    - whether the page has a <form> that posts somewhere off-domain
    Returns a dict; all zeros if the fetch fails (offline / unreachable).
    """
    import requests
    from bs4 import BeautifulSoup

    default = {
        "has_password_field": 0,
        "external_link_ratio": 0.0,
        "form_action_external": 0,
        "fetch_failed": 1,
    }
    try:
        resp = requests.get(url, timeout=timeout, headers={"User-Agent": "Mozilla/5.0"})
        soup = BeautifulSoup(resp.text, "html.parser")
        host = urlparse(url).netloc

        has_password = int(bool(soup.find("input", {"type": "password"})))

        links = soup.find_all("a", href=True)
        if links:
            external = sum(1 for a in links if host not in a["href"] and a["href"].startswith("http"))
            ext_ratio = external / len(links)
        else:
            ext_ratio = 0.0

        forms = soup.find_all("form", action=True)
        form_external = int(any(
            f["action"].startswith("http") and host not in f["action"] for f in forms
        ))

        return {
            "has_password_field": has_password,
            "external_link_ratio": ext_ratio,
            "form_action_external": form_external,
            "fetch_failed": 0,
        }
    except Exception:
        return default

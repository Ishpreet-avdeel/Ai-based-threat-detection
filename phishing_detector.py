"""
phishing_detector.py
--------------------
Supervised machine-learning phishing email detector.

Model:  TfidfVectorizer  ->  LogisticRegression   (one sklearn Pipeline)
Saved:  models/phishing_model.pkl  (via joblib)

Public API:
    train_or_load_model(model_path, train_csv) -> Pipeline
    analyze_emails(df, model=None)             -> DataFrame
    analyze_text(text, model=None)             -> DataFrame (one row)

Input email CSV schema (validated like detector._prepare):
    required: timestamp, sender, subject, body
    optional: recipient, user_id   (passed through to the output if present)

Training CSV schema:
    email_text (subject + body as one string), label (1 = phishing, 0 = legitimate)
"""

import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent

DEFAULT_MODEL_PATH = "models/phishing_model.pkl"
DEFAULT_TRAIN_CSV = "data/phishing_train.csv"

REQUIRED_EMAIL_COLUMNS = ["timestamp", "sender", "subject", "body"]
OPTIONAL_EMAIL_COLUMNS = ["recipient", "user_id"]
OUTPUT_COLUMNS = ["timestamp", "sender", "subject", "prediction", "confidence", "severity", "details"]
TRAIN_COLUMNS = ["email_text", "label"]

# Phishing probability thresholds
CRITICAL_THRESHOLD = 0.85
HIGH_THRESHOLD = 0.60          # also the cut-off for prediction = 1 (phishing)
TOP_KEYWORDS = 5
RANDOM_STATE = 42


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _resolve(path) -> Path:
    """Relative paths are resolved next to this file, so the cwd does not matter."""
    path = Path(path)
    return path if path.is_absolute() else BASE_DIR / path


def _empty_results(extra_columns=()) -> pd.DataFrame:
    return pd.DataFrame(columns=OUTPUT_COLUMNS + list(extra_columns))


def _build_pipeline() -> Pipeline:
    """TF-IDF features (words + word pairs) feeding a logistic-regression classifier."""
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2),
                                  sublinear_tf=True, min_df=2)),
        ("clf", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   random_state=RANDOM_STATE)),
    ])


# --------------------------------------------------------------------------- #
# Training / loading
# --------------------------------------------------------------------------- #
def _ensure_training_csv(train_path: Path) -> None:
    """Auto-generate the training CSV if it does not exist yet."""
    if train_path.exists():
        return
    if str(BASE_DIR) not in sys.path:
        sys.path.insert(0, str(BASE_DIR))
    try:
        from generate_phishing_data import write_training_csv
    except ImportError as exc:
        raise FileNotFoundError(
            f"Training file not found: {train_path}. Run `python generate_phishing_data.py` first."
        ) from exc
    print(f"Training file missing - generating {train_path}")
    write_training_csv(train_path)


def _load_training_data(train_path: Path) -> pd.DataFrame:
    _ensure_training_csv(train_path)
    data = pd.read_csv(train_path)

    missing = [c for c in TRAIN_COLUMNS if c not in data.columns]
    if missing:
        raise ValueError(f"Training CSV is missing column(s): {', '.join(missing)}. "
                         f"Expected: {', '.join(TRAIN_COLUMNS)}.")

    data = data[TRAIN_COLUMNS].dropna()
    data["email_text"] = data["email_text"].astype(str)
    data["label"] = pd.to_numeric(data["label"], errors="coerce")
    data = data.dropna(subset=["label"])
    data["label"] = data["label"].astype(int)

    if set(data["label"].unique()) != {0, 1}:
        raise ValueError("Training CSV needs both classes: label 1 (phishing) and label 0 (legitimate).")
    if len(data) < 20:
        raise ValueError("Training CSV has too few rows to train a model (need at least 20).")
    return data.reset_index(drop=True)


def train_model(train_csv=DEFAULT_TRAIN_CSV) -> Pipeline:
    """Train the pipeline on a labeled CSV and return it (not saved)."""
    data = _load_training_data(_resolve(train_csv))
    texts, labels = data["email_text"], data["label"]

    # Hold-out check so we can report an honest accuracy number
    x_train, x_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, stratify=labels, random_state=RANDOM_STATE)
    check = _build_pipeline().fit(x_train, y_train)
    print(f"Phishing model hold-out accuracy: {check.score(x_test, y_test):.1%} "
          f"({len(x_test)} unseen training-set emails)")

    # Final model uses all of the data
    return _build_pipeline().fit(texts, labels)


def train_or_load_model(model_path=DEFAULT_MODEL_PATH, train_csv=DEFAULT_TRAIN_CSV) -> Pipeline:
    """
    Load the saved model if it exists; otherwise train on `train_csv`
    (auto-generating it if missing), save to `model_path` and return it.
    """
    model_file = _resolve(model_path)

    if model_file.exists():
        try:
            return joblib.load(model_file)
        except Exception as exc:  # corrupt file or incompatible scikit-learn version
            print(f"Could not load {model_file} ({exc}). Retraining...")

    model = train_model(train_csv)
    try:
        model_file.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_file)
    except OSError as exc:
        print(f"Warning: trained model could not be saved to {model_file}: {exc}")
    return model


# --------------------------------------------------------------------------- #
# Explaining a prediction
# --------------------------------------------------------------------------- #
def _top_trigger_terms(model: Pipeline, texts: list, top_k: int = TOP_KEYWORDS) -> list:
    """
    For each text, return the terms that pushed the score toward 'phishing' the most.

    For logistic regression, a term's contribution = its TF-IDF weight in this email
    x its learned coefficient. We keep the largest positive contributions and drop
    overlapping terms (e.g. keep 'verify' rather than also 'verify account').
    """
    tfidf, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    terms = tfidf.get_feature_names_out()
    coef = clf.coef_[0]
    matrix = tfidf.transform(texts).tocsr()

    all_triggers = []
    for i in range(matrix.shape[0]):
        start, end = matrix.indptr[i], matrix.indptr[i + 1]
        cols, weights = matrix.indices[start:end], matrix.data[start:end]
        contributions = weights * coef[cols]

        chosen, used_words = [], set()
        for j in np.argsort(contributions)[::-1]:
            if contributions[j] <= 0:
                break
            term = terms[cols[j]]
            words = set(term.split())
            if words & used_words:  # overlaps an already chosen, stronger term
                continue
            chosen.append(term)
            used_words |= words
            if len(chosen) == top_k:
                break
        all_triggers.append(chosen)
    return all_triggers


# --------------------------------------------------------------------------- #
# Scanning emails
# --------------------------------------------------------------------------- #
def _prepare_emails(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the schema and clean the columns (mirrors detector._prepare)."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("analyze_emails expects a pandas DataFrame.")

    missing = [c for c in REQUIRED_EMAIL_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {', '.join(REQUIRED_EMAIL_COLUMNS)} "
            f"(optional: {', '.join(OPTIONAL_EMAIL_COLUMNS)})."
        )

    keep = REQUIRED_EMAIL_COLUMNS + [c for c in OPTIONAL_EMAIL_COLUMNS if c in df.columns]
    data = df[keep].copy()
    # Unlike log rows, an email with a bad timestamp is still worth scanning
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    for col in ("sender", "subject", "body"):
        data[col] = data[col].fillna("").astype(str).str.strip()
    data = data[(data["subject"] != "") | (data["body"] != "")]  # nothing to scan
    return data.reset_index(drop=True)


def analyze_emails(df: pd.DataFrame, model=None) -> pd.DataFrame:
    """
    Scan every email and return one result row per email.

    Columns: timestamp, sender, subject, prediction, confidence, severity, details
    (plus recipient / user_id if they were in the input).

        prediction : 1 = phishing, 0 = legitimate (phishing probability >= 60%)
        confidence : phishing probability in percent (0-100)
        severity   : Critical (>= 85%), High (>= 60%), Safe (< 60%)
        details    : plain-English summary with the top trigger keywords

    Results are sorted with the most dangerous emails first. `model` is optional;
    if omitted the saved model is loaded (or trained) automatically.
    """
    data = _prepare_emails(df)
    extra = [c for c in OPTIONAL_EMAIL_COLUMNS if c in data.columns]
    if data.empty:
        return _empty_results(extra)

    if model is None:
        model = train_or_load_model()

    texts = (data["subject"] + " " + data["body"]).tolist()
    phishing_col = list(model.classes_).index(1)
    probability = model.predict_proba(texts)[:, phishing_col]

    data["confidence"] = np.round(probability * 100, 1)
    data["prediction"] = (probability >= HIGH_THRESHOLD).astype(int)
    data["severity"] = np.select(
        [probability >= CRITICAL_THRESHOLD, probability >= HIGH_THRESHOLD],
        ["Critical", "High"], default="Safe")

    # Trigger keywords are only worked out for flagged emails
    details = [""] * len(data)
    flagged_idx = np.flatnonzero(data["prediction"].to_numpy() == 1)
    triggers = _top_trigger_terms(model, [texts[i] for i in flagged_idx]) if len(flagged_idx) else []
    trigger_by_row = dict(zip(flagged_idx, triggers))

    for i, conf in enumerate(data["confidence"]):
        if i in trigger_by_row:
            keywords = ", ".join(trigger_by_row[i]) or "none identified"
            details[i] = f"Flagged phishing ({conf:.1f}% confidence) \u2014 Trigger keywords: {keywords}"
        else:
            details[i] = f"Classified as safe ({conf:.1f}% phishing probability)"
    data["details"] = details

    result = data[OUTPUT_COLUMNS + extra]
    return result.sort_values(["confidence", "timestamp"], ascending=[False, False]).reset_index(drop=True)


def analyze_text(text: str, model=None) -> pd.DataFrame:
    """Scan one pasted email (subject and body together) and return a one-row result."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Email text is empty.")
    single = pd.DataFrame([{
        "timestamp": pd.Timestamp.now(), "sender": "(pasted email)", "subject": "", "body": text,
    }])
    return analyze_emails(single, model=model)

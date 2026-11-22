"""
detector.py
-----------
Threat detection engine. The public entry point is `analyze_logs(df)`, which runs
three detection modules and returns one unified DataFrame of flagged threats.

Input schema (exactly 5 columns, one event per row):
    timestamp, user_id, ip_address, action, status

Detection modules:
    1. AFTER_HOURS_LOGIN     - LOGIN outside 09:00-16:59           (Medium)
    2. BRUTE_FORCE_ATTEMPT   - > 5 failed logins in 5 minutes      (High)
    3. ABNORMAL_RPM_SPIKE    - IsolationForest on requests/minute  (Critical)
"""

import pandas as pd
from sklearn.ensemble import IsolationForest

REQUIRED_COLUMNS = ["timestamp", "user_id", "ip_address", "action", "status"]
OUTPUT_COLUMNS = ["timestamp", "user_id", "ip_address", "threat_type", "severity", "details"]

WORK_START_HOUR = 9    # inclusive
WORK_END_HOUR = 17     # exclusive (17:00 and later is after hours)
BRUTE_FORCE_THRESHOLD = 5
IF_CONTAMINATION = 0.05
IF_RANDOM_STATE = 42
# By default each tree only sees 256 random rows, so on large logs a rare spike is
# usually missing from the trees. Let every tree see (up to) this many rows instead.
IF_MAX_SAMPLES_CAP = 200_000


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _empty_alerts() -> pd.DataFrame:
    """An empty result frame that still has the correct columns."""
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def _prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Validate the schema, normalise text columns and parse timestamps."""
    if not isinstance(df, pd.DataFrame):
        raise TypeError("analyze_logs expects a pandas DataFrame.")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required column(s): {', '.join(missing)}. "
            f"Expected columns: {', '.join(REQUIRED_COLUMNS)}."
        )

    data = df[REQUIRED_COLUMNS].copy()
    data["timestamp"] = pd.to_datetime(data["timestamp"], errors="coerce")
    data = data.dropna(subset=["timestamp"])  # drop rows with unparseable timestamps

    for col in ("user_id", "ip_address", "action", "status"):
        data[col] = data[col].astype(str).str.strip()
    data["action"] = data["action"].str.upper()
    data["status"] = data["status"].str.upper()
    return data.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# Rule 1 - After-hours login
# --------------------------------------------------------------------------- #
def detect_after_hours_login(df: pd.DataFrame) -> pd.DataFrame:
    """Flag LOGIN events before 09:00 or from 17:00 onwards."""
    hour = df["timestamp"].dt.hour
    mask = (df["action"] == "LOGIN") & ((hour < WORK_START_HOUR) | (hour >= WORK_END_HOUR))
    hits = df.loc[mask, ["timestamp", "user_id", "ip_address"]].copy()
    if hits.empty:
        return _empty_alerts()

    hits["threat_type"] = "AFTER_HOURS_LOGIN"
    hits["severity"] = "Medium"
    hits["details"] = hits["timestamp"].dt.strftime("%H:%M:%S").map(
        lambda t: f"Login detected outside working hours (9 AM - 5 PM) at {t}"
    )
    return hits[OUTPUT_COLUMNS]


# --------------------------------------------------------------------------- #
# Rule 2 - Brute-force attempt
# --------------------------------------------------------------------------- #
def detect_brute_force(df: pd.DataFrame) -> pd.DataFrame:
    """Flag users with more than 5 failed logins inside a 5-minute window."""
    failed = df[(df["action"] == "LOGIN") & (df["status"] == "FAILED")].copy()
    if failed.empty:
        return _empty_alerts()

    # pd.Grouper consumes the 'timestamp' column, so keep a copy to aggregate on
    failed["event_time"] = failed["timestamp"]
    grouped = (
        failed.groupby(["user_id", pd.Grouper(key="timestamp", freq="5min")])
        .agg(
            failed_count=("event_time", "size"),
            last_attempt=("event_time", "max"),
            ip_address=("ip_address", "last"),
        )
        .reset_index()
    )

    hits = grouped[grouped["failed_count"] > BRUTE_FORCE_THRESHOLD].copy()
    if hits.empty:
        return _empty_alerts()

    hits["timestamp"] = hits["last_attempt"]  # time of the final failed attempt
    hits["threat_type"] = "BRUTE_FORCE_ATTEMPT"
    hits["severity"] = "High"
    hits["details"] = hits["failed_count"].map(
        lambda n: f"Detected {n} failed login attempts within 5 minutes"
    )
    return hits[OUTPUT_COLUMNS]


# --------------------------------------------------------------------------- #
# Rule 3 - AI anomaly detection on requests per minute
# --------------------------------------------------------------------------- #
def detect_rpm_spikes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute requests_per_minute (events per user per 1-minute window), fit an
    IsolationForest on it, and flag windows that are BOTH model anomalies AND
    above mean + 2*std (so unusually low counts are never flagged).
    """
    if df.empty:
        return _empty_alerts()

    work = df.assign(minute=df["timestamp"].dt.floor("1min"))
    rpm = (
        work.groupby(["user_id", "minute"])
        .agg(
            requests_per_minute=("timestamp", "size"),
            ip_address=("ip_address", lambda s: s.mode().iat[0]),
        )
        .reset_index()
    )

    # IsolationForest and std() need at least a couple of observations
    if len(rpm) < 2:
        return _empty_alerts()

    model = IsolationForest(
        contamination=IF_CONTAMINATION,
        max_samples=min(len(rpm), IF_MAX_SAMPLES_CAP),
        random_state=IF_RANDOM_STATE,
    )
    rpm["prediction"] = model.fit_predict(rpm[["requests_per_minute"]])

    mean = rpm["requests_per_minute"].mean()
    std = rpm["requests_per_minute"].std()
    threshold = mean + 2 * (0.0 if pd.isna(std) else std)
    rpm["user_avg"] = rpm.groupby("user_id")["requests_per_minute"].transform("mean")

    hits = rpm[(rpm["prediction"] == -1) & (rpm["requests_per_minute"] > threshold)].copy()
    if hits.empty:
        return _empty_alerts()

    hits["timestamp"] = hits["minute"]
    hits["threat_type"] = "ABNORMAL_RPM_SPIKE"
    hits["severity"] = "Critical"
    hits["details"] = [
        f"AI detected abnormal traffic spike: {r} requests/min (User avg: {a:.1f})"
        for r, a in zip(hits["requests_per_minute"], hits["user_avg"])
    ]
    return hits[OUTPUT_COLUMNS]


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def analyze_logs(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run all detection modules and return flagged threats, newest first.

    Returns a DataFrame with columns:
        timestamp, user_id, ip_address, threat_type, severity, details
    (an empty frame with these columns when nothing is flagged).

    Raises:
        TypeError / ValueError for non-DataFrame input or missing columns.
    """
    data = _prepare(df)
    if data.empty:
        return _empty_alerts()

    results = [
        detect_after_hours_login(data),
        detect_brute_force(data),
        detect_rpm_spikes(data),
    ]
    results = [r for r in results if not r.empty]  # avoid concat warnings on empties
    if not results:
        return _empty_alerts()

    alerts = pd.concat(results, ignore_index=True)
    return alerts.sort_values("timestamp", ascending=False).reset_index(drop=True)[OUTPUT_COLUMNS]

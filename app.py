"""
app.py
------
Streamlit dashboard for the AI-Powered Log Threat Detection System.

Two tabs:
    1. System Log Threat Detection  (rules + IsolationForest anomaly detection)
    2. Phishing Email Detection     (supervised TF-IDF + Logistic Regression)

Run with:
    streamlit run app.py
"""

from pathlib import Path

import pandas as pd
import streamlit as st

from detector import analyze_logs
from phishing_detector import analyze_emails, analyze_text, train_or_load_model

SAMPLE_PATH = Path(__file__).resolve().parent / "data" / "system_logs.csv"
SAMPLE_EMAILS_PATH = Path(__file__).resolve().parent / "data" / "sample_emails.csv"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# Colours used for the severity badges in the alerts tables
SEVERITY_STYLES = {
    "Critical": "background-color: #b71c1c; color: white;",
    "High": "background-color: #ef6c00; color: white;",
    "Medium": "background-color: #fbc02d; color: black;",
    "Low": "background-color: #1976d2; color: white;",
}
# The phishing tab reuses the same colours and adds one for "Safe"
EMAIL_SEVERITY_STYLES = {**SEVERITY_STYLES, "Safe": "background-color: #2e7d32; color: white;"}

st.set_page_config(page_title="AI-Powered Log Threat Detection System", page_icon="🛡️", layout="wide")


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def style_severity(styler, styles=SEVERITY_STYLES):
    """Apply colour-coded severity styling (works across pandas versions)."""
    def css(value):
        if value in styles:
            return styles[value] + " font-weight: bold; text-align: center;"
        return ""

    apply_fn = styler.map if hasattr(styler, "map") else styler.applymap
    return apply_fn(css, subset=["severity"])


def sync_data_source(uploaded, use_sample, key_name, loader, sample_path, sample_label):
    """
    Decide what to (re)load after a Streamlit rerun.

    * Sample button clicked  -> load the sample file.
    * New file uploaded      -> analyse it once (remembered by name + size).
    * Nothing uploaded       -> forget the remembered upload.
    """
    if use_sample:
        loader(sample_path, sample_label)
        if uploaded is not None:  # remember it so it is not reloaded on the next rerun
            st.session_state[key_name] = (uploaded.name, uploaded.size)
    elif uploaded is not None:
        upload_key = (uploaded.name, uploaded.size)
        if st.session_state.get(key_name) != upload_key:  # analyse each upload only once
            st.session_state[key_name] = upload_key
            loader(uploaded, f"Uploaded: {uploaded.name}")
    else:
        st.session_state.pop(key_name, None)


# --------------------------------------------------------------------------- #
# Tab 1 - system logs: data loading
# --------------------------------------------------------------------------- #
def load_and_analyze(source, label: str) -> None:
    """Read a CSV, run the detector and store the results in session state."""
    try:
        logs = pd.read_csv(source)
        alerts = analyze_logs(logs)
    except FileNotFoundError:
        st.sidebar.error("Sample dataset not found. Run `python generate_data.py` first.")
        return
    except pd.errors.EmptyDataError:
        st.sidebar.error("The CSV file is empty.")
        return
    except (ValueError, TypeError) as exc:
        st.sidebar.error(f"Invalid log file: {exc}")
        return
    except Exception as exc:  # last-resort guard so the dashboard never crashes
        st.sidebar.error(f"Unexpected error while analysing logs: {exc}")
        return

    st.session_state["total_logs"] = len(logs)
    st.session_state["alerts"] = alerts
    st.session_state["source_label"] = label


# --------------------------------------------------------------------------- #
# Tab 1 - system logs: dashboard
# --------------------------------------------------------------------------- #
def render_log_tab() -> None:
    if "alerts" not in st.session_state:
        st.info("Upload a CSV or click **Load Sample Dataset** in the sidebar to begin.")
        return

    alerts: pd.DataFrame = st.session_state["alerts"]
    total_logs: int = st.session_state["total_logs"]
    st.caption(f"Source: {st.session_state['source_label']}")

    # --- KPI metrics ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Logs Processed", f"{total_logs:,}")
    k2.metric("Total Threats Flagged", f"{len(alerts):,}")
    k3.metric("Unique Suspicious Users", alerts["user_id"].nunique() if not alerts.empty else 0)
    k4.metric("Most Common Threat Type", alerts["threat_type"].mode().iat[0] if not alerts.empty else "None")

    if alerts.empty:
        st.success("No threats detected in this dataset.")
        return

    # --- Charts ---
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Threat Count by Type")
        st.bar_chart(alerts["threat_type"].value_counts())
    with c2:
        st.subheader("Top Flagged Users")
        st.bar_chart(alerts["user_id"].value_counts().head(10))

    # --- Filterable alerts table ---
    st.divider()
    st.subheader("Flagged Security Alerts")

    f1, f2 = st.columns(2)
    selected_types = f1.multiselect(
        "Filter by threat type", sorted(alerts["threat_type"].unique()), help="Leave empty to show all"
    )
    selected_users = f2.multiselect(
        "Filter by user", sorted(alerts["user_id"].unique()), help="Leave empty to show all"
    )

    filtered = alerts
    if selected_types:
        filtered = filtered[filtered["threat_type"].isin(selected_types)]
    if selected_users:
        filtered = filtered[filtered["user_id"].isin(selected_users)]

    display = filtered.copy()
    display["timestamp"] = display["timestamp"].dt.strftime(TIMESTAMP_FORMAT)
    st.caption(f"Showing {len(display):,} of {len(alerts):,} alerts")
    st.dataframe(style_severity(display.style), hide_index=True)

    # --- Export ---
    st.download_button(
        label="Download Threat Report (CSV)",
        data=alerts.to_csv(index=False).encode("utf-8"),
        file_name="threat_report.csv",
        mime="text/csv",
    )


# --------------------------------------------------------------------------- #
# Tab 2 - phishing emails
# --------------------------------------------------------------------------- #
@st.cache_resource
def get_phishing_model():
    """Load (or train on first use) the phishing model once and reuse it."""
    return train_or_load_model()


def load_and_scan_emails(source, label: str) -> None:
    """Read an email CSV, scan it and store the results in session state."""
    try:
        with st.spinner("Scanning emails..."):
            emails = pd.read_csv(source)
            results = analyze_emails(emails, model=get_phishing_model())
    except FileNotFoundError as exc:
        st.error(f"File not found: {exc}. For the sample file, run `python generate_phishing_data.py` first.")
        return
    except pd.errors.EmptyDataError:
        st.error("The email CSV file is empty.")
        return
    except (ValueError, TypeError) as exc:
        st.error(f"Invalid email file: {exc}")
        return
    except Exception as exc:  # last-resort guard so the dashboard never crashes
        st.error(f"Unexpected error while scanning emails: {exc}")
        return

    st.session_state["email_results"] = results
    st.session_state["email_source"] = label


def build_email_display(results: pd.DataFrame) -> pd.DataFrame:
    """Readable copy of the results for the table and the CSV report."""
    out = results.copy()
    out["timestamp"] = out["timestamp"].dt.strftime(TIMESTAMP_FORMAT).fillna("")
    out["prediction"] = out["prediction"].map({1: "Phishing", 0: "Legitimate"})
    return out


def render_single_email_scanner() -> None:
    with st.expander("Single Email Text Scanner"):
        text = st.text_area(
            "Paste an email (subject and body) to test it live",
            height=180,
            key="single_email_text",
            placeholder="URGENT: Your account has been suspended. Verify your password within 24 hours...",
        )
        if st.button("Scan Email Text", key="scan_single_email"):
            if not text.strip():
                st.warning("Paste some email text first.")
                return
            try:
                row = analyze_text(text, model=get_phishing_model()).iloc[0]
            except Exception as exc:
                st.error(f"Could not scan this text: {exc}")
                return

            confidence = float(row["confidence"])
            if row["prediction"] == 1:
                st.error(f"Phishing detected: {confidence:.1f}% confidence ({row['severity']})")
            else:
                st.success(f"Looks safe: {confidence:.1f}% phishing probability")
            st.progress(min(max(confidence / 100, 0.0), 1.0))
            st.write(row["details"])


def render_phishing_tab() -> None:
    st.caption(
        "Supervised model (TF-IDF + Logistic Regression) trained on labeled emails. "
        "The bundled training data is synthetic, so treat scores as a demo."
    )

    render_single_email_scanner()

    # --- Data source ---
    col_upload, col_sample = st.columns([2, 1])
    with col_upload:
        uploaded = st.file_uploader("Upload email CSV", type=["csv"], key="email_uploader")
    with col_sample:
        st.write("")  # nudge the button down, level with the uploader
        use_sample = st.button("Load Sample Emails (data/sample_emails.csv)")
    st.caption("Required columns: timestamp, sender, subject, body (optional: recipient or user_id)")

    sync_data_source(uploaded, use_sample, "email_upload_key", load_and_scan_emails,
                     SAMPLE_EMAILS_PATH, "Sample emails")

    if "email_results" not in st.session_state:
        st.info("Upload an email CSV or click **Load Sample Emails** to begin.")
        return

    results: pd.DataFrame = st.session_state["email_results"]
    st.caption(f"Source: {st.session_state['email_source']}")

    if results.empty:
        st.warning("No scannable emails found (every row had an empty subject and body).")
        return

    flagged = results[results["prediction"] == 1]
    total = len(results)

    # --- KPI metrics ---
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Emails Scanned", f"{total:,}")
    k2.metric("Phishing Emails Flagged", f"{len(flagged):,}")
    k3.metric("Safe Emails", f"{total - len(flagged):,}")
    k4.metric("Highest Phishing Confidence (%)", f"{results['confidence'].max():.1f}%")

    # --- Charts ---
    st.divider()
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Email Classification Breakdown")
        st.bar_chart(pd.Series({"Phishing": len(flagged), "Safe": total - len(flagged)}))
    with c2:
        st.subheader("Top Suspicious Senders")
        if flagged.empty:
            st.info("No suspicious senders.")
        else:
            st.bar_chart(flagged["sender"].value_counts().head(10))

    # --- Alerts table ---
    st.divider()
    st.subheader("Phishing Email Alerts")
    show_safe = st.checkbox("Include safe emails in the table", value=False)
    table = results if show_safe else flagged

    if table.empty:
        st.success("No phishing emails flagged in this file.")
    else:
        display = build_email_display(table)
        st.caption(f"Showing {len(display):,} of {total:,} scanned emails")
        styled = style_severity(display.style.format({"confidence": "{:.1f}%"}), EMAIL_SEVERITY_STYLES)
        st.dataframe(styled, hide_index=True)

    # --- Export ---
    st.download_button(
        label="Download Phishing Report (CSV)",
        data=build_email_display(flagged).to_csv(index=False).encode("utf-8"),
        file_name="phishing_report.csv",
        mime="text/csv",
        disabled=flagged.empty,
    )


# --------------------------------------------------------------------------- #
# Page layout
# --------------------------------------------------------------------------- #
st.title("AI-Powered Log Threat Detection System")

# Sidebar: data source for the System Log tab
with st.sidebar:
    st.header("System Log Data Source")
    uploaded_logs = st.file_uploader("Upload log CSV", type=["csv"])
    use_sample_logs = st.button("Load Sample Dataset (data/system_logs.csv)")
    st.caption("Required columns: timestamp, user_id, ip_address, action, status")
    st.caption("This sidebar feeds the System Log tab. The Phishing tab has its own upload.")

sync_data_source(uploaded_logs, use_sample_logs, "upload_key", load_and_analyze,
                 SAMPLE_PATH, "Sample dataset")

tab_logs, tab_phishing = st.tabs(["System Log Threat Detection", "Phishing Email Detection"])
with tab_logs:
    render_log_tab()
with tab_phishing:
    render_phishing_tab()

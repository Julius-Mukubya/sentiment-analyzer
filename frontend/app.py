"""
frontend/app.py — Single-file Streamlit app using st.navigation + st.Page.

Three pages defined inline as functions:
  page_home()      — landing page with model info and API status
  page_feedback()  — feedback form → POST /feedback
  page_dashboard() — charts + filterable table → GET /feedback + /feedback/stats

Run with: streamlit run frontend/app.py
"""

import os
import re

from dotenv import load_dotenv
import pandas as pd
import plotly.express as px
import requests
import streamlit as st

load_dotenv()

# ── Shared config ──────────────────────────────────────────────────────────────

API_URL = os.environ.get("API_URL", "http://127.0.0.1:8000")

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
CATEGORIES = ["Product", "Service", "General"]
SENTIMENT_CONFIG = {
    "POSITIVE": {"icon": "😊", "color": "#2ecc71", "label": "Positive"},
    "NEGATIVE": {"icon": "😟", "color": "#e74c3c", "label": "Negative"},
}


def check_api_health() -> bool:
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        return r.status_code == 200
    except requests.exceptions.ConnectionError:
        return False


def _health_banner() -> None:
    """Render the API health indicator — shared across all pages."""
    if check_api_health():
        st.success("API is online", icon="✅")
    else:
        st.error(
            f"Cannot reach the API at {API_URL}. "
            "Start the backend with: `uvicorn backend.app:app --reload`",
            icon="🔴",
        )


# ── API helpers ────────────────────────────────────────────────────────────────

def api_submit_feedback(payload: dict) -> dict:
    r = requests.post(f"{API_URL}/feedback", json=payload, timeout=60)
    if r.status_code == 200:
        return r.json()
    elif r.status_code == 422:
        detail = r.json().get("detail", "Invalid input.")
        if isinstance(detail, list):
            raise ValueError("; ".join(e.get("msg", str(e)) for e in detail))
        raise ValueError(detail)
    else:
        raise RuntimeError(f"API error {r.status_code}: {r.text}")


def api_fetch_feedback() -> list[dict]:
    try:
        r = requests.get(f"{API_URL}/feedback", timeout=10)
        return r.json() if r.status_code == 200 else []
    except Exception:
        return []


def api_fetch_stats() -> dict | None:
    try:
        r = requests.get(f"{API_URL}/feedback/stats", timeout=10)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


# ── Pages ──────────────────────────────────────────────────────────────────────

def page_home():
    _health_banner()
    st.title("💬 Sentiment Analyzer")
    st.markdown(
        """
        Welcome! This app collects feedback and uses an AI model to automatically
        detect whether each submission carries a **positive** or **negative** sentiment.

        ---

        ### How it works

        1. Go to the **Feedback** page and fill in the form.
        2. The backend runs your text through
           [distilbert-base-uncased-finetuned-sst-2-english](https://huggingface.co/distilbert/distilbert-base-uncased-finetuned-sst-2-english)
           — a DistilBERT model fine-tuned on the SST-2 sentiment dataset.
        3. The sentiment label (POSITIVE / NEGATIVE) and confidence score are saved
           alongside your feedback in a local SQLite database.
        4. Visit the **Dashboard** page to see all submissions, summary statistics,
           and interactive charts.

        ---

        ### Model
        | Detail | Value |
        |---|---|
        | Model | `distilbert-base-uncased-finetuned-sst-2-english` |
        | Parameters | ~67 million |
        | Task | Binary sentiment classification |
        | Labels | `POSITIVE` / `NEGATIVE` |
        | Training data | Stanford Sentiment Treebank v2 (SST-2) |

        Use the sidebar to navigate between pages.
        """
    )


def page_feedback():
    _health_banner()
    st.title("📝 Submit Feedback")
    st.caption("Share your experience. The AI will analyse the sentiment of your feedback automatically.")
    st.divider()

    with st.form("feedback_form", clear_on_submit=True):
        name  = st.text_input("Name (optional)", placeholder="Your name")
        email = st.text_input("Email *", placeholder="you@example.com")
        category = st.selectbox("Category *", options=CATEGORIES)
        rating = st.slider(
            "Overall rating *", min_value=1, max_value=5, value=3,
            help="1 = very poor  ·  5 = excellent",
        )
        st.caption(f"Your rating: {'⭐' * rating}{'☆' * (5 - rating)} ({rating}/5)")
        feedback_text = st.text_area(
            "Feedback *", height=160,
            placeholder="Tell us what you think… (min 10 characters)",
        )
        submitted = st.form_submit_button("Submit", type="primary", use_container_width=True)

    if submitted:
        errors = []
        if not email.strip():
            errors.append("Email is required.")
        elif not _EMAIL_RE.match(email.strip()):
            errors.append("Please enter a valid email address.")
        if not feedback_text.strip():
            errors.append("Feedback text is required.")
        elif len(feedback_text.strip()) < 10:
            errors.append("Feedback must be at least 10 characters.")

        if errors:
            for err in errors:
                st.error(err)
        else:
            with st.spinner("Analysing sentiment and saving…"):
                try:
                    result = api_submit_feedback({
                        "name":     name.strip() or None,
                        "email":    email.strip(),
                        "category": category,
                        "rating":   rating,
                        "feedback": feedback_text.strip(),
                    })
                except ValueError as e:
                    st.error(str(e)); return
                except RuntimeError as e:
                    st.error(str(e)); return
                except requests.exceptions.Timeout:
                    st.error("Request timed out — the model may still be loading. Try again."); return
                except requests.exceptions.ConnectionError:
                    st.error("Lost connection to the API."); return

            sentiment  = result["sentiment"]
            confidence = result["confidence"]
            cfg        = SENTIMENT_CONFIG[sentiment]

            st.success("Thank you! Your feedback has been submitted. ✅")
            st.divider()
            st.subheader("Sentiment Analysis Result")

            c1, c2, c3 = st.columns(3)
            c1.metric("Sentiment",   f"{cfg['icon']} {cfg['label']}")
            c2.metric("Confidence",  f"{confidence * 100:.1f}%")
            c3.metric("Your Rating", f"{'⭐' * rating} ({rating}/5)")

            st.markdown(
                f"""
                <div style="margin-top:8px">
                    <div style="font-size:0.8rem;color:#888;margin-bottom:4px">Confidence</div>
                    <div style="background:#e0e0e0;border-radius:6px;height:14px">
                        <div style="background:{cfg['color']};width:{confidence*100:.1f}%;
                                    height:14px;border-radius:6px"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.caption("Go to the **Dashboard** page to see all submissions and charts.")


def page_dashboard():
    st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")
    _health_banner()

    stats = api_fetch_stats()
    rows  = api_fetch_feedback()

    st.title("📊 Feedback Dashboard")
    st.caption("Live view of all submitted feedback and sentiment breakdown.")

    if not stats or stats["total"] == 0:
        st.info("No feedback submitted yet. Go to the **Feedback** page to add the first entry.")
        return

    # KPI strip
    st.divider()
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Submissions", stats["total"])
    k2.metric("Positive 😊",       stats["positive"])
    k3.metric("Negative 😟",       stats["negative"])
    k4.metric("Avg Rating",        f"⭐ {stats['avg_rating']}")
    pos_pct = round(stats["positive"] / stats["total"] * 100) if stats["total"] else 0
    k5.metric("Positive Rate",     f"{pos_pct}%")
    st.divider()

    df = pd.DataFrame(rows)
    df["created_at"]     = pd.to_datetime(df["created_at"], utc=True)
    df["date"]           = df["created_at"].dt.date
    df["confidence_pct"] = (df["confidence"] * 100).round(1)
    df["stars"]          = df["rating"].apply(lambda r: "⭐" * r)

    # Charts row 1
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Sentiment Breakdown")
        counts = df["sentiment"].value_counts().reset_index()
        counts.columns = ["Sentiment", "Count"]
        fig = px.pie(
            counts, names="Sentiment", values="Count", color="Sentiment",
            color_discrete_map={"POSITIVE": "#2ecc71", "NEGATIVE": "#e74c3c"}, hole=0.45,
        )
        fig.update_traces(textposition="inside", textinfo="percent+label")
        fig.update_layout(showlegend=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("Submissions by Category")
        cat_df = df.groupby(["category", "sentiment"]).size().reset_index(name="count")
        fig = px.bar(
            cat_df, x="category", y="count", color="sentiment", barmode="group",
            color_discrete_map={"POSITIVE": "#2ecc71", "NEGATIVE": "#e74c3c"},
            labels={"category": "Category", "count": "Count", "sentiment": "Sentiment"},
        )
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    # Charts row 2
    col3, col4 = st.columns(2)
    with col3:
        st.subheader("Star Rating Distribution")
        rat_df = df["rating"].value_counts().sort_index().reset_index()
        rat_df.columns = ["Rating", "Count"]
        rat_df["label"] = rat_df["Rating"].apply(lambda r: f"{'⭐'*r} ({r})")
        fig = px.bar(
            rat_df, x="label", y="Count", color="Count",
            color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
            labels={"label": "Rating"},
        )
        fig.update_layout(coloraxis_showscale=False, margin=dict(t=10, b=10, l=10, r=10))
        st.plotly_chart(fig, use_container_width=True)

    with col4:
        st.subheader("Confidence Score Distribution")
        fig = px.histogram(
            df, x="confidence_pct", color="sentiment", nbins=20, barmode="overlay", opacity=0.75,
            color_discrete_map={"POSITIVE": "#2ecc71", "NEGATIVE": "#e74c3c"},
            labels={"confidence_pct": "Confidence (%)", "sentiment": "Sentiment"},
        )
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    # Submissions over time
    if df["date"].nunique() > 1:
        st.subheader("Submissions Over Time")
        time_df = df.groupby(["date", "sentiment"]).size().reset_index(name="count")
        fig = px.line(
            time_df, x="date", y="count", color="sentiment", markers=True,
            color_discrete_map={"POSITIVE": "#2ecc71", "NEGATIVE": "#e74c3c"},
            labels={"date": "Date", "count": "Submissions", "sentiment": "Sentiment"},
        )
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), legend_title_text="")
        st.plotly_chart(fig, use_container_width=True)

    # Filterable table
    st.divider()
    st.subheader("All Feedback")
    f1, f2, f3 = st.columns(3)
    with f1:
        fs = st.selectbox("Filter by sentiment", ["All", "POSITIVE", "NEGATIVE"])
    with f2:
        fc = st.selectbox("Filter by category", ["All"] + sorted(df["category"].unique().tolist()))
    with f3:
        fr = st.selectbox("Filter by rating", ["All"] + [str(i) for i in range(1, 6)])

    filtered = df.copy()
    if fs != "All": filtered = filtered[filtered["sentiment"] == fs]
    if fc != "All": filtered = filtered[filtered["category"]  == fc]
    if fr != "All": filtered = filtered[filtered["rating"]    == int(fr)]

    display = (
        filtered[["created_at", "name", "email", "category", "stars",
                   "sentiment", "confidence_pct", "feedback"]]
        .rename(columns={
            "created_at": "Submitted", "name": "Name", "email": "Email",
            "category": "Category", "stars": "Rating", "sentiment": "Sentiment",
            "confidence_pct": "Confidence %", "feedback": "Feedback",
        })
    )
    display["Submitted"] = display["Submitted"].dt.strftime("%Y-%m-%d %H:%M UTC")

    st.dataframe(
        display, use_container_width=True, hide_index=True,
        column_config={
            "Confidence %": st.column_config.NumberColumn(format="%.1f%%"),
            "Feedback":     st.column_config.TextColumn(width="large"),
        },
    )
    st.caption(f"Showing {len(filtered)} of {len(df)} submissions.")


# ── Navigation ─────────────────────────────────────────────────────────────────

pg = st.navigation([
    st.Page(page_home,      title="Home",      icon="🏠", default=True),
    st.Page(page_feedback,  title="Feedback",  icon="📝"),
    st.Page(page_dashboard, title="Dashboard", icon="📊"),
])
pg.run()

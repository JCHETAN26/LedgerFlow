"""
LedgerFlow dashboard — an interactive view over the pipeline's outputs.

Run with:

    streamlit run dashboard/app.py

It reads the DVC-produced artifacts (metrics, feature importance, ROC/PR curves,
the feature matrix, raw events, trained models) and adds a live-scoring tab that
runs the *same* point-in-time feature code used in training — demonstrating the
no-training-serving-skew design end to end.
"""

from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

# Make the repo importable when run via `streamlit run dashboard/app.py`.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dashboard.loaders import Artifacts, metrics_table  # noqa: E402

st.set_page_config(page_title="LedgerFlow", page_icon="📊", layout="wide")

ART = Artifacts()
AVAIL = ART.available()


def _require(*keys: str) -> bool:
    """Show a friendly message and return False if any artifact is missing."""
    missing = [k for k in keys if not AVAIL.get(k)]
    if missing:
        st.warning(
            f"Missing artifact(s): {', '.join(missing)}. "
            "Run `dvc repro` (or `dvc pull`) to generate them."
        )
        return False
    return True


st.title("📊 LedgerFlow")
st.caption(
    "Point-in-time feature engineering + offline model evaluation, "
    "fully versioned with DVC."
)

if not any(AVAIL.values()):
    st.error(
        "No pipeline artifacts found. From the project root run `dvc repro` "
        "(or `dvc pull` if you have the S3 remote configured)."
    )
    st.stop()

overview, models, curves, features, data, scoring, memo = st.tabs(
    ["Overview", "Models", "Curves", "Features", "Data", "Live scoring", "Memo"]
)

# --------------------------------------------------------------------------- #
# Overview
# --------------------------------------------------------------------------- #
with overview:
    if _require("eval_metrics"):
        ev = ART.load_eval_metrics()
        rec = ev["recommended"]
        m = ev["models"][rec]
        st.subheader(f"Recommended model: :green[{rec}]")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("AUC-ROC", f"{m['auc_roc']:.3f}")
        c2.metric("Avg precision", f"{m['avg_precision']:.3f}")
        c3.metric("Brier (↓ better)", f"{m['brier_score']:.3f}")
        c4.metric("Precision @ 5% FPR", f"{m['precision_at_5pct_fpr']:.3f}")
        st.info(ev.get("recommendation_reason", ""))

    if AVAIL.get("feature_runtime"):
        rt = ART.load_feature_runtime()
        c1, c2, c3 = st.columns(3)
        c1.metric("Users", f"{rt.get('n_users', '—'):,}")
        c2.metric("Features", rt.get("n_features", "—"))
        c3.metric("Featurize runtime", f"{rt.get('runtime_seconds', 0):.2f}s")

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
with models:
    if _require("eval_metrics"):
        ev = ART.load_eval_metrics()
        table = metrics_table(ev)
        st.subheader("Model comparison")
        st.dataframe(
            table.style.format("{:.4f}").highlight_max(
                subset=["auc_roc", "avg_precision", "precision_at_5pct_fpr", "f1"],
                color="#1b5e20",
            ).highlight_min(subset=["brier_score", "log_loss"], color="#1b5e20"),
            use_container_width=True,
        )
        chart_df = table.reset_index()
        col1, col2 = st.columns(2)
        with col1:
            st.caption("AUC-ROC (higher is better)")
            st.bar_chart(chart_df, x="model", y="auc_roc", horizontal=True)
        with col2:
            st.caption("Brier score (lower is better — calibration)")
            st.bar_chart(chart_df, x="model", y="brier_score", horizontal=True)

# --------------------------------------------------------------------------- #
# Curves
# --------------------------------------------------------------------------- #
with curves:
    if _require("curves", "eval_metrics"):
        ev = ART.load_eval_metrics()
        model_names = list(ev["models"].keys())

        roc_frames, pr_frames = [], []
        for name in model_names:
            try:
                roc = ART.load_curve(name, "roc")
                roc_frames.append(
                    pd.DataFrame({"fpr": roc["fpr"], "tpr": roc["tpr"], "model": name})
                )
                pr = ART.load_curve(name, "pr")
                pr_frames.append(
                    pd.DataFrame(
                        {"recall": pr["recall"], "precision": pr["precision"], "model": name}
                    )
                )
            except FileNotFoundError:
                continue

        col1, col2 = st.columns(2)
        if roc_frames:
            roc_df = pd.concat(roc_frames)
            chance = pd.DataFrame({"fpr": [0, 1], "tpr": [0, 1]})
            roc_chart = (
                alt.Chart(roc_df)
                .mark_line()
                .encode(x="fpr", y="tpr", color="model")
                .properties(title="ROC curve", height=360)
            )
            diag = (
                alt.Chart(chance)
                .mark_line(strokeDash=[4, 4], color="gray")
                .encode(x="fpr", y="tpr")
            )
            col1.altair_chart(roc_chart + diag, use_container_width=True)
        if pr_frames:
            pr_df = pd.concat(pr_frames)
            pr_chart = (
                alt.Chart(pr_df)
                .mark_line()
                .encode(x="recall", y="precision", color="model")
                .properties(title="Precision-Recall curve", height=360)
            )
            col2.altair_chart(pr_chart, use_container_width=True)

# --------------------------------------------------------------------------- #
# Features
# --------------------------------------------------------------------------- #
with features:
    if _require("feature_importance"):
        imp = ART.load_feature_importance().sort_values(
            "mean_importance", ascending=False
        )
        top_n = st.slider("Top N features", 5, min(35, len(imp)), 15)
        top = imp.head(top_n)[["feature", "mean_importance"]]
        st.subheader("Feature importance (mean across models)")
        st.bar_chart(top, x="feature", y="mean_importance", horizontal=True)

    if AVAIL.get("low_signal"):
        ls = ART.load_low_signal()
        st.subheader("Low-signal features (removal candidates)")
        st.caption(f"Threshold: {ls['threshold']} — {len(ls['features'])} flagged")
        if ls["features"]:
            st.write(", ".join(f"`{f}`" for f in ls["features"]))
        else:
            st.success("No features below the importance threshold.")

# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
with data:
    if _require("events"):
        events = ART.load_events()
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Event-type distribution")
            st.bar_chart(events["event_type"].value_counts())
        with c2:
            st.caption("Daily event volume")
            daily = events.set_index("event_timestamp").resample("1D").size()
            st.line_chart(daily)
        st.caption("Purchase amount distribution (99th pct clipped)")
        amt = events.loc[events["event_type"] == "purchase", "amount"].dropna()
        if not amt.empty:
            clipped = amt.clip(upper=amt.quantile(0.99))
            hist = pd.cut(clipped, bins=40).value_counts().sort_index()
            st.bar_chart(hist.reset_index(drop=True))

    if AVAIL.get("features"):
        feat = ART.load_features()
        feat_cols = [c for c in feat.columns if c.startswith("purchase_")]
        st.subheader("Feature matrix")
        st.caption(f"{feat.shape[0]:,} users × {len(feat_cols)} features")
        st.dataframe(feat[feat_cols].describe().T, use_container_width=True)

# --------------------------------------------------------------------------- #
# Live scoring (demonstrates the single compute() path)
# --------------------------------------------------------------------------- #
with scoring:
    st.caption(
        "Scores a single user with the **same** point-in-time feature code used "
        "in training (`transform_single`) — no training-serving skew."
    )
    model_names = ART.list_models()
    if not (AVAIL.get("events") and model_names):
        _require("events")
        if not model_names:
            st.warning("No trained models found. Run `dvc repro` first.")
    else:
        from LedgerFlow.features.time_windows import ALL_FEATURES
        from LedgerFlow.pipeline import FeaturePipeline

        events = ART.load_events()
        feature_cols = [f.name for f in ALL_FEATURES]

        c1, c2 = st.columns(2)
        model_name = c1.selectbox("Model", model_names)
        # Offer users that actually have purchase events, for a livelier demo.
        buyers = (
            events.loc[events["event_type"] == "purchase", "user_id"]
            .value_counts()
            .index.tolist()
        )
        user_id = c2.selectbox("User", buyers[:200] or events["user_id"].unique()[:200])

        history = events[events["user_id"] == user_id].sort_values("event_timestamp")
        ref_time = history["event_timestamp"].max()

        pipe = FeaturePipeline()
        row = pipe.transform_single(history, ref_time)
        x = pd.DataFrame([row])[feature_cols]

        model = ART.load_model(model_name)
        proba = float(model.predict_proba(x)[:, 1][0])

        st.metric(f"Predicted positive probability ({model_name})", f"{proba:.1%}")
        st.progress(min(max(proba, 0.0), 1.0))

        with st.expander(f"User event history ({len(history)} events)"):
            st.dataframe(history, use_container_width=True)
        with st.expander("Computed feature vector (non-zero)"):
            nz = {k: v for k, v in row.items() if v != 0}
            st.json(nz or {"(all features zero)": 0})

# --------------------------------------------------------------------------- #
# Memo
# --------------------------------------------------------------------------- #
with memo:
    if _require("memo"):
        st.markdown(ART.load_memo())

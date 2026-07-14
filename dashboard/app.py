"""
LedgerFlow dashboard — a product-grade view over the pipeline's outputs.

Run with:

    streamlit run dashboard/app.py

It reads the DVC-produced artifacts (metrics, feature importance, ROC/PR curves,
the feature matrix, raw events, trained models) and adds a live-scoring tab that
runs the *same* point-in-time feature code used in training — demonstrating the
no-training-serving-skew design end to end.

Visual language follows a single validated palette (colorblind-safe categorical
hues in fixed per-model order, recessive axes, direct labels) so every chart
reads as one system.
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

# --------------------------------------------------------------------------- #
# Design tokens (validated light-mode palette — see dataviz reference)
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"
PLANE = "#f9f9f7"
PRIMARY = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
ACCENT = "#2a78d6"
FONT = "system-ui, -apple-system, 'Segoe UI', sans-serif"

# Status ramp (fixed — never themed).
GOOD = "#0ca30c"
WARNING = "#fab219"
SERIOUS = "#ec835a"
CRITICAL = "#d03b3b"

# Categorical hues, assigned to models in a FIXED order (identity, not rank).
MODEL_COLORS = {
    "LogisticRegression": "#2a78d6",  # blue
    "XGBoost": "#1baf7a",  # aqua
    "LightGBM": "#eda100",  # yellow
}
BLUE_RAMP = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab"]  # sequential (magnitude)

st.set_page_config(page_title="LedgerFlow", page_icon="📊", layout="wide")

ART = Artifacts()
AVAIL = ART.available()


# --------------------------------------------------------------------------- #
# Styling helpers
# --------------------------------------------------------------------------- #
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        #MainMenu, footer {{visibility: hidden;}}
        [data-testid="stToolbar"], [data-testid="stDecoration"] {{display: none;}}
        header[data-testid="stHeader"] {{background: transparent; height: 0;}}
        .block-container {{
            padding-top: 1.4rem; padding-bottom: 3rem; max-width: 1200px;
        }}

        .hero {{
            display: flex; justify-content: space-between; align-items: center;
            gap: 1rem; padding: 1.4rem 1.6rem; border-radius: 16px;
            background: linear-gradient(135deg, #0d366b 0%, #1c5cab 55%, #2a78d6 100%);
            color: #fff; margin-bottom: 1.1rem;
        }}
        .hero-title {{
            font-size: 1.9rem; font-weight: 700;
            letter-spacing: -0.02em; line-height: 1;
        }}
        .hero-tag {{
            font-size: 0.95rem; opacity: 0.9;
            margin-top: 0.45rem; max-width: 46ch;
        }}
        .hero-badges {{
            display: flex; flex-direction: column;
            gap: 0.5rem; align-items: flex-end;
        }}
        .badge {{
            font-size: 0.78rem; font-weight: 600; padding: 0.32rem 0.7rem;
            border-radius: 999px; background: rgba(255,255,255,0.15);
            border: 1px solid rgba(255,255,255,0.25); white-space: nowrap;
        }}
        .badge-accent {{background: #fff; color: #1c5cab; border-color: #fff;}}

        .kpi {{
            background: {SURFACE}; border: 1px solid rgba(11,11,11,0.10);
            border-radius: 14px; padding: 1rem 1.1rem; height: 100%;
        }}
        .kpi-label {{
            font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
            letter-spacing: 0.06em; color: {MUTED};
        }}
        .kpi-value {{
            font-size: 1.75rem; font-weight: 700;
            line-height: 1.15; margin-top: 0.15rem;
        }}
        .kpi-sub {{font-size: 0.8rem; color: {SECONDARY}; margin-top: 0.2rem;}}

        .callout {{
            border-left: 3px solid {ACCENT}; background: #eef4fd;
            padding: 0.85rem 1.1rem; border-radius: 0 10px 10px 0;
            color: {PRIMARY}; font-size: 0.92rem; margin: 0.4rem 0 0.2rem 0;
        }}
        .section-h {{
            font-size: 1.05rem; font-weight: 700; color: {PRIMARY};
            margin: 0.2rem 0 0.6rem 0;
        }}
        .caption {{color: {SECONDARY}; font-size: 0.85rem; margin-bottom: 0.3rem;}}

        .stTabs [data-baseweb="tab-list"] {{gap: 0.3rem;}}
        .stTabs [data-baseweb="tab"] {{
            font-weight: 600; padding: 0.4rem 0.9rem; border-radius: 8px 8px 0 0;
        }}

        .risk-pill {{
            display: inline-block; padding: 0.3rem 0.85rem; border-radius: 999px;
            font-weight: 700; font-size: 0.9rem; color: #fff;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi(col, label: str, value: str, sub: str = "", accent: str = PRIMARY) -> None:
    col.markdown(
        f"""<div class="kpi">
              <div class="kpi-label">{label}</div>
              <div class="kpi-value" style="color:{accent}">{value}</div>
              <div class="kpi-sub">{sub}</div>
            </div>""",
        unsafe_allow_html=True,
    )


def section(title: str) -> None:
    st.markdown(f'<div class="section-h">{title}</div>', unsafe_allow_html=True)


def caption(text: str) -> None:
    st.markdown(f'<div class="caption">{text}</div>', unsafe_allow_html=True)


def style_chart(chart: alt.Chart, height: int = 320) -> alt.Chart:
    """Apply the shared, recessive chart chrome to any Altair chart."""
    return (
        chart.properties(height=height)
        .configure_view(strokeWidth=0)
        .configure_axis(
            labelColor=MUTED,
            titleColor=SECONDARY,
            labelFontSize=12,
            titleFontSize=12,
            labelFont=FONT,
            titleFont=FONT,
            gridColor=GRID,
            domainColor=BASELINE,
            tickColor=BASELINE,
        )
        .configure_axisX(grid=False)
        .configure_legend(
            labelColor=SECONDARY,
            titleColor=SECONDARY,
            labelFont=FONT,
            titleFont=FONT,
            orient="top",
            title=None,
        )
    )


def model_scale() -> alt.Scale:
    return alt.Scale(
        domain=list(MODEL_COLORS.keys()), range=list(MODEL_COLORS.values())
    )


def _require(*keys: str) -> bool:
    missing = [k for k in keys if not AVAIL.get(k)]
    if missing:
        st.warning(
            f"Missing artifact(s): {', '.join(missing)}. "
            "Run `dvc repro` (or `dvc pull`) to generate them."
        )
        return False
    return True


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
inject_css()

if not any(AVAIL.values()):
    st.error(
        "No pipeline artifacts found. From the project root run `dvc repro` "
        "(or `dvc pull` if you have the S3 remote configured)."
    )
    st.stop()

# ---- Hero (recommended model pulled live from metrics) --------------------- #
rec_name = None
n_events = n_rows = pos_rate = None
if AVAIL.get("eval_metrics"):
    _ev = ART.load_eval_metrics()
    rec_name = _ev.get("recommended")
if AVAIL.get("feature_runtime"):
    _rt = ART.load_feature_runtime()
    n_rows = _rt.get("n_rows")
# Prefer the ingest summary for provenance if present.
_summary_path = ART.root / "data/raw/ingest_summary.json"
dataset_badge = "Event-log pipeline"
if _summary_path.exists():
    import json as _json

    _s = _json.loads(_summary_path.read_text())
    n_events = _s.get("n_events")
    pos_rate = _s.get("positive_rate")
    src = _s.get("source", "data")
    if n_events:
        dataset_badge = f"{src.title()} · {n_events / 1e6:.1f}M txns"

rec_badge = f"{rec_name} recommended" if rec_name else "offline evaluation"
st.markdown(
    f"""
    <div class="hero">
      <div class="hero-left">
        <div class="hero-title">📊 LedgerFlow</div>
        <div class="hero-tag">Fraud &amp; risk scoring from raw event logs —
          point-in-time features, no training-serving skew, leakage-free splits.</div>
      </div>
      <div class="hero-badges">
        <span class="badge">{dataset_badge}</span>
        <span class="badge badge-accent">{rec_badge}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

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

        c1, c2, c3, c4, c5 = st.columns(5)
        kpi(c1, "Recommended", rec, "highest test AUC-ROC", accent=ACCENT)
        kpi(c2, "AUC-ROC", f"{m['auc_roc']:.3f}", "ranking power", accent=PRIMARY)
        kpi(
            c3,
            "Brier",
            f"{m['brier_score']:.3f}",
            "calibration · lower better",
            accent=GOOD if m["brier_score"] < 0.05 else PRIMARY,
        )
        if n_rows:
            kpi(c4, "Snapshots", f"{n_rows:,}", "point-in-time rows", accent=PRIMARY)
        if pos_rate is not None:
            kpi(
                c5,
                "Fraud rate",
                f"{100 * pos_rate:.2f}%",
                "real class imbalance",
                accent=SERIOUS,
            )

        st.markdown(
            f'<div class="callout">{ev.get("recommendation_reason", "")}</div>',
            unsafe_allow_html=True,
        )

    st.write("")
    section("How the number was produced")
    caption(
        "Raw events → Pandera schema validation → point-in-time feature matrix "
        "(each row as of its own decision_time) → time-ordered split → offline "
        "model comparison → auto-generated recommendation. The same feature code "
        "runs in training and in the Live-scoring tab — no skew."
    )
    if AVAIL.get("feature_runtime"):
        rt = ART.load_feature_runtime()
        c1, c2, c3 = st.columns(3)
        kpi(c1, "Features", str(rt.get("n_features", "—")), "windowed aggregations")
        kpi(c2, "Cards", f"{rt.get('n_users', 0):,}", "distinct users")
        kpi(
            c3,
            "Featurize time",
            f"{rt.get('runtime_seconds', 0):.0f}s",
            "point-in-time",
        )

# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
with models:
    if _require("eval_metrics"):
        ev = ART.load_eval_metrics()
        table = metrics_table(ev)
        section("Model comparison")
        st.dataframe(
            table.style.format("{:.4f}")
            .highlight_max(
                subset=["auc_roc", "avg_precision", "precision_at_5pct_fpr", "f1"],
                color="#dff0df",
            )
            .highlight_min(subset=["brier_score", "log_loss"], color="#dff0df"),
            use_container_width=True,
        )

        chart_df = table.reset_index()
        col1, col2 = st.columns(2)
        for col, field, title, better in (
            (col1, "auc_roc", "AUC-ROC", "higher is better"),
            (col2, "brier_score", "Brier score", "lower is better · calibration"),
        ):
            # Headroom on the x-axis so the direct value labels never clip.
            xmax = float(chart_df[field].max()) * 1.18
            # Explicit y order (largest bar on top) — avoids the layered-chart
            # "-x" sort ambiguity across the bar + label layers.
            order = chart_df.sort_values(field, ascending=False)["model"].tolist()
            base = alt.Chart(chart_df).encode(
                y=alt.Y("model:N", title=None, sort=order)
            )
            bars = base.mark_bar(cornerRadiusEnd=4, height=26).encode(
                x=alt.X(f"{field}:Q", title=None, scale=alt.Scale(domain=[0, xmax])),
                color=alt.Color("model:N", scale=model_scale(), legend=None),
            )
            # Separate text layer (no color encoding) → labels stay in text ink,
            # never illegible series yellow.
            labels = base.mark_text(
                align="left", dx=5, color=SECONDARY, fontSize=12, fontWeight=600
            ).encode(
                x=alt.X(f"{field}:Q"),
                text=alt.Text(f"{field}:Q", format=".3f"),
            )
            col.markdown(f'<div class="caption">{title} — {better}</div>', True)
            col.altair_chart(
                style_chart(bars + labels, height=180),
                use_container_width=True,
                theme=None,
            )

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
                        {
                            "recall": pr["recall"],
                            "precision": pr["precision"],
                            "model": name,
                        }
                    )
                )
            except FileNotFoundError:
                continue

        col1, col2 = st.columns(2)
        if roc_frames:
            roc_df = pd.concat(roc_frames)
            chance = pd.DataFrame({"fpr": [0, 1], "tpr": [0, 1]})
            roc_line = (
                alt.Chart(roc_df)
                .mark_line(strokeWidth=2.5)
                .encode(
                    x=alt.X("fpr:Q", title="False positive rate"),
                    y=alt.Y("tpr:Q", title="True positive rate"),
                    color=alt.Color("model:N", scale=model_scale()),
                )
            )
            diag = (
                alt.Chart(chance)
                .mark_line(strokeDash=[4, 4], color=MUTED, strokeWidth=1)
                .encode(x="fpr:Q", y="tpr:Q")
            )
            col1.markdown('<div class="caption">ROC — ranking (vs chance)</div>', True)
            col1.altair_chart(
                style_chart(diag + roc_line, height=340),
                use_container_width=True,
                theme=None,
            )
        if pr_frames:
            pr_df = pd.concat(pr_frames)
            pr_line = (
                alt.Chart(pr_df)
                .mark_line(strokeWidth=2.5)
                .encode(
                    x=alt.X("recall:Q", title="Recall"),
                    y=alt.Y("precision:Q", title="Precision"),
                    color=alt.Color("model:N", scale=model_scale()),
                )
            )
            col2.markdown(
                '<div class="caption">Precision-Recall — the imbalanced view</div>',
                True,
            )
            col2.altair_chart(
                style_chart(pr_line, height=340), use_container_width=True, theme=None
            )

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
        section("Feature importance (mean across models)")
        bars = (
            alt.Chart(top)
            .mark_bar(cornerRadiusEnd=4)
            .encode(
                x=alt.X("mean_importance:Q", title=None),
                y=alt.Y("feature:N", title=None, sort="-x"),
                color=alt.Color(
                    "mean_importance:Q",
                    scale=alt.Scale(range=BLUE_RAMP),
                    legend=None,
                ),
            )
        )
        st.altair_chart(
            style_chart(bars, height=28 * top_n + 30),
            use_container_width=True,
            theme=None,
        )

    if AVAIL.get("low_signal"):
        ls = ART.load_low_signal()
        section("Low-signal features (removal candidates)")
        caption(f"Threshold {ls['threshold']} — {len(ls['features'])} flagged")
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
        section("Event stream")
        c1, c2, c3 = st.columns(3)
        kpi(c1, "Events", f"{len(events):,}", "raw transactions")
        kpi(c2, "Cards", f"{events['user_id'].nunique():,}", "distinct users")
        span = (events["event_timestamp"].max() - events["event_timestamp"].min()).days
        kpi(c3, "Span", f"{span} days", "observation window")

        daily = (
            events.set_index("event_timestamp")
            .resample("1D")
            .size()
            .rename("events")
            .reset_index()
        )
        vol = (
            alt.Chart(daily)
            .mark_area(
                line={"color": ACCENT, "strokeWidth": 2},
                color=alt.Gradient(
                    gradient="linear",
                    stops=[
                        alt.GradientStop(color="#ffffff", offset=0),
                        alt.GradientStop(color="#cde2fb", offset=1),
                    ],
                    x1=1,
                    x2=1,
                    y1=1,
                    y2=0,
                ),
            )
            .encode(
                x=alt.X("event_timestamp:T", title=None),
                y=alt.Y("events:Q", title="daily events"),
            )
        )
        caption("Daily event volume")
        st.altair_chart(
            style_chart(vol, height=240), use_container_width=True, theme=None
        )

        col1, col2 = st.columns(2)
        amt = events.loc[events["event_type"] == "purchase", "amount"].dropna()
        if not amt.empty:
            clipped = amt.clip(upper=amt.quantile(0.99)).rename("amount")
            hist = (
                alt.Chart(clipped.reset_index())
                .mark_bar(color=ACCENT, cornerRadiusEnd=2)
                .encode(
                    x=alt.X("amount:Q", bin=alt.Bin(maxbins=40), title="amount ($)"),
                    y=alt.Y("count():Q", title="transactions"),
                )
            )
            col1.markdown(
                '<div class="caption">Purchase amount (99th pct clipped)</div>', True
            )
            col1.altair_chart(
                style_chart(hist, height=240), use_container_width=True, theme=None
            )
        etype = (
            events["event_type"]
            .value_counts()
            .rename_axis("type")
            .reset_index(name="n")
        )
        etype_bar = (
            alt.Chart(etype)
            .mark_bar(color=ACCENT, cornerRadiusEnd=4)
            .encode(
                x=alt.X("n:Q", title="events"),
                y=alt.Y("type:N", title=None, sort="-x"),
            )
        )
        col2.markdown('<div class="caption">Event-type mix</div>', True)
        col2.altair_chart(
            style_chart(etype_bar, height=240), use_container_width=True, theme=None
        )

    if AVAIL.get("features"):
        feat = ART.load_features()
        feat_cols = [c for c in feat.columns if c.startswith("purchase_")]
        section("Feature matrix")
        caption(f"{feat.shape[0]:,} rows × {len(feat_cols)} features")
        st.dataframe(feat[feat_cols].describe().T, use_container_width=True)

# --------------------------------------------------------------------------- #
# Live scoring (demonstrates the single compute() path)
# --------------------------------------------------------------------------- #
with scoring:
    section("Score one card with the training-time feature code")
    caption(
        "Runs the same point-in-time featurizer used in training "
        "(transform_single) — the visible proof there is no training-serving skew."
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
        buyers = (
            events.loc[events["event_type"] == "purchase", "user_id"]
            .value_counts()
            .index.tolist()
        )
        user_id = c2.selectbox("Card", buyers[:200] or events["user_id"].unique()[:200])

        history = events[events["user_id"] == user_id].sort_values("event_timestamp")
        ref_time = history["event_timestamp"].max()

        pipe = FeaturePipeline()
        row = pipe.transform_single(history, ref_time)
        x = pd.DataFrame([row])[feature_cols]

        model = ART.load_model(model_name)
        proba = float(model.predict_proba(x)[:, 1][0])

        if proba < 0.02:
            band, band_color = "Low risk", GOOD
        elif proba < 0.10:
            band, band_color = "Elevated risk", WARNING
        else:
            band, band_color = "High risk", CRITICAL

        left, right = st.columns([1, 2])
        kpi(
            left,
            f"Fraud probability · {model_name}",
            f"{proba:.1%}",
            "",
            accent=band_color,
        )
        right.write("")
        right.markdown(
            f'<span class="risk-pill" style="background:{band_color}">{band}</span>',
            unsafe_allow_html=True,
        )
        right.progress(min(max(proba, 0.0), 1.0))

        with st.expander(f"Card event history ({len(history)} events)"):
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

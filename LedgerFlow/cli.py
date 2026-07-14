"""
``ledgerflow`` command-line interface.

A thin, dependency-free (stdlib ``argparse``) wrapper over the pipeline stages so
LedgerFlow is usable as a tool, not just a library:

    ledgerflow run --source synthetic          # zero-download end-to-end demo
    ledgerflow run --events fraudTrain.csv      # real Sparkov data
    ledgerflow score --user cc_1234             # score one card live
    ledgerflow dashboard                        # launch the Streamlit UI

Every command wraps the same ``*_main()`` entry points the DVC pipeline calls, so
the CLI and ``dvc repro`` do byte-for-byte identical work — no logic is
duplicated here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Quiet the Pandera import banner before any heavy import happens.
os.environ.setdefault("DISABLE_PANDERA_IMPORT_WARNING", "True")

from . import __version__

# --------------------------------------------------------------------------- #
# Tiny terminal styling (no dependency on rich/colorama)
# --------------------------------------------------------------------------- #
_TTY = sys.stdout.isatty()


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


def _bold(s: str) -> str:
    return _c("1", s)


def _green(s: str) -> str:
    return _c("32", s)


def _cyan(s: str) -> str:
    return _c("36", s)


def _dim(s: str) -> str:
    return _c("2", s)


def _yellow(s: str) -> str:
    return _c("33", s)


def _red(s: str) -> str:
    return _c("31", s)


def _step(n: int, total: int, msg: str) -> None:
    print(f"{_cyan(f'[{n}/{total}]')} {_bold(msg)}")


def _ok(msg: str) -> None:
    print(f"  {_green('✓')} {msg}")


def _info(msg: str) -> None:
    print(f"  {_dim(msg)}")


def _err(msg: str) -> None:
    print(f"{_red('error:')} {msg}", file=sys.stderr)


# --------------------------------------------------------------------------- #
# Stage wrappers
# --------------------------------------------------------------------------- #
def _ingest(args: argparse.Namespace) -> None:
    """Run the ingest stage, writing data/raw/{events,labels}.parquet."""
    if args.source == "synthetic":
        from .data.synthetic import generate_main

        generate_main(n_users=args.n_users, days=args.days)
        _ok(f"synthetic events generated ({args.n_users} users, {args.days} days)")
        return

    from .data.sparkov import sparkov_main

    sparkov_main(
        csv_path=args.events,
        label_horizon_days=args.horizon,
        snapshot_freq=args.snapshot,
    )
    summary_path = Path("data/raw/ingest_summary.json")
    if summary_path.exists():
        s = json.loads(summary_path.read_text())
        _ok(
            f"{s['n_events']:,} events · {s['n_users']:,} cards → "
            f"{s['n_samples']:,} {s['snapshot_freq']}-snapshots "
            f"({100 * s['positive_rate']:.2f}% positive)"
        )


def _featurize(_: argparse.Namespace) -> None:
    from .pipeline import featurize_main

    featurize_main()
    rt_path = Path("reports/feature_runtime.json")
    if rt_path.exists():
        rt = json.loads(rt_path.read_text())
        _ok(
            f"{rt['n_rows']:,} × {rt['n_features']} feature matrix "
            f"in {rt['runtime_seconds']:.0f}s"
        )


def _split(_: argparse.Namespace) -> None:
    from .data.splitter import split_main

    paths = split_main()
    _ok(f"time-ordered splits written ({', '.join(sorted(paths))})")


def _evaluate(_: argparse.Namespace) -> None:
    from .evaluation.compare import evaluate_main

    evaluate_main()
    _print_recommendation()


def _report(_: argparse.Namespace) -> None:
    from .evaluation.report import generate_memo

    out = generate_memo()
    _ok(f"recommendation memo → {out}")


def _print_recommendation() -> None:
    """Print the recommended model + a compact metrics table, if available."""
    metrics_path = Path("reports/eval_metrics.json")
    if not metrics_path.exists():
        return
    ev = json.loads(metrics_path.read_text())
    rec = ev.get("recommended")
    print()
    print(f"  {_bold('Recommended:')} {_green(rec)}")
    header = f"  {'model':<22}{'AUC':>8}{'Brier':>9}{'AvgPrec':>9}"
    print(_dim(header))
    for name, m in ev["models"].items():
        marker = _green("→") if name == rec else " "
        row = (
            f"{marker} {name:<22}{m['auc_roc']:>8.3f}"
            f"{m['brier_score']:>9.3f}{m['avg_precision']:>9.3f}"
        )
        print(row)


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_run(args: argparse.Namespace) -> int:
    """Full pipeline: ingest → featurize → split → evaluate → report."""
    total = 4 if args.skip_report else 5
    src = "synthetic generator" if args.source == "synthetic" else "Sparkov"
    print(_bold(f"LedgerFlow · running the pipeline on {src}\n"))

    _step(1, total, "Ingest")
    _ingest(args)
    _step(2, total, "Featurize (point-in-time)")
    _featurize(args)
    _step(3, total, "Split (time-ordered)")
    _split(args)
    _step(4, total, "Evaluate (LogReg · XGBoost · LightGBM)")
    _evaluate(args)
    if not args.skip_report:
        _step(5, total, "Report")
        _report(args)

    print()
    print(_green("✓ pipeline complete."))
    print(_dim("  Explore results: ledgerflow dashboard"))
    return 0


def cmd_score(args: argparse.Namespace) -> int:
    """Score a single card/user through the training-time feature code."""
    import pandas as pd

    events_path = Path("data/raw/events.parquet")
    if not events_path.exists():
        _err(
            "no event data found at data/raw/events.parquet. "
            "Run `ledgerflow run` first."
        )
        return 1

    events = pd.read_parquet(events_path)
    history = events[events["user_id"] == args.user]
    if history.empty:
        sample = ", ".join(events["user_id"].drop_duplicates().head(5).tolist())
        _err(f"no events for user '{args.user}'. Examples: {sample}")
        return 1

    # Pick the model: explicit flag, else the recommended one, else first found.
    model_name = args.model
    if model_name is None:
        metrics_path = Path("reports/eval_metrics.json")
        if metrics_path.exists():
            model_name = json.loads(metrics_path.read_text()).get("recommended")
    models_dir = Path("models")
    if model_name is None or not (models_dir / model_name / "model.pkl").exists():
        available = [p.name for p in models_dir.glob("*") if (p / "model.pkl").exists()]
        if not available:
            _err("no trained models found. Run `ledgerflow run` first.")
            return 1
        model_name = model_name if model_name in available else available[0]

    import joblib

    from .features.time_windows import ALL_FEATURES
    from .pipeline import FeaturePipeline

    history = history.sort_values("event_timestamp")
    ref_time = history["event_timestamp"].max()
    row = FeaturePipeline().transform_single(history, ref_time)
    x = pd.DataFrame([row])[[f.name for f in ALL_FEATURES]]

    model = joblib.load(models_dir / model_name / "model.pkl")
    proba = float(model.predict_proba(x)[:, 1][0])

    if proba < 0.02:
        band, paint = "LOW", _green
    elif proba < 0.10:
        band, paint = "ELEVATED", _yellow
    else:
        band, paint = "HIGH", _red

    print(f"{_bold('user')}      {args.user}")
    print(f"{_bold('model')}     {model_name}")
    print(f"{_bold('events')}    {len(history):,} (as of {ref_time})")
    print(f"{_bold('fraud p')}   {paint(f'{proba:.1%}')}  [{paint(band + ' risk')}]")
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    """Launch the Streamlit dashboard over the pipeline outputs."""
    import shutil
    import subprocess

    app = Path(__file__).resolve().parent.parent / "dashboard" / "app.py"
    if not app.exists():
        _err(f"dashboard app not found at {app}")
        return 1
    if shutil.which("streamlit") is None:
        _err(
            "streamlit is not installed. "
            "Install it with: pip install 'ledgerflow[dashboard]'"
        )
        return 1
    cmd = ["streamlit", "run", str(app), "--server.port", str(args.port)]
    print(_dim(f"$ {' '.join(cmd)}"))
    return subprocess.call(cmd)


def cmd_version(_: argparse.Namespace) -> int:
    print(f"ledgerflow {__version__}")
    return 0


# Individual-stage commands: run one stage, always exit 0 on success.
def cmd_ingest(args: argparse.Namespace) -> int:
    _ingest(args)
    return 0


def cmd_featurize(args: argparse.Namespace) -> int:
    _featurize(args)
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    _split(args)
    return 0


def cmd_evaluate(args: argparse.Namespace) -> int:
    _evaluate(args)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    _report(args)
    return 0


# --------------------------------------------------------------------------- #
# Parser
# --------------------------------------------------------------------------- #
def _add_ingest_opts(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--source",
        choices=["sparkov", "synthetic"],
        default="sparkov",
        help="ingest source (default: sparkov)",
    )
    p.add_argument(
        "--events",
        metavar="CSV",
        default=None,
        help="path to the Sparkov CSV (or set LEDGERFLOW_SPARKOV_CSV)",
    )
    p.add_argument(
        "--snapshot",
        choices=["D", "W"],
        default=None,
        help="snapshot cadence: D=daily, W=weekly (default: params.yaml)",
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="label horizon in snapshot periods (default: params.yaml)",
    )
    p.add_argument(
        "--n-users",
        type=int,
        default=2000,
        help="synthetic source: number of users (default: 2000)",
    )
    p.add_argument(
        "--days",
        type=int,
        default=30,
        help="synthetic source: history length in days (default: 30)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ledgerflow",
        description="Turn raw event logs into leakage-free ML features, models, "
        "and a data-driven model recommendation.",
        epilog="examples:\n"
        "  ledgerflow run --source synthetic        # zero-download demo\n"
        "  ledgerflow run --events fraudTrain.csv   # real Sparkov data\n"
        "  ledgerflow score --user cc_1234\n"
        "  ledgerflow dashboard",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", metavar="<command>")

    run_p = sub.add_parser("run", help="run the full pipeline end-to-end")
    _add_ingest_opts(run_p)
    run_p.add_argument(
        "--skip-report", action="store_true", help="stop after evaluation"
    )
    run_p.set_defaults(func=cmd_run)

    ing_p = sub.add_parser("ingest", help="ingest raw events + labels")
    _add_ingest_opts(ing_p)
    ing_p.set_defaults(func=cmd_ingest)

    feat_p = sub.add_parser("featurize", help="compute the point-in-time features")
    feat_p.set_defaults(func=cmd_featurize)

    split_p = sub.add_parser("split", help="time-ordered train/val/test split")
    split_p.set_defaults(func=cmd_split)

    eval_p = sub.add_parser("evaluate", help="train + compare models")
    eval_p.set_defaults(func=cmd_evaluate)

    rep_p = sub.add_parser("report", help="render the recommendation memo")
    rep_p.set_defaults(func=cmd_report)

    score_p = sub.add_parser("score", help="score a single user/card")
    score_p.add_argument("--user", required=True, help="user_id to score")
    score_p.add_argument(
        "--model", default=None, help="model name (default: recommended)"
    )
    score_p.set_defaults(func=cmd_score)

    dash_p = sub.add_parser("dashboard", help="launch the Streamlit dashboard")
    dash_p.add_argument("--port", type=int, default=8501, help="port (default: 8501)")
    dash_p.set_defaults(func=cmd_dashboard)

    ver_p = sub.add_parser("version", help="print the version")
    ver_p.set_defaults(func=cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

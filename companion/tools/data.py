"""Data analysis + plotting capabilities — Pandas / NumPy / Matplotlib.

Accepts CSVs, rows of dicts, lists of lists, or a raw CSV string, then
produces a profile, aggregations and (optionally) a chart. All imports are
lazy.
"""

from __future__ import annotations

import base64
import io
import json
from typing import Any

from .common import as_rows, load


def run_data_analysis(payload: dict[str, Any]) -> dict[str, Any]:
    """Profile, clean and summarize a dataset (Pandas + NumPy)."""
    data = payload.get("data")
    params = payload.get("params") or {}
    if data is None:
        raise ValueError(
            "No data provided. Send `data` as a CSV string, a list of dicts "
            "(rows), or a dict with `rows`/`columns`."
        )

    pd = load("pandas")
    np = load("numpy")

    columns, rows = as_rows(data)
    if not rows:
        # A plain list of numbers is still analyzable as a single series.
        from .common import as_numbers

        numbers = as_numbers(data)
        if numbers:
            df = pd.DataFrame({"value": numbers})
            df = df.reset_index().rename(columns={"index": "index"})
        else:
            raise ValueError(
                "Could not parse tabular data. Send a CSV string, a list of "
                "row-dicts, or a dict with `rows`/`columns`."
            )
    else:
        df = pd.DataFrame(rows, columns=columns if columns else None)
    if df.shape[1] == 0:
        df = df.reset_index(drop=True)
    # Coerce numeric-looking columns (CSV cells arrive as strings).
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except (ValueError, TypeError):
            pass

    profile: dict[str, Any] = {
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "column_names": [str(c) for c in df.columns],
        "dtypes": {str(c): str(t) for c, t in df.dtypes.items()},
        "missing": {str(c): int(df[c].isna().sum()) for c in df.columns},
        "duplicate_rows": int(df.duplicated().sum()),
    }

    numeric_cols = list(df.select_dtypes(include=[np.number]).columns)
    profile["numeric_columns"] = [str(c) for c in numeric_cols]
    profile["numeric_summary"] = {}
    for c in numeric_cols:
        desc = df[c].describe().to_dict()
        profile["numeric_summary"][str(c)] = {
            "mean": float(desc.get("mean", 0)),
            "std": float(desc.get("std", 0)),
            "min": float(desc.get("min", 0)),
            "q1": float(desc.get("25%", 0)),
            "median": float(desc.get("50%", 0)),
            "q3": float(desc.get("75%", 0)),
            "max": float(desc.get("max", 0)),
        }

    # Categorical top values (capped).
    categorical_cols = [c for c in df.columns if c not in numeric_cols]
    profile["categorical_summary"] = {}
    for c in categorical_cols:
        counts = df[c].dropna().astype(str).value_counts().head(5)
        profile["categorical_summary"][str(c)] = [
            {"value": k, "count": int(v)} for k, v in counts.items()
        ]

    # Correlation highlights for numeric columns (top 5 pairs).
    if len(numeric_cols) >= 2:
        corr = df[numeric_cols].corr().unstack().dropna()
        corr = corr[abs(corr) < 1.0]
        if not corr.empty:
            top = corr.abs().sort_values(ascending=False).head(5)
            profile["top_correlations"] = [
                {"columns": [str(a), str(b)], "correlation": round(float(corr[(a, b)]), 4)}
                for (a, b) in top.index
            ]

    # Optional grouping.
    group_by = params.get("group_by")
    if group_by and group_by in [str(c) for c in df.columns] and numeric_cols:
        agg = params.get("agg") or "mean"
        grouped = (
            df.groupby(str(group_by), dropna=False)[numeric_cols[0]].agg(agg).head(10)
        )
        profile["grouped"] = {
            "column": str(group_by),
            "agg": str(agg),
            "target": str(numeric_cols[0]),
            "values": {str(k): (round(float(v), 4) if isinstance(v, (int, float)) else str(v)) for k, v in grouped.items()},
        }

    # Optional chart alongside the profile.
    if params.get("chart"):
        chart = _chart(df, params, numeric_cols, pd)
        profile["chart"] = chart

    summary = (
        f"{profile['rows']} rows × {profile['columns']} columns "
        f"({len(numeric_cols)} numeric). "
    )
    if profile["missing"] and any(v for v in profile["missing"].values()):
        total_missing = sum(profile["missing"].values())
        summary += f"{total_missing} missing cells. "
    if profile.get("top_correlations"):
        strongest = profile["top_correlations"][0]
        summary += (
            f"Strongest correlation: {strongest['columns'][0]} ↔ "
            f"{strongest['columns'][1]} ({strongest['correlation']:.2f})."
        )

    return {
        "result": profile,
        "summary": summary,
        "libraries_used": ["pandas", "numpy"],
    }


def run_plot(payload: dict[str, Any]) -> dict[str, Any]:
    """Render a chart from tabular data (Matplotlib → base64 PNG)."""
    data = payload.get("data")
    params = payload.get("params") or {}
    if data is None:
        raise ValueError("No data provided for plotting.")

    pd = load("pandas")
    np = load("numpy")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    columns, rows = as_rows(data)
    if not rows:
        from .common import as_numbers

        numbers = as_numbers(data)
        if numbers:
            df = pd.DataFrame({"index": range(len(numbers)), "value": numbers})
        else:
            raise ValueError("Could not parse tabular data for plotting.")
    else:
        df = pd.DataFrame(rows, columns=columns if columns else None)
    kind = str(params.get("kind") or _detect_kind(payload.get("task") or "")).lower()

    fig, ax = plt.subplots(figsize=(7, 4.5))
    try:
        if kind == "hist":
            column = str(params.get("x") or params.get("column") or df.columns[0])
            df[column] = pd.to_numeric(df[column], errors="coerce")
            ax.hist(df[column].dropna().astype(float), bins=int(params.get("bins") or 20))
            ax.set_xlabel(column)
            ax.set_ylabel("count")
            title = f"Histogram of {column}"
        elif kind == "scatter":
            xc = str(params.get("x") or df.columns[0])
            yc = str(params.get("y") or df.columns[1] if len(df.columns) > 1 else df.columns[0])
            xs = pd.to_numeric(df[xc], errors="coerce").astype(float)
            ys = pd.to_numeric(df[yc], errors="coerce").astype(float)
            ax.scatter(xs, ys, alpha=0.6)
            ax.set_xlabel(xc)
            ax.set_ylabel(yc)
            title = f"{yc} vs {xc}"
        elif kind == "bar":
            xc = str(params.get("x") or df.columns[0])
            yc = str(params.get("y") or (df.columns[1] if len(df.columns) > 1 else None))
            if yc is None:
                counts = df[xc].astype(str).value_counts().head(int(params.get("top") or 10))
                labels, heights = list(counts.index), list(counts)
            else:
                agg = df.groupby(xc)[yc].sum()
                labels, heights = [str(k) for k in agg.index], list(agg)
            ax.bar(labels[: int(params.get("top") or 10)], heights[: int(params.get("top") or 10)])
            ax.set_xlabel(xc)
            ax.set_ylabel(yc or "count")
            ax.tick_params(axis="x", rotation=45)
            title = f"{yc or 'count'} by {xc}"
        else:  # line
            xc = str(params.get("x") or df.columns[0])
            yc = str(params.get("y") or df.columns[1] if len(df.columns) > 1 else df.columns[0])
            xs = pd.to_numeric(df[xc], errors="coerce")
            ys = pd.to_numeric(df[yc], errors="coerce")
            order = xs.argsort()
            ax.plot(xs.iloc[order], ys.iloc[order])
            ax.set_xlabel(xc)
            ax.set_ylabel(yc)
            title = f"{yc} over {xc}"
        ax.set_title(title)
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        png = base64.b64encode(buf.read()).decode("ascii")
    finally:
        plt.close(fig)

    return {
        "result": {
            "kind": kind,
            "title": title,
            "png_base64": png,
            "image_url": f"data:image/png;base64,{png[:64]}...",
        },
        "summary": f"Rendered a {kind} chart: {title}.",
        "libraries_used": ["matplotlib", "pandas", "numpy"],
    }


def _chart(df, params: dict[str, Any], numeric_cols: list, pd) -> dict[str, Any]:
    """Small inline chart reused by run_data_analysis when params.chart is set."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    kind = str(params.get("chart") or "bar").lower()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    try:
        if kind == "hist" and numeric_cols:
            col = numeric_cols[0]
            ax.hist(pd.to_numeric(df[col], errors="coerce").dropna().astype(float), bins=20)
            ax.set_title(f"Histogram of {col}")
        elif numeric_cols and len(numeric_cols) >= 2:
            ax.scatter(
                pd.to_numeric(df[numeric_cols[0]], errors="coerce").astype(float),
                pd.to_numeric(df[numeric_cols[1]], errors="coerce").astype(float),
                alpha=0.6,
            )
            ax.set_title(f"{numeric_cols[1]} vs {numeric_cols[0]}")
        else:
            col = df.columns[0]
            counts = df[col].astype(str).value_counts().head(10)
            ax.bar([str(k) for k in counts.index], list(counts))
            ax.tick_params(axis="x", rotation=45)
            ax.set_title(f"{col} counts")
        fig.tight_layout()
        buf = io.BytesIO()
        fig.savefig(buf, format="png")
        buf.seek(0)
        png = base64.b64encode(buf.read()).decode("ascii")
    finally:
        plt.close(fig)
    return {"kind": kind, "png_base64": png}


def _detect_kind(task: str) -> str:
    text = task.lower()
    if "histogram" in text or "distribution" in text:
        return "hist"
    if "scatter" in text:
        return "scatter"
    if "bar" in text:
        return "bar"
    return "line"

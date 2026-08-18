#!/usr/bin/env python3
"""
plot_hurst_baseline_panels.py

Figure-12-style(Steinburgge Hurst exponent vs baseline panels for icy-world ROI summary workbooks.

This version is designed for workbooks like roi_grouped_summary.xlsx with sheets such as:
    Ganymede_Product_Good
    Europa_Product_Good
    Tethys_Product_GOod
    Enceladus_Product_GOod

What the plot means
-------------------
For each ROI with an accepted breakpoint, the script creates two plotted points:
    Short-scale point: x = Pixel Scale m/pixel, y = H Before BP
    Long-scale point:  x = Breakpoint m,        y = H After BP

python scripts/plotting/plot_hurst_baseline_panels_clean.py \
  --excel roi_grouped_summary.xlsx \
  --sheets Ganymede_Product_Good Europa_Product_Good Enceladus_Product_Good Tethys_Product_Good \
  --output-dir figures/figure_individual_clean \
  --split-by-body \
  --plot-mode individual \
  --color-by Site \
  --legend-position right \
  --legend-ncols 4 \
  --max-legend-label-chars 16 \
  --verbose

For the mean ± standard deviation version
Use aggregate mode:
python scripts/plotting/plot_hurst_baseline_panels_clean.py \
  --excel roi_grouped_summary.xlsx \
  --sheets Ganymede_Product_Good Europa_Product_Good Enceladus_Product_Good \
  --output-dir figures/figure_aggregate_clean \
  --split-by-body \
  --plot-mode aggregate \
  --color-by Site \
  --legend-position right \
  --legend-ncols 4 \
  --max-legend-label-chars 16 \
  --verbose
Figure design
-------------
    - One panel per terrain type.
    - One separate figure per body/world when --split-by-body is used.
    - Colors represent sites/study areas by default.
    - Marker shapes represent product types by default.
    - Optional thin lines connect the short-scale and long-scale points for the same ROI.
    - Optional aggregate mode shows mean +/- standard deviation error bars.

Important note
--------------
In aggregate mode, error bars show spread across ROIs in the same group. They are NOT
vertical DTM errors and NOT Monte Carlo uncertainties unless your input H values already
represent those uncertainty calculations.
"""

from __future__ import annotations

import argparse
import math
import re
import textwrap
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DEFAULT_SHEETS = [
    "Ganymede_Product_Good",
    "Europa_Product_Good",
    "Enceladus_Product_Good",
]

KNOWN_BODIES = ["Europa", "Ganymede", "Enceladus", "Tethys"]

CANONICAL_COLUMNS = {
    "roi_id": "ROI_ID",
    "roi id": "ROI_ID",
    "body": "Body",
    "dataset": "Dataset",
    "site": "Site",
    "terrain type": "Terrain Type",
    "terrain_type": "Terrain Type",
    "product type": "Product Type",
    "product_type": "Product Type",
    "pixel scale m/pixel": "Pixel Scale m/pixel",
    "pixel scale": "Pixel Scale m/pixel",
    "pixel_scale_m_per_pixel": "Pixel Scale m/pixel",
    "fit min baseline m": "Fit Min Baseline m",
    "fit max baseline m": "Fit Max Baseline m",
    "fit min pixels": "Fit Min Pixels",
    "fit max pixels": "Fit Max Pixels",
    "fit range pixels": "Fit Range Pixels",
    "bad pixel fraction": "Bad Pixel Fraction",
    "qc status": "QC Status",
    "breakpoint accepted": "Breakpoint Accepted",
    "breakpoint ac": "Breakpoint Accepted",
    "accepted breakpoint": "Breakpoint Accepted",
    "breakpoint m": "Breakpoint m",
    "breakpoint pixels": "Breakpoint pixels",
    "h overall": "H Overall",
    "h before bp": "H Before BP",
    "h after bp": "H After BP",
    "delta h": "Delta H",
    "delta h before": "Delta H",
    "delta h before-after": "Delta H",
    "delta h before after": "Delta H",
    "delta h interpretation": "Delta H Interpretation",
}

REQUIRED_FOR_FIG12 = [
    "ROI_ID",
    "Body",
    "Terrain Type",
    "Product Type",
    "Pixel Scale m/pixel",
    "Breakpoint m",
    "H Before BP",
    "H After BP",
]

SUMMARY_WORDS = (
    "n rois", "n roi", "median", "mean", "min", "max", "range", "std", "count",
    "figure", "plot", "interpretation", "caption", "body", "site", "terrain", "product",
    "notes", "summary", "use", "check",
)

PRODUCT_MARKERS = {
    "Stereo": "o",
    "SFS(T)": "s",
    "SFS": "s",
    "Combined(Stereo + SFS[ZT])": "^",
    "Combined(Stereo + SFS[ZT]) ": "^",
    "Combined": "^",
    "Combined Stereo + SFS": "^",
    "Combined(Stereo+SFS[ZT])": "^",
}

FALLBACK_MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">", "*"]

# -----------------------------
# Plot styling for readability
# -----------------------------
FIG_W_PER_COL = 5.3
FIG_H_PER_ROW = 4.2

SUPTITLE_FS = 19
PANEL_TITLE_FS = 14
AXIS_LABEL_FS = 12
TICK_LABEL_FS = 10

LEGEND_FS = 10
LEGEND_TITLE_FS = 11

MARKER_SIZE = 70
ERRORBAR_MARKER_SIZE = 7.5
LINE_WIDTH = 1.2
GRID_ALPHA = 0.35

WSPACE = 0.22
HSPACE = 0.32

def normalize_text(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip().replace("\n", " ")
    return re.sub(r"\s+", " ", text)


def canonicalize_columns(columns: Iterable[object]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for col in columns:
        raw = normalize_text(col)
        low = raw.lower()
        canonical = CANONICAL_COLUMNS.get(low, raw)
        if canonical in seen:
            seen[canonical] += 1
            canonical = f"{canonical}_{seen[canonical]}"
        else:
            seen[canonical] = 0
        out.append(canonical)
    return out


def make_unique_columns(columns: Sequence[str]) -> list[str]:
    seen: dict[str, int] = {}
    out: list[str] = []
    for col in columns:
        name = str(col)
        if name in seen:
            seen[name] += 1
            name = f"{name}_{seen[name]}"
        else:
            seen[name] = 0
        out.append(name)
    return out


def infer_body_from_sheet(sheet_name: str) -> str | None:
    low = sheet_name.lower()
    for body in KNOWN_BODIES:
        if body.lower() in low:
            return body
    return None


def parse_bool(value: object) -> bool:
    if isinstance(value, bool):
        return bool(value)
    if pd.isna(value):
        return False
    text = str(value).strip().lower()
    return text in {"true", "t", "yes", "y", "1", "accepted", "accept", "ok"}


def to_numeric(series: pd.Series) -> pd.Series:
    def convert_one(x: object) -> float:
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.number)):
            return float(x)
        text = str(x).strip().replace(",", "")
        if text == "":
            return np.nan
        if text.endswith("%"):
            try:
                return float(text[:-1]) / 100.0
            except ValueError:
                return np.nan
        try:
            return float(text)
        except ValueError:
            return np.nan
    return series.map(convert_one)


def safe_name(text: object) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", str(text)).strip("_") or "unknown"


def row_has_roi_header(row: pd.Series) -> bool:
    normalized = [normalize_text(v).lower() for v in row.values]
    return "roi_id" in normalized or "roi id" in normalized


def clean_roi_rows(df: pd.DataFrame, inferred_body: str | None = None, sheet_name: str | None = None) -> pd.DataFrame:
    df = df.copy()
    df.columns = canonicalize_columns(df.columns)
    if "ROI_ID" not in df.columns:
        return pd.DataFrame()

    df = df[df["ROI_ID"].notna()].copy()
    df["ROI_ID"] = df["ROI_ID"].astype(str).str.strip()
    df = df[df["ROI_ID"] != ""]

    low = df["ROI_ID"].astype(str).str.strip().str.lower()
    df = df[~low.isin({"roi_id", "roi id"})]
    df = df[~low.str.startswith(SUMMARY_WORDS)]

    # Fill/infer body if a sheet already represents one body and Body cells are blank/merged.
    if "Body" not in df.columns:
        df["Body"] = inferred_body or "Unknown"
    else:
        body_text = df["Body"].map(normalize_text)
        if inferred_body:
            df.loc[body_text.eq("") | body_text.str.lower().isin({"nan", "unknown"}), "Body"] = inferred_body
        df["Body"] = df["Body"].fillna(inferred_body or "Unknown")

    if "Site" not in df.columns:
        df["Site"] = "Unknown"
    if "Product Type" not in df.columns:
        df["Product Type"] = "Unknown"
    if sheet_name is not None:
        df["Source Sheet"] = sheet_name

    # Remove rows that still look like summary rows by checking that at least one core numeric column exists.
    core_numeric = [c for c in ["Pixel Scale m/pixel", "Breakpoint m", "H Before BP", "H After BP", "H Overall"] if c in df.columns]
    if core_numeric:
        numeric_any = pd.Series(False, index=df.index)
        for c in core_numeric:
            numeric_any = numeric_any | to_numeric(df[c]).notna()
        df = df[numeric_any].copy()

    return df.reset_index(drop=True)


def read_block_style_sheet(excel_path: Path, sheet_name: str, verbose: bool = False) -> pd.DataFrame:
    inferred_body = infer_body_from_sheet(sheet_name)
    raw = pd.read_excel(excel_path, sheet_name=sheet_name, header=None, dtype=object)
    header_rows = [idx for idx, row in raw.iterrows() if row_has_roi_header(row)]

    if not header_rows:
        df = pd.read_excel(excel_path, sheet_name=sheet_name, dtype=object)
        return clean_roi_rows(df, inferred_body=inferred_body, sheet_name=sheet_name)

    blocks: list[pd.DataFrame] = []
    header_rows_plus_end = header_rows + [len(raw)]

    for i in range(len(header_rows_plus_end) - 1):
        start = header_rows_plus_end[i]
        end = header_rows_plus_end[i + 1]
        header = raw.iloc[start].tolist()
        keep_cols = [j for j, v in enumerate(header) if normalize_text(v) != ""]
        if not keep_cols:
            continue
        columns = canonicalize_columns([header[j] for j in keep_cols])
        columns = make_unique_columns(columns)
        data = raw.iloc[start + 1:end, keep_cols].copy()
        data.columns = columns
        data = clean_roi_rows(data, inferred_body=inferred_body, sheet_name=sheet_name)
        if not data.empty:
            blocks.append(data)

    if not blocks:
        if verbose:
            print(f"[warning] {sheet_name}: found ROI_ID header rows, but no data blocks survived cleaning.")
        return pd.DataFrame()
    return pd.concat(blocks, ignore_index=True, sort=False)


def resolve_sheet_names(excel_path: Path, requested_sheets: Sequence[str]) -> list[str]:
    xls = pd.ExcelFile(excel_path)
    available = xls.sheet_names
    lower_map = {s.lower(): s for s in available}
    resolved: list[str] = []
    missing: list[str] = []
    for name in requested_sheets:
        if name in available:
            resolved.append(name)
        elif name.lower() in lower_map:
            resolved.append(lower_map[name.lower()])
        else:
            missing.append(name)
    if missing:
        print("[warning] These requested sheets were not found and will be skipped:", missing)
        print("[info] Available sheets are:")
        for s in available:
            print("  -", repr(s))
    # Preserve order but deduplicate.
    final: list[str] = []
    for s in resolved:
        if s not in final:
            final.append(s)
    return final


def calculate_missing_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = canonicalize_columns(df.columns)

    numeric_candidates = [
        "Pixel Scale m/pixel",
        "Fit Min Baseline m",
        "Fit Max Baseline m",
        "Fit Min Pixels",
        "Fit Max Pixels",
        "Fit Range Pixels",
        "Bad Pixel Fraction",
        "Breakpoint m",
        "Breakpoint pixels",
        "H Overall",
        "H Before BP",
        "H After BP",
        "Delta H",
    ]
    for col in numeric_candidates:
        if col in df.columns:
            df[col] = to_numeric(df[col])

    if "Breakpoint Accepted" not in df.columns:
        # If no accepted column exists, infer accepted only where all breakpoint fields exist.
        df["Breakpoint Accepted Bool"] = (
            df.get("Breakpoint m", pd.Series(np.nan, index=df.index)).notna()
            & df.get("H Before BP", pd.Series(np.nan, index=df.index)).notna()
            & df.get("H After BP", pd.Series(np.nan, index=df.index)).notna()
        )
        df["Breakpoint Accepted"] = df["Breakpoint Accepted Bool"]
    else:
        df["Breakpoint Accepted Bool"] = df["Breakpoint Accepted"].map(parse_bool)

    if "QC Status" not in df.columns:
        # Product_Good sheets are usually already filtered to Good, so do not remove them later.
        df["QC Status"] = "Good"
    else:
        df["QC Status"] = df["QC Status"].fillna("Good").astype(str).str.strip()

    if "Breakpoint pixels" not in df.columns and {"Breakpoint m", "Pixel Scale m/pixel"}.issubset(df.columns):
        df["Breakpoint pixels"] = df["Breakpoint m"] / df["Pixel Scale m/pixel"]
    if "Delta H" not in df.columns and {"H Before BP", "H After BP"}.issubset(df.columns):
        df["Delta H"] = df["H Before BP"] - df["H After BP"]
    if "Fit Min Pixels" not in df.columns and {"Fit Min Baseline m", "Pixel Scale m/pixel"}.issubset(df.columns):
        df["Fit Min Pixels"] = df["Fit Min Baseline m"] / df["Pixel Scale m/pixel"]
    if "Fit Max Pixels" not in df.columns and {"Fit Max Baseline m", "Pixel Scale m/pixel"}.issubset(df.columns):
        df["Fit Max Pixels"] = df["Fit Max Baseline m"] / df["Pixel Scale m/pixel"]
    if "Fit Range Pixels" not in df.columns and {"Fit Min Pixels", "Fit Max Pixels"}.issubset(df.columns):
        df["Fit Range Pixels"] = df["Fit Max Pixels"] - df["Fit Min Pixels"]

    for col in ["ROI_ID", "Body", "Site", "Terrain Type", "Product Type", "Source Sheet"]:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()
            df[col] = df[col].replace({"": "Unknown", "nan": "Unknown", "None": "Unknown"})

    # Clean a few product-name variants.
    if "Product Type" in df.columns:
        df["Product Type"] = df["Product Type"].str.replace("Combined(Stereo+SFS", "Combined(Stereo + SFS", regex=False)
    return df


def load_all_data(excel_path: Path, sheets: Sequence[str], verbose: bool) -> pd.DataFrame:
    resolved = resolve_sheet_names(excel_path, sheets)
    if not resolved:
        raise ValueError("No requested sheets were found. Check --sheets against the actual Excel sheet names.")

    frames: list[pd.DataFrame] = []
    for sheet in resolved:
        df = read_block_style_sheet(excel_path, sheet, verbose=verbose)
        if not df.empty:
            # Force body from sheet where Body was missing or merged/blank.
            inferred = infer_body_from_sheet(sheet)
            if inferred and "Body" in df.columns:
                body_clean = df["Body"].map(normalize_text)
                df.loc[body_clean.eq("") | body_clean.str.lower().isin({"unknown", "nan"}), "Body"] = inferred
            frames.append(df)
        if verbose:
            print(f"[load] {sheet}: {len(df)} candidate ROI rows")
            if not df.empty:
                cols_preview = [c for c in ["Body", "Terrain Type", "Product Type", "Site"] if c in df.columns]
                if cols_preview:
                    print(df[cols_preview].head(8).to_string(index=False))

    if not frames:
        raise ValueError("No ROI rows were found. The workbook may not contain ROI_ID data blocks in the requested sheets.")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    combined = calculate_missing_columns(combined)
    return combined


def make_long_hurst_table(
    df: pd.DataFrame,
    qc: str,
    bodies: Sequence[str] | None,
    products: Sequence[str] | None,
    include_unaccepted: bool = False,
) -> pd.DataFrame:
    df = calculate_missing_columns(df)
    missing = [c for c in REQUIRED_FOR_FIG12 if c not in df.columns]
    if missing:
        raise ValueError(
            "Missing required columns for Figure-12-style plot: "
            f"{missing}\nAvailable columns: {list(df.columns)}"
        )

    use = df.copy()
    start_n = len(use)
    if qc.lower() != "all" and "QC Status" in use.columns:
        use = use[use["QC Status"].astype(str).str.lower() == qc.lower()].copy()
    after_qc = len(use)

    if not include_unaccepted:
        use = use[use["Breakpoint Accepted Bool"]].copy()
    after_bp = len(use)

    if bodies:
        use = use[use["Body"].isin(bodies)].copy()
    after_body = len(use)

    if products:
        use = use[use["Product Type"].isin(products)].copy()
    after_product = len(use)

    use = use.dropna(subset=["Pixel Scale m/pixel", "Breakpoint m", "H Before BP", "H After BP"]).copy()
    after_required = len(use)

    if after_required == 0:
        print("[filter debug] Rows loaded:", start_n)
        print("[filter debug] After QC filter:", after_qc)
        print("[filter debug] After Breakpoint Accepted filter:", after_bp)
        print("[filter debug] After body filter:", after_body)
        print("[filter debug] After product filter:", after_product)
        print("[filter debug] After required numeric values:", after_required)
        print("[filter debug] Breakpoint Accepted values seen:")
        if "Breakpoint Accepted" in df.columns:
            print(df["Breakpoint Accepted"].value_counts(dropna=False).head(20).to_string())
        print("[filter debug] QC values seen:")
        if "QC Status" in df.columns:
            print(df["QC Status"].value_counts(dropna=False).head(20).to_string())
        return pd.DataFrame(columns=[
            "ROI_ID", "Body", "Site", "Terrain Type", "Product Type", "Point Type",
            "Baseline m", "Hurst Exponent", "Pixel Scale m/pixel", "Breakpoint m",
            "Breakpoint pixels", "Delta H", "Source Sheet",
        ])

    rows: list[dict[str, object]] = []
    for _, r in use.iterrows():
        common = {
            "ROI_ID": r.get("ROI_ID"),
            "Body": r.get("Body"),
            "Site": r.get("Site", "Unknown"),
            "Terrain Type": r.get("Terrain Type"),
            "Product Type": r.get("Product Type", "Unknown"),
            "Pixel Scale m/pixel": r.get("Pixel Scale m/pixel"),
            "Breakpoint m": r.get("Breakpoint m"),
            "Breakpoint pixels": r.get("Breakpoint pixels", np.nan),
            "Delta H": r.get("Delta H", np.nan),
            "Source Sheet": r.get("Source Sheet", "Unknown"),
        }
        rows.append({**common, "Point Type": "Short-scale", "Baseline m": r["Pixel Scale m/pixel"], "Hurst Exponent": r["H Before BP"]})
        rows.append({**common, "Point Type": "Long-scale", "Baseline m": r["Breakpoint m"], "Hurst Exponent": r["H After BP"]})

    long = pd.DataFrame(rows)
    long = long.replace([np.inf, -np.inf], np.nan)
    long = long.dropna(subset=["Baseline m", "Hurst Exponent", "Terrain Type", "Body"])
    long = long[(long["Baseline m"] > 0) & (long["Hurst Exponent"].between(-0.25, 1.25))]
    return long.reset_index(drop=True)


def aggregate_long_table(long: pd.DataFrame, color_by: str) -> pd.DataFrame:
    # Aggregate across ROIs within terrain/body/site/product/point-type groups.
    group_cols = ["Terrain Type", "Body", "Site", "Product Type", "Point Type"]
    if color_by not in group_cols and color_by in long.columns:
        group_cols.append(color_by)
    agg = (
        long.groupby(group_cols, dropna=False)
        .agg(
            n=("ROI_ID", "count"),
            mean_baseline_m=("Baseline m", "mean"),
            std_baseline_m=("Baseline m", "std"),
            median_baseline_m=("Baseline m", "median"),
            min_baseline_m=("Baseline m", "min"),
            max_baseline_m=("Baseline m", "max"),
            mean_h=("Hurst Exponent", "mean"),
            std_h=("Hurst Exponent", "std"),
            median_h=("Hurst Exponent", "median"),
            min_h=("Hurst Exponent", "min"),
            max_h=("Hurst Exponent", "max"),
        )
        .reset_index()
    )
    agg["std_h"] = agg["std_h"].fillna(0.0)
    agg["std_baseline_m"] = agg["std_baseline_m"].fillna(0.0)
    return agg


def choose_grid(n: int, ncols: int | None = None) -> tuple[int, int]:
    if n <= 0:
        return 1, 1
    if ncols is None:
        ncols = 3 if n >= 6 else min(3, n)
    return math.ceil(n / ncols), ncols


def make_category_colors(values: Sequence[object], cmap_name: str = "tab20") -> dict[str, object]:
    clean_values = [str(v) for v in values]
    unique = sorted(set(clean_values))
    cmap = plt.get_cmap(cmap_name)
    if len(unique) <= 1:
        return {unique[0]: cmap(0) if unique else "black"}
    return {v: cmap(i % cmap.N) for i, v in enumerate(unique)}


def make_product_markers(products: Sequence[object]) -> dict[str, str]:
    unique = sorted(set(str(p) for p in products))
    markers: dict[str, str] = {}
    fallback_i = 0
    for p in unique:
        if p in PRODUCT_MARKERS:
            markers[p] = PRODUCT_MARKERS[p]
        elif "combined" in p.lower():
            markers[p] = "^"
        elif "sfs" in p.lower():
            markers[p] = "s"
        elif "stereo" in p.lower():
            markers[p] = "o"
        else:
            markers[p] = FALLBACK_MARKERS[fallback_i % len(FALLBACK_MARKERS)]
            fallback_i += 1
    return markers


def format_axis(ax: plt.Axes, terrain: object, x_min: float, x_max: float, y_min: float, y_max: float) -> None:
    ax.set_title(str(terrain).replace("_", " "), fontsize=PANEL_TITLE_FS, pad=8)
    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_xlabel("Baseline [m]", fontsize=AXIS_LABEL_FS)
    ax.set_ylabel("Hurst exponent", fontsize=AXIS_LABEL_FS)
    ax.grid(True, which="both", linewidth=0.5, alpha=GRID_ALPHA)
    ax.tick_params(axis="both", labelsize=TICK_LABEL_FS)


def wrap_legend_label(label: object, max_chars: int = 22) -> str:
    """Wrap long legend labels so they do not run into neighboring labels."""
    text = str(label).replace("_", " ").strip()
    if max_chars <= 0:
        return text
    return "\n".join(textwrap.wrap(text, width=max_chars, break_long_words=False, break_on_hyphens=False)) or text


def add_legends(
    fig: plt.Figure,
    color_title: str,
    color_map: dict[str, object],
    marker_title: str,
    marker_map: dict[str, str],
    *,
    legend_position: str = "right",
    legend_ncols: int | None = None,
    max_label_chars: int = 22,
) -> tuple[float, float, float, float]:
    if legend_position == "none":
        return (0.05, 0.06, 0.98, 0.95)

    color_handles = [
        Line2D(
            [0], [0],
            marker="o",
            linestyle="None",
            label=wrap_legend_label(name, max_label_chars),
            markerfacecolor=color,
            markeredgecolor="black",
            markersize=8
        )
        for name, color in color_map.items()
    ]

    marker_handles = [
        Line2D(
            [0], [0],
            marker=marker,
            linestyle="None",
            label=wrap_legend_label(name, max_label_chars),
            markerfacecolor="white",
            markeredgecolor="black",
            markersize=8
        )
        for name, marker in marker_map.items()
    ]

    if legend_position == "right":
        if marker_handles:
            fig.legend(
                marker_handles,
                [h.get_label() for h in marker_handles],
                title=marker_title,
                loc="upper left",
                bbox_to_anchor=(0.86, 0.93),
                fontsize=LEGEND_FS,
                title_fontsize=LEGEND_TITLE_FS,
                frameon=False,
                borderaxespad=0.0,
                labelspacing=0.55,
                handletextpad=0.5,
            )

        if color_handles:
            fig.legend(
                color_handles,
                [h.get_label() for h in color_handles],
                title=color_title,
                loc="upper left",
                bbox_to_anchor=(0.86, 0.63),
                fontsize=LEGEND_FS,
                title_fontsize=LEGEND_TITLE_FS,
                frameon=False,
                borderaxespad=0.0,
                labelspacing=0.55,
                handletextpad=0.5,
            )

        # Less wasted space than before
        return (0.05, 0.07, 0.82, 0.95)

    if marker_handles:
        fig.legend(
            marker_handles,
            [h.get_label() for h in marker_handles],
            title=marker_title,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.065),
            ncol=min(4, len(marker_handles)),
            fontsize=LEGEND_FS,
            title_fontsize=LEGEND_TITLE_FS,
            frameon=False,
            labelspacing=0.6,
            columnspacing=1.0,
        )

    if color_handles:
        ncol = legend_ncols or min(4, max(1, len(color_handles)))
        fig.legend(
            color_handles,
            [h.get_label() for h in color_handles],
            title=color_title,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.005),
            ncol=ncol,
            fontsize=LEGEND_FS,
            title_fontsize=LEGEND_TITLE_FS,
            frameon=False,
            labelspacing=0.6,
            columnspacing=1.0,
        )

    return (0.05, 0.16, 0.98, 0.95)

def plot_individual_panels(
    long: pd.DataFrame,
    output_path: Path,
    title: str,
    color_by: str,
    shape_by_product: bool,
    connect_rois: bool,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    ncols: int | None,
    legend_position: str,
    legend_ncols: int | None,
    max_legend_label_chars: int,
) -> None:
    terrains = sorted(long["Terrain Type"].dropna().astype(str).unique())
    nrows, ncols_final = choose_grid(len(terrains), ncols)
    fig, axes = plt.subplots(
    nrows,
    ncols_final,
    figsize=(FIG_W_PER_COL * ncols_final, FIG_H_PER_ROW * nrows),
    squeeze=False
    )
    axes_flat = axes.ravel()

    color_col = color_by if color_by in long.columns else "Site"
    color_map = make_category_colors(long[color_col].dropna().astype(str).unique())
    marker_map = make_product_markers(long["Product Type"].dropna().astype(str).unique())

    for ax, terrain in zip(axes_flat, terrains):
        sub = long[long["Terrain Type"].astype(str) == terrain]
        if connect_rois:
            for _, g in sub.groupby("ROI_ID", dropna=False):
                if len(g) < 2:
                    continue
                g = g.sort_values("Baseline m")
                cval = str(g[color_col].iloc[0])
                ax.plot(g["Baseline m"], g["Hurst Exponent"], color=color_map.get(cval, "black"), alpha=0.28, linewidth=0.9, zorder=1)
        for _, r in sub.iterrows():
            cval = str(r[color_col])
            pval = str(r["Product Type"])
            marker = marker_map.get(pval, "o") if shape_by_product else {"Short-scale": "o", "Long-scale": "s"}.get(str(r["Point Type"]), "o")
            ax.scatter(
                r["Baseline m"], r["Hurst Exponent"],
                s=MARKER_SIZE,
                color=color_map.get(cval, "black"),
                marker=marker,
                edgecolor="black",
                linewidth=0.45,
                alpha=0.9,
                zorder=2,
            )
        format_axis(ax, terrain, x_min, x_max, y_min, y_max)

    for ax in axes_flat[len(terrains):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=SUPTITLE_FS, y=0.98)
    layout_rect = add_legends(
        fig, color_col, color_map, "Product type", marker_map if shape_by_product else {},
        legend_position=legend_position, legend_ncols=legend_ncols, max_label_chars=max_legend_label_chars
    )
    fig.tight_layout(rect=layout_rect, pad=1.0)
    fig.subplots_adjust(wspace=WSPACE, hspace=HSPACE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_aggregate_panels(
    agg: pd.DataFrame,
    output_path: Path,
    title: str,
    color_by: str,
    shape_by_product: bool,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    ncols: int | None,
    show_xerr: bool,
    legend_position: str,
    legend_ncols: int | None,
    max_legend_label_chars: int,
    annotate_n: bool,
) -> None:
    terrains = sorted(agg["Terrain Type"].dropna().astype(str).unique())
    nrows, ncols_final = choose_grid(len(terrains), ncols)
    fig, axes = plt.subplots(
    nrows,
    ncols_final,
    figsize=(FIG_W_PER_COL * ncols_final, FIG_H_PER_ROW * nrows),
    squeeze=False
    )
    axes_flat = axes.ravel()

    color_col = color_by if color_by in agg.columns else "Site"
    color_map = make_category_colors(agg[color_col].dropna().astype(str).unique())
    marker_map = make_product_markers(agg["Product Type"].dropna().astype(str).unique())

    for ax, terrain in zip(axes_flat, terrains):
        sub = agg[agg["Terrain Type"].astype(str) == terrain]
        for _, r in sub.iterrows():
            cval = str(r[color_col])
            pval = str(r["Product Type"])
            marker = marker_map.get(pval, "o") if shape_by_product else {"Short-scale": "o", "Long-scale": "s"}.get(str(r["Point Type"]), "o")
            ax.errorbar(
                r["mean_baseline_m"], r["mean_h"],
                yerr=r["std_h"],
                xerr=r["std_baseline_m"] if show_xerr else None,
                fmt=marker,
                markersize=ERRORBAR_MARKER_SIZE,
                color=color_map.get(cval, "black"),
                markeredgecolor="black",
                markeredgewidth=0.55,
                elinewidth=1.25,
                capsize=4,
                alpha=0.95,
            )
            if annotate_n:
                ax.annotate(f"n={int(r['n'])}", (r["mean_baseline_m"], r["mean_h"]), xytext=(4, 4), textcoords="offset points", fontsize=7, alpha=0.8)
        format_axis(ax, terrain, x_min, x_max, y_min, y_max)

    for ax in axes_flat[len(terrains):]:
        ax.axis("off")

    fig.suptitle(title, fontsize=SUPTITLE_FS, y=0.98)
    layout_rect = add_legends(
        fig, color_col, color_map, "Product type", marker_map if shape_by_product else {},
        legend_position=legend_position, legend_ncols=legend_ncols, max_label_chars=max_legend_label_chars
    )
    fig.tight_layout(rect=layout_rect, pad=1.0)
    fig.subplots_adjust(wspace=WSPACE, hspace=HSPACE)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    fig.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def print_debug(df: pd.DataFrame, long: pd.DataFrame, agg: pd.DataFrame | None = None) -> None:
    print("\n[debug] Loaded ROI rows by Source Sheet and Body:")
    if {"Source Sheet", "Body"}.issubset(df.columns):
        print(df.groupby(["Source Sheet", "Body"]).size().reset_index(name="n_rows").to_string(index=False))
    print("\n[debug] Plotted ROI counts by Body/Terrain/Product:")
    if not long.empty:
        print(long.groupby(["Body", "Terrain Type", "Product Type"])["ROI_ID"].nunique().reset_index(name="n_roi").to_string(index=False))
    if agg is not None and not agg.empty:
        print("\n[debug] Aggregate groups with std_h:")
        print(agg[["Body", "Site", "Terrain Type", "Product Type", "Point Type", "n", "mean_baseline_m", "mean_h", "std_h"]].to_string(index=False))


def plot_for_dataset(
    long: pd.DataFrame,
    output_dir: Path,
    label: str,
    plot_mode: str,
    color_by: str,
    shape_by_product: bool,
    connect_rois: bool,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    ncols: int | None,
    show_xerr: bool,
    legend_position: str,
    legend_ncols: int | None,
    max_legend_label_chars: int,
    annotate_n: bool,
) -> None:
    safe_label = safe_name(label)
    if plot_mode in {"individual", "both"}:
        plot_individual_panels(
            long,
            output_dir / f"figure12_hurst_baseline_{safe_label}_individual.png",
            f"{label}: Hurst exponent vs baseline by terrain",
            color_by,
            shape_by_product,
            connect_rois,
            x_min,
            x_max,
            y_min,
            y_max,
            ncols,
            legend_position,
            legend_ncols,
            max_legend_label_chars,
        )
    if plot_mode in {"aggregate", "both"}:
        agg = aggregate_long_table(long, color_by=color_by)
        agg.to_csv(output_dir / f"figure12_hurst_baseline_{safe_label}_aggregate_table.csv", index=False)
        plot_aggregate_panels(
            agg,
            output_dir / f"figure12_hurst_baseline_{safe_label}_aggregate_mean_std.png",
            f"{label}: Hurst exponent vs baseline by terrain (mean ± std dev)",
            color_by,
            shape_by_product,
            x_min,
            x_max,
            y_min,
            y_max,
            ncols,
            show_xerr,
            legend_position,
            legend_ncols,
            max_legend_label_chars,
            annotate_n,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Make Figure-12-style Hurst exponent vs baseline terrain panels with block-style Excel reading and non-overlapping legends.")
    parser.add_argument("--excel", required=True, help="Path to Excel workbook, e.g. roi_grouped_summary.xlsx")
    parser.add_argument("--sheets", nargs="+", default=DEFAULT_SHEETS, help="Sheet names to read.")
    parser.add_argument("--output-dir", default="figures/figure12_hurst_baseline_v4", help="Output directory.")
    parser.add_argument("--qc", default="all", help="QC Status to use, or 'all'. Default: all, because Product_Good sheets are usually already filtered.")
    parser.add_argument("--bodies", nargs="*", default=None, help="Optional body filter: Europa Ganymede Enceladus Tethys")
    parser.add_argument("--products", nargs="*", default=None, help="Optional exact Product Type filter.")
    parser.add_argument("--plot-mode", choices=["individual", "aggregate", "both"], default="individual", help="individual = every ROI point; aggregate = mean +/- std error bars; both = both versions.")
    parser.add_argument("--split-by-body", action="store_true", help="Make one separate figure for each body/world.")
    parser.add_argument("--also-combined", action="store_true", help="Also make one combined all-body figure.")
    parser.add_argument("--color-by", default="Site", choices=["Site", "Body", "Product Type"], help="Color points by this column. Default: Site.")
    parser.add_argument("--legend-position", choices=["right", "bottom", "none"], default="bottom", help="Legend placement. Use 'right' to prevent overlapping long site names but bottom to make it look more compact. Default: bottom.")
    parser.add_argument("--legend-ncols", type=int, default=None, help="Bottom legend only: number of legend columns for site/body/product labels.")
    parser.add_argument("--max-legend-label-chars", type=int, default=22, help="Wrap legend labels after this many characters. Use 0 to disable wrapping.")
    parser.add_argument("--shape-by-product", action="store_true", default=True, help="Use marker shape for product type. Default: on.")
    parser.add_argument("--no-shape-by-product", dest="shape_by_product", action="store_false", help="Turn off product-type marker shapes.")
    parser.add_argument("--connect-rois", action="store_true", default=True, help="Connect short- and long-scale points for each ROI. Default: on.")
    parser.add_argument("--no-connect-rois", dest="connect_rois", action="store_false", help="Do not connect ROI point pairs.")
    parser.add_argument("--show-xerr", action="store_true", help="Aggregate mode only: show horizontal baseline std dev error bars too.")
    parser.add_argument("--annotate-n", action="store_true", help="Aggregate mode only: annotate each point with sample size n. Default off to avoid label overlap.")
    parser.add_argument("--x-min", type=float, default=5.0)
    parser.add_argument("--x-max", type=float, default=10000.0)
    parser.add_argument("--y-min", type=float, default=0.0)
    parser.add_argument("--y-max", type=float, default=1.0)
    parser.add_argument("--ncols", type=int, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    excel_path = Path(args.excel)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_data(excel_path, args.sheets, verbose=args.verbose)
    long = make_long_hurst_table(
        df,
        qc=args.qc,
        bodies=args.bodies,
        products=args.products,
        include_unaccepted=False,
    )
    if long.empty:
        raise ValueError("No Figure-12 plotting points remained. Run with --verbose and check sheet names, body names, Breakpoint Accepted, and numeric H columns.")

    long.to_csv(output_dir / "figure12_hurst_baseline_long_table.csv", index=False)
    full_agg = aggregate_long_table(long, color_by=args.color_by)
    full_agg.to_csv(output_dir / "figure12_hurst_baseline_aggregate_table.csv", index=False)

    if args.verbose:
        print_debug(df, long, full_agg)

    if args.also_combined or not args.split_by_body:
        plot_for_dataset(
            long,
            output_dir,
            "All selected bodies",
            args.plot_mode,
            args.color_by,
            args.shape_by_product,
            args.connect_rois,
            args.x_min,
            args.x_max,
            args.y_min,
            args.y_max,
            args.ncols,
            args.show_xerr,
            args.legend_position,
            args.legend_ncols,
            args.max_legend_label_chars,
            args.annotate_n,
        )

    if args.split_by_body:
        bodies = sorted(long["Body"].dropna().astype(str).unique())
        for body in bodies:
            sub = long[long["Body"].astype(str) == body].copy()
            if sub.empty:
                continue
            sub.to_csv(output_dir / f"figure12_hurst_baseline_long_table_{safe_name(body)}.csv", index=False)
            plot_for_dataset(
                sub,
                output_dir,
                body,
                args.plot_mode,
                args.color_by,
                args.shape_by_product,
                args.connect_rois,
                args.x_min,
                args.x_max,
                args.y_min,
                args.y_max,
                args.ncols,
                args.show_xerr,
                args.legend_position,
                args.legend_ncols,
                args.max_legend_label_chars,
                args.annotate_n,
            )

    print(f"[done] Saved Figure-12-style plots and CSVs to: {output_dir}")
    print("[note] Individual mode uses every ROI point. Aggregate mode uses mean ± standard deviation across ROIs in each site/product/point group.")


if __name__ == "__main__":
    main()

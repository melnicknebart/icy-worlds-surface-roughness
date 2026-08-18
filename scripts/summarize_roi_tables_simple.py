#!/usr/bin/env python3
"""
Simplified ROI summary script.

Goal:
  Read one or more roi_analysis_summary.csv files and create a small, digestible
  summary workbook focused only on the quantities needed for science conclusions:

  - Good QC filtering
  - H Overall comparison
  - H Before BP, H After BP, and Delta H = H Before BP - H After BP
  - Breakpoint medians/means in meters and pixels
  - Bad-pixel fraction checks
  - Pixel scale and fit-range checks
  - QC counts by body/site/terrain/product

Example:
python scripts/summarize_roi_tables_simple.py \
  --inputs results/europa/summaries/tables/roi_analysis_summary.csv results/ganymede/summaries/tables/ganymede_roi_analysis_summary.csv \
  --output simple_roi_summary.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Columns we actually care about for interpretation
# -----------------------------------------------------------------------------
KEEP_COLUMNS = [
    "ROI_ID",
    "Dataset",
    "Body",
    "Site",
    "Terrain Type",
    "Product Type",
    "Pixel Scale m/pixel",
    "QC Status",
    "Bad Pixel Fraction",
    "Interpolation Method",
    "Fit Min Baseline m",
    "Fit Max Baseline m",
    "H Overall",
    "H Overall OLS SE",
    "Breakpoint Accepted",
    "Breakpoint m",
    "H Before BP",
    "H After BP",
]

GROUP_METRICS = [
    "n_rois",
    "median_pixel_scale_m_per_px",
    "median_bad_pixel_fraction",
    "mean_bad_pixel_fraction",
    "median_H_overall",
    "mean_H_overall",
    "median_H_before_BP",
    "median_H_after_BP",
    "median_delta_H",
    "mean_delta_H",
    "median_breakpoint_m",
    "mean_breakpoint_m",
    "median_breakpoint_pixels",
    "mean_breakpoint_pixels",
    "median_fit_max_m",
    "median_fit_max_pixels",
]


def as_bool(series: pd.Series) -> pd.Series:
    """Convert True/False-like values to bool."""
    if series.dtype == bool:
        return series.fillna(False)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "t", "yes", "y", "1"])
    )


def load_inputs(paths: list[str]) -> pd.DataFrame:
    frames = []
    for p in paths:
        path = Path(p)
        if not path.exists():
            raise FileNotFoundError(f"Input file not found: {path}")
        df = pd.read_csv(path)
        df["Source File"] = path.name
        frames.append(df)

    combined = pd.concat(frames, ignore_index=True)

    # Keep only columns that exist. This lets the script work if future files
    # have extra columns or are missing optional columns.
    cols = [c for c in KEEP_COLUMNS if c in combined.columns] + ["Source File"]
    combined = combined[cols].copy()

    # Ensure needed columns exist, even if blank in a future table.
    for col in KEEP_COLUMNS:
        if col not in combined.columns:
            combined[col] = np.nan

    # Numeric conversion
    numeric_cols = [
        "Pixel Scale m/pixel",
        "Bad Pixel Fraction",
        "Fit Min Baseline m",
        "Fit Max Baseline m",
        "H Overall",
        "H Overall OLS SE",
        "Breakpoint m",
        "H Before BP",
        "H After BP",
    ]
    for col in numeric_cols:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")

    combined["Breakpoint Accepted Bool"] = as_bool(combined["Breakpoint Accepted"])

    # Derived science columns
    combined["Delta H"] = combined["H Before BP"] - combined["H After BP"]
    combined["Breakpoint pixels"] = combined["Breakpoint m"] / combined["Pixel Scale m/pixel"]
    combined["Fit Min pixels"] = combined["Fit Min Baseline m"] / combined["Pixel Scale m/pixel"]
    combined["Fit Max pixels"] = combined["Fit Max Baseline m"] / combined["Pixel Scale m/pixel"]
    combined["Fit Range m"] = combined["Fit Max Baseline m"] - combined["Fit Min Baseline m"]
    combined["Fit Range pixels"] = combined["Fit Range m"] / combined["Pixel Scale m/pixel"]

    # Simple interpretation categories
    combined["Delta H Category"] = pd.cut(
        combined["Delta H"],
        bins=[-np.inf, 0.10, 0.30, np.inf],
        labels=["Small transition", "Moderate transition", "Large transition"],
    )
    combined["Bad Pixel Category"] = pd.cut(
        combined["Bad Pixel Fraction"],
        bins=[-np.inf, 0.01, 0.05, 0.10, np.inf],
        labels=["Clean (<1%)", "Low (1-5%)", "Moderate (5-10%)", "High (>10%)"],
    )

    combined["Use For Overall H Science"] = combined["QC Status"].eq("Good") & combined["H Overall"].notna()
    combined["Use For BP Science"] = (
        combined["QC Status"].eq("Good")
        & combined["Breakpoint Accepted Bool"]
        & combined["H Before BP"].notna()
        & combined["H After BP"].notna()
        & combined["Breakpoint m"].notna()
    )

    return combined


def summarize_group(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Return compact median/mean summary for a grouping."""
    if df.empty:
        return pd.DataFrame(columns=group_cols + GROUP_METRICS)

    out = (
        df.groupby(group_cols, dropna=False)
        .agg(
            n_rois=("ROI_ID", "count"),
            median_pixel_scale_m_per_px=("Pixel Scale m/pixel", "median"),
            median_bad_pixel_fraction=("Bad Pixel Fraction", "median"),
            mean_bad_pixel_fraction=("Bad Pixel Fraction", "mean"),
            median_H_overall=("H Overall", "median"),
            mean_H_overall=("H Overall", "mean"),
            median_H_before_BP=("H Before BP", "median"),
            median_H_after_BP=("H After BP", "median"),
            median_delta_H=("Delta H", "median"),
            mean_delta_H=("Delta H", "mean"),
            median_breakpoint_m=("Breakpoint m", "median"),
            mean_breakpoint_m=("Breakpoint m", "mean"),
            median_breakpoint_pixels=("Breakpoint pixels", "median"),
            mean_breakpoint_pixels=("Breakpoint pixels", "mean"),
            median_fit_max_m=("Fit Max Baseline m", "median"),
            median_fit_max_pixels=("Fit Max pixels", "median"),
        )
        .reset_index()
        .sort_values(group_cols + ["n_rois"], ascending=[True] * len(group_cols) + [False])
    )
    return out


def make_qc_counts(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["Body", "QC Status"], dropna=False)
        .agg(
            n_rois=("ROI_ID", "count"),
            median_bad_pixel_fraction=("Bad Pixel Fraction", "median"),
            mean_bad_pixel_fraction=("Bad Pixel Fraction", "mean"),
        )
        .reset_index()
        .sort_values(["Body", "QC Status"])
    )


def make_site_qc(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["Body", "Site", "QC Status"], dropna=False)
        .agg(
            n_rois=("ROI_ID", "count"),
            median_bad_pixel_fraction=("Bad Pixel Fraction", "median"),
            median_pixel_scale_m_per_px=("Pixel Scale m/pixel", "median"),
        )
        .reset_index()
        .sort_values(["Body", "Site", "QC Status"])
    )


def make_flagged_rois(df: pd.DataFrame) -> pd.DataFrame:
    """Rows needing caution: not Good, no accepted BP, high bad pixels, or suspicious breakpoint scale."""
    flag = (
        ~df["QC Status"].eq("Good")
        | (df["Bad Pixel Fraction"] >= 0.05)
        | (~df["Breakpoint Accepted Bool"])
        | (df["Breakpoint pixels"] < 5)
        | (df["Breakpoint pixels"] > 25)
    )
    cols = [
        "ROI_ID", "Body", "Site", "Terrain Type", "Product Type", "QC Status",
        "Bad Pixel Fraction", "Bad Pixel Category", "Pixel Scale m/pixel",
        "H Overall", "Breakpoint Accepted Bool", "Breakpoint m", "Breakpoint pixels",
        "H Before BP", "H After BP", "Delta H", "Delta H Category",
        "Fit Min Baseline m", "Fit Max Baseline m", "Fit Range pixels",
    ]
    return df.loc[flag, cols].sort_values(["Body", "QC Status", "Bad Pixel Fraction"], ascending=[True, True, False])


def write_outputs(tables: dict[str, pd.DataFrame], output_xlsx: str) -> None:
    output_path = Path(output_xlsx)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    readme = pd.DataFrame(
        {
            "Sheet": [
                "01_BP_by_body_terrain_product",
                "02_Good_by_body_terrain",
                "03_BP_by_body_product",
                "04_QC_counts",
                "05_Site_QC",
                "06_Flagged_ROIs",
                "07_Cleaned_Key_Rows",
            ],
            "Use": [
                "Main science table: Good QC + accepted breakpoints, grouped by body/terrain/product.",
                "Good QC only: compare H Overall by body and terrain, even if no breakpoint.",
                "Check product effects: Stereo vs SFS vs Combined.",
                "Counts and bad-pixel fractions by body and QC status.",
                "Choose which sites are strong, cautious, or not useful.",
                "Rows to inspect before using in science conclusions.",
                "Trimmed raw table with only important columns and derived metrics.",
            ],
        }
    )
    tables = {"00_README": readme, **tables}

    # Excel workbook
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        for sheet, table in tables.items():
            # Excel sheet names max 31 characters.
            safe_sheet = sheet[:31]
            table.to_excel(writer, sheet_name=safe_sheet, index=False)

            ws = writer.book[safe_sheet]
            ws.freeze_panes = "A2"
            for col_cells in ws.columns:
                header = str(col_cells[0].value) if col_cells[0].value is not None else ""
                width = min(max(len(header) + 2, 12), 28)
                ws.column_dimensions[col_cells[0].column_letter].width = width

    # Also save each sheet as a CSV folder for easy GitHub tracking.
    csv_dir = output_path.with_suffix("")
    csv_dir.mkdir(exist_ok=True)
    for sheet, table in tables.items():
        table.to_csv(csv_dir / f"{sheet}.csv", index=False)

    print(f"Wrote Excel workbook: {output_path}")
    print(f"Wrote CSV folder:     {csv_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create simplified science summaries from ROI analysis tables.")
    parser.add_argument("--inputs", nargs="+", required=True, help="One or more roi_analysis_summary.csv files.")
    parser.add_argument("--output", required=True, help="Output .xlsx workbook path.")
    args = parser.parse_args()

    df = load_inputs(args.inputs)

    good = df[df["Use For Overall H Science"]].copy()
    bp = df[df["Use For BP Science"]].copy()

    key_cols = [
        "ROI_ID", "Body", "Site", "Terrain Type", "Product Type", "QC Status",
        "Bad Pixel Fraction", "Bad Pixel Category", "Pixel Scale m/pixel",
        "H Overall", "H Overall OLS SE", "Breakpoint Accepted Bool",
        "Breakpoint m", "Breakpoint pixels", "H Before BP", "H After BP",
        "Delta H", "Delta H Category", "Fit Min Baseline m", "Fit Max Baseline m",
        "Fit Max pixels", "Fit Range pixels", "Source File",
    ]

    tables = {
        "01_BP_by_body_terrain_product": summarize_group(bp, ["Body", "Terrain Type", "Product Type"]),
        "02_Good_by_body_terrain": summarize_group(good, ["Body", "Terrain Type"]),
        "03_BP_by_body_product": summarize_group(bp, ["Body", "Product Type"]),
        "04_QC_counts": make_qc_counts(df),
        "05_Site_QC": make_site_qc(df),
        "06_Flagged_ROIs": make_flagged_rois(df),
        "07_Cleaned_Key_Rows": df[key_cols].copy(),
    }

    write_outputs(tables, args.output)


if __name__ == "__main__":
    main()

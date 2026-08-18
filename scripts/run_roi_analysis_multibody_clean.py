#!/usr/bin/env python3
"""Run the multi-body icy-worlds roughness analysis.

This is the canonical ROI-analysis entry point for Europa, Ganymede, and
Enceladus. It combines:

1. catalog and ROI geometry handling;
2. raw DTM loading from ISIS CUB crops or saved rotated ROI arrays;
3. raw-DTM QC and documented missing-pixel treatment;
4. least-squares plane detrending;
5. PSD diagnostics;
6. deterministic RMS-deviation roughness curves;
7. Hurst fits and candidate segmented breakpoints;
8. per-ROI provenance plus body-level summary tables.


Run from the repository root, for example:

    python scripts/run_roi_analysis_multibody.py \
        --dataset europa --roi CONA12hr_CHAOS_008

Run every cataloged ROI for one body:

    python scripts/run_roi_analysis_multibody.py \
        --dataset europa --all

Required neighboring modules:

    scripts/roi_processing.py
    scripts/icy_workflow_common.py
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

import roi_processing as processing
from icy_workflow_common import (
    clean_text,
    detect_segmented_breakpoint,
    ensure_dataset_directories,
    exact_rms_deviation_curve_from_dtm,
    fit_power_law,
    get_catalog_value,
    json_default,
    load_project_config,
    predict_power_law,
    repository_root_from_script,
    resolve_dataset_config,
    safe_positive_float,
    save_curve_csv,
    save_fit_bins_csv,
    select_fit_bins,
    update_summary_index,
)


# ---------------------------------------------------------------------------
# Science settings
# ---------------------------------------------------------------------------

ACTIVE_DATASET = "europa"

MAX_BASELINE_M: float | None = 5000.0
MIN_FIT_BASELINE_PIXELS = 1.0
MIN_PAIRS_PER_BIN = 100

USE_SHEPARD_10_PERCENT_FIT_LIMIT = True
FIT_RANGE_FRACTION_OF_MIN_ROI_SIDE = 0.10

MIN_POINTS_EACH_SIDE = 5
EDGE_BUFFER_BINS = 4
MIN_RESIDUAL_IMPROVEMENT_FRACTION = 0.20

NULL_VALUE = -9999
NULL_THRESHOLD = -9000

GOOD_MAX_BAD_FRACTION = 0.01
QUESTIONABLE_MAX_BAD_FRACTION = 0.05
NEEDS_REVIEW_MAX_BAD_FRACTION = 0.10

SKIP_REJECTED_ROIS = True
STOP_IF_LARGE_NULL_REGION = False
MEDIAN_FILTER_SIZE = 3

ALLOW_INTERPOLATION_FOR_REJECTED_ROIS = True
INTERPOLATION_METHOD = "linear_then_nearest"
SAVE_INTERPOLATION_MASK = True

SAVE_LINEAR_BASELINE_DIAGNOSTIC = True


# ---------------------------------------------------------------------------
# CLI and processing-module configuration
# ---------------------------------------------------------------------------

def parse_cli(repo_root: Path) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic multi-body DTM QC, PSD, RMS-deviation, Hurst, "
            "and breakpoint analysis."
        )
    )
    parser.add_argument(
        "--config",
        default=str(repo_root / "config" / "icy_worlds_config.json"),
        help="Path to the central workflow JSON config.",
    )
    parser.add_argument(
        "--dataset",
        default=ACTIVE_DATASET,
        help="Dataset key from config, e.g. europa, ganymede, enceladus.",
    )

    roi_group = parser.add_mutually_exclusive_group()
    roi_group.add_argument(
        "--roi",
        action="append",
        default=None,
        help="ROI_ID to run. Repeat this flag for multiple ROIs.",
    )
    roi_group.add_argument(
        "--all",
        action="store_true",
        help="Run every nonblank ROI_ID in the selected body's catalog.",
    )

    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop at the first failed ROI.",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="Print configured dataset keys and exit.",
    )
    return parser.parse_args()


def configure_processing_module(
    dataset: dict[str, Any],
    repo_root: Path,
) -> None:
    """Apply the shared dataset/QC settings to roi_processing."""
    processing.REPO_ROOT = repo_root
    processing.ROI_CATALOG = Path(dataset["catalog_path"])
    processing.OUTPUT_ROOT = Path(dataset["reproducible_rois_root"])

    processing.NULL_VALUE = NULL_VALUE
    processing.NULL_THRESHOLD = NULL_THRESHOLD
    processing.GOOD_MAX_BAD_FRACTION = GOOD_MAX_BAD_FRACTION
    processing.QUESTIONABLE_MAX_BAD_FRACTION = QUESTIONABLE_MAX_BAD_FRACTION
    processing.NEEDS_REVIEW_MAX_BAD_FRACTION = NEEDS_REVIEW_MAX_BAD_FRACTION
    processing.SKIP_REJECTED_ROIS = SKIP_REJECTED_ROIS
    processing.STOP_IF_LARGE_NULL_REGION = STOP_IF_LARGE_NULL_REGION
    processing.MEDIAN_FILTER_SIZE = MEDIAN_FILTER_SIZE
    processing.ALLOW_INTERPOLATION_FOR_REJECTED_ROIS = (
        ALLOW_INTERPOLATION_FOR_REJECTED_ROIS
    )
    processing.INTERPOLATION_METHOD = INTERPOLATION_METHOD
    processing.SAVE_INTERPOLATION_MASK = SAVE_INTERPOLATION_MASK


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def draw_fit_and_breakpoint(
    ax: plt.Axes,
    baseline_fit: np.ndarray,
    full_fit: dict[str, Any],
    breakpoint_info: dict[str, Any],
) -> None:
    """Draw the full power-law fit and accepted segmented fits."""
    x = np.asarray(baseline_fit, dtype=float)

    ax.loglog(
        x,
        predict_power_law(x, full_fit),
        "-",
        linewidth=2.2,
        label=(
            f"Full fit: H = {full_fit['H']:.3f}, "
            f"σ₀ = {full_fit['sigma0_m_at_1_m_extrapolated']:.3g} m"
        ),
    )

    if not breakpoint_info.get("accepted", False):
        return

    split = int(breakpoint_info["candidate_breakpoint_index"])
    before_fit = {
        "H": breakpoint_info["H_before_breakpoint"],
        "log10_sigma0": breakpoint_info["log10_sigma0_before"],
    }
    after_fit = {
        "H": breakpoint_info["H_after_breakpoint"],
        "log10_sigma0": breakpoint_info["log10_sigma0_after"],
    }

    ax.loglog(
        x[:split],
        predict_power_law(x[:split], before_fit),
        "--",
        linewidth=2.0,
        label=f"Before BP: H = {breakpoint_info['H_before_breakpoint']:.3f}",
    )
    ax.loglog(
        x[split:],
        predict_power_law(x[split:], after_fit),
        "--",
        linewidth=2.0,
        label=f"After BP: H = {breakpoint_info['H_after_breakpoint']:.3f}",
    )
    ax.axvline(
        breakpoint_info["breakpoint_baseline_m"],
        linestyle=":",
        linewidth=2.0,
        label=f"Candidate BP = {breakpoint_info['breakpoint_baseline_m']:.1f} m",
    )


def save_rms_breakpoint_plot(
    roi_id: str,
    curve: dict[str, Any],
    baseline_fit: np.ndarray,
    rms_fit: np.ndarray,
    full_fit: dict[str, Any],
    breakpoint_info: dict[str, Any],
    output_path: Path,
) -> None:
    """Save the RMS-deviation curve and Hurst/breakpoint fits."""
    fig, ax = plt.subplots(figsize=(9.2, 6.5))
    ax.loglog(
        curve["baseline_center_m"],
        curve["rms_deviation_m"],
        "o",
        markersize=4.2,
        alpha=0.65,
        label="All valid resolution-width bins",
    )
    ax.loglog(
        baseline_fit,
        rms_fit,
        "o",
        markersize=5.5,
        label="Bins used for fit",
    )
    draw_fit_and_breakpoint(ax, baseline_fit, full_fit, breakpoint_info)

    if not breakpoint_info.get("accepted", False):
        ax.text(
            0.02,
            0.02,
            breakpoint_info.get("reason", "No accepted breakpoint"),
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment="bottom",
            bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "0.7"},
        )

    ax.set_xlabel("Baseline length, Δx (m)")
    ax.set_ylabel("RMS deviation, y(Δx) (m)")
    ax.set_title(f"{roi_id} — RMS-deviation structure function")
    ax.grid(True, which="both", alpha=0.25)
    ax.legend(fontsize=8.4, loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_linear_baseline_plot(
    roi_id: str,
    curve: dict[str, Any],
    output_path: Path,
) -> None:
    """Save a linear-axis diagnostic of the RMS-deviation curve."""
    fig, ax = plt.subplots(figsize=(9.0, 4.8))
    ax.plot(
        curve["baseline_center_m"],
        curve["rms_deviation_m"],
        "o-",
        markersize=3.3,
        linewidth=1.0,
    )
    ax.set_xlabel("Baseline length, Δx (m)")
    ax.set_ylabel("RMS deviation, y(Δx) (m)")
    ax.set_title(f"{roi_id} — RMS deviation in one-resolution-width bins")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_boolean_mask(
    mask: np.ndarray,
    output_path: Path,
    title: str,
) -> None:
    """Save a boolean mask as a grayscale diagnostic."""
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(mask, cmap="gray")
    ax.set_title(title)
    ax.set_xlabel("X pixel")
    ax.set_ylabel("Y pixel")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------------------
# One ROI
# ---------------------------------------------------------------------------

def run_one_roi(
    dataset: dict[str, Any],
    roi_id: str,
    repo_root: Path,
) -> dict[str, Any]:
    """Run QC, cleaning, detrending, PSD, RMS-deviation, and summary steps."""
    dataset_name = dataset["dataset_name"]
    body_name = dataset["body_name"]
    timestamp = datetime.now().isoformat(timespec="seconds")

    log_lines: list[str] = [
        f"Analysis started: {timestamp}",
        f"Dataset: {dataset_name}",
        f"Body: {body_name}",
        f"ROI_ID: {roi_id}",
        "Analysis: deterministic RMS-deviation Hurst/breakpoint workflow.",
        "Monte Carlo vertical-error propagation: not included.",
    ]

    # 1. Catalog and geometry
    row = processing.load_roi_catalog_row(roi_id)
    geometry = processing.prepare_roi_geometry_for_analysis(row, repo_root=repo_root)

    roi_shape = geometry["roi_shape"]
    analysis_width = int(geometry["analysis_width_pixels"])
    analysis_height = int(geometry["analysis_height_pixels"])

    catalog_qc_status = clean_text(row.get("QC Status")).casefold()
    if (
        SKIP_REJECTED_ROIS
        and catalog_qc_status == "reject"
        and not ALLOW_INTERPOLATION_FOR_REJECTED_ROIS
    ):
        raise ValueError(
            f"{roi_id} is marked Reject in {dataset['catalog_path']}."
        )

    if catalog_qc_status == "reject" and ALLOW_INTERPOLATION_FOR_REJECTED_ROIS:
        log_lines.append(
            "WARNING: Catalog QC is Reject; interpolation is enabled."
        )

    if geometry.get("warnings"):
        log_lines.append("CATALOG / GEOMETRY WARNINGS:")
        log_lines.extend(f"- {warning}" for warning in geometry["warnings"])

    pixel_scale_m = safe_positive_float(
        get_catalog_value(row, "Pixel Scale m/pixel", "Pixel Scale m per pixel"),
        "Pixel Scale m/pixel",
    )

    source_cub_path = processing.resolve_project_path(
        row.get("Source CUB"),
        repo_root=repo_root,
    )
    source_cub_catalog_value = clean_text(row.get("Source CUB"))

    if not geometry["is_rotated_roi"]:
        if source_cub_path is None:
            raise ValueError("Axis-aligned ROI requires Source CUB.")
        if not source_cub_path.exists():
            raise FileNotFoundError(
                f"Source CUB does not exist: {source_cub_path}"
            )
    elif source_cub_path is not None and not source_cub_path.exists():
        log_lines.append(
            "NOTE: Source CUB is unavailable; rotated ROI uses its saved NPY."
        )

    # 2. Per-ROI folders
    roi_dir, data_dir, figure_dir, log_dir = processing.make_roi_folders(roi_id)

    cropped_cub = data_dir / f"{roi_id}_cropped.cub"
    ascii_txt = data_dir / f"{roi_id}_ascii.txt"
    raw_dtm_path = data_dir / "raw_DTM.npy"
    cleaned_dtm_path = data_dir / "cleaned_DTM.npy"
    detrended_dtm_path = data_dir / "detrended_DTM.npy"
    plane_path = data_dir / "best_fit_plane.npy"

    psd_csv_path = data_dir / "psd.csv"
    curve_csv_path = data_dir / "steinbruegge_rms_deviation_curve.csv"
    fit_bins_csv_path = data_dir / "steinbruegge_rms_deviation_fit_bins.csv"

    raw_preview_path = figure_dir / "raw_DTM_preview.png"
    raw_masked_preview_path = figure_dir / "raw_DTM_preview_nulls_masked.png"
    null_mask_path = figure_dir / "null_mask.png"
    cleaned_preview_path = figure_dir / "cleaned_DTM_preview.png"
    detrended_preview_path = figure_dir / "detrended_DTM_preview.png"
    psd_diagnostic_path = figure_dir / "psd_diagnostic_multipanel.png"
    psd_plot_path = figure_dir / "psd_plot.png"
    breakpoint_plot_path = (
        figure_dir / "steinbruegge_rms_deviation_breakpoint_plot.png"
    )
    linear_plot_path = (
        figure_dir / "steinbruegge_rms_deviation_linear_bins.png"
    )

    metadata_path = log_dir / "roi_metadata.json"
    geometry_path = log_dir / "analysis_geometry.json"
    qc_path = log_dir / "quality_control_from_analysis.json"
    summary_path = log_dir / "steinbruegge_rms_deviation_summary.json"
    log_path = log_dir / "analysis_log.txt"

    geometry_record = dict(geometry)
    geometry_record.update(
        {
            "ROI_ID": roi_id,
            "Dataset": dataset_name,
            "Body": body_name,
            "pixel_scale_m_per_pixel": pixel_scale_m,
            "source_cub_catalog_value": source_cub_catalog_value,
            "source_cub_resolved_path": (
                str(source_cub_path) if source_cub_path is not None else ""
            ),
            "source_cub_exists": (
                source_cub_path.exists() if source_cub_path is not None else False
            ),
            "catalog_coordinates_zero_based": (
                processing.CATALOG_COORDINATES_ARE_ZERO_BASED
            ),
            "schema_rule": (
                "axis_aligned_rect: X/Y/Width/Height define the science ROI; "
                "catalog X/Y are zero-based image coordinates and are converted "
                "to one-based ISIS SAMPLE/LINE during crop. rotated_rect: "
                "Rotated DTM NPY is the frozen science ROI."
            ),
        }
    )
    with geometry_path.open("w", encoding="utf-8") as handle:
        json.dump(geometry_record, handle, indent=2, default=json_default)

    metadata = dict(row)
    metadata.update(
        {
            "Analysis ROI Folder": str(roi_dir),
            "Analysis Timestamp": timestamp,
            "Script": Path(__file__).name,
            "Dataset": dataset_name,
            "Body": body_name,
            "ROI Shape Used For Analysis": roi_shape,
            "Analysis Input Mode": geometry["input_mode"],
            "Analysis Width Pixels": analysis_width,
            "Analysis Height Pixels": analysis_height,
            "Analysis Geometry Record": str(geometry_path),
            "Catalog / Geometry Warnings": geometry.get("warnings", []),
        }
    )
    with metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, default=json_default)

    # 3. Raw DTM and QC
    raw_dtm, raw_source_info = processing.load_raw_dtm_for_roi(
        geometry=geometry,
        source_cub_path=source_cub_path,
        cropped_cub=cropped_cub,
        ascii_txt=ascii_txt,
        raw_dtm_path=raw_dtm_path,
        log_lines=log_lines,
    )
    processing.save_dtm_preview(
        raw_dtm, raw_preview_path, f"{roi_id} raw DTM"
    )
    processing.save_dtm_preview_mask_nulls(
        raw_dtm,
        raw_masked_preview_path,
        f"{roi_id} raw DTM nulls masked",
    )
    processing.save_null_mask(raw_dtm, null_mask_path, f"{roi_id} null mask")

    qc_info = processing.evaluate_raw_dtm_quality(raw_dtm)
    qc_info.update(
        {
            "ROI_ID": roi_id,
            "Dataset": dataset_name,
            "Body": body_name,
            "Timestamp": timestamp,
            "Raw DTM path": str(raw_dtm_path),
            "Null mask path": str(null_mask_path),
            "Raw DTM preview": str(raw_preview_path),
            "Raw DTM preview nulls masked": str(raw_masked_preview_path),
            "Raw DTM source info": raw_source_info,
            "ROI shape": roi_shape,
            "Analysis input mode": geometry["input_mode"],
            "Analysis geometry record": str(geometry_path),
            "Analysis width pixels": analysis_width,
            "Analysis height pixels": analysis_height,
        }
    )
    with qc_path.open("w", encoding="utf-8") as handle:
        json.dump(qc_info, handle, indent=2, default=json_default)

    log_lines.extend(
        [
            "",
            "QUALITY CONTROL",
            f"Bad pixel count: {qc_info['bad_pixel_count']}",
            f"Total pixel count: {qc_info['total_pixel_count']}",
            f"Bad pixel fraction: {qc_info['bad_pixel_fraction']:.6f}",
            f"Large null region: {qc_info['large_null_region']}",
            f"QC status: {qc_info['qc_status']}",
            f"QC notes: {qc_info['qc_notes']}",
        ]
    )

    if STOP_IF_LARGE_NULL_REGION and qc_info["large_null_region"]:
        raise ValueError("Stopped: a large contiguous null region was detected.")

    if (
        qc_info["qc_status"] == "Reject"
        and not ALLOW_INTERPOLATION_FOR_REJECTED_ROIS
    ):
        raise ValueError("Stopped: raw-DTM QC rejected this ROI.")

    # 4. Cleaning/interpolation
    interpolation_mask_path = data_dir / "interpolated_pixel_mask.npy"
    interpolation_mask_plot_path = figure_dir / "interpolated_pixel_mask.png"

    if (
        qc_info["qc_status"] == "Reject"
        and ALLOW_INTERPOLATION_FOR_REJECTED_ROIS
    ):
        cleaned_dtm, interpolation_mask, cleaning_info = (
            processing.interpolate_missing_pixels(
                raw_dtm,
                qc_info,
                method=INTERPOLATION_METHOD,
            )
        )
        if SAVE_INTERPOLATION_MASK:
            np.save(interpolation_mask_path, interpolation_mask)
            save_boolean_mask(
                interpolation_mask,
                interpolation_mask_plot_path,
                f"{roi_id} interpolated-pixel mask",
            )
            cleaning_info["interpolation_mask_npy"] = str(
                interpolation_mask_path
            )
            cleaning_info["interpolation_mask_plot"] = str(
                interpolation_mask_plot_path
            )
    else:
        cleaned_dtm, cleaning_info = processing.clean_dtm_with_median_filter(
            raw_dtm, qc_info
        )

    np.save(cleaned_dtm_path, cleaned_dtm)
    processing.save_dtm_preview(
        cleaned_dtm,
        cleaned_preview_path,
        f"{roi_id} cleaned/interpolated DTM",
    )

    # 5. Detrending and PSD
    detrended_dtm, plane, plane_coefficients = processing.detrend_plane(
        cleaned_dtm
    )
    np.save(detrended_dtm_path, detrended_dtm)
    np.save(plane_path, plane)
    processing.save_dtm_preview(
        detrended_dtm,
        detrended_preview_path,
        f"{roi_id} detrended DTM",
    )

    processing.save_psd_diagnostic_multipanel_plot(
        dtm=detrended_dtm,
        pixel_scale_m=pixel_scale_m,
        output_path=psd_diagnostic_path,
        title=f"{roi_id} PSD diagnostic",
    )
    frequencies, psd, _ = processing.power_spectrum_1d(
        detrended_dtm, pixel_scale_m
    )
    np.savetxt(
        psd_csv_path,
        np.column_stack((frequencies, psd)),
        delimiter=",",
        header="spatial_frequency_1_per_m,power",
        comments="",
    )
    processing.save_psd_plot(
        frequencies, psd, psd_plot_path, f"{roi_id} PSD"
    )

    # 6. Deterministic RMS-deviation / Hurst analysis
    curve = exact_rms_deviation_curve_from_dtm(
        detrended_dtm,
        pixel_scale_m=pixel_scale_m,
        max_baseline_m=MAX_BASELINE_M,
        min_pairs_per_bin=MIN_PAIRS_PER_BIN,
    )
    baseline_fit, rms_fit, counts_fit, fit_range_info = select_fit_bins(
        curve,
        pixel_scale_m,
        analysis_width,
        analysis_height,
        max_baseline_m=MAX_BASELINE_M,
        min_fit_baseline_pixels=MIN_FIT_BASELINE_PIXELS,
        use_shepard_10_percent_limit=USE_SHEPARD_10_PERCENT_FIT_LIMIT,
        fit_range_fraction_of_min_roi_side=FIT_RANGE_FRACTION_OF_MIN_ROI_SIDE,
        min_points_each_side=MIN_POINTS_EACH_SIDE,
    )
    full_fit = fit_power_law(baseline_fit, rms_fit)
    breakpoint_info = detect_segmented_breakpoint(
        baseline_fit,
        rms_fit,
        min_points_each_side=MIN_POINTS_EACH_SIDE,
        edge_buffer_bins=EDGE_BUFFER_BINS,
        min_residual_improvement_fraction=MIN_RESIDUAL_IMPROVEMENT_FRACTION,
    )

    save_curve_csv(curve_csv_path, curve)
    save_fit_bins_csv(
        fit_bins_csv_path, baseline_fit, rms_fit, counts_fit
    )
    save_rms_breakpoint_plot(
        roi_id,
        curve,
        baseline_fit,
        rms_fit,
        full_fit,
        breakpoint_info,
        breakpoint_plot_path,
    )
    if SAVE_LINEAR_BASELINE_DIAGNOSTIC:
        save_linear_baseline_plot(roi_id, curve, linear_plot_path)

    # 7. Provenance summary
    summary = {
        "ROI_ID": roi_id,
        "Dataset": dataset_name,
        "Body": body_name,
        "timestamp": timestamp,
        "script": Path(__file__).name,
        "source_detrended_dtm": str(detrended_dtm_path),
        "catalog_metadata": {
            "site": clean_text(row.get("Site")),
            "terrain_type": clean_text(row.get("Terrain Type")),
            "product_type": clean_text(row.get("Product Type")),
            "roi_shape": clean_text(row.get("roi_shape")) or "axis_aligned_rect",
            "rotation_deg": clean_text(row.get("rotation_deg")),
            "pixel_scale_m_per_pixel": pixel_scale_m,
        },
        "analysis_geometry": geometry_record,
        "raw_source_info": raw_source_info,
        "quality_control": qc_info,
        "cleaning_info": cleaning_info,
        "detrending": {
            "method": "least_squares_best_fit_plane_subtraction",
            "plane_coefficients": plane_coefficients.tolist(),
            "plane_npy": str(plane_path),
        },
        "psd_outputs": {
            "psd_csv": str(psd_csv_path),
            "psd_plot": str(psd_plot_path),
            "psd_diagnostic_multipanel": str(psd_diagnostic_path),
            "note": "PSD uses the same detrended DTM as the roughness analysis.",
        },
        "steinbruegge_rms_deviation_method": {
            "equation": "y(Δx) = sqrt(mean([h(x_i) - h(x_i + Δx)]^2))",
            "pair_aggregation": curve["method"],
            "binning": "Linear baseline bins one DTM pixel wide.",
            "h_convention": (
                "H = slope of log10(RMS deviation) versus log10(baseline)."
            ),
            "breakpoint_method": breakpoint_info.get("method"),
            "curve_csv": str(curve_csv_path),
            "fit_bins_csv": str(fit_bins_csv_path),
            "breakpoint_plot": str(breakpoint_plot_path),
            "linear_bin_plot": (
                str(linear_plot_path)
                if SAVE_LINEAR_BASELINE_DIAGNOSTIC
                else ""
            ),
        },
        "curve_info": {
            "dtm_shape": curve["dtm_shape"],
            "valid_pixel_count": curve["valid_pixel_count"],
            "total_pixel_count": curve["total_pixel_count"],
            "valid_pixel_fraction": curve["valid_pixel_fraction"],
            "bin_width_m": curve["bin_width_m"],
            "max_baseline_used_m": curve["max_baseline_used_m"],
            "min_pairs_per_bin": MIN_PAIRS_PER_BIN,
        },
        "fit_range_info": fit_range_info,
        "full_power_law_fit": full_fit,
        "segmented_breakpoint": breakpoint_info,
        "uncertainty_note": (
            "This deterministic workflow does not generate vertical DTM error "
            "bars. Processing-sensitivity diagnostics are not a substitute for "
            "a validated product-specific error model."
        ),
        "interpretation_notes": [
            (
                "sigma0 is extrapolated to a 1 m baseline unless the DTM "
                "itself has 1 m sampling."
            ),
            (
                "A candidate breakpoint is not automatically a ridge spacing "
                "or radar wavelength."
            ),
            (
                "PSD is a separate Fourier-domain diagnostic using the same "
                "detrended DTM."
            ),
            (
                "Cross-world comparisons should match product type, "
                "projection, pixel scale, detrending, interpolation policy, "
                "and baseline range."
            ),
        ],
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=json_default)

    log_lines.extend(
        [
            "",
            "RMS-DEVIATION RESULTS",
            f"Full H: {full_fit['H']:.6f}",
            (
                "Full sigma0 at 1 m (extrapolated): "
                f"{full_fit['sigma0_m_at_1_m_extrapolated']:.6g} m"
            ),
            (
                "Fit baseline range: "
                f"{fit_range_info['fit_min_baseline_m']:.3f} to "
                f"{fit_range_info['fit_max_baseline_m']:.3f} m"
            ),
            f"Accepted breakpoint: {breakpoint_info.get('accepted', False)}",
            f"Breakpoint result: {breakpoint_info.get('reason', '')}",
        ]
    )
    if breakpoint_info.get("accepted", False):
        log_lines.extend(
            [
                (
                    "Candidate breakpoint: "
                    f"{breakpoint_info['breakpoint_baseline_m']:.6g} m"
                ),
                (
                    "H before breakpoint: "
                    f"{breakpoint_info['H_before_breakpoint']:.6f}"
                ),
                (
                    "H after breakpoint: "
                    f"{breakpoint_info['H_after_breakpoint']:.6f}"
                ),
            ]
        )

    log_lines.extend(
        ["", f"Analysis completed: {datetime.now().isoformat(timespec='seconds')}"]
    )
    log_path.write_text("\n".join(log_lines), encoding="utf-8")

    index_row = {
        "ROI_ID": roi_id,
        "Dataset": dataset_name,
        "Body": body_name,
        "Site": clean_text(row.get("Site")),
        "Terrain Type": clean_text(row.get("Terrain Type")),
        "Product Type": clean_text(row.get("Product Type")),
        "Pixel Scale m/pixel": pixel_scale_m,
        "ROI Shape": roi_shape,
        "QC Status": qc_info["qc_status"],
        "Bad Pixel Fraction": qc_info["bad_pixel_fraction"],
        "Interpolation Method": cleaning_info.get("cleaning_method", ""),
        "Fit Min Baseline m": fit_range_info["fit_min_baseline_m"],
        "Fit Max Baseline m": fit_range_info["fit_max_baseline_m"],
        "H Overall": full_fit["H"],
        "H Overall OLS SE": full_fit["H_standard_error_ols"],
        "Sigma0 m at 1m Extrapolated": (
            full_fit["sigma0_m_at_1_m_extrapolated"]
        ),
        "Breakpoint Accepted": breakpoint_info.get("accepted", False),
        "Breakpoint m": breakpoint_info.get("breakpoint_baseline_m", ""),
        "H Before BP": breakpoint_info.get("H_before_breakpoint", ""),
        "H After BP": breakpoint_info.get("H_after_breakpoint", ""),
        "RMS Curve CSV": str(curve_csv_path),
        "Fit Bins CSV": str(fit_bins_csv_path),
        "Summary JSON": str(summary_path),
        "Breakpoint Plot": str(breakpoint_plot_path),
        "PSD Plot": str(psd_plot_path),
        "Analysis Log": str(log_path),
        "Updated": timestamp,
    }
    update_summary_index(
        dataset["summaries_root"] / "tables" / "roi_analysis_summary.csv",
        index_row,
    )
    return summary


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point."""
    repo_root = repository_root_from_script(__file__)
    args = parse_cli(repo_root)
    config = load_project_config(args.config, repo_root)

    if args.list_datasets:
        for name, entry in config["datasets"].items():
            print(f"{name}: {entry['body_name']}")
        return

    dataset = resolve_dataset_config(config, args.dataset, repo_root)
    ensure_dataset_directories(dataset)
    configure_processing_module(dataset, repo_root)

    if args.all:
        roi_ids = processing.list_roi_ids()
    elif args.roi:
        roi_ids = [clean_text(roi_id) for roi_id in args.roi if clean_text(roi_id)]
    else:
        raise SystemExit(
            "Provide at least one --roi ROI_ID or use --all."
        )

    if not roi_ids:
        raise SystemExit("No ROI IDs were found to run.")

    failures: list[tuple[str, str]] = []

    for roi_id in roi_ids:
        try:
            summary = run_one_roi(dataset, roi_id, repo_root)
            print(
                f"Completed {roi_id}: "
                f"H={summary['full_power_law_fit']['H']:.3f}; "
                "breakpoint accepted="
                f"{summary['segmented_breakpoint'].get('accepted', False)}"
            )
        except Exception as exc:
            message = "".join(
                traceback.format_exception(type(exc), exc, exc.__traceback__)
            )
            failures.append((roi_id, message))

            failure_dir = dataset["troubleshooting_root"] / "failures"
            failure_dir.mkdir(parents=True, exist_ok=True)
            failure_path = failure_dir / (
                f"{roi_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                "_failure.txt"
            )
            failure_path.write_text(message, encoding="utf-8")
            print(
                f"FAILED {roi_id}. Details: {failure_path}",
                file=sys.stderr,
            )
            if args.fail_fast:
                raise

    if failures:
        failed_ids = ", ".join(roi_id for roi_id, _ in failures)
        raise SystemExit(
            f"{len(failures)} ROI(s) failed: {failed_ids}"
        )


if __name__ == "__main__":
    main()


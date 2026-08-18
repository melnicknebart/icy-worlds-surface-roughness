#!/usr/bin/env python3
""" Shared helpers for the reproducible icy-world-surface-roughness workflow.

This module deliberately contains no body-specific science settings. Edit only
``config/icy_worlds_config.json`` to direct Europa, Ganymede, or Enceladus
catalogs and results into separate folders.

The Hurst calculation implemented here follows the second-order RMS-deviation
structure-function form used by Steinbrügge et al. (2020):

    y(Δx) = sqrt(mean([h(x_i) - h(x_i + Δx)]**2))

It uses all valid pair *displacements* through FFT cross-correlations, then
combines them into isotropic, linear baseline bins one pixel wide. This is a
reproducible deterministic estimator; it is not an error model by itself.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.signal import fftconvolve


REQUIRED_DATASET_KEYS = {"body_name", "catalog_path", "results_root"}


def json_default(value: Any) -> Any:
    """Convert common NumPy/Path objects to JSON-safe values."""
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Cannot JSON-serialize {type(value)!r}")


def clean_text(value: Any) -> str:
    """Return a stripped string while handling None values safely."""
    return "" if value is None else str(value).strip()


def normalize_header(header: Any) -> str:
    """Normalize catalog headers for forgiving matching."""
    return "".join(ch for ch in clean_text(header).casefold() if ch.isalnum())


def get_catalog_value(row: dict[str, Any], *possible_headers: str) -> str:
    """Return the first nonblank value matching any header spelling."""
    normalized = {normalize_header(key): value for key, value in row.items()}
    for header in possible_headers:
        value = normalized.get(normalize_header(header), "")
        if clean_text(value):
            return clean_text(value)
    return ""


def safe_positive_float(value: Any, field_name: str) -> float:
    """Parse a finite positive float with a useful error message."""
    text = clean_text(value)
    if text == "" or text.casefold() in {"nan", "none", "unknown"}:
        raise ValueError(f"{field_name} is blank or invalid.")
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(f"Could not parse {field_name}={text!r} as a number.") from exc
    if not np.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{field_name} must be a finite positive number; got {text!r}.")
    return parsed


def repository_root_from_script(script_path: str | Path) -> Path:
    """Assume this module lives in <repo>/scripts/ and return <repo>."""
    return Path(script_path).resolve().parents[1]


def load_project_config(config_path: str | Path, repo_root: str | Path) -> dict[str, Any]:
    """Load and validate the central multi-body workflow config."""
    config_path = Path(config_path)
    if not config_path.is_absolute():
        config_path = Path(repo_root) / config_path
    if not config_path.exists():
        raise FileNotFoundError(
            f"Could not find workflow config: {config_path}. "
            "Copy icy_worlds_config.json into <repo>/config/."
        )
    with open(config_path, "r", encoding="utf-8") as handle:
        config = json.load(handle)
    datasets = config.get("datasets")
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("Workflow config must contain a nonempty 'datasets' object.")
    for dataset_name, dataset in datasets.items():
        if not isinstance(dataset, dict):
            raise ValueError(f"Dataset {dataset_name!r} must be an object.")
        missing = REQUIRED_DATASET_KEYS - set(dataset)
        if missing:
            raise ValueError(f"Dataset {dataset_name!r} is missing keys: {sorted(missing)}")
    config["_config_path"] = str(config_path)
    return config


def resolve_dataset_config(
    config: dict[str, Any], dataset_name: str, repo_root: str | Path
) -> dict[str, Any]:
    """Resolve one dataset's relative paths to absolute paths."""
    datasets = config["datasets"]
    if dataset_name not in datasets:
        raise KeyError(
            f"Unknown dataset {dataset_name!r}. Choose one of: {sorted(datasets)}"
        )
    raw = dict(datasets[dataset_name])
    root = Path(repo_root)
    resolved = dict(raw)
    resolved["dataset_name"] = dataset_name
    resolved["catalog_path"] = (root / raw["catalog_path"]).resolve()
    resolved["results_root"] = (root / raw["results_root"]).resolve()
    resolved["selection_root"] = resolved["results_root"] / "selection"
    resolved["reproducible_rois_root"] = resolved["results_root"] / "reproducible_rois"
    resolved["summaries_root"] = resolved["results_root"] / "summaries"
    resolved["troubleshooting_root"] = resolved["results_root"] / "troubleshooting"
    return resolved


def ensure_dataset_directories(dataset: dict[str, Any]) -> None:
    """Create the standard non-destructive folder tree for one icy world."""
    dataset["catalog_path"].parent.mkdir(parents=True, exist_ok=True)
    for directory in (
        dataset["selection_root"] / "previews",
        dataset["selection_root"] / "qc_previews",
        dataset["selection_root"] / "logs",
        dataset["selection_root"] / "rotated_dtms",
        dataset["selection_root"] / "temporary_qc",
        dataset["reproducible_rois_root"],
        dataset["summaries_root"] / "figures",
        dataset["summaries_root"] / "tables",
        dataset["troubleshooting_root"] / "failures",
    ):
        directory.mkdir(parents=True, exist_ok=True)


def cross_correlation_2d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Full 2-D cross-correlation through FFT convolution."""
    return fftconvolve(a, b[::-1, ::-1], mode="full")


def exact_rms_deviation_curve_from_dtm(
    detrended_dtm: np.ndarray,
    pixel_scale_m: float,
    max_baseline_m: float | None,
    min_pairs_per_bin: int,
) -> dict[str, Any]:
    """Compute a Steinbrügge-style RMS-deviation curve from a 2-D DTM.

    The literal N(N-1)/2 list of pixel pairs would be impractical for normal
    ROIs. Instead, FFT correlations calculate the exact pair count and exact
    sum of squared elevation differences for each integer x/y displacement.
    Distances are then grouped into linear bins centered at p, 2p, 3p, ...,
    where p is the DTM pixel scale.
    """
    dtm = np.asarray(detrended_dtm, dtype=np.float64)
    if dtm.ndim != 2:
        raise ValueError(f"Expected a 2-D DTM, received shape {dtm.shape}.")
    ny, nx = dtm.shape
    if ny < 2 or nx < 2:
        raise ValueError("DTM needs at least two rows and two columns.")
    if pixel_scale_m <= 0 or not np.isfinite(pixel_scale_m):
        raise ValueError("pixel_scale_m must be a positive finite number.")

    valid = np.isfinite(dtm)
    if int(valid.sum()) < 2:
        raise ValueError("DTM has fewer than two finite elevations.")

    mask = valid.astype(np.float64)
    z = np.where(valid, dtm, 0.0)
    z2 = np.where(valid, dtm * dtm, 0.0)

    pair_count_grid = cross_correlation_2d(mask, mask)
    left_sq_grid = cross_correlation_2d(z2, mask)
    right_sq_grid = cross_correlation_2d(mask, z2)
    cross_grid = cross_correlation_2d(z, z)
    sum_sq_grid = left_sq_grid + right_sq_grid - 2.0 * cross_grid

    # Remove only negligible FFT roundoff below true zero.
    tolerance = 1e-9 * max(1.0, float(np.nanmax(np.abs(sum_sq_grid))))
    sum_sq_grid[(sum_sq_grid < 0) & (sum_sq_grid > -tolerance)] = 0.0
    sum_sq_grid = np.maximum(sum_sq_grid, 0.0)

    dy_grid, dx_grid = np.indices(pair_count_grid.shape)
    dy_grid = dy_grid - (ny - 1)
    dx_grid = dx_grid - (nx - 1)

    # One half plane counts each unordered pair only once.
    half_plane = (dy_grid > 0) | ((dy_grid == 0) & (dx_grid > 0))
    dy = dy_grid[half_plane].astype(float)
    dx = dx_grid[half_plane].astype(float)
    pair_counts = np.rint(pair_count_grid[half_plane]).astype(np.int64)
    sum_sq = sum_sq_grid[half_plane]

    distance_m = np.hypot(dx, dy) * float(pixel_scale_m)
    keep = (pair_counts > 0) & np.isfinite(distance_m) & (distance_m > 0)
    if max_baseline_m is not None:
        keep &= distance_m <= float(max_baseline_m)

    distance_m = distance_m[keep]
    pair_counts = pair_counts[keep]
    sum_sq = sum_sq[keep]
    if distance_m.size == 0:
        raise ValueError("No valid pair displacements remain after baseline filtering.")

    # Bins are one pixel scale wide: center p, 2p, 3p, ...
    bin_index = np.floor(distance_m / float(pixel_scale_m) + 0.5).astype(int)
    bin_index = np.maximum(bin_index, 1)
    max_bin = int(bin_index.max())

    count_by_bin = np.bincount(
        bin_index, weights=pair_counts.astype(float), minlength=max_bin + 1
    )
    sum_sq_by_bin = np.bincount(bin_index, weights=sum_sq, minlength=max_bin + 1)

    bin_numbers = np.arange(1, max_bin + 1)
    baseline_m = bin_numbers.astype(float) * float(pixel_scale_m)
    n_pairs = np.rint(count_by_bin[1:]).astype(np.int64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rms = np.sqrt(sum_sq_by_bin[1:] / n_pairs)

    usable = (n_pairs >= int(min_pairs_per_bin)) & np.isfinite(rms) & (rms > 0)
    if int(usable.sum()) < 6:
        raise ValueError(
            "Too few RMS-deviation bins passed the pair-count threshold. "
            "Use a larger ROI or lower min_pairs_per_bin."
        )

    return {
        "baseline_center_m": baseline_m[usable],
        "bin_lower_m": np.maximum(0.0, baseline_m[usable] - 0.5 * pixel_scale_m),
        "bin_upper_m": baseline_m[usable] + 0.5 * pixel_scale_m,
        "rms_deviation_m": rms[usable],
        "n_distinct_pixel_pairs": n_pairs[usable],
        "all_baseline_center_m": baseline_m,
        "all_rms_deviation_m": rms,
        "all_n_distinct_pixel_pairs": n_pairs,
        "valid_pixel_count": int(valid.sum()),
        "total_pixel_count": int(dtm.size),
        "valid_pixel_fraction": float(valid.mean()),
        "dtm_shape": [int(ny), int(nx)],
        "bin_width_m": float(pixel_scale_m),
        "max_baseline_used_m": float(np.max(baseline_m[usable])),
        "method": (
            "Exact all-valid-unordered-pixel-pair aggregation through FFT "
            "cross-correlations; isotropic linear distance bins one DTM "
            "resolution element wide."
        ),
    }


def select_fit_bins(
    curve: dict[str, Any],
    pixel_scale_m: float,
    analysis_width_pixels: int,
    analysis_height_pixels: int,
    *,
    max_baseline_m: float | None,
    min_fit_baseline_pixels: float,
    use_shepard_10_percent_limit: bool,
    fit_range_fraction_of_min_roi_side: float,
    min_points_each_side: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply transparent baseline rules and separate full-fit from breakpoint eligibility.

    A ROI can contain enough scale bins for one overall Hurst fit but not enough
    for a statistically supported two-segment breakpoint test.  That is an
    informative outcome, not a fatal error: the caller receives the full-fit
    bins and ``breakpoint_test_eligible=False`` in the returned metadata.
    """
    baseline = np.asarray(curve["baseline_center_m"], dtype=float)
    rms = np.asarray(curve["rms_deviation_m"], dtype=float)
    counts = np.asarray(curve["n_distinct_pixel_pairs"], dtype=np.int64)

    finite_positive_baselines = baseline[np.isfinite(baseline) & (baseline > 0)]
    if finite_positive_baselines.size == 0:
        raise ValueError("The RMS-deviation curve contains no positive finite baseline bins.")

    min_baseline_m = float(min_fit_baseline_pixels) * float(pixel_scale_m)
    max_by_curve = float(np.max(finite_positive_baselines))
    max_by_config = max_by_curve if max_baseline_m is None else min(max_by_curve, float(max_baseline_m))

    effective_min_side_m = min(analysis_width_pixels, analysis_height_pixels) * float(pixel_scale_m)
    if use_shepard_10_percent_limit:
        max_by_shepard = fit_range_fraction_of_min_roi_side * effective_min_side_m
        fit_max_m = min(max_by_config, max_by_shepard)
        fit_rule = "min(configured max, fraction × minimum ROI side)"
    else:
        max_by_shepard = None
        fit_max_m = max_by_config
        fit_rule = "configured maximum baseline"

    keep = (
        (baseline >= min_baseline_m)
        & (baseline <= fit_max_m)
        & np.isfinite(baseline)
        & np.isfinite(rms)
        & (baseline > 0)
        & (rms > 0)
        & (counts > 0)
    )

    n_bins_used = int(np.count_nonzero(keep))
    # A log-log line with an estimated residual spread needs at least 3 bins.
    min_bins_for_full_fit = 3
    # This mirrors detect_segmented_breakpoint(), which requires an interior
    # split plus the requested number of bins for each fitted segment.
    effective_min_points_each_side = max(3, int(min_points_each_side))
    min_bins_for_breakpoint = 2 * effective_min_points_each_side + 1
    breakpoint_test_eligible = n_bins_used >= min_bins_for_breakpoint

    info = {
        "fit_range_rule": fit_rule,
        "fit_min_baseline_m": float(min_baseline_m),
        "fit_max_baseline_m": float(fit_max_m),
        "configured_max_baseline_m": max_baseline_m,
        "shepard_10_percent_limit_enabled": bool(use_shepard_10_percent_limit),
        "fit_range_fraction_of_min_roi_side": (
            float(fit_range_fraction_of_min_roi_side)
            if use_shepard_10_percent_limit
            else None
        ),
        "effective_min_roi_side_m": float(effective_min_side_m),
        "shepard_max_baseline_m": float(max_by_shepard) if max_by_shepard is not None else None,
        "min_fit_baseline_pixels": float(min_fit_baseline_pixels),
        "n_bins_total": int(len(baseline)),
        "n_bins_used_for_fit": n_bins_used,
        "n_bins_required_for_full_fit": min_bins_for_full_fit,
        "n_bins_required_for_breakpoint_test": min_bins_for_breakpoint,
        "effective_min_points_each_side": effective_min_points_each_side,
        "full_fit_eligible": n_bins_used >= min_bins_for_full_fit,
        "breakpoint_test_eligible": breakpoint_test_eligible,
        "breakpoint_status": (
            "eligible"
            if breakpoint_test_eligible
            else "insufficient_fit_bins_for_two_segment_test"
        ),
    }

    if n_bins_used < min_bins_for_full_fit:
        raise ValueError(
            "Not enough usable baseline bins for even one overall Hurst fit: "
            f"kept {n_bins_used}, need at least {min_bins_for_full_fit}. "
            "Use a larger/cleaner ROI or review the fit-range limits."
        )

    print(
        "[FIT RANGE] "
        f"kept {n_bins_used}/{len(baseline)} bins; "
        f"range {min_baseline_m:.6g}–{fit_max_m:.6g} m; "
        f"breakpoint test: {info['breakpoint_status']} "
        f"(needs {min_bins_for_breakpoint} bins)."
    )
    return baseline[keep], rms[keep], counts[keep], info

def fit_power_law(baseline_m: np.ndarray, rms_m: np.ndarray) -> dict[str, Any]:
    """Fit y = sigma0 * baseline**H in log10 space with OLS diagnostics."""
    baseline = np.asarray(baseline_m, dtype=float)
    rms = np.asarray(rms_m, dtype=float)
    keep = np.isfinite(baseline) & np.isfinite(rms) & (baseline > 0) & (rms > 0)
    x = np.log10(baseline[keep])
    y = np.log10(rms[keep])
    if len(x) < 3:
        raise ValueError("Need at least three valid bins for a power-law fit.")

    design = np.column_stack((x, np.ones_like(x)))
    beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    slope, intercept = float(beta[0]), float(beta[1])
    residual = y - design @ beta
    rss = float(np.sum(residual**2))
    dof = int(len(x) - 2)
    residual_variance = float(rss / dof) if dof > 0 else float("nan")
    try:
        covariance = residual_variance * np.linalg.inv(design.T @ design)
        h_se = float(np.sqrt(max(covariance[0, 0], 0.0)))
        intercept_se = float(np.sqrt(max(covariance[1, 1], 0.0)))
    except np.linalg.LinAlgError:
        covariance = np.full((2, 2), np.nan)
        h_se = float("nan")
        intercept_se = float("nan")

    return {
        "H": slope,
        "log10_sigma0": intercept,
        "sigma0_m_at_1_m_extrapolated": float(10.0**intercept),
        "residual_sum_squares": rss,
        "degrees_of_freedom": dof,
        "residual_variance_log10": residual_variance,
        "residual_standard_deviation_log10": float(np.sqrt(residual_variance)) if residual_variance >= 0 else float("nan"),
        "H_standard_error_ols": h_se,
        "log10_sigma0_standard_error_ols": intercept_se,
        "coefficient_covariance_log10": covariance.tolist(),
    }


def predict_power_law(baseline_m: np.ndarray, fit: dict[str, Any]) -> np.ndarray:
    """Evaluate a fitted power law at positive baseline values."""
    x = np.asarray(baseline_m, dtype=float)
    return (10.0 ** float(fit["log10_sigma0"])) * (x ** float(fit["H"]))


def detect_segmented_breakpoint(
    baseline_m: np.ndarray,
    rms_m: np.ndarray,
    *,
    min_points_each_side: int,
    edge_buffer_bins: int,
    min_residual_improvement_fraction: float,
) -> dict[str, Any]:
    """Deterministic two-segment breakpoint diagnostic in log-log space."""
    baseline = np.asarray(baseline_m, dtype=float)
    rms = np.asarray(rms_m, dtype=float)
    n = len(baseline)
    # Each individual line fit needs at least three bins for a slope and
    # residual-variance estimate, even when a caller requests a lower value.
    effective_min_points = max(3, int(min_points_each_side))
    if n < 2 * effective_min_points + 1:
        return {
            "available": False,
            "accepted": False,
            "method": "project_segmented_least_squares_loglog",
            "reason": "Not enough bins for a two-segment breakpoint test.",
            "n_points_total": int(n),
        }

    full = fit_power_law(baseline, rms)
    start = max(effective_min_points, int(edge_buffer_bins))
    stop = min(n - effective_min_points, n - int(edge_buffer_bins))

    best: dict[str, Any] | None = None
    for split in range(start, stop):
        before = fit_power_law(baseline[:split], rms[:split])
        after = fit_power_law(baseline[split:], rms[split:])
        two_rss = before["residual_sum_squares"] + after["residual_sum_squares"]
        if best is None or two_rss < best["two_segment_residual_sum_squares"]:
            best = {
                "candidate_breakpoint_index": int(split),
                "breakpoint_baseline_m": float(baseline[split]),
                "breakpoint_log10_baseline": float(np.log10(baseline[split])),
                "H_before_breakpoint": float(before["H"]),
                "log10_sigma0_before": float(before["log10_sigma0"]),
                "sigma0_before_m_at_1_m_extrapolated": float(before["sigma0_m_at_1_m_extrapolated"]),
                "H_after_breakpoint": float(after["H"]),
                "log10_sigma0_after": float(after["log10_sigma0"]),
                "sigma0_after_m_at_1_m_extrapolated": float(after["sigma0_m_at_1_m_extrapolated"]),
                "n_points_before": int(split),
                "n_points_after": int(n - split),
                "two_segment_residual_sum_squares": float(two_rss),
            }

    if best is None:
        return {
            "available": False,
            "accepted": False,
            "method": "project_segmented_least_squares_loglog",
            "reason": "No interior breakpoint candidate could be tested.",
            "n_points_total": int(n),
        }

    full_rss = float(full["residual_sum_squares"])
    improvement = 0.0 if full_rss <= 0 else (full_rss - best["two_segment_residual_sum_squares"]) / full_rss
    accepted = improvement >= float(min_residual_improvement_fraction)
    result = {
        "available": True,
        "accepted": bool(accepted),
        "method": "project_segmented_least_squares_loglog",
        "reason": (
            "Accepted candidate breakpoint using the project's residual-improvement rule."
            if accepted
            else "No accepted breakpoint: two segments did not improve residuals enough."
        ),
        "n_points_total": int(n),
        "full_fit_residual_sum_squares": full_rss,
        "residual_improvement_fraction": float(improvement),
        "minimum_required_improvement_fraction": float(min_residual_improvement_fraction),
        "min_points_each_side": int(effective_min_points),
        "edge_buffer_bins": int(edge_buffer_bins),
        "note": (
            "This is the project's explicit segmented-fit diagnostic. It is not claimed "
            "to be the unpublished exact numerical breakpoint procedure of any paper."
        ),
    }
    result.update(best)
    return result


def save_curve_csv(path: Path, curve: dict[str, Any]) -> None:
    """Save the full usable RMS-deviation curve."""
    np.savetxt(
        path,
        np.column_stack(
            (
                curve["baseline_center_m"],
                curve["bin_lower_m"],
                curve["bin_upper_m"],
                curve["rms_deviation_m"],
                curve["n_distinct_pixel_pairs"],
            )
        ),
        delimiter=",",
        header="baseline_center_m,bin_lower_m,bin_upper_m,rms_deviation_m,n_distinct_pixel_pairs",
        comments="",
    )


def save_fit_bins_csv(path: Path, baseline: np.ndarray, rms: np.ndarray, counts: np.ndarray) -> None:
    """Save the exact bins included in the Hurst/breakpoint fit."""
    np.savetxt(
        path,
        np.column_stack((baseline, rms, counts)),
        delimiter=",",
        header="baseline_m,rms_deviation_m,n_distinct_pixel_pairs",
        comments="",
    )


def read_numeric_csv(path: str | Path) -> dict[str, np.ndarray]:
    """Read a small numeric CSV written by this workflow."""
    data = np.genfromtxt(path, delimiter=",", names=True, dtype=float, encoding="utf-8")
    if data.size == 0:
        raise ValueError(f"No rows found in {path}")
    if data.ndim == 0:
        data = np.array([data])
    return {name: np.asarray(data[name], dtype=float) for name in data.dtype.names or ()}


def update_summary_index(index_path: Path, summary_row: dict[str, Any]) -> None:
    """Update one body-level table, replacing an older row with the same ROI_ID."""
    index_path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, str]] = []
    fieldnames: list[str] = []
    if index_path.exists() and index_path.stat().st_size > 0:
        with open(index_path, "r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fieldnames = list(reader.fieldnames or [])
            existing = [dict(row) for row in reader]

    for key in summary_row:
        if key not in fieldnames:
            fieldnames.append(key)

    normalized = {key: "" if value is None else str(value) for key, value in summary_row.items()}
    roi_id = normalized.get("ROI_ID", "")
    replacement_done = False
    updated: list[dict[str, str]] = []
    for row in existing:
        if row.get("ROI_ID", "") == roi_id:
            merged = {field: row.get(field, "") for field in fieldnames}
            merged.update(normalized)
            updated.append(merged)
            replacement_done = True
        else:
            updated.append({field: row.get(field, "") for field in fieldnames})
    if not replacement_done:
        updated.append({field: normalized.get(field, "") for field in fieldnames})

    with open(index_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated)

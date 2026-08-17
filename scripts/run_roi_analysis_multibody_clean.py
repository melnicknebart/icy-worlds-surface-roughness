"""Interactive multibody ROI selector with separated body-specific catalogs and outputs.

Switch body by changing ACTIVE_DATASET below; edit config/icy_worlds_config.json
to change where that body stores its catalog and results.
"""

import cv2
import csv
import json
import re
import shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path
from datetime import date, datetime

from icy_workflow_common import (
    ensure_dataset_directories,
    load_project_config,
    repository_root_from_script,
    resolve_dataset_config,
)

#python scripts/select_roi_for_catalog_multibody_clean.py

# ============================================================
# USER SETTINGS — EDIT THESE EACH TIME YOU SELECT A NEW ROI
# ============================================================

ROI_ID = "SITE_TERRAIN_TYPE_001"
SITE = "Site_Name"
TERRAIN_TYPE = "Terrain_Type"
PRODUCT_FOLDER = "Folder_Name"
PRODUCT_TYPE = "Product_Type_Name"

SOURCE_CUB = "path/file.cub"
SOURCE_TIFF = "path/file.tif"

PIXEL_SCALE_M =  # Check using metadata, spreadsheet, or mentor guidance.

REASON_FOR_ROI = "Representative terrain_type ROI selected for ROI provenance."
QUALITY_FLAG = "Good"
STATUS = "Selected"

SELECTED_BY = "Selector name"


# ============================================================
# ROTATED ROI SETTINGS
# ============================================================

# This is the key setting.
# The script rotates the TIFF display by this many degrees BEFORE selection.
# You then draw a normal rectangle on that rotated view.
# The script maps that rectangle back to the original image as a rotated ROI.
#
# Try positive or negative values until the feature you care about looks horizontal.
# Example:
#   0    = normal axis-aligned ROI
#   25   = rotate display 25 degrees counterclockwise before selecting
#   -25  = rotate display 25 degrees clockwise before selecting
SELECTION_ROTATION_DEG = 0

# Resize very large images for easier selection.
# 1.0 = original display size
# 0.5 = half size, faster for huge files
# 0.25 = quarter size, much faster
# 2.0 = enlarged, useful only for small images
DISPLAY_SCALE = 0.5

# ============================================================
# PREVIEW-SAVING SETTINGS
# ============================================================

# The full preview is always saved. It shows:
#   red polygon = true science ROI
#   blue rectangle = temporary ISIS/CUB bounding-box crop
#
# Save the straightened crop preview only when selection used a real
# non-zero rotation. At zero rotation, the full preview is enough.
SAVE_ROTATED_CROP_PREVIEW_ONLY_WHEN_ROTATED = True

# This was useful while debugging coordinate transforms, but is not needed
# for normal cataloging. Turn it on only when debugging a new change.
SAVE_OPENCV_SELECTED_WINDOW_PREVIEW = False

# Treat very small angles as zero.
ROTATION_ZERO_TOLERANCE_DEG = 1e-6


# ============================================================
# AUTOMATIC QC SETTINGS
# ============================================================

AUTO_QC_BEFORE_SAVE = True

NULL_VALUE = -9999
NULL_THRESHOLD = -9000

GOOD_MAX_BAD_FRACTION = 0.01
QUESTIONABLE_MAX_BAD_FRACTION = 0.05
NEEDS_REVIEW_MAX_BAD_FRACTION = 0.20

# If True, a large connected null region causes rejection even if total bad fraction is low.
REJECT_LARGE_NULL_REGION = False


# ============================================================
# DATASET / OUTPUT LOCATIONS
# ============================================================
#
# The central JSON config is the only place where dataset catalog/output paths
# are defined. Change ACTIVE_DATASET here to select a body, or edit the JSON
# config to relocate that body's catalog and results root.

ACTIVE_DATASET = "europa"
REPO_ROOT = repository_root_from_script(__file__)
CONFIG_PATH = REPO_ROOT / "config" / "icy_worlds_config.json"
WORKFLOW_CONFIG = load_project_config(CONFIG_PATH, REPO_ROOT)
DATASET = resolve_dataset_config(WORKFLOW_CONFIG, ACTIVE_DATASET, REPO_ROOT)
ensure_dataset_directories(DATASET)

BODY_NAME = DATASET["body_name"]
CATALOG_PATH = DATASET["catalog_path"]
SELECTION_ROOT = DATASET["selection_root"]
PREVIEW_DIR = SELECTION_ROOT / "previews"
LOG_DIR = SELECTION_ROOT / "logs"
TEMP_QC_DIR = SELECTION_ROOT / "temporary_qc"
QC_PREVIEW_DIR = SELECTION_ROOT / "qc_previews"
ROTATED_DTM_DIR = SELECTION_ROOT / "rotated_dtms"

# No manually created output folder is required beyond ensure_dataset_directories,
# but keeping these mkdir calls makes this selector safe when copied on its own.
for _directory in (PREVIEW_DIR, LOG_DIR, TEMP_QC_DIR, QC_PREVIEW_DIR, ROTATED_DTM_DIR):
    _directory.mkdir(parents=True, exist_ok=True)
CATALOG_PATH.parent.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPER FUNCTIONS — DISPLAY / ROTATION / GEOMETRY
# ============================================================

def normalize_for_display(img):
    """
    Convert image to uint8 so OpenCV can display it clearly.

    This does NOT change the scientific source file.
    It only makes a viewable image for selecting ROI.
    """
    display = img.copy()

    if display.ndim > 2:
        display = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)

    if display.dtype != np.uint8:
        display = display.astype(np.float32)
        finite_mask = np.isfinite(display)

        if not finite_mask.any():
            raise ValueError("Image has no finite values to display.")

        display_min = display[finite_mask].min()
        display_max = display[finite_mask].max()

        if display_max == display_min:
            display = np.zeros_like(display, dtype=np.uint8)
        else:
            display = (display - display_min) / (display_max - display_min)
            display = np.clip(display * 255, 0, 255).astype(np.uint8)

    return display


def rotate_image_bound(img, angle_deg, border_value=0):
    """
    Rotate image by an arbitrary angle while expanding the canvas
    so the rotated image is not clipped.

    Returns:
        rotated_img
        affine_matrix_original_to_rotated
    """
    height, width = img.shape[:2]
    center = (width / 2.0, height / 2.0)

    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)

    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])

    new_width = int((height * sin) + (width * cos))
    new_height = int((height * cos) + (width * sin))

    matrix[0, 2] += (new_width / 2.0) - center[0]
    matrix[1, 2] += (new_height / 2.0) - center[1]

    rotated = cv2.warpAffine(
        img,
        matrix,
        (new_width, new_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=border_value,
    )

    return rotated, matrix


def apply_affine_to_points(points, matrix):
    """
    Apply a 2x3 affine transform to a list/array of xy points.
    """
    points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
    transformed = cv2.transform(points, matrix).reshape(-1, 2)
    return transformed


def selected_rectangle_to_original_geometry(
    x_display,
    y_display,
    w_display,
    h_display,
    display_scale,
    matrix_original_to_rotated,
):
    """
    Convert the rectangle selected on the scaled, rotated display back into
    a rotated quadrilateral in original image coordinates.
    """
    # First undo display scaling.
    x_rot = x_display / display_scale
    y_rot = y_display / display_scale
    w_rot = w_display / display_scale
    h_rot = h_display / display_scale

    # Corners in the rotated-image coordinate system.
    # Order matters: top-left, top-right, bottom-right, bottom-left.
    corners_rotated = np.array(
        [
            [x_rot, y_rot],
            [x_rot + w_rot - 1, y_rot],
            [x_rot + w_rot - 1, y_rot + h_rot - 1],
            [x_rot, y_rot + h_rot - 1],
        ],
        dtype=np.float32,
    )

    # Map rotated-image coordinates back to original image coordinates.
    matrix_rotated_to_original = cv2.invertAffineTransform(matrix_original_to_rotated)
    corners_original = apply_affine_to_points(corners_rotated, matrix_rotated_to_original)

    rotated_width = int(round(w_rot))
    rotated_height = int(round(h_rot))

    rotated_width = max(rotated_width, 1)
    rotated_height = max(rotated_height, 1)

    return corners_original, rotated_width, rotated_height


def bounding_box_from_corners(corners, original_width, original_height):
    """
    Get a clipped axis-aligned bounding box around rotated ROI corners.
    """
    corners = np.asarray(corners, dtype=np.float32)

    min_x = int(np.floor(np.min(corners[:, 0])))
    min_y = int(np.floor(np.min(corners[:, 1])))
    max_x = int(np.ceil(np.max(corners[:, 0])))
    max_y = int(np.ceil(np.max(corners[:, 1])))

    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(original_width - 1, max_x)
    max_y = min(original_height - 1, max_y)

    width = max_x - min_x + 1
    height = max_y - min_y + 1

    if width <= 0 or height <= 0:
        raise ValueError("Rotated ROI bounding box has invalid width/height.")

    return min_x, min_y, width, height


def warp_array_to_rotated_roi(
    array,
    corners_original,
    bbox_x,
    bbox_y,
    rotated_width,
    rotated_height,
    is_dtm=False,
):
    """
    Extract the true rotated ROI from an axis-aligned bounding crop using
    a perspective transform.

    For DTM arrays, missing values are preserved using a warped valid-data mask.
    """
    array = np.asarray(array)

    src = np.asarray(corners_original, dtype=np.float32).copy()
    src[:, 0] -= bbox_x
    src[:, 1] -= bbox_y

    dst = np.array(
        [
            [0, 0],
            [rotated_width - 1, 0],
            [rotated_width - 1, rotated_height - 1],
            [0, rotated_height - 1],
        ],
        dtype=np.float32,
    )

    transform = cv2.getPerspectiveTransform(src, dst)

    if is_dtm:
        dtm = array.astype(np.float32)

        bad_mask = (~np.isfinite(dtm)) | (dtm <= NULL_THRESHOLD)

        dtm_for_warp = dtm.copy()
        dtm_for_warp[bad_mask] = np.nan

        valid_mask = (~bad_mask).astype(np.uint8)

        warped = cv2.warpPerspective(
            dtm_for_warp,
            transform,
            (rotated_width, rotated_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=np.nan,
        )

        warped_valid = cv2.warpPerspective(
            valid_mask,
            transform,
            (rotated_width, rotated_height),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        warped[(warped_valid == 0) | (~np.isfinite(warped))] = NULL_VALUE

        return warped

    else:
        warped = cv2.warpPerspective(
            array,
            transform,
            (rotated_width, rotated_height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        return warped


def corners_to_json(corners):
    """
    Convert corner coordinates to a JSON string for CSV storage.
    """
    corners = np.asarray(corners, dtype=float)
    rounded = [[round(float(x), 3), round(float(y), 3)] for x, y in corners]
    return json.dumps(rounded)


# ============================================================
# HELPER FUNCTIONS — ISIS TEMP QC AND ROTATED DTM EXTRACTION
# ============================================================

def run_command(command):
    """
    Run a terminal command and raise an error if it fails.
    """
    result = subprocess.run(
        [str(c) for c in command],
        capture_output=True,
        text=True
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed:\n{' '.join(str(c) for c in command)}\n\n"
            f"STDOUT:\n{result.stdout}\n\n"
            f"STDERR:\n{result.stderr}"
        )

    return result


def crop_cub_for_qc(source_cub, output_cub, x, y, width, height):
    """
    Temporarily crop the source CUB using an axis-aligned bounding box.

    Note:
    This follows the same coordinate convention as the previous selector.
    """
    command = [
        "crop",
        f"from={source_cub}",
        f"to={output_cub}",
        f"SAMPLE={x}",
        f"LINE={y}",
        f"NSAMPLES={width}",
        f"NLINES={height}",
    ]

    run_command(command)


def cub_to_ascii_for_qc(input_cub, output_txt):
    """
    Convert temporary cropped CUB to ASCII so NumPy can read it.
    """
    command = [
        "isis2ascii",
        f"from={input_cub}",
        f"to={output_txt}",
        "HEADER=false",
        "SETPIXELVALUE=true",
        f"NULLVALUE={NULL_VALUE}",
    ]

    run_command(command)


def load_bbox_dtm_from_cub(source_cub, roi_id, bbox_x, bbox_y, bbox_w, bbox_h):
    """
    Crop the source CUB temporarily using the bounding box and load as NumPy.
    """
    temp_cub = TEMP_QC_DIR / f"{roi_id}_bbox_temp_qc.cub"
    temp_txt = TEMP_QC_DIR / f"{roi_id}_bbox_temp_qc.txt"

    if temp_cub.exists():
        temp_cub.unlink()
    if temp_txt.exists():
        temp_txt.unlink()

    crop_cub_for_qc(source_cub, temp_cub, bbox_x, bbox_y, bbox_w, bbox_h)
    cub_to_ascii_for_qc(temp_cub, temp_txt)

    dtm = np.loadtxt(temp_txt)

    return dtm, temp_cub, temp_txt


def load_rotated_dtm_from_selected_roi(
    source_cub,
    roi_id,
    bbox_x,
    bbox_y,
    bbox_w,
    bbox_h,
    corners_original,
    rotated_width,
    rotated_height,
):
    """
    Load a real rotated DTM extraction.

    Process:
    1. Crop the axis-aligned bounding box from the CUB using ISIS.
    2. Convert that temporary CUB to ASCII.
    3. Use the rotated ROI corners to perspective-warp the DTM into
       a true rectangular rotated ROI.
    """
    bbox_dtm, temp_cub, temp_txt = load_bbox_dtm_from_cub(
        source_cub,
        roi_id,
        bbox_x,
        bbox_y,
        bbox_w,
        bbox_h,
    )

    rotated_dtm = warp_array_to_rotated_roi(
        bbox_dtm,
        corners_original,
        bbox_x,
        bbox_y,
        rotated_width,
        rotated_height,
        is_dtm=True,
    )

    return rotated_dtm, bbox_dtm, temp_cub, temp_txt


def evaluate_dtm_quality(dtm):
    """
    Evaluate whether the selected ROI is usable based on null pixels.
    """
    null_mask = (~np.isfinite(dtm)) | (dtm <= NULL_THRESHOLD)

    bad_pixel_count = int(null_mask.sum())
    total_pixel_count = int(dtm.size)
    bad_fraction = bad_pixel_count / total_pixel_count

    row_null_fraction = null_mask.mean(axis=1)
    col_null_fraction = null_mask.mean(axis=0)

    large_null_region = bool(
        row_null_fraction.max() > 0.5 or col_null_fraction.max() > 0.5
    )

    if REJECT_LARGE_NULL_REGION and large_null_region:
        qc_status = "Reject"
        qc_notes = (
            "Large contiguous null-data region detected in rotated ROI. "
            "Reselect ROI away from DTM coverage boundary."
        )

    elif bad_fraction < GOOD_MAX_BAD_FRACTION:
        qc_status = "Good"
        qc_notes = "Very low bad-pixel fraction."

    elif bad_fraction < QUESTIONABLE_MAX_BAD_FRACTION:
        qc_status = "Questionable"
        qc_notes = "Small bad-pixel fraction. Usable with caution."

    elif bad_fraction < NEEDS_REVIEW_MAX_BAD_FRACTION:
        qc_status = "Needs Review"
        qc_notes = "Moderate bad-pixel fraction. Review before final analysis."

    else:
        qc_status = "Reject"
        qc_notes = "Bad-pixel fraction is too high. Reselect ROI."

    return {
        "bad_pixel_count": bad_pixel_count,
        "total_pixel_count": total_pixel_count,
        "bad_pixel_fraction": bad_fraction,
        "large_null_region": large_null_region,
        "qc_status": qc_status,
        "qc_notes": qc_notes,
    }


def save_qc_preview_images(dtm, roi_id, roi_kind_label):
    """
    Save null-mask and masked raw preview for temporary QC.

    roi_kind_label should be either "rotated ROI" or "axis-aligned ROI"
    so that the saved labels accurately describe the selection geometry.
    """
    null_mask = (~np.isfinite(dtm)) | (dtm <= NULL_THRESHOLD)

    null_mask_path = QC_PREVIEW_DIR / f"{roi_id}_null_mask.png"
    masked_preview_path = QC_PREVIEW_DIR / f"{roi_id}_raw_DTM_nulls_masked.png"

    plt.figure(figsize=(7, 6))
    plt.imshow(null_mask, cmap="gray")
    plt.title(f"{roi_id} {roi_kind_label} null mask")
    plt.xlabel("ROI X pixel")
    plt.ylabel("ROI Y pixel")
    plt.tight_layout()
    plt.savefig(null_mask_path, dpi=300)
    plt.close()

    dtm_plot = dtm.astype(float).copy()
    dtm_plot[null_mask] = np.nan

    plt.figure(figsize=(7, 6))
    plt.imshow(dtm_plot, cmap="gray")
    plt.colorbar(label="Elevation / relative height")
    plt.title(f"{roi_id} {roi_kind_label} raw DTM with nulls masked")
    plt.xlabel("ROI X pixel")
    plt.ylabel("ROI Y pixel")
    plt.tight_layout()
    plt.savefig(masked_preview_path, dpi=300)
    plt.close()

    return null_mask_path, masked_preview_path


# ============================================================
# HELPER FUNCTIONS — CSV
# ============================================================

BASE_FIELDNAMES = [
    "ROI_ID",
    "Body",
    "Site",
    "Product Folder",
    "Product Type",
    "Terrain Type",
    "Source CUB",
    "Source TIFF",
    "X",
    "Y",
    "Width",
    "Height",
    "Pixel Scale m/pixel",
    "Approx Width m",
    "Approx Height m",
    "Selection Method",
    "Display Rotation Degrees",
    "Display Scale",
    "roi_shape",
    "rotation_deg",
    "Rotated Corners Original Image",
    "Bounding Box X",
    "Bounding Box Y",
    "Bounding Box Width",
    "Bounding Box Height",
    "Rotated Width Pixels",
    "Rotated Height Pixels",
    "Rotated DTM NPY",
    "Selection Date",
    "Selected By",
    "Reason for ROI",
    "Quality Flag",
    "Status",
    "Full Preview Image",
    "Crop Preview Image",
    "OpenCV Selected Window Preview",
    "Log File",
    "Notes",
    "Timestamp",
    "Original Image Width",
    "Original Image Height",
    "Bad Pixel Fraction",
    "Large Null Region?",
    "QC Status",
    "QC Notes",
    "QC Null Mask",
    "QC Masked Raw Preview",
]

# ------------------------------------------------------------
# CATALOG REPAIR / BACKFILL SETTINGS
# ------------------------------------------------------------
# The old catalog can contain duplicate headers caused by invisible spaces,
# slightly different spellings, or earlier versions of this selector.
# This script normalizes those headers before writing the next row.
REPAIR_EXISTING_CATALOG_HEADERS = True
BACKFILL_LEGACY_APPROX_DIMENSIONS = True
BACKFILL_LEGACY_SELECTED_BY = True

# Existing records may not contain a reliable Display Scale. Leave this as
# None unless you know every old blank record used the same display scale.
# Example: LEGACY_DISPLAY_SCALE_IF_BLANK = 0.5
LEGACY_DISPLAY_SCALE_IF_BLANK = None


def clean_catalog_value(value):
    """Return a safe CSV cell value without None, NaN, or surrounding spaces."""
    if value is None:
        return ""

    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return ""

    return str(value).strip()


def normalize_header_for_matching(header):
    """Normalize a header only for comparing spelling/spacing variants."""
    text = clean_catalog_value(header).replace("\ufeff", "")
    text = re.sub(r"\s+", " ", text)
    return re.sub(r"[^a-z0-9]+", "", text.casefold())


# Canonicalize both exact field names and common older variants.
HEADER_ALIASES = {
    normalize_header_for_matching(field): field
    for field in BASE_FIELDNAMES
}
HEADER_ALIASES.update({
    normalize_header_for_matching("Approximate Width m"): "Approx Width m",
    normalize_header_for_matching("Approximate Width (m)"): "Approx Width m",
    normalize_header_for_matching("Approx Width (m)"): "Approx Width m",
    normalize_header_for_matching("Approximate Height m"): "Approx Height m",
    normalize_header_for_matching("Approximate Height (m)"): "Approx Height m",
    normalize_header_for_matching("Approx Height (m)"): "Approx Height m",
    normalize_header_for_matching("Selected by"): "Selected By",
    normalize_header_for_matching("Display scale"): "Display Scale",
    # Historical typo in the current catalog header.
    normalize_header_for_matching("Dsplay Scale"): "Display Scale",
    normalize_header_for_matching("Display rotation degrees"): "Display Rotation Degrees",
    # Historical typo in the current catalog header.
    normalize_header_for_matching("QC Masked Raw Review"): "QC Masked Raw Preview",
    normalize_header_for_matching("Pixel scale m per pixel"): "Pixel Scale m/pixel",
    normalize_header_for_matching("Pixel scale (m/pixel)"): "Pixel Scale m/pixel",
    normalize_header_for_matching("Body Name"): "Body",
    normalize_header_for_matching("Planetary Body"): "Body",
})


def canonical_catalog_header(header):
    """Return a consistent header name for CSV writing."""
    cleaned = clean_catalog_value(header).replace("\ufeff", "")
    cleaned = re.sub(r"\s+", " ", cleaned)

    if cleaned == "":
        return ""

    return HEADER_ALIASES.get(
        normalize_header_for_matching(cleaned),
        cleaned,
    )


def set_if_blank(row, field, value):
    """Fill a field only when the current value is blank."""
    if clean_catalog_value(row.get(field)) == "" and clean_catalog_value(value) != "":
        row[field] = clean_catalog_value(value)


def safe_legacy_float(value):
    """Convert a legacy CSV cell to float, returning None for blanks/non-numbers."""
    try:
        cleaned = clean_catalog_value(value)
        return float(cleaned) if cleaned != "" else None
    except (TypeError, ValueError):
        return None


def backfill_legacy_catalog_fields(row):
    """
    Fill only safe missing legacy values.

    Approximate dimensions can be reconstructed from pixel scale and either
    rotated dimensions or ordinary width/height. Selected By is filled only
    from the user-level SELECTED_BY setting. Display Scale is filled only if
    LEGACY_DISPLAY_SCALE_IF_BLANK is explicitly set.
    """
    if BACKFILL_LEGACY_APPROX_DIMENSIONS:
        pixel_scale = safe_legacy_float(row.get("Pixel Scale m/pixel"))
        rotated_width = safe_legacy_float(row.get("Rotated Width Pixels"))
        rotated_height = safe_legacy_float(row.get("Rotated Height Pixels"))
        width = safe_legacy_float(row.get("Width"))
        height = safe_legacy_float(row.get("Height"))

        science_width = rotated_width if rotated_width is not None else width
        science_height = rotated_height if rotated_height is not None else height

        if pixel_scale is not None and science_width is not None:
            set_if_blank(
                row,
                "Approx Width m",
                round(science_width * pixel_scale, 3),
            )

        if pixel_scale is not None and science_height is not None:
            set_if_blank(
                row,
                "Approx Height m",
                round(science_height * pixel_scale, 3),
            )

    if BACKFILL_LEGACY_SELECTED_BY:
        set_if_blank(row, "Selected By", SELECTED_BY)

    if LEGACY_DISPLAY_SCALE_IF_BLANK is not None:
        set_if_blank(
            row,
            "Display Scale",
            LEGACY_DISPLAY_SCALE_IF_BLANK,
        )

    return row


def normalize_catalog_row(raw_row):
    """
    Normalize a row's header names and merge duplicate header columns.

    When duplicate columns exist, the first nonblank value is preserved.
    This fixes old CSVs where, for example, 'Display Scale' and
    'Display Scale ' both existed.
    """
    normalized_row = {}

    for raw_header, raw_value in raw_row.items():
        header = canonical_catalog_header(raw_header)

        if header == "":
            continue

        value = clean_catalog_value(raw_value)

        if header not in normalized_row:
            normalized_row[header] = value
        elif clean_catalog_value(normalized_row[header]) == "" and value != "":
            normalized_row[header] = value

    return normalized_row


def read_catalog_with_normalized_headers():
    """
    Read catalog using csv.reader so duplicate headers can be repaired safely.
    csv.DictReader loses earlier duplicate columns, so it is not used here.
    """
    if not CATALOG_PATH.exists() or CATALOG_PATH.stat().st_size == 0:
        return [], [], False

    with open(CATALOG_PATH, "r", newline="", encoding="utf-8-sig") as f:
        rows = list(csv.reader(f))

    if not rows:
        return [], [], False

    raw_headers = rows[0]
    canonical_headers = [canonical_catalog_header(header) for header in raw_headers]

    # Header needs repair if names changed or canonicalization created duplicates.
    nonblank_headers = [header for header in canonical_headers if header != ""]
    needs_repair = (
        raw_headers != canonical_headers
        or len(nonblank_headers) != len(set(nonblank_headers))
    )

    fieldnames = []
    for header in canonical_headers:
        if header != "" and header not in fieldnames:
            fieldnames.append(header)

    normalized_rows = []
    for values in rows[1:]:
        raw_row = {}

        for index, raw_header in enumerate(raw_headers):
            raw_value = values[index] if index < len(values) else ""
            canonical_header = canonical_catalog_header(raw_header)

            if canonical_header == "":
                continue

            # Preserve the first nonblank value among duplicate columns.
            if canonical_header not in raw_row:
                raw_row[canonical_header] = raw_value
            elif clean_catalog_value(raw_row[canonical_header]) == "":
                raw_row[canonical_header] = raw_value

        normalized_row = normalize_catalog_row(raw_row)
        normalized_row = backfill_legacy_catalog_fields(normalized_row)
        normalized_rows.append(normalized_row)

    return normalized_rows, fieldnames, needs_repair


def validate_new_catalog_row(row):
    """Stop before saving if core fields that should always be present are blank."""
    required_fields = [
        "ROI_ID",
        "Pixel Scale m/pixel",
        "Approx Width m",
        "Approx Height m",
        "Display Scale",
        "Selected By",
    ]

    missing = [
        field for field in required_fields
        if clean_catalog_value(row.get(field)) == ""
    ]

    if missing:
        raise ValueError(
            "Refusing to save ROI because these catalog fields are blank: "
            + ", ".join(missing)
            + ". Check PIXEL_SCALE_M, DISPLAY_SCALE, and SELECTED_BY at the top of the script."
        )


def append_to_roi_catalog(row):
    """
    Append one ROI row while repairing catalog-header inconsistencies.

    This function:
    - normalizes duplicate/space-variant column names;
    - preserves existing nonblank values;
    - backfills safe legacy Approx Width/Height and Selected By values;
    - writes the current row under one canonical copy of every header.
    """
    row = normalize_catalog_row(row)
    validate_new_catalog_row(row)

    existing_rows, existing_fieldnames, header_needed_repair = (
        read_catalog_with_normalized_headers()
    )

    fieldnames = []
    for field in BASE_FIELDNAMES + existing_fieldnames + list(row.keys()):
        canonical_field = canonical_catalog_header(field)
        if canonical_field != "" and canonical_field not in fieldnames:
            fieldnames.append(canonical_field)

    # Ensure every old and new row has every canonical column.
    for existing_row in existing_rows:
        for field in fieldnames:
            existing_row.setdefault(field, "")

    for field in fieldnames:
        row.setdefault(field, "")

    # Save a one-time backup before rewriting a catalog with repaired headers.
    if header_needed_repair and REPAIR_EXISTING_CATALOG_HEADERS:
        backup_dir = CATALOG_PATH.parent / "catalog_backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        backup_path = backup_dir / (
            f"{CATALOG_PATH.stem}_before_header_repair_"
            f"{datetime.now().strftime('%Y%m%d_%H%M%S')}{CATALOG_PATH.suffix}"
        )
        shutil.copy2(CATALOG_PATH, backup_path)
        print(f"Catalog backup saved before header repair: {backup_path}")

    existing_rows.append(row)

    with open(CATALOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(existing_rows)

    return fieldnames


def verify_catalog_row_written(roi_id):
    """Read the rewritten catalog and return the latest row for ROI_ID."""
    rows, _, _ = read_catalog_with_normalized_headers()

    matching_rows = [
        row for row in rows
        if clean_catalog_value(row.get("ROI_ID")) == clean_catalog_value(roi_id)
    ]

    if not matching_rows:
        raise RuntimeError(f"Could not verify saved ROI_ID in catalog: {roi_id}")

    return matching_rows[-1]


# ============================================================
# MAIN SCRIPT
# ============================================================

def main():
    source_tiff_path = Path(SOURCE_TIFF)
    source_cub_path = Path(SOURCE_CUB)

    if not source_tiff_path.exists():
        raise FileNotFoundError(f"Could not find TIFF file: {SOURCE_TIFF}")

    if not source_cub_path.exists():
        raise FileNotFoundError(
            f"Could not find Source CUB file: {SOURCE_CUB}\n"
            "The CUB is required for automatic QC and later frozen analysis."
        )

    if PIXEL_SCALE_M is None:
        raise ValueError(
            "PIXEL_SCALE_M must be set before saving an ROI so approximate "
            "physical dimensions can be recorded."
        )

    img = cv2.imread(str(source_tiff_path), cv2.IMREAD_UNCHANGED)

    if img is None:
        raise FileNotFoundError(f"Could not open image: {SOURCE_TIFF}")

    original_height, original_width = img.shape[:2]
    display = normalize_for_display(img)

    # Rotate only the display used for manual selection.
    display_rotated, matrix_original_to_rotated = rotate_image_bound(
        display,
        SELECTION_ROTATION_DEG,
        border_value=0,
    )

    if DISPLAY_SCALE != 1.0:
        display_for_selection = cv2.resize(
            display_rotated,
            None,
            fx=DISPLAY_SCALE,
            fy=DISPLAY_SCALE,
            interpolation=cv2.INTER_AREA,
        )
    else:
        display_for_selection = display_rotated

    print("\nROI selection")
    print("-------------")
    print(f"ROI_ID: {ROI_ID}")
    print(f"Selection rotation angle: {SELECTION_ROTATION_DEG} degrees")
    print(f"Display scale: {DISPLAY_SCALE}")
    print("\nSelect ROI on the display, then press ENTER or SPACE.")
    print("Press C to cancel.\n")

    roi = cv2.selectROI(
        f"Select ROI: {ROI_ID}",
        display_for_selection,
        showCrosshair=True,
    )

    cv2.destroyAllWindows()

    x_display, y_display, w_display, h_display = roi

    if w_display == 0 or h_display == 0:
        print("No ROI selected. Exiting.")
        return

    corners_original, selected_width, selected_height = (
        selected_rectangle_to_original_geometry(
            x_display,
            y_display,
            w_display,
            h_display,
            DISPLAY_SCALE,
            matrix_original_to_rotated,
        )
    )

    bbox_x, bbox_y, bbox_w, bbox_h = bounding_box_from_corners(
        corners_original,
        original_width,
        original_height,
    )

    # A real rotated ROI exists only when the display was rotated by more than
    # the configured floating-point tolerance.
    is_rotated_roi = (
        abs(SELECTION_ROTATION_DEG) > ROTATION_ZERO_TOLERANCE_DEG
    )

    roi_shape = (
        "rotated_rect"
        if is_rotated_roi
        else "axis_aligned_rect"
    )
    rotation_deg = float(SELECTION_ROTATION_DEG)

    # For an axis-aligned ROI, the selected rectangle and the CUB crop are
    # the same thing. For a rotated ROI, the selected_width/height describe
    # the true science strip, while bbox_* describe only the temporary CUB crop.
    science_width_pixels = selected_width if is_rotated_roi else bbox_w
    science_height_pixels = selected_height if is_rotated_roi else bbox_h

    approx_width_m = round(
        float(science_width_pixels) * float(PIXEL_SCALE_M),
        3,
    )
    approx_height_m = round(
        float(science_height_pixels) * float(PIXEL_SCALE_M),
        3,
    )

    print("\nOriginal-image ROI corners:")
    for i, point in enumerate(corners_original):
        print(f"  Corner {i + 1}: x={point[0]:.2f}, y={point[1]:.2f}")

    print("\nAxis-aligned CUB crop geometry:")
    print(f"  X={bbox_x}, Y={bbox_y}, Width={bbox_w}, Height={bbox_h}")

    if is_rotated_roi:
        print("\nTrue rotated ROI size:")
        print(f"  Rotated Width={selected_width} px")
        print(f"  Rotated Height={selected_height} px")
    else:
        print("\nAxis-aligned ROI size:")
        print(f"  Width={bbox_w} px")
        print(f"  Height={bbox_h} px")

    # ------------------------------------------------------------
    # Rotated DTM extraction only for a genuinely rotated selection
    # ------------------------------------------------------------
    rotated_dtm_path = ""
    qc_dtm = None
    qc_source_note = ""

    if is_rotated_roi:
        rotated_dtm_path = (
            ROTATED_DTM_DIR / f"{ROI_ID}_rotated_raw_DTM.npy"
        )

        print("\nExtracting true rotated DTM from CUB...")

        rotated_dtm, _, _, _ = load_rotated_dtm_from_selected_roi(
            SOURCE_CUB,
            ROI_ID,
            bbox_x,
            bbox_y,
            bbox_w,
            bbox_h,
            corners_original,
            selected_width,
            selected_height,
        )

        np.save(rotated_dtm_path, rotated_dtm)
        qc_dtm = rotated_dtm
        qc_source_note = "true rotated DTM extracted from CUB"
        print(f"Rotated DTM saved to: {rotated_dtm_path}")

    else:
        # No perspective warp, no rotated .npy, and no rotated-only metadata.
        # The frozen-analysis script will later use X/Y/Width/Height to crop
        # the CUB normally for this axis-aligned ROI.
        qc_source_note = "axis-aligned temporary CUB crop (not saved as rotated DTM)"
        print("\nAxis-aligned selection detected.")
        print("No rotated DTM .npy will be created.")
        print("Rotated-only catalog fields will be left blank.")

    # ------------------------------------------------------------
    # Automatic QC
    # ------------------------------------------------------------
    qc_result = {
        "bad_pixel_fraction": "",
        "large_null_region": "",
        "qc_status": "Pending",
        "qc_notes": "Automatic QC not run.",
        "qc_null_mask_path": "",
        "qc_masked_preview_path": "",
    }

    if AUTO_QC_BEFORE_SAVE:
        print("\nRunning automatic QC...")

        try:
            # A rotated ROI already has its true science DTM. An axis-aligned
            # ROI uses the ordinary CUB crop only for QC and does not save NPY.
            if qc_dtm is None:
                qc_dtm, _, _ = load_bbox_dtm_from_cub(
                    SOURCE_CUB,
                    ROI_ID,
                    bbox_x,
                    bbox_y,
                    bbox_w,
                    bbox_h,
                )

            qc_eval = evaluate_dtm_quality(qc_dtm)

            qc_label = (
                "rotated ROI"
                if is_rotated_roi
                else "axis-aligned ROI"
            )

            null_mask_path, masked_preview_path = save_qc_preview_images(
                qc_dtm,
                ROI_ID,
                qc_label,
            )

            qc_result = {
                "bad_pixel_fraction": qc_eval["bad_pixel_fraction"],
                "large_null_region": (
                    "Yes"
                    if qc_eval["large_null_region"]
                    else "No"
                ),
                "qc_status": qc_eval["qc_status"],
                "qc_notes": qc_eval["qc_notes"],
                "qc_null_mask_path": str(null_mask_path),
                "qc_masked_preview_path": str(masked_preview_path),
            }

            print("\nQC result:")
            print(f"QC source: {qc_source_note}")
            print(f"Bad pixel fraction: {qc_eval['bad_pixel_fraction']:.4f}")
            print(f"Large null region: {qc_eval['large_null_region']}")
            print(f"QC Status: {qc_eval['qc_status']}")
            print(f"QC Notes: {qc_eval['qc_notes']}")
            print(f"Null mask preview: {null_mask_path}")
            print(f"Masked raw DTM preview: {masked_preview_path}")

            if qc_eval["qc_status"] == "Reject":
                print("\nThis ROI was rejected by automatic QC.")
                print("It will NOT be saved to roi_catalog.csv.")
                print("Please rerun the script and select a cleaner region.")
                return

        except Exception as e:
            print("\nWARNING: Automatic QC failed.")
            print(f"Reason: {e}")
            print("The ROI can still be saved, but QC Status will be Pending.")

    # ------------------------------------------------------------
    # Save visual previews
    # ------------------------------------------------------------
    # The full preview is always saved:
    #   red polygon = true selected science ROI
    #   blue rectangle = temporary axis-aligned CUB crop
    #
    # For a non-rotated ROI, the red and blue outlines coincide because the
    # selected ROI and the temporary CUB crop are exactly the same rectangle.
    full_preview = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

    corners_int = np.round(corners_original).astype(np.int32)
    cv2.polylines(
        full_preview,
        [corners_int],
        isClosed=True,
        color=(0, 0, 255),  # BGR = red
        thickness=3,
    )

    cv2.rectangle(
        full_preview,
        (bbox_x, bbox_y),
        (bbox_x + bbox_w - 1, bbox_y + bbox_h - 1),
        (255, 0, 0),  # BGR = blue
        thickness=2,
    )

    full_preview_path = (
        PREVIEW_DIR / f"{ROI_ID}_full_preview.png"
    )
    crop_preview_path = ""
    selected_window_preview_path = ""
    log_path = LOG_DIR / f"{ROI_ID}.json"

    cv2.imwrite(str(full_preview_path), full_preview)

    # Save the straightened crop preview only when a real rotation was used.
    if (
        is_rotated_roi
        or not SAVE_ROTATED_CROP_PREVIEW_ONLY_WHEN_ROTATED
    ):
        display_bbox = display[
            bbox_y:bbox_y + bbox_h,
            bbox_x:bbox_x + bbox_w,
        ].copy()

        if display_bbox.size == 0:
            raise ValueError(
                "Could not create the TIFF bounding-box crop for "
                "the rotated ROI preview."
            )

        crop_preview = warp_array_to_rotated_roi(
            display_bbox,
            corners_original,
            bbox_x,
            bbox_y,
            selected_width,
            selected_height,
            is_dtm=False,
        )

        # Save explicitly as neutral BGR grayscale to avoid false viewer tint.
        if crop_preview.ndim == 2:
            crop_preview_to_save = cv2.cvtColor(
                crop_preview,
                cv2.COLOR_GRAY2BGR,
            )
        else:
            crop_preview_to_save = crop_preview

        crop_preview_path = (
            PREVIEW_DIR / f"{ROI_ID}_rotated_crop_preview.png"
        )
        cv2.imwrite(str(crop_preview_path), crop_preview_to_save)

    # Optional debug-only preview. It has no analysis role.
    if SAVE_OPENCV_SELECTED_WINDOW_PREVIEW:
        selected_window_preview = display_for_selection[
            y_display:y_display + h_display,
            x_display:x_display + w_display,
        ].copy()

        if selected_window_preview.size == 0:
            raise ValueError(
                "Could not create the selected-window validation preview."
            )

        if selected_window_preview.ndim == 2:
            selected_window_preview = cv2.cvtColor(
                selected_window_preview,
                cv2.COLOR_GRAY2BGR,
            )

        selected_window_preview_path = PREVIEW_DIR / (
            f"{ROI_ID}_opencv_selected_window_preview.png"
        )
        cv2.imwrite(
            str(selected_window_preview_path),
            selected_window_preview,
        )

    selection_date = str(date.today())
    timestamp = datetime.now().isoformat(timespec="seconds")

    # These fields are deliberately blank for axis-aligned ROIs. This makes
    # the catalog unambiguous: a populated Rotated DTM NPY always means a
    # real rotated extraction exists.
    rotated_corners_value = (
        corners_to_json(corners_original)
        if is_rotated_roi
        else ""
    )
    rotated_bbox_x_value = bbox_x if is_rotated_roi else ""
    rotated_bbox_y_value = bbox_y if is_rotated_roi else ""
    rotated_bbox_w_value = bbox_w if is_rotated_roi else ""
    rotated_bbox_h_value = bbox_h if is_rotated_roi else ""
    rotated_width_value = selected_width if is_rotated_roi else ""
    rotated_height_value = selected_height if is_rotated_roi else ""
    rotated_dtm_npy_value = (
        str(rotated_dtm_path)
        if is_rotated_roi
        else ""
    )

    selection_method = (
        "Manual OpenCV rotated ROI selection; bounding CUB crop plus "
        "perspective-warped true rotated DTM extraction"
        if is_rotated_roi
        else "Manual OpenCV axis-aligned ROI selection; "
        "ordinary CUB crop used by frozen analysis, no rotated DTM extraction"
    )

    notes = (
        "For rotated_rect ROIs, X/Y/Width/Height are the axis-aligned "
        "temporary CUB crop. Use Rotated DTM NPY for the true rotated "
        "science extraction. The full preview shows red science ROI and "
        "blue temporary CUB bounding box."
        if is_rotated_roi
        else "Axis-aligned ROI. X/Y/Width/Height define the science ROI "
        "directly. Rotated-only fields and Rotated DTM NPY are intentionally "
        "blank because no perspective-warped extraction was created."
    )

    roi_info = {
        "ROI_ID": ROI_ID,
        "Body": BODY_NAME,
        "Site": SITE,
        "Product Folder": PRODUCT_FOLDER,
        "Product Type": PRODUCT_TYPE,
        "Terrain Type": TERRAIN_TYPE,
        "Source CUB": SOURCE_CUB,
        "Source TIFF": SOURCE_TIFF,

        # These are always the direct science rectangle for axis-aligned
        # selections and the temporary CUB bounding box for rotated selections.
        "X": bbox_x,
        "Y": bbox_y,
        "Width": bbox_w,
        "Height": bbox_h,

        "Pixel Scale m/pixel": PIXEL_SCALE_M,
        "Approx Width m": approx_width_m,
        "Approx Height m": approx_height_m,

        "Selection Method": selection_method,
        "Display Rotation Degrees": rotation_deg,
        "Display Scale": DISPLAY_SCALE,

        "roi_shape": roi_shape,
        "rotation_deg": rotation_deg,
        "Rotated Corners Original Image": rotated_corners_value,
        "Bounding Box X": rotated_bbox_x_value,
        "Bounding Box Y": rotated_bbox_y_value,
        "Bounding Box Width": rotated_bbox_w_value,
        "Bounding Box Height": rotated_bbox_h_value,
        "Rotated Width Pixels": rotated_width_value,
        "Rotated Height Pixels": rotated_height_value,
        "Rotated DTM NPY": rotated_dtm_npy_value,

        "Selection Date": selection_date,
        "Selected By": SELECTED_BY,
        "Reason for ROI": REASON_FOR_ROI,
        "Quality Flag": QUALITY_FLAG,
        "Status": STATUS,
        "Full Preview Image": str(full_preview_path),
        "Crop Preview Image": str(crop_preview_path),
        "OpenCV Selected Window Preview": str(selected_window_preview_path),
        "Log File": str(log_path),
        "Notes": notes,
        "Timestamp": timestamp,
        "Original Image Width": original_width,
        "Original Image Height": original_height,
        "Bad Pixel Fraction": qc_result["bad_pixel_fraction"],
        "Large Null Region?": qc_result["large_null_region"],
        "QC Status": qc_result["qc_status"],
        "QC Notes": qc_result["qc_notes"],
        "QC Null Mask": qc_result["qc_null_mask_path"],
        "QC Masked Raw Preview": qc_result["qc_masked_preview_path"],
    }

    # Save JSON log and append a canonicalized CSV row.
    with open(log_path, "w") as f:
        json.dump(roi_info, f, indent=4)

    append_to_roi_catalog(roi_info)
    saved_catalog_row = verify_catalog_row_written(ROI_ID)

    print("\nCatalog field verification")
    print("--------------------------")
    print(f"Approx Width m: {saved_catalog_row.get('Approx Width m', '')}")
    print(f"Approx Height m: {saved_catalog_row.get('Approx Height m', '')}")
    print(f"Selected By: {saved_catalog_row.get('Selected By', '')}")
    print(f"Display Scale: {saved_catalog_row.get('Display Scale', '')}")
    print(f"ROI shape: {saved_catalog_row.get('roi_shape', '')}")
    print(
        "Rotated DTM NPY: "
        f"{saved_catalog_row.get('Rotated DTM NPY', '')}"
    )

    print("\nROI selected and saved.")
    print(f"Dataset/body: {ACTIVE_DATASET} / {BODY_NAME}")
    print(f"ROI_ID: {ROI_ID}")
    print(f"roi_shape: {roi_shape}")
    print(f"rotation_deg: {rotation_deg}")
    print(f"Source TIFF: {SOURCE_TIFF}")
    print(f"Source CUB: {SOURCE_CUB}")
    print(f"X: {bbox_x}")
    print(f"Y: {bbox_y}")
    print(f"Width: {bbox_w}")
    print(f"Height: {bbox_h}")
    print(f"Approx Width m: {approx_width_m}")
    print(f"Approx Height m: {approx_height_m}")
    print(f"Bad Pixel Fraction: {qc_result['bad_pixel_fraction']}")
    print(f"Large Null Region?: {qc_result['large_null_region']}")
    print(f"QC Status: {qc_result['qc_status']}")
    print(f"Full preview saved to: {full_preview_path}")

    if is_rotated_roi:
        print(f"Rotated Width Pixels: {selected_width}")
        print(f"Rotated Height Pixels: {selected_height}")
        print(f"Rotated DTM NPY: {rotated_dtm_path}")
        if crop_preview_path:
            print(f"Rotated crop preview saved to: {crop_preview_path}")
    else:
        print(
            "No rotated DTM or rotated crop preview was saved because "
            "SELECTION_ROTATION_DEG is 0."
        )

    if selected_window_preview_path:
        print(
            "OpenCV selected-window preview saved to: "
            f"{selected_window_preview_path}"
        )

    print(f"Log saved to: {log_path}")
    print(f"Catalog updated: {CATALOG_PATH}")


if __name__ == "__main__":
    main()

# Icy Worlds Surface Roughness

A reproducible Python/USGS ISIS workflow for scale-dependent surface-roughness analysis of **Europa, Ganymede, Enceladus, and other icy bodies**.

This repository contains the analysis pipeline developed for the project:

**Europa, Enceladus, and Ganymede: Comparative Topography of Active Icy Worlds**

The workflow extracts regions of interest (ROIs) from digital terrain models (DTMs), evaluates data quality, cleans and detrends the topography, computes deterministic RMS-deviation roughness curves, fits Hurst exponents, identifies candidate roughness-transition breakpoints, calculates power spectral density (PSD), and generates reproducible summary tables and figures.

---

## Workflow Overview

```text
ISIS CUB DTM
     ↓
TIFF preview
     ↓
ROI selection + metadata
     ↓
ROI catalog
     ↓
Raw DTM extraction
     ↓
Quality control / missing-pixel treatment
     ↓
Plane detrending
     ↓
RMS-deviation structure function
     ↓
Hurst exponent + candidate breakpoint
     ↓
PSD analysis
     ↓
Per-ROI results
     ↓
Body-level summary CSVs
     ↓
Combined Excel workbook
     ↓
Comparative plots
```

For reproducing an **existing ROI**, the manual ROI-selection step is not required. The saved ROI catalog contains the geometry needed to reproduce the analysis.

---

# Repository Structure

The repository layout is:

```text
icy-worlds-surface-roughness/
│
├── README.md
├── environment.yml
├── pyproject.toml
├── .gitignore
│
├── config/
│   └── icy_worlds_config.json
│
├── catalogs/
│   ├── europa_roi_catalog.csv
│   ├── ganymede_roi_catalog.csv
│   └── enceladus_roi_catalog.csv
│
├── scripts/
│   ├── create_tiff_from_cub.py
│   ├── select_roi_for_catalog_multibody_clean.py
│   ├── run_roi_analysis_multibody.py
│   ├── roi_processing.py
│   ├── icy_workflow_common.py
│   ├── summarize_roi_tables_simple.py
│   │
│   ├── plotting/
│       └── plot_hurst_baseline_panels.py
│
├── data/
│   ├── README.md
│   ├── data_manifest.csv
│   └── raw/
│       ├── europa/
│       ├── ganymede/
│       └── enceladus/
│
├── results/
│   ├── europa/
│   ├── ganymede/
│   ├── enceladus/
│   └── roi_grouped_summary.xlsx
│
└── figures/
```

---

# Installation

## 1. Clone the repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd icy-worlds-surface-roughness
```

---

## 2. Install USGS ISIS

This project relies on command-line programs from the **USGS Integrated Software for Imagers and Spectrometers (ISIS)**.

The workflow requires ISIS programs including:

```text
catlab
isis2std
crop
isis2ascii
```

ISIS should be installed and activated before running analyses that use `.cub` files.

If you already have a functioning ISIS Conda environment, it may be preferable to use that environment and install the additional Python dependencies into it rather than replacing it.

An `environment.yml` file is included to document the Python environment used for this project.

If creating a new environment:

```bash
conda env create -f environment.yml
```

Then activate the environment specified in the YAML file.

If you already have an ISIS environment, activate it instead and install the additional project dependencies as needed.

---

## 3. Install the Python project

From the repository root:

```bash
pip install -e .
```

This installs the project in editable mode.

---

## 4. Verify the environment

Check that ISIS is available:

```bash
which python
which isis2std
which crop
which isis2ascii
which catlab
```

Check the main Python dependencies:

```bash
python -c "import numpy, scipy, pandas, matplotlib, cv2; print('Python packages OK')"
```

---

# Configuration

The multi-body workflow uses:

```text
config/icy_worlds_config.json
```

Example:

```json
{
  "datasets": {
    "europa": {
      "body_name": "Europa",
      "catalog_path": "catalogs/europa_roi_catalog.csv",
      "results_root": "results/europa"
    },
    "ganymede": {
      "body_name": "Ganymede",
      "catalog_path": "catalogs/ganymede_roi_catalog.csv",
      "results_root": "results/ganymede"
    },
    "enceladus": {
      "body_name": "Enceladus",
      "catalog_path": "catalogs/enceladus_roi_catalog.csv",
      "results_root": "results/enceladus"
    }
  }
}
```

Paths should be stored relative to the repository root whenever possible so the workflow can run on different computers.

---

# Data

The original planetary DTM products may be too large to store directly in a normal GitHub repository.

Raw data should therefore be placed under:

```text
data/raw/
├── europa/
├── ganymede/
└── enceladus/
```

The repository should include a:

```text
data/data_manifest.csv
```

describing the required source products, including:

* body
* site
* product type
* filename
* pixel scale
* original source
* which analysis uses the file

The ROI catalogs should refer to repository-relative data locations rather than machine-specific paths.

For example:

```text
data/raw/europa/conamara/example.cub
```

rather than:

```text
/Users/username/.../example.cub
```

---

# Reproducing Existing Results

For ROIs that have already been selected and stored in the catalogs, **you do not need to manually select the ROI again**.

The basic reproduction workflow is:

```text
Obtain source DTM
      ↓
Use committed ROI catalog
      ↓
Run ROI analysis
      ↓
Generate summary tables
      ↓
Generate final figures
```

---

# 1. Create a TIFF Preview from a CUB

This step is primarily needed when inspecting a DTM or selecting a **new ROI**.

Run:

```bash
python scripts/create_tiff_from_cub.py
```

The script uses ISIS tools equivalent to:

```bash
catlab from=<SOURCE_CUB>
isis2std from=<SOURCE_CUB> to=<OUTPUT_TIFF> format=tiff
```

The TIFF is used only for visualization and ROI selection.

The scientific analysis uses either:

* the original ISIS `.cub` file for an axis-aligned ROI, or
* the saved rotated DTM `.npy` array for a rotated ROI.

---

# 2. Select a New ROI

This step is required only when adding a new ROI.

The ROI selector is:

```text
scripts/select_roi_for_catalog_multibody_clean.py
```

Set the dataset and ROI metadata near the top of the script.

Example:

```python
ACTIVE_DATASET = "europa"

ROI_ID = "TEST_CHAOS_001"
SITE = "conamara(12hr)"
TERRAIN_TYPE = "Chaos"

PRODUCT_FOLDER = "sfs_data"
PRODUCT_TYPE = "Combined(Stereo + SFS[ZT])"

SOURCE_CUB = "data/raw/europa/conamara/example.cub"
SOURCE_TIFF = "data/raw/europa/conamara/example.tif"

PIXEL_SCALE_M = 9

SELECTION_ROTATION_DEG = 0
DISPLAY_SCALE = 0.5
```

Run:

```bash
python scripts/select_roi_for_catalog_multibody_clean.py
```

In the OpenCV selection window:

```text
Draw ROI rectangle
ENTER or SPACE = accept
C = cancel
```

The selector records the ROI geometry and metadata in the appropriate body catalog.

---

## Axis-Aligned ROIs

For a normal rectangular ROI:

```python
SELECTION_ROTATION_DEG = 0
```

The catalog records:

```text
X
Y
Width
Height
```

During analysis, these coordinates are used to reproduce the same region from the original CUB.

The selector stores image coordinates using zero-based indexing. The analysis workflow converts these coordinates to the one-based `SAMPLE` and `LINE` convention required by ISIS `crop`.

---

## Rotated ROIs

For a rotated ROI:

```python
SELECTION_ROTATION_DEG = -25
```

The selector creates a rotated science array such as:

```text
results/<body>/selection/rotated_dtms/<ROI_ID>_rotated_raw_DTM.npy
```

This `.npy` file is the science input used when reproducing that rotated ROI.

The catalog also records the rotated ROI geometry and dimensions.

Existing rotated `.npy` products should therefore be preserved when exact reproduction of those ROIs is required.

---

# 3. Run ROI Analysis

The main analysis entry point is:

```text
scripts/run_roi_analysis_multibody.py
```

It uses:

```text
scripts/roi_processing.py
```

for:

* catalog parsing
* ROI geometry handling
* ISIS CUB extraction
* DTM loading
* QC
* missing-pixel treatment
* detrending
* PSD calculations

and:

```text
scripts/icy_workflow_common.py
```

for:

* deterministic RMS-deviation calculations
* Hurst fitting
* candidate breakpoint fitting
* configuration utilities
* summary-table utilities

---

## List Available Datasets

```bash
python scripts/run_roi_analysis_multibody.py --list-datasets
```

Example output:

```text
europa: Europa
ganymede: Ganymede
enceladus: Enceladus
```

---

## Run One ROI

```bash
python scripts/run_roi_analysis_multibody.py \
    --dataset europa \
    --roi TEST_CHAOS_001
```

---

## Run Multiple ROIs

Repeat the `--roi` flag:

```bash
python scripts/run_roi_analysis_multibody.py \
    --dataset europa \
    --roi ROI_001 \
    --roi ROI_002 \
    --roi ROI_003
```

---

## Run Every ROI in One Catalog

```bash
python scripts/run_roi_analysis_multibody.py \
    --dataset europa \
    --all
```

The same command can be used for another body:

```bash
python scripts/run_roi_analysis_multibody.py \
    --dataset ganymede \
    --all
```

---

# 4. Analysis Pipeline

For each ROI, the main script performs:

```text
1. Read ROI metadata
2. Interpret ROI geometry
3. Load raw DTM
4. Save raw DTM diagnostics
5. Evaluate missing/null pixels
6. Apply documented cleaning/interpolation
7. Remove a least-squares best-fit plane
8. Calculate PSD
9. Calculate deterministic RMS-deviation curve
10. Select the baseline fitting range
11. Fit an overall Hurst exponent
12. Search for a candidate segmented breakpoint
13. Save figures and numerical outputs
14. Save metadata and provenance
15. Update the body-level summary CSV
```

---

# 5. Quality Control and Missing Data

Null or invalid DTM values are identified using:

```text
NULL_VALUE = -9999
NULL_THRESHOLD = -9000
```

The workflow records:

* bad-pixel count
* bad-pixel fraction
* large null regions
* QC status
* cleaning method
* interpolation fraction
* interpolation-distance diagnostics

For small isolated missing regions, local median-filter estimates may be used.

For ROIs requiring broader interpolation, the workflow can use:

```text
linear interpolation
        ↓
nearest-neighbor fallback
```

All interpolated pixels can be saved in an explicit interpolation mask.

Interpolation may suppress measured topographic roughness and affect PSD or breakpoint behavior, so ROIs requiring substantial interpolation should be interpreted cautiously.

---

# 6. Plane Detrending

Before roughness and PSD analysis, the cleaned DTM is detrended by fitting and subtracting a least-squares plane:

```text
z(x,y) = ax + by + c
```

The workflow saves both:

```text
best_fit_plane.npy
detrended_DTM.npy
```

so the transformation can be inspected and reproduced.

---

# 7. RMS-Deviation Roughness Analysis

The current workflow uses a deterministic RMS-deviation structure function:

```text
y(Δx) = sqrt(mean([h(x_i) - h(x_i + Δx)]²))
```

where:

* `Δx` is horizontal baseline length
* `h` is detrended elevation
* `y(Δx)` is RMS elevation difference at that scale

The analysis uses deterministic pair aggregation rather than the older random-pair sampling workflow.

Baseline bins are approximately one DTM pixel wide in physical distance.

---

# 8. Hurst Exponent

The Hurst exponent is obtained from the slope of:

```text
log10(RMS deviation)
```

versus:

```text
log10(baseline)
```

so that:

```text
RMS deviation ∝ baseline^H
```

The workflow reports:

```text
H Overall
```

for a single fit over the accepted baseline range.

---

# 9. Candidate Breakpoint Analysis

The RMS-deviation curve is also tested for a possible two-segment fit.

When a breakpoint is accepted, the workflow reports:

```text
Breakpoint m
H Before BP
H After BP
```

where:

```text
H Before BP
```

represents the shorter-baseline roughness behavior and:

```text
H After BP
```

represents the longer-baseline behavior.

The difference can be written as:

```text
Delta H = H Before BP - H After BP
```

A breakpoint is treated only as a **candidate scale-dependent roughness transition**.

It is not automatically interpreted as:

* ridge spacing
* fracture spacing
* radar wavelength
* ice-shell thickness
* another specific physical length scale

without additional independent evidence.

---

# 10. PSD Analysis

The same detrended DTM is also analyzed in the Fourier domain.

The workflow computes:

```text
2D FFT power spectrum
        ↓
radial averaging
        ↓
1D PSD
```

The PSD uses **spatial frequency**, with units of approximately:

```text
1/m
```

It should not be confused with transmitted radar frequency.

PSD is a complementary diagnostic and is separate from the Hurst-fitting procedure.

---

# 11. Per-ROI Outputs

Each ROI is stored under:

```text
results/<body>/reproducible_rois/<ROI_ID>/
```

A typical output folder contains:

```text
data/
├── raw_DTM.npy
├── cleaned_DTM.npy
├── detrended_DTM.npy
├── best_fit_plane.npy
├── psd.csv
├── steinbruegge_rms_deviation_curve.csv
└── steinbruegge_rms_deviation_fit_bins.csv

figures/
├── raw_DTM_preview.png
├── raw_DTM_preview_nulls_masked.png
├── null_mask.png
├── cleaned_DTM_preview.png
├── detrended_DTM_preview.png
├── psd_plot.png
├── psd_diagnostic_multipanel.png
├── steinbruegge_rms_deviation_breakpoint_plot.png
└── steinbruegge_rms_deviation_linear_bins.png

logs/
├── roi_metadata.json
├── analysis_geometry.json
├── quality_control_from_analysis.json
├── steinbruegge_rms_deviation_summary.json
└── analysis_log.txt
```

These files preserve the major transformations between the raw terrain data and the final roughness measurements.

---

# 12. Body-Level Summary Tables

After each ROI finishes, the workflow updates:

```text
results/<body>/summaries/tables/roi_analysis_summary.csv
```

The table includes values such as:

```text
ROI_ID
Body
Site
Terrain Type
Product Type
Pixel Scale m/pixel
QC Status
Bad Pixel Fraction
H Overall
Breakpoint Accepted
Breakpoint m
H Before BP
H After BP
```

---

# 13. Create the Combined Summary Workbook

After the body-level analyses are complete:

```bash
python scripts/summarize_roi_tables_simple.py \
    --inputs \
    results/europa/summaries/tables/roi_analysis_summary.csv \
    results/ganymede/summaries/tables/roi_analysis_summary.csv \
    results/enceladus/summaries/tables/roi_analysis_summary.csv \
    --output results/roi_grouped_summary.xlsx
```

Useful workbook sheets include:

```text
01_BP_by_body_terrain_product
02_Good_by_body_terrain
03_BP_by_body_product
04_QC_counts
05_Site_QC
06_Flagged_ROIs
07_Cleaned_Key_Rows
```

Additional sheets may contain body-, terrain-, product-, or site-specific subsets.

---

# 14. Generate Hurst-vs-Baseline Figures

Example:

```bash
python scripts/plotting/plot__hurst_baseline_panels.py \
    --excel results/roi_grouped_summary.xlsx \
    --sheets \
    Ganymede_Product_Good \
    Europa_Product_Good \
    Enceladus_Product_Good \
    --output-dir figures/figure_individual_clean \
    --split-by-body \
    --plot-mode individual \ or aggregate \
    --color-by Site \
    --legend-position bottom \
    --legend-ncols 4 \
    --max-legend-label-chars 16 \
    --x-min 1 \
    --x-max 10000 \
    --verbose
```

For accepted-breakpoint ROIs, the plot uses:

```text
Short-scale point:
x = Pixel Scale m/pixel
y = H Before BP
```

and:

```text
Long-scale point:
x = Breakpoint m
y = H After BP
```

These plots visualize changes in roughness behavior across spatial scales.

---

# Interpretation Notes

### H Overall

One Hurst slope across the complete accepted fitting range.

### H Before BP

The fitted slope on the shorter-baseline side of an accepted candidate breakpoint.

### H After BP

The fitted slope on the longer-baseline side of an accepted candidate breakpoint.

### Delta H

```text
Delta H = H Before BP - H After BP
```

### Candidate Breakpoint

The breakpoint represents a possible change in the scale dependence of surface roughness.

It should not automatically be interpreted as a specific geological or geophysical length scale.

### PSD

PSD measures power as a function of **spatial frequency**.

It does not use radar transmission frequency.

### Radar Interpretation

These measurements provide terrain-specific topographic roughness context that may be relevant to radar surface-clutter studies.

The workflow does **not** directly simulate radar returns or REASON instrument observations.

---

# Important Comparison Considerations

Comparisons across icy worlds should be made carefully.

Before directly comparing Hurst exponents or candidate breakpoints, consider differences in:

* source mission
* DTM product type
* stereo versus shape-from-shading processing
* map projection
* pixel scale
* ROI dimensions
* detrending
* interpolation fraction
* terrain class
* fitted baseline range

A difference between two ROIs is not necessarily a geological difference if their source products or effective resolutions differ substantially.

---

# Quick Reproduction Example

Assuming the required source DTMs have already been placed in the locations documented by the repository:

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd icy-worlds-surface-roughness
```

Activate the project/ISIS environment:

```bash
conda activate <YOUR-ISIS-ENVIRONMENT>
```

Verify the tools:

```bash
which crop
which isis2ascii
which isis2std
which catlab
```

Run one existing Europa ROI:

```bash
python scripts/run_roi_analysis_multibody.py \
    --dataset europa \
    --roi CONA12hr_CHAOS_008
```

Run all Europa ROIs:

```bash
python scripts/run_roi_analysis_multibody.py \
    --dataset europa \
    --all
```

Build the combined summary workbook:

```bash
python scripts/summarize_roi_tables_simple.py \
    --inputs \
    results/europa/summaries/tables/roi_analysis_summary.csv \
    results/ganymede/summaries/tables/roi_analysis_summary.csv \
    results/enceladus/summaries/tables/roi_analysis_summary.csv \
    --output results/roi_grouped_summary.xlsx
```

Generate comparative figures:

```bash
python scripts/plotting/plot_baseline_panels.py \
    --excel results/roi_grouped_summary.xlsx \
    --sheets \
    Ganymede_Product_Good \
    Europa_Product_Good \
    Enceladus_Product_Good \
    --output-dir figures/figure_individual_clean \
    --split-by-body \
    --plot-mode individual \
    --color-by Site \
    --legend-position bottom \
    --legend-ncols 4 \
    --max-legend-label-chars 16 \
    --x-min 1 \
    --x-max 10000 \
    --verbose
```

---


The canonical analysis entry point is:

```text
scripts/run_roi_analysis_multibody.py
```

---

# Reproducibility

A successful reproduction should allow a new user to:

1. obtain the required source DTM;
2. clone this repository;
3. install or activate ISIS;
4. install the Python dependencies;
5. reproduce an existing ROI using the committed catalog;
6. regenerate the per-ROI numerical and graphical outputs;
7. regenerate body-level summary CSVs;
8. regenerate the grouped workbook; and
9. regenerate the comparative figures.

For exact reproduction, source data versions, pixel scales, ROI catalog entries, rotated ROI arrays, configuration files, and science settings should remain unchanged.

---

# Project Context

This workflow was developed as part of a comparative study of the surface topography and scale-dependent roughness of active icy worlds.

The analysis focuses on terrain-scale roughness derived from digital terrain models of:

* Europa
* Ganymede
* Enceladus

and is designed so that additional icy bodies or datasets can be added through the same configuration-based workflow.

---

# Acknowledgments

This project was developed during a summer research internship at the Lunar and Planetary Institute.

Scientific interpretation and use of planetary datasets should follow the citation and acknowledgment requirements of the original data providers and processing teams.



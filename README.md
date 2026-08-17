# icy-worlds-surface-roughness
Reproducible Python/ISIS workflow for scale-dependent surface roughness analysis of Europa, Ganymede, and Enceladus.
# Europa, Enceladus, and Ganymede Roughness Workflow

This repository contains the workflow for:

**Europa, Enceladus, and Ganymede: Comparative Topography of Active Icy Worlds**

The workflow selects regions of interest from icy-world digital terrain models, evaluates ROI quality, computes RMS-deviation roughness curves, fits Hurst exponents, detects candidate breakpoints, saves PSD diagnostics, and generates summary tables and plots.

---

## Workflow overview

```text
CUB file → TIFF preview → ROI catalog → reproducible ROI analysis → summary CSV → Excel summary → plots
```

Main scripts:

```text
scripts/create_tiff_from_cub.py
scripts/select_roi_for_catalog_multibody_clean.py
scripts/run_roi_analysis_multibody_clean.py
scripts/icy_workflow_common.py
scripts/summarize_roi_tables_simple.py
scripts/plotting/plot_figure12_hurst_baseline_panels.py
```

---

## Setup

```bash
cd /Users/mnebart/icy/github/europa-surface-analysis
conda activate isis9.0.0
```

Check required tools:

```bash
which python
which isis2std
which crop
which isis2ascii
which catlab
python -c "import numpy, scipy, pandas, matplotlib, cv2; print('Python packages OK')"
```

---

## Config file

The multi-body workflow uses:

```text
config/icy_worlds_config.json
```

Example:

```json
{
  "frozen_workflow_module": "run_frozen_roi_analysis",
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

---

## 1. Create TIFF preview from CUB

Edit `SOURCE_CUB` in:

```text
scripts/create_tiff_from_cub.py
```

Run:

```bash
python scripts/create_tiff_from_cub.py
```

This uses:

```bash
catlab from=<SOURCE_CUB>
isis2std from=<SOURCE_CUB> to=<same_name>.tif format=tiff
```

The TIFF is for visual ROI selection only. Scientific analysis still uses the CUB file, or the rotated DTM `.npy` for rotated ROIs.

---

## 2. Select ROI

Edit the user settings at the top of:

```text
scripts/select_roi_for_catalog_multibody_clean.py
```

Important fields:

```python
ACTIVE_DATASET = "europa"
ROI_ID = "TEST_CHAOS_001"
SITE = "conamara(12hr)"
TERRAIN_TYPE = "Chaos"
PRODUCT_FOLDER = "sfs_data"
PRODUCT_TYPE = "Combined(Stereo + SFS[ZT])"
SOURCE_CUB = "/path/to/file.cub"
SOURCE_TIFF = "/path/to/file.tif"
PIXEL_SCALE_M = 9
SELECTION_ROTATION_DEG = 0
DISPLAY_SCALE = 0.5
```

Run:

```bash
python scripts/select_roi_for_catalog_multibody_clean.py
```

In the OpenCV window:

```text
Draw ROI rectangle
ENTER or SPACE = accept ROI
C = cancel
```

For rotated ROI selection, use a nonzero angle:

```python
SELECTION_ROTATION_DEG = -25
```

For rotated ROIs, the selector saves a true rotated science DTM at:

```text
results/<body>/selection/rotated_dtms/<ROI_ID>_rotated_raw_DTM.npy
```

---

## 3. Run ROI analysis

List datasets:

```bash
python scripts/run_roi_analysis_multibody_clean.py --list-datasets
```

Run one ROI:

```bash
python scripts/run_roi_analysis_multibody_clean.py   --dataset europa   --roi TEST_CHAOS_001
```

Run multiple ROIs:

```bash
python scripts/run_roi_analysis_multibody_clean.py   --dataset europa   --roi ROI_001   --roi ROI_002   --roi ROI_003
```

---

## 4. Analysis outputs

For each ROI, outputs are saved to:

```text
results/<body>/reproducible_rois/<ROI_ID>/
```

Important output files:

```text
figures/steinbruegge_rms_deviation_breakpoint_plot.png
figures/detrended_DTM_preview.png
figures/psd_diagnostic_multipanel.png
logs/analysis_log.txt
logs/steinbruegge_rms_deviation_summary.json
data/steinbruegge_rms_deviation_curve.csv
data/steinbruegge_rms_deviation_fit_bins.csv
```

The body summary table is updated here:

```text
results/<body>/summaries/tables/roi_analysis_summary.csv
```

---

## 5. Make grouped summary workbook

Run:

```bash
python scripts/summarize_roi_tables_simple.py   --inputs     results/europa/summaries/tables/roi_analysis_summary.csv     results/ganymede/summaries/tables/roi_analysis_summary.csv     results/enceladus/summaries/tables/roi_analysis_summary.csv   --output results/roi_grouped_summary.xlsx
```

Useful workbook sheets:

```text
01_BP_by_body_terrain_product
02_Good_by_body_terrain
03_BP_by_body_product
04_QC_counts
05_Site_QC
06_Flagged_ROIs
07_Cleaned_Key_Rows
```

---

## 6. Make Hurst-vs-baseline plots

Run:

```bash
python scripts/plotting/plot_figure12_hurst_baseline_panels.py   --excel results/roi_grouped_summary.xlsx   --sheets Ganymede_Product_Good Europa_Product_Good Enceladus_Product_Good   --output-dir figures/figure12_clean_readable   --split-by-body   --plot-mode individual   --color-by Site   --legend-position bottom   --legend-ncols 4   --max-legend-label-chars 16   --x-min 1   --x-max 10000   --verbose
```

The plot uses:

```text
Short-scale point: x = Pixel Scale m/pixel, y = H Before BP
Long-scale point: x = Breakpoint m, y = H After BP
```

---

## Interpretation notes

* `H Overall` is one slope over the full fitted baseline range.
* `H Before BP` is the short-baseline slope before the accepted breakpoint.
* `H After BP` is the long-baseline slope after the accepted breakpoint.
* `Delta H = H Before BP - H After BP`.
* Breakpoint is a candidate roughness transition scale.
* Breakpoint is not automatically a ridge spacing, fracture spacing, radar wavelength, or ice-shell property.
* PSD uses spatial frequency, not radar frequency.
* These measurements provide terrain-specific topographic roughness context for radar surface clutter; they do not directly simulate REASON radar returns.

---

## Quick reproduction example

```bash
cd /Users/mnebart/icy/github/europa-surface-analysis
conda activate isis9.0.0

python scripts/create_tiff_from_cub.py
python scripts/select_roi_for_catalog_multibody_clean.py

python scripts/run_roi_analysis_multibody_clean.py   --dataset europa   --roi TEST_CHAOS_001

python scripts/summarize_roi_tables_simple.py   --inputs     results/europa/summaries/tables/roi_analysis_summary.csv     results/ganymede/summaries/tables/roi_analysis_summary.csv     results/enceladus/summaries/tables/roi_analysis_summary.csv   --output results/roi_grouped_summary.xlsx

python scripts/plotting/plot_figure12_hurst_baseline_panels.py --excel results/roi_grouped_summary.xlsx --sheets Ganymede_Product_Good Europa_Product_Good Enceladus_Product_Good --output-dir figures/figure12_clean_readable --split-by-body --plot-mode individual --color-by Site --legend-position bottom --legend-ncols 4 --max-legend-label-chars 16 --x-min 1 --x-max 10000 --verbose
```

---


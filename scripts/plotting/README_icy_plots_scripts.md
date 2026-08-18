# Icy Worlds Roughness Plotting Scripts

This script reads an Excel workbook that contains ROI summary sheets such as:

- `Ganymede_Product_Good`
- `Europa_Product_Good`
- `Tethys_Product_Good`
- `Enceladus_Product_Good`

They are designed to handle the block-style tables where each sheet has repeated `ROI_ID` header rows, terrain/product blocks, and summary boxes beneath the ROI rows.

## Install requirements

```bash
python -m pip install pandas matplotlib openpyxl numpy
```

## Script: Hurst vs baseline panels

File: `plot_hurst_baseline_panels.py`

This creates multi-panel plots:

- x-axis: baseline distance in meters, log scale
- y-axis: Hurst exponent
- panels: terrain types
- short-scale point: x = Pixel Scale m/pixel, y = H Before BP
- long-scale point: x = Breakpoint m, y = H After BP

Example:

```bash
python scripts/plotting/plot_hurst_baseline_panels_clean.py \
  --excel roi_grouped_summary.xlsx \
  --sheets Ganymede_Product_Good Europa_Product_Good Enceladus_Product_Good Tethys_Product_Good \
  --output-dir figures/figure12_individual_clean \
  --split-by-body \
  --plot-mode individual \
  --color-by Site \
  --legend-position right \
  --legend-ncols 4 \
  --max-legend-label-chars 16 \
  --verbose
```
For the mean ± standard deviation version
Use aggregate mode:
```bash
python scripts/plotting/plot_hurst_baseline_panels_clean.py \
  --excel roi_grouped_summary.xlsx \
  --sheets Ganymede_Product_Good Europa_Product_Good Enceladus_Product_Good \
  --output-dir figures/figure12_aggregate_clean \
  --split-by-body \
  --plot-mode aggregate \
  --color-by Site \
  --legend-position right \
  --legend-ncols 4 \
  --max-legend-label-chars 16 \
  --verbose
```
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
## Recommended scientific filters

Use:

- `QC Status = Good` for main plots
- `Breakpoint Accepted = TRUE` for H Before BP, H After BP, Delta H, Breakpoint m, and Breakpoint pixels
- Keep Questionable/Reject rows out of final science figures unless explicitly labeled as exploratory.

## Caption caution

For Figure-12-style plots:

> The short-scale point is plotted at the DTM pixel scale using H Before BP, and the long-scale point is plotted at the accepted breakpoint baseline using H After BP. These H values are fitted exponents for scale ranges, not single-baseline measurements.

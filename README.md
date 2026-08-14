# SEM/TEM Particle Toolkit
## User manual and configuration reference

**Documentation status:** 25 June 2026
**Repository:** `https://github.com/VitPavelka/sem_coverage`
**Default Git branch:** `master`

This manual is intended for users with little or no Python experience.
It covers installation, updates, interactive viewers, batch export,
every current configuration parameter, and troubleshooting.

**Important:** This software is still under development. 
Every result is an automated image-analysis *estimate*.
Before using statistics, inspect overlays from at least several representative images.
Pixel-based settings depend on magnification, resolution, and particle size.
At first, try to keep separate configuration presets for different acquisition protocols.

## 1. Included tools

| Task                                           | Launcher                        | Configuration                     | Input                                   |
|------------------------------------------------|---------------------------------|-----------------------------------|-----------------------------------------|
| Unified interactive diagnostic tuning          | `run_diagnostic_viewer.py`      | `sem_bead_viewer_config.json` or `sem_coverage_viewer_config.json` | SEM `.tif` plus optional paired `.hdr`; modes `beads` and `coverage`; temporary tuning only; TEM diagnostics are not implemented yet |
| Size analysis of bright SEM beads              | `run_bead_viewer.py`            | `sem_bead_viewer_config.json`     | `.tif` plus optional paired `.hdr`      |
| Ag coverage and Ag-count estimate on SEM beads | `run_sem_coverage_viewer.py`    | `sem_coverage_viewer_config.json` | `.tif` plus optional paired `.hdr`      |
| Size analysis of dark TEM nanoparticles        | `run_tem_particle_viewer.py`    | `tem_particle_viewer_config.json` | `.png`, `.jpg`, `.jpeg`                 |
| Batch SEM export                               | `batch_export_protocols.py`     | the two SEM configs as templates  | directory tree containing `.tif` files  |
| Batch TEM export                               | `batch_export_tem_protocols.py` | TEM config as a template          | directory tree containing PNG/JPG files |
| Additional SEM summary re-export               | `export_output_summaries.py`    | command-line arguments            | previously generated SEM JSON files     |

## 2. Installation and updates

### 2.1 Easiest one-time installation from a ZIP archive

1. Install **Python 3.10** or higher (Preferably 3.13).
2. During Windows installation, select **Add Python to PATH**.
3. Extract the entire archive to a writable folder, for example `C:\SEM_TEM_analysis`.
4. Open that folder in File Explorer, click the address bar, type `cmd`, and press Enter.
5. Create an isolated environment and install dependencies:

```bat
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For every later terminal session, you have to activate the environment again:

```bat
.venv\Scripts\activate
```

Installation check:
```bat
python -c "import numpy, scipy, matplotlib, skimage, tifffile; print('Installation OK')"
```

On Linux, activate with `source .venv/bin/activate`.

### 2.2 Recommended installation from a Git repository

1. Install **Git for Windows** (or use the system Git package on Linux).
2. Open folder you want to store this software in File Explorer, click the address bar, type `cmd`, and press Enter.
3. In the terminal run:

```bat
git clone https://github.com/VitPavelka/sem_coverage.git
cd sem_coverage
py -3 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The software is a clone of the original repository and therefore can receive updates. 
The repository currently uses the `master` branch.

### 2.3 Checking for updates without checking local files (`git fetch`)

Use this when you only want to see whether newer commits exist:

```bat
.venv\Scripts\activate
git status -sb
git fetch origin
git log --oneline HEAD..origin/master
```

- `git fetch origin` dowlnoads information about remote changes but does **not** modify your project files.
- If the final command prints no commits, the working copy is up to date.
- `git status -sb` also shows local file edits that may need attention before updating.
- `git log --oneline HEAD..origin/master` shows the changes that have been made since the last update.

### 2.4 Updating the project (`git pull`)

When there are no local edits:

```bat
git pull --ff-only
python -m pip install -r requirements.txt --upgrade
```

`git pull --ff-only` updates only when Git can perform a clean fast-forward merge.
It refuses rather than silently creating a merge commit.

### 2.5 Protecting locally edited configuration files during an update

Git tracks the distributed JSON configuration files. 
If you edit them directly, Git may refuse an update or report a conflict.
For non-technical users, the safest procedure is:

1. Run `git status -sb`.
2. Copy every locally edited `*_config.json` file to a backup folder outside the repository.
3. Restore the tracked config files to the repository version:
    ```bat
    git restore sem_bead_viewer_config.json
    git restore sem_coverage_viewer_config.json
    git restore tem_particle_viewer_config.json
    ```
4. run `git pull --ff-only`.
5. Reapply only the required paths and tuned values from the backup.

Do *not* at any circumstance use `git reset --hard`. 

### 2.6 Development and extending the code

This project is (and will be for some time) under construction.
Any developers are more than welcome to contribute:
    If this is you, please create a separate branch before changing code and contact me (`pavelka.vit69@gmail.com`).

## 3. Configuration files and path handling

Open the JSON files in Notepad, Notepad++, or VS Code.
Routine users normally change only paths and values under `viewer`.

### JSON rules

- Text must use double quotes: `"text""`.
- Boolean values are lowercase: `true`/`false`.
- An empty value is `null`.
- Do not place a comma after the final item in an object.
- JSON does not support comments.
- On Windows, prefer `/`: `"C:/Data/TEM"`. Alternatively, escape backslashes: `"C:\\Data\\TEM"`.
- A single unescaped backslash is not valid JSON. For example, `"C:\Data\SEM"` is invalid JSON.
- If editing long Windows paths in JSON is inconvenient, use CLI `--folder` or `--file` overrides instead.

### Common top-level fields

| Field               | Meaning                                                                                          | Recommendation                                                        |
|---------------------|--------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| `folder`            | Folder containing input images.                                                                  | The path must exist.                                                  |
| `file`              | Optional single file in SEM coverage and TEM configurations (`null` means all supported images). | Enter a file name or path. SEM bead viewer has no `file` field.       |
| `summary_json_path` | Optional path for a summary JSON. `null` is the simplest interactive mode.                       | Routine users should normally leave it `null` and use batch scripts.  |

Example:

```json
{
  "folder": "C:/Data/TEM/sample_01",
  "file": null,
  "summary_json_path": null,
  "viewer": {
    "detector": "sauvola"
  }
}
```

Relative config input paths keep the current legacy behavior:

1. absolute paths are used directly;
2. a relative input path first tries the current working directory;
3. if that path does not exist, the same relative path is tried next to the config file.

This keeps existing working-directory-based setups working while allowing more portable configs.

For `file` fields, an absolute file path is used directly. A relative file path is resolved inside the effective `folder`.

Grouped coverage manifests use a separate path rule for `configs_root`: a
relative `configs_root` is resolved next to the manifest itself. Paths inside
each referenced coverage config retain the ordinary config rules above.

## 4. Recommended workflow

1. **Diagnostic viewer**
   Tune and understand parameters on representative images.
2. **Regular viewer**
   Verify the tuned configuration on several images without the full diagnostic control panel.
3. **Batch export**
   Process complete folders and generate final JSON, PNG, CSV, and/or XLSX outputs.
4. **Optional SEM summary re-export**
   If SEM JSON outputs already exist, regenerate CSV/XLSX tables or histograms later with `export_output_summaries.py`.

## 5. Interactive diagnostic viewer

Run commands from the project folder while `.venv` is active.

The unified diagnostic viewer currently supports **SEM bead** and **SEM coverage** modes:

```bat
python run_diagnostic_viewer.py --mode beads --config sem_bead_viewer_config.json
python run_diagnostic_viewer.py --mode coverage --config sem_coverage_viewer_config.json
```

Use `--file FILE` to select a specific TIFF, `--folder FOLDER` to temporarily
override the configured folder, `--output-config PATH` to choose the tuned-copy
destination, or `--no-async` to debug with synchronous recalculation. Parameter
changes are temporary and update the image automatically; the source JSON is not
changed. Click **Save tuned config** or press `S` to write a separate tuned copy
(`*_tuned.json` by default).

Sliders now update their numeric values while dragging but commit on mouse
release. The **Apply changes** button commits the current pending values
explicitly, which is useful after keyboard edits or coordinated multi-control
changes. The title and status area distinguishes `PENDING`, `RUNNING`, and
`UP TO DATE`.

- Left/Right or the Previous/Next buttons navigate images; Home/End select the first/last image.
- Up/Down cycles mode-specific diagnostic stages without rerunning analysis.
- `R` resets the current parameter group; `Shift+R` resets every parameter to the
  values loaded from the source JSON; `F5` reloads the image list and source config.
- `H` displays keyboard help. Hover over a parameter control for tuning help.

Beads mode provides stages for the display image, DoG feature, candidate mask,
labels, valid mask, outlier mask, and overlay. Overlay checkboxes independently
control valid, rejected, and candidate boundaries, dimension lines, measurement
labels, and the scale bar.

Coverage mode provides stages for the overlay, display image, normalized image,
raw/refined bead masks, Ag count and coverage features, Ag count and coverage
masks, Ag peak map, and ROI index map. Coverage overlay checkboxes independently
control bead boundaries, ROI inclusion status, diameter lines, diameter labels,
Ag coverage boundaries, Ag count boundaries, Ag peak markers, ROI index labels,
and the scale bar.

Coverage mode also supports ROI inspection. Use the Previous ROI, Next ROI, and
All ROIs buttons or the `[`, `]`, and `\` keys to switch between All ROIs and
individual bead ROIs. ROI changes update the visualization and status only; they
do not rerun analysis. Failed coverage segmentations keep the preview image
visible so the parameters can be tuned until a valid ROI is recovered.

Some coverage controls are conditionally inactive. For example, the single
coverage top-hat radius is inactive while the multi-radius list is populated,
adaptive-threshold settings are inactive when adaptive thresholding is disabled,
and secondary-coverage controls are inactive while the secondary coverage branch
is disabled. The status/help area explains why a control is inactive.

Coverage global sphere filters update ROI inclusion colors and aggregate
statistics immediately without rerunning segmentation. Coverage display
percentiles also update the displayed contrast immediately without starting a
full analysis.

Select the parameter group on the right to expose the relevant controls. Beads
currently offers Preprocessing, Detection, Morphology and size, Watershed
splitting, and Filtering. Coverage offers Preprocessing and display, Primary
bead segmentation, Morphology fallback, Bead splitting, Ag count detector,
Ag coverage detector, and Global sphere filters.

TEM diagnostics are still planned but are not implemented in this unified
diagnostic application yet.

## 6. Regular interactive viewers

Run commands from the project folder while `.venv` is active.

### 6.1 SEM bead size

```bat
python run_bead_viewer.py
python run_bead_viewer.py --help
python run_bead_viewer.py --config sem_bead_viewer_config.json
```

The viewer loads lowercase **`.tif`** files directly inside `folder`.
Left/right arrows change images.
Checkboxes control the scale bar, boundaries, and size labels.
Use `--folder PATH` for a temporary folder override without changing the JSON config.

### 6.2 SEM Ag coverage

```bat
python run_sem_coverage_viewer.py
python run_sem_coverage_viewer.py --help
python run_sem_coverage_viewer.py --config sem_coverage_viewer_config.json --folder "C:/Data/SEM coverage"
```

Left/right arrows change images. Up/down arrows switch diagnostic layers:

- `display` - original image scaled for display,
- `norm` - normalization used for bead detection,
- `bead_raw` - initial ROI candidates,
- `bead_refined` - final bead ROIs,
- `ag_count_feature` - top-hat response used for Ag peaks,
- `ag_coverage_feature` - response used for the coverage mask.

The launcher also supports `--file PATH` for a temporary single-image selection.

### 6.3 TEM particle size

```bat
python run_tem_particle_viewer.py
python run_tem_particle_viewer.py --help
python run_tem_particle_viewer.py --config tem_particle_viewer_config.json --file "TEM SeNPs 1.png"
```

Left/right arrows change images. The window shows the original image, detector feature map,
overlay, and histogram/statistics.

Regular viewers can optionally write summary JSON files when `summary_json_path`
is configured. For routine final multi-sample processing, the batch exporters are
recommended instead. CSV and XLSX output belongs to the batch/export layer, not
the regular viewer layer.

## 7. Input files and physical calibration

### 7.1 SEM TIFF and HDR

For `sample.tif`, the SEM viewers look for metadata in:

```text
sample-tif.hdr
```

The HDR file supplies pixel size, magnification, and measurement information.
Analysis can run without HDR, but sizes and scale bars may remain in pixels.

### 7.2 TEM

The TEM loader currently supports only grayscale, RGB, and RGBA PNG/JPG images.
Enter either `pixel_size_nm` or `fov_nm` for physical sizes.
A direct `pixel_size_nm` value take precedence.

## 8. Batch exports and table formats

### 8.1 SEM bead and coverage export

Beads only:

```bat
python batch_export_protocols.py --bead-root "C:/Data/SEM/size" --outputs-dir "C:/Data/results_sem" --clean
python batch_export_protocols.py --bead-root "C:/Data/SEM/size" --bead-config sem_bead_viewer_config.json --outputs-dir outputs
python batch_export_protocols.py --bead-config sem_bead_viewer_config.json --outputs-dir outputs
```

`--bead-root` overrides the bead config's top-level `folder`. If it is omitted,
an explicitly selected `--bead-config` supplies the input folder. The batch does
not modify that source configuration file.

Coverage only:

```bat
python batch_export_protocols.py --coverage-root "C:/Data/SEM/coverage" --outputs-dir "C:/Data/results_sem" --clean
python batch_export_protocols.py --coverage-config sem_coverage_viewer_config.json --outputs-dir outputs
python batch_export_protocols.py --batch-config batch_export_protocols.example.json --outputs-dir outputs
```

`--coverage-root` overrides the folder/file source in one ordinary
`--coverage-config`. Without that override, the config's source is used.
`--batch-config` is a distinct grouped mode and cannot be combined with either
`--coverage-config` or `--coverage-root`.

A grouped manifest is a JSON object whose top-level names are scientific sample
identifiers. Each sample lists tuned coverage configs:

```json
{
  "p1-b": {
    "configs_root": "Projects/coverage_configs",
    "config_names": ["p1-b1.json", "p1-b2"]
  },
  "p2-a": {
    "configs_root": "Projects/coverage_configs",
    "config_names": ["p2-a1.json"]
  }
}
```

Names with and without `.json` are accepted; no other filename guessing is
performed. The full manifest is validated before image analysis. Within one
scientific sample, assigning the same resolved TIFF through two configs is an
error. For each requested coverage branch, all ROI records from the sample's
tuned configs are pooled and the existing global-summary builder runs once on
that complete ROI population; subconfig means are never averaged.

The default `--coverage-branches configured` mode respects
`ag_enable_secondary_coverage` in each individual coverage config. It does not
create a branch directory: JSON and tables are written directly to
`--outputs-dir`, and grouped overlays appear as
`coverage_png/p1-b/<TIFF stem>.png`. Different tuned configs inside one
scientific sample may therefore select one-layer or two-layer analysis
individually while contributing to the same pooled summary.

The explicit comparison modes `one-layer`, `two-layers`, and `both` override
that config field and isolate their outputs under `coverage_one_layer/` and/or
`coverage_two_layers/`. Tuned config names do not create extra directories.
Colliding TIFF stems receive a deterministic readable suffix instead of being
overwritten. Rich JSON image and ROI records include `analysis_config_name` and
the resolved `analysis_config_path`. Group JSON also lists every distinct source
path; its scalar `source_path` is empty when multiple sources exist. Compact
CSV/XLSX schemas are unchanged.

Both tasks:

```bat
python batch_export_protocols.py --bead-root "C:/Data/SEM/size" --coverage-root "C:/Data/SEM/coverage" --outputs-dir "C:/Data/results_sem" --clean
python batch_export_protocols.py --bead-root "C:/Data/SEM/size" --outputs-dir outputs --table-format both
```

Optional arguments:

- `--bead-config FILE` - alternate bead-analysis template,
- `--coverage-config FILE` - alternate coverage template,
- `--batch-config FILE` - grouped scientific-sample manifest for multiple tuned coverage configs,
- `--coverage-branches {configured,one-layer,two-layers,both}` - use each config's branch setting by default, or explicitly override it for comparisons,
- `--clean` - remove previous JSON/CSV/PNG outputs in the target structure before running,
- `--table-format {csv,xlsx,both,none}` - choose summary table output,
- `--sort-by {name,path,none}` - control deterministic natural sorting,
- `--no-export` - create per-sample JSON and overlay PNG files but skip final table/histogram generation,
- `--no-csv`, `--no-bead-csv`, and `--no-coverage-csv` - suppress only CSV files,
- `--no-histograms` - keep summary tables but do not write bead histogram PNG files or their matching one-column TXT source-data files.

Typical outputs include per-sample JSON, `size_png`, `coverage_png`,
`bead_global_summaries.csv`, `coverage_global_summaries.csv`,
`sem_global_summaries.xlsx`, and `bead_histograms`.

#### Bead histogram source data

Bead histogram export automatically writes the exact numerical size vector
used to construct each histogram. No additional CLI switch is required. The
metric is selected with `size_distribution_metric` in the bead configuration.
The default is mean X/Y diameter:

```text
mean_xy_diameter = (x_diameter + y_diameter) / 2
```

`equivalent_diameter` remains available as an alternative. It is the diameter
of a circle with the segmented bead area:

```text
equivalent_diameter = 2 * sqrt(area / pi)
```

The selection affects distributions and summary statistics only. It does not
change segmentation, equivalent-diameter acceptance filters, or X/Y overlay
diagnostics. Only valid beads included in the matching histogram are written
to the data file.

Mean X/Y diameter is the current default because, for this dataset and this
implementation, it has been empirically less sensitive to local area loss
when touching beads are imperfectly separated by watershed. This is a
dataset- and implementation-specific robustness consideration, not a general
claim that mean X/Y diameter is always more accurate. Both X/Y dimensions and
equivalent diameter remain available in the outputs for comparison.

For each bead summary, the exporter creates a PNG/TXT pair in
`<outputs-dir>/bead_histograms/`:

```text
<sample>_bead_mean_xy_diameter_um_histogram.png
<sample>_bead_mean_xy_diameter_um_values.txt
```

For data without physical calibration, the corresponding files use pixel
units:

```text
<sample>_bead_mean_xy_diameter_px_histogram.png
<sample>_bead_mean_xy_diameter_px_values.txt
```

When `size_distribution_metric` is `equivalent_diameter`, the corresponding
filenames use `equivalent_diameter` instead of `mean_xy_diameter`.

Combined batch outputs use the same convention:

```text
all_bead_mean_xy_diameter_um_histogram.png
all_bead_mean_xy_diameter_um_values.txt
```

or the corresponding `_px_` filenames when calibration is unavailable.

Each TXT file is UTF-8 plain text with one column and one value per line. Its
header identifies the selected metric and unit, for example
`mean_xy_diameter_um`, `mean_xy_diameter_px`, `equivalent_diameter_um`, or
`equivalent_diameter_px`. It contains exactly the same valid size vector supplied
to the matching histogram. The files can be opened directly in Excel, Origin,
Python, R, or other statistics software.

```text
mean_xy_diameter_um
1.2384
1.1942
1.2715
1.2250
```

Normal bead export writes both the histogram PNG and its matching values TXT.
`--no-export` skips final tables, histograms, and histogram-value TXT files.
`--no-histograms` skips both bead histogram PNG files and their matching TXT
files. `--table-format` controls CSV/XLSX summary tables only; it does not
control histogram source-data TXT files. `export_output_summaries.py` can
regenerate both histogram PNG and matching TXT files from existing bead JSON
summaries.

### 8.2 TEM batch export

```bat
python batch_export_tem_protocols.py --root "C:/Data/TEM" --outputs-dir "C:/Data/results_tem" --clean
python batch_export_tem_protocols.py --root "C:/Data/TEM" --outputs-dir outputs_tem --table-format xlsx
```

Optional arguments:

- `--config FILE` - alternate TEM template,
- `--clean` - delete old TEM outputs,
- `--table-format {csv,xlsx,both,none}` - choose summary table output,
- `--sort-by {name,path,none}` - control deterministic natural sorting,
- `--no-csv` - skip CSV creation while leaving XLSX enabled,
- `--no-histograms` - skip histogram creation.

Outputs include per-sample and global JSON, per-image JSON, overlay PNG files,
two CSV summaries, `tem_summaries.xlsx`, and histograms.

### 8.3 Regenerating SEM summaries from existing JSON files

```bat
python export_output_summaries.py --outputs-dir "C:/Data/results_sem"
python export_output_summaries.py --outputs-dir "C:/Data/results_sem" --table-format both
```

Use this when SEM JSON files already exist and you want to regenerate CSV,
XLSX, or histogram outputs later. The legacy `--no-csv`, `--no-bead-csv`,
`--no-coverage-csv`, and `--no-histograms` switches remain available here.

### 8.4 Extracting compact publication tables

After CSV/XLSX export, create compact sibling tables recursively with:

```bat
python extract_output_tables.py "C:/Data/results_sem"
```

The defaults retain the existing `caps` coverage metric
(`projected_over_cap_surface`) in the `homo` homogeneity-domain annulus. Both
choices are configurable without changing code:

```bat
python extract_output_tables.py "C:/Data/results_sem" --coverage-metric surfw --region homo
python extract_output_tables.py "C:/Data/results_sem" --coverage-metric caps --region cap
```

Recognized CSV files and recognized coverage worksheets are reduced to row
identity/provenance, essential counts and validity fields, and the selected
scientific family. For example,
`coverage_global_summaries.csv` becomes
`coverage_global_summaries-extracted.csv`, and
`coverage_summaries.xlsx` becomes `coverage_summaries-extracted.xlsx`, next to
the source. Source files are untouched. Rows are not aggregated, values are not
rounded or recalculated, and no image analysis is run.

The extractor skips unrelated tables, spreadsheet temporary files, and stems
already ending in `-extracted`, so repeated recursive runs do not create
`-extracted-extracted` files. Existing sibling outputs are skipped unless
`--overwrite` is supplied. Use `--dry-run` to preview recognized tables and
column-count reductions without writing anything; `--suffix` changes the
sibling suffix. Add `--drop-medians` for an even smaller overview that removes
only schema-classified descriptive population medians when the same table also
retains the mean of that population. It is off by default. Rotation-robust
polar/total/residual estimators, and medians without an equivalent mean, remain
untouched.

The current metric choices are `proj`, `caps`, and `surfw`; the current region
choices are `cap` and `homo`. Local-cell, radial-profile, and polar-sector
tables describe only the `homo` domain and are skipped for `--region cap`.
The long-form Polar rotations table stores metric families in rows rather than
columns, so it is explicitly omitted from an extracted workbook (or skipped as
a standalone CSV) to keep this utility strictly column-selecting. Unrecognized
workbook sheets are preserved unchanged.

# 9. Configuration reference: `sem_bead_viewer_config.json`

This tool detects **bright beads on a darker background**, measures x/y dimensions, and classifies candidates as valid (green) or rejected (red).

| Parameter                              | What it controls                                                                                    | Tuning advice / common problems                                                                                                            |
|----------------------------------------|-----------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------|
| `infobar_tail_rows` = `320`            | Number of bottom rows searched for the bright SEM information bar.                                  | Increase if the bar starts higher. Decrease for very short images or to limit the search area.                                             |
| `infobar_k_mad` = `8.0`                | Sensitivity of automatic bright-bar detection based on the median and MAD.                          | Lower values crop more sensitively but can cut image content. Higher values are more conservative.                                         |
| `infobar_min_run` = `10`               | Minimum number of consecutive bright rows required to identify the bar.                             | Decrease if a short footer is missed. Increase if a bright image region causes false cropping.                                             |                                                                                                    |
| `display_percentiles` = `[0.5, 99.5]`  | Percentile contrast scaling. In this viewer the scaled image also participates in segmentation.     | A narrower range such as `[1, 99]` increases contrast but can saturate extremes. A wider range preserves tones but may weaken faint beads. |
| `dog_sigma_small` = `1.2`              | Small Gaussian sigma in the Difference of Gaussians (DoG); defines the fine spatial scale.          | Lower values retain smaller objects and more noise. Higher values suppress noise but can remove small beads.                               |
| `dog_sigma_large` = `8.0`              | Large Gaussian sigma representing slowly varying background.                                        | Must exceed `dog_sigma_small`. Increase for larger objects, but excessive values can merge nearby structures.                              |
| `dog_foreground_percentile` = `80.0`   | Strong-response subset of the DoG image used to derive the automatic Otsu threshold.                | Higher values retain only strong structures and may miss faint beads. Lower values are more sensitive but noiser.                          |
| `intensity_percentile` = `97.5`        | A candidate must be brighter than this image-intensity percentile.                                  | Decrease when darker or smaller beads are missed. Increase when bright background or artifacts are accepted.                               |
| `closing_radius` = `2`                 | Morphological closing fills small gaps and corrects nearby regions.                                 | Increase to repair broken bead masks; decrease when neighboring beads merge.                                                               |
| `opening_radius` = `1`                 | Morphological opening removes small noise and thin connecting bridges.                              | Increase to remove noise or break thin bridges. Too large values erases small beads.                                                       |
| `min_object_area_px` = `50`            | Minimum candidate area in pixels; smaller components are removed.                                   | Decrease when small beads are missed. Increase when granular noise is counted.                                                             |
| `diameter_size_limits` = `false`       | Enables minimum and maximum equivalent-diameter filtering.                                          | Set to `false` during diagnosis. It changes acceptance into statistics, not the initial segmentation.                                      |
| `min_diameter_px` = `14.0`             | Smallest accepted equivalent diameter in pixels.                                                    | Decrease for smaller beads. Active only when `diameter_size_limits=true`                                                                   |
| `max_diameter_px` = `60.0`             | Largest accepted equivalent diameter in pixels.                                                     | Increase for larger beads. Too low a value marks valid objects red.                                                                        |
| `size_distribution_metric` = `"mean_xy_diameter"` | Scalar used for bead histograms and summary statistics: `mean_xy_diameter` or `equivalent_diameter`. | Default `mean_xy_diameter` is `(d_x + d_y) / 2`, the mean full horizontal/vertical mask extent. It is the current dataset-specific default because it is empirically less sensitive to local watershed area loss for imperfectly separated touching beads; this is not a universal accuracy claim. Optional `equivalent_diameter` is `2 * sqrt(A / pi)`, the diameter of a circle with the segmented bead area. This reporting choice does not affect segmentation or existing equivalent-diameter filters; X/Y dimensions remain available for anisotropy and overlays. |
| `peak_min_distance_px` = `8`           | Reserved parameter for peak detection.                                                              | Not used by the current code; changing it has no effect.                                                                                   |
| `peak_threshold_px` = `3.5`            | Reserved parameter for peak detection.                                                              | Not used by the current code; changing it has no effect.                                                                                   |
| `use watershed_split` = `true`         | Enables watershed separation of connected bead candidates.                                          | Disable if single beads are over-split. Enable when touching beads are counted as one.                                                     |
| `split_only_suspicious` = `true`       | Attempts watershed only on candidates that are unusually large, elongated, or insufficiently solid. | `true` is safer. `false` attempts to split every candidqte and increases over-splitting risk.                                              |
| `split_min_distance_px` = `10`         | Minimum distance between watershed markers.                                                         | Lower values allow more markers and more aggressive splitting. Higher values produce fewer splits.                                         |
| `split_threshold_px` = `5.0`           | Minimum height of a distance-transform maximum in pixels.                                           | Lower values accept weaker markers and produce more splitting. Higher values are more conservative.                                        |
| `split_min_peak_count` = `2`           | Minimum number of markers required before a plit is accepted.                                       | Normally 2. A larger value rejects ordinary two-bead splits.                                                                               |
| `split_max_peak_count` = `4`           | Maximum allowed marker count within one parent candidate.                                           | Decrease when splitting is excessive. Increase only for genuine multi-bead clusters.                                                       |
| `split_min_child_area_px` = `120`      | Minimum area of each child after splitting.                                                         | Increase to reject tiny fragments. Decrease for genuinely small beads.                                                                     |
| `split_min_child_diameter_px` = `14.0` | Minimum equivalent diameter of a split child.                                                       | Same role as minimum child area, expressed as a more intuitive diameter.                                                                   |
| `split_max_child_diameter_px` = `60.0` | Maximum equivalent diameter of a split child.                                                       | Increase if a valid large beads prevent acceptance of a split.                                                                             |
| `split_trigger_diameter_px` = `34.0`   | Diameter above which a candidate is considered a suspiciously large cluster.                        | Decrease to attempt splitting more often. Increase if large single beads are unnecessarily split.                                          |
| `split_trigger_axis_ratio` = `1.18`    | Axis ratio above which a candidate is considered suspiciously elongated.                            | Decrease for more aggressive splitting of elongated candidates. Increase for more tolerance.                                               |
| `split_trigger_solidity_below` = `0.9` | Solidity below which a candidate is considered an irregular cluster.                                | Increase to attempt splitting more irregular regions. Too high a value also targets normal beads.                                          |
| `boundary_linewidth` = `1.0`           | Reserved boundary-line width setting.                                                               | Not used by the current viewer; changing it has no effect.                                                                                 |
| `outlier_axis_ratio` = `1.22`          | Maximum accepted x/y dimension ratio before an object is flagged as anisotropic.                    | Increase for irregular or perspective-distorted beads. Decrease for a stricter circularity requirement.                                    |
| `global_size_outliers` = `false`       | Enables robust size-outlier rejection within one image.                                             | Disable for genuinely multimodal size mixtures. Enable for nominally monodisperse samples.                                                 |
| `outlier_mad_zscore` = `3.5`           | Robust MAD z-score threshold used for global size outliers.                                         | Higher values are more tolerant. Lower values mark more size outliers red.                                                                 |
| `min_solidity` = `0.72`                | Minimum ratio of object area to convex-hull area.                                                   | Decrease for irregular beads or partial overlaps. Increase to reject lobed clusters.                                                       |
| `max_eccentricity` = `0.95`            | Maximum region eccentricity; 0 is circular and values near 1 are elongated.                         | Decrease for stricter circularity. Increase to retain more elongated objects.                                                              |
| `edge_touch_margin_px` = `0`           | Image-border margin considered to be edge contact.                                                  | Increase to flag objects close to the border. Zero checks direct contact only.                                                             |
| `include_edge_candidates` = `true`     | Whether objects touching the image edge may enter statistics.                                       | `false` is more conservative because a clipped object has an incomplete measured size.                                                     |
| `default_show_scale` = `true`          | Initial visibility of the scale bar.                                                                | Display only; it does not change the analysis.                                                                                             |
| `default_show_boundaries` = `true`     | Initial visibility of valid/rejected boundaries.                                                    | Display only.                                                                                                                              |
| `default_show_measures` = `true`       | Initial visibility of size crosses and labels.                                                      | Disable for dense images where labels obscure the image.                                                                                   |

# 10. Configuration reference: `sem_coverage_viewer_config.json`

The viewer first identifies bead ROIs and then estimates projected Ag coverage and Ag peak count.

## 10.1 `viewer.analyzer`

| Parameter                             | What it controls                                                                             | Tuning advice / common problems                                                                                               |
|---------------------------------------|----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| `infobar_tail_rows` = `320`           | Number of bottom rows searched for the bright SEM information bar.                           | Increase if the bar starts higher; decrease for short images.                                                                 |
| `infobar_k_mad` = `8.0`               | Sensitivity of information-bar detection.                                                    | Lower is more sensitive; higher is more conservative.                                                                         |
| `infobar_min_run` = `10`              | Minimum continuous height of the bar.                                                        | Decrease for a short bar; increase to avoid false cropping.                                                                   |
| `norm_percentiles` = `[2.0, 98.0]`    | Percentile normalization used for bead detection, not directly for the Ag top-hat operation. | A narrower range enhances the bead but may saturate. Change cautiously if ROI detection fails.                                |
| `bead_blur_sigma` = `2.0`             | Gaussian smoothing before Otsu segmentation of the bead.                                     | Higher values suppress texture and smooth the mask, but can merge neighbors or lose small beads.                              |
| `bead_closing_radius` = `5`           | Closing radius for the bead mask.                                                            | Increase to fill gaps; decrease when neighboring objects merge.                                                               |
| `bead_opening_radius` = `3`           | Opening radius for the bead mask.                                                            | Increase to remove noise and thin bridges, but note that small beads may shrink.                                              |
| `bead_hole_area` = `5000`             | Fills holes in the bead mask below this area.                                                | Increase if a correct bead contains a large unfilled hole. Excessive values can fill real cavities.                           |
| `ag_tophat_radius` = `9`              | White top-hat radius that sets the bright-feature scale for the primary Ag/count detector.   | Smaller values favor small sharp particles. Larger values capture broader structures. Top-hat selects scale; it is not a complete denoising step by itself. |
| `ag_mask_threshold_rel` = `1.0`       | Multiplier on the Otsu threshold used to create the primary Ag/count mask.                   | Increase above 1 for a more conservative mask; decrease below 1 for a more sensitive mask. This can change primary-only coverage. |
| `ag_opening_radius` = `1`             | Opening radius applied to the primary Ag/count mask; `0` disables opening.                   | Increase to remove narrow protrusions and small structures morphologically. It does not control the secondary coverage morphology. |
| `ag_min_object_size` = `5`            | Minimum final connected-component area in the primary mask, enforced after opening.          | Decrease when small nanoparticles are missed. Increase to suppress grain and opening-created fragments.                         |
| `ag_erode_bead_radius` = `2`          | Erosion of the bead ROI before Ag detection, excluding the bead edge.                        | Increase to suppress a false bright rim. Decrease to include Ag close to the edge.                                            |
| `ag_use_log` = `false`                | Applies `log1p` to original intensities before top-hat filtering.                            | Useful for a very broad dynamic range, but on sparse/noisy images it can reduce contrast between strong particles and moderate texture. |
| `count_min_distance` = `5`            | Minimum separation between local maxima counted as Ag particles.                             | Increase to avoid multiple counts per particle. Decrease for tightly packed particles.                                        |
| `count_thr_rel` = `1.0`               | Multiplier on the effective primary-mask threshold used only to accept local-maxima peaks.   | Higher gives fewer, stronger peak counts. It changes projected particle count and peak markers, but never the primary or coverage mask. |
| `display_percentiles` = `[0.5, 99.5]` | Contrast scaling used only for display.                                                      | Does not change numerical Ag segmentation; change only for viewer readability.                                                |

## 10.2 Other `viewer` parameters
| Parameter                                     | What it controls                                                                                               | Tuning advice / common problems                                                                                                                         |
|-----------------------------------------------|----------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------|
| `detector_choice_index` = `0`                 | Tile index for a multi-detector TIFF according to `ViewFiieldsCountX/Y` in the HDR file; indexing starts at 0. | Keep 0 for a normal single image. An invalid index raises an error.                                                                                     |
| `min_bead_area_px` = `500`                    | Initial minimum component area and part of final ROI validation.                                               | Decrease for smaller beads. Increase to reject small false objects.                                                                                     |
| `min_roi_eq_diameter_px` = `140.0`            | Minimum equivalent diameter of the final bead ROI.                                                             | Decrease when small beads disappear. Increase to reject smaller artefacts.                                                                              |
| `min_roi_solidity` = `0.82`                   | Minimum compactness of an accepted ROI.                                                                        | Decrease for irregular or partly obscured beads. Increase to reject lobed clusters.                                                                     |
| `max_roi_anisotropy_ratio` = `1.65`           | Maximum major/minor-axis ratio allowed for an ROI.                                                             | Increase for distored or partial beads; decrease for stricter spherical appearance.                                                                     |
| `sphere_anisotropy_check` = `true`            | Additional anisotropy filter used only for inclusion in global statistics.                                     | The ROI and per-image result remain visible; a failing ROI is omitted from the global summary.                                                          |
| `max_global_sphere_anisotropy_ratio` = `1.25` | Threshold for the global anisotropy filter.                                                                    | Higher is more tolerant. Used only when `sphere_anisotropy_check` is enabled.                                                                           |
| `sphere_soliditiy_check` = `true`             | Additional global-statistics filter based on solidity.                                                         | Affects summary inclusion, not the segmentation itself.                                                                                                 |
| `min_global_sphere_solidity` = `0.9`          | Minimum solidity for inclusion in global statistics.                                                           | Decrease for irregular ROIs; increase for a stricter selection.                                                                                         |
| `salvage_open_radius_px` = `7`                | If a candidate fails filters, opening attempts to isolate and recover its compact core.                        | Increase to rescue a bead connected by a thin bridge, but excessive opening deforms the mask.                                                           |
| `bead_morph_fallback` = `true`                | Enables gradient-based fallback bead segmentation when the primary method fails.                               | Recommended on. Disable only when diagnosing failure of the primary method.                                                                             |
| `bead_morph_downscale` = `0.25`               | Image downscaling factor used by fallback segmentation.                                                        | Lower is faster but less detailed. Higher is more accurate and slower. Allowed range: 0.05-1.0.                                                         |
| `bead_morph_blur_sigma` = `4.0`               | Smoothing of the downscaled image before gradient computation.                                                 | Increase to suppress texture; excessive smoothing blurs small edges.                                                                                    |
| `bead_morph_gradient_percentile` = `80.0`     | Gradient percentile above which edges are retained.                                                            | Higher keeps only the strongest edges and may break the outline. Lower retains more edges and noise.                                                    |
| `bead_morph_close_radius` = `2`               | Closing radius for edges in fallback segmentation.                                                             | Increase to connect a broken outline; too large a radius joins objects.                                                                                 |
| `bead_morph_dilate_radius` = `2`              | Edge dilation before hole filling in fallback segmentation.                                                    | Increase to close gaps. Excessive dilation enlarges or joins ROIs.                                                                                      |
| `bead_morph_erode_radius_px` = `20`           | Erosion of the final fallback mask after returning to full resolution.                                         | Increase if the fallback ROI exceeds the real bead edge. Decrease if the ROI is too small.                                                              |
| `bead_morph_min_object_area_ratio` = `0.08`   | Minimum fallback-component area as a fraction of the whole image.                                              | Decrease for smaller beads. Increase to reject broad noise regions or unclosed areas.                                                                   |
| `split_touching_beads` = `true`               | Enables watershed splitting of touching bead ROIs.                                                             | Disable for over-splitting; enable when two beads remain merged.                                                                                        |
| `split_trigger_eq_diameter_px` = `430.0`      | Equivalent diameter above which an ROI is considered a suspiciously large cluster.                             | Decrease to try splitting more often. Increase to avoid splitting large individual beads.                                                               |
| `split_trigger_anisotropy_ratio` = `1.45`     | Anisotropy ratio that triggers a split attempt.                                                                | Decrease for more aggressive splitting of elongated candidates.                                                                                         |
| `split_trigger_solidity_below` = `0.9`        | Low-solidity threshold that triggers a split attempt.                                                          | Increase to try splitting more irregular regions.                                                                                                       |
| `split_min_distance_px` = `70`                | Minimum distance between watershed markers.                                                                    | Lower produces more splitting. Higher is more conservative.                                                                                             |
| `split_peak_threshold_rel` = `0.55`           | Minimum distance-peak height as a fraction of the strongest peak within the parent.                            | Lower accepts weaker markers and more splits. Higher retains only pronounced centers.                                                                   |
| `split_max_peaks` = `4`                       | Maximum number of allowed split markers.                                                                       | Decrease when over-splitting occurs. Increase for genuine multi-bead clusters.                                                                          |
| `split_min_child_area_ratio` = `0.18`         | Every child must occupy at least this fraction of the parent area.                                             | Increase to reject tiny fragments. Decrease when touching beads differ strongly in size.                                                                |
| `ag_enable_secondary_coverage` = `false`      | Enables a separate, usually more sensitive mask for coverage. If `false`, coverage uses the primary count mask. | When `false`, the following `ag_coverage_*` settings are effectively ignored. A sensitive secondary branch can worsen false positives on sparse noisy samples, so verify its overlay. |
| `ag_coverage_tophat_radius` = `15`            | Single top-hat radius for the secondary coverage mask.                                                         | Used when `ag_coverage_tophat_radii` is not provided.                                                                                                   |
| `ag_coverage_tophat_radii` = `[15, 25]`       | Optional list of top-hat radii; the maximum response across scales is used.                                    | Use for Ag structures with varied sizes. More or larger radii are slower and may include background.                                                    |
| `ag_coverage_threshold_rel` = `0.8`           | Multiplier of the Otsu threshold for the secondary coverage mask.                                              | Lower gives more and more sensitive coverage. Higher is more conservative.                                                                              |
| `ag_coverage_adaptive_threshold` = `true`     | Adds a local threshold of `local_mean + k*local_std`.                                                          | Useful for uneven brightness, but it can include texture; verify the overlay.                                                                           |
| `ag_coverage_adaptive_block_size` = `151`     | Neighborhood size for the adaptive threshold; the code enforces an odd value of at least 15.                   | Smaller reacts more locally and can follow noise. Larger is smoother and more global.                                                                   |
| `ag_coverage_adaptive_k_std` = `2.0`          | Strictness of the adaptive threshold in local standard deviations.                                             | Higher gives fewer coverage pixels. Lower gives more coverage an more noise.                                                                            |
| `ag_coverage_min_object_size` = `9`           | Minimum component area in the secondary coverage mask.                                                         | Decrease for small Ag dots. Increase to suppress grain.                                                                                                 |
| `ag_coverage_closing_radius` = `0`            | Closing radius for the secondary coverage mask; the intentional default is disabled.                           | Increasing it joins nearby Ag pixels and usually raises coverage. Too high a value fills real gaps.                                                     |
| `ag_coverage_use_union_with_count` = `true`   | Adds every count-mask pixel to the coverage mask.                                                              | `true` ensures count candidates remain in coverage but may raise the coverage value. `false` keeps masks independent.                                   |
| `default_show_scale` = `true`                 | Initial scale-bar visibility.                                                                                  | Display only.                                                                                                                                           |
| `default_show_bead_boundary` = `true`         | Initial bead-boundary visibility.                                                                              | Grenn beads are included in summaries; red beads fail global sphere filters.                                                                            |
| `default_show_diameter_lines` = `true`        | Initial visibility of bead size crosses.                                                                       | Display only.                                                                                                                                           |
| `default_show_ag_boundary` = `true`           | Initial visibility of the red coverage-mask boundary.                                                          | Display only.                                                                                                                                           |
| `default_show_ag_count_boundary` = `false`    | Initial visibility of the yellow count-mask boundary.                                                          | Enable while comparing count and coverage segmentation.                                                                                                 |
| `default_show_ag_peaks` = `false`             | Initial visibility of cyan peaks used for Ag counting.                                                         | Useful when tuning `count_min_distance` and `count_thr_rel`                                                                                             |

# 11. Configuration reference: `tem_particle_viewer_config.json`

This viewer detects **dark TEM particles on a brighter background**, optionally separates touching regions,
measures boundary-constrained axes, and reports a size distribution.

| Parameter                             | What it controls                                                                                   | Tuning advice / common problems                                                                                                             |
|---------------------------------------|----------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------|
| `strip_rows` = `null`                 | Fixed number of bottom rows to remove. When not `null`, this overrides automatic footer detection. | Most reliable for data with a constant footer height.                                                                                       |
| `footer_tail_rows` = `160`            | Number of bottom rows searched during automatic dark-footer detection.                             | Increase when the footer begins higher. Ignored when fixed `strip_rows` is used.                                                            |
| `dark_footer_k_mad` = `6.0`           | Sensitivity of dark-footer detection.                                                              | Lower is more sensitive. Increase if real image content is cropped.                                                                         |
| `dark_footer_min_run` = `12`          | Minimum number of consecutive dark rows required.                                                  | Decrease if the footer is missed. Increase to avoid mistaking a dark cluster at the bottom for the footer.                                  |
| `display_percentiles` = `[1.0, 99.5]` | Contrast scaling of the original TEM image for display only.                                       | Does not affect segmentation, which uses the loaded grayscale image.                                                                        |
| `feature_percentiles` = `[1.0, 99.5]` | Contrast scaling of the feature map for display only.                                              | Adjust if the feature panel is nearly black or saturated.                                                                                   |
| `pixel_size_nm` = `null`              | Direct calibration in nm/pixel. Takes precedence over `fov_nm`.                                    | Use when precisely known. An incorrect value scales every hysical dimension.                                                                |
| `fov_nm` = `835.0`                    | Image field-of-view width in nm; pixel size is FOV divided by image width.                         | Enter the value for the whole image width. Removing the bottom footer does not change width.                                                |
| `detector` = `"sauvola"`              | Segmentation method: `sauvola`, `dog`, or `invert_tophat_otsu`.                                    | `sauvola` is currently recommended for dark TEM particles on uneven background. Parameters of inactive methods are ignored.                 |
| `dog_sigma_small` = `1.2`             | Fine Gaussian sigma for DoG.                                                                       | Used only with `detector="dog"`. Smaller values detect smaller structures and more noise.                                                   |
| `dog_sigma_large` = `6.0`             | Coarse Gaussian sigma for DoG.                                                                     | Used only for DoG and must exceed the small sigma.                                                                                          |
| `dog_foreground_percentile` = `99.0`  | Percentile threshold of the DoG feature map.                                                       | Higher retains fewer, stronger regions. Lower retains more candidates and noise.                                                            |
| `intensity_percentile_dark` = `35.0`  | DoG candidates must be darker than this percentile of the original image.                          | Higher accepts more and brighter structures. Lower is stricter about darkness.                                                              |
| `tophat_radius` = `7`                 | Radius of the white top-hat applied to the inverted image.                                         | Used only for `invert_tophat_otsu`. It must match particle scale; too small a radius detects texture.                                       |
| `sauvola_window_size` = `151`         | Sauvola local-window size; the code requires an odd number and minimum 15.                         | Larger gives smoother local background and suits large particles but can merge clusters. Smaller is more local and more texture-sensitive.  |
| `sauvola_k` = `0.2`                   | Strictness of the Sauvola threshold for dark objects.                                              | Higher values usually make segmentation stricter: fewer pixels and less merging. Lower values retain fainter particles and more background. |
| `closing_radius` = `3`                | Closing radius of the candidate mask.                                                              | Increase to fill holes, but note that neighboring particles may merge. Reduce to 0 when mergin occurs.                                      |
| `opening_radius` = `2`                | Opening radius of the candidate mask.                                                              | Increase to remove noise and thin bridges. Too large value removes small particles.                                                         |
| `min_area_px` = `500`                 | Minimum particle area in pixels, used during cleanup and classification.                           | Decrease for small particles. Increase to reject noise and tiny false regions.                                                              |
| `max_area_px` = `null`                | Optional maximum area; `null` disables the upper limmit.                                           | Use to flag unsplit large clusters as outliers. It rejects them but does not split them.                                                    |
| `max_anisotropy_ratio` = `3.0`        | Maximum ratio of measured major/minor boundary-constrained chord lengths.                          | Increase for elongated or irregular particles. Decrease for stricter shape filtering.                                                       |
| `min_solidity` = `0.2`                | Minimum particle compactness.                                                                      | Decrease for lobed valid particles. Increase to reject clusters and concave regions.                                                        |
| `split_touching` = `true`             | Enables watershed splitting of touching particles.                                                 | Disable when single particles are over-split. Enable when neighbors remain merged.                                                          |
| `split_min_distance` = `5`            | Minimum separation between watershed markers.                                                      | Lower permits more markers and more aggressive splitting. Higher produces fewer splits.                                                     |
| `split_threshold_rel` = `0.15`        | Minimum distance peak as a fraction of the global distance-map maximum.                            | Lower accepts weaker centers and more splits. Higher is more conservative. The threshold is global for the image.                           |
| `split_exclude_border` = `false`      | Whether the peak detector excludes markers near the mask/image border.                             | `false` permits splitting near the border. `true` may reduce false border markers but can lose edge particles.                              |
| `measurement_mode` = `"mask_chords"`  | Method used for displayed particle axes.                                                           | Currently only `mask_chords` is implemented; other values fall back with a warning.                                                         |
| `measure_step_px` = `0.5`             | Sampling step while tracing each axis inside the particle mask.                                    | Smaller is more precise but slower. 0.5 px is a reasonable compromise.                                                                      |
| `histogram_metric` = `"mean_axes"`    | Primary histogram metric: `mean_axes`, `eq_diameter`, `major_axis`, or `minor_axis`.               | `mean_axes` averages the two boundary-constrained axes and is suitable for mildly anisotropic particles.                                    |
| `major_axis_color` = `"cyan"`         | Matplotlib color used for the major axis and its label.                                            | Any Matplotlib color is valid, for example `cyan`, `lime`, or `#00ffff`.                                                                    |
| `minor_axis_color` = `"orange"`       | Color used for the minor axis and its label.                                                       | Choose a color that contrasts with the major axis and image.                                                                                |
| `default_show_scale` = `true`         | Initial scale-bar visibility.                                                                      | Display only.                                                                                                                               |
| `default_show_boundaries` = `true`    | Initial visibility of green/red boundaries.                                                        | Green means valid; red means rejected.                                                                                                      |
| `default_show_measures` = `false`     | Initial visibility of axes and labels.                                                             | For dense clusters, `false` is recommended; enable with the checkbox for inspection.                                                        |
| `default_show_histogram` = `true`     | Initial histogram visbility in the fourth panel.                                                   | If `false`, only text statistics are shown.                                                                                                 |

# 12. Troubleshooting

## 12.1 General problems

| Problem                                          | Likely cause                                                               | Solution                                                                                                        |
|--------------------------------------------------|----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------|
| `python` or `py` is not recognized               | Python is missing or not in PATH                                           | Install Python with Add Python to PATH, then reopen the terminal.                                               |
| `git` is not recognized                          | Git for Windows is missing or not in PATH                                  | Install Git fro Windows and reopen the terminal.                                                                |
| `ModuleNotFoundError`                            | `.venv` is inactive or a package is missing                                | Activate `.venv`, then run `python -m pip install -r requirements.txt`.                                         |
| `JSONDecodeError`                                | Missing/extra comma, comment, wrong quotes, or invalid boolean/null syntax | Open the JSON configuration file and fix sytax. Use `true`, `false`, and `null` (or check with ChatGPT).        |
| `FileNotFoundError`                              | Incorrect `folder` or `file` path                                          | Verify the path, use `/`, and run from the project folder.                                                      |
| Git pull says local changes would be overwritten | A tracked config or code file was edited                                   | Back up local configs, inspect `git status -sb`, restore the tracked file, then pull and reapply settings.      |
| Viewer reports no TIFF files                     | Files use uppercase `.TIF` or are in subfolders                            | Interactive SEM viewers read direct lowercase `*.tif` files. Rename the extension or choose the correct folder. |
| Scale bar is missing / units remain px           | Metadata are absent or mismatched                                          | For SEM check `<stem>-tif.hdr` and PixelSizeX/Y; for TEM set `pixel_size_nm` or `fov_nm`.                       |
| Results change strongly with magnification       | Parameters are expressen in pixels                                         | Use a separate config for each magnification/resolution.                                                        |
| First image is slow                              | Initial analysis and morphology can be expensive                           | Wait for completion; viewer results are cached during navigation. Batch scripts show progress.                  |
| Labels obscure the image                         | The image is dense                                                         | Disable the Measures checkbox or set the relevant `default_show_measures=false`.                                |

## 12.2 SEM bead viewer

| Symptom                                            | Parameters to try                                                                                                                                         |
|----------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| Small beads are missed                             | Decrease `min_object_area_px`, `min_diameter_px`, `opening_radius`, and possibly `intensity percentile`; consider smaller DoG sigmas.                     |
| Darker beads are missed                            | Decrease `intensity_percentile`; cautiously adjust `display_percentiles`.                                                                                 |
| Granular noise is detected                         | Increase `min_object_area_px`, `opening_radius`, `intensity_percentile`, or `dog_foreground_percentile`.                                                  |
| Two beads remain one object                        | Decrease `closing_radius`; slightly increase `opening_radius`; for stronger watershed splitting decrease `split_min_distance_px` and `split_threshold_px` |
| One bead splits into several                       | Increase `split_min_distance_px`, `split_threshold_px`, and `split_min_child_area_px`; decrease `split_max_peak_count` or disable watershed.              |
| Correct beads are red                              | Read rejection reasons in the information panel; relax `outlier_axis_ratio`, `min_solidity`, `max_eccentricity`, size limits, or `outlier_mad_zscore`.    |
| One mode of a bimodal mixture is marked as outlier | Set `global_size_outliers=false`.                                                                                                                         |
| Cropped beads enter statistics                     | Set `include_edge_candidates=false`; optionally increase `edge_touch_margin_px`                                                                           |

## 12.3 SEM coverage viewer

| Symptom                                    | Parameters to try                                                                                                                                                                  |
|--------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| No bead ROI is found                       | Decrease `min_bead_area_px`, `min_roi_eq_diameter_px`, and `min_roi_solidity`; increase `max_roi_anisotropy_ratio`; inspect `norm_percentiles` and keep `bead_morph_fallback=true` |
| ROI extends beyond the bead                | Increase `bead_morph_erode_radius_px`, reduce fallback dilation, or adjust bead opening/closing.                                                                                   |
| Two beads remain connected                 | Decrease `split_min_distance_px` and `split_peak_threshold_rel`; lower split-trigger thresholds; increase `split_max_peaks`.                                                       |
| One bead is over-split                     | Increase `split_min_distance_px` and `split_peak_threshold_rel`; reduce `split_max_peaks`; increase `split_min_child_area_ratio`.                                                  |
| Primary Ag mask misses particles           | Decrease `ag_mask_threshold_rel`, `ag_min_object_size`, or `ag_opening_radius`; inspect the feature scale selected by `ag_tophat_radius`.                                           |
| Primary Ag mask includes weak texture      | Increase `ag_mask_threshold_rel`, `ag_min_object_size`, or `ag_opening_radius`; compare log and linear intensity on sparse/noisy images.                                            |
| Ag peak count is too low                   | Decrease `count_thr_rel` or `count_min_distance`; display `Ag peaks` while tuning. These controls do not enlarge the primary mask.                                                  |
| Ag peak count is too high                  | Increase `count_thr_rel` or `count_min_distance`; display `Ag peaks` while tuning.                                                                                                |
| Coverage is too low                        | With secondary coverage, decrease `ag_coverage_threshold_rel`, `ag_coverage_adaptive_k_std`, or `ag_coverage_min_object_size`; consider adaptive threshold or union.               |
| Coverage is too high / texture is included | Increase `ag_coverage_threshold_rel`, `ag_coverage_adaptive_k_std`, and `ag_coverage_min_object_size`; decrease closing; consider disabling adaptive threshold or union.           |
| Bright bead rim is classified as Ag        | Increase `ag_erode_bead_radius`, while checking that true edge particles are not lost.                                                                                             |
| Changing `ag_coverage_*` has no effect     | `ag_enable_secondary_coverage` is `false`; enable it or remember coverage is then derived from the count mask.                                                                     |

## 12.4 TEM viewer

| Symptom                                   | Parameters to try                                                                                                                                                          |
|-------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Small/faint particles are missed          | Decrease `min_area_px` and `opening_radius`; for Sauvola decrease `sauvola_k` and possibly reduce the window.                                                              |
| Background texture is detected            | Increase `min_areapx`, `opening_radius`, or `sauvola_k`; optionally enlarge `sauvola_window_size`.                                                                         |
| Neighboring particles merge               | Decrease `closing_radius`, cautiously increase `opening_radius`, and increase `sauvola_k`; for stronger splitting decrease `split_min_distance` and `split_threshold_rel`. |
| One particle splits into several          | Increase `split_min_distance` and `split_threshold_rel`, or set `split_touching=false`.                                                                                    |
| Correct elongated particles are red       | Increase `max_anisotropy_ratio`.                                                                                                                                           |
| Correct irregular particles are red       | Decrease `min_solidity`.                                                                                                                                                   |
| A large unsplit cluster enters statistics | Set `max_area_px` and tune the splitter. The upper limit rejects a region but does not split it.                                                                           |
| Footer remains in the image               | Set fixed `strip_rows`, or increase `footer_tail_rows`; decrease `dark_footer_k_mad`/`dark_footer_min_run` for more sensitive detection.                                   |
| Image is cropped too high                 | Increase `dark_footer_k_mad` or `dark_footer_min_run`; fixed correct `strip_rows` is most reliable.                                                                        |
| Sizes in nm are wrong                     | Verify `pixel_size_nm`/`fov_nm`; direct pixel size has priority. Confirm the FOV refers to the full image width.                                                           |

# 13. Current behavior and limitations

- Interactive SEM viewers search only for lowercase `.tif`.
- Interactive SEM viewers do not recurse into subfolders. SEM batch processing does recurse and treats each folder containing TIFF files as one sample.
- TEM viewer first searches the selected folder directly; if no images are found, it uses recursive search.
- `peak_min_distance_px`, `peak_threshold_px`, and `boundary_linewidth` in the bead config are currently reserved and have no effect.
- `summary_json_path` behaves somewhat differently between viewers; use batch scripts for routine exports.
- SEM Ag count is the number of local maxima in the count feature map. Coverage is Ag-mask pixels divided by bead-ROI pixels in the 2D projection.
- `sphere_ag_count_est` is currently twice the projected count and is a model estimate, not a direct 3D measurement.

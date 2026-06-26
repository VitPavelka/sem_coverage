# SEM/TEM Particle Toolkit
## User manual and configuration reference

**Documentation status:** 25 June 2026
**Repository:** `https://github.com/VitPavelka/sem_coverage`
**Default Git branch:** `master`

This manual is intended for users with little or no Python experience. 
It covers installation, updates, interactive viewers, batch export, 
every current configuration parameter, and some troubleshooting.

**Important:** This software is still under development. 
Every result is an automated image-analysis *estimate*.
Before using statistics, inspect overlays from at least several representative images.
Pixel-based settings depend on magnification, resolution, and particle size.
At first, try to keep separate configuration presets for different acquisition protocols.

## 1. Included tools

| Task                                           | Launcher                        | Configuration                     | Input                                   |
|------------------------------------------------|---------------------------------|-----------------------------------|-----------------------------------------|
| Size analysis of bright SEM beads              | `run_bead_viewer.py`            | `sem_bead_viewer_config.json`     | `.tif` plus optional paired `.hdr`      |
| Ag coverage and Ag-count estimate on SEM beads | `run_sem_coverage_viewer.py`    | `sem_coverage_viewer_config.json` | `.tif` plus optional paired `.hdr`      |
| Size analysis of dark TEM nanoparticles        | `run_tem_particle_viewer.py`    | `tem_particle_viewer_config.json` | `.png`, `.jpg`, `.jpeg`                 |
| Batch SEM export                               | `batch_export_protocols.py`     | the two SEM configs as templates  | directory tree containing `.tif` files  |
| Batch TEM export                               | `batch_export_tem_protocols.py` | TEM config as a template          | directory tree containing PNG/JPG files |
| Additional SEM CSV/histogram export            | `export_output_summaries.py`    | command-line arguments            | previously generated SEM JSON files     |

## 2. Installation

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

## 3. Editing JSON configuration files

Open the JSON files in Notepad, Notepad++, or VS Code.
Routine users normally change only paths and values under `viewer`.

### JSON rules

- Text must use double quotes: `"text""`.
- Boolean values are lowercase: `true`/`false`.
- An empty value is `null`.
- Do not place a comma after the final item in an object.
- JSON does not support comments.
- On Windows, prefer `/`: `"C:/Data/TEM"`. Alternatively, escape backslashes: `"C:\\Data\\TEM"`.

### Common top-level fields

| Field               | Meaning                                                                                          | Recommendation                                                        |
|---------------------|--------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------|
| `folder`            | Folder containing input images.                                                                  | The past must exist.                                                  |
| `file`              | Optional single file in SEM coverage and TEM configurations (`null` means all supported images). | Enter a file name or path. SEM bead viewer has no `file` field.       |
| `summary_json_path` | Optional path for a summary JSON. `null` is the simplest interactive mode.                       | Routine users should noramlly leave it `null` and use batch scripts.  |

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

## 4. Running the interactive viewers

Run commands from the project folder while `.venv` is active.

### 4.1 SEM bead size

```bat
python run_bead_viewer.py
```

The viewer loads lowercase **`.tif`** files directly inside `folder`.
Left/right arrows change images.
Checkboxes control the scale bar, boundaries, and size labels.

### 4.2 SEM Ag coverage

```bat
python run_sem_coverage_viewer.py
```

Left/right arrows change images. Up/down arrows switch diagnostic layers:

- `display` - original image scaled for display,
- `norm` - normalization used for bead detection,
- `bead_raw` - initial ROI candidates,
- `bead_refined` - final bead ROIs,
- `ag_count_feature` - top-hat response used for Ag peaks,
- `ag_coverage_feature` - response used for the coverage mask.

### 4.3 TEM particle size

```bat
python run_tem_particle_viewer.py
```

Left/right arrows change images. The window shows the original image, detector feature map,
overlay, and histogram/statistics.

## 5. Input files and physical calibration

### 5.1 SEM TIFF and HDR

For `sample.tif`, the SEM viewers look for metadata in:

```text
sample-tif.hdr
```

The HDR file supplies pixel size, magnification, and measurement information.
Analysis can run without HDR, but sizes and scale bars may remain in pixels.

### 5.2 TEM

The TEM loader currently supports only grayscale, RGB, and RGBA PNG/JPG images.
Enter either `pixel_size_nm` or `fov_nm` for physical sizes.
A direct `pixel_size_nm` value take precedence.

## 6. Batch exports

### 6.1 SEM bead and coverage export

Beads only:

```bat
python batch_export_protolols.py --bead-root "C:/Data/SEM/size" --output-dir "C:/Data/results_sem" --clean
```

Coverage only:

```bat
python batch_export_protolols.py --coverage-root "C:/Data/SEM/coverage" --output-dir "C:/Data/results_sem" --clean
```

Both tasks:

```bat
python batch_export_protolols.py --bead-root "C:/Data/SEM/size" --coverage-root "C:/Data/SEM/coverage" --output-dir "C:/Data/results_sem" --clean
```

Optional arguments:

- `--bead-config FILE` - alternate bead-analysis template,
- `--coverage-config FILE` - alternate coverage template,
- `--clean` - remove previous JSON/CSV/PNG outputs in the target structure before running,
- `--no-export` - crete per-sample JSON and overlay PNG files but skip final CSV/histogram generation.

Typical outputs include per-sample JSON, `size_png`, `coverage_png`, `bead_global_summaries.csv`, 
`coverage_global_summaries.csv`, and `bead_histograms`.

### 6.2 TEM batch export

```bat
python batch_export_tem_protocols.py --root "C:/Data/TEM" --output-dir "C:/Data/results_tem" --clean
```

Optional arguments:

- `--config FILE` - alternate TEM template,
- `--clean` - delete old TEM outputs,
- `--no-csv` - skip CSV creation,
- `--no-histograms` - skip histogram creation.

Outputs include per-sample and global JSON, per-image JSON, overlay PNG files, two CSV summaries, and histograms.

### 6.3 Regenerating SEM summaries from existing JSON files

```bat
python export_output_summaries.py --outputs-dir "C:/Data/results_sem"
```

Use this when SEM JSON files already exist, and you want to regenerate the CSV/histogram outputs.

## 7. Recommended tuning workflow

1. Verify footer cropping and physical calibration first.
2. Tune using at least several representative images, not one ideal image.
3. Tune coarse segmentation first, touching-object splitting second, and outlier filters last.
4. Change one parameter group at a time, restart the viewer, and compare overlays. 
5. Keep a separate configuration for each image type and/or acquisition protocol.
6. Before batch export, manually review several images from every sample group.

# 8. Configuration reference: `sem_bead_viewer_config.json`

This tool detects **bright beads on a darker background**, measures x/y dimensions, and classifies candidates as valid (green) or rejected (red).

| Parameter (current value)              | What it controls                                                                                    | Tuning advice / common problems                                                                                                            |
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

# 9. Configuration reference: `sem_coverage_viewer_config.json`

The viewer first identifies bead ROIs and then estimates projected Ag coverage and Ag peak count.

## 9.1 `viewer.analyzer`

| Parameter (current value)             | What it controls                                                                             | Tuning advice / common problems                                                                                               |
|---------------------------------------|----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------|
| `infobar_tail_rows` = `320`           | Number of bottom rows searched for the bright SEM information bar.                           | Increase if the bar starts higher; decrease for short images.                                                                 |
| `infobar_k_mad` = `8.0`               | Sensitivity of information-bar detection.                                                    | Lower is more sensitive; higher is more conservative.                                                                         |
| `infobar_min_run` = `10`              | Minimum continuous height of the bar.                                                        | Decrease for a short bar; increase to avoid false cropping.                                                                   |
| `norm_percentiles` = `[2.0, 98.0]`    | Percentile normalization used for bead detection, not directly for the Ag top-hat operation. | A narrower range enhances the bead but may saturate. Change cautiously if ROI detection fails.                                |
| `bead_blur_sigma` = `2.0`             | Gaussian smoothing before Otsu segmentation of the bead.                                     | Higher values suppress texture and smooth the mask, but can merge neighbors or lose small beads.                              |
| `bead_closing_radius` = `5`           | Closing radius for the bead mask.                                                            | Increase to fill gaps; decrease when neighboring objects merge.                                                               |
| `bead_opening_radius` = `3`           | Opening radius for the bead mask.                                                            | Increase to remove noise and thin bridges, but note that small beads may shrink.                                              |
| `bead_hole_area` = `5000`             | Fills holes in the bead mask below this area.                                                | Increase if a correct bead contains a large unfilled hole. Excessive values can fill real cavities.                           |
| `ag_tophat_radius` = `9`              | White top-hat radius used for small bright Ag structures and peak counting.                  | Smaller values favor small sharp particles. Larger values capture broader structures but may respond to background variation. |
| `ag_min_object_size` = `5`            | Minimum Ag candidate area in pixels.                                                         | Decrease when small nanoparticles are missed. Increase to suppress grain and isolated pixels.                                 |
| `ag_erode_bead_radius` = `2`          | Erosion of the bead ROI before Ag detection, excluding the bead edge.                        | Increase to suppress a false bright rim. Decrease to include Ag close to the edge.                                            |
| `ag_use_log` = `false`                | Applies `log1p` to original intensities before top-hat filtering.                            | Useful for a very broad dynamic range. Disable if compression weakens Ag-to-background differences.                           |
| `count_min_distance` = `5`            | Minimum separation between local maxima counted as Ag particles.                             | Increase to avoid multiple counts per particle. Decrease for tightly packed particles.                                        |
| `count_thr_rel` = `1.0`               | Multiplier applied to the automatic threshold for accepting a peak.                          | Higher gives fewer, stronger peaks. Lower gives more particles and more possible false peaks.                                 |
| `display_percentiles` = `[0.5, 99.5]` | Contrast scaling used only for display.                                                      | Does not change numerical Ag segmentation; change only for viewer readability.                                                |

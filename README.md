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

| Parameter (current value)   | What it controls                                                           | Tuning advice / common problems                                                                    |
|-----------------------------|----------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------|
| `infobar_tail_rows` = `320` | Number of bottom rows searched for the bright SEM information bar.         | Increase if the bar starts higher. Decrease for very short images or to limit the search area.     |
| `infobar_k_mad` = `8.0`     | Sensitivity of automatic bright-bar detection based on the median and MAD. | Lower values crop more sensitively but can cut image content. Higher values are more conservative. |
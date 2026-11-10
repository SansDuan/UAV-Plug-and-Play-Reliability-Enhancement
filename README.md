# UAV Plug-and-Play Reliability Enhancement

AnyVisLoc matching data, data-generation utilities, and evaluation results for
*A Plug-and-Play Lightweight Reliability Enhancement Network for Zero-Shot
Generalization in Scene-Matching Visual Navigation*.

## Dataset

[Download the matching data from Google Drive](https://drive.google.com/file/d/1qs_mPLWHgElHm9K3kmRLrSdgrP_U2HzR/view?usp=sharing)

The matching archive provides precomputed UAV-to-reference correspondences.
Frame directories use identifiers such as `Scene_21_L21_0001_aerial`; NPZ files
use matcher names as filenames. `data/methods.txt` lists the 50 point-matching
methods used for point-correspondence processing.

Reference crops are selected around the ground-truth location and pre-rotated
using camera heading. The evaluation studies correspondence reliability and
geometric localization within these local reference regions.

## AnyVisLoc Results

The evaluation uses Scenes 21-24, with 1,728 shared frames and ten target
matchers, totaling 17,280 matcher-frame instances.

| Method | Coverage (%) | Median (m) | RMSE (m) | P95 (m) | SR@5 (%) | F30 (%) |
|---|---:|---:|---:|---:|---:|---:|
| Raw PnP | 68.67 | 40.92 | 99.23 | 139.75 | 4.25 | 68.83 |
| MAGSAC++ | 95.17 | 28.55 | 69.67 | 120.48 | 19.75 | 43.63 |
| CLNet | 95.17 | 36.80 | 71.84 | 121.71 | 3.50 | 56.03 |
| OANet | 95.17 | 35.50 | 71.67 | 122.05 | 4.18 | 52.48 |
| MHGC | 95.17 | 21.99 | 48.26 | 83.85 | 19.88 | 39.18 |
| CMCGC | 95.17 | 21.75 | 48.18 | 83.75 | 21.17 | 38.99 |

Saved query-level errors are in `tables/query_errors/`. Recompute the table and
check its values using Python 3.10+:

```bash
python3 code/reproduce_main_table.py
python3 code/test_release.py
```

Median, RMSE and P95 are computed over valid outputs per matcher and then
macro-averaged. Coverage, SR@5 and F30 account for every query. The saved-error
convention treats nonfinite values and errors >= 9999 as invalid outputs.

## Data Generation

Install the preprocessing and matching dependencies:

```bash
pip install -r requirements.txt
```

Prepare UAV/reference pairs with heading pre-rotation and DSM crops:

```bash
python3 code/preprocessing/anyvisloc_precrop_scene.py \
  --zip /path/to/AnyVisLoc/Scene_21.zip \
  --out-root /path/to/pairs --mode aerial
```

The matcher worker uses IMCUI and local upstream checkpoints. A pair CSV has the
columns `run_name,image0,image1,out`; use absolute paths for images and output
directories. Run a matcher with:

```bash
python3 code/matching/run_matching.py \
  --method sift --pairs-csv pairs.csv \
  --checkpoints /path/to/checkpoints --resume
```

The worker reuses one model across the listed pairs, saves correspondences as
NPZ files, and uses CUDA when available. The launcher sets CPU thread counts to
one. Matcher-specific upstream dependencies and checkpoints follow their
respective projects' installation instructions.

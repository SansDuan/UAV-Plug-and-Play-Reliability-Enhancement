# UAV Plug-and-Play Reliability Enhancement

Public data and evaluation results for *A Plug-and-Play Lightweight Reliability Enhancement Network for Zero-Shot Generalization in Scene-Matching Visual Navigation*.

**Release status:** Matching data and evaluation results are available. Source code, reproduction scripts, and model weights are temporarily withheld from the public release.

## Dataset

[Download the matching data from Google Drive](https://drive.google.com/file/d/1qs_mPLWHgElHm9K3kmRLrSdgrP_U2HzR/view?usp=sharing)

The matching archive contains precomputed UAV-to-reference correspondences. Frame directories use identifiers such as `Scene_21_L21_0001_aerial`, and NPZ files are named after the matching method. See [data/methods.txt](data/methods.txt) for the 50 point-matching methods used for correspondence processing.

Reference crops are selected around the ground-truth location and pre-rotated using camera heading. Evaluation examines correspondence reliability and geometric localization within these local reference regions.

## AnyVisLoc Evaluation Results

The evaluation uses Scenes 21–24, with 1,728 shared frames and ten target matchers, totaling 17,280 matcher-frame instances.

| Method | Coverage (%) | Median (m) | RMSE (m) | P95 (m) | SR@5 (%) | F30 (%) |
|---|---:|---:|---:|---:|---:|---:|
| Raw PnP | 68.67 | 40.92 | 99.23 | 139.75 | 4.25 | 68.83 |
| MAGSAC++ | 95.17 | 28.55 | 69.67 | 120.48 | 19.75 | 43.63 |
| CLNet | 95.17 | 36.80 | 71.84 | 121.71 | 3.50 | 56.03 |
| OANet | 95.17 | 35.50 | 71.67 | 122.05 | 4.18 | 52.48 |
| MHGC | 95.17 | 21.99 | 48.26 | 83.85 | 19.88 | 39.18 |
| CMCGC | 95.17 | 21.75 | 48.18 | 83.75 | 21.17 | 38.99 |

Saved query-level errors are in [tables/query_errors](tables/query_errors). Aggregate numbers and verification metadata are in [tables/main_table_verified.csv](tables/main_table_verified.csv) and [tables/paper_value_checks.json](tables/paper_value_checks.json).

Median, RMSE, and P95 are computed over valid outputs per matcher and then macro-averaged. Coverage, SR@5, and F30 account for all queries. Nonfinite outputs and errors >= 9999 are treated as invalid.

#!/usr/bin/env python3
"""Run one IMCUI method over many already-cropped image pairs."""

from __future__ import annotations

import argparse
import copy
import csv
import gc
import json
import os
import time
import traceback
from pathlib import Path

import cv2
import numpy as np
import torch

from imcui_eval_pair import (
    DEFAULT_CHECKPOINTS,
    completed_method_outputs,
    load_manifest,
    load_matcher_zoo,
    prediction_arrays,
    safe_method_name,
    summarize_prediction,
)


ROOT = Path(__file__).resolve().parents[1]


class PersistentMatcher:
    def __init__(self, method: str) -> None:
        if not hasattr(np, "int"):
            np.int = int
        if not hasattr(np, "bool"):
            np.bool = bool

        from imcui.hloc import extract_features, extractors, match_dense, match_features, matchers
        from imcui.hloc.utils.base_model import dynamic_load

        self.extract_features = extract_features
        self.match_dense = match_dense
        self.match_features = match_features
        self.method = method
        self.zoo = load_matcher_zoo()
        self.raw = self.zoo[method]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.set_grad_enabled(False)

        if self.raw["standalone"]:
            self.conf = copy.deepcopy(match_dense.confs[self.raw["matcher"]])
            Model = dynamic_load(matchers, self.conf["model"]["name"])
            self.model = Model(self.conf["model"]).eval().to(self.device)
            self.extractor = None
            self.matcher = None
            self.feature_conf = None
            self.matcher_conf = self.conf
        else:
            self.feature_conf = copy.deepcopy(extract_features.confs[self.raw["feature"]])
            self.matcher_conf = copy.deepcopy(match_features.confs[self.raw["matcher"]])
            if self.raw.get("feature") == "sift":
                max_kpts = int(
                    os.environ.get(
                        "IMCUI_SIFT_LIGHTGLUE_MAX_KEYPOINTS" if method == "sift+lightglue" else "IMCUI_SIFT_MAX_KEYPOINTS",
                        "2048",
                    )
                )
                resize_max = int(os.environ.get("IMCUI_SIFT_RESIZE_MAX", "1024"))
                self.feature_conf.setdefault("model", {})["max_keypoints"] = max_kpts
                self.feature_conf.setdefault("preprocessing", {})["resize_max"] = resize_max
            if self.raw.get("feature") == "rdd":
                self.feature_conf.setdefault("model", {})["max_keypoints"] = int(os.environ.get("IMCUI_RDD_MAX_KEYPOINTS", "4096"))
                self.feature_conf.setdefault("preprocessing", {})["resize_max"] = int(os.environ.get("IMCUI_RDD_RESIZE_MAX", "1024"))
            Extractor = dynamic_load(extractors, self.feature_conf["model"]["name"])
            Matcher = dynamic_load(matchers, self.matcher_conf["model"]["name"])
            self.extractor = Extractor(self.feature_conf["model"]).eval().to(self.device)
            self.matcher = Matcher(self.matcher_conf["model"]).eval().to(self.device)
            self.model = None

    def match(self, image0_path: Path, image1_path: Path) -> tuple[dict, dict]:
        image0_bgr = cv2.imread(str(image0_path), cv2.IMREAD_COLOR)
        image1_bgr = cv2.imread(str(image1_path), cv2.IMREAD_COLOR)
        if image0_bgr is None or image1_bgr is None:
            raise FileNotFoundError(f"Failed to read image pair: {image0_path}, {image1_path}")
        image0 = cv2.cvtColor(image0_bgr, cv2.COLOR_BGR2RGB)
        image1 = cv2.cvtColor(image1_bgr, cv2.COLOR_BGR2RGB)

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        result = {
            "method": self.method,
            "standalone": bool(self.raw["standalone"]),
            "device": self.device,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "image0": str(image0_path),
            "image1": str(image1_path),
        }
        if self.raw["standalone"]:
            result["matcher_conf"] = self.matcher_conf
            with torch.inference_mode():
                pred = self.match_dense.match_images(self.model, image0, image1, self.matcher_conf["preprocessing"], device=self.device)
        else:
            result["feature_conf"] = self.feature_conf
            result["matcher_conf"] = self.matcher_conf
            with torch.inference_mode():
                pred0 = self.extract_features.extract(self.extractor, image0, self.feature_conf["preprocessing"])
                pred1 = self.extract_features.extract(self.extractor, image1, self.feature_conf["preprocessing"])
                try:
                    pred = self.match_features.match_images(self.matcher, pred0, pred1)
                except IndexError as exc:
                    empty_feature_error = (
                        "too many indices" in str(exc)
                        or "Expected reduction dim" in str(exc)
                        or "non-zero size" in str(exc)
                    )
                    if not empty_feature_error:
                        raise
                    pred = {
                        "mkeypoints0_orig": np.empty((0, 2), dtype=np.float32),
                        "mkeypoints1_orig": np.empty((0, 2), dtype=np.float32),
                        "mkeypoints0": np.empty((0, 2), dtype=np.float32),
                        "mkeypoints1": np.empty((0, 2), dtype=np.float32),
                        "mconf": np.empty((0,), dtype=np.float32),
                    }
                    result["empty_match_fallback"] = str(exc)
        if torch.cuda.is_available():
            result["cuda_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
            result["cuda_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 3)
        return result, pred


def append_manifest(out_dir: Path, record: dict) -> None:
    manifest_path = out_dir / "manifest.json"
    manifest = load_manifest(manifest_path)
    manifest = [row for row in manifest if row.get("method") != record.get("method")]
    manifest.append(record)
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(manifest_path)


def write_json_atomic(path: Path, payload: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    tmp.replace(path)


def write_npz_atomic(path: Path, arrays: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays)
    tmp.replace(path)


def load_pairs(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--pairs-csv", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    args = parser.parse_args()

    if os.environ.get("OPENCV_NUM_THREADS"):
        cv2.setNumThreads(int(os.environ["OPENCV_NUM_THREADS"]))

    method = args.method
    safe = safe_method_name(method)
    pairs = load_pairs(args.pairs_csv)
    matcher = PersistentMatcher(method)
    print(json.dumps({"method": method, "status": "model-loaded", "pairs": len(pairs)}, ensure_ascii=False), flush=True)

    failures = 0
    for idx, row in enumerate(pairs, 1):
        out_dir = Path(row["out"])
        result_path = out_dir / f"{safe}.json"
        npz_path = out_dir / f"{safe}.npz"
        if args.resume and completed_method_outputs(out_dir, method):
            print(json.dumps({"method": method, "run_name": row["run_name"], "status": "skipped-existing"}, ensure_ascii=False), flush=True)
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        record = {
            "method": method,
            "run_name": row["run_name"],
            "returncode": 1,
            "elapsed_sec": 0.0,
            "result": str(result_path),
            "npz": str(npz_path),
            "result_exists": False,
            "npz_exists": False,
            "batch_worker": True,
        }
        try:
            result, pred = matcher.match(Path(row["image0"]), Path(row["image1"]))
            arrays = prediction_arrays(pred)
            result["prediction"] = summarize_prediction(pred)
            result["saved_arrays"] = {key: list(value.shape) for key, value in arrays.items()}
            write_npz_atomic(npz_path, arrays)
            write_json_atomic(result_path, result)
            record.update(
                {
                    "returncode": 0,
                    "elapsed_sec": round(time.time() - start, 3),
                    "result_exists": result_path.exists(),
                    "npz_exists": npz_path.exists(),
                }
            )
        except Exception as exc:
            failures += 1
            record.update(
                {
                    "elapsed_sec": round(time.time() - start, 3),
                    "error": str(exc),
                    "traceback": traceback.format_exc(limit=12),
                }
            )
            err_path = out_dir / f"{safe}.error.json"
            write_json_atomic(err_path, record)
        append_manifest(out_dir, record)
        print(json.dumps({"idx": idx, **record}, ensure_ascii=False), flush=True)
        try:
            del arrays
        except NameError:
            pass
        try:
            del pred
        except NameError:
            pass
        try:
            del result
        except NameError:
            pass
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

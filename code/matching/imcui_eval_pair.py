#!/usr/bin/env python3
"""Run Image Matching WebUI methods on one image pair with original configs.

The parent mode runs each method in a separate child process so a heavy model can
fail without poisoning the rest of the benchmark.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import shutil
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHECKPOINTS = ROOT / "third_party" / "imcui_checkpoints_full"
DEFAULT_OUT = ROOT / "outputs" / "imcui_pair_eval"


def app_yaml_path() -> Path:
    spec = importlib.util.find_spec("imcui")
    if spec and spec.submodule_search_locations:
        candidate = Path(next(iter(spec.submodule_search_locations))) / "config" / "app.yaml"
        if candidate.exists():
            return candidate
    legacy = ROOT / ".conda-imcui" / "lib" / "python3.10" / "site-packages" / "imcui" / "config" / "app.yaml"
    if legacy.exists():
        return legacy
    raise FileNotFoundError("Cannot find imcui/config/app.yaml in the current Python environment.")


def safe_method_name(method: str) -> str:
    return method.replace("/", "_").replace(" ", "_")


def load_matcher_zoo() -> dict:
    with app_yaml_path().open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg["matcher_zoo"]


def prepare_env(checkpoints: Path) -> dict:
    env = os.environ.copy()
    env.setdefault("PYTHONNOUSERSITE", "1")
    env.setdefault("CUDA_MODULE_LOADING", "LAZY")
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128")
    env.setdefault("IMCUI_SKIP_PYCOLMAP", "1")
    env["IMCUI_CHECKPOINTS_DIR"] = str(checkpoints)
    env.setdefault("TORCH_HOME", str(checkpoints))
    env.setdefault("HF_HOME", str(checkpoints / "huggingface"))
    try:
        import certifi

        for key in ("SSL_CERT_FILE", "REQUESTS_CA_BUNDLE"):
            if not env.get(key) or not Path(env[key]).exists():
                env[key] = certifi.where()
    except Exception:
        pass
    return env


def read_gpu_mem_mb(pid: int) -> int:
    try:
        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            text=True,
            capture_output=True,
            timeout=5,
        )
    except Exception:
        return 0
    total = 0
    for line in proc.stdout.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 2:
            continue
        try:
            if int(parts[0]) == pid:
                total += int(parts[1])
        except ValueError:
            continue
    return total


def monitor_process(
    proc: subprocess.Popen,
    max_ram_gb: float | None,
    max_vram_gb: float | None,
    timeout_sec: float | None,
) -> dict:
    disable_ram_monitor = os.environ.get("IMCUI_DISABLE_RAM_MONITOR") == "1"
    ps = None if disable_ram_monitor else psutil.Process(proc.pid)
    peak_rss = 0
    peak_vram = 0
    killed = None
    start = time.time()
    terminate_at = None
    while proc.poll() is None:
        if ps is not None:
            try:
                rss = ps.memory_info().rss
                peak_rss = max(peak_rss, rss)
            except psutil.Error:
                pass
        vram = 0 if os.environ.get("IMCUI_DISABLE_VRAM_MONITOR") == "1" else read_gpu_mem_mb(proc.pid)
        peak_vram = max(peak_vram, vram)
        if max_ram_gb is not None and peak_rss > max_ram_gb * 1024**3:
            killed = f"RAM limit exceeded: {peak_rss / 1024**3:.2f} GB"
            terminate_process_tree(proc)
            terminate_at = terminate_at or time.time()
        if max_vram_gb is not None and peak_vram > max_vram_gb * 1024:
            killed = f"VRAM limit exceeded: {peak_vram / 1024:.2f} GB"
            terminate_process_tree(proc)
            terminate_at = terminate_at or time.time()
        if timeout_sec is not None and time.time() - start > timeout_sec:
            killed = f"timeout exceeded: {timeout_sec:.1f} sec"
            terminate_process_tree(proc)
            terminate_at = terminate_at or time.time()
        if terminate_at is not None and time.time() - terminate_at > 10:
            kill_process_tree(proc)
        time.sleep(0.5)
    if killed and proc.returncode is None:
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    return {
        "peak_ram_gb": round(peak_rss / 1024**3, 3),
        "peak_vram_gb": round(peak_vram / 1024, 3),
        "killed": killed,
    }


def terminate_process_tree(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except Exception:
        try:
            proc.terminate()
        except Exception:
            pass


def kill_process_tree(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass


def load_manifest(path: Path) -> list[dict]:
    if not path.exists():
        return []
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError:
        return []
    return data if isinstance(data, list) else []


def completed_method_outputs(out_dir: Path, method: str) -> bool:
    safe_name = safe_method_name(method)
    result_path = out_dir / f"{safe_name}.json"
    npz_path = out_dir / f"{safe_name}.npz"
    if not (result_path.is_file() and npz_path.is_file()):
        return False
    if result_path.stat().st_size <= 0 or npz_path.stat().st_size <= 0:
        return False
    try:
        with result_path.open("r", encoding="utf-8") as f:
            json.load(f)
    except Exception:
        return False
    return True


def run_parent(args: argparse.Namespace) -> int:
    zoo = load_matcher_zoo()
    methods = list(zoo.keys()) if args.all else args.method
    args.out.mkdir(parents=True, exist_ok=True)
    env = prepare_env(args.checkpoints)
    manifest_path = args.out / "manifest.json"
    manifest = load_manifest(manifest_path) if args.resume else []
    successful = {r.get("method") for r in manifest if r.get("returncode") == 0 and completed_method_outputs(args.out, str(r.get("method", "")))}
    if args.resume:
        successful.update(method for method in methods if completed_method_outputs(args.out, method))
    for method in methods:
        safe_name = safe_method_name(method)
        log_path = args.out / f"{safe_name}.log"
        result_path = args.out / f"{safe_name}.json"
        npz_path = args.out / f"{safe_name}.npz"
        if args.resume and method in successful and result_path.exists() and npz_path.exists():
            print(json.dumps({"method": method, "status": "skipped-existing", "result": str(result_path)}, ensure_ascii=False))
            continue
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--method",
            method,
            "--image0",
            str(args.image0),
            "--image1",
            str(args.image1),
            "--out",
            str(args.out),
            "--checkpoints",
            str(args.checkpoints),
        ]
        if os.environ.get("IMCUI_USE_GNU_TIMEOUT") == "1" and args.timeout_sec and shutil.which("timeout"):
            cmd = [
                "timeout",
                "--kill-after=10s",
                f"{int(args.timeout_sec)}s",
                *cmd,
            ]
        start = time.time()
        with log_path.open("w", encoding="utf-8") as log:
            proc = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            usage = monitor_process(proc, args.max_ram_gb, args.max_vram_gb, args.timeout_sec)
        elapsed = round(time.time() - start, 3)
        record = {
            "method": method,
            "returncode": proc.returncode,
            "elapsed_sec": elapsed,
            "log": str(log_path),
            "result": str(result_path),
            "npz": str(npz_path),
            "result_exists": result_path.exists(),
            "npz_exists": npz_path.exists(),
            **usage,
        }
        manifest = [r for r in manifest if r.get("method") != method]
        manifest.append(record)
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        print(json.dumps(record, ensure_ascii=False))
    return 0


def summarize_prediction(pred: dict) -> dict:
    summary = {"keys": sorted(pred.keys())}
    for key in ("mkeypoints0_orig", "keypoints0_orig", "keypoints0", "matches0"):
        if key in pred:
            try:
                summary[f"{key}_shape"] = list(pred[key].shape)
            except AttributeError:
                pass
    if "mconf" in pred:
        summary["num_matches"] = int(len(pred["mconf"]))
    elif "matches0" in pred:
        matches = pred["matches0"]
        summary["num_matches"] = int((matches >= 0).sum())
    elif "keypoints0" in pred:
        summary["num_matches"] = int(len(pred["keypoints0"]))
    return summary


def prediction_arrays(pred: dict) -> dict:
    arrays = {}
    skip_keys = {"image0", "image1", "image0_orig", "image1_orig"}
    for key, value in pred.items():
        if key in skip_keys or not hasattr(value, "shape"):
            continue
        try:
            import torch

            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
        except Exception:
            pass
        try:
            arrays[key] = np.asarray(value)
        except Exception:
            continue
    return arrays


def run_child(args: argparse.Namespace) -> int:
    import cv2
    import numpy as np
    import torch

    if not hasattr(np, "int"):
        np.int = int  # Compatibility for older third-party matcher code.
    if not hasattr(np, "bool"):
        np.bool = bool  # Compatibility for older third-party matcher code.

    from imcui.hloc import extract_features, match_dense, match_features, matchers, extractors
    from imcui.hloc.utils.base_model import dynamic_load

    method = args.method[0] if isinstance(args.method, list) else args.method
    zoo = load_matcher_zoo()
    raw = zoo[method]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image0 = cv2.cvtColor(cv2.imread(str(args.image0), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    image1 = cv2.cvtColor(cv2.imread(str(args.image1), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    if image0 is None or image1 is None:
        raise FileNotFoundError("Failed to read one of the input images.")

    torch.set_grad_enabled(False)
    result = {
        "method": method,
        "standalone": bool(raw["standalone"]),
        "device": device,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "image0": str(args.image0),
        "image1": str(args.image1),
    }
    if raw["standalone"]:
        conf = copy.deepcopy(match_dense.confs[raw["matcher"]])
        result["matcher_conf"] = conf
        Model = dynamic_load(matchers, conf["model"]["name"])
        model = Model(conf["model"]).eval().to(device)
        with torch.inference_mode():
            pred = match_dense.match_images(model, image0, image1, conf["preprocessing"], device=device)
    else:
        feature_conf = copy.deepcopy(extract_features.confs[raw["feature"]])
        matcher_conf = copy.deepcopy(match_features.confs[raw["matcher"]])
        if raw.get("feature") == "sift":
            max_kpts = int(
                os.environ.get(
                    "IMCUI_SIFT_LIGHTGLUE_MAX_KEYPOINTS" if method == "sift+lightglue" else "IMCUI_SIFT_MAX_KEYPOINTS",
                    "2048",
                )
            )
            resize_max = int(os.environ.get("IMCUI_SIFT_RESIZE_MAX", "1024"))
            feature_conf.setdefault("model", {})["max_keypoints"] = max_kpts
            feature_conf.setdefault("preprocessing", {})["resize_max"] = resize_max
            result["sift_max_keypoints"] = max_kpts
            result["sift_resize_max"] = resize_max
        result["feature_conf"] = feature_conf
        result["matcher_conf"] = matcher_conf
        Extractor = dynamic_load(extractors, feature_conf["model"]["name"])
        Matcher = dynamic_load(matchers, matcher_conf["model"]["name"])
        extractor = Extractor(feature_conf["model"]).eval().to(device)
        matcher = Matcher(matcher_conf["model"]).eval().to(device)
        with torch.inference_mode():
            pred0 = extract_features.extract(extractor, image0, feature_conf["preprocessing"])
            pred1 = extract_features.extract(extractor, image1, feature_conf["preprocessing"])
            try:
                pred = match_features.match_images(matcher, pred0, pred1)
            except IndexError as exc:
                empty_feature_error = (
                    "too many indices" in str(exc)
                    or "Expected reduction dim" in str(exc)
                    or "non-zero size" in str(exc)
                )
                if not empty_feature_error:
                    raise
                # Some sparse matchers crash while rescaling empty LightGlue matches.
                # Preserve the run as a valid zero-match result so dataset jobs continue.
                pred = {
                    "mkeypoints0_orig": np.empty((0, 2), dtype=np.float32),
                    "mkeypoints1_orig": np.empty((0, 2), dtype=np.float32),
                    "mkeypoints0": np.empty((0, 2), dtype=np.float32),
                    "mkeypoints1": np.empty((0, 2), dtype=np.float32),
                    "mconf": np.empty((0,), dtype=np.float32),
                }
                result["empty_match_fallback"] = str(exc)

    result["prediction"] = summarize_prediction(pred)
    if torch.cuda.is_available():
        result["cuda_peak_allocated_gb"] = round(torch.cuda.max_memory_allocated() / 1024**3, 3)
        result["cuda_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved() / 1024**3, 3)

    args.out.mkdir(parents=True, exist_ok=True)
    safe_name = safe_method_name(method)
    arrays = prediction_arrays(pred)
    result["saved_arrays"] = {key: list(value.shape) for key, value in arrays.items()}
    np.savez_compressed(args.out / f"{safe_name}.npz", **arrays)
    with (args.out / f"{safe_name}.json").open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2, default=str)
    print(json.dumps(result["prediction"], ensure_ascii=False))
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--list", action="store_true", help="List methods from the Space config and exit.")
    parser.add_argument("--all", action="store_true", help="Run all methods sequentially.")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--method", action="append", default=[], help="Method name from app.yaml; can be repeated.")
    parser.add_argument("--image0", type=Path)
    parser.add_argument("--image1", type=Path)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--checkpoints", type=Path, default=DEFAULT_CHECKPOINTS)
    parser.add_argument("--max-ram-gb", type=float, default=None)
    parser.add_argument("--max-vram-gb", type=float, default=None)
    parser.add_argument("--timeout-sec", type=float, default=None, help="Per-method wall-clock timeout in seconds.")
    parser.add_argument("--resume", action="store_true", help="Keep previous manifest records and skip completed methods.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.list:
        for name, conf in load_matcher_zoo().items():
            print(f"{name}\tstandalone={conf['standalone']}\tfeature={conf.get('feature', '-')}\tmatcher={conf.get('matcher', '-')}")
        return 0
    if not args.image0 or not args.image1:
        raise SystemExit("--image0 and --image1 are required when running methods.")
    if args.child:
        if len(args.method) != 1:
            raise SystemExit("Child mode requires exactly one --method.")
        return run_child(args)
    if not args.all and not args.method:
        raise SystemExit("Use --list, --all, or at least one --method.")
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())

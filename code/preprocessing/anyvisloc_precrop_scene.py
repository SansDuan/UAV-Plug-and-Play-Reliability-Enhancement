#!/usr/bin/env python3
"""Pre-crop one AnyVisLoc scene by loading map/DSM once."""

from __future__ import annotations

import argparse
import io
import json
import os
import re
import zipfile
from pathlib import Path

import cv2
import numpy as np

from anyvisloc_prepare_reference_pair import (
    crop_dsm_with_padding,
    crop_with_padding,
    jsonable,
    read_npz,
    read_png,
    rotate_image,
)


ROOT = Path(__file__).resolve().parents[1]


def write_state(state_file: Path | None, **payload) -> None:
    if state_file is None:
        return
    state_file.parent.mkdir(parents=True, exist_ok=True)
    tmp = state_file.with_suffix(state_file.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp.replace(state_file)


def sample_key(name: str):
    match = re.search(r"_(\d+)\.npz$", name)
    return int(match.group(1)) if match else name


def complete_pair(out_dir: Path, mode: str) -> bool:
    required = [
        out_dir / "pair_meta.json",
        out_dir / "uav.png",
        out_dir / f"{mode}_crop_prerotated.png",
        out_dir / f"{mode}_crop_raw.png",
        out_dir / f"{mode}_dsm_crop.npy",
    ]
    return all(path.exists() and path.stat().st_size > 0 for path in required)


def safe_method_name(method: str) -> str:
    return method.replace("/", "_").replace(" ", "_")


def load_required_methods(path: Path) -> list[str]:
    methods = []
    for line in path.read_text(encoding="utf-8").splitlines():
        method = line.strip()
        if method and not method.startswith("#"):
            methods.append(method)
    return methods


def method_outputs_exist(match_dir: Path, method: str) -> bool:
    safe = safe_method_name(method)
    return (match_dir / f"{safe}.json").is_file() and (match_dir / f"{safe}.npz").is_file()


def matched_status(match_dir: Path, required_methods: list[str]) -> tuple[bool, int, list[str]]:
    found = {method for method in required_methods if method_outputs_exist(match_dir, method)}
    manifest = match_dir / "manifest.json"
    if manifest.exists():
        try:
            rows = json.loads(manifest.read_text(encoding="utf-8"))
        except Exception:
            rows = []
        if isinstance(rows, list):
            for row in rows:
                if not isinstance(row, dict) or row.get("returncode") != 0:
                    continue
                method = row.get("method")
                if isinstance(method, str) and method in required_methods and method_outputs_exist(match_dir, method):
                    found.add(method)
    missing = [method for method in required_methods if method not in found]
    return not missing, len(found), missing


def write_pair(
    sample_name: str,
    sample: dict[str, np.ndarray],
    reference: dict,
    mode_name: str,
    ref_image: np.ndarray,
    dsm: np.ndarray,
    zip_path: Path,
    out_dir: Path,
    crop_size: int,
    resize_max: int,
    no_prerotate: bool,
) -> None:
    mode = reference["modes"][mode_name]
    image = np.asarray(sample["image"])
    if image.ndim == 3:
        uav_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    else:
        uav_bgr = image

    map_res_x, map_res_y = [float(v) for v in mode["map_resolution"]]
    origin_x, origin_y = [float(v) for v in mode["map_origin_local"]]
    xyz = np.asarray(sample["xyz"], dtype=np.float64)
    px = (xyz[0] - origin_x) / map_res_x
    py = (xyz[1] - origin_y) / map_res_y

    crop, crop_x0, crop_y0 = crop_with_padding(ref_image, px, py, crop_size, crop_size)

    dsm_res_x, dsm_res_y = [float(v) for v in mode["dsm_resolution"]]
    dsm_origin_x, dsm_origin_y = [float(v) for v in mode["dsm_origin_local"]]
    dsm_cx = (xyz[0] - dsm_origin_x) / dsm_res_x
    dsm_cy = (xyz[1] - dsm_origin_y) / dsm_res_y
    dsm_crop_w = max(1, int(np.ceil((crop_size * map_res_x) / abs(dsm_res_x))) + 4)
    dsm_crop_h = max(1, int(np.ceil((crop_size * map_res_y) / abs(dsm_res_y))) + 4)
    dsm_crop, dsm_crop_x0, dsm_crop_y0 = crop_dsm_with_padding(dsm, dsm_cx, dsm_cy, dsm_crop_w, dsm_crop_h)

    yaw = float(np.asarray(sample.get("euler_deg", [0, 0, 0]))[2])
    prerotate_deg = 0.0 if no_prerotate else -yaw
    crop_aligned, affine = rotate_image(crop, prerotate_deg)

    if resize_max and max(crop_aligned.shape[:2]) > resize_max:
        scale = resize_max / max(crop_aligned.shape[:2])
        crop_aligned = cv2.resize(
            crop_aligned,
            (max(1, int(round(crop_aligned.shape[1] * scale))), max(1, int(round(crop_aligned.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        scale = 1.0

    out_dir.mkdir(parents=True, exist_ok=True)
    uav_path = out_dir / "uav.png"
    ref_path = out_dir / f"{mode_name}_crop_prerotated.png"
    raw_ref_path = out_dir / f"{mode_name}_crop_raw.png"
    dsm_crop_path = out_dir / f"{mode_name}_dsm_crop.npy"
    cv2.imwrite(str(uav_path), uav_bgr)
    cv2.imwrite(str(ref_path), crop_aligned)
    cv2.imwrite(str(raw_ref_path), crop)
    np.save(dsm_crop_path, dsm_crop)

    meta = {
        "zip": str(zip_path),
        "sample": sample_name,
        "mode": mode_name,
        "uav": str(uav_path),
        "reference": str(ref_path),
        "reference_raw": str(raw_ref_path),
        "dsm_crop": str(dsm_crop_path),
        "dsm_crop_x0": int(dsm_crop_x0),
        "dsm_crop_y0": int(dsm_crop_y0),
        "dsm_crop_size": [int(dsm_crop.shape[1]), int(dsm_crop.shape[0])],
        "crop_x0": int(crop_x0),
        "crop_y0": int(crop_y0),
        "crop_size": int(crop_size),
        "reference_resize_scale": float(scale),
        "reference_affine_raw_crop_to_prerotated_before_resize": affine.tolist(),
        "reference_prerotate_deg": float(prerotate_deg),
        "reference_json": reference,
        "sample_fields": {k: jsonable(v) for k, v in sample.items() if k != "image"},
    }
    with (out_dir / "pair_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps({"sample": sample_name, "out": str(out_dir), "dsm_crop": str(dsm_crop_path)}, ensure_ascii=False))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--mode", choices=["aerial", "satellite"], default="aerial")
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--crop-size", type=int, default=1536)
    parser.add_argument("--resize-max", type=int, default=1024)
    parser.add_argument("--start-at-sample", default="")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--match-root", type=Path, default=ROOT.parent / "work" / "matches")
    parser.add_argument("--skip-matched", action="store_true")
    parser.add_argument("--method-list", type=Path, default=ROOT.parent / "data" / "methods.txt")
    parser.add_argument("--min-matched-methods", type=int, default=50)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-prerotate", action="store_true")
    parser.add_argument("--state-file", type=Path, default=None)
    args = parser.parse_args()

    if os.environ.get("OPENCV_NUM_THREADS"):
        cv2.setNumThreads(int(os.environ["OPENCV_NUM_THREADS"]))

    with zipfile.ZipFile(args.zip) as zf:
        samples = sorted((n for n in zf.namelist() if n.endswith(".npz")), key=sample_key)
        if args.start_at_sample:
            samples = [name for name in samples if name >= args.start_at_sample]
        if args.max_samples is not None:
            samples = samples[: args.max_samples]
        if not samples:
            print(json.dumps({"zip": str(args.zip), "samples": 0}, ensure_ascii=False))
            return 0

        if args.skip_matched and not args.force:
            required_methods = load_required_methods(args.method_list)
            todo = []
            skipped_matched = 0
            for sample_name in samples:
                run_name = f"{args.zip.stem}_{Path(sample_name).stem}_{args.mode}"
                complete, found_count, missing = matched_status(args.match_root / run_name, required_methods)
                if complete:
                    write_state(
                        args.state_file,
                        status="skipped-matched",
                        zip=str(args.zip),
                        mode=args.mode,
                        out_root=str(args.out_root),
                        sample=sample_name,
                        run_name=run_name,
                    )
                    skipped_matched += 1
                    print(
                        json.dumps(
                            {
                                "sample": sample_name,
                                "status": "skipped-matched",
                                "match": str(args.match_root / run_name),
                                "found_required": found_count,
                            },
                            ensure_ascii=False,
                        )
                    )
                else:
                    if found_count:
                        print(
                            json.dumps(
                                {
                                    "sample": sample_name,
                                    "status": "partial-existing",
                                    "match": str(args.match_root / run_name),
                                    "found_required": found_count,
                                    "required": len(required_methods),
                                    "missing": missing,
                                },
                                ensure_ascii=False,
                            )
                        )
                    todo.append(sample_name)
            samples = todo
            if not samples:
                print(
                    json.dumps(
                        {
                            "zip": str(args.zip),
                            "mode": args.mode,
                            "generated": 0,
                            "skipped": 0,
                            "skipped_matched": skipped_matched,
                            "failed": 0,
                        },
                        ensure_ascii=False,
                    )
                )
                return 0
        else:
            skipped_matched = 0

        scene_dir = str(Path(samples[0]).parent)
        ref_name = next(n for n in zf.namelist() if n.startswith(scene_dir + "/") and n.endswith("_reference.json"))
        reference = json.loads(zf.read(ref_name).decode("utf-8"))
        mode = reference["modes"][args.mode]
        ref_image = read_png(zf, f"{scene_dir}/{mode['map_path']}")
        dsm = np.load(io.BytesIO(zf.read(f"{scene_dir}/{mode['dsm_path']}")))

        done = skipped = failed = 0
        for sample_name in samples:
            run_name = f"{args.zip.stem}_{Path(sample_name).stem}_{args.mode}"
            out_dir = args.out_root / run_name
            if not args.force and complete_pair(out_dir, args.mode):
                write_state(
                    args.state_file,
                    status="skipped-existing",
                    zip=str(args.zip),
                    mode=args.mode,
                    out_root=str(args.out_root),
                    sample=sample_name,
                    run_name=run_name,
                )
                skipped += 1
                print(json.dumps({"sample": sample_name, "status": "skipped-existing", "out": str(out_dir)}, ensure_ascii=False))
                continue
            try:
                sample = read_npz(zf, sample_name)
                write_pair(
                    sample_name,
                    sample,
                    reference,
                    args.mode,
                    ref_image,
                    dsm,
                    args.zip,
                    out_dir,
                    args.crop_size,
                    args.resize_max,
                    args.no_prerotate,
                )
                write_state(
                    args.state_file,
                    status="generated",
                    zip=str(args.zip),
                    mode=args.mode,
                    out_root=str(args.out_root),
                    sample=sample_name,
                    run_name=run_name,
                )
                done += 1
            except Exception as exc:
                write_state(
                    args.state_file,
                    status="failed",
                    zip=str(args.zip),
                    mode=args.mode,
                    out_root=str(args.out_root),
                    sample=sample_name,
                    run_name=run_name,
                    error=str(exc),
                )
                failed += 1
                print(json.dumps({"sample": sample_name, "status": "failed", "error": str(exc)}, ensure_ascii=False))

    write_state(
        args.state_file,
        status="scene-complete" if failed == 0 else "scene-failed",
        zip=str(args.zip),
        mode=args.mode,
        out_root=str(args.out_root),
        sample=samples[-1] if samples else "",
        failed=failed,
    )
    print(
        json.dumps(
            {
                "zip": str(args.zip),
                "mode": args.mode,
                "generated": done,
                "skipped": skipped,
                "skipped_matched": skipped_matched,
                "failed": failed,
            },
            ensure_ascii=False,
        )
    )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

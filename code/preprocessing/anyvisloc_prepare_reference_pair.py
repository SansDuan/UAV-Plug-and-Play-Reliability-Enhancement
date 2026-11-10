#!/usr/bin/env python3
"""Prepare a UAV-to-reference-map AnyVisLoc pair for absolute PnP tests."""

from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_npz(zf: zipfile.ZipFile, name: str) -> dict[str, np.ndarray]:
    data = np.load(io.BytesIO(zf.read(name)), allow_pickle=True)
    return {k: data[k] for k in data.files}


def read_png(zf: zipfile.ZipFile, name: str) -> np.ndarray:
    arr = np.frombuffer(zf.read(name), dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot decode {name}")
    return image


def rotate_image(image: np.ndarray, degree: float) -> tuple[np.ndarray, np.ndarray]:
    h, w = image.shape[:2]
    radians = np.deg2rad(degree)
    new_h = max(1, int(round(w * abs(np.sin(radians)) + h * abs(np.cos(radians)))))
    new_w = max(1, int(round(h * abs(np.sin(radians)) + w * abs(np.cos(radians)))))
    mat = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), degree, 1.0).astype(np.float32)
    mat[0, 2] += (new_w - w) / 2.0
    mat[1, 2] += (new_h - h) / 2.0
    rotated = cv2.warpAffine(image, mat, (new_w, new_h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0))
    return rotated, mat


def crop_with_padding(image: np.ndarray, cx: float, cy: float, width: int, height: int) -> tuple[np.ndarray, int, int]:
    h, w = image.shape[:2]
    x0 = int(round(cx - width / 2))
    y0 = int(round(cy - height / 2))
    x1 = x0 + width
    y1 = y0 + height
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)
    if sx0 >= sx1 or sy0 >= sy1:
        raise ValueError(f"Crop outside image: center=({cx},{cy}), size=({width},{height}), image=({w},{h})")
    crop = np.zeros((height, width, image.shape[2]), dtype=image.dtype)
    dx0, dy0 = sx0 - x0, sy0 - y0
    crop[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = image[sy0:sy1, sx0:sx1]
    return crop, x0, y0


def crop_dsm_with_padding(dsm: np.ndarray, cx: float, cy: float, width: int, height: int) -> tuple[np.ndarray, int, int]:
    h, w = dsm.shape[:2]
    x0 = int(round(cx - width / 2))
    y0 = int(round(cy - height / 2))
    x1 = x0 + width
    y1 = y0 + height
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x1), min(h, y1)
    if sx0 >= sx1 or sy0 >= sy1:
        raise ValueError(f"DSM crop outside image: center=({cx},{cy}), size=({width},{height}), dsm=({w},{h})")
    crop = np.full((height, width), np.nan, dtype=np.float32)
    dx0, dy0 = sx0 - x0, sy0 - y0
    crop[dy0 : dy0 + (sy1 - sy0), dx0 : dx0 + (sx1 - sx0)] = dsm[sy0:sy1, sx0:sx1].astype(np.float32)
    return crop, x0, y0


def jsonable(value):
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--sample", default="Scene_01/L01_0001.npz")
    parser.add_argument("--mode", choices=["aerial", "satellite"], default="aerial")
    parser.add_argument("--crop-size", type=int, default=1536)
    parser.add_argument("--resize-max", type=int, default=1024)
    parser.add_argument("--out", type=Path, default=ROOT / "outputs" / "anyvisloc_ref_pair")
    parser.add_argument("--no-prerotate", action="store_true")
    args = parser.parse_args()

    if os.environ.get("OPENCV_NUM_THREADS"):
        cv2.setNumThreads(int(os.environ["OPENCV_NUM_THREADS"]))
    args.out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.zip) as zf:
        sample = read_npz(zf, args.sample)
        scene_dir = str(Path(args.sample).parent)
        ref_name = next(n for n in zf.namelist() if n.startswith(scene_dir + "/") and n.endswith("_reference.json"))
        reference = json.loads(zf.read(ref_name).decode("utf-8"))
        mode = reference["modes"][args.mode]
        ref_image = read_png(zf, f"{scene_dir}/{mode['map_path']}")
        dsm = np.load(io.BytesIO(zf.read(f"{scene_dir}/{mode['dsm_path']}")))

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

    crop, crop_x0, crop_y0 = crop_with_padding(ref_image, px, py, args.crop_size, args.crop_size)
    dsm_res_x, dsm_res_y = [float(v) for v in mode["dsm_resolution"]]
    dsm_origin_x, dsm_origin_y = [float(v) for v in mode["dsm_origin_local"]]
    dsm_cx = (xyz[0] - dsm_origin_x) / dsm_res_x
    dsm_cy = (xyz[1] - dsm_origin_y) / dsm_res_y
    crop_world_w = args.crop_size * map_res_x
    crop_world_h = args.crop_size * map_res_y
    dsm_crop_w = max(1, int(np.ceil(crop_world_w / abs(dsm_res_x))) + 4)
    dsm_crop_h = max(1, int(np.ceil(crop_world_h / abs(dsm_res_y))) + 4)
    dsm_crop, dsm_crop_x0, dsm_crop_y0 = crop_dsm_with_padding(dsm, dsm_cx, dsm_cy, dsm_crop_w, dsm_crop_h)
    yaw = float(np.asarray(sample.get("euler_deg", [0, 0, 0]))[2])
    prerotate_deg = 0.0 if args.no_prerotate else -yaw
    crop_aligned, affine = rotate_image(crop, prerotate_deg)

    if args.resize_max and max(crop_aligned.shape[:2]) > args.resize_max:
        scale = args.resize_max / max(crop_aligned.shape[:2])
        crop_aligned = cv2.resize(
            crop_aligned,
            (max(1, int(round(crop_aligned.shape[1] * scale))), max(1, int(round(crop_aligned.shape[0] * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    else:
        scale = 1.0

    uav_path = args.out / "uav.png"
    ref_path = args.out / f"{args.mode}_crop_prerotated.png"
    raw_ref_path = args.out / f"{args.mode}_crop_raw.png"
    dsm_crop_path = args.out / f"{args.mode}_dsm_crop.npy"
    cv2.imwrite(str(uav_path), uav_bgr)
    cv2.imwrite(str(ref_path), crop_aligned)
    cv2.imwrite(str(raw_ref_path), crop)
    np.save(dsm_crop_path, dsm_crop)

    meta = {
        "zip": str(args.zip),
        "sample": args.sample,
        "mode": args.mode,
        "uav": str(uav_path),
        "reference": str(ref_path),
        "reference_raw": str(raw_ref_path),
        "dsm_crop": str(dsm_crop_path),
        "dsm_crop_x0": dsm_crop_x0,
        "dsm_crop_y0": dsm_crop_y0,
        "dsm_crop_size": [int(dsm_crop.shape[1]), int(dsm_crop.shape[0])],
        "crop_x0": crop_x0,
        "crop_y0": crop_y0,
        "crop_size": args.crop_size,
        "reference_resize_scale": scale,
        "reference_affine_raw_crop_to_prerotated_before_resize": affine.tolist(),
        "reference_prerotate_deg": prerotate_deg,
        "reference_json": reference,
        "sample_fields": {k: jsonable(v) for k, v in sample.items() if k != "image"},
    }
    with (args.out / "pair_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(json.dumps({"uav": str(uav_path), "reference": str(ref_path), "meta": str(args.out / "pair_meta.json")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
Generate a self-rendered training dataset for the chief pose CNN.

Replaces SPEED+ (wrong geometry/range regime -- Tango satellite, 2-10m) with
images rendered by render/chief_renderer.py using the sim's own chief
geometry and camera model. Labels are written in the exact schema
pose_cnn/dataset.py::SpeedPlusDataset already parses, so the existing loader
needs no changes -- only --data-root changes.

Range is capped at ~150m: beyond that the chief (0.8-1.5m half-extents) is
sub-pixel at this camera's 800px focal length, so renders would be
indistinguishable blank frames and useless for training.

Range floor is 1.5m, just above the chief's circumscribing-sphere radius
(~1.24m, from CHIEF_BODY_HALF_EXTENTS_M) -- below that the camera can end up
inside the object's own geometry, which this rasterizer (no near-plane
clipping) renders as a degenerate near-blank frame. The existing analytic
ChiefPoseEstimator already never runs its orientation PnP below 2m for the
same reason (it coasts instead) -- 1.5m keeps a small margin below that
transition for the position channel without entering the degenerate regime.

The camera boresight always points exactly at the chief by construction (see
render/chief_renderer.py::_camera_frame). Both bearing direction and range
are randomized so the CNN sees the chief from all viewpoints — necessary for
orientation observability (a fixed bearing leaves rotations around that axis
nearly invisible).
"""

import argparse
import json
import os

import numpy as np
from PIL import Image

from render.chief_renderer import render_chief

MIN_RANGE_M = 1.5
MAX_RANGE_M = 30.0   # beyond ~30m retroreflectors go sub-pixel; restrict to useful range


def _random_quaternion(rng):
    v = rng.normal(size=4)
    return v / np.linalg.norm(v)


def _random_unit_vector(rng):
    v = rng.normal(size=3)
    return v / np.linalg.norm(v)


def _sample_range(rng):
    return float(np.exp(rng.uniform(np.log(MIN_RANGE_M), np.log(MAX_RANGE_M))))


def generate_split(n_images, images_dir, rng, start_idx=0):
    os.makedirs(images_dir, exist_ok=True)
    records = []
    for i in range(n_images):
        r = _sample_range(rng)
        dr_lvlh = _random_unit_vector(rng) * r   # randomise bearing — needed for orientation observability
        q_chief = _random_quaternion(rng)
        sun_lvlh = _random_unit_vector(rng)

        img = render_chief(dr_lvlh, q_chief, sun_lvlh, rng=rng)
        filename = f"render_{start_idx + i:06d}.jpg"
        Image.fromarray(img).save(os.path.join(images_dir, filename), quality=90)
        records.append({
            "filename": filename,
            "r_Vo2To_vbs_true": dr_lvlh.tolist(),
            "q_vbs2tango_true": q_chief.tolist(),
        })
    return records


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-images", type=int, default=5000)
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--out-root", default="pose_cnn/data/chief_render")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    images_dir = os.path.join(args.out_root, "images")

    n_val = int(args.n_images * args.val_frac)
    n_train = args.n_images - n_val

    print(f"Generating {n_train} train + {n_val} val images -> {images_dir}")
    train_records = generate_split(n_train, images_dir, rng, start_idx=0)
    val_records = generate_split(n_val, images_dir, rng, start_idx=n_train)

    os.makedirs(args.out_root, exist_ok=True)
    with open(os.path.join(args.out_root, "train.json"), "w") as f:
        json.dump(train_records, f)
    with open(os.path.join(args.out_root, "validation.json"), "w") as f:
        json.dump(val_records, f)

    print(f"Wrote {len(train_records)} train / {len(val_records)} val records")


if __name__ == "__main__":
    main()

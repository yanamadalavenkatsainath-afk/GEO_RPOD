"""
Range-stratified evaluation of the orientation CNN.
Shows whether error is dominated by long-range (uninformative) images
vs near-range (mission-critical) images.
"""

import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from pose_cnn.dataset import SpeedPlusDataset
from pose_cnn.model import PoseRegressionNet


def quat_angle_error_deg(q_pred, q_true):
    dot = np.clip(np.abs(np.sum(q_pred * q_true, axis=1)), 0.0, 1.0)
    return np.degrees(2.0 * np.arccos(dot))


BINS = [
    ("  <5m",    0,    5),
    (" 5-10m",   5,   10),
    ("10-20m",  10,   20),
    ("20-30m",  20,   30),
    ("30-50m",  30,   50),
    ("50-150m", 50,  150),
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",  default="pose_cnn/data/chief_render")
    parser.add_argument("--split-json", default="validation.json")
    parser.add_argument("--checkpoint", default="pose_cnn/checkpoints/pose_net.pt")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--workers",    type=int, default=4)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds     = SpeedPlusDataset(args.data_root, args.split_json)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=True)

    model = PoseRegressionNet(pretrained=False).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    all_err, all_range = [], []
    with torch.no_grad():
        for img, pos, quat in loader:
            img  = img.to(device)
            pos_d = pos.to(device)
            bearing = pos_d / pos_d.norm(dim=1, keepdim=True).clamp_min(1e-3)
            quat_pred = model(img, bearing).cpu().numpy()
            errs = quat_angle_error_deg(quat_pred, quat.numpy())
            ranges = pos.norm(dim=1).numpy()
            all_err.extend(errs)
            all_range.extend(ranges)

    all_err   = np.array(all_err)
    all_range = np.array(all_range)

    print(f"\n{args.data_root}/{args.split_json}  (N={len(all_err)})")
    print(f"  OVERALL  mean={all_err.mean():.1f}deg  median={np.median(all_err):.1f}deg  "
          f"std={all_err.std():.1f}deg  p95={np.percentile(all_err, 95):.1f}deg\n")

    print(f"  {'Range':8s}  {'N':>5s}  {'mean':>7s}  {'median':>7s}  {'p95':>7s}")
    print(f"  {'-'*45}")
    for label, lo, hi in BINS:
        mask = (all_range >= lo) & (all_range < hi)
        n = mask.sum()
        if n == 0:
            print(f"  {label:8s}  {n:>5d}  {'—':>7s}  {'—':>7s}  {'—':>7s}")
            continue
        e = all_err[mask]
        print(f"  {label:8s}  {n:>5d}  {e.mean():>6.1f}°  {np.median(e):>6.1f}°  "
              f"{np.percentile(e,95):>6.1f}°")


if __name__ == "__main__":
    main()

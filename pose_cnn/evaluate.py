"""
Evaluate trained orientation CNN and report attitude error statistics.
Used to calibrate sigma_att_deg in sensors/uncooperative_pose_sensor.py.
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

    ori_err = []
    with torch.no_grad():
        for img, pos, quat in loader:
            img  = img.to(device)
            pos  = pos.to(device)
            bearing = pos / pos.norm(dim=1, keepdim=True).clamp_min(1e-3)
            quat_pred = model(img, bearing).cpu().numpy()
            ori_err.extend(quat_angle_error_deg(quat_pred, quat.numpy()))

    ori_err = np.array(ori_err)
    print(f"\n{args.data_root}/{args.split_json}  (N={len(ori_err)})")
    print(f"  ori_err  mean={ori_err.mean():.2f}deg  "
          f"median={np.median(ori_err):.2f}deg  "
          f"std={ori_err.std():.2f}deg  "
          f"p95={np.percentile(ori_err, 95):.2f}deg")


if __name__ == "__main__":
    main()

"""
Train orientation CNN (image + lidar bearing -> quaternion) on self-rendered
GEO chief imagery.  The bearing unit vector from the flash lidar resolves the
viewing-direction ambiguity that makes pure-image attitude estimation degenerate
when the camera always centres the chief.
"""

import argparse
import os
import time

import torch
from torch.utils.data import DataLoader

from pose_cnn.dataset import SpeedPlusDataset
from pose_cnn.model import PoseRegressionNet


def quat_geodesic_loss(q_pred, q_true):
    dot = (q_pred * q_true).sum(dim=1).abs().clamp(max=1.0)
    return (1.0 - dot).mean()


def run_epoch(model, loader, device, optimizer=None):
    train = optimizer is not None
    model.train(train)
    total_loss, n = 0.0, 0
    for img, pos, quat in loader:
        img, pos, quat = img.to(device), pos.to(device), quat.to(device)
        bearing = pos / pos.norm(dim=1, keepdim=True).clamp_min(1e-3)
        with torch.set_grad_enabled(train):
            quat_pred = model(img, bearing)
            loss = quat_geodesic_loss(quat_pred, quat)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * img.size(0)
        n += img.size(0)
    return total_loss / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root",  default="pose_cnn/data/chief_render")
    parser.add_argument("--epochs",     type=int,   default=50)
    parser.add_argument("--batch-size", type=int,   default=64)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--workers",    type=int,   default=4)
    parser.add_argument("--limit",      type=int,   default=None)
    parser.add_argument("--out",        default="pose_cnn/checkpoints/pose_net.pt")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    train_ds = SpeedPlusDataset(args.data_root, "train.json",      limit=args.limit)
    val_ds   = SpeedPlusDataset(args.data_root, "validation.json", limit=args.limit)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.workers, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                               num_workers=args.workers, pin_memory=True)

    model     = PoseRegressionNet(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best_val = float("inf")
    for epoch in range(args.epochs):
        t0 = time.time()
        train_loss = run_epoch(model, train_loader, device, optimizer)
        val_loss   = run_epoch(model, val_loader,   device)
        scheduler.step()
        dt = time.time() - t0
        print(f"epoch {epoch+1}/{args.epochs}  "
              f"train ori={train_loss:.4f}  val ori={val_loss:.4f}  ({dt:.0f}s)")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), args.out)
            print(f"  saved new best checkpoint -> {args.out}")


if __name__ == "__main__":
    main()

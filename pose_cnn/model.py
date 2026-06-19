"""Orientation CNN for GEO chief spacecraft — image + lidar bearing input."""

import torch
import torch.nn as nn
import torchvision.models as models


def rot6d_to_quat(r6d):
    """Convert 6D rotation representation to unit quaternion.

    Input  : (B, 6) — first two columns of rotation matrix, stacked
    Output : (B, 4) — unit quaternion [w, x, y, z]

    Avoids the quaternion double-cover ambiguity (q and -q same rotation)
    that causes near-180° prediction errors.
    """
    a1 = r6d[:, :3]
    a2 = r6d[:, 3:]
    b1 = nn.functional.normalize(a1, dim=1)
    b2 = nn.functional.normalize(a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1, dim=1)
    b3 = torch.cross(b1, b2, dim=1)
    R  = torch.stack([b1, b2, b3], dim=2)          # (B, 3, 3) rotation matrix

    # rotation matrix -> quaternion (Shepperd method, numerically stable)
    trace = R[:, 0, 0] + R[:, 1, 1] + R[:, 2, 2]
    w = torch.sqrt((1.0 + trace).clamp_min(1e-8)) / 2.0
    x = (R[:, 2, 1] - R[:, 1, 2]) / (4.0 * w.clamp_min(1e-8))
    y = (R[:, 0, 2] - R[:, 2, 0]) / (4.0 * w.clamp_min(1e-8))
    z = (R[:, 1, 0] - R[:, 0, 1]) / (4.0 * w.clamp_min(1e-8))
    q = torch.stack([w, x, y, z], dim=1)
    return nn.functional.normalize(q, dim=1)


class PoseRegressionNet(nn.Module):
    """ResNet18 image branch + bearing MLP branch -> quaternion via 6D output.

    Inputs
    ------
    x       : (B, 3, 224, 224) image tensor
    bearing : (B, 3) unit vector from deputy to chief in LVLH frame

    Returns
    -------
    quat : (B, 4) normalised quaternion [w, x, y, z]
    """

    def __init__(self, pretrained=True):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.bearing_mlp = nn.Sequential(
            nn.Linear(3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 64),
            nn.ReLU(inplace=True),
        )

        self.head = nn.Sequential(
            nn.Linear(512 + 64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 6),   # 6D rotation representation
        )

    def forward(self, x, bearing):
        feat_img     = self.backbone(x)
        feat_bearing = self.bearing_mlp(bearing)
        feat         = torch.cat([feat_img, feat_bearing], dim=1)
        r6d          = self.head(feat)
        return rot6d_to_quat(r6d)

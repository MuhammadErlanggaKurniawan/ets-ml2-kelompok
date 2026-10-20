"""Track C (Rijal, quail egg yolk): EfficientNet-B0 image embeddings -> CSV.

STATUS: waiting for data. This script has NOT been run yet because the yolk images are not in the
repo. It needs extra packages that are not in pyproject.toml:

    uv add torch torchvision pillow

Expected layout (labels come from the Yolk Index measured with a caliper, never from the image):

    data/egg/labels.csv     image,yolk_index,storage_day,temperature   (one row per egg)
    data/egg/images/<image>

Class thresholds on the Yolk Index must be fixed in Rijal's protocol *before* modelling; pass them
with --thresholds (low/medium boundary, medium/high boundary).

    uv run python scripts/egg_extract_embeddings.py --thresholds 0.40 0.45
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EGG = ROOT / "data" / "egg"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thresholds", nargs=2, type=float, required=True)
    args = ap.parse_args()

    import torch
    from PIL import Image
    from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0

    weights = EfficientNet_B0_Weights.IMAGENET1K_V1
    net = efficientnet_b0(weights=weights)
    net.classifier = torch.nn.Identity()          # 1280-d global-average-pooled features
    net.eval()
    prep = weights.transforms()

    labels = pd.read_csv(EGG / "labels.csv")
    lo, hi = args.thresholds
    labels["y"] = np.select([labels.yolk_index < lo, labels.yolk_index < hi], [0, 1], 2)  # low/med/high

    feats = []
    with torch.no_grad():
        for name in labels["image"]:
            img = prep(Image.open(EGG / "images" / name).convert("RGB")).unsqueeze(0)
            feats.append(net(img).squeeze(0).numpy())
    F = pd.DataFrame(np.vstack(feats), columns=[f"f{i}" for i in range(len(feats[0]))])
    out = pd.concat([labels.reset_index(drop=True), F], axis=1)
    out.to_csv(EGG / "embeddings_efficientnet_b0.csv", index=False)
    print(f"saved {out.shape} -> {EGG / 'embeddings_efficientnet_b0.csv'}")


if __name__ == "__main__":
    main()

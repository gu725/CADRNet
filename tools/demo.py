import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cadrnet import CADRNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a CADRNet forward pass on random tensors.")
    parser.add_argument("--backbone", default="resnet18", help="timm backbone name supported by cadrnet.utils.backbone")
    parser.add_argument("--exchange", default="le", help="feature exchange mode: le, rle, ce, rce, se, rse, fdse")
    parser.add_argument("--size", type=int, default=128, help="input image height/width")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pretrained", action="store_true", help="load timm pretrained weights if available")
    parser.add_argument("--train-mode", action="store_true", help="also pass random labels and return auxiliary losses")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    model = CADRNet(
        model_name=args.backbone,
        exchange_mode=args.exchange,
        decoder_refiner="gdcr",
        cdam=1,
        pretrained=args.pretrained,
    ).to(device)
    model.train(args.train_mode)

    x_a = torch.randn(args.batch_size, 3, args.size, args.size, device=device)
    x_b = torch.randn(args.batch_size, 3, args.size, args.size, device=device)
    gt = None
    if args.train_mode:
        gt = torch.randint(0, 2, (args.batch_size, args.size, args.size), device=device)

    with torch.set_grad_enabled(args.train_mode):
        output = model(x_a, x_b, gt=gt)

    logits = output["logits"] if isinstance(output, dict) else output
    print("logits:", [tuple(item.shape) for item in logits])
    if isinstance(output, dict):
        print("aux keys:", sorted(output["aux"].keys()))


if __name__ == "__main__":
    main()

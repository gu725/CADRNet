import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cadrnet import CADRNet


class RandomChangeDataset(Dataset):
    def __init__(self, length: int, size: int) -> None:
        self.length = int(length)
        self.size = int(size)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int):
        del index
        x_a = torch.randn(3, self.size, self.size)
        x_b = torch.randn(3, self.size, self.size)
        y = torch.randint(0, 2, (self.size, self.size), dtype=torch.long)
        return x_a, x_b, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a tiny CADRNet training loop on random data.")
    parser.add_argument("--backbone", default="resnet18")
    parser.add_argument("--exchange", default="le")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--aux-weight", type=float, default=0.05)
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
    model.train()

    loader = DataLoader(
        RandomChangeDataset(length=max(args.steps * args.batch_size, args.batch_size), size=args.size),
        batch_size=args.batch_size,
        shuffle=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    for step, (x_a, x_b, y) in enumerate(loader, start=1):
        if step > args.steps:
            break
        x_a = x_a.to(device)
        x_b = x_b.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        output = model(x_a, x_b, gt=y)
        logits = output["logits"] if isinstance(output, dict) else output
        main_loss = sum(F.cross_entropy(item, y) for item in logits) / len(logits)
        aux_loss = torch.zeros((), device=device)
        if isinstance(output, dict):
            for key, value in output["aux"].items():
                if key.startswith("loss_") and torch.is_tensor(value):
                    aux_loss = aux_loss + value
        loss = main_loss + args.aux_weight * aux_loss
        loss.backward()
        optimizer.step()

        print(
            f"step={step} "
            f"loss={loss.item():.4f} "
            f"main={main_loss.item():.4f} "
            f"aux={aux_loss.item():.4f}"
        )


if __name__ == "__main__":
    main()

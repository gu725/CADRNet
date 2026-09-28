import argparse
import random
import sys
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cadrnet import CADRNet


class TxtChangeDataset(Dataset):
    """Change-detection dataset backed by train/val/test txt files.

    Each line must contain three paths:
        image_t1 image_t2 mask

    Paths can be absolute or relative to the dataset root.
    """

    def __init__(
        self,
        dataset_root: Path,
        split: str,
        crop_size: int,
        train: bool,
        resize_size: int = 0,
    ) -> None:
        self.dataset_root = Path(dataset_root)
        self.split = split
        self.crop_size = int(crop_size)
        self.train = bool(train)
        self.resize_size = int(resize_size)
        txt_path = self.dataset_root / f"{split}.txt"
        if not txt_path.is_file():
            raise FileNotFoundError(f"Missing split file: {txt_path}")
        self.samples = self._read_txt(txt_path)
        if not self.samples:
            raise ValueError(f"No samples found in {txt_path}")

    def _resolve(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        return self.dataset_root / path

    def _read_txt(self, txt_path: Path) -> List[Tuple[Path, Path, Path]]:
        samples: List[Tuple[Path, Path, Path]] = []
        with txt_path.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.replace(",", " ").split()
                if len(parts) < 3:
                    raise ValueError(f"{txt_path}:{line_no} must contain image_a image_b mask")
                samples.append((self._resolve(parts[0]), self._resolve(parts[1]), self._resolve(parts[2])))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    @staticmethod
    def _load_rgb(path: Path) -> Image.Image:
        return Image.open(path).convert("RGB")

    @staticmethod
    def _load_mask(path: Path) -> Image.Image:
        return Image.open(path).convert("L")

    def _resize_if_needed(self, img_a: Image.Image, img_b: Image.Image, mask: Image.Image):
        if self.resize_size > 0:
            size = [self.resize_size, self.resize_size]
            return (
                TF.resize(img_a, size, interpolation=TF.InterpolationMode.BILINEAR),
                TF.resize(img_b, size, interpolation=TF.InterpolationMode.BILINEAR),
                TF.resize(mask, size, interpolation=TF.InterpolationMode.NEAREST),
            )
        min_side = min(img_a.size[0], img_a.size[1])
        if min_side >= self.crop_size:
            return img_a, img_b, mask
        size = [self.crop_size, self.crop_size]
        return (
            TF.resize(img_a, size, interpolation=TF.InterpolationMode.BILINEAR),
            TF.resize(img_b, size, interpolation=TF.InterpolationMode.BILINEAR),
            TF.resize(mask, size, interpolation=TF.InterpolationMode.NEAREST),
        )

    def _crop(self, img_a: Image.Image, img_b: Image.Image, mask: Image.Image):
        width, height = img_a.size
        crop = min(self.crop_size, height, width)
        if self.train:
            top = random.randint(0, height - crop) if height > crop else 0
            left = random.randint(0, width - crop) if width > crop else 0
        else:
            top = max((height - crop) // 2, 0)
            left = max((width - crop) // 2, 0)
        return (
            TF.crop(img_a, top, left, crop, crop),
            TF.crop(img_b, top, left, crop, crop),
            TF.crop(mask, top, left, crop, crop),
        )

    def __getitem__(self, index: int):
        path_a, path_b, path_mask = self.samples[index]
        img_a = self._load_rgb(path_a)
        img_b = self._load_rgb(path_b)
        mask = self._load_mask(path_mask)
        img_a, img_b, mask = self._resize_if_needed(img_a, img_b, mask)
        img_a, img_b, mask = self._crop(img_a, img_b, mask)

        if self.train and random.random() < 0.5:
            img_a = TF.hflip(img_a)
            img_b = TF.hflip(img_b)
            mask = TF.hflip(mask)
        if self.train and random.random() < 0.5:
            img_a = TF.vflip(img_a)
            img_b = TF.vflip(img_b)
            mask = TF.vflip(mask)

        x_a = TF.to_tensor(img_a)
        x_b = TF.to_tensor(img_b)
        y = (TF.pil_to_tensor(mask).squeeze(0) > 0).long()
        return x_a, x_b, y


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CADRNet on txt-based change-detection datasets.")
    parser.add_argument("--dataset", required=True, help="dataset root containing train.txt and optionally val.txt")
    parser.add_argument("--backbone", default="convnext_base")
    parser.add_argument("--exchange", default="le")
    parser.add_argument("--decoder-scale", default="single", choices=["single", "multi"])
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--resize-size", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=25000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup", type=int, default=3000)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--cdam-loss-weight", type=float, default=1.0)
    parser.add_argument("--gdcr-loss-weight", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--work-dir", default="work_dirs")
    parser.add_argument("--exp-name", default="cadrnet_run")
    parser.add_argument("--val-interval", type=int, default=1000)
    parser.add_argument("--save-interval", type=int, default=5000)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def unpack_logits(output):
    return output["logits"] if isinstance(output, dict) else output


def aux_loss_from_output(output, cdam_weight: float, gdcr_weight: float, device: torch.device) -> torch.Tensor:
    loss = torch.zeros((), device=device)
    if not isinstance(output, dict):
        return loss
    aux = output.get("aux", {})
    if "loss_cdam" in aux:
        loss = loss + cdam_weight * aux["loss_cdam"]
    if "loss_gdcr" in aux:
        loss = loss + gdcr_weight * aux["loss_gdcr"]
    return loss


@torch.no_grad()
def validate(model: CADRNet, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_inter = 0.0
    total_union = 0.0
    total_batches = 0
    for x_a, x_b, y in loader:
        x_a = x_a.to(device)
        x_b = x_b.to(device)
        y = y.to(device)
        logits = unpack_logits(model(x_a, x_b))
        merged = torch.stack(list(logits), dim=0).mean(dim=0)
        loss = F.cross_entropy(merged, y)
        pred = merged.argmax(dim=1)
        inter = ((pred == 1) & (y == 1)).sum().item()
        union = ((pred == 1) | (y == 1)).sum().item()
        total_loss += loss.item()
        total_inter += inter
        total_union += union
        total_batches += 1
    miou = total_inter / max(total_union, 1.0)
    return total_loss / max(total_batches, 1), miou


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    run_dir = Path(args.work_dir) / args.exp_name
    run_dir.mkdir(parents=True, exist_ok=True)

    train_set = TxtChangeDataset(Path(args.dataset), split="train", crop_size=args.crop_size, train=True, resize_size=args.resize_size)
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
    )
    val_loader = None
    if (Path(args.dataset) / "val.txt").is_file():
        val_set = TxtChangeDataset(Path(args.dataset), split="val", crop_size=args.crop_size, train=False, resize_size=args.resize_size)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = CADRNet(
        model_name=args.backbone,
        exchange_mode=args.exchange,
        decoder_scale=args.decoder_scale,
        decoder_refiner="gdcr",
        cdam=1,
        pretrained=args.pretrained,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def lr_factor(step: int) -> float:
        if step < args.warmup:
            raw = float(step) / float(max(1, args.warmup))
        else:
            progress = float(step - args.warmup) / float(max(1, args.max_steps - args.warmup))
            raw = max(0.0, (1.0 - progress) ** 3.0)
        return max(raw, args.min_lr / args.lr)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)

    step = 0
    model.train()
    while step < args.max_steps:
        for x_a, x_b, y in train_loader:
            if step >= args.max_steps:
                break
            step += 1
            x_a = x_a.to(device)
            x_b = x_b.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)
            output = model(x_a, x_b, gt=y)
            logits = unpack_logits(output)
            main_loss = sum(F.cross_entropy(item, y) for item in logits) / len(logits)
            aux_loss = aux_loss_from_output(output, args.cdam_loss_weight, args.gdcr_loss_weight, device)
            loss = main_loss + aux_loss
            loss.backward()
            optimizer.step()
            scheduler.step()

            if step == 1 or step % 50 == 0:
                print(
                    f"step={step:06d} loss={loss.item():.4f} "
                    f"main={main_loss.item():.4f} aux={aux_loss.item():.4f} "
                    f"lr={scheduler.get_last_lr()[0]:.6g}"
                )
            if val_loader is not None and args.val_interval > 0 and step % args.val_interval == 0:
                val_loss, val_iou = validate(model, val_loader, device)
                print(f"val step={step:06d} loss={val_loss:.4f} change_iou={val_iou:.4f}")
                model.train()
            if args.save_interval > 0 and step % args.save_interval == 0:
                ckpt_path = run_dir / f"step_{step:06d}.pth"
                torch.save({"model": model.state_dict(), "args": vars(args), "step": step}, ckpt_path)
                print(f"saved {ckpt_path}")

    final_path = run_dir / "final.pth"
    torch.save({"model": model.state_dict(), "args": vars(args), "step": step}, final_path)
    print(f"training finished, saved {final_path}")


if __name__ == "__main__":
    main()

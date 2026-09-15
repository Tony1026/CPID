#!/usr/bin/env python3
"""Inference entry point using the released model implementation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import torch
from PIL import Image, ImageOps
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF
from transformers import CLIPVisionModel

from module.image_model import ImageDRGDModel

CLIP_NAME = "openai/clip-vit-large-patch14"
MEAN = (0.48145466, 0.4578275, 0.40821073)
STD = (0.26862954, 0.26130258, 0.27577711)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def native_tiles(image: Image.Image, crop_size: int = 224) -> torch.Tensor:
    image = image.convert("RGB")
    width, height = image.size
    pad_w, pad_h = max(0, crop_size - width), max(0, crop_size - height)
    if pad_w or pad_h:
        image = ImageOps.expand(
            image,
            border=(pad_w // 2, pad_h // 2,
                    pad_w - pad_w // 2, pad_h - pad_h // 2),
            fill=tuple(round(255 * x) for x in MEAN),
        )
    width, height = image.size
    xs = list(range(0, max(1, width - crop_size + 1), crop_size))
    ys = list(range(0, max(1, height - crop_size + 1), crop_size))
    if xs[-1] != width - crop_size:
        xs.append(max(0, width - crop_size))
    if ys[-1] != height - crop_size:
        ys.append(max(0, height - crop_size))
    tiles = []
    for y in ys:
        for x in xs:
            crop = TF.crop(image, y, x, crop_size, crop_size)
            tiles.append(TF.normalize(TF.to_tensor(crop), MEAN, STD))
    return torch.stack(tiles)


def center_view(image: Image.Image, crop_size: int = 224) -> torch.Tensor:
    image = image.convert("RGB")
    image = TF.resize(image, crop_size, interpolation=InterpolationMode.BICUBIC, antialias=True)
    image = TF.center_crop(image, crop_size)
    return TF.normalize(TF.to_tensor(image), MEAN, STD).unsqueeze(0)


def image_paths(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
        return
    for item in sorted(path.rglob("*")):
        if item.is_file() and item.suffix.lower() in IMAGE_EXTS:
            yield item


class Detector:
    def __init__(self, checkpoint: Path, clip_name: str, device: torch.device):
        self.device = device
        self.encoder = CLIPVisionModel.from_pretrained(clip_name).to(device).eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.model = ImageDRGDModel(dim=1024, minimal_lgnd=True).to(device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        missing, unexpected = self.model.load_state_dict(state, strict=False)
        if unexpected or set(missing) - {"nuisance_to_auth.weight"}:
            raise RuntimeError(f"checkpoint mismatch: missing={missing}, unexpected={unexpected}")
        self.model.eval()

    @torch.inference_mode()
    def predict_tiles(self, tiles: torch.Tensor) -> float:
        logits = []
        for batch in tiles.split(32):
            features = self.encoder(pixel_values=batch.to(self.device)).last_hidden_state
            logits.append(self.model.test_struct(features))
        return float(torch.softmax(torch.cat(logits).mean(0), dim=-1)[1].cpu())


def main() -> None:
    parser = argparse.ArgumentParser(description="Run released CPID inference.")
    parser.add_argument("--input", required=True, type=Path, help="image file or directory")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--clip-name", default=CLIP_NAME)
    parser.add_argument("--view", choices=("native-tiles", "center"), default="native-tiles")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    detector = Detector(args.checkpoint, args.clip_name, device)
    results = {}
    for path in image_paths(args.input):
        with Image.open(path) as image:
            tiles = native_tiles(image) if args.view == "native-tiles" else center_view(image)
        probability = detector.predict_tiles(tiles)
        results[str(path)] = {
            "fake_probability": probability,
            "label": "fake" if probability >= 0.5 else "real",
            "num_views": int(len(tiles)),
        }
        print(f"{path}\t{results[str(path)]['label']}\t{probability:.4f}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"saved: {args.output}")


if __name__ == "__main__":
    main()

"""FFHQ test images shipped with SILO (external/SILO/data/ffhq)."""

import numpy as np
import torch
from datasets import load_from_disk

from paseo_implementation.silo_operator import SILO_DIR


def load_ffhq(num_images, device="cuda"):
    """First num_images FFHQ images as a (N, 3, 512, 512) tensor in [-1, 1]."""
    dataset = load_from_disk(str(SILO_DIR / "data" / "ffhq"))
    images = []
    for i in range(num_images):
        image = dataset[i]["image"].convert("RGB").resize((512, 512))
        images.append(torch.from_numpy(np.array(image)).permute(2, 0, 1))
    return (torch.stack(images).float() / 127.5 - 1).to(device)

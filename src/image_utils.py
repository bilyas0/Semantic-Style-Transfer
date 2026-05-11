from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image


class ImageProcessor:
    """Görsel okuma, boyutlandırma ve format dönüşümlerini yöneten yardımcı sınıf."""

    @staticmethod
    def load_image(path: str | Path, image_size: int = 384) -> Image.Image:
        """Dosya yolundan görsel okur, RGB'ye çevirir ve kare boyuta getirir."""
        image = Image.open(path).convert("RGB")
        return image.resize((image_size, image_size), Image.LANCZOS)

    @staticmethod
    def resize_image(image: Image.Image, image_size: int = 384) -> Image.Image:
        """PIL görseli RGB'ye çevirip kare boyuta getirir."""
        return image.convert("RGB").resize((image_size, image_size), Image.LANCZOS)

    @staticmethod
    def pil_to_tensor(image: Image.Image, device: torch.device) -> torch.Tensor:
        """PIL görseli [1, 3, H, W] PyTorch tensörüne çevirir."""
        return T.ToTensor()(image).unsqueeze(0).to(device)

    @staticmethod
    def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
        """[1, 3, H, W] tensörü PIL görsele çevirir."""
        tensor = tensor.detach().cpu().squeeze(0).clamp(0, 1)
        return T.ToPILImage()(tensor)

    @staticmethod
    def pil_to_numpy(image: Image.Image) -> np.ndarray:
        """PIL görseli 0-1 arası float numpy formatına çevirir."""
        return np.asarray(image).astype(np.float32) / 255.0

    @staticmethod
    def tensor_to_numpy(tensor: torch.Tensor) -> np.ndarray:
        """[1, 3, H, W] tensörü [H, W, 3] numpy formatına çevirir."""
        tensor = tensor.detach().cpu().squeeze(0).clamp(0, 1)
        return tensor.permute(1, 2, 0).numpy()

    @staticmethod
    def numpy_to_pil(image_np: np.ndarray) -> Image.Image:
        """0-1 arası numpy görseli PIL görsele çevirir."""
        image_np = np.clip(image_np, 0, 1)
        return Image.fromarray((image_np * 255).astype(np.uint8))

    @staticmethod
    def save_image(image: Image.Image | np.ndarray, output_path: str | Path) -> None:
        """PIL veya numpy görseli diske kaydeder."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(image, np.ndarray):
            image = ImageProcessor.numpy_to_pil(image)

        image.save(output_path)

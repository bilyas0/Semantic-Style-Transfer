from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision.models.segmentation import DeepLabV3_ResNet50_Weights, deeplabv3_resnet50


class PersonSegmenter:
    """DeepLabV3-ResNet50 hazır modeliyle insan/person maskesi çıkarır."""

    def __init__(self, device: torch.device, kernel_size: int = 7) -> None:
        self.device = device
        self.kernel_size = kernel_size

        self.weights = DeepLabV3_ResNet50_Weights.DEFAULT
        self.model = deeplabv3_resnet50(weights=self.weights).to(device).eval()
        self.preprocess = self.weights.transforms()

        categories = self.weights.meta["categories"]
        self.person_class_id = categories.index("person")

    @torch.no_grad()
    def predict_mask(self, image: Image.Image) -> np.ndarray:
        """Görselden person maskesi üretir. Çıktı boyutu: [H, W], değer aralığı: 0-1."""
        original_width, original_height = image.size
        input_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        output = self.model(input_tensor)["out"][0]
        prediction = output.argmax(0).cpu().numpy()

        raw_mask = (prediction == self.person_class_id).astype(np.uint8)
        raw_mask = cv2.resize(
            raw_mask,
            (original_width, original_height),
            interpolation=cv2.INTER_NEAREST,
        )

        return self._clean_mask(raw_mask)

    def _clean_mask(self, mask: np.ndarray) -> np.ndarray:
        """Küçük gürültüleri azaltmak için morfolojik açma/kapama uygular."""
        mask_uint8 = (mask * 255).astype(np.uint8)
        kernel = np.ones((self.kernel_size, self.kernel_size), np.uint8)

        opened = cv2.morphologyEx(mask_uint8, cv2.MORPH_OPEN, kernel)
        closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        return (closed > 127).astype(np.float32)

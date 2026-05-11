from __future__ import annotations

import cv2
import numpy as np
import torch
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

from .image_utils import ImageProcessor
from .nst_model import NeuralStyleTransfer
from .segmenter import PersonSegmenter


class SemanticStyleTransferPipeline:
    """
    Projenin ana OOP pipeline sınıfı.

    Akış:
    1. Content ve style görsellerini hazırlar.
    2. Content görselindeki insan bölgesini DeepLabV3 ile bulur.
    3. VGG19 ile tüm görsele Neural Style Transfer uygular.
    4. İnsan bölgesini orijinalden, arka planı stilize görselden alarak semantic sonuç üretir.
    """

    def __init__(self, image_size: int = 384, device: torch.device | None = None) -> None:
        self.image_size = image_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_processor = ImageProcessor()
        self.segmenter = PersonSegmenter(device=self.device)
        self.nst_model = NeuralStyleTransfer(device=self.device)

    def run(
        self,
        content_image: Image.Image,
        style_image: Image.Image,
        steps: int = 150,
        content_weight: float = 5.0,
        style_weight: float = 100000.0,
        blur_size: int = 21,
        learning_rate: float = 0.03,
    ) -> dict:
        """Tam semantic style transfer sürecini çalıştırır."""
        content_image = self.image_processor.resize_image(content_image, self.image_size)
        style_image = self.image_processor.resize_image(style_image, self.image_size)

        content_tensor = self.image_processor.pil_to_tensor(content_image, self.device)
        style_tensor = self.image_processor.pil_to_tensor(style_image, self.device)
        content_np = self.image_processor.pil_to_numpy(content_image)

        person_mask = self.segmenter.predict_mask(content_image)

        styled_tensor, loss_history = self.nst_model.transfer(
            content_tensor=content_tensor,
            style_tensor=style_tensor,
            steps=steps,
            content_weight=content_weight,
            style_weight=style_weight,
            learning_rate=learning_rate,
        )

        styled_image = self.image_processor.tensor_to_pil(styled_tensor)
        styled_np = self.image_processor.tensor_to_numpy(styled_tensor)

        semantic_np, soft_mask = self._combine_person_and_styled_background(
            content_np=content_np,
            styled_np=styled_np,
            person_mask=person_mask,
            blur_size=blur_size,
        )

        semantic_image = self.image_processor.numpy_to_pil(semantic_np)
        metrics = self._calculate_metrics(content_np, styled_np, semantic_np, person_mask)

        return {
            "content_image": content_image,
            "style_image": style_image,
            "person_mask": person_mask,
            "soft_mask": soft_mask,
            "full_styled_image": styled_image,
            "semantic_image": semantic_image,
            "loss_history": loss_history,
            "metrics": metrics,
        }

    @staticmethod
    def _combine_person_and_styled_background(
        content_np: np.ndarray,
        styled_np: np.ndarray,
        person_mask: np.ndarray,
        blur_size: int = 21,
    ) -> tuple[np.ndarray, np.ndarray]:
        """İnsanı orijinalden, arka planı stilize görselden alır."""
        height, width = content_np.shape[:2]

        styled_np = cv2.resize(styled_np, (width, height))
        mask = cv2.resize(person_mask.astype(np.float32), (width, height), interpolation=cv2.INTER_NEAREST)

        if blur_size % 2 == 0:
            blur_size += 1

        soft_mask = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
        soft_mask = np.clip(soft_mask, 0, 1)
        mask_3ch = np.repeat(soft_mask[:, :, None], 3, axis=2)

        semantic_np = content_np * mask_3ch + styled_np * (1 - mask_3ch)
        return np.clip(semantic_np, 0, 1), soft_mask

    @staticmethod
    def _calculate_metrics(
        content_np: np.ndarray,
        styled_np: np.ndarray,
        semantic_np: np.ndarray,
        person_mask: np.ndarray,
    ) -> dict:
        """Full NST ve Semantic NST için temel metrikleri hesaplar."""
        def image_metrics(reference: np.ndarray, output: np.ndarray) -> dict:
            return {
                "mse": float(np.mean((reference - output) ** 2)),
                "psnr": float(peak_signal_noise_ratio(reference, output, data_range=1.0)),
                "ssim": float(structural_similarity(reference, output, channel_axis=2, data_range=1.0)),
            }

        def person_mse(reference: np.ndarray, output: np.ndarray, mask: np.ndarray) -> float:
            mask_3ch = np.repeat(mask[:, :, None], 3, axis=2)
            denominator = np.sum(mask_3ch) + 1e-8
            return float(np.sum(((reference - output) ** 2) * mask_3ch) / denominator)

        return {
            "full_nst": {
                **image_metrics(content_np, styled_np),
                "person_mse": person_mse(content_np, styled_np, person_mask),
            },
            "semantic_nst": {
                **image_metrics(content_np, semantic_np),
                "person_mse": person_mse(content_np, semantic_np, person_mask),
            },
        }

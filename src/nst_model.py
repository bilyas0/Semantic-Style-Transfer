from __future__ import annotations

import torch
import torch.nn.functional as F
from torchvision.models import VGG19_Weights, vgg19


class NeuralStyleTransfer:
    """VGG19 tabanlı Neural Style Transfer sınıfı."""

    STYLE_LAYERS = {
        "0": "conv1_1",
        "5": "conv2_1",
        "10": "conv3_1",
        "19": "conv4_1",
        "28": "conv5_1",
    }

    CONTENT_LAYERS = {
        "21": "conv4_2",
    }

    STYLE_LAYER_WEIGHTS = {
        "conv1_1": 1.0,
        "conv2_1": 0.8,
        "conv3_1": 0.5,
        "conv4_1": 0.3,
        "conv5_1": 0.1,
    }

    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.layers = {**self.STYLE_LAYERS, **self.CONTENT_LAYERS}

        weights = VGG19_Weights.DEFAULT
        self.vgg = vgg19(weights=weights).features.to(device).eval()

        for parameter in self.vgg.parameters():
            parameter.requires_grad_(False)

        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def transfer(
        self,
        content_tensor: torch.Tensor,
        style_tensor: torch.Tensor,
        steps: int = 150,
        content_weight: float = 5.0,
        style_weight: float = 100000.0,
        tv_weight: float = 1e-6,
        learning_rate: float = 0.03,
    ) -> tuple[torch.Tensor, dict[str, list[float]]]:
        """Content ve style tensörlerinden stil aktarımı sonucu üretir."""
        content_features = self._extract_features(self._normalize(content_tensor))
        style_features = self._extract_features(self._normalize(style_tensor))

        style_grams = {
            layer_name: self._gram_matrix(style_features[layer_name])
            for layer_name in self.STYLE_LAYER_WEIGHTS
        }

        target = content_tensor.clone().detach().requires_grad_(True)
        optimizer = torch.optim.Adam([target], lr=learning_rate)

        history = {
            "total_loss": [],
            "content_loss": [],
            "style_loss": [],
            "tv_loss": [],
        }

        for step in range(1, steps + 1):
            optimizer.zero_grad()

            target_features = self._extract_features(self._normalize(target))

            content_loss = F.mse_loss(
                target_features["conv4_2"],
                content_features["conv4_2"],
            )

            style_loss = 0.0
            for layer_name, layer_weight in self.STYLE_LAYER_WEIGHTS.items():
                target_gram = self._gram_matrix(target_features[layer_name])
                style_gram = style_grams[layer_name]
                style_loss = style_loss + layer_weight * F.mse_loss(target_gram, style_gram)

            tv_loss = self._total_variation_loss(target)
            total_loss = (
                content_weight * content_loss
                + style_weight * style_loss
                + tv_weight * tv_loss
            )

            total_loss.backward()
            optimizer.step()

            with torch.no_grad():
                target.clamp_(0, 1)

            history["total_loss"].append(float(total_loss.item()))
            history["content_loss"].append(float(content_loss.item()))
            history["style_loss"].append(float(style_loss.item()))
            history["tv_loss"].append(float(tv_loss.item()))

            if step == 1 or step % max(1, steps // 5) == 0:
                print(
                    f"Step {step}/{steps} | "
                    f"Total: {total_loss.item():.4f} | "
                    f"Content: {content_loss.item():.4f} | "
                    f"Style: {style_loss.item():.8f}"
                )

        return target.detach(), history

    def _normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

    def _extract_features(self, tensor: torch.Tensor) -> dict[str, torch.Tensor]:
        features = {}
        x = tensor

        for layer_index, layer in self.vgg._modules.items():
            x = layer(x)
            if layer_index in self.layers:
                features[self.layers[layer_index]] = x

        return features

    @staticmethod
    def _gram_matrix(tensor: torch.Tensor) -> torch.Tensor:
        batch_size, channels, height, width = tensor.size()
        features = tensor.view(batch_size, channels, height * width)
        gram = torch.bmm(features, features.transpose(1, 2))
        return gram / (channels * height * width)

    @staticmethod
    def _total_variation_loss(tensor: torch.Tensor) -> torch.Tensor:
        horizontal_loss = torch.mean(torch.abs(tensor[:, :, :, :-1] - tensor[:, :, :, 1:]))
        vertical_loss = torch.mean(torch.abs(tensor[:, :, :-1, :] - tensor[:, :, 1:, :]))
        return horizontal_loss + vertical_loss

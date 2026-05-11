from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

from src.semantic_style_transfer import SemanticStyleTransferPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Semantic Background Neural Style Transfer")

    parser.add_argument("--content", required=True, help="Content image path")
    parser.add_argument("--style", required=True, help="Style image path")
    parser.add_argument("--output", default="outputs/result.png", help="Output image path")

    parser.add_argument("--image-size", type=int, default=384, help="Image size: 256, 384 or 512 recommended")
    parser.add_argument("--steps", type=int, default=150, help="NST optimization steps")
    parser.add_argument("--content-weight", type=float, default=5.0)
    parser.add_argument("--style-weight", type=float, default=100000.0)
    parser.add_argument("--blur-size", type=int, default=21)
    parser.add_argument("--learning-rate", type=float, default=0.03)

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    content_image = Image.open(args.content).convert("RGB")
    style_image = Image.open(args.style).convert("RGB")

    pipeline = SemanticStyleTransferPipeline(image_size=args.image_size)

    result = pipeline.run(
        content_image=content_image,
        style_image=style_image,
        steps=args.steps,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        blur_size=args.blur_size,
        learning_rate=args.learning_rate,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result["semantic_image"].save(output_path)

    mask_path = output_path.parent / "person_mask.png"
    styled_path = output_path.parent / "full_styled_result.png"

    result["full_styled_image"].save(styled_path)
    result["semantic_image"].save(output_path)

    import matplotlib.pyplot as plt

    plt.imsave(mask_path, result["person_mask"], cmap="gray")

    print("İşlem tamamlandı.")
    print("Semantic sonuç:", output_path)
    print("Full NST sonuç:", styled_path)
    print("Person mask:", mask_path)
    print("Metrikler:", result["metrics"])


if __name__ == "__main__":
    main()

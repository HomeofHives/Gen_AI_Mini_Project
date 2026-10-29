"""Task 5: Text-to-Image ad generation using Stable Diffusion and OpenAI."""

import argparse
import base64
from datetime import datetime
import io
import json
import os
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image


DEFAULT_PROMPT = (
    "Create a premium advertisement for wireless headphones designed for college students, "
    "displayed in a modern university environment, professional product photography, "
    "attractive lighting and a clean composition."
)

DEFAULT_NEGATIVE_PROMPT = "blurry, low quality, distorted, deformed, text, watermark, extra limbs"


def generate_sd(
    prompt: str = DEFAULT_PROMPT,
    negative_prompt: str = DEFAULT_NEGATIVE_PROMPT,
    out_dir: Path = Path("results/ads"),
    steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int = 42,
    size: Tuple[int, int] = (512, 512),
) -> Tuple[Path, Path]:
    """Generate ad image using Stable Diffusion v1.5 via diffusers pipeline.

    Args:
        prompt (str): Generation text prompt.
        negative_prompt (str): Negative guidance prompt.
        out_dir (Path): Output directory for image and metadata.
        steps (int): Diffusion inference steps.
        guidance_scale (float): Classifier-free guidance weight.
        seed (int): Deterministic random seed.
        size (Tuple[int, int]): Image dimensions (width, height).

    Returns:
        Tuple[Path, Path]: Saved (image_path, json_path).
    """
    import torch
    from diffusers import StableDiffusionPipeline

    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.float16 if device == "cuda" else torch.float32

    model_id = "stable-diffusion-v1-5/stable-diffusion-v1-5"
    print(f"Loading Stable Diffusion pipeline ({model_id}) on {device} ({torch_dtype})...")

    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
        )
    except Exception:
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch_dtype,
            safety_checker=None,
        )

    pipe = pipe.to(device)

    # Use deterministic generator
    generator = torch.Generator(device=device).manual_seed(seed)

    print(f"Generating image with {steps} steps, guidance={guidance_scale}, seed={seed}...")
    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        num_inference_steps=steps,
        guidance_scale=guidance_scale,
        generator=generator,
        width=size[0],
        height=size[1],
    )
    image: Image.Image = result.images[0]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_filename = f"ad_sd_{timestamp}.png"
    json_filename = f"ad_sd_{timestamp}.json"

    img_path = out_dir / img_filename
    json_path = out_dir / json_filename

    image.save(img_path)

    metadata = {
        "backend": "stable-diffusion",
        "model_id": model_id,
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "steps": steps,
        "guidance": guidance_scale,
        "seed": seed,
        "size": list(size),
        "timestamp": timestamp,
        "device": device,
        "torch_version": torch.__version__,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved Stable Diffusion image to {img_path}")
    print(f"Saved metadata to {json_path}")
    return img_path, json_path


def generate_openai(
    prompt: str = DEFAULT_PROMPT,
    out_dir: Path = Path("results/ads"),
    seed: Optional[int] = None,
) -> Tuple[Path, Path]:
    """Generate ad image using OpenAI Images API.

    Args:
        prompt (str): Generation prompt text.
        out_dir (Path): Output directory for image and metadata.
        seed (Optional[int]): Seed value for reference.

    Returns:
        Tuple[Path, Path]: Saved (image_path, json_path).
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY environment variable is not set. "
            "Please set your OpenAI API key in your environment before using the OpenAI backend."
        )

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating image with OpenAI Images API (model: gpt-image-1, size: 1024x1024)...")
    response = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size="1024x1024",
        response_format="b64_json",
    )

    b64_data = response.data[0].b64_json
    image_bytes = base64.b64decode(b64_data)
    image = Image.open(io.BytesIO(image_bytes))

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    img_filename = f"ad_openai_{timestamp}.png"
    json_filename = f"ad_openai_{timestamp}.json"

    img_path = out_dir / img_filename
    json_path = out_dir / json_filename

    image.save(img_path)

    metadata = {
        "backend": "openai",
        "model_id": "gpt-image-1",
        "prompt": prompt,
        "negative_prompt": None,
        "steps": None,
        "guidance": None,
        "seed": seed,
        "size": [1024, 1024],
        "timestamp": timestamp,
        "device": "cloud",
        "torch_version": None,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved OpenAI image to {img_path}")
    print(f"Saved metadata to {json_path}")
    return img_path, json_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate advertisement image via SD or OpenAI")
    parser.add_argument(
        "--backend",
        choices=["sd", "openai"],
        default="sd",
        help="Backend to use: 'sd' (Stable Diffusion) or 'openai'",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=DEFAULT_PROMPT,
        help="Custom text prompt",
    )
    parser.add_argument(
        "--negative-prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt for SD",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed for generation",
    )
    args = parser.parse_args()

    if args.backend == "sd":
        generate_sd(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            seed=args.seed,
        )
    elif args.backend == "openai":
        generate_openai(
            prompt=args.prompt,
            seed=args.seed,
        )

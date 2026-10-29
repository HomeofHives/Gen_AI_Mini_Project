"""Task 4: GPT-2 product description generation using Hugging Face transformers."""

import argparse
from pathlib import Path
from typing import Optional, Tuple

from src.metrics import save_metrics


PROMPT_TEMPLATE = (
    "Product: Running shoes\n"
    "Features: lightweight, breathable mesh, cushioned sole\n"
    "Audience: beginner runners\n"
    "Advertisement: Hit the pavement with confidence in every stride. Engineered with ultra-breathable "
    "mesh and responsive cushioning, these lightweight running shoes protect your joints while keeping you moving "
    "comfortably. Start your running journey on the right foot today.\n\n"
    "Product: Stainless steel water bottle\n"
    "Features: keeps drinks cold for 24 hours, leak-proof lid, fits in a backpack\n"
    "Audience: hikers\n"
    "Advertisement: Conquer the trail without ever losing your cool. Designed with double-wall insulation to "
    "keep your hydration ice-cold for a full 24 hours, this rugged bottle slips effortlessly into any pack. "
    "Reliable, leak-proof, and built for your wildest adventures.\n\n"
    "Product: Wireless headphones\n"
    "Features: noise cancellation, lightweight, long battery life\n"
    "Audience: college students\n"
    "Advertisement:"
)

TARGET_KEYWORDS = ["noise", "light", "battery", "headphone", "student"]


def post_process_generation(raw_text: str) -> str:
    """Post-process generated text:

    - Keep text up to the first blank line or next 'Product:' line.
    - Trim to the last complete sentence ending in ., !, or ?.
    """
    clean_text = raw_text.strip()
    lines = clean_text.split("\n")
    kept_chunks = []
    for line in lines:
        stripped = line.strip()
        if not stripped:
            break
        if stripped.lower().startswith("product:"):
            break
        kept_chunks.append(stripped)

    merged = " ".join(kept_chunks).strip()

    # Find the last sentence-ending punctuation (. ! ?)
    last_idx = max(merged.rfind("."), merged.rfind("!"), merged.rfind("?"))
    if last_idx != -1:
        merged = merged[: last_idx + 1].strip()

    return merged


def generate_description(
    seed: int = 42,
    device: Optional[str] = None,
    out_path: Path = Path("results/text/product_description.txt"),
) -> Tuple[str, int]:
    """Generate product advertisement using gpt2-medium few-shot prompting."""
    import torch
    import transformers
    from transformers import pipeline

    transformers.set_seed(seed)

    # Determine device index (-1 for CPU, 0+ for GPU)
    if device is None:
        device_idx = 0 if torch.cuda.is_available() else -1
    elif device.lower() in ("cuda", "gpu") and torch.cuda.is_available():
        device_idx = 0
    else:
        device_idx = -1

    print(f"Loading pipeline for 'gpt2-medium' on device index {device_idx}...")
    text_generator = pipeline(
        "text-generation",
        model="gpt2-medium",
        device=device_idx,
    )

    outputs = text_generator(
        PROMPT_TEMPLATE,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.8,
        top_p=0.92,
        repetition_penalty=1.15,
        return_full_text=False,
    )

    generated_continuation = outputs[0]["generated_text"]
    result_text = post_process_generation(generated_continuation)

    if not result_text:
        raise ValueError("Generation produced empty text after post-processing.")

    # Write output to file
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(result_text + "\n")
    print(f"Saved product description to {out_path}\n")

    # Print output
    print("-" * 60)
    print("Generated Advertisement:")
    print(result_text)
    print("-" * 60)

    # Keyword check
    lower_text = result_text.lower()
    keyword_hits = sum(1 for kw in TARGET_KEYWORDS if kw in lower_text)
    print(f"Keyword hits ({keyword_hits}/5): {[kw for kw in TARGET_KEYWORDS if kw in lower_text]}")
    if keyword_hits < 2:
        print(f"WARNING: Fewer than two target keywords found in generated text (hits={keyword_hits})!")

    save_metrics("gpt_keyword_hits", keyword_hits)
    return result_text, keyword_hits


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate product description with GPT-2 medium")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for generation")
    args = parser.parse_args()

    generate_description(seed=args.seed)

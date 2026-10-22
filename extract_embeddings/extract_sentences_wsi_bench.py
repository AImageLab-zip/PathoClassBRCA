import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch

from utils.utils import init_seeds
from plip.plip import PLIP


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract sentences from WSI-Bench reports and build a PLIP text knowledge bank."
    )

    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input WSI-Bench JSON file.",
    )

    parser.add_argument(
        "--output_sentences",
        type=str,
        required=True,
        help="Path where the extracted sentences JSON will be saved.",
    )

    parser.add_argument(
        "--output_embeddings",
        type=str,
        required=True,
        help="Path where the PLIP embeddings (.pt) will be saved.",
    )

    parser.add_argument(
        "--plip_checkpoint",
        type=str,
        required=True,
        help="Path to the PLIP checkpoint directory.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for PLIP text encoding. Default: 32.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42.",
    )

    return parser.parse_args()


def extract_sentences(data):
    """
    Extract GPT-generated text from WSI-Bench conversations
    and split it into individual sentences.
    """
    sentences = []

    for item in data:
        for conv in item.get("conversations", []):
            if conv.get("from") != "gpt":
                continue

            text = conv.get("value", "").strip()

            if not text:
                continue

            # Split on sentence-ending punctuation
            current_sentences = re.split(r"(?<=[.!?])\s+", text)

            current_sentences = [
                sentence.strip()
                for sentence in current_sentences
                if sentence.strip()
            ]

            sentences.extend(current_sentences)

    return sentences


def save_sentences(sentences, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            sentences,
            f,
            indent=4,
            ensure_ascii=False,
        )

    print(f"Saved {len(sentences)} sentences to:")
    print(f"  {output_path}")


def build_knowledge_bank(
    sentences,
    plip_checkpoint,
    batch_size,
):
    """
    Encode sentences using PLIP and L2-normalize the embeddings.
    """

    print("\nLoading PLIP...")
    plip = PLIP(plip_checkpoint)

    print(f"Encoding {len(sentences)} sentences...")
    print(f"Batch size: {batch_size}")

    text_embeddings = plip.encode_text(
        sentences,
        batch_size=batch_size,
    )

    # L2 normalization
    norms = np.linalg.norm(
        text_embeddings,
        ord=2,
        axis=-1,
        keepdims=True,
    )

    text_embeddings = text_embeddings / np.clip(
        norms,
        a_min=1e-12,
        a_max=None,
    )

    text_embeddings = torch.from_numpy(
        text_embeddings
    ).float()

    return text_embeddings


def save_embeddings(embeddings, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(embeddings, output_path)

    print("\nSaved knowledge bank to:")
    print(f"  {output_path}")
    print(f"Knowledge bank shape: {tuple(embeddings.shape)}")


def main():
    args = parse_args()

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    init_seeds(args.seed)

    # ---------------------------------------------------------
    # Load input JSON
    # ---------------------------------------------------------

    print(f"Loading input file:\n  {args.input_file}")

    with open(args.input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # ---------------------------------------------------------
    # Extract sentences
    # ---------------------------------------------------------

    sentences = extract_sentences(data)

    if len(sentences) == 0:
        raise ValueError(
            "No GPT sentences were found in the input JSON."
        )

    save_sentences(
        sentences,
        args.output_sentences,
    )

    # ---------------------------------------------------------
    # Build PLIP knowledge bank
    # ---------------------------------------------------------

    text_embeddings = build_knowledge_bank(
        sentences=sentences,
        plip_checkpoint=args.plip_checkpoint,
        batch_size=args.batch_size,
    )

    save_embeddings(
        text_embeddings,
        args.output_embeddings,
    )


if __name__ == "__main__":
    main()
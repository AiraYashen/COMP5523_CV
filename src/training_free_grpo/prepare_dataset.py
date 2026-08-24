from __future__ import annotations

import argparse

from src.training_free_grpo.dataset import export_seed_dataset_from_logs


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export a training-free GRPO seed dataset from logged VLM inputs"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="outputs/vlm_inputs",
        help="Directory containing logged VLM metadata JSON files",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/training_free_grpo/train_dataset.jsonl",
        help="Target JSONL path",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    export_seed_dataset_from_logs(args.log_dir, args.output_path)


if __name__ == "__main__":
    main()

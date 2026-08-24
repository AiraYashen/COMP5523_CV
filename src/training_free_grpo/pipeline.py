from __future__ import annotations

import argparse

from src.training_free_grpo.dataset import export_seed_dataset_from_logs
from src.training_free_grpo.trainer import TrainingFreeGRPOTrainer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-shot dataset export + training-free GRPO training pipeline"
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="outputs/vlm_inputs",
        help="Source VLM log directory",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="outputs/training_free_grpo/train_dataset.jsonl",
        help="Intermediate JSONL dataset path",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/training_free_grpo",
        help="Directory for rollouts and learned profiles",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--rollout-temperature", type=float, default=None)
    parser.add_argument("--rollout-concurrency", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    export_seed_dataset_from_logs(args.log_dir, args.dataset_path)
    trainer = TrainingFreeGRPOTrainer()
    trainer.train(
        dataset_path=args.dataset_path,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        group_size=args.group_size,
        rollout_temperature=args.rollout_temperature,
        rollout_concurrency=args.rollout_concurrency,
    )


if __name__ == "__main__":
    main()

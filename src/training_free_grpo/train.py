from __future__ import annotations

import argparse

from src.training_free_grpo.trainer import TrainingFreeGRPOTrainer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Training-free GRPO prompt trainer")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="JSONL dataset generated from VLM logs and optional reference answers",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/training_free_grpo",
        help="Directory for step rollouts and learned profiles",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--group-size", type=int, default=None)
    parser.add_argument("--rollout-temperature", type=float, default=None)
    parser.add_argument("--rollout-concurrency", type=int, default=None)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    trainer = TrainingFreeGRPOTrainer()
    trainer.train(
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        group_size=args.group_size,
        rollout_temperature=args.rollout_temperature,
        rollout_concurrency=args.rollout_concurrency,
    )


if __name__ == "__main__":
    main()

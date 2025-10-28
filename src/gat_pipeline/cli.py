from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

from .config import PipelineConfig, load_config
from .data import (
    build_fold_graphs,
    convert_fasta_to_bingo_format,
    generate_embeddings,
    generate_kfold_splits,
)
from .explain import run_node_explainer
from .inference import infer_fasta, infer_sequence
from .training import train_fold


def _load_config(path: Optional[str]) -> PipelineConfig:
    return load_config(path)


def _prepare_data(args: argparse.Namespace) -> None:
    config = _load_config(args.config)

    if args.pathogenesis_fasta and args.non_pathogenesis_fasta and not args.skip_dataset:
        convert_fasta_to_bingo_format(
            args.pathogenesis_fasta,
            args.non_pathogenesis_fasta,
            config.root_path / config.species,
        )

    if not args.skip_embeddings:
        generate_embeddings(config)

    if not args.skip_splits:
        generate_kfold_splits(config)

    if not args.skip_graphs:
        build_fold_graphs(config)


def _train_fold(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    if args.model and args.model not in {"gat", "gcn", "sage"}:
        raise ValueError("Model must be one of: gat, gcn, sage")
    summary = train_fold(
        config,
        fold=args.fold,
        model_name=args.model,
        use_wandb=False if args.no_wandb else None,
    )
    payload = {
        "best_aupr_path": str(summary.best_aupr_path) if summary.best_aupr_path else None,
        "best_loss_path": str(summary.best_loss_path) if summary.best_loss_path else None,
    }
    print(json.dumps(payload, indent=2))


def _infer_sequence(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    if args.sequence_file:
        sequence = Path(args.sequence_file).read_text().strip()
    else:
        sequence = args.sequence
    if not sequence:
        raise ValueError("Provide --sequence or --sequence-file")
    result = infer_sequence(
        sequence=sequence,
        sequence_id=args.name,
        checkpoint_path=Path(args.model_checkpoint),
        config=config,
    )
    print(json.dumps(result.__dict__, indent=2))


def _infer_fasta(args: argparse.Namespace) -> None:
    config = _load_config(args.config)
    results = infer_fasta(
        fasta_path=Path(args.fasta),
        checkpoint_path=Path(args.model_checkpoint),
        config=config,
        output_csv=Path(args.output),
    )
    print(f"Wrote {len(results)} predictions to {args.output}")


def _explain_nodes(args: argparse.Namespace) -> None:
    config = _load_config(args.config) if args.config else None
    if args.sequence_file:
        sequence = Path(args.sequence_file).read_text().strip()
    else:
        sequence = args.sequence
    if not sequence:
        raise ValueError("Provide --sequence or --sequence-file")
    ratio = args.ratio
    if ratio is None and config is not None:
        ratio = config.ratio
    ratio = ratio if ratio is not None else 0.2
    run_node_explainer(
        sequence=sequence,
        model_path=Path(args.model_checkpoint),
        output_name=args.name,
        output_dir=Path(args.output_dir),
        ratio=ratio,
        top_fraction=args.top_fraction,
        steps=args.steps,
        epochs=args.epochs,
        seed=args.seed,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="gat-pipeline", description="Utilities for the GAT fungal pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare-data", help="Prepare embeddings, splits and graphs")
    prepare_parser.add_argument("--config", default="configs/fungi.yaml")
    prepare_parser.add_argument("--pathogenesis-fasta", default=None)
    prepare_parser.add_argument("--non-pathogenesis-fasta", default=None)
    prepare_parser.add_argument("--skip-dataset", action="store_true")
    prepare_parser.add_argument("--skip-embeddings", action="store_true")
    prepare_parser.add_argument("--skip-splits", action="store_true")
    prepare_parser.add_argument("--skip-graphs", action="store_true")
    prepare_parser.set_defaults(func=_prepare_data)

    train_parser = subparsers.add_parser("train-fold", help="Train a specific fold")
    train_parser.add_argument("--config", default="configs/fungi.yaml")
    train_parser.add_argument("--fold", type=int, required=True)
    train_parser.add_argument("--model", default=None)
    train_parser.add_argument("--no-wandb", action="store_true")
    train_parser.set_defaults(func=_train_fold)

    single_parser = subparsers.add_parser("infer-sequence", help="Infer a single protein sequence")
    single_parser.add_argument("--config", default="configs/fungi.yaml")
    single_parser.add_argument("--sequence", default=None)
    single_parser.add_argument("--sequence-file", default=None)
    single_parser.add_argument("--name", required=True)
    single_parser.add_argument("--model-checkpoint", required=True)
    single_parser.set_defaults(func=_infer_sequence)

    batch_parser = subparsers.add_parser("infer-fasta", help="Infer all sequences in a FASTA file")
    batch_parser.add_argument("--config", default="configs/fungi.yaml")
    batch_parser.add_argument("--fasta", required=True)
    batch_parser.add_argument("--model-checkpoint", required=True)
    batch_parser.add_argument("--output", required=True)
    batch_parser.set_defaults(func=_infer_fasta)

    explain_parser = subparsers.add_parser("explain-nodes", help="Run GNNExplainer on a single sequence")
    explain_parser.add_argument("--config", default=None)
    explain_parser.add_argument("--sequence", default=None)
    explain_parser.add_argument("--sequence-file", default=None)
    explain_parser.add_argument("--name", required=True)
    explain_parser.add_argument("--model-checkpoint", required=True)
    explain_parser.add_argument("--output-dir", default="gnn_results")
    explain_parser.add_argument("--ratio", default=None, type=float)
    explain_parser.add_argument("--top-fraction", default=0.1, type=float)
    explain_parser.add_argument("--steps", default=11, type=int)
    explain_parser.add_argument("--epochs", default=None, type=int)
    explain_parser.add_argument("--seed", default=42, type=int)
    explain_parser.set_defaults(func=_explain_nodes)

    return parser


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":  # pragma: no cover
    main()

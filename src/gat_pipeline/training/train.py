from __future__ import annotations

import json
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch_geometric.loader import DataLoader

from ..config import PipelineConfig
from ..data import ProteinGraphDataset
from ..models import FGM_GAT, FGM_GCN, FGM_SAGE, GATNet, GCNNet, SAGENet
from ..utils import compute_classification_metrics, ensure_dir

try:  # Optional dependency management
    import wandb
except ImportError:  # pragma: no cover - optional integration
    wandb = None  # type: ignore


@dataclass
class TrainSummary:
    best_aupr_path: Optional[Path]
    best_loss_path: Optional[Path]
    history: Dict[str, Any]


def _save_checkpoint(model: torch.nn.Module, path: Path, config: PipelineConfig, fold: int, model_name: str) -> None:
    torch.save(model.state_dict(), path)
    metadata = {
        "model": model_name,
        "fold": fold,
        "drop_prob": config.drop_prob,
        "ratio": config.ratio,
        "lr": config.lr,
        "weight_decay": config.weight_decay,
        "train_batch_size": config.train_batch_size,
        "test_batch_size": config.test_batch_size,
        "timestamp": datetime.utcnow().isoformat(),
    }
    meta_path = path.with_name(path.name + ".meta.json")
    with meta_path.open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def _make_model(model_name: str, drop_prob: float, n_output: int) -> tuple[torch.nn.Module, Any]:
    if model_name == "gcn":
        model = GCNNet(esm_embeds=1280, drop_prob=drop_prob, n_output=n_output)
        fgm = FGM_GCN(model)
    elif model_name == "gat":
        model = GATNet(esm_embeds=1280, n_heads=2, drop_prob=drop_prob, n_output=n_output)
        fgm = FGM_GAT(model)
    elif model_name == "sage":
        model = SAGENet(esm_embeds=1280, drop_prob=drop_prob, n_output=n_output)
        fgm = FGM_SAGE(model)
    else:
        raise ValueError(f"Unsupported model '{model_name}'")
    return model, fgm


def _adversarial_training(model, loader, device, optimizer, loss_fn, fgm_model):
    model.train()
    epoch_loss = 0.0
    epoch_loss_adv = 0.0
    train_num = 0
    preds = []
    labels = []

    for batch in loader:
        batch = batch.to(device)
        batch_y = batch.y.view(-1).float()
        optimizer.zero_grad()

        out = model(batch).view(-1)
        loss = loss_fn(out, batch_y)
        loss.backward()

        fgm_model.attack()
        out_adv = model(batch).view(-1)
        loss_adv = loss_fn(out_adv, batch_y)
        loss_adv.backward()
        fgm_model.restore()

        optimizer.step()

        batch_size = batch.num_graphs
        train_num += batch_size
        epoch_loss += loss.item() * batch_size
        epoch_loss_adv += loss_adv.item() * batch_size

        with torch.no_grad():
            preds.append(torch.sigmoid(out).detach().cpu().numpy())
            labels.append(batch_y.detach().cpu().numpy())

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    preds_arr = np.concatenate(preds)
    labels_arr = np.concatenate(labels)
    metrics = compute_classification_metrics(labels_arr, preds_arr)

    return epoch_loss / train_num, epoch_loss_adv / train_num, metrics


def _predict_with_loss(loader, model, device, loss_fn):
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            batch_y = batch.y.view(-1).float()
            output = model(batch).view(-1)
            loss = loss_fn(output, batch_y)
            total_loss += loss.item() * batch.num_graphs
            total_samples += batch.num_graphs
            all_preds.append(output.detach().cpu().numpy())
            all_labels.append(batch_y.detach().cpu().numpy())

    preds_arr = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    metrics = compute_classification_metrics(labels_arr, preds_arr)
    avg_loss = total_loss / max(total_samples, 1)
    return metrics, avg_loss


def _predict(loader, model, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch).view(-1)
            all_preds.append(output.detach().cpu().numpy())
            all_labels.append(batch.y.view(-1).detach().cpu().numpy())

    preds_arr = np.concatenate(all_preds)
    labels_arr = np.concatenate(all_labels)
    return compute_classification_metrics(labels_arr, preds_arr)


def train_fold(
    config: PipelineConfig,
    fold: int,
    model_name: Optional[str] = None,
    use_wandb: Optional[bool] = None,
    wandb_run_name: Optional[str] = None,
) -> TrainSummary:
    """Run adversarial training for a specific fold."""

    model_name = model_name or config.model
    device = torch.device(config.cuda_name if torch.cuda.is_available() else "cpu")
    fold_path = config.kfold_root_path / f"fold{fold}"
    processed_train = fold_path / "processed" / "train.pt"
    processed_test = fold_path / "processed" / "test.pt"
    if not processed_train.exists() or not processed_test.exists():
        raise FileNotFoundError(
            "Processed graph data missing. Run gat-pipeline prepare-data first."
        )

    train_data_path = fold_path / "train_data.txt"
    test_data_path = fold_path / "test_data.txt"
    train_list = pd.read_csv(train_data_path, sep="\t")["GeneSymbol"].tolist()
    test_list = pd.read_csv(test_data_path, sep="\t")["GeneSymbol"].tolist()

    full_train_dataset = ProteinGraphDataset(
        root=fold_path,
        gene_list=train_list,
        mode="train",
        ratio=config.ratio,
        raw_data_path=config.raw_data_path,
    )
    test_dataset = ProteinGraphDataset(
        root=fold_path,
        gene_list=test_list,
        mode="test",
        ratio=config.ratio,
        raw_data_path=config.raw_data_path,
    )

    train_size = int(0.8 * len(full_train_dataset))
    valid_size = len(full_train_dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        full_train_dataset, [train_size, valid_size]
    )

    train_loader = DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config.test_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.test_batch_size, shuffle=False)

    model, fgm_model = _make_model(model_name, config.drop_prob, n_output=1)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    loss_fn = nn.BCEWithLogitsLoss().to(device)

    run_wandb = use_wandb if use_wandb is not None else bool(config.use_wandb)
    wandb_run = None
    if run_wandb and wandb is not None:
        wandb_config = config.to_dict()
        wandb_config.update({"fold": fold, "model": model_name})
        run_name = wandb_run_name or f"{config.wandb_run_prefix}_{model_name}_fold_{fold}"
        wandb_run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=run_name,
            config=wandb_config,
        )

    best_val_aupr = 0.0
    best_val_loss = float("inf")

    model_dir = config.experiments_dir / model_name / f"fold_{fold}"
    ensure_dir(model_dir)
    best_aupr_path = model_dir / "best_aupr.pt"
    best_loss_path = model_dir / "best_loss.pt"

    history: Dict[str, Any] = {"epochs": []}

    for epoch in range(config.num_epochs):
        epoch_loss, epoch_loss_adv, train_metrics = _adversarial_training(
            model, train_loader, device, optimizer, loss_fn, fgm_model
        )
        val_metrics, val_loss = _predict_with_loss(valid_loader, model, device, loss_fn)

        metrics_dict = {
            "epoch": epoch,
            "train_loss": epoch_loss,
            "train_loss_adv": epoch_loss_adv,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "val_loss": val_loss,
        }
        history["epochs"].append(metrics_dict)

        if run_wandb and wandb_run is not None:
            log_payload = {
                "epoch": epoch,
                "train/loss": epoch_loss,
                "train/loss_adv": epoch_loss_adv,
                "validation/loss": val_loss,
                "train/auc": train_metrics[6],
                "train/aupr": train_metrics[7],
                "train/f1": train_metrics[8],
                "train/accuracy": train_metrics[9],
                "train/recall": train_metrics[10],
                "train/specificity": train_metrics[11],
                "train/precision": train_metrics[12],
                "validation/auc": val_metrics[6],
                "validation/aupr": val_metrics[7],
                "validation/f1": val_metrics[8],
                "validation/accuracy": val_metrics[9],
                "validation/recall": val_metrics[10],
                "validation/specificity": val_metrics[11],
                "validation/precision": val_metrics[12],
            }
            wandb_run.log(log_payload)

        val_aupr = val_metrics[7]
        if val_aupr > best_val_aupr:
            best_val_aupr = val_aupr
            _save_checkpoint(model, best_aupr_path, config, fold, model_name)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            _save_checkpoint(model, best_loss_path, config, fold, model_name)

    best_aupr_model = _load_model(model_name, best_aupr_path, config.drop_prob).to(device)
    best_loss_model = _load_model(model_name, best_loss_path, config.drop_prob).to(device)
    test_metrics_aupr = _predict(test_loader, best_aupr_model, device)
    test_metrics_loss = _predict(test_loader, best_loss_model, device)

    history["test"] = {
        "best_aupr": test_metrics_aupr,
        "best_loss": test_metrics_loss,
    }

    with (model_dir / "history.json").open("w", encoding="utf-8") as fh:
        json.dump(history, fh, indent=2)

    if run_wandb and wandb_run is not None:
        wandb_run.log({
            "test/best_aupr_auc": test_metrics_aupr[6],
            "test/best_aupr_aupr": test_metrics_aupr[7],
            "test/best_loss_auc": test_metrics_loss[6],
            "test/best_loss_aupr": test_metrics_loss[7],
        })
        wandb_run.finish()

    return TrainSummary(best_aupr_path, best_loss_path, history)


def _load_model(model_name: str, checkpoint_path: Path, drop_prob: float):
    model, _ = _make_model(model_name, drop_prob=drop_prob, n_output=1)
    state_dict = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model

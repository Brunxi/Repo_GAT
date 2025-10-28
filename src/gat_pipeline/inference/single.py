from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from ..config import PipelineConfig
from ..data.esm import embed_sequence, load_esm_model
from ..models import GATNet
from ..utils import cmap_to_graph


@dataclass
class InferenceResult:
    sequence_id: str
    probability: float
    prediction: int
    sequence_length: int


class GATNetWithAttention(torch.nn.Module):
    def __init__(self, trained_model: GATNet):
        super().__init__()
        self.gat_model = trained_model

    def forward_with_attention(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.dropout(x, p=self.gat_model.drop_prob, training=self.gat_model.training)
        x1, attn1 = self.gat_model.gcn1(x, edge_index, return_attention_weights=True)
        x1 = F.elu(x1)
        x1 = F.dropout(x1, p=self.gat_model.drop_prob, training=self.gat_model.training)
        x2, attn2 = self.gat_model.gcn2(x1, edge_index, return_attention_weights=True)
        x2 = self.gat_model.relu(x2)
        x2 = self.gat_model.fc_g1(x2)
        x2 = self.gat_model.relu(x2)
        out = self.gat_model.fc_g2(x2)
        return out, attn1, attn2


def _build_graph(sequence_id: str, sequence: str, config: PipelineConfig):
    model_bundle = load_esm_model()
    _, features, contact_map = embed_sequence(sequence_id, sequence, model_bundle)
    node_features, edge_index = cmap_to_graph(features, contact_map, ratio=config.ratio)
    graph_data = Data(
        x=torch.as_tensor(node_features, dtype=torch.float32),
        edge_index=torch.as_tensor(edge_index, dtype=torch.long),
        y=torch.tensor([0.0], dtype=torch.float32),
    )
    graph_data.batch = torch.zeros(graph_data.num_nodes, dtype=torch.long)
    return graph_data


def load_gat_model(checkpoint_path: Path, drop_prob: float, device: torch.device) -> GATNet:
    model = GATNet(esm_embeds=1280, n_heads=2, drop_prob=drop_prob, n_output=1)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def infer_sequence(
    sequence: str,
    sequence_id: str,
    checkpoint_path: Path,
    config: PipelineConfig,
    device: Optional[torch.device] = None,
) -> InferenceResult:
    device = device or torch.device(config.cuda_name if torch.cuda.is_available() else "cpu")
    graph = _build_graph(sequence_id, sequence, config)
    loader = DataLoader([graph], batch_size=1, shuffle=False)

    model = load_gat_model(checkpoint_path, config.drop_prob, device)
    model_wrapper = GATNetWithAttention(model)

    with torch.no_grad():
        batch = next(iter(loader)).to(device)
        logits, *_ = model_wrapper.forward_with_attention(batch)
        probability = torch.sigmoid(logits).cpu().numpy().flatten()[0]
        prediction = int(probability >= 0.5)

    return InferenceResult(
        sequence_id=sequence_id,
        probability=float(probability),
        prediction=prediction,
        sequence_length=len(sequence),
    )

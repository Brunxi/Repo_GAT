from __future__ import annotations

from functools import lru_cache
from typing import Iterable, Iterator, Tuple

import torch


@lru_cache(maxsize=1)
def load_esm_model() -> Tuple[torch.nn.Module, callable]:
    """Load ESM-2 from fair-esm using the native contact-head outputs."""

    import os

    os.environ.setdefault("APEX_DISABLED", "1")
    import esm  # type: ignore

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    return model, batch_converter


def embed_sequence(identifier: str, sequence: str, model_bundle: Tuple[torch.nn.Module, callable]):
    """Return representations and the official ESM contact map for a single sequence."""

    model, batch_converter = model_bundle
    device = next(model.parameters()).device

    trimmed_sequence = sequence[:1024] if len(sequence) > 1024 else sequence
    gene_ids, _, tokens = batch_converter([(identifier, trimmed_sequence)])
    tokens = tokens.to(device)

    with torch.no_grad():
        results = model(tokens, repr_layers=[33], return_contacts=True)
        representations = results["representations"][33].squeeze(0)[1:-1, :].cpu()
        contact_map = results["contacts"].squeeze(0).cpu()

    return gene_ids[0], representations, contact_map


def embed_sequences(bag: Iterable[Tuple[str, str]], model_bundle: Tuple[torch.nn.Module, callable]) -> Iterator[Tuple[str, torch.Tensor, torch.Tensor]]:
    for identifier, sequence in bag:
        yield embed_sequence(identifier, sequence, model_bundle)

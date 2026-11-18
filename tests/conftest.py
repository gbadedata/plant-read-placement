from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from prp.config import Panel, Species


@pytest.fixture
def rng():
    return np.random.default_rng(0)


def make_panel(extra: list[Species] | None = None) -> Panel:
    sp = [
        Species("Solanum lycopersicum", "Solanum", "Solanaceae", "panel", pretrain="false"),
        Species("Coffea canephora", "Coffea", "Rubiaceae", "panel", pretrain="false"),
        Species("Solanum pennellii", "Solanum", "Solanaceae", "val_relative", pretrain="false"),
        Species("Theobroma cacao", "Theobroma", "Malvaceae", "val_new", pretrain="false"),
        Species("Solanum tuberosum", "Solanum", "Solanaceae", "test_relative", pretrain="false"),
        Species("Beta vulgaris", "Beta", "Amaranthaceae", "test_new", pretrain="false"),
    ]
    return Panel("t", sp + (extra or []))


@pytest.fixture
def panel():
    return make_panel()


# --------------------------------------------------------------------- fake PlantCaduceus components

VOCAB = {"[SEP]": 1, "A": 7, "C": 8, "G": 9, "T": 10, "N": 11}
COMP_ID = {7: 10, 10: 7, 8: 9, 9: 8, 11: 11, 1: 1}


class FakeTokenizer:
    """Character tokenizer that appends [SEP], like the Caduceus/HyenaDNA style tokenizers."""

    def convert_tokens_to_ids(self, toks):
        return [VOCAB[t] for t in toks]

    def encode_plus(self, seq, return_attention_mask=False, return_token_type_ids=False):
        return {"input_ids": [VOCAB[c] for c in seq] + [VOCAB["[SEP]"]]}


class _Out:
    def __init__(self, hs):
        self.hidden_states = hs


class FakeRCPSModel(nn.Module):
    """Copies the Caduceus RCPS output layout (modeling_caduceus.py): channels are
    [forward stream | reverse-complement stream flipped along sequence AND channel axes]."""

    def __init__(self, H: int = 6):
        super().__init__()
        self.config = type("C", (), {"rcps": True})()
        self.emb = nn.Embedding(16, H)
        self.backbone = nn.Linear(H, H)
        lut = torch.arange(16)
        for a, b in COMP_ID.items():
            lut[a] = b
        self.register_buffer("comp", lut)

    def forward(self, input_ids, output_hidden_states=True):
        fwd = self.backbone(self.emb(input_ids))
        rc_ids = self.comp[torch.flip(input_ids, dims=[1])]
        rc_stream = self.backbone(self.emb(rc_ids))  # RC stream, in its own coordinates
        return _Out([torch.cat([fwd, rc_stream.flip(dims=[-2, -1])], dim=-1)])

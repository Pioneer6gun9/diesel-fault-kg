"""Shared NER model, BIO span decoding, tokenizer and relation-type helpers."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch import nn
from transformers import BertConfig, BertTokenizerFast
from transformers import RobertaConfig, RobertaModel


FAULT_TYPES = {"故障现象", "故障状态", "故障原因"}
# These are explicit engineering constraints, not labels learned from independent
# causal ground truth. Callers may supply a different ontology explicitly.
DEFAULT_ONTOLOGY = {
    "包含": [({"发动机型号", "故障部位"}, {"故障部位"})],
    "发生于": [(FAULT_TYPES | {"影响后果"}, {"故障部位"})],
    "导致": [(FAULT_TYPES | {"故障部位"}, {"故障现象", "故障状态", "影响后果"})],
    "影响": [(FAULT_TYPES | {"故障部位"}, FAULT_TYPES | {"影响后果", "故障部位"})],
    "由…引起": [({"故障现象", "故障状态", "影响后果"}, {"故障原因", "故障部位"})],
    "表现为": [({"故障部位", "故障原因", "故障状态", "发动机型号"}, {"故障现象", "故障状态"})],
    "适用于": [({"维修建议", "检测手段", "故障部位"}, {"发动机型号"})],
    "通过…检测": [(FAULT_TYPES | {"故障部位", "发动机型号"}, {"检测手段"})],
    "需…维修": [(FAULT_TYPES | {"故障部位", "发动机型号"}, {"维修建议"})],
}


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_tokenizer(directory: Path, default_do_lower_case: bool = True) -> BertTokenizerFast:
    # Both old training files use BertTokenizerFast, including the NER encoder
    # named 'chinese-roberta'. Instantiate from local vocab to prevent downloads.
    tokenizer_config = directory / "tokenizer_config.json"
    settings = json.loads(tokenizer_config.read_text()) if tokenizer_config.exists() else {}
    return BertTokenizerFast(vocab_file=str(directory / "vocab.txt"),
                             do_lower_case=settings.get("do_lower_case", default_do_lower_case),
                             clean_up_tokenization_spaces=False)


def encoder_config(directory: Path, state: dict[str, torch.Tensor], prefix: str):
    values = json.loads((directory / "config.json").read_text())
    embedding = state[f"{prefix}.embeddings.word_embeddings.weight"]
    layer_indices = [int(key.split(".layer.")[1].split(".")[0]) for key in state
                     if key.startswith(f"{prefix}.encoder.layer.")]
    values.update(vocab_size=embedding.shape[0], hidden_size=embedding.shape[1],
                  num_hidden_layers=max(layer_indices) + 1,
                  max_position_embeddings=state[f"{prefix}.embeddings.position_embeddings.weight"].shape[0],
                  type_vocab_size=state[f"{prefix}.embeddings.token_type_embeddings.weight"].shape[0],
                  intermediate_size=state[f"{prefix}.encoder.layer.0.intermediate.dense.weight"].shape[0])
    if values["hidden_size"] % values.get("num_attention_heads", 12):
        raise ValueError("Local encoder config has incompatible attention head count")
    cls = RobertaConfig if prefix == "roberta" else BertConfig
    return cls.from_dict(values)


class NERModel(nn.Module):
    def __init__(self, config: RobertaConfig, label_count: int, with_pooler: bool):
        super().__init__()
        self.roberta = RobertaModel(config, add_pooling_layer=with_pooler)
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(config.hidden_size, label_count)

    def forward(self, input_ids, attention_mask):
        hidden = self.roberta(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        return self.classifier(self.dropout(hidden))


def read_bio(path: Path):
    rows, units, labels = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines() + [""]:
        if not line.strip():
            if units:
                offsets, cursor = [], 0
                for unit in units:
                    offsets.append((cursor, cursor + len(unit)))
                    cursor += len(unit)
                rows.append({"text": "".join(units), "labels": labels, "offsets": offsets})
                units, labels = [], []
        else:
            fields = line.split("\t", 1)
            if len(fields) != 2 or not fields[0]:
                raise ValueError(f"Invalid BIO record in {path}")
            units.append(fields[0]); labels.append(fields[1].strip())
    return rows


def bio_spans(labels, offsets, probabilities=None):
    spans, current = [], None
    for index, (label, (start, end)) in enumerate(zip(labels, offsets)):
        if end <= start:
            continue
        kind = label[2:] if label.startswith(("B-", "I-")) else None
        continuation = label.startswith("I-") and current is not None and current["type"] == kind and start == current["end"]
        if not continuation:
            if current:
                current["confidence"] = float(np.mean(current.pop("scores")))
                spans.append(current)
            current = None
            if kind:
                current = {"type": kind, "start": start, "end": end, "scores": []}
        elif current:
            current["end"] = end
        if current:
            current["scores"].append(float(probabilities[index]) if probabilities is not None else 1.0)
    if current:
        current["confidence"] = float(np.mean(current.pop("scores")))
        spans.append(current)
    return spans


def predict_ner(model, tokenizer, texts, id2label, device, max_length=512):
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_length,
                        return_offsets_mapping=True, return_tensors="pt")
    offsets = encoded.pop("offset_mapping").tolist()
    with torch.inference_mode():
        probabilities = model(encoded["input_ids"].to(device), encoded["attention_mask"].to(device)).softmax(-1).cpu()
    scores, indices = probabilities.max(-1)
    results = []
    for row_offsets, row_indices, row_scores in zip(offsets, indices.tolist(), scores.tolist()):
        valid = [(offset, index, score) for offset, index, score in zip(row_offsets, row_indices, row_scores) if offset[1] > offset[0]]
        results.append({"offsets": [tuple(v[0]) for v in valid],
                        "labels": [id2label[v[1]] for v in valid], "scores": [v[2] for v in valid]})
    return results


def relation_allowed(ontology, relation, head_type, tail_type):
    return any(head_type in heads and tail_type in tails for heads, tails in ontology.get(relation, []))


def extract_long_cases(model, tokenizer, texts, id2label, device, threshold, batch_size):
    """Batch all overflow windows while preserving each original character span."""
    encoded = tokenizer(texts, truncation=True, max_length=512, stride=64,
                        return_overflowing_tokens=True, return_offsets_mapping=True,
                        padding=True, return_tensors="pt")
    offsets = encoded["offset_mapping"].tolist()
    sample_indices = encoded["overflow_to_sample_mapping"].tolist()
    # Use the window where a token has most context, not the highest class score.
    chosen_by_sample = [{} for _ in texts]
    for first_window in range(0, len(offsets), batch_size):
        last_window = min(first_window + batch_size, len(offsets))
        with torch.inference_mode():
            scores, indices = model(encoded["input_ids"][first_window:last_window].to(device),
                                    encoded["attention_mask"][first_window:last_window].to(device)).softmax(-1).cpu().max(-1)
        for batch_row, window in enumerate(range(first_window, last_window)):
            row_offsets = offsets[window]
            chosen = chosen_by_sample[sample_indices[window]]
            valid = [i for i, (a, b) in enumerate(row_offsets) if b > a]
            for position, i in enumerate(valid):
                offset = tuple(row_offsets[i]); context = min(position, len(valid) - 1 - position)
                if offset not in chosen or context > chosen[offset][0]:
                    chosen[offset] = (context, id2label[int(indices[batch_row, i])], float(scores[batch_row, i]))
    result = []
    for text, chosen in zip(texts, chosen_by_sample):
        ordered = sorted(chosen)
        spans = bio_spans([chosen[o][1] for o in ordered], ordered, [chosen[o][2] for o in ordered])
        result.append([{**span, "text": text[span["start"]:span["end"]],
                        "confidence_kind": "mean_uncalibrated_token_softmax"}
                       for span in spans if span["confidence"] >= threshold])
    return result


#!/usr/bin/env python3
"""Reviewed relationship-existence extension of the original nine-class RE.

Audits supplied positive-only files without generating fake negatives. A full
ten-class source interaction-MLP training/evaluation path is available for
reviewed span annotations. This extension is disclosed new implementation,
not recovered author methodology or a completed external benchmark.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import random
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, f1_score, precision_recall_fscore_support

from diesel_kg.inference.paired import source_classes
from diesel_kg.inference.relations import pair_window, validate_mentions
from diesel_kg.models.loading import load_tokenizer, file_sha256


def dump(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def normalized_body(text):
    return "".join(text.split())


def body_hash(text): return hashlib.sha256(normalized_body(text).encode()).hexdigest()


def audit_source(directory, output):
    output.mkdir(parents=True, exist_ok=True)
    data = {split: json.loads((directory / f"{split}.json").read_text()) for split in ["train", "dev", "test"]}
    bodies = {s: {body_hash(r["text"]) for r in records} for s, records in data.items()}
    samples = {s: {json.dumps([normalized_body(r["text"]), r.get("subject"), r.get("object"), r["relation"]], ensure_ascii=False) for r in records} for s, records in data.items()}
    labels = sorted({r["relation"] for records in data.values() for r in records if r["relation"] != "no_relation"})
    audit = {"status": "not_evaluable_without_reviewed_negative_annotations", "splits": {},
             "positive_relation_classes": labels, "n_positive_classes": len(labels),
             "cross_split_exact_normalized_body_overlap": {}, "cross_split_exact_sample_overlap": {},
             "binary_precision": None, "binary_recall": None, "binary_F1": None, "binary_AUPRC": None,
             "negative_generation": "none; unannotated pairs are unknown and not relabeled as no_relation",
             "original_source_is_preserved": True, "script_sha256": file_sha256(Path(__file__)),
             "command": __import__('sys').argv, "input_sha256": {s: file_sha256(directory / f"{s}.json") for s in data}}
    for split, records in data.items():
        audit["splits"][split] = {"samples": len(records), "relation_counts": dict(collections.Counter(r["relation"] for r in records)),
                                  "unique_normalized_bodies": len(bodies[split]),
                                  "n_no_relation": sum(r["relation"] == "no_relation" for r in records),
                                  "n_reviewed_no_relation": sum(r["relation"] == "no_relation" and r.get("reviewed_negative") is True for r in records),
                                  "n_span_annotated_pairs": sum(isinstance(r.get("head"), dict) and isinstance(r.get("tail"), dict) for r in records)}
    for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
        audit["cross_split_exact_normalized_body_overlap"][f"{a}/{b}"] = len(bodies[a] & bodies[b])
        audit["cross_split_exact_sample_overlap"][f"{a}/{b}"] = len(samples[a] & samples[b])
    dump(output / "source_positive_only_audit.json", audit)
    dump(output / "annotation_schema.json", {"format": "one reviewed ordered mention pair per JSONL record",
        "required_fields": {"sample_id": "unique string", "case_id": "original source case ID", "text": "complete original case body",
                            "head": {"text": "exact source substring", "start": "inclusive integer character offset", "end": "exclusive integer offset", "type": "entity type"},
                            "tail": {"text": "exact source substring", "start": "inclusive integer character offset", "end": "exclusive integer offset", "type": "entity type"},
                            "relation": labels + ["no_relation"]},
        "negative_requirements": {"reviewed_negative": True, "reviewer_id": "identifier of the actual adjudicator", "review_evidence": "actual pair-specific rationale"},
        "optional_group_id": "shared source/scenario/template group; unioned with case IDs and identical normalized bodies before split",
        "unknown_pair_policy": "unreviewed, ambiguous and unlabeled pairs are not negatives",
        "split_policy": "group by linked case_id, normalized body hash and provided group_id; disjoint train/dev/test",
        "initialization": "original checkpoint encoder/entity_interact only; entire ten-class classifier initialized anew",
        "implementation_status": "training and heldout evaluation implemented; valid negative benchmark data currently absent"})
    return audit


def validate_annotation(record, allowed):
    for name in ["sample_id", "case_id", "text", "head", "tail", "relation"]:
        if name not in record: raise ValueError(f"Missing annotation field {name}")
    if not isinstance(record["sample_id"], str) or not record["sample_id"]: raise ValueError("sample_id must be a nonempty string")
    if not isinstance(record["case_id"], str) or not record["case_id"]: raise ValueError("case_id must be a nonempty string")
    if record["relation"] not in allowed: raise ValueError("Relation label not in ten-class ontology")
    validate_mentions({"case_id": record["case_id"], "text": record["text"], "entities": [record["head"], record["tail"]]})
    if record["relation"] == "no_relation":
        if record.get("reviewed_negative") is not True or not str(record.get("reviewer_id", "")).strip() or not str(record.get("review_evidence", "")).strip():
            raise ValueError("no_relation requires reviewed_negative=true and actual reviewer_id/review_evidence; unknown pairs are not negative")


def group_records(records):
    """Union-linked cases/bodies/source groups, preventing chained leakage."""
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i: parent[i] = parent[parent[i]]; i = parent[i]
        return i
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[b] = a
    seen = {}
    for i, record in enumerate(records):
        keys = [("case", record["case_id"]), ("body", body_hash(record["text"]))]
        if record.get("group_id"): keys.append(("group", str(record["group_id"])))
        for key in keys:
            if key in seen: union(i, seen[key])
            else: seen[key] = i
    groups = collections.defaultdict(list)
    for i, record in enumerate(records): groups[find(i)].append(record)
    return list(groups.values())


def grouped_split(records, seed=42, dev_fraction=.15, test_fraction=.15):
    if not (0 < dev_fraction < 1 and 0 < test_fraction < 1 and dev_fraction + test_fraction < 1):
        raise ValueError("Positive dev/test fractions must sum to less than one")
    groups = group_records(records)
    if len(groups) < 3: raise ValueError("At least three independent case/body/source groups required")
    random.Random(seed).shuffle(groups)
    n = len(groups); nt = max(1, round(n * test_fraction)); nd = max(1, round(n * dev_fraction))
    if nt + nd >= n: raise ValueError("Insufficient groups for nonempty training split")
    return {"test": [r for g in groups[:nt] for r in g], "dev": [r for g in groups[nt:nt + nd] for r in g],
            "train": [r for g in groups[nt + nd:] for r in g]}


def check_disjoint_splits(splits):
    if len({r["sample_id"] for values in splits.values() for r in values}) != sum(len(v) for v in splits.values()):
        raise ValueError("Duplicate sample IDs across splits")
    seen = {}
    for split, records in splits.items():
        for r in records:
            keys = [("case", r["case_id"]), ("body", body_hash(r["text"]))]
            if r.get("group_id"): keys.append(("group", str(r["group_id"])))
            for key in keys:
                if key in seen and seen[key] != split: raise ValueError(f"Case/body/source-group split leakage: {key[0]}")
                seen[key] = split


def calculate_metrics(labels, probabilities, no_relation_id, threshold=.5):
    labels, probabilities = np.asarray(labels), np.asarray(probabilities)
    if labels.ndim != 1 or probabilities.ndim != 2 or len(labels) != len(probabilities) or not len(labels): raise ValueError("Nonempty aligned labels/probabilities required")
    if not 0 <= threshold <= 1: raise ValueError("Binary threshold must be in [0,1]")
    if not np.isfinite(probabilities).all() or (probabilities < 0).any() or not np.allclose(probabilities.sum(1), 1, atol=1e-5): raise ValueError("Invalid probability distributions")
    if not (0 <= no_relation_id < probabilities.shape[1]) or (labels < 0).any() or (labels >= probabilities.shape[1]).any(): raise ValueError("Label outside classifier range")
    gold = (labels != no_relation_id).astype(int); scores = 1 - probabilities[:, no_relation_id]
    prediction = probabilities.argmax(1)
    result = {"n_samples": len(labels), "n_actual_positive_relations": int(gold.sum()), "n_actual_reviewed_negatives": int((1 - gold).sum()),
              "binary_threshold": threshold, "binary_score": "sum of positive-class probabilities = 1-P(no_relation)",
              "multi_class_accuracy": float(accuracy_score(labels, prediction)),
              "multi_class_macroF1_all_classifier_classes": float(f1_score(labels, prediction, labels=list(range(probabilities.shape[1])), average="macro", zero_division=0)),
              "multi_class_confusion_matrix": confusion_matrix(labels, prediction, labels=list(range(probabilities.shape[1]))).tolist()}
    if len(set(gold.tolist())) < 2:
        result.update(binary_status="not_evaluable_without_both_binary_classes", binary_precision=None, binary_recall=None, binary_F1=None, binary_AUPRC=None)
    else:
        p, r, f, _ = precision_recall_fscore_support(gold, scores >= threshold, average="binary", zero_division=0)
        result.update(binary_status="evaluated", binary_precision=float(p), binary_recall=float(r), binary_F1=float(f), binary_AUPRC=float(average_precision_score(gold, scores)))
    return result


class SpanPairDataset(Dataset):
    def __init__(self, records, tokenizer, relation_to_id, max_length=128):
        self.records, self.features, self.rejections = records, [], []
        for record in records:
            enc = tokenizer(record["text"], add_special_tokens=False, return_offsets_mapping=True, truncation=False)
            window, reason = pair_window(enc["offset_mapping"], record["head"], record["tail"], max_length - 2)
            if reason:
                self.rejections.append({"sample_id": record["sample_id"], "case_id": record["case_id"], "reason": reason}); continue
            left, right, h, t = window
            padded = tokenizer.pad({"input_ids": tokenizer.build_inputs_with_special_tokens(enc["input_ids"][left:right])},
                                   padding="max_length", max_length=max_length, return_tensors="pt")
            self.features.append({"input_ids": padded["input_ids"].flatten(), "attention_mask": padded["attention_mask"].flatten(),
                                  "head_position": torch.tensor(h), "tail_position": torch.tensor(t),
                                  "label": torch.tensor(relation_to_id[record["relation"]]), "sample_id": record["sample_id"]})
    def __len__(self): return len(self.features)
    def __getitem__(self, index): return self.features[index]


def evaluate(model, loader, no_relation_id):
    model.eval(); labels, probabilities, predictions = [], [], []
    with torch.inference_mode():
        for batch in loader:
            logits = model(batch["input_ids"], batch["attention_mask"], batch["head_position"], batch["tail_position"])
            probs = logits.softmax(-1).numpy(); gold = batch["label"].numpy()
            labels.extend(gold.tolist()); probabilities.extend(probs.tolist())
            predictions.extend({"sample_id": sid, "true_label_id": int(y), "predicted_label_id": int(p.argmax()), "probabilities": p.tolist()}
                               for sid, y, p in zip(batch["sample_id"], gold, probs))
    return calculate_metrics(labels, probabilities, no_relation_id), predictions


def train(a, source_audit):
    begun = time.perf_counter(); random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed); torch.set_num_threads(a.threads)
    if not all([a.source, a.pretrained, a.tokenizer, a.initial_checkpoint]): raise ValueError("Training requires local source/pretrained/tokenizer/initial-checkpoint")
    labels = source_audit["positive_relation_classes"]
    if len(labels) != 9: raise ValueError("Expected source nine-positive-class ontology")
    relation_to_id = {r: i for i, r in enumerate(labels + ["no_relation"])}
    if a.annotations:
        records = jsonl(a.annotations)
        for r in records: validate_annotation(r, relation_to_id)
        if len({r["sample_id"] for r in records}) != len(records): raise ValueError("Duplicate sample IDs")
        splits = grouped_split(records, a.seed, a.dev_fraction, a.test_fraction)
    else:
        if not all([a.train_jsonl, a.dev_jsonl, a.test_jsonl]): raise ValueError("Provide --annotations or all explicit train/dev/test JSONL paths")
        splits = {s: jsonl(getattr(a, s + "_jsonl")) for s in ["train", "dev", "test"]}
        for values in splits.values():
            for r in values: validate_annotation(r, relation_to_id)
    check_disjoint_splits(splits)
    for s, records in splits.items():
        if len({r["relation"] == "no_relation" for r in records}) != 2:
            raise ValueError(f"{s} requires both reviewed negatives and real positive relations; no synthetic balancing")
        with (a.output / f"{s}_grouped_annotations.jsonl").open("w") as handle:
            for r in records: handle.write(json.dumps(r, ensure_ascii=False) + "\n")
    tokenizer = load_tokenizer(a.tokenizer, default_do_lower_case=False)
    datasets = {s: SpanPairDataset(r, tokenizer, relation_to_id, a.max_length) for s, r in splits.items()}
    coverage = {s: {"total_annotated": len(splits[s]), "eligible_model_pairs": len(d), "explicitly_unclassified": len(d.rejections), "rejections": d.rejections} for s, d in datasets.items()}
    dump(a.output / "window_coverage.json", coverage)
    for s, d in datasets.items():
        if not len(d) or len({f["label"].item() == relation_to_id["no_relation"] for f in d.features}) != 2:
            raise ValueError(f"{s} requires both binary classes after explicit window coverage audit")
    generator = torch.Generator().manual_seed(a.seed)
    loaders = {s: DataLoader(d, batch_size=a.batch_size, shuffle=s == "train", generator=generator if s == "train" else None) for s, d in datasets.items()}
    _, model_class, source_hash = source_classes(a.source)
    model = model_class(str(a.pretrained), 10, dropout_rate=.2).cpu()
    saved = torch.load(a.initial_checkpoint, weights_only=True, map_location="cpu", mmap=True)
    copied = {k: v for k, v in saved["model_state_dict"].items() if k.startswith("bert.") or k.startswith("entity_interact.")}
    missing, unexpected = model.load_state_dict(copied, strict=False)
    if unexpected or any(not k.startswith("classifier.") for k in missing): raise ValueError("Unexpected architecture mismatch initializing original encoder/interaction")
    del saved
    freq = collections.Counter(f["label"].item() for f in datasets["train"].features)
    class_weights = torch.tensor([len(datasets["train"]) / (10 * freq[i]) if freq[i] else 0. for i in range(10)], dtype=torch.float32)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-7)
    history, best, stale = [], (-1., -1.), 0
    checkpoint = a.output / "best_ten_class_model.pt"
    config = {"extension_method": "ten-class reviewed relationship existence; new implementation distinct from source nine-class classifier",
        "seed": a.seed, "device": "cpu", "epochs": a.epochs, "batch_size": a.batch_size, "max_length": a.max_length,
        "learning_rate": a.learning_rate, "optimizer": "AdamW weight_decay=1e-4", "scheduler": "CosineAnnealingWarmRestarts(T0=3,Tmult=2,eta_min=1e-7), per minibatch",
        "class_balance": "inverse-frequency weighted cross entropy; no synthesized negatives or oversampling",
        "initialization": "preserve supplied checkpoint bert and entity_interact; all classifier layers initialized anew",
        "best_model_selection": "dev binary F1 at fixed .5 threshold, ties broken by all-10-class macroF1; heldout test once",
        "patience": a.patience, "group_disjoint": True, "annotation_input_sha256": {str(p): file_sha256(p) for p in [a.annotations, a.train_jsonl, a.dev_jsonl, a.test_jsonl] if p},
        "initial_checkpoint_sha256": file_sha256(a.initial_checkpoint), "source_class_hashes": source_hash, "rel2id": relation_to_id,
        "script_sha256": file_sha256(Path(__file__)), "training_class_counts": {name: freq[i] for name, i in relation_to_id.items()},
        "class_weights": class_weights.tolist(), "command": __import__('sys').argv,
        "trainable_parameter_names": [name for name, p in model.named_parameters() if p.requires_grad]}
    dump(a.output / "training_config.json", config)
    for epoch in range(1, a.epochs + 1):
        model.train(); loss_sum = 0.
        for batch in loaders["train"]:
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch["input_ids"], batch["attention_mask"], batch["head_position"], batch["tail_position"])
            loss = criterion(logits, batch["label"]); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step(); scheduler.step(); loss_sum += loss.item() * len(batch["label"])
        dev, _ = evaluate(model, loaders["dev"], relation_to_id["no_relation"])
        value = (dev["binary_F1"], dev["multi_class_macroF1_all_classifier_classes"])
        history.append({"epoch": epoch, "training_loss": loss_sum / len(datasets["train"]), "dev_metrics": dev})
        if value > best:
            best, stale = value, 0
            torch.save({"model_state_dict": model.state_dict(), "rel2id": relation_to_id, "best_epoch": epoch, "dev_metrics": dev, "config": config}, checkpoint)
        else: stale += 1
        dump(a.output / "training_history.json", history)
        if stale >= a.patience: break
    selected = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    model.load_state_dict(selected["model_state_dict"], strict=True)
    test, predictions = evaluate(model, loaders["test"], relation_to_id["no_relation"])
    with (a.output / "heldout_test_predictions.jsonl").open("w") as handle:
        for record in predictions: handle.write(json.dumps(record) + "\n")
    result = {"status": "completed_reviewed_group_heldout_evaluation", "best_epoch": selected["best_epoch"], "epochs_run": len(history),
        "test_metrics": test, "test_coverage": coverage["test"], "group_split_sample_counts": {s: len(r) for s, r in splits.items()},
        "checkpoint_sha256": file_sha256(checkpoint), "elapsed_seconds": time.perf_counter() - begun,
        "limitations": ["This is the disclosed ten-class extension, not the original paper's positive-only RE protocol.", "External annotation correctness and near-template group completeness depend on actual adjudication.",
                       "Heldout metrics cover model-eligible token windows; unclassified pairs and full denominator are reported separately.", "Relation-existence performance is not diagnosis or causal propagation accuracy."]}
    dump(a.output / "run_summary.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit-source-dir", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--train", action="store_true")
    for name in ["annotations", "train-jsonl", "dev-jsonl", "test-jsonl", "source", "pretrained", "tokenizer", "initial-checkpoint"]: p.add_argument("--" + name, type=Path)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--threads", type=int, default=2); p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32); p.add_argument("--max-length", type=int, default=128); p.add_argument("--learning-rate", type=float, default=1e-5)
    p.add_argument("--patience", type=int, default=3); p.add_argument("--dev-fraction", type=float, default=.15); p.add_argument("--test-fraction", type=float, default=.15)
    a = p.parse_args()
    audit = audit_source(a.audit_source_dir, a.output)
    if a.train:
        try: result = train(a, audit)
        except Exception as error:
            dump(a.output / "run_summary.json", {"status": "not_evaluable_invalid_or_missing_reviewed_annotations", "error": str(error), "metrics": None})
            raise
    else:
        result = audit; dump(a.output / "run_summary.json", result)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()

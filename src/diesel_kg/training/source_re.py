"""Execute the unchanged author RE algorithm with isolated local runtime IO.

The default executes source train(), REDataset, RelationModel, optimizer,
sampling, scheduler, loss, pooling, splits and early stopping directly.
Runtime paths/device/fonts/seed are adapted and recorded; --locator offset
explicitly repairs only the entity-location function with a saved code diff.
"""
from __future__ import annotations

import argparse
from collections import Counter
import difflib
import hashlib
import importlib.metadata
import inspect
import json
import logging
import os
from pathlib import Path
import random
import runpy
import shutil
import sys
import textwrap
import time

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import torch
from sklearn.metrics import classification_report, accuracy_score, f1_score
from transformers import BertTokenizerFast


def sha256(path):
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(4194304), b""):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def correct_offset_locator(self, text, entity, offsets):
    """Return inclusive token indices covering the complete first mention."""
    start = text.find(entity)
    if start < 0:
        raise ValueError("entity_not_in_text")
    end = start + len(entity)
    positions = [index for index, (left, right) in enumerate(offsets)
                 if right > left and left < end and right > start]
    if not positions or offsets[positions[0]][0] > start or offsets[positions[-1]][1] < end:
        raise ValueError("entity_truncated_or_unaligned")
    return positions[0], positions[-1]


def write_schema_subset_audits(output_dir, data_dir, final_test):
    """Analyse already produced predictions; never invoke or choose a model."""
    rows_by_split = {split: json.loads((data_dir / f"{split}.json").read_text())
                     for split in ("train", "dev", "test")}
    rows = rows_by_split["test"]
    indices = final_test.get("source_test_scored_indices", list(range(len(final_test["predicted_ids"]))))
    if len(indices) != len(final_test["predicted_ids"]): raise ValueError("Prediction/index count mismatch")
    id2rel = {int(key): value for key, value in final_test["id2rel"].items()}
    labels = sorted(id2rel)
    subset_metrics = {"computed_from_existing_predictions": True, "new_model_inference_calls": 0,
                      "definition": "Subsets are source-schema groups. Explicit type fields do not establish independent or correct annotations."}
    for name, wants_types in (("without_explicit_types", False), ("with_explicit_subject_object_types", True)):
        positions = [position for position, index in enumerate(indices)
                     if bool(rows[index].get("subject_type") and rows[index].get("object_type")) == wants_types]
        truth = [final_test["true_ids"][position] for position in positions]
        predicted = [final_test["predicted_ids"][position] for position in positions]
        subset_metrics[name] = {
            "n": len(positions), "correct": sum(a == b for a, b in zip(truth, predicted)),
            "accuracy": float(accuracy_score(truth, predicted)) if positions else None,
            "macro_f1_all_nine_classes": float(f1_score(truth, predicted, average="macro", labels=labels, zero_division=0)) if positions else None,
            "macro_f1_true_supported_classes": float(f1_score(truth, predicted, average="macro", labels=sorted(set(truth)), zero_division=0)) if positions else None,
            "original_indices": [indices[position] for position in positions],
            "gold_relation_counts": dict(Counter(id2rel[label] for label in truth)),
            "classification_report": classification_report(truth, predicted, labels=labels,
                target_names=[id2rel[label] for label in labels], output_dict=True, zero_division=0) if positions else {}}
    write_json(output_dir / "subset_metrics.json", subset_metrics)
    # Transparent strict type constraints already used in candidate filtering.
    # Their violations are audit flags, not independently verified label errors.
    faults = {"故障现象", "故障状态", "故障原因"}
    rules = {
        "包含": ({"发动机型号", "故障部位"}, {"故障部位"}),
        "发生于": (faults | {"影响后果"}, {"故障部位"}),
        "导致": (faults | {"故障部位"}, {"故障现象", "故障状态", "影响后果"}),
        "影响": (faults | {"故障部位"}, faults | {"影响后果", "故障部位"}),
        "由…引起": ({"故障现象", "故障状态", "影响后果"}, {"故障原因", "故障部位"}),
        "表现为": ({"故障部位", "故障原因", "故障状态", "发动机型号"}, {"故障现象", "故障状态"}),
        "适用于": ({"维修建议", "检测手段", "故障部位"}, {"发动机型号"}),
        "通过…检测": (faults | {"故障部位", "发动机型号"}, {"检测手段"}),
        "需…维修": (faults | {"故障部位", "发动机型号"}, {"维修建议"})}
    schema = {"criterion": "Declared strict type constraints, identical to the reproduction candidate filter; not independent ontology gold.",
              "rules": {relation: {"head_types": sorted(heads), "tail_types": sorted(tails)}
                        for relation, (heads, tails) in rules.items()},
              "splits": {}, "qualification": "Schema groups identify metadata formats. A strict-rule incompatibility is a suspected direction/type issue for manual review, not proof that the label or the paper is wrong. No test data or label is changed."}
    for split, split_rows in rows_by_split.items():
        typed = [(index, row) for index, row in enumerate(split_rows)
                 if row.get("subject_type") and row.get("object_type")]
        flags, compatible = [], 0
        type_triples = Counter((row["relation"], row["subject_type"], row["object_type"]) for _, row in typed)
        for index, row in typed:
            heads, tails = rules.get(row["relation"], (set(), set()))
            if row["subject_type"] in heads and row["object_type"] in tails:
                compatible += 1
            else:
                flags.append({"original_index": index, "text": row["text"], "subject": row["subject"],
                              "object": row["object"], "source_relation": row["relation"],
                              "subject_type": row["subject_type"], "object_type": row["object_type"],
                              "status": "suspected_strict_type_or_direction_incompatibility"})
        schema["splits"][split] = {
            "samples": len(split_rows), "source_sha256": sha256(data_dir / f"{split}.json"),
            "with_both_explicit_type_fields": len(typed), "without_both_explicit_type_fields": len(split_rows) - len(typed),
            "strict_rule_compatible": compatible, "strict_rule_incompatible": len(flags),
            "explicit_relation_type_triples": [{"relation": relation, "subject_type": head, "object_type": tail, "count": count}
                                              for (relation, head, tail), count in type_triples.items()],
            "flags": flags}
    write_json(output_dir / "data_schema_audit.json", schema)
    return subset_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--pretrained-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("mps", "cpu", "cuda"), default="mps")
    parser.add_argument("--locator", choices=("original", "offset"), default="original",
                        help="Only offset changes the author's entity locator; all training settings stay identical")
    args = parser.parse_args()
    for key in ("source", "data_dir", "pretrained_dir", "output_dir", "work_dir"):
        setattr(args, key, getattr(args, key).resolve())
    original_tree = args.source.parent.parent
    for path in (args.output_dir, args.work_dir):
        if path == original_tree or original_tree in path.parents:
            parser.error("Runtime IO must be outside the untouched original kg_build tree")
        path.mkdir(parents=True, exist_ok=True)
    if args.device == "mps" and not torch.backends.mps.is_available():
        parser.error("MPS unavailable")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA unavailable")
    device = torch.device(args.device)
    source_hash = sha256(args.source)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    if args.device == "mps": torch.mps.manual_seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(42)
    torch.set_num_threads(4)
    # The old module creates relative log/model directories at import. Direct
    # these harmless initialization writes into an isolated work directory.
    os.chdir(args.work_dir)
    namespace = runpy.run_path(str(args.source), run_name="original_re_algorithm_isolated")
    original_train = namespace["train"]
    globals_ = original_train.__globals__
    original_parameters = {key: globals_[key] for key in
                           ("MAX_SEQ_LEN", "BATCH_SIZE", "LEARNING_RATE", "EPOCHS", "WARMUP_RATIO", "WEIGHT_DECAY", "DROPOUT_RATE")}
    expected = {"MAX_SEQ_LEN": 128, "BATCH_SIZE": 32, "LEARNING_RATE": 1e-5,
                "EPOCHS": 15, "WARMUP_RATIO": 0.05, "WEIGHT_DECAY": 1e-4, "DROPOUT_RATE": 0.2}
    if original_parameters != expected:
        raise RuntimeError("Original source configuration differs from the audited author protocol")
    log_dir = args.output_dir / "logs"
    model_dir = args.output_dir / "best_re_model"
    visual_dir = log_dir / "visualizations"
    snapshot_dir = args.output_dir / "best_checkpoints_by_epoch"
    for path in (log_dir, model_dir, visual_dir, snapshot_dir): path.mkdir(exist_ok=True, parents=True)
    globals_.update(DEVICE=device, BERT_MODEL_NAME=str(args.pretrained_dir), DATA_DIR=str(args.data_dir),
                    LOG_DIR=str(log_dir), MODEL_SAVE_DIR=str(model_dir), VISUAL_SAVE_DIR=str(visual_dir))
    logger = logging.getLogger()
    for handler in list(logger.handlers): handler.close(); logger.removeHandler(handler)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    for handler in (logging.FileHandler(log_dir / "re_train.log", encoding="utf-8"), logging.StreamHandler(sys.stdout)):
        handler.setFormatter(formatter); logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    # Font selection affects plots only, not training or evaluated predictions.
    globals_["plt"].rcParams["font.sans-serif"] = ["Arial Unicode MS", "DejaVu Sans"]
    locator_audit = None
    original_locator_source = textwrap.dedent(inspect.getsource(globals_["REDataset"]._locate_entity))
    if args.locator == "offset":
        # Audit before training. Invalid or truncated mentions are rejected with
        # source indices; labels, duplicates and the supplied split stay intact.
        tokenizer = BertTokenizerFast.from_pretrained(str(args.pretrained_dir), local_files_only=True)
        if tokenizer.do_lower_case:
            raise ValueError("Official local BERT tokenizer must use do_lower_case=false")
        allowed_signatures = {}
        locator_audit = {"algorithm": "correct inclusive offset span; reject absent or truncated mention",
                         "ambiguous_occurrence_policy": "Gold annotations contain mention strings, not occurrence offsets; the first full occurrence is used.",
                         "splits": {}}
        for split in ("train", "dev", "test"):
            path = args.data_dir / f"{split}.json"
            rows = json.loads(path.read_text()); allowed = set(); rejected = []; ambiguous = 0
            for index, row in enumerate(rows):
                offsets = tokenizer(row["text"], padding="max_length", truncation=True,
                                    max_length=globals_["MAX_SEQ_LEN"], return_offsets_mapping=True)["offset_mapping"]
                try:
                    for name in ("subject", "object"):
                        if row["text"].count(row[name]) > 1: ambiguous += 1
                        correct_offset_locator(None, row["text"], row[name], offsets)
                except ValueError as error:
                    rejected.append({"original_index": index, "reason": str(error)})
                    continue
                allowed.add(tuple(row[key] for key in ("text", "subject", "object", "relation")))
            allowed_signatures[str(path.resolve())] = allowed
            locator_audit["splits"][split] = {"raw_samples": len(rows), "rejected": rejected,
                                              "kept_samples": len(rows) - len(rejected),
                                              "ambiguous_mentions_first_occurrence": ambiguous}
        dataset_class = globals_["REDataset"]
        original_load_data = dataset_class.load_data
        def load_data_with_explicit_mention_rejection(self, path):
            data = original_load_data(self, path)
            allowed = allowed_signatures[str(Path(path).resolve())]
            return [row for row in data if tuple(row[key] for key in ("text", "subject", "object", "relation")) in allowed]
        dataset_class._locate_entity = correct_offset_locator
        dataset_class.load_data = load_data_with_explicit_mention_rejection
        patched_source = textwrap.dedent(inspect.getsource(correct_offset_locator))
        (args.output_dir / "offset_locator_patch.py").write_text(patched_source, encoding="utf-8")
        (args.output_dir / "offset_locator.diff").write_text("".join(difflib.unified_diff(
            original_locator_source.splitlines(True), patched_source.splitlines(True),
            fromfile="original_REDdataset._locate_entity", tofile="correct_offset_locator")), encoding="utf-8")
        locator_audit["patched_method_sha256"] = hashlib.sha256(patched_source.encode()).hexdigest()
        locator_audit["original_method_sha256"] = hashlib.sha256(original_locator_source.encode()).hexdigest()
        write_json(args.output_dir / "locator_patch_audit.json", locator_audit)
    config = {
        "protocol": "direct execution of current original RelationExtraction.py train()",
        "protocol_variant": "original" if args.locator == "original" else "original_architecture_with_offset_locator_repair",
        "entity_locator": args.locator,
        "original_locator_method_sha256": hashlib.sha256(original_locator_source.encode()).hexdigest(),
        "locator_patch_audit": locator_audit,
        "source": str(args.source), "source_sha256": source_hash, "original_parameters": original_parameters,
        "seed": 42, "device": str(device), "data_dir": str(args.data_dir),
        "dependency_versions": {name: importlib.metadata.version(name) for name in
                                 ("torch", "transformers", "numpy", "scikit-learn", "matplotlib", "pandas", "seaborn")},
        "scheduler_semantics": {"class": "CosineAnnealingWarmRestarts", "T_0": 3, "T_mult": 2,
                                "eta_min": 1e-7, "step_frequency": "every minibatch as original",
                                "declared_WARMUP_RATIO_used_by_original_train": False},
        "data": {split: {"count": len(json.loads((args.data_dir / f"{split}.json").read_text())),
                           "sha256": sha256(args.data_dir / f"{split}.json")} for split in ("train", "dev", "test")},
        "pretrained_dir": str(args.pretrained_dir),
        "pretrained_asset_sha256": {name: sha256(args.pretrained_dir / name) for name in
                                     ("model.safetensors", "config.json", "vocab.txt", "tokenizer_config.json")},
        "pretrained_repository": "google-bert/bert-base-chinese",
        "pretrained_revision": "8f23c25b06e129b6c986331a13d8d025a92cf0ea",
        "adaptations": ["Isolated cwd before runpy prevents original-directory import writes.",
                        "Absolute data, log, model and visualization paths.",
                        "Local fresh official BERT with official tokenizer do_lower_case=false; HF offline.",
                        "MPS device instead of original CUDA-or-CPU detection.",
                        "Fixed Python/NumPy/Torch execution seed42 and CPU threads4.",
                        "Available macOS fonts replace SimHei for plotting only.",
                        "Instrumentation records epoch evaluation and copies every original best checkpoint; training math is unchanged."],
        "preserved": ["Original 800/200/225 splits without deduplication or replacement.",
                      "Original entity locator including its fallback, inclusive mean pooling and interaction MLP.",
                      "Original relation vocabulary construction and nine positive classes.",
                      "Original oversampling, class weights, CrossEntropyLoss, AdamW, CosineAnnealingWarmRestarts and gradient clipping.",
                      "Original last12 parameter-tensor freezing, dropout and best-dev micro-F1 early stopping patience3.",
                      "Original final test after loading its saved best_model.pt."],
        "test_policy": "Source train() performs one final test; dev chooses the original best checkpoint. No test tuning."}
    if args.locator == "offset":
        config["adaptations"].append("Explicit offset locator repair with rejected-mention audit; original method code and patch SHA/diff saved.")
        config["preserved"][1] = "Same original inclusive mean pooling and interaction MLP; only entity locator replaced with complete offset matching."
    write_json(args.output_dir / "run_configuration.json", config)
    epochs, best_files, final_test = [], [], {}
    timing = {"epoch_start": None}
    class TrainingStartFilter(logging.Filter):
        def filter(self, record):
            if record.getMessage() == "开始训练..." and timing["epoch_start"] is None:
                timing["epoch_start"] = time.perf_counter()
            return True
    start_filter = TrainingStartFilter(); globals_["logger"].addFilter(start_filter)
    original_evaluate = globals_["evaluate"]
    def record_evaluate(model, dataloader, id2rel, num_relations):
        result = original_evaluate(model, dataloader, id2rel, num_relations)
        role_count = len(dataloader.dataset)
        if role_count == config["data"]["dev"]["count"]:
            now = time.perf_counter()
            row = {"epoch": len(epochs) + 1, "dev_accuracy": float(result[0]), "dev_micro_f1": float(result[1]),
                   "train_plus_dev_seconds": now - timing["epoch_start"] if timing["epoch_start"] is not None else None}
            epochs.append(row); timing["epoch_start"] = now
            write_json(args.output_dir / "epoch_metrics.json", epochs)
            print("AUTHOR_PROTOCOL_EPOCH " + json.dumps(row), flush=True)
        else:
            labels = [id2rel[i] for i in range(num_relations)]
            truth, predicted = [int(i) for i in result[3]], [int(i) for i in result[4]]
            final_test.update({"scored_samples": len(truth), "accuracy": float(result[0]), "micro_f1": float(result[1]),
                               "macro_f1": float(f1_score(truth, predicted, average="macro", labels=list(range(num_relations)), zero_division=0)),
                               "classification_report": classification_report(truth, predicted, labels=list(range(num_relations)), target_names=labels, output_dict=True, zero_division=0),
                               "true_ids": truth, "predicted_ids": predicted, "id2rel": id2rel})
            write_json(args.output_dir / "final_test_metrics.json", final_test)
        return result
    globals_["evaluate"] = record_evaluate
    original_model_class = globals_["RelationModel"]
    def record_model(*positional, **keyword):
        model = original_model_class(*positional, **keyword)
        info = {"class": "original RelationModel", "total_parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "trainable_bert_tensor_names": [name for name, value in model.bert.named_parameters() if value.requires_grad],
                "head_tensor_shapes": {name: list(value.shape) for name, value in model.state_dict().items() if not name.startswith("bert.")}}
        write_json(args.output_dir / "original_model_structure.json", info)
        return model
    globals_["RelationModel"] = record_model
    original_save = torch.save
    def record_save(obj, destination, *positional, **keyword):
        original_save(obj, destination, *positional, **keyword)
        if Path(destination) == model_dir / "best_model.pt":
            epoch = int(obj["epoch"]) + 1
            snapshot = snapshot_dir / f"epoch_{epoch:02d}_best_model.pt"
            shutil.copy2(destination, snapshot)
            best_files.append({"epoch": epoch, "path": str(snapshot), "dev_micro_f1": epochs[-1]["dev_micro_f1"]})
            write_json(args.output_dir / "best_checkpoint_history.json", best_files)
    torch.save = record_save
    started = time.perf_counter()
    try:
        original_train()
    finally:
        torch.save = original_save; globals_["logger"].removeFilter(start_filter)
    if sha256(args.source) != source_hash:
        raise RuntimeError("Original source changed during execution")
    best_path = model_dir / "best_model.pt"
    saved = torch.load(best_path, map_location="cpu", weights_only=True, mmap=True)
    rows = json.loads((args.data_dir / "test.json").read_text())
    rejected_indices = {row["original_index"] for row in locator_audit["splits"]["test"]["rejected"]} if locator_audit else set()
    scored_indices = [index for index, row in enumerate(rows)
                      if index not in rejected_indices and all(isinstance(row.get(key), str) and row[key].strip()
                         for key in ("text", "subject", "object", "relation"))]
    if len(scored_indices) != len(final_test["predicted_ids"]):
        raise ValueError("Original test source indices do not align to final predictions")
    final_test["source_test_scored_indices"] = scored_indices
    write_json(args.output_dir / "final_test_metrics.json", final_test)
    with (args.output_dir / "test_predictions.jsonl").open("w", encoding="utf-8") as handle:
        for index, true, prediction in zip(scored_indices, final_test["true_ids"], final_test["predicted_ids"]):
            row = rows[index]
            handle.write(json.dumps({"original_index": index, **row, "gold_relation_id": true,
                                     "predicted_relation_id": prediction, "predicted_relation": final_test["id2rel"][prediction]}, ensure_ascii=False) + "\n")
    summary = {"status": "complete", "protocol": config["protocol"], "protocol_variant": config["protocol_variant"],
               "entity_locator": args.locator, "source_sha256": source_hash,
               "epochs_completed": len(epochs), "original_early_stopping_possible": True,
               "best_epoch": int(saved["epoch"]) + 1, "best_checkpoint": str(best_path),
               "best_checkpoint_sha256": sha256(best_path), "elapsed_seconds": time.perf_counter() - started,
               "test": final_test, "best_checkpoint_history": best_files,
               "qualification": "Current original RE training component executed with " + ("the author's original locator" if args.locator == "original" else "an explicitly audited offset locator repair") + ". This is nine-class gold-mention relation classification with the original supplied splits. It does not alone reproduce the paper's propagation graph or pattern-mining experiments."}
    write_json(args.output_dir / "run_summary.json", summary)
    write_schema_subset_audits(args.output_dir, args.data_dir, final_test)
    print("AUTHOR_PROTOCOL_COMPLETE " + json.dumps({k: v for k, v in summary.items() if k not in ("test", "best_checkpoint_history")}), flush=True)


if __name__ == "__main__":
    main()

"""Run the supplied NER and the actual interaction-MLP RE on every corpus case.

The RE source classes are compiled from their original AST, without importing
the database builder. An encoder cache avoids recomputing identical case text
for every entity pair; numerical equivalence to source forward is checked.
Every positive-class prediction is retained in a compressed audit file. A
separate, explicitly qualified candidate graph uses fixed type/length/confidence
rules. Neither graph is independent causal ground truth.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter
import gzip
import hashlib
import json
import logging
import os
from pathlib import Path
import time

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from transformers import BertModel

from diesel_kg.models.loading import NERModel, encoder_config, extract_long_cases, file_sha256, load_tokenizer, DEFAULT_ONTOLOGY, relation_allowed
from diesel_kg.data.neo4j import stable_id


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def source_classes(path):
    text = path.read_text(encoding="utf-8-sig")
    tree = ast.parse(text)
    names = {"REDataset", "RelationModel"}
    selected = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name in names]
    if {n.name for n in selected} != names:
        raise ValueError("Expected original REDataset and RelationModel")
    module = ast.Module(body=selected, type_ignores=[])
    namespace = {"torch": torch, "nn": nn, "Dataset": Dataset, "BertModel": BertModel,
                 "json": json, "logger": logging.getLogger("author_inference")}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["REDataset"], namespace["RelationModel"], {
        n.name: hashlib.sha256(ast.get_source_segment(text, n).encode()).hexdigest() for n in selected}


def locate_offsets(offsets, start, end):
    positions = [i for i, (a, b) in enumerate(offsets) if b > a and a < end and b > start]
    if not positions or offsets[positions[0]][0] > start or offsets[positions[-1]][1] < end:
        return None
    return (positions[0], positions[-1])


def infer(args):
    started = time.perf_counter()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = torch.device(args.device)
    cases = [json.loads(line) for line in args.cases.read_text().splitlines() if line.strip()]
    if args.case_limit:
        cases = cases[:args.case_limit]
    if args.entities_input:
        entity_records = [json.loads(line) for line in args.entities_input.read_text().splitlines() if line.strip()]
        if [(r["case_id"], r["text"]) for r in entity_records] != [(r["case_id"], r["text"]) for r in cases]:
            raise ValueError("Reused NER records must exactly match the ordered cases and their cleaned bodies")
        for row in entity_records:
            for entity in row["entities"]:
                if row["text"][entity["start"]:entity["end"]] != entity["text"]:
                    raise ValueError("Reused entity span is not grounded in its case")
        print(f"Reusing {len(entity_records)} complete, exact-case-matched NER records", flush=True)
    else:
        checkpoint = torch.load(args.ner_checkpoint, map_location="cpu", weights_only=True, mmap=True)
        state = checkpoint["model_state_dict"]
        ner = NERModel(encoder_config(args.ner_tokenizer, state, "roberta"), len(checkpoint["label2id"]),
                       "roberta.pooler.dense.weight" in state)
        ner.load_state_dict(state, strict=True)
        ner.eval().to(device)
        ner_tokenizer = load_tokenizer(args.ner_tokenizer)
        id2label = {int(i): label for label, i in checkpoint["label2id"].items()}
        del checkpoint, state
        entity_records = []
        for first in range(0, len(cases), args.batch_size):
            batch = cases[first:first + args.batch_size]
            outputs = extract_long_cases(ner, ner_tokenizer, [row["text"] for row in batch], id2label,
                                         device, 0.0, args.batch_size)
            for row, entities in zip(batch, outputs):
                # The original builder de-duplicates exact name/type per case.
                seen, unique = set(), []
                for entity in entities:
                    signature = (entity["text"], entity["type"])
                    if signature not in seen:
                        seen.add(signature); unique.append(entity)
                entity_records.append({**row, "entities": unique, "status": "supplied_ner_checkpoint_prediction"})
            if first % 160 == 0:
                print(f"NER {min(first + len(batch), len(cases))}/{len(cases)}", flush=True)
        del ner
    with (args.output / "corpus_entities.jsonl").open("w", encoding="utf-8") as handle:
        for row in entity_records:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if device.type == "mps": torch.mps.empty_cache()

    dataset_class, model_class, class_hashes = source_classes(args.re_source)
    re_tokenizer = load_tokenizer(args.re_tokenizer, default_do_lower_case=False)
    saved = torch.load(args.re_checkpoint, map_location="cpu", weights_only=True, mmap=True)
    re_model = model_class(str(args.pretrained), len(saved["rel2id"]), dropout_rate=0.2)
    re_model.load_state_dict(saved["model_state_dict"], strict=True)
    re_model.eval().to(device)
    id2rel = {int(i): label for label, i in saved["rel2id"].items()}
    del saved
    locator = object.__new__(dataset_class)
    locator.tokenizer = re_tokenizer
    counts = Counter(); rejection = Counter(); relation_counts = Counter(); accepted_relations = Counter()
    nodes, edges = {}, {}
    raw_nodes, raw_edges = set(), set()
    checks = []
    case_rows = []
    raw_path = args.output / "all_pair_predictions.jsonl.gz"
    triple_path = args.output / "qualified_triples.jsonl"
    with gzip.open(raw_path, "wt", encoding="utf-8") as raw_handle, triple_path.open("w", encoding="utf-8") as triples_handle:
        for first in range(0, len(entity_records), args.batch_size):
            batch = entity_records[first:first + args.batch_size]
            encoded = re_tokenizer([row["text"] for row in batch], padding="max_length", truncation=True,
                                   max_length=128, return_offsets_mapping=True, return_tensors="pt")
            offsets = encoded.pop("offset_mapping").tolist()
            input_ids, masks = encoded["input_ids"].to(device), encoded["attention_mask"].to(device)
            with torch.inference_mode():
                hidden_batch = re_model.bert(input_ids=input_ids, attention_mask=masks).last_hidden_state
            # One batch transfer avoids thousands of small MPS reduction
            # launches. Inclusive float32 means are compared with source
            # forward below; model weights and classification math are intact.
            cpu_hidden_batch = hidden_batch.cpu()
            for local, row in enumerate(batch):
                entities = row["entities"]
                hidden = hidden_batch[local]
                cpu_hidden = cpu_hidden_batch[local]
                cpu_mask = encoded["attention_mask"][local].tolist()
                positions = []
                for entity in entities:
                    if args.locator == "original":
                        position = locator._locate_entity(row["text"], entity["text"], offsets[local])
                        actual = locate_offsets(offsets[local], entity["start"], entity["end"])
                        counts["original_locator_disagrees_with_true_span"] += int(tuple(position) != actual)
                        counts["original_locator_at_pad"] += int(cpu_mask[max(0, min(position[0], 127))] == 0)
                    else:
                        position = locate_offsets(offsets[local], entity["start"], entity["end"])
                        if position is None: rejection["truncated_entity_mention"] += 1
                    positions.append(position)
                features = []; feature_by_span = {}
                for position in positions:
                    if position is None:
                        features.append(torch.zeros(re_model.hidden_size)); continue
                    lo = max(0, min(position[0], 127)); hi = max(lo, min(position[1], 127))
                    if (lo, hi) not in feature_by_span:
                        feature_by_span[lo, hi] = cpu_hidden[lo:hi + 1].mean(0)
                    features.append(feature_by_span[lo, hi])
                features = torch.stack(features).to(device) if features else torch.empty((0, re_model.hidden_size), device=device)
                pairs = [(i, j) for i in range(len(entities)) for j in range(len(entities))
                         if i != j and entities[i]["text"] != entities[j]["text"]]
                raw_pairs = []; accepted_this_case = 0
                for pair_first in range(0, len(pairs), 512):
                    subset = pairs[pair_first:pair_first + 512]
                    valid = [(i, j) for i, j in subset if positions[i] is not None and positions[j] is not None]
                    rejection["pair_with_truncated_entity"] += len(subset) - len(valid)
                    if not valid: continue
                    head_indices = torch.tensor([i for i, _ in valid], device=device)
                    tail_indices = torch.tensor([j for _, j in valid], device=device)
                    with torch.inference_mode():
                        head_features = features[head_indices]; tail_features = features[tail_indices]
                        interacted = tail_features + re_model.entity_interact(torch.cat([head_features, tail_features], -1))
                        combined = torch.cat([hidden[0].expand(len(valid), -1), head_features, interacted], -1)
                        logits = re_model.classifier(re_model.dropout(combined))
                        if len(checks) < 12:
                            i, j = valid[0]
                            source_logits = re_model(input_ids[local:local + 1], masks[local:local + 1],
                                                      torch.tensor([positions[i]], device=device),
                                                      torch.tensor([positions[j]], device=device))
                            error = float((source_logits[0] - logits[0]).abs().max().cpu())
                            if error > 3e-5: raise AssertionError(f"Encoder cache differs from source forward by {error}")
                            checks.append({"case_id": row["case_id"], "head_index": i, "tail_index": j,
                                           "maximum_logit_difference": error, "atol": 3e-5})
                        scores, labels = logits.softmax(-1).max(-1)
                    for (i, j), score, label in zip(valid, scores.cpu().tolist(), labels.cpu().tolist()):
                        head, tail, relation = entities[i], entities[j], id2rel[label]
                        raw_pairs.append([i, j, label, score])
                        counts["classified_ordered_pairs"] += 1; relation_counts[relation] += 1
                        raw_nodes.update([(head["type"], head["text"]), (tail["type"], tail["text"])])
                        raw_edges.add((head["type"], head["text"], relation, tail["type"], tail["text"]))
                        if len(head["text"]) < 2 or len(tail["text"]) < 2:
                            rejection["short_entity_candidate"] += 1; continue
                        if score < args.re_confidence:
                            rejection["qualified_confidence_filter"] += 1; continue
                        if not relation_allowed(DEFAULT_ONTOLOGY, relation, head["type"], tail["type"]):
                            rejection["qualified_type_filter"] += 1; continue
                        triple = {"case_id": row["case_id"], "head": head["text"], "head_type": head["type"],
                                  "relation": relation, "tail": tail["text"], "tail_type": tail["type"],
                                  "probability": score, "head_start": head["start"], "tail_start": tail["start"],
                                  "status": "original_interaction_mlp_prediction_fixed_candidate_filters"}
                        triples_handle.write(json.dumps(triple, ensure_ascii=False) + "\n")
                        counts["qualified_candidate_observations"] += 1; accepted_this_case += 1
                        accepted_relations[relation] += 1
                        node_ids = []
                        for entity in (head, tail):
                            nid = stable_id("n_", [entity["type"], entity["text"]])
                            node = nodes.setdefault(nid, {"id": nid, "name": entity["text"], "entity_type": entity["type"],
                                                        "analysis_eligible": True, "labels": ["Entity"], "case_ids": [],
                                                        "prediction_not_gold": True})
                            if row["case_id"] not in node["case_ids"]: node["case_ids"].append(row["case_id"])
                            node_ids.append(nid)
                        eid = stable_id("e_", [node_ids[0], relation, node_ids[1]])
                        edge = edges.setdefault(eid, {"id": eid, "source": node_ids[0], "target": node_ids[1],
                                                     "relation_type": relation, "analysis_eligible": True,
                                                     "case_ids": [], "probabilities": [], "causal_gold_label": False})
                        edge["case_ids"].append(row["case_id"]); edge["probabilities"].append(score)
                counts["processed_cases"] += 1; counts["ner_entities"] += len(entities)
                counts["cases_with_qualified_edges"] += bool(accepted_this_case)
                case_rows.append({"case_id": row["case_id"], "entities": len(entities),
                                  "classified_pairs": len(raw_pairs), "qualified_edges": accepted_this_case})
                raw_handle.write(json.dumps({"case_id": row["case_id"], "entities": entities,
                                             "predictions_ij_relation_id_probability": raw_pairs}, ensure_ascii=False) + "\n")
            raw_handle.flush(); triples_handle.flush()
            print(f"RE {min(first + len(batch), len(entity_records))}/{len(entity_records)}; pairs={counts['classified_ordered_pairs']}; qualified={counts['qualified_candidate_observations']}", flush=True)
    for edge in edges.values():
        edge["mean_probability"] = float(np.mean(edge.pop("probabilities")))
        edge["case_ids"] = sorted(set(edge["case_ids"]))
    graph = {"schema": {"role": "model_predicted_qualified_candidates_not_gold", "raw_node_count": len(raw_nodes),
                         "raw_edge_count": len(raw_edges), "eligible_diesel_nodes": len(nodes), "eligible_diesel_edges": len(edges),
                         "source_cases": len(cases), "has_independent_causal_gold": False},
             "nodes": list(nodes.values()), "edges": list(edges.values())}
    write_json(args.output / "qualified_predicted_graph.json", graph)
    write_json(args.output / "case_counts.json", case_rows)
    summary = {"status": "complete", "processed_all_requested_cases": counts["processed_cases"] == len(cases),
               "counts": dict(counts), "rejection_counts": dict(rejection), "raw_relation_counts": dict(relation_counts),
               "qualified_relation_counts": dict(accepted_relations), "raw_graph_nodes": len(raw_nodes),
               "raw_graph_edges": len(raw_edges), "qualified_graph_nodes": len(nodes), "qualified_graph_edges": len(edges),
               "elapsed_seconds": time.perf_counter() - started, "encoder_cache_checks": checks,
               "pooling_execution": "float32 inclusive means on CPU from exact cached encoder outputs; source forward tolerance checked",
               "executed_wrapper_sha256": file_sha256(Path(__file__)),
               "locator": args.locator, "re_model": "original RelationModel entity_interact+MLP, strict=True",
               "ner": "supplied checkpoint, exact checkpoint label2id, strict=True; inference on cleaned case bodies",
               "ner_reuse": {"input": str(args.entities_input) if args.entities_input else None,
                             "sha256": file_sha256(args.entities_input) if args.entities_input else None,
                             "ordered_case_text_and_span_match_verified": True},
               "candidate_policy": {"all_ordered_pairs": True, "pair_cap": None, "raw_argmax_retained": True,
                                    "qualification": "both entity names length>=2, RE probability>=threshold, explicit entity-type ontology",
                                    "re_confidence": args.re_confidence, "ner_confidence": 0.0},
               "source_class_sha256": class_hashes,
               "input_sha256": {str(path): file_sha256(path) for path in
                                 (args.cases, args.ner_checkpoint, args.re_checkpoint, args.re_source)},
               "relation_id_map": id2rel,
               "limitations": ["No no_relation class exists: all-pair argmax is not reliable relation existence detection.",
                               "The supplied NER checkpoint produces fragmented mentions on this corpus; all outputs retained for audit.",
                               "Confidence is uncalibrated; qualification is a fixed implementation filter, not causal validation.",
                               "This inference graph is kept separate from the supplied historical Neo4j export."]}
    write_json(args.output / "run_summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("status", "counts", "raw_graph_nodes", "raw_graph_edges", "qualified_graph_nodes", "qualified_graph_edges", "elapsed_seconds")}, ensure_ascii=False), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", required=True, type=Path)
    p.add_argument("--ner-checkpoint", required=True, type=Path)
    p.add_argument("--ner-tokenizer", required=True, type=Path)
    p.add_argument("--re-checkpoint", required=True, type=Path)
    p.add_argument("--re-tokenizer", required=True, type=Path)
    p.add_argument("--re-source", required=True, type=Path)
    p.add_argument("--pretrained", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--locator", choices=("original", "offset"), default="original")
    p.add_argument("--device", choices=("cpu", "mps"), default="mps")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--re-confidence", type=float, default=0.8)
    p.add_argument("--case-limit", type=int, default=0)
    p.add_argument("--entities-input", type=Path, help="Reuse a complete, exact-case-matched NER output without changing the paired experiment")
    infer(p.parse_args())


if __name__ == "__main__":
    main()

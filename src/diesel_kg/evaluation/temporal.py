#!/usr/bin/env python3
"""Evaluate real timestamped fault episodes without inventing temporal data.

Observed event chronology is descriptive. Propagation-link delays additionally
require externally supported source/target event links. Association weights and
static graph path length are never converted into temporal speed measurements.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path

import numpy as np

EVENT_FIELDS = ["device_id", "episode_id", "event_id", "node_id", "timestamp", "level", "event_type", "system_id", "community_id", "evidence_ref"]
LINK_FIELDS = ["device_id", "episode_id", "source_event_id", "target_event_id", "evidence_ref"]
LEVELS = {"component": 0, "subsystem": 1, "engine": 2}
KINDS = {"fault_onset", "recovery", "intervention"}


def dump(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_csv(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fields})


def timestamp(value):
    t = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if t.tzinfo is None or t.utcoffset() is None:
        raise ValueError("timestamp must contain an explicit UTC offset or Z")
    return t.timestamp()


def describe(values):
    a = np.asarray(values, dtype=float)
    if not len(a):
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {"n": len(a), "mean": float(a.mean()), "median": float(np.median(a)), "min": float(a.min()), "max": float(a.max())}


def evaluate(events, links=None):
    valid, errors, seen = [], [], set()
    for row_no, r in enumerate(events, 2):
        try:
            if any(not r.get(k) for k in ["device_id", "episode_id", "event_id", "node_id", "timestamp", "event_type", "evidence_ref"]):
                raise ValueError("missing required identity/timestamp/evidence field")
            if r["event_type"] not in KINDS:
                raise ValueError("event_type must be fault_onset, recovery or intervention")
            if r["event_type"] == "fault_onset" and r.get("level") not in LEVELS:
                raise ValueError("fault_onset level must be component, subsystem or engine")
            key = (r["device_id"], r["episode_id"], r["event_id"])
            if key in seen:
                raise ValueError("duplicate event identity")
            seen.add(key)
            valid.append({**r, "timestamp_unix_seconds": timestamp(r["timestamp"])})
        except (ValueError, KeyError) as e:
            errors.append({"csv_row": row_no, "reason": str(e)})
    episodes = defaultdict(list)
    by_id = {}
    for r in valid:
        episodes[r["device_id"], r["episode_id"]].append(r)
        by_id[r["device_id"], r["episode_id"], r["event_id"]] = r
    chronological, transitions = [], []
    for (device, episode), ev in sorted(episodes.items()):
        ev.sort(key=lambda r: (r["timestamp_unix_seconds"], r["event_id"]))
        onsets = [r for r in ev if r["event_type"] == "fault_onset"]
        interventions = [r["timestamp_unix_seconds"] for r in ev if r["event_type"] == "intervention"]
        # First onset for each externally supplied level is observed chronology.
        first = {level: next((r for r in onsets if r["level"] == level), None) for level in LEVELS}
        complete = all(first.values())
        ordered = complete and first["component"]["timestamp_unix_seconds"] <= first["subsystem"]["timestamp_unix_seconds"] <= first["engine"]["timestamp_unix_seconds"]
        chronological.append({"device_id": device, "episode_id": episode, "n_onsets": len(onsets),
                              "n_interventions": len(interventions), "has_all_three_levels": complete,
                              "has_ordered_three_level_onsets": bool(ordered),
                              "first_component_event": first["component"]["event_id"] if first["component"] else None,
                              "first_subsystem_event": first["subsystem"]["event_id"] if first["subsystem"] else None,
                              "first_engine_event": first["engine"]["event_id"] if first["engine"] else None})
        for a, b in [("component", "subsystem"), ("subsystem", "engine"), ("component", "engine")]:
            if first[a] and first[b]:
                ta, tb = first[a]["timestamp_unix_seconds"], first[b]["timestamp_unix_seconds"]
                censored = any(ta < t <= tb for t in interventions)
                transitions.append({"device_id": device, "episode_id": episode, "source_level": a, "target_level": b,
                                    "source_event_id": first[a]["event_id"], "target_event_id": first[b]["event_id"],
                                    "observed_onset_delay_seconds": tb - ta, "positive_chronological_order": tb > ta,
                                    "intervention_between_onsets": censored, "verified_causal_propagation": False})
    supported_links, rejected_links = [], []
    for row_no, r in enumerate(links or [], 2):
        try:
            if any(not r.get(k) for k in LINK_FIELDS):
                raise ValueError("missing required link identity/evidence field")
            key = (r["device_id"], r["episode_id"])
            u = by_id[key + (r["source_event_id"],)]
            v = by_id[key + (r["target_event_id"],)]
            if u["event_type"] != "fault_onset" or v["event_type"] != "fault_onset":
                raise ValueError("propagation links must join two fault_onset events")
            delay = v["timestamp_unix_seconds"] - u["timestamp_unix_seconds"]
            if not math.isfinite(delay) or delay <= 0:
                raise ValueError("propagation delay must be strictly positive; simultaneous events are unresolved at this time resolution")
            intervention = any(e["event_type"] == "intervention" and u["timestamp_unix_seconds"] < e["timestamp_unix_seconds"] <= v["timestamp_unix_seconds"] for e in episodes[key])
            if intervention:
                raise ValueError("intervention occurs between linked onsets; natural propagation interval is censored")
            relation = "unmapped"
            if u.get("community_id") and v.get("community_id"):
                relation = "within_community" if u["community_id"] == v["community_id"] else "between_community"
            supported_links.append({**r, "source_node_id": u["node_id"], "target_node_id": v["node_id"],
                                    "source_level": u["level"], "target_level": v["level"], "delay_seconds": delay,
                                    "inverse_delay_per_hour": 3600 / delay, "community_relation": relation,
                                    "evidence_source": "externally supplied event-link evidence; not inferred merely from event ordering"})
        except (KeyError, ValueError) as e:
            rejected_links.append({"csv_row": row_no, "reason": str(e)})
    level_stats = {}
    for a, b in [("component", "subsystem"), ("subsystem", "engine"), ("component", "engine")]:
        usable = [r["observed_onset_delay_seconds"] for r in transitions if r["source_level"] == a and r["target_level"] == b and r["positive_chronological_order"] and not r["intervention_between_onsets"]]
        level_stats[a + "_to_" + b] = describe(usable)
    comparison = {scope: describe([r["delay_seconds"] for r in supported_links if r["community_relation"] == scope]) for scope in ["within_community", "between_community", "unmapped"]}
    summary = {"status": "computed" if valid else "not_evaluable", "n_events_input": len(events), "n_valid_events": len(valid),
               "invalid_event_rows": errors, "n_devices": len({r["device_id"] for r in valid}), "n_episodes": len(episodes),
               "n_complete_three_level_episodes": sum(r["has_all_three_levels"] for r in chronological),
               "n_ordered_three_level_episodes": sum(r["has_ordered_three_level_onsets"] for r in chronological),
               "observed_level_onset_delay_seconds": level_stats,
               "supported_propagation_link_status": "computed" if supported_links else "not_evaluable",
               "n_supported_positive_uncensored_links": len(supported_links), "invalid_link_rows": rejected_links,
               "within_between_link_delay_seconds": comparison,
               "inverse_delay_is_physical_spreading_velocity": False,
               "scope": "real timestamped episode onset chronology and externally evidenced link delays",
               "cannot_infer_causality_from_timestamps_alone": True,
               "qualitative_slow_medium_fast_cutoffs": None,
               "slow_medium_fast_reason": "paper provides no calibrated cutoffs; actual delay distributions are retained without arbitrary labels"}
    return summary, chronological, transitions, supported_links


def run(args):
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "events_input_template.csv", [], EVENT_FIELDS)
    write_csv(output / "propagation_links_input_template.csv", [], LINK_FIELDS)
    dump(output / "input_schema.json", {"events": {"required_fields": ["device_id", "episode_id", "event_id", "node_id", "timestamp", "level", "event_type", "evidence_ref"],
        "timestamp": "ISO 8601 with explicit timezone, e.g. UTC offset or Z; recorded event time, never graph position",
        "levels": LEVELS, "event_types": sorted(KINDS), "optional_fields": ["system_id", "community_id"],
        "evidence_ref": "real sensor/maintenance/event record reference"}, "propagation_links": {"required_fields": LINK_FIELDS,
        "definition": "externally supported causal source/target event links; chronology alone is insufficient"},
        "no_real_input_behavior": "not_evaluable with null results; no static weight/path-to-speed conversion"})
    if not args.events:
        summary = {"status": "not_evaluable", "propagation_delay_seconds": None, "three_level_temporal_evolution": None,
                   "missing_input": "real device/episode/event IDs, onset timestamps, level labels, interventions and evidence references",
                   "additional_input_for_propagation_speed_comparison": "externally supported source-target event links and community IDs",
                   "execution_code_implemented": True, "static_graph_used_as_time_data": False}
    else:
        events = read_csv(args.events)
        links = read_csv(args.links) if args.links else None
        summary, chronology, transitions, supported = evaluate(events, links)
        write_csv(output / "episode_chronology.csv", chronology, list(chronology[0]) if chronology else ["device_id", "episode_id"])
        write_csv(output / "observed_level_transitions.csv", transitions, list(transitions[0]) if transitions else ["device_id", "episode_id"])
        write_csv(output / "supported_event_link_delays.csv", supported, list(supported[0]) if supported else LINK_FIELDS)
        summary["input_provenance"] = {"events": {"path": str(args.events), "sha256": sha(args.events)},
                                      "links": {"path": str(args.links), "sha256": sha(args.links)} if args.links else None}
    dump(output / "analysis_summary.json", summary)
    print(json.dumps({"output": str(output), "status": summary["status"]}, ensure_ascii=False))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--events")
    p.add_argument("--links")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    if args.links and not args.events:
        p.error("--links requires --events")
    run(args)


if __name__ == "__main__":
    main()

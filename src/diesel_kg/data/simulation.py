"""Disclosed synthetic verification inputs, never measured paper evidence.

This generator plants mechanisms and then corrupts their graphs. It does not
run or estimate the performance of the trained NER/RE models. All quantities
except the paper's SIS weights are simulation assumptions.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timedelta, timezone
from decimal import Decimal, ROUND_HALF_UP
import hashlib
import json
import math
from pathlib import Path
import random
from statistics import mean

STATUS = "SIMULATED"
SEED = 20261008
SIS_WEIGHTS = {"S_SC": .30, "S_PC": .25, "S_CC": .25, "S_DA": .20}
FUSION_WEIGHTS = {"physical": .50, "semantic": .30, "cooccurrence": .20}
# Illustrative engineering narratives, not validated mechanisms for an engine model.
SYSTEMS = [
    ("fuel_combustion", "燃油喷射与燃烧", "喷油器", "喷油器堵塞", "燃烧不充分", "发动机动力下降", "intake_exhaust", "排气温度升高", "进气量不足"),
    ("hydraulic", "液压系统", "液压泵", "液压泵磨损", "液压压力不足", "执行响应迟缓", "transmission_steering", "换挡执行迟滞", "换挡阀卡滞"),
    ("cooling_heat", "冷却与热管理", "节温器", "节温器卡滞", "冷却效能下降", "发动机过热", "fuel_combustion", "燃烧稳定性下降", "喷油正时偏差"),
    ("electrical_control", "电气与控制", "温度传感器", "传感器信号偏差", "喷油控制偏差", "故障报警", "fuel_combustion", "喷油正时偏差", "喷油器磨损"),
    ("mechanical_sealing", "机械磨损与密封", "轴承", "润滑不足", "摩擦损失增大", "发动机振动增大", "cooling_heat", "冷却负荷升高", "冷却液不足"),
    ("intake_exhaust", "进排气系统", "空气滤清器", "空气滤清器堵塞", "进气量不足", "排气异常", "fuel_combustion", "空燃比偏低", "喷油量偏大"),
    ("transmission_steering", "传动与转向", "同步器", "同步器磨损", "负载波动增大", "传动冲击增大", "mechanical_sealing", "轴承载荷升高", "轴承间隙过大"),
]
SYSTEM_NAMES = {s[0]: s[1] for s in SYSTEMS}
PAPER_NEEDS = [
    {"section": "§3.1、§4.1–4.2、§6", "claim": "关系存在性、根因、完整传播链及核心节点", "needed": "逐对正负关系、症状查询、根因集合、完整有向链、核心节点", "files": "relation_labels.jsonl; cases.jsonl; graphs.json", "use": "检查关系标注结构、路径覆盖和根因检索的计算流程", "limit": "生成机制真值；未人工审核，不能替代真实标注"},
    {"section": "§3.3、§5", "claim": "社团与物理系统一致、系统耦合", "needed": "独立于社团算法的物理系统标签", "files": "physical_nodes.csv", "use": "示范物理系统标签如何关联节点", "limit": "预设的系统标签；没有生成或宣称真实社团一致性指标"},
    {"section": "§3.4、§4.1、表1", "claim": "物理、语义和共现融合的传播边权", "needed": "定量物理强度、语义分量、共现分量、融合参数", "files": "physical_edges.csv", "use": "演示融合权重和缺边、噪声边对路径的影响", "limit": "强度和融合系数均为假设；未校准为因果概率，不凑表1次数与均值"},
    {"section": "§6、表2、SIS公式", "claim": "诊断可解释性改善", "needed": "同案例、同评分人的四项成对评分", "files": "sis_ratings.csv", "use": "按原文0.30/0.25/0.25/0.20计算SIS", "limit": "虚拟评分人；不是专家评分，未开展趋势预测，因此S_DA上限8"},
    {"section": "§6、图8", "claim": "根因定位时间缩短", "needed": "工程师、案例、试验顺序、起止时刻、正确性和超时", "files": "engineer_trials.csv", "use": "成功成对试验的定位耗时比较，单列失败与超时", "limit": "虚拟工程师和时间；不是实测人工效率，也不是程序运行时间"},
    {"section": "§3.3、§5；§7局限", "claim": "局部/跨系统传播延迟、三级演化", "needed": "设备/事件/时间戳/层级、事件链、干预与恢复记录", "files": "events.csv; event_links.csv", "use": "时序扩展示例：计算正延迟及干预截断", "limit": "原文为静态离线方法；这些数据不证明其已实现动态传播预测"},
    {"section": "§2、§3.1–3.3", "claim": "真实语料、NER/RE、主题扫描、已有图谱分析", "needed": "现有项目中的真实输入、模型和实测结果", "files": "../experiments/", "use": "继续使用原有实测数据，不模拟替换", "limit": "本数据包不修改实测指标和权重，不伪造NER/RE准确率"},
]


def clipped(value, lo=0., hi=1.):
    return round(max(lo, min(hi, value)), 4)


def stamp(seconds):
    return (datetime(2024, 1, 1, tzinfo=timezone(timedelta(hours=8)))
            + timedelta(seconds=seconds)).isoformat(timespec="seconds")


def fusion_score(physical, semantic, cooccurrence):
    # Explicit decimal half-up rounding matches Excel ROUND at four places.
    values = {"physical": physical, "semantic": semantic, "cooccurrence": cooccurrence}
    total = sum(Decimal(str(FUSION_WEIGHTS[k])) * Decimal(str(values[k])) for k in values)
    return float(total.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP))


def build(seed=SEED, cases_per_system=12):
    if cases_per_system < 3 or cases_per_system % 3:
        raise ValueError("cases_per_system must be a positive multiple of 3 (at least 3)")
    import networkx as nx
    from diesel_kg.evaluation.engineering import infer_query
    from diesel_kg.evaluation.temporal import evaluate as temporal_evaluate

    rng = random.Random(seed)
    cases, nodes, edges, relations, ratings, trials, events, event_links = ([] for _ in range(8))
    graph_edges = {name: [] for name in ("base", "enhanced")}
    case_metrics = []
    rater_bias = {f"SIM_RATER_{r:02}": rng.gauss(0, .35) for r in range(1, 4)}
    engineer_speed = {f"SIM_ENGINEER_{r:02}": rng.lognormvariate(0, .15) for r in range(1, 7)}

    for sys_index, system in enumerate(SYSTEMS):
        sid, system_name, component, root_name, state_name, outcome_name, other_sid, cross_name, other_cause = system
        for replicate in range(cases_per_system):
            index = sys_index * cases_per_system + replicate + 1
            cid = f"SIM_CASE_{index:03}"
            mode = ("linear", "branching", "convergent")[replicate % 3]
            device, episode = f"SIM_DEVICE_{sys_index + 1:02}_{replicate % 3 + 1:02}", f"SIM_EPISODE_{index:03}"
            local = []

            def node(role, name, kind, level, system_id):
                record = {"node_id": f"{cid}_{role}", "case_id": cid, "name": name,
                          "entity_type": kind, "level": level, "system_id": system_id,
                          "system_name": SYSTEM_NAMES[system_id], "component_context": component,
                          "label_source": "planted_generation_rule_not_community_output",
                          "adjudicated": False, "data_status": STATUS}
                nodes.append(record); local.append(record)
                return record["node_id"]

            root = node("R", root_name, "故障原因", "component", sid)
            middle = node("M", state_name, "故障状态", "subsystem", sid)
            outcome = node("O", outcome_name, "影响后果", "engine", sid)
            distractor = node("D", "无关接插件松动", "故障原因", "component", sid)
            roots, paths = [root], [[root, middle, outcome]]
            if mode == "branching":
                cross = node("X", cross_name, "故障状态", "subsystem", other_sid)
                paths.append([root, middle, cross, outcome])
            elif mode == "convergent":
                second = node("R2", other_cause, "故障原因", "component", other_sid)
                roots.append(second); paths.append([second, middle, outcome])
            by_id = {n["node_id"]: n for n in local}
            truth_pairs = sorted({(u, v) for path in paths for u, v in zip(path, path[1:])})
            local_edges = []
            for number, (u, v) in enumerate(truth_pairs + [(distractor, outcome)], 1):
                true = (u, v) in truth_pairs
                cross_system = by_id[u]["system_id"] != by_id[v]["system_id"]
                physical = rng.betavariate(6, 3) if true and not cross_system else rng.betavariate(4, 4) if true else rng.uniform(.05, .65)
                semantic = rng.betavariate(7, 2) if true else rng.betavariate(3, 4)
                context_count = rng.randint(6, 20) if true else rng.randint(0, 12)
                physical, semantic = clipped(physical), clipped(semantic)
                fused = fusion_score(physical, semantic, context_count / 20)
                present = {"base": rng.random() < (.70 if true else .12),
                           "enhanced": rng.random() < (.84 if true else .15)}
                row = {"edge_id": f"{cid}_EDGE_{number:02}", "case_id": cid,
                       "source": u, "target": v, "source_name": by_id[u]["name"], "target_name": by_id[v]["name"],
                       "source_system": by_id[u]["system_id"], "target_system": by_id[v]["system_id"],
                       "oracle_relation_exists": true, "physical_strength_assumed": clipped(physical),
                       "semantic_similarity_assumed": clipped(semantic), "synthetic_context_count": context_count,
                       "context_count_reference": 20, "cooccurrence_normalized": context_count / 20,
                       "base_weight": clipped(physical), "enhanced_weight": fused,
                       "base_present": present["base"], "enhanced_present": present["enhanced"],
                       "evidence_ref": f"SIM_GENERATOR:{cid}:edge{number}", "data_status": STATUS}
                edges.append(row); local_edges.append(row)
                for name in present:
                    if present[name]:
                        graph_edges[name].append({"id": row["edge_id"], "source": u, "target": v,
                            "weight": row[name + "_weight"], "kind": "causal_candidate",
                            "case_support": [cid], "basis_base_edge_ids": [], "data_status": STATUS})
                head, tail = by_id[u]["name"], by_id[v]["name"]
                prefix = f"{cid}模拟机制：" if true else f"{cid}模拟反例："
                connector = "导致" if true else "与"
                suffix = "。" if true else "在该生成机制中没有直接关系。"
                text = prefix + head + connector + tail + suffix
                start_h, start_t = len(prefix), len(prefix + head + connector)
                relations.append({"sample_id": row["edge_id"], "case_id": cid, "text": text,
                    "head": {"node_id": u, "text": head, "start": start_h, "end": start_h + len(head), "type": by_id[u]["entity_type"]},
                    "tail": {"node_id": v, "text": tail, "start": start_t, "end": start_t + len(tail), "type": by_id[v]["entity_type"]},
                    "relation": "导致" if true else "no_relation", "reviewed_negative": not true,
                    "reviewer_id": "SIM_ANNOTATOR_01", "review_is_simulated": True,
                    "review_evidence": row["evidence_ref"], "data_status": STATUS})

            severity = rng.uniform(.3, .95)
            # Censor one in seven episodes after subsystem onset; no invented later onset.
            interrupted = index % 7 == 0
            onset = {root: 0, middle: rng.randint(90, 540)}
            if mode == "branching":
                onset[cross] = onset[middle] + rng.randint(150, 1100)
            elif mode == "convergent":
                onset[second] = rng.randint(10, 80)
            if not interrupted:
                onset[outcome] = max(onset.values()) + rng.randint(120, 1200)
            clock = index * 86400
            event_ids = {}
            for sequence, (nid, delay) in enumerate(sorted(onset.items(), key=lambda pair: pair[1]), 1):
                event_id = f"{episode}_ONSET_{sequence:02}"
                event_ids[nid] = event_id
                n = by_id[nid]
                events.append({"device_id": device, "episode_id": episode, "case_id": cid, "event_id": event_id,
                    "node_id": nid, "node_name": n["name"], "timestamp": stamp(clock + delay), "level": n["level"],
                    "event_type": "fault_onset", "system_id": n["system_id"], "community_id": "SIM_SYSTEM_" + n["system_id"],
                    "evidence_ref": f"SIM_GENERATOR:{event_id}", "data_status": STATUS})
            if interrupted:
                for kind, delay in [("intervention", max(onset.values()) + 60), ("recovery", max(onset.values()) + 600)]:
                    events.append({"device_id": device, "episode_id": episode, "case_id": cid,
                        "event_id": f"{episode}_{kind.upper()}", "node_id": root, "node_name": root_name,
                        "timestamp": stamp(clock + delay), "level": "component", "event_type": kind,
                        "system_id": sid, "community_id": "SIM_SYSTEM_" + sid,
                        "evidence_ref": f"SIM_GENERATOR:{episode}:{kind}", "data_status": STATUS})
            for u, v in truth_pairs:
                if u in onset and v in onset:
                    event_links.append({"device_id": device, "episode_id": episode, "case_id": cid,
                        "source_event_id": event_ids[u], "target_event_id": event_ids[v],
                        "source_node_id": u, "target_node_id": v, "delay_seconds": onset[v] - onset[u],
                        "system_relation": "within_system" if by_id[u]["system_id"] == by_id[v]["system_id"] else "between_system",
                        "evidence_ref": f"SIM_GENERATOR:{cid}:{u}->{v}", "data_status": STATUS})

            presentation = f"假定下游查询症状为{outcome_name}" if interrupted else f"模拟观察到{outcome_name}"
            record = {"case_id": cid, "source_group_id": f"SIM_TEMPLATE_{sid}_{mode}",
                "system_id": sid, "system_name": system_name, "mode": mode, "severity_assumed": round(severity, 4),
                "text": f"{cid}，虚拟{component}工况，负载系数{severity:.2f}，{presentation}；预设机制为{root_name}、{state_name}。",
                "device_id": device, "episode_id": episode, "query_symptom_node_ids": [outcome],
                "root_cause_node_ids": roots, "core_node_ids": [root, middle] + ([cross] if mode == "branching" else [second] if mode == "convergent" else []),
                "propagation_paths": paths, "path_names": [[by_id[n]["name"] for n in p] for p in paths],
                "adjudicated": False, "annotator_ids": ["SIM_ANNOTATOR_01", "SIM_ANNOTATOR_02"],
                "review_is_simulated": True, "engine_onset_observed": not interrupted,
                "query_is_hypothetical_when_interrupted": interrupted, "data_status": STATUS}
            cases.append(record)
            metrics = {"case_id": cid, "mode": mode, "system_id": sid, "data_status": STATUS}
            for name in ("base", "enhanced"):
                g = nx.DiGraph()
                g.add_nodes_from(by_id)
                present_pairs = set()
                for edge in local_edges:
                    if edge[name + "_present"]:
                        w = edge[name + "_weight"]
                        g.add_edge(edge["source"], edge["target"], weight=w, cost=-math.log(w),
                                   kinds=["causal_candidate"], edge_ids=[edge["edge_id"]], basis_base_edge_ids=[], case_support=[cid])
                        present_pairs.add((edge["source"], edge["target"]))
                result = infer_query([outcome], {n: {**v, "id": n} for n, v in by_id.items()}, g, set(roots + [distractor]), core_k=3)
                top1 = result["candidate_roots"][0]["node_id"] if result["candidate_roots"] else ""
                complete = sum(all((u, v) in present_pairs for u, v in zip(p, p[1:])) for p in paths) / len(paths)
                core = [n["node_id"] for n in result["core_nodes"][:3]]
                metrics.update({name + "_root_top1": int(top1 in roots), name + "_predicted_root": top1,
                    name + "_gold_path_coverage": complete,
                    name + "_core_precision_at3": len(set(core) & set(record["core_node_ids"])) / 3,
                    name + "_cross_edge_coverage": mean([int(e[name + "_present"]) for e in local_edges if e["oracle_relation_exists"] and e["source_system"] != e["target_system"]]) if mode != "linear" else None})
            case_metrics.append(metrics)

            # SIS uses the non-linear subset, where cross-system coupling is relevant.
            if mode != "linear":
                for rater, bias in rater_bias.items():
                    for name in ("base", "enhanced"):
                        latent = {"S_SC": (5.4 if name == "base" else 7.0) - .5 * severity,
                            "S_PC": 3.0 + 5.4 * metrics[name + "_gold_path_coverage"],
                            "S_CC": 3.0 + 5.2 * metrics[name + "_cross_edge_coverage"],
                            "S_DA": 4.2 + 2.8 * metrics[name + "_root_top1"]}
                        scores = {k: round(clipped(v + bias + rng.gauss(0, .65), 0, 8 if k == "S_DA" else 10), 1) for k, v in latent.items()}
                        ratings.append({"case_id": cid, "graph": name, "rater_id": rater, **scores,
                            "SIS": round(sum(SIS_WEIGHTS[k] * scores[k] for k in scores), 4),
                            "rating_source": "virtual_rater_conditional_on_generated_graph_not_actual_expert",
                            **{"evidence_" + k[2:]: f"SIM_GENERATOR:{cid}:{name}:{k}" for k in scores},
                            "trend_evidence_ref": "", "data_status": STATUS})

            for engineer in rng.sample(sorted(engineer_speed), 2):
                base_minutes = (9 + severity * 10 + (3 if mode != "linear" else 0)) * engineer_speed[engineer] * rng.lognormvariate(0, .15)
                # Heterogeneous benefit: some assisted trials take longer.
                multiplier = rng.lognormvariate(math.log(.78), .28)
                trial = {"pair_id": f"{cid}_{engineer}", "case_id": cid, "engineer_id": engineer,
                    "order": rng.choice(["base_then_enhanced", "enhanced_then_base"]),
                    "timeout_limit_seconds": 1800, "evidence_ref": f"SIM_GENERATOR:{cid}:{engineer}", "data_status": STATUS}
                for name, minutes in [("base", base_minutes), ("enhanced", base_minutes * multiplier)]:
                    correct = rng.random() < (.70 + .23 * metrics[name + "_root_top1"])
                    attempted = max(60, round(minutes * 60 * (1 if correct else 1.35)))
                    timeout = attempted >= 1800
                    duration = min(1800, attempted)
                    slot = 0 if trial["order"].startswith(name) else 1
                    start = 366 * 86400 + index * 86400 + int(engineer[-2:]) * 7200 + slot * 3600
                    trial.update({name + "_start": stamp(start), name + "_end": stamp(start + duration),
                        name + "_duration_seconds": duration, name + "_success": bool(correct and not timeout), name + "_timeout": timeout})
                trial["paired_success"] = trial["base_success"] and trial["enhanced_success"]
                trial["paired_time_reduction_fraction"] = (trial["base_duration_seconds"] - trial["enhanced_duration_seconds"]) / trial["base_duration_seconds"] if trial["paired_success"] else None
                trials.append(trial)

    temporal, _, _, _ = temporal_evaluate(events, event_links)
    temporal.update(data_status=STATUS, scope="SIMULATED temporal extension; planted event links and assumed system groups, no real causal evidence",
                    community_labels_are_algorithm_output=False)
    successful_pairs = [t for t in trials if t["paired_success"]]
    summary = {"cases": len(cases), "systems": 7, "nodes": len(nodes), "relation_pairs": len(relations),
        "positive_relations": sum(r["relation"] != "no_relation" for r in relations),
        "negative_relations": sum(r["relation"] == "no_relation" for r in relations),
        "gold_paths": sum(len(c["propagation_paths"]) for c in cases), "sis_cases": len({r["case_id"] for r in ratings}),
        "sis_raters": 3, "sis_rows": len(ratings), "engineers": 6, "timing_pairs": len(trials),
        "timing_trials": 2 * len(trials), "paired_successful_timing_pairs": len(successful_pairs),
        "events": len(events), "event_links": len(event_links), "data_status": STATUS}
    for name in ("base", "enhanced"):
        summary[name] = {"root_top1": mean(r[name + "_root_top1"] for r in case_metrics),
            "gold_path_coverage": mean(r[name + "_gold_path_coverage"] for r in case_metrics),
            "core_precision_at3": mean(r[name + "_core_precision_at3"] for r in case_metrics),
            "SIS": mean(r["SIS"] for r in ratings if r["graph"] == name),
            "paired_success_mean_minutes": mean(t[name + "_duration_seconds"] / 60 for t in successful_pairs) if successful_pairs else None,
            "trial_success_rate": mean(t[name + "_success"] for t in trials),
            "timeouts": sum(t[name + "_timeout"] for t in trials)}
    summary["paired_mean_time_reduction_fraction"] = mean(t["paired_time_reduction_fraction"] for t in successful_pairs) if successful_pairs else None
    return {"cases": cases, "physical_nodes": nodes, "physical_edges": edges, "relations": relations,
        "sis_ratings": ratings, "engineer_trials": trials, "events": events, "event_links": event_links,
        "graphs": {"data_status": STATUS, "graph_generation": "oracle edges with assumed omissions/noise; not trained model outputs",
                   "nodes": [{**n, "id": n["node_id"]} for n in nodes], "analysis_edges": graph_edges},
        "manifest": {"data_status": STATUS, "purpose": "原论文正文验证方案配套仿真数据与计算验证", "seed": seed, "cases_per_system": cases_per_system,
            "paper_needs": PAPER_NEEDS, "sis_weights": SIS_WEIGHTS, "fusion_weights_assumed": FUSION_WEIGHTS,
            "summary": summary, "case_metrics": case_metrics, "temporal_calculation": temporal,
            "assumptions": ["七系统×每系统案例数；线性、分支、汇聚均衡；同模板案例不构成独立临床/工业样本",
                "机制叙述为工程常识启发的简化假设，未针对具体柴油机型号确认；节点为SIM_CASE独有，不能直接冒充真实语料节点",
                "物理系统标签在生成时指定，未读取Louvain或主题分类结果；不由本数据包宣称社团一致性或燃油系统主导性",
                "真关系物理强度：系统内Beta(6,3)、跨系统Beta(4,4)；无关系噪声Uniform(0.05,0.65)",
                "语义分量真关系Beta(7,2)、噪声Beta(3,4)；虚拟共现计数真关系6–20、噪声0–12，除以20归一化",
                "基础/增强图真边保留概率0.70/0.84、噪声边保留概率0.12/0.15；增强优势是设定，非模型实测增益",
                "基础边权为假定物理强度；增强融合0.50物理+0.30语义+0.20共现，原文没有指定这些系数；分量非因果概率",
                "路径完整率为生成真值链在图中的完整覆盖率；根因Top1由现有反向最大乘积检索实际计算；核心precision@3分母固定3",
                "SIS只覆盖有跨系统链的分支/汇聚案例；3名虚拟评分人，评分受生成图条件与噪声影响，不能代替专家效度检验",
                "S_DA不高于8，不假造趋势预测验证；SIS权重唯一来源是原文，其他评分分布为假设",
                "6名虚拟工程师每案例抽2名、顺序随机平衡；忽略学习/记忆效应；单试验30分钟超时",
                "基础计时=(9+10*严重度+非线性加3)分钟×工程师速度×LogNormal(0,0.15)；增强倍率LogNormal(log(0.78),0.28)",
                "计时正确概率=0.70+0.23×图根因Top1命中；失败耗时乘1.35；均值耗时改善仅用两方法均成功的同一案例/工程师",
                "虚拟时区UTC+08，起点2024-01-01；正延迟独立抽样，系统内/跨系统分布重叠；每7例1例干预截断，不生成后续发动机发作",
                "事件组标签是预设物理系统，不是检测得到的社团；时间戳是扩展演示，不能证明静态图能预测演化",
                "没有复刻原文表1次数/均值、图中目标数值或NER/RE指标；没有读取、修改真实实验预测/模型权重"],
            "not_real_expert_or_engineer_records": True, "supports_original_empirical_claims": False}}


def validate(data):
    """Check joins, spans, units, graph truth and missing/censored outcomes."""
    cases = {c["case_id"]: c for c in data["cases"]}
    nodes = {n["node_id"]: n for n in data["physical_nodes"]}
    for key in ("cases", "physical_nodes", "physical_edges", "relations", "sis_ratings", "engineer_trials", "events", "event_links"):
        if any(r.get("data_status") != STATUS for r in data[key]):
            raise ValueError(f"Missing simulation label in {key}")
    for r in data["relations"]:
        for role in ("head", "tail"):
            mention = r[role]
            if r["text"][mention["start"]:mention["end"]] != mention["text"] or mention["node_id"] not in nodes:
                raise ValueError("Relation span or node join invalid")
    truth = {(e["source"], e["target"]) for e in data["physical_edges"] if e["oracle_relation_exists"]}
    for case in cases.values():
        for path in case["propagation_paths"]:
            if any((u, v) not in truth for u, v in zip(path, path[1:])):
                raise ValueError("Gold path not supported by planted edges")
    panels = {}
    for r in data["sis_ratings"]:
        if r["case_id"] not in cases or any(not 0 <= r[k] <= 10 for k in SIS_WEIGHTS):
            raise ValueError("SIS panel invalid")
        key = (r["case_id"], r["rater_id"])
        if r["graph"] in panels.setdefault(key, set()):
            raise ValueError("Duplicate SIS observation")
        panels[key].add(r["graph"])
    if any(panel != {"base", "enhanced"} for panel in panels.values()):
        raise ValueError("Unpaired SIS panel")
    for t in data["engineer_trials"]:
        for name in ("base", "enhanced"):
            delta = (datetime.fromisoformat(t[name + "_end"]) - datetime.fromisoformat(t[name + "_start"])).total_seconds()
            if delta != t[name + "_duration_seconds"] or not 0 < delta <= t["timeout_limit_seconds"]:
                raise ValueError("Timing endpoints or units inconsistent")
        if (t["paired_time_reduction_fraction"] is not None) != t["paired_success"]:
            raise ValueError("Failed/censored pair enters timing improvement")
    event_ids = {e["event_id"]: e for e in data["events"]}
    if len(event_ids) != len(data["events"]):
        raise ValueError("Duplicate event IDs")
    for link in data["event_links"]:
        u, v = (event_ids[link[k + "_event_id"]] for k in ("source", "target"))
        if u["episode_id"] != v["episode_id"] or u["episode_id"] != link["episode_id"]:
            raise ValueError("Event link crosses episodes")
        delta = (datetime.fromisoformat(v["timestamp"]) - datetime.fromisoformat(u["timestamp"])).total_seconds()
        if delta <= 0 or delta != link["delay_seconds"]:
            raise ValueError("Non-positive or inconsistent propagation delay")
    temporal = data["manifest"]["temporal_calculation"]
    if temporal["invalid_event_rows"] or temporal["invalid_link_rows"]:
        raise ValueError("Temporal evaluator rejected synthetic inputs")
    return {"status": "passed", "data_status": STATUS, "checks": ["all-row provenance", "relation spans and joins",
        "gold chain edges", "paired SIS", "timing endpoints and censoring", "event identity and positive delays", "existing temporal evaluator"]}


def write_bundle(output, data, paper_source=None):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Refusing to overwrite a non-empty output directory: {output}")
    validation = validate(data)
    output.mkdir(parents=True, exist_ok=True)
    for name, key in [("cases.jsonl", "cases"), ("relation_labels.jsonl", "relations")]:
        (output / name).write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in data[key]), encoding="utf-8")
    for key in ("physical_nodes", "physical_edges", "sis_ratings", "engineer_trials", "events", "event_links"):
        with (output / f"{key}.csv").open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[key][0]))
            writer.writeheader(); writer.writerows(data[key])
    (output / "graphs.json").write_text(json.dumps(data["graphs"], ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    manifest = data["manifest"]
    manifest["validation"] = validation
    manifest["generator_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    if paper_source:
        manifest["paper_source_sha256"] = hashlib.sha256(Path(paper_source).read_bytes()).hexdigest()
    manifest["files_sha256"] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(output.iterdir())}
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def main():
    parser = argparse.ArgumentParser(description="Generate SIMULATED paper-verification data; no real evidence is created")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--paper-source", type=Path)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--cases-per-system", type=int, default=12)
    args = parser.parse_args()
    manifest = write_bundle(args.output, build(args.seed, args.cases_per_system), args.paper_source)
    print(json.dumps({"data_status": STATUS, "output": str(args.output), "summary": manifest["summary"], "validation": manifest["validation"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

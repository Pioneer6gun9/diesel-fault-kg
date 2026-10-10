"""Check literal evidence for the ten manuscript chains with frozen aliases.

Counts mean three concept mentions in the same original case. They are neither
ordered causal observations nor an exact recovery of the author's chain IDs.
The manuscript counts/weights are copied only into separate reference columns.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import re

ALIASES = [
    [["供油异常", "供油不足", "燃油供给异常", "燃油异常", "燃油不足"], ["燃烧不充分", "不完全燃烧", "燃烧不完全", "燃烧不足"], ["动力下降", "动力不足", "功率下降", "加速无力"]],
    [["冷却失效", "冷却故障", "散热不良", "散热故障", "冷却不足", "散热器堵塞"], ["温度升高", "温度过高", "水温过高", "过热", "高温", "温度异常"], ["启动困难", "难启动", "无法启动"]],
    [["喷油器堵塞", "喷嘴堵塞", r"喷油器[^，。；,.;]{0,15}堵塞"], ["喷射效率低", "喷油效率低", "效率下降", "效率低"], ["排放异常", "排气异常", "黑烟", "冒烟", "排放超标"]],
    [["机油污染", "液压油污染", "油液污染", "油污染", "油液杂质"], ["阀芯卡滞", "阀芯卡死", "滑阀卡滞", r"阀芯[^，。；,.;]{0,15}卡滞"], ["压力不足", "压力下降", "工作压力不足"]],
    [["噪声", "噪音", "异响"], ["齿轮磨损", r"齿轮[^，。；,.;]{0,15}磨损"], ["振动增大", "振动增加", "振动过大", "震动增大"]],
    [["滤清器堵塞", "空气滤芯堵塞", "进气滤清器堵塞", r"滤清器[^，。；,.;]{0,15}堵塞"], ["进气不足", "进气量不足", "进气量下降"], ["排气压力高", "排气压力过高", "排气背压高", "排气背压过高"]],
    [["换挡困难", "挂挡困难"], ["同步器磨损", r"同步器[^，。；,.;]{0,15}磨损"], ["换挡冲击", "挂挡冲击"]],
    [["传感器异常", "传感器故障", "信号漂移", "传感器误差"], ["ECU误判", "ECU误判断", "ECU判断错误"], ["故障报警", "故障码", "故障告警", "报警"]],
    [["节温器卡滞", r"节温器[^，。；,.;]{0,15}卡滞"], ["温度异常", "水温异常", "水温过高", "发动机过热"], ["热风险", "过热风险", "热损伤", "热负荷", "发动机损坏"]],
    [["润滑不良", "润滑不足", "机油不足"], ["轴承磨损", r"轴承[^，。；,.;]{0,15}磨损"], ["振动", "震动"]],
]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases", type=Path, required=True)
    p.add_argument("--paper-targets", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); a.output.mkdir(exist_ok=True, parents=True)
    cases = [json.loads(line) for line in a.cases.read_text().splitlines() if line.strip()]
    reference = json.loads(a.paper_targets.read_text())["tables"]["1"]["rows"]
    if len(reference) != len(ALIASES): raise ValueError("The alias protocol must explicitly map every manuscript chain")
    results, evidence = [], []
    for index, (target, concepts) in enumerate(zip(reference, ALIASES), 1):
        support, term_support = [], [set(), set(), set()]
        for case in cases:
            hits = [[{"pattern": pattern, "span": list(match.span()), "mention": match.group()}
                     for pattern in patterns for match in re.finditer(pattern, case["text"], re.I)] for patterns in concepts]
            for j, hit in enumerate(hits):
                if hit: term_support[j].add(case["case_id"])
            if all(hits):
                support.append(case)
                evidence.append({"chain_index": index, "case_id": case["case_id"], "text": case["text"],
                                 "three_concept_matches": hits, "observed_causal_sequence": False})
        results.append({"chain_index": index, "paper_chain": target["chain"], "paper_reference_count": target["count"],
                        "paper_reference_weight": target["average_weight"], "computed_three_concept_comention_cases": len(support),
                        "computed_unique_body_support": len({case["text"] for case in support}),
                        "concept1_case_count": len(term_support[0]), "concept2_case_count": len(term_support[1]), "concept3_case_count": len(term_support[2]),
                        "case_ids": [case["case_id"] for case in support], "computed_average_propagation_weight": None,
                        "qualification": "fixed_alias_literal_co_mentions_only_not_observed_causal_frequency; absent original node/chain IDs"})
    with (a.output / "paper_table1_literal_support.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0])); writer.writeheader()
        writer.writerows([{**r, "case_ids": json.dumps(r["case_ids"], ensure_ascii=False)} for r in results])
    (a.output / "matched_case_evidence.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in evidence))
    protocol = {"aliases_in_paper_table_order": ALIASES, "case_count": len(cases), "duplicates_retained": True,
                "count_definition": "all three fixed bilingual concepts mentioned in one numbered case; no causal/temporal gold",
                "aliases_not_adapted_to_reference_counts": True,
                "source_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in (a.cases, a.paper_targets, Path(__file__))},
                "limitations": ["An alias miss can be an unlisted synonym, not proof that the paper chain is absent.", "A match can include repair/negated mentions and does not validate propagation order.", "Average weights need original canonical chain IDs; no propagation weights inferred from mention matching."],
                "results": results}
    (a.output / "run_summary.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"n_paper_chains_checked": len(results), "measured_literal_support_counts": [r["computed_three_concept_comention_cases"] for r in results]}, ensure_ascii=False))


if __name__ == "__main__": main()

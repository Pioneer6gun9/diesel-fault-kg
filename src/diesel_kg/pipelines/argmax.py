"""Independent CPU execution of the saved-argmax threshold-zero graph branch.

This branch preserves the main 0.8 outputs. It changes no saved class labels,
ontology, fusion coefficients or source case support. A completed execution
establishes computational provenance, not precision or causal validation.
"""
from __future__ import annotations
from diesel_kg.paths import PROJECT_ROOT, resolve_script, source_dependencies

import argparse
import ast
from collections import defaultdict
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from diesel_kg.validation.framework import sha, load, rows, csv_rows, _simple_paths, _candidate_in_case

HERE = PROJECT_ROOT
ROLE = "independent_saved_argmax_threshold0_candidate_comparison_not_causal_gold"


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def file_refs(paths):
    return {str(Path(p).resolve()): sha(p) for p in paths}


def dependencies(script):
    return source_dependencies(script, HERE)


def check_threshold_ready(threshold_dir, topics_dir, cases):
    """Reject partial exports and inputs that have changed after the audit."""
    threshold_dir, topics_dir, cases = map(Path, (threshold_dir, topics_dir, cases))
    summary = load(threshold_dir / "analysis_summary.json")
    configuration = load(threshold_dir / "run_config.json")
    if (summary.get("status") != "computed_sensitivity_diagnostic_only" or
            summary.get("main_0_8_results_modified_or_replaced") is not False or
            summary.get("argmax_labels_changed_to_fit_ontology") is not False or
            not configuration.get("export_argmax_graph")):
        raise ValueError("Threshold audit is incomplete or is not an independent saved-label comparison")
    binding = summary.get("source_RE_binding", {})
    if (binding.get("status") != "recorded_metadata_and_complete_raw_counts_consistent" or
            binding.get("saved_argmax_label_mapping_verified_against_source_relation_counts") is not True or
            binding.get("main_0_8_rejection_and_retention_counts_reproduced") is not True):
        raise ValueError("Threshold audit is not bound to the unchanged complete source RE run")
    inputs = configuration.get("inputs", {})
    for name in ["raw_predictions", "entities", "rel2id", "source_RE_run_summary"]:
        ref = inputs.get(name, {})
        if not Path(ref.get("path", "")).is_file() or sha(ref["path"]) != ref.get("sha256"):
            raise ValueError("Audited input is missing or changed: " + name)
    corpus = rows(cases)
    if len(corpus) != len({c["case_id"] for c in corpus}): raise ValueError("Duplicate case IDs")
    topic_summary = load(topics_dir / "analysis_summary.json")
    if (topic_summary.get("input", {}).get("entities_sha256") != inputs["entities"]["sha256"] or
            topic_summary.get("input", {}).get("cases_sha256") != sha(cases)):
        raise ValueError("Topic weights must come from this actual NER dictionary and case corpus")
    source = load(threshold_dir / "threshold0_qualified_predicted_graph.json")
    schema = source.get("schema", {})
    if (schema.get("support_mode") != "model_occurrences" or schema.get("confidence_threshold") != 0 or
            schema.get("role") != "model_qualified_candidates_not_causal_gold"):
        raise ValueError("Argmax graph has the wrong evidence scope")
    measured = summary["thresholds"]["0.0"]
    if len(corpus) != measured["n_all_cases_denominator"] or len(source["edges"]) != measured["n_typed_canonical_edges"]:
        raise ValueError("Argmax graph/corpus totals disagree with the completed audit")
    if not source["edges"] and (source["nodes"] or measured["n_retained_occurrence_pairs"]):
        raise ValueError("Empty argmax graph cannot contain orphan nodes or retained observations")
    (threshold_dir / "threshold0_qualified_triples.jsonl").stat()
    for filename in ["document_topic.csv", "case_system_probabilities.csv"]:
        (topics_dir / "paper_fixed_K07" / filename).stat()
    return summary, configuration, source, corpus


def verify_argmax_graph(threshold_dir, topics_dir, cases_file, folder):
    """Independent case-support, uncapped path and downstream-source checks."""
    from diesel_kg.graph.analysis import RELATIONS, MECHANISMS, FAULT_TYPES
    threshold_dir, topics_dir, cases_file, folder = map(Path, (threshold_dir, topics_dir, cases_file, folder))
    threshold, threshold_config, source, cases = check_threshold_ready(threshold_dir, topics_dir, cases_file)
    checks = []
    def check(name, passed, detail=None): checks.append({"name": name, "passed": bool(passed), "detail": detail})
    case_text = {c["case_id"]: c["text"] for c in cases}; case_ids = set(case_text)
    nodes = {n["id"]: n for n in source["nodes"]}; edges = {e["id"]: e for e in source["edges"]}
    semantic = {(n["entity_type"], n["name"]): nid for nid,n in nodes.items()}
    observations = rows(threshold_dir / "threshold0_qualified_triples.jsonl")
    node_support, edge_support = defaultdict(set), defaultdict(set); observations_valid = True
    for item in observations:
        cid, pair = item["case_id"], []
        for key in ["head", "tail"]:
            endpoint = item[key]; meaning = (endpoint["type"], endpoint["text"]); pair.append(meaning)
            start, end = endpoint.get("start"), endpoint.get("end")
            valid = (isinstance(start, int) and not isinstance(start, bool) and isinstance(end, int) and not isinstance(end, bool)
                     and cid in case_text and 0 <= start < end <= len(case_text[cid]) and case_text[cid][start:end] == endpoint["text"])
            observations_valid &= valid and meaning in semantic
            if meaning in semantic: node_support[semantic[meaning]].add(cid)
        edge_support[pair[0], item["relation"], pair[1]].add(cid)
        observations_valid &= item.get("causal_gold") is False
    source_valid = observations_valid and len(nodes)==len(source["nodes"])==len(semantic) and len(edges)==len(source["edges"])
    source_valid &= len(observations)==threshold["thresholds"]["0.0"]["n_retained_occurrence_pairs"]
    for nid,node in nodes.items(): source_valid &= bool(node_support[nid]) and set(node["case_ids"])==node_support[nid]<=case_ids
    edge_keys = set()
    for edge in edges.values():
        head,tail = nodes[edge["source"]],nodes[edge["target"]]
        key = ((head["entity_type"],head["name"]),edge["relation_type"],(tail["entity_type"],tail["name"])); edge_keys.add(key)
        support = set(edge["case_ids"])
        source_valid &= bool(support) and support==edge_support[key] and support<=set(head["case_ids"]) & set(tail["case_ids"])
    check("actual_saved_occurrence_offsets_and_exact_source_graph_support", source_valid and edge_keys==set(edge_support))
    basic, enhanced, summary = [load(folder / name) for name in ["base_graph.json", "enhanced_graph.json", "analysis_summary.json"]]
    config = load(folder / "run_config.json")
    check("model_occurrences_mode_and_actual_graph_input", all(d.get("support_mode")=="model_occurrences" for d in [basic,enhanced,summary]) and
          config["arguments"]["support_mode"]=="model_occurrences" and config["input_sha256"]["cases"]==sha(cases_file) and
          config["input_sha256"]["export"]==sha(threshold_dir/"threshold0_qualified_predicted_graph.json") and
          Path(config["arguments"]["export_json"]).resolve()==(threshold_dir/"threshold0_qualified_predicted_graph.json").resolve())
    base_nodes = {n["id"]: n for n in basic["nodes"]}; base_edges = {e["id"]: e for e in basic["edges"]}
    preserved = set(base_nodes)==set(nodes) and set(base_edges)==set(edges)
    for nid,node in base_nodes.items():
        raw = nodes.get(nid,{})
        preserved &= (node.get("name"),node.get("entity_type"))==(raw.get("name"),raw.get("entity_type"))
        preserved &= set(node.get("case_support",[]))==set(raw.get("case_ids",[]))
    for eid,edge in base_edges.items():
        raw = edges.get(eid,{})
        preserved &= (edge.get("source"),edge.get("target"),edge.get("relation_type"))==(raw.get("source"),raw.get("target"),raw.get("relation_type"))
        preserved &= set(edge.get("case_support",[]))==set(raw.get("case_ids",[]))
    check("base_case_support_not_expanded_to_same_words_in_other_cases", preserved)
    valid_analysis = True; eligible = {eid for eid,e in base_edges.items() if e["relation_type"] in RELATIONS}
    for e in basic["analysis_edges"]:
        basis = e.get("basis_base_edge_ids",[])
        if len(basis)!=1 or basis[0] not in eligible: valid_analysis=False; continue
        raw = base_edges[basis[0]]; kind,reverse = RELATIONS[raw["relation_type"]]
        u,v = (raw["target"],raw["source"]) if reverse else (raw["source"],raw["target"])
        physical = {"故障部位","部件","组件","系统"}
        if kind=="spatial_context" and base_nodes[v]["entity_type"] in physical and base_nodes[u]["entity_type"] not in physical: u,v=v,u
        valid_analysis &= (e["source"],e["target"],e["kind"])==(u,v,kind) and set(e["case_support"])==set(raw["case_support"]) and e.get("causal_gold") is False
    valid_analysis &= {e["id"] for e in basic["analysis_edges"]}=={"analysis_"+eid for eid in eligible}
    check("metadata_not_promoted_and_original_relation_mapping_unchanged",valid_analysis)
    augmented_nodes = {n["id"]:n for n in enhanced["nodes"]}
    augmentation = enhanced["basic_edges"]==basic["edges"] and set(base_nodes)<=set(augmented_nodes)
    for nid,node in augmented_nodes.items():
        if nid in base_nodes:
            augmentation &= node==base_nodes[nid]; continue
        mechanism=nid.removeprefix("mechanism_")
        expected={i for i,n in base_nodes.items() if n["entity_type"] in FAULT_TYPES and any(term in n["name"] for term in MECHANISMS.get(mechanism,[]))}
        support=set().union(*(set(base_nodes[i]["case_support"]) for i in expected))
        augmentation &= nid.startswith("mechanism_") and mechanism in MECHANISMS and node.get("layer")=="mechanism" and bool(expected)
        augmentation &= set(node.get("member_entity_ids",[]))==expected and set(node.get("case_support",[]))==support
    expected_augmented_ids={e["id"] for e in basic["analysis_edges"]}|{e["id"] for e in enhanced["mechanism_edges"]}
    augmentation &= len(enhanced["analysis_edges"])==len(expected_augmented_ids) and {e["id"] for e in enhanced["analysis_edges"]}==expected_augmented_ids
    for e in enhanced["mechanism_edges"]:
        basis = set(e.get("basis_base_edge_ids",[])); support = set(e.get("case_support",[]))
        allowed = set().union(*(set(base_edges[b]["case_support"]) for b in basis if b in base_edges))
        augmentation &= bool(basis) and basis<=set(base_edges) and e.get("causal_gold") is False
        augmentation &= support<=allowed & set(augmented_nodes[e["source"]]["case_support"]) & set(augmented_nodes[e["target"]]["case_support"])
        if e["kind"]!="mechanism_membership_context": augmentation &= basis<=eligible
        if e["kind"]=="mechanism_chain_inherited_candidate":
            augmentation &= all(RELATIONS[base_edges[b]["relation_type"]][0] in {"causal_candidate","functional_influence"} for b in basis if b in eligible)
    check("mechanism_edges_inherit_only_actual_basis_case_support",augmentation)
    topic_valid = True
    for name,path_key,hash_key in [("document_topic.csv","document_topic","document_topic_sha256"),
                                   ("case_system_probabilities.csv","case_system_probabilities","case_system_sha256")]:
        path = topics_dir/"paper_fixed_K07"/name
        topic_valid &= Path(summary["topic_sources"][path_key]).resolve()==path.resolve() and summary["topic_sources"][hash_key]==sha(path)
    check("actual_source_NER_dictionary_K7_topic_hashes",topic_valid)
    strict_counts = {}
    for label,data in [("base",basic),("enhanced",enhanced)]:
        layer_nodes = {n["id"]:n for n in data["nodes"]}; pair_edges,pairs_by_case = defaultdict(list),defaultdict(set)
        for e in data["analysis_edges"]:
            pair_edges[e["source"],e["target"]].append(e)
            for cid in e["case_support"]: pairs_by_case[cid].add((e["source"],e["target"]))
        paths = rows(folder/f"{label}_case_grounded_paths.jsonl"); actual,strict = defaultdict(set),defaultdict(set); path_valid=True
        for p in paths:
            ids,support = tuple(p["node_ids"]),set(p["case_ids"])
            path_valid &= len(ids)>=2 and len(ids)==len(set(ids)) and support<=case_ids and len(support)==p["literal_case_frequency"]==p.get("model_qualified_case_frequency")
            for cid in support:
                actual[cid].add(ids); path_valid &= all((u,v) in pairs_by_case[cid] for u,v in zip(ids,ids[1:]))
                if _candidate_in_case(ids,layer_nodes,pair_edges,cid): strict[ids].add(cid)
        completion = csv_rows(folder/f"{label}_case_path_completion.csv"); by_id={r["case_id"]:r for r in completion}; cache={}; total=0
        complete = len(completion)==len(cases) and set(by_id)==case_ids
        for cid in case_ids:
            signature=tuple(sorted(pairs_by_case[cid]))
            if signature not in cache: cache[signature]=_simple_paths(signature)
            expected=cache[signature]; total+=len(expected); item=by_id.get(cid,{})
            complete &= actual[cid]==expected and item.get("complete")=="True" and item.get("n_simple_paths")==str(len(expected)) and item.get("n_grounded_edges")==str(len(signature))
        stats=summary["case_path_statistics"]["before" if label=="base" else "after"]
        complete &= stats["n_cases_complete"]==len(cases) and stats["n_cases_incomplete"]==0 and stats["hop_cap"] is None and stats["path_case_incidences"]==total
        complete &= len(paths)==len({tuple(p["node_ids"]) for p in paths})==stats["n_unique_case_grounded_simple_paths"]
        candidates=rows(folder/f"{label}_propagation_candidate_paths.jsonl"); strict_counts[label]=len(candidates)
        check(label+"_all_numbered_cases_uncapped_path_enumeration",complete and path_valid,{"cases":len(cases),"independent_path_case_incidences":total})
        check(label+"_strict_paths_case_specific_influence",len(candidates)==len(strict) and {tuple(p["node_ids"]):set(p["case_ids"]) for p in candidates}==dict(strict))
    check("no_strict_paths_stay_explicit_not_evaluable",bool(strict_counts["enhanced"]) or summary.get("propagation_evaluation_status")=="not_evaluable_no_qualified_propagation_candidates")
    downstream=True
    for subdir in ["patterns","hub_hierarchy"]:
        item=load(folder/subdir/"analysis_summary.json"); downstream &= item.get("support_mode")=="model_occurrences" and item["input_sha256"]["cases"]==sha(cases_file)
        for label in ["base","enhanced"]: downstream &= item["input_sha256"][label+"_graph"]==sha(folder/f"{label}_graph.json")
    check("pattern_and_hub_input_hash_binding",downstream)
    modes=load(folder/"graph_modes_verification.json")
    check("pattern_offsets_topology_hub_and_scope_verifier_passed",modes.get("status")=="passed" and all(modes.get("checks",{}).values()))
    engineering=load(folder/"engineering/analysis_summary.json"); eng_config=load(folder/"engineering/run_config.json"); scope=engineering.get("source_scope",{})
    expected_scope="model_qualified_occurrence_graph_retrospective_symptom_query_replay"
    check("engineering_model_candidate_source_scope_not_historical_or_gold",scope.get("support_mode")=="model_occurrences" and
          scope.get("graph_source_kind")=="model_qualified_occurrence_relation_candidates" and scope.get("evaluation_scope")==expected_scope==engineering["retrospective_query_comparison"]["evaluation_scope"] and
          scope.get("graph_source_is_historical_export") is False and scope.get("case_support_is_observed_causal_gold") is False and eng_config.get("source_scope")==scope)
    bound=all(Path(eng_config["graph_inputs"][l]["path"]).resolve()==(folder/f"{l}_graph.json").resolve() and eng_config["graph_inputs"][l]["sha256"]==sha(folder/f"{l}_graph.json") for l in ["base","enhanced"])
    queries=rows(folder/"engineering/case_queries_and_inference.jsonl")
    check("engineering_all_case_rows_and_fresh_input_hashes",bound and len(queries)==len(cases) and {r["case_id"] for r in queries}==case_ids)
    return {"status":"passed" if all(c["passed"] for c in checks) else "failed","role":ROLE,"checks":checks,
            "cases":len(cases),"n_observations":len(observations),"strict_path_counts":strict_counts,
            "causal_gold_validated":False,"relation_precision":None,"relationship_existence_accuracy":None}


def execute(threshold_dir, topics_dir, cases, output=None):
    threshold_dir,topics_dir,cases=map(lambda p:Path(p).resolve(),(threshold_dir,topics_dir,cases))
    output=Path(output).resolve() if output else threshold_dir/"argmax_graph_analysis"
    summary,configuration,source,corpus=check_threshold_ready(threshold_dir,topics_dir,cases)
    output.mkdir(parents=True,exist_ok=True); log_dir=output/"execution_logs"; log_dir.mkdir(exist_ok=True)
    primary_dir=Path(configuration["inputs"]["source_RE_run_summary"]["path"]).parent
    inputs=[threshold_dir/name for name in ["analysis_summary.json","run_config.json","threshold0_qualified_predicted_graph.json","threshold0_qualified_triples.jsonl"]]
    inputs += [cases,topics_dir/"analysis_summary.json",topics_dir/"paper_fixed_K07/document_topic.csv",topics_dir/"paper_fixed_K07/case_system_probabilities.csv"]
    inputs += [Path(ref["path"]) for ref in configuration["inputs"].values()]
    primary=[primary_dir/name for name in ["run_summary.json","qualified_predicted_graph.json","qualified_triples.jsonl","case_counts_and_rejections.jsonl"] if (primary_dir/name).is_file()]
    before=file_refs(set(inputs+primary)); record={"status":"running","role":ROLE,"command":sys.argv,"started_unix":time.time(),
        "input_sha256":before,"main_0_8_artifacts_before_sha256":file_refs(primary),"commands":[],"historical_export_inserted":False,
        "relation_labels_changed":False,"new_gold_annotation_performed":False,"support_mode":"model_occurrences","implementation_sha256":dependencies(__file__)}
    record_path=output/"execution.json"; write(record_path,record)
    if not source["edges"]:
        for name in ["base_graph.json","enhanced_graph.json"]:
            path=output/name
            if path.exists():
                prior=load(path)
                if any(prior.get(k) for k in ["nodes","edges","basic_edges","analysis_edges"]):
                    record.update(status="failed",error="Empty input has stale nonempty graph output: "+str(path)); write(record_path,record)
                    raise ValueError(record["error"])
        empty={"status":"not_evaluable_empty_qualified_model_graph","qualified_graph_nodes":0,"qualified_graph_edges":0,
               "historical_export_inserted":False,"role":ROLE,"n_original_cases":len(corpus),
               "upstream_threshold_summary":str(threshold_dir/"analysis_summary.json"),"upstream_threshold_summary_sha256":sha(threshold_dir/"analysis_summary.json")}
        if any(not Path(p).is_file() or sha(p)!=h for p,h in before.items()):
            record.update(status="failed",error="Input/main-0.8 artifact changed during empty-graph assessment"); write(record_path,record)
            raise ValueError(record["error"])
        write(output/"analysis_summary.json",empty); record.update(status=empty["status"],elapsed_seconds=time.time()-record["started_unix"],commands=[],
            main_0_8_artifacts_unchanged=True,main_0_8_artifacts_after_sha256=file_refs(primary),
            output_artifacts_sha256=file_refs([output/"analysis_summary.json"]))
        write(record_path,record); return record
    stages=[
        ("graph",'diesel_kg.graph.analysis',["--topics-dir",topics_dir,"--export-json",threshold_dir/"threshold0_qualified_predicted_graph.json","--support-mode","model_occurrences","--cases",cases,"--output",output],
         [output/p for p in ["analysis_summary.json","run_config.json","base_graph.json","enhanced_graph.json","base_case_path_completion.csv","enhanced_case_path_completion.csv"]]),
        ("patterns",'diesel_kg.graph.patterns',["--graph-dir",output,"--cases",cases,"--output",output/"patterns"],[output/"patterns/analysis_summary.json"]),
        ("hub_hierarchy",'diesel_kg.graph.hubs',["--graph-dir",output,"--cases",cases,"--output",output/"hub_hierarchy"],[output/"hub_hierarchy/analysis_summary.json"]),
        ("engineering",'diesel_kg.evaluation.engineering',["--graph-dir",output,"--cases",cases,"--output",output/"engineering","--repeats",3],[output/"engineering/analysis_summary.json",output/"engineering/run_config.json",output/"engineering/case_queries_and_inference.jsonl"]),
        ("graph_modes_verification",'diesel_kg.validation.graph_modes',["--graph-dir",output,"--cases",cases,"--patterns-dir",output/"patterns","--hub-dir",output/"hub_hierarchy","--output",output/"graph_modes_verification.json"],[output/"graph_modes_verification.json"])]
    try:
        for name,script,args,required in stages:
            command=[sys.executable,str(resolve_script(script, HERE)),*map(str,args)]; log=log_dir/(name+".log")
            stage={"stage":name,"command":command,"cwd":str(HERE),"status":"running","started_unix":time.time(),"log":str(log),"local_code_sha256":dependencies(script)}
            record["commands"].append(stage); write(record_path,record); print(name+": executing; log="+str(log),flush=True)
            with log.open("w") as handle:
                result=subprocess.run(command,cwd=HERE,stdout=handle,stderr=subprocess.STDOUT,env={**os.environ,"TOKENIZERS_PARALLELISM":"false"})
            stage.update(exit_code=result.returncode,elapsed_seconds=time.time()-stage["started_unix"],missing_artifacts=[str(p) for p in required if not p.is_file()])
            stage["artifacts_sha256"]=file_refs(p for p in required if p.is_file())
            stage["status"]="completed" if result.returncode==0 and not stage["missing_artifacts"] else "failed"; write(record_path,record)
            if stage["status"]!="completed": raise RuntimeError(name+" failed; inspect "+str(log))
            if any(sha(p)!=h for p,h in stage["local_code_sha256"].items()): raise ValueError("Code changed during stage: "+name)
            print(name+": completed in "+str(round(stage["elapsed_seconds"],2))+" s",flush=True)
        verification=verify_argmax_graph(threshold_dir,topics_dir,cases,output)
        write(output/"argmax_evidence_verification.json",verification)
        if verification["status"]!="passed": raise ValueError("Independent argmax graph evidence check failed")
        if any(not Path(p).is_file() or sha(p)!=h for p,h in before.items()): raise ValueError("Input/main-0.8 artifact changed during independent execution")
        record.update(status="completed_candidate_computation_not_causal_validation",elapsed_seconds=time.time()-record["started_unix"],
            main_0_8_artifacts_unchanged=True,main_0_8_artifacts_after_sha256=file_refs(primary),verification_sha256=sha(output/"argmax_evidence_verification.json"))
        record["output_artifacts_sha256"]=file_refs(p for p in output.rglob("*") if p.is_file() and p!=record_path)
        write(record_path,record); return record
    except Exception as exc:
        record.update(status="failed",error=str(exc),elapsed_seconds=time.time()-record["started_unix"]); write(record_path,record); raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ["threshold-dir","topics-dir","cases"]: parser.add_argument("--"+name,type=Path,required=True)
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(); result=execute(args.threshold_dir,args.topics_dir,args.cases,args.output)
    print(json.dumps({"status":result["status"],"role":ROLE,"stages":[c["stage"] for c in result["commands"]]},ensure_ascii=False))


if __name__=="__main__":main()

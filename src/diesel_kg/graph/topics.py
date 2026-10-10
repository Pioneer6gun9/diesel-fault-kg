#!/usr/bin/env python3
"""Author-protocol LDA reconstruction with explicit c_v and system-mapping assumptions.

The available manuscript names LDA and topic coherence but does not specify the
coherence implementation, tokenizer, priors, seed, or training parameters. This
module fills those missing choices transparently; it never tunes toward a
desired topic count/coherence and never interprets engineering mappings as gold.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import gensim
import jieba
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from gensim.corpora import Dictionary, MmCorpus
from gensim.models import CoherenceModel, LdaModel


SYSTEMS = {
    "fuel_combustion": {
        "label": "Fuel injection / combustion",
        "terms": "喷油器 喷油嘴 喷油泵 喷油 高压油泵 高压共轨 共轨 油轨 供油泵 输油泵 燃油泵 燃油滤清器 柴油滤清器 柴油滤芯 燃油滤芯 油水分离器 燃油 燃烧 雾化 喷射 针阀 燃油压力 喷油正时 供油 燃油管 高压油管 回油管 预燃室 点火 着火 失火 燃油计量阀".split(),
    },
    "hydraulic": {
        "label": "Hydraulic",
        "terms": "液压 液压泵 液压油 液压缸 液压阀 液压管 液压系统 液压油箱 液压油滤芯 液压马达 溢流阀 换向阀 比例阀 柱塞泵 齿轮泵 吸油管 液力".split(),
    },
    "cooling_heat": {
        "label": "Cooling / heat dissipation",
        "terms": "冷却 冷却液 冷却水 冷却系统 冷却风扇 散热器 散热 风扇 水泵 节温器 水箱 水温 冷却水管 水套 防冻液 热交换器 油冷器 过热 高温 水垢 水管 冷却液温度".split(),
    },
    "electrical_control": {
        "label": "Electrical / control",
        "terms": "电气 电路 电线 线束 接线 继电器 蓄电池 电池 发电机 起动机 启动电机 启动机 ECU 控制器 电控 传感器 电磁阀 电源 电压 电流 短路 断路 接地 接触不良 插接件 接插件 保险丝 保险 电子 油门踏板".split(),
    },
    "mechanical_sealing": {
        "label": "Mechanical wear / sealing",
        "terms": "磨损 密封 油封 密封圈 活塞 活塞环 缸套 气缸 缸体 气缸盖 缸盖 缸垫 气缸垫 曲轴 轴瓦 连杆 轴承 凸轮 凸轮轴 气门 气门导管 气门座 气门间隙 摇臂 弹簧 螺栓 螺母 曲轴箱 机油泵 润滑 机油 机油滤清器 机油滤芯 齿轮 活塞销 配气 正时齿轮 疲劳 裂纹 拉缸 抱瓦".split(),
    },
    "intake_exhaust": {
        "label": "Intake / exhaust",
        "terms": "进气 排气 增压 涡轮增压器 增压器 涡轮 中冷器 空气滤清器 空气滤芯 空滤 进气歧管 排气歧管 进气管 排气管 废气 EGR DPF SCR DOC 后处理 尿素 催化 三元 颗粒捕集器 排烟 黑烟 蓝烟 白烟 进气压力 排气背压".split(),
    },
    "transmission_steering": {
        "label": "Transmission / steering",
        "terms": "传动 转向 变速箱 变速器 离合器 传动轴 转向泵 转向器 方向盘 转向油 转向助力 制动 刹车 差速器 驱动桥 减速器 扭矩转换器 液力变矩器 齿轮箱 换挡 换档 挂挡 挂档".split(),
    },
}
EXTRA_COMPONENTS = "柴油机 发动机 柴油发动机 潍柴 玉柴 康明斯 动力不足 怠速不稳 无法启动 启动困难 机油压力 燃油压力 水温过高 气门间隙 温度传感器 压力传感器 转速传感器 喷油器针阀 曲轴位置传感器 增压压力 燃油计量单元 机油消耗 排气冒烟 冷却液泄漏 油压不足 压缩压力 爆震 异响 振动 漏油 漏水 漏气 堵塞 腐蚀 卡滞 松动 老化 泄漏 渗漏 故障报警 故障码 油耗增加 烧机油".split()
FIXED_STOPWORDS = set("的 了 和 与 及 或 在 是 为 有 无 将 对 被 把 由 于 中 其 该 此 这 那 并 且 而 等 后 前 时 上 下 内 外 之间 进行 通过 发现 检查 检测 更换 更换后 维修 修理 处理 解决 排除 故障 恢复 正常 恢复正常 重新 及时 立即 随后 最后 之后 之前 导致 造成 因为 因此 从而 出现 存在 使用 采用 需要 由于 发生 案例 柴油机 发动机 柴油发动机 一台 一辆 一部 一次 一个 一种 部分 其中 相关 情况 现象 原因 结果 问题 表现 工作 运行 运转 试车 分析 确认 进一步 后来".split())
FIXED_STOPWORDS.update("标准 拆检 调整 达到 降至 升至 清除 排查 复查 复至".split())
CASE_PREFIX = re.compile(r"^\s*(?:案例|case)\s*[#＃]?\s*(\d+)\s*(?:[：:、.．)）-]\s*|\s+)(.*)$", re.I | re.S)
NUMBER_PREFIX = re.compile(r"^\s*(\d+)\s*[：:、.．)）]\s*(.*)$", re.S)
BAD_ENTITY = re.compile(r"[\s，,。；;：:！!？?（）()\[\]{}]|导致|造成|因为|从而|但是|更换|检查|发现|恢复|调整|达到|降至|升至|复至|排除|清除|拆检|(?:至|为|后)$")
SYSTEM_KEYS = list(SYSTEMS)


@dataclass(frozen=True)
class Settings:
    seed: int = 42
    topic_min: int = 3
    topic_max: int = 12
    fixed_paper_k: int = 7
    passes: int = 20
    iterations: int = 200
    min_df: int = 2
    coherence_topn: int = 10
    keyword_topn: int = 20
    cv_window_size: int = 110
    mapping_min_top_affinity: float = 0.45
    mapping_min_margin: float = 0.15


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: expected a JSON object")
            rows.append(row)
    return rows


def clean_case_text(text: str) -> tuple[int | None, str]:
    """Only remove a leading case enumeration, preserving the body and duplicates."""
    text = str(text).strip()
    match = CASE_PREFIX.match(text) or NUMBER_PREFIX.match(text)
    return (int(match.group(1)), match.group(2).strip()) if match else (None, text)


def flatten_entities(value):
    if isinstance(value, list):
        for item in value:
            yield from flatten_entities(item)
    elif isinstance(value, dict):
        if "entities" in value:
            yield from flatten_entities(value["entities"])
        elif isinstance(value.get("text"), str):
            yield value


def professional_entity_rejection_reason(text: str, base_frequencies: dict | None = None):
    if not 2 <= len(text) <= 15:
        return "outside_2_to_15_characters"
    if BAD_ENTITY.search(text):
        return "punctuation_or_clause_or_maintenance_action"
    if text.isnumeric() or not re.search(r"[\u4e00-\u9fffA-Za-z]", text):
        return "numeric_or_nonlexical"
    if text in FIXED_STOPWORDS:
        return "fixed_stopword"
    anchors = {w for v in SYSTEMS.values() for w in v["terms"]} | set(EXTRA_COMPONENTS)
    has_anchor = any(w.lower() in text.lower() for w in anchors)
    if re.search(r"[\u4e00-\u9fff]", text) and re.search(r"[A-Za-z0-9]", text) and not has_anchor:
        return "mixed_script_fragment_without_engineering_anchor"
    if base_frequencies is not None and re.fullmatch(r"[\u4e00-\u9fff]+", text) and not has_anchor and base_frequencies.get(text, 0) == 0:
        return "unrecognized_span_without_engineering_anchor"
    return None


def accepted_professional_entity(text: str) -> bool:
    return professional_entity_rejection_reason(text) is None


def read_cases(cases_path: Path, entity_rows: list[dict], sample_size: int | None = None):
    by_id = {str(r.get("case_id")): r for r in entity_rows if "case_id" in r}
    result = []
    for index, row in enumerate(read_jsonl(cases_path), 1):
        case_id = str(row.get("case_id", f"case_{index:05d}"))
        original_number, text = clean_case_text(row.get("text", ""))
        if not text:
            raise ValueError(f"Empty body in case {case_id}")
        ner_number, ner_text = clean_case_text(by_id.get(case_id, {}).get("text", ""))
        if original_number is not None:
            number_source = "case_text_prefix"
        elif ner_number is not None:
            original_number, number_source = ner_number, "aligned_NER_text_prefix"
        else:
            suffix = re.search(r"(\d+)$", case_id)
            original_number, number_source = (int(suffix.group(1)), "case_id_numeric_suffix") if suffix else (index, "input_order_assumption")
        result.append({"case_id": case_id, "original_case_number": original_number,
                       "number_source": number_source, "text": text,
                       "ner_text_matches_body": bool(ner_text == text) if ner_text else None})
    if len({r["case_id"] for r in result}) != len(result):
        raise ValueError("Duplicate case identifiers; distinct numbered records must have distinct identifiers")
    return result[:sample_size] if sample_size is not None else result


def vocabulary_terms(entity_rows: list[dict]):
    reference_tokenizer = jieba.Tokenizer()
    reference_tokenizer.initialize()
    terms = {w for v in SYSTEMS.values() for w in v["terms"]} | set(EXTRA_COMPONENTS)
    reasons = {w: {"fixed_engineering_dictionary"} for w in terms}
    rejected = Counter()
    accepted_ner = Counter()
    for row in entity_rows:
        for entity in flatten_entities(row.get("entities", row if "entities" not in row else [])):
            text = str(entity.get("text", "")).strip()
            rejection = professional_entity_rejection_reason(text, reference_tokenizer.FREQ)
            if rejection is None:
                terms.add(text)
                reasons.setdefault(text, set()).add("provided_NER_entity")
                accepted_ner[text] += 1
            else:
                rejected[(text, rejection)] += 1
    return terms, reasons, accepted_ner, rejected


def tokenize_cases(cases: list[dict], terms: set[str]):
    tokenizer = jieba.Tokenizer()
    tokenizer.initialize()
    for word in sorted(terms):
        tokenizer.add_word(word, freq=100000)
    documents = []
    for case in cases:
        tokens = []
        for word in tokenizer.cut(case["text"], HMM=False):
            word = word.strip()
            if (len(word) >= 2 and word not in FIXED_STOPWORDS and not word.isnumeric()
                    and re.fullmatch(r"[\u4e00-\u9fffA-Za-z0-9_+-]+", word)):
                tokens.append(word)
        documents.append(tokens)
    return documents


def write_csv(path: Path, rows: list[dict], fields: list[str] | None = None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def topic_keywords(model: LdaModel, topn: int) -> list[list[tuple[str, float]]]:
    return [[(word, float(weight)) for word, weight in model.show_topic(i, topn)]
            for i in range(model.num_topics)]


def system_mapping(keywords: list[list[tuple[str, float]]], settings: Settings):
    affinities = np.zeros((len(keywords), len(SYSTEM_KEYS)), dtype=float)
    details = []
    for topic_index, words in enumerate(keywords):
        scores = np.zeros(len(SYSTEM_KEYS), dtype=float)
        matches = []
        for word, probability in words:
            for system_index, key in enumerate(SYSTEM_KEYS):
                anchors = [anchor for anchor in SYSTEMS[key]["terms"] if anchor.lower() in word.lower()]
                if anchors:
                    # One keyword contributes once per system; avoid double-counting nested anchors.
                    scores[system_index] += probability
                    matches.append({"topic_id": f"T{topic_index + 1:02d}", "keyword": word,
                                    "keyword_probability": probability, "system": key,
                                    "matched_anchors": "|".join(sorted(anchors))})
        total = float(scores.sum())
        if total > 0:
            affinities[topic_index] = scores / total
            order = np.argsort(-affinities[topic_index], kind="stable")
            top = float(affinities[topic_index, order[0]])
            margin = top - float(affinities[topic_index, order[1]])
            decision = "ambiguous" if top < settings.mapping_min_top_affinity or margin < settings.mapping_min_margin else "mapped"
            label = SYSTEM_KEYS[int(order[0])] if decision == "mapped" else "ambiguous"
        else:
            top, margin, decision, label = 0.0, 0.0, "unmapped", "unmapped"
        detail = {"topic_id": f"T{topic_index + 1:02d}", "mapping_decision": decision,
                  "assigned_system": label, "matched_keyword_mass": total,
                  "top_affinity": top, "affinity_margin": margin}
        detail.update({"affinity_" + key: float(affinities[topic_index, i]) for i, key in enumerate(SYSTEM_KEYS)})
        detail.update({"raw_score_" + key: float(scores[i]) for i, key in enumerate(SYSTEM_KEYS)})
        details.append((detail, matches))
    return affinities, [x[0] for x in details], [m for x in details for m in x[1]]


def document_topic_matrix(model: LdaModel, corpus) -> np.ndarray:
    matrix = np.zeros((len(corpus), model.num_topics), dtype=float)
    for row_index, bow in enumerate(corpus):
        for topic_index, probability in model.get_document_topics(bow, minimum_probability=0):
            matrix[row_index, topic_index] = probability
    sums = matrix.sum(axis=1, keepdims=True)
    if not np.isfinite(matrix).all() or np.any(sums <= 0):
        raise ValueError("Invalid document-topic probabilities")
    return matrix / sums


def fit_lda(k: int, dictionary: Dictionary, corpus, texts, settings: Settings):
    start = time.perf_counter()
    model = LdaModel(corpus=corpus, id2word=dictionary, num_topics=k,
                     random_state=settings.seed, passes=settings.passes,
                     iterations=settings.iterations, chunksize=2000,
                     update_every=0, alpha="symmetric", eta=None,
                     minimum_probability=0.0, eval_every=None, dtype=np.float32)
    trained_seconds = time.perf_counter() - start
    cv_start = time.perf_counter()
    coherence = CoherenceModel(model=model, texts=texts, dictionary=dictionary,
                               coherence="c_v", topn=min(settings.coherence_topn, len(dictionary)),
                               window_size=settings.cv_window_size, processes=1)
    per_topic = [float(x) for x in coherence.get_coherence_per_topic()]
    cv = float(np.mean(per_topic))
    if not math.isfinite(cv) or not all(math.isfinite(x) for x in per_topic):
        raise ValueError(f"Nonfinite c_v at K={k}; data/dictionary must be inspected")
    return model, {"k": k, "coherence_c_v": cv,
                   "lda_training_seconds": trained_seconds,
                   "coherence_seconds": time.perf_counter() - cv_start}, per_topic


def export_model_results(output: Path, model: LdaModel, corpus, cases, settings: Settings,
                         per_topic_cv: list[float], serialize: bool = True,
                         frozen_document_topics: np.ndarray | None = None):
    output.mkdir(parents=True, exist_ok=True)
    keywords = topic_keywords(model, min(settings.keyword_topn, len(model.id2word)))
    topic_rows = [{"topic_id": f"T{i + 1:02d}", "rank": rank, "word": word,
                   "probability": probability, "topic_coherence_c_v": per_topic_cv[i]}
                  for i, words in enumerate(keywords) for rank, (word, probability) in enumerate(words, 1)]
    write_csv(output / "top_words.csv", topic_rows)
    write_json(output / "topic_keywords.json", {
        "k": model.num_topics,
        "topics": [{"topic_id": f"T{i+1:02d}", "topic_index": i,
                    "coherence_c_v": per_topic_cv[i],
                    "keywords": [{"word": word, "probability": probability} for word, probability in words]}
                   for i, words in enumerate(keywords)],
    })
    # Gensim initializes inference gamma randomly. Reuse the one computed
    # posterior matrix when exporting the same fitted model under two labels.
    matrix = document_topic_matrix(model, corpus) if frozen_document_topics is None else frozen_document_topics
    doc_rows = []
    for case, distribution in zip(cases, matrix):
        row = {"case_id": case["case_id"], "original_case_number": case["original_case_number"]}
        row.update({f"T{i + 1:02d}": float(x) for i, x in enumerate(distribution)})
        doc_rows.append(row)
    write_csv(output / "document_topic.csv", doc_rows)
    if serialize:
        model.save(str(output / "lda.model"), separately=[])
    return keywords, matrix


def figure_font():
    available = {font.name for font in font_manager.fontManager.ttflist}
    for name in ["Arial Unicode MS", "PingFang SC", "Heiti TC", "Noto Sans CJK SC"]:
        if name in available:
            plt.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
            break
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["svg.fonttype"] = "none"


def save_figure(fig, stem: Path):
    fig.savefig(stem.with_suffix(".png"), dpi=200, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def export_system_results(output: Path, keywords, matrix, cases, settings: Settings, figures: bool):
    affinity, mappings, matches = system_mapping(keywords, settings)
    write_csv(output / "topic_system_affinities.csv", mappings)
    write_json(output / "topic_system_affinities.json", {
        "system_order": SYSTEM_KEYS,
        "topic_order": [r["topic_id"] for r in mappings],
        "affinity_matrix": affinity.tolist(),
        "mapping_decisions": [r["mapping_decision"] for r in mappings],
        "assigned_systems": [r["assigned_system"] for r in mappings],
        "mapping_is_gold": False,
    })
    write_csv(output / "engineering_keyword_matches.csv", matches,
              ["topic_id", "keyword", "keyword_probability", "system", "matched_anchors"])
    unmapped = np.array([r["mapping_decision"] == "unmapped" for r in mappings], dtype=float)
    ambiguous = np.array([r["mapping_decision"] == "ambiguous" for r in mappings], dtype=float)
    probabilities = matrix @ affinity
    unmapped_mass = matrix @ unmapped
    ambiguity_mass = matrix @ ambiguous
    if not np.allclose(probabilities.sum(axis=1) + unmapped_mass, 1.0, atol=1e-6):
        raise AssertionError("Seven-system plus unmapped probability mass must sum to one")
    rows = []
    for i, case in enumerate(cases):
        row = {"case_id": case["case_id"], "original_case_number": case["original_case_number"]}
        row.update({key: float(probabilities[i, j]) for j, key in enumerate(SYSTEM_KEYS)})
        row.update({"unmapped_probability": float(unmapped_mass[i]),
                    "ambiguity_topic_mass": float(ambiguity_mass[i])})
        rows.append(row)
    write_csv(output / "case_system_probabilities.csv", rows)
    write_json(output / "mapping_summary.json", {
        "mapping_is_gold": False,
        "decisions": dict(Counter(r["mapping_decision"] for r in mappings)),
        "case_unmapped_mass_mean": float(unmapped_mass.mean()),
        "case_ambiguity_topic_mass_mean": float(ambiguity_mass.mean()),
        "system_probability_mass_mean": {key: float(probabilities[:, j].mean()) for j, key in enumerate(SYSTEM_KEYS)},
        "case_probability_rule": "doc_topic @ normalized engineering keyword affinities; unmapped topic mass is retained separately; ambiguous topics keep their soft affinities",
    })
    if figures:
        fig, ax = plt.subplots(figsize=(11.4, max(4.1, 0.44 * len(keywords))))
        img = ax.imshow(affinity, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_xticks(range(7), [SYSTEMS[k]["label"].replace(" / ", " /\n") for k in SYSTEM_KEYS], rotation=28, ha="right", fontsize=9)
        ax.set_yticks(range(len(keywords)), [r["topic_id"] + " (" + r["mapping_decision"] + ")" for r in mappings])
        ax.set_title(f"Engineering keyword affinities: K={len(keywords)}\nSoft mapping assumption; no gold subsystem labels")
        for i in range(len(keywords)):
            for j in range(7):
                ax.text(j, i, f"{affinity[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if affinity[i, j] > 0.55 else "black")
        fig.colorbar(img, ax=ax, label="Normalized keyword affinity")
        fig.tight_layout()
        save_figure(fig, output / "topic_system_affinity")
        fig, ax = plt.subplots(figsize=(11.4, 4.3))
        means = list(probabilities.mean(axis=0)) + [float(unmapped_mass.mean())]
        ax.bar(range(8), means, color=["#315b83"] * 7 + ["#999999"])
        ax.set_xticks(range(8), [SYSTEMS[k]["label"].replace(" / ", " /\n") for k in SYSTEM_KEYS] + ["Unmapped"], rotation=28, ha="right", fontsize=8)
        ax.set_ylim(0, min(1.0, max(means) * 1.25 + 0.02))
        ax.set_ylabel("Mean case probability mass")
        ax.set_title(f"Case-level subsystem probability: K={len(keywords)}\nDescriptive engineering mapping; original numbered cases retained")
        for i, x in enumerate(means):
            ax.text(i, x + 0.005, f"{x:.3f}", ha="center", fontsize=8)
        fig.tight_layout()
        save_figure(fig, output / "case_system_probability")
        fig, axes = plt.subplots(len(keywords), 1, figsize=(10.3, max(5, len(keywords) * 1.35)), squeeze=False)
        for i, words in enumerate(keywords):
            top = words[:8]
            ax = axes[i, 0]
            ax.bar(range(len(top)), [p for _, p in top], color="#315b83")
            ax.set_xticks(range(len(top)), [w for w, _ in top], fontsize=9)
            ax.set_ylabel(f"T{i+1:02d}\np(word)", fontsize=9)
        fig.suptitle(f"LDA topic keywords: K={len(keywords)}", y=1.003)
        fig.tight_layout()
        save_figure(fig, output / "topic_keywords")
    return mappings


def run_analysis(cases_path: Path, entities_path: Path, output: Path, settings: Settings,
                 sample_size: int | None = None, figures: bool = True):
    start = time.perf_counter()
    if settings.topic_min < 2 or settings.topic_max < settings.topic_min or settings.min_df < 1:
        raise ValueError("Invalid topic range or min_df")
    output.mkdir(parents=True, exist_ok=True)
    entity_rows = read_jsonl(entities_path)
    cases = read_cases(cases_path, entity_rows, sample_size)
    terms, provenance, accepted_ner, rejected = vocabulary_terms(entity_rows)
    tokenized = tokenize_cases(cases, terms)
    dictionary = Dictionary(tokenized)
    before_filter = len(dictionary)
    dictionary.filter_extremes(no_below=settings.min_df, no_above=1.0, keep_n=None)
    dictionary.compactify()
    if len(dictionary) < 3:
        raise ValueError(f"Only {len(dictionary)} vocabulary items survive min_df={settings.min_df}")
    retained = set(dictionary.token2id)
    texts = [[w for w in text if w in retained] for text in tokenized]
    corpus = [dictionary.doc2bow(text) for text in texts]
    dictionary.save(str(output / "dictionary.gensim"))
    MmCorpus.serialize(str(output / "corpus.mm"), corpus)
    write_csv(output / "vocabulary.csv", [{"token_id": i, "token": dictionary[i], "document_frequency": dictionary.dfs[i],
                                          "corpus_frequency": dictionary.cfs[i],
                                          "term_source": "|".join(sorted(provenance.get(dictionary[i], {"jieba_base_dictionary"})))}
                                         for i in range(len(dictionary))])
    write_csv(output / "professional_dictionary.csv", [{"term": w, "source": "|".join(sorted(provenance[w])),
                                                        "NER_occurrences": accepted_ner[w]} for w in sorted(terms)])
    write_csv(output / "rejected_NER_dictionary_entries.csv", [{"text": t, "rejection_reason": reason, "occurrences": n} for (t, reason), n in sorted(rejected.items())])
    write_csv(output / "stopwords.csv", [{"token": x, "source": "explicit_fixed_function_and_boilerplate_words"} for x in sorted(FIXED_STOPWORDS)])
    write_csv(output / "case_metadata.csv", [{k: v for k, v in r.items() if k != "text"} for r in cases])
    with (output / "tokenized_cases.jsonl").open("w", encoding="utf-8") as handle:
        for case, tokens in zip(cases, texts):
            handle.write(json.dumps({**case, "tokens": tokens}, ensure_ascii=False) + "\n")
    write_json(output / "engineering_system_dictionary.json", SYSTEMS)
    figure_font()
    models, measures, cv_by_k, results = {}, [], {}, {}
    candidate_ks = list(range(settings.topic_min, settings.topic_max + 1))
    fixed_k = settings.fixed_paper_k
    training_order = [fixed_k] + [k for k in candidate_ks if k != fixed_k]
    decision_counts = {}
    for k in training_order:
        model, row, per_topic = fit_lda(k, dictionary, corpus, texts, settings)
        models[k], cv_by_k[k] = model, per_topic
        row["selection_candidate"] = k in candidate_ks
        measures.append(row)
        results[k] = export_model_results(output / "topic_sweep" / f"K{k:02d}", model, corpus, cases, settings, per_topic)
        write_json(output / "topic_sweep" / f"K{k:02d}" / "training_metrics.json", {
            **row, "settings": asdict(settings), "input_cases_sha256": sha256(cases_path),
            "input_entities_sha256": sha256(entities_path), "script_sha256": sha256(Path(__file__)),
        })
        print(f"K={k:02d} c_v={row['coherence_c_v']:.8f} training={row['lda_training_seconds']:.3f}s coherence={row['coherence_seconds']:.3f}s", flush=True)
        if k == fixed_k:
            label = f"paper_fixed_K{fixed_k:02d}"
            directory = output / label
            keywords, matrix = export_model_results(directory, model, corpus, cases, settings, per_topic,
                                                     serialize=False, frozen_document_topics=results[k][1])
            mappings = export_system_results(directory, keywords, matrix, cases, settings, figures)
            decision_counts[label] = dict(Counter(r["mapping_decision"] for r in mappings))
            write_json(directory / "model_reference.json", {"k": k, "model_path_relative": f"../topic_sweep/K{k:02d}/lda.model",
                                                              "selection": "paper seven-topic interpretation, independent of optimal-K selection"})
            write_json(output / "fixed_K07_ready.json", {"fixed_k": k, "case_count": len(cases),
                        "coherence_c_v": row["coherence_c_v"], "remaining_K_search_in_progress": True,
                        "input_cases_sha256": sha256(cases_path), "input_entities_sha256": sha256(entities_path)})
    best_k = min([r for r in measures if r["selection_candidate"]], key=lambda row: (-row["coherence_c_v"], row["k"]))["k"]
    measures.sort(key=lambda row: row["k"])
    write_csv(output / "topic_coherence_curve.csv", measures)
    for label, k in [(f"selected_best_K{best_k:02d}", best_k)]:
        directory = output / label
        keywords, matrix = export_model_results(directory, models[k], corpus, cases, settings, cv_by_k[k],
                                                 serialize=False, frozen_document_topics=results[k][1])
        mappings = export_system_results(directory, keywords, matrix, cases, settings, figures)
        decision_counts[label] = dict(Counter(r["mapping_decision"] for r in mappings))
        write_json(directory / "model_reference.json", {"k": k, "model_path_relative": f"../topic_sweep/K{k:02d}/lda.model",
                                                      "selection": "highest observed c_v" if label.startswith("selected") else "paper seven-topic interpretation, independent of optimal-K selection"})
    if figures:
        candidates = [r for r in measures if r["selection_candidate"]]
        fig, ax = plt.subplots(figsize=(9, 4.4))
        ax.plot([r["k"] for r in candidates], [r["coherence_c_v"] for r in candidates], "o-", color="#315b83")
        ax.axvline(best_k, color="#a34b31", linestyle="--", label=f"Observed best K={best_k}")
        if fixed_k != best_k:
            ax.axvline(fixed_k, color="#737373", linestyle=":", label=f"Paper configuration K={fixed_k}")
        ax.set_xticks([r["k"] for r in candidates])
        ax.set_xlabel("Number of LDA topics (K)")
        ax.set_ylabel("Topic coherence (gensim c_v)")
        ax.set_title("Observed topic coherence curve\nc_v is a disclosed reconstruction choice; no target score imposed")
        ax.legend(fontsize=9)
        fig.tight_layout()
        save_figure(fig, output / "topic_coherence_curve")
    model_hashes = {str(p.relative_to(output)): sha256(p) for p in (output / "topic_sweep").rglob("lda.model*") if p.is_file()}
    summary = {
        "status": "author-method reconstruction from supplied cases/code and manuscript; missing analysis implementation filled under disclosed assumptions",
        "settings": asdict(settings),
        "coherence": {"implementation": "gensim.models.CoherenceModel c_v", "window_size": settings.cv_window_size,
                       "topn_actual": min(settings.coherence_topn, len(dictionary)), "processes": 1,
                       "paper_specifies_variant": False, "assumption": "c_v is the primary index because the paper does not name the variant; it is not the previous NPMI curve"},
        "lda_additional_assumptions": {"batch_update_every": 0, "chunksize": 2000, "alpha": "symmetric (1/K)",
                                       "eta": "gensim default symmetric (1/K)", "eval_every": None, "dtype": "float32"},
        "tokenization": {"implementation": "jieba.Tokenizer", "HMM": False, "added_word_frequency": 100000,
                          "NER_entry_rule": "2-15 characters; no punctuation/whitespace/clause or maintenance-action fragments; mixed Chinese/ASCII or unknown all-Chinese spans need an explicit engineering anchor; otherwise known jieba words are allowed. Standalone entities are preserved as words, never concatenated into sentence tokens.",
                          "stopwords": "explicit fixed function and maintenance-boilerplate words; exported without adaptive topic-score tuning",
                          "vocabulary_filter": {"no_below_document_count": settings.min_df, "no_above": 1.0, "keep_n": None}},
        "input": {"cases": str(cases_path.resolve()), "entities": str(entities_path.resolve()),
                  "cases_sha256": sha256(cases_path), "entities_sha256": sha256(entities_path)},
        "case_count": len(cases), "sample_size_requested": sample_size,
        "case_body_duplicates_retained": len(cases) - len({r["text"] for r in cases}),
        "duplicate_body_policy": "keep all separately numbered input records; strip enumeration only",
        "ner_body_mismatch_count": sum(r["ner_text_matches_body"] is False for r in cases),
        "vocabulary_before_min_df": before_filter, "vocabulary_after_min_df": len(dictionary),
        "empty_bow_cases_retained": sum(not bow for bow in corpus),
        "best_k": best_k, "best_coherence_c_v": next(r["coherence_c_v"] for r in measures if r["k"] == best_k),
        "paper_fixed_k": fixed_k, "paper_fixed_coherence_c_v": next(r["coherence_c_v"] for r in measures if r["k"] == fixed_k),
        "selection_rule": "maximum observed c_v across stated K range; exact ties choose smaller K; fixed seven-topic output is independently reported",
        "training_order": training_order,
        "document_inference_policy": "one document-topic inference per fitted model; frozen posterior matrix reused in sweep/selected/fixed exports to avoid initialization jitter",
        "mapping": {"rule": "sum top-keyword probabilities once per matched engineering system; normalize each topic across seven systems; retain zero-match topics as unmapped",
                    "case_probability_rule": "document-topic probabilities multiplied by topic-system affinity; unmapped mass retained, ambiguous mapping recorded",
                    "gold_labels": False, "decisions": decision_counts},
        "model_file_sha256": model_hashes, "script_sha256": sha256(Path(__file__)),
        "versions": {"python": sys.version.split()[0], "gensim": gensim.__version__, "jieba": jieba.__version__,
                     "numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "runtime_seconds": time.perf_counter() - start,
        "limits": ["No desired K or coherence score is hard-coded as an observed result.",
                   "Engineering system dictionaries and affinity thresholds are disclosed assumptions, not manually labeled diagnostic gold.",
                   "Repeated case bodies may be repeated source evidence; retaining them does not create independent cases.",
                   "LDA topic distributions and seven-system affinities do not establish causal fault propagation or held-out diagnostic accuracy."],
    }
    write_json(output / "analysis_summary.json", summary)
    write_json(output / "fixed_K07_ready.json", {"fixed_k": fixed_k, "case_count": len(cases),
                "coherence_c_v": summary["paper_fixed_coherence_c_v"], "remaining_K_search_in_progress": False,
                "input_cases_sha256": sha256(cases_path), "input_entities_sha256": sha256(entities_path)})
    print(json.dumps({"case_count": len(cases), "vocabulary": len(dictionary), "best_k": best_k,
                      "best_c_v": summary["best_coherence_c_v"], "fixed_K7_c_v": summary["paper_fixed_coherence_c_v"],
                      "runtime_seconds": summary["runtime_seconds"]}, ensure_ascii=False), flush=True)
    return summary



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path)
    parser.add_argument("--entities", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--topic-min", type=int, default=3)
    parser.add_argument("--topic-max", type=int, default=12)
    parser.add_argument("--passes", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-figures", action="store_true")
    args = parser.parse_args()
    if not args.cases or not args.entities or not args.output_dir:
        parser.error("--cases, --entities and --output-dir are required")
    if args.sample_size is not None and args.sample_size < 1:
        parser.error("--sample-size must be positive")
    settings = Settings(seed=args.seed, topic_min=args.topic_min, topic_max=args.topic_max,
                        passes=args.passes, iterations=args.iterations, min_df=args.min_df)
    run_analysis(args.cases, args.entities, args.output_dir, settings,
                 sample_size=args.sample_size, figures=not args.no_figures)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# Diesel Fault Knowledge Graph

面向柴油机维护案例的语义增强知识图谱项目，提供实体与关系抽取、LDA 主题分析、图谱构建、候选传播路径挖掘、静态结构分析和实验评价。

## 安装环境

使用 Python 3.12。安装命令通过 POSIX shell 执行，依赖版本由 `requirements.lock.txt` 固定。

```bash
git clone https://github.com/Pioneer6gun9/diesel-fault-kg.git
cd diesel-fault-kg
./setup.sh
```

`setup.sh` 创建项目内的 `.venv/`，安装依赖并检查依赖兼容性。指定其他 Python 3.12 解释器时：

```bash
DIESEL_KG_PYTHON=/path/to/python3.12 ./setup.sh
```

生成论文 PDF 还需要包含 `latexmk` 的 TeX 环境；只生成图表和 LaTeX 源码时不需要执行 PDF 编译。

## 数据与模型准备

当前实验使用仓库提供的固定语料、数据划分和输入校验清单。配置位于 `configs/default.json`，所有路径相对于项目根目录解析。

| 目录或文件 | 用途 |
| --- | --- |
| `data/kg_build/` | 案例语料、NER/RE 数据划分、原算法和原始检查点 |
| `data/topic_dictionary_entities.jsonl` | 主题分析使用的固定实体词典 |
| `data/manifest.json` | 固定输入文件的 SHA256 校验清单 |
| `assets/pretrained/`、`assets/tokenizers/` | 预训练模型、模型配置和词表 |
| `results/experiments/` | 默认权重复用所需的模型、训练记录和历史参考结果 |
| `.cache/ner_domain_features/` | 域适配模型校验使用的特征分片 |

下载并校验固定版本的公开预训练模型：

```bash
./run assets --download
```

模型权重、训练缓存和运行产物需另行准备。原始检查点应放在以下位置；默认完整流程还需要 `results/experiments/` 中的模型与配套记录。

```text
data/kg_build/NER/best_roberta_ner_model/best_roberta_ner_model.pt
data/kg_build/RE/best_re_model/best_re_model.pt
```

`--train` 会重新训练模型，但完整流程仍会审计原始检查点并读取历史图谱参考结果。预训练模型下载完成并不代表全部实验资源已经齐备。

资源准备完成后检查环境、输入文件和模型状态：

```bash
./run doctor
./run assets
./run status
```

训练设备自动选择可用的 MPS，否则使用 CPU；也可通过 `--device cpu` 显式指定。复制实验资源时应保留相对目录结构和内部符号链接。

## 运行

```bash
./run                                      # 使用已有权重执行完整流程
./run --train --output results/train_run    # 重新训练模型并执行完整流程
./run --dry-run --output results/new_run    # 查看执行计划
./run --resume --output results/run         # 校验记录并继续已有运行
./run --device cpu --output results/cpu_run # 指定 CPU
```

默认流程包含 19 个顶层阶段，依次执行数据审计、NER/RE 全语料推理、主题分析、图谱与传播路径分析、消融和模型对比、计算校验、指标汇总及论文图表生成。`--train` 模式增加模型训练阶段。

新运行默认输出到 `results/run/`，也可通过 `--output` 指定 `results/` 下的其他空目录。已有结果不会被直接覆盖；中断后使用相同输出目录和 `--resume` 继续。

默认运行会复用并校验已有权重，运行记录标记为 `recorded_checkpoints`。`--train` 模式的训练与后续评价结果均写入指定输出目录。

## 图表与结果

在完整实验结果的基础上，可以单独执行图谱分析或图表生成：

```bash
./run graph --experiment results/run --output results/graph_run
./run plot --experiment results/run --output results/plots
./run manuscript --experiment results/run --output results/paper --compile
```

`--experiment` 指定包含输入、模型、预测和评价的一套完整实验目录，默认读取 `results/experiments/`。这些命令的输出目录同样需要为空。

| 输出 | 内容 |
| --- | --- |
| `pipeline.json` | 完整运行状态、阶段命令、源码哈希和权重信息 |
| `execution_logs/` | 各阶段的执行日志 |
| `ner/`、`ner_domain/adaptation/` | NER 模型记录、评价及逐例实体预测 |
| `recorded_RE_training/`、`re_clean/` | RE 模型记录、固定划分及测试预测 |
| `topics*`、`graph_k7/`、`graph_k10/` | 主题扫描、图谱和传播路径分析 |
| `graph_ablation/`、`adapted_model_graph/` | 图谱消融与模型预测图谱 |
| `cross_model_silver_evaluation/` | 使用同一固定银标参考的模型对比 |
| `measured_metrics.csv`、`metrics_provenance.json` | 25 项汇总指标及对应来源文件、字段和哈希 |
| `manuscript/` | 论文图表、LaTeX 源码包、编译结果及数据来源记录 |

论文图表脚本位于 `src/diesel_kg/visualization/manuscript.py`，模板输入位于 `data/manuscript/`。默认模板生成 8 张图、5 张数据表和 1 张专家评分规则表，表格采用左对齐排版。专家评分规则表不包含实际评分。

`manuscript/overleaf/` 保存 LaTeX 源码和图片，`overleaf.zip` 为上传包，`paper.pdf` 为编译结果。`data_sources.json` 记录图表输入与生成程序的哈希。模板正文包含数值一致性校验；实验结果发生变化时，需要核对正文和模板中的对应数值。

计算结果描述候选关系、图谱结构和静态传播模式。诊断准确率、专家评价及动态演化结论需要独立标注、真实评分或时间序列证据。

## 项目结构

```text
diesel-fault-kg/
├── run / run.py                 # 统一命令行入口
├── setup.sh                     # 环境安装
├── configs/                     # 默认配置、分析目标和参考目录
├── data/
│   ├── kg_build/                # 固定语料、数据划分和原算法
│   ├── paper/                   # 论文源文件及原始图片
│   ├── manuscript/              # 论文图表模板与原理图
│   ├── topic_dictionary_entities.jsonl
│   └── manifest.json            # 输入校验清单
├── assets/                      # 预训练模型与词表
├── src/diesel_kg/
│   ├── data/                    # 数据处理与资源准备
│   ├── models/                  # 模型定义与加载
│   ├── training/                # NER/RE 训练
│   ├── inference/               # 实体与关系推理
│   ├── graph/                   # 主题、图谱、路径和消融
│   ├── evaluation/              # 实验评价与指标汇总
│   ├── validation/              # 数据、模型和计算一致性校验
│   ├── visualization/           # 实验图表生成
│   └── pipelines/               # 流程调度
├── results/                     # 模型记录、预测、评价和图表
├── requirements.txt             # 直接依赖
├── requirements.lock.txt        # 固定版本依赖
├── LICENSE
├── .venv/                       # 安装时创建的 Python 环境
└── .cache/                      # 模型特征缓存
```

原算法 `EntityExtraction.py` 和 `RelationExtraction.py` 保存在 `data/kg_build/`，统一实现与流程调度位于 `src/diesel_kg/`。

## 模拟数据

生成带有 **SIMULATED** 标记的流程验证数据：

```bash
./run simulate --seed 20261008 --cases-per-system 12 --output results/simulated
```

该命令生成 CSV/JSON 数据和记录生成假设、随机种子及校验信息的 `manifest.json`，不生成 Excel 文件。模拟数据包括虚拟关系标注、根因与传播链、物理标签、SIS 评分、故障定位计时和事件时间戳。模拟数据独立保存，完整实测流程不会将其作为真实验证证据。

## 许可证

采用 Apache-2.0 许可证，详见 [LICENSE](LICENSE)。

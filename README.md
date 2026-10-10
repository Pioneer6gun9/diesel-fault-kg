# Diesel Fault Knowledge Graph

柴油机故障知识图谱：语义增强的故障传播分析与演化模式挖掘。

本地完整项目包含源码、原始数据、预训练模型、已训练权重和实验结果，统一由 `run` 启动。一套完整实验保存在同一个目录内，包含数据处理、NER/RE、主题分析、图谱构建、传播路径、消融、评价和图片。

## GitHub 版本

仓库保存源码、配置、依赖清单、输入数据、论文图表模板，以及模型配置和词表。模型权重（`.pt`、`.bin`、`.safetensors` 等）、`.venv/`、`.cache/` 和 `results/` 不随 Git 提交，本地文件仍保留。GitHub 普通 Git 仓库不接受大于 100 MiB 的文件；当前单个模型权重约为 390–465 MiB，见 [GitHub 文件限制](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github)。

```bash
git clone https://github.com/Pioneer6gun9/diesel-fault-kg.git
cd diesel-fault-kg
./setup.sh
```

克隆后需要恢复本地完整项目中的原检查点、已训练模型记录和域适配特征分片，保持 `data/kg_build/`、`results/experiments/` 和 `.cache/ner_domain_features/` 的相对目录结构及符号链接。预训练权重可通过 `./run assets --download` 获取固定版本。只克隆源码并安装 Python 依赖，尚不足以直接执行默认完整流程；资源准备齐全后，先运行 `./run doctor` 和 `./run assets`，再运行 `./run`。输入校验清单见 `data/manifest.json`。

## 运行

```bash
cd diesel-fault-kg
./run                            # 使用已有权重，运行完整实验流程
./run --train                    # 重新训练全部模型，再运行完整实验流程
./run --dry-run                  # 查看执行步骤，不启动实验
./run --resume                   # 校验本次运行记录并继续未完成阶段
./run status                     # 查看已有权重和实测汇总
./run doctor                     # 检查输入数据和运行环境
```

完整流程依次执行：固定输入与数据审计 → 原算法 NER/RE 及全语料推理 → LDA、图谱和传播模式分析 → NER 域适配与分组 RE → 图谱消融和固定参考对比 → 校验、实测指标汇总 → 投稿稿彩色图、左对齐表格、源码包及 PDF。默认使用已有权重，共 19 个顶层阶段。

默认复用模式会校验已训练权重，并在运行记录中明确标记 `recorded_checkpoints`。`--train` 才会重新训练原算法、NER 域适配和分组 RE。两种模式使用同一个实验目录和流程入口。

新运行默认写入 `results/run/`；也可使用 `./run --output results/自定义目录`。入口要求空目录，并保护已有的 `results/experiments/`、模拟数据和溯源记录。`pipeline.json` 记录整套运行状态、每阶段命令、代码哈希和权重复用信息；日志在本次目录的 `execution_logs/`。

当前 `results/run/` 已实际完成全部 19 个阶段，复用已有权重重新执行全语料推理、图谱分析、评价和投稿稿生成；25 项汇总指标与保存的实测结果一致。该目录已经存在，再次校验续跑使用 `./run --resume`；从头运行时用 `--output` 指定另一个空目录。

只重跑图谱或图片时：

```bash
./run graph                      # 使用现有权重和预测重跑图谱实验
./run plot                       # 从已有实测结果重新出图
./run manuscript --compile      # 单独生成投稿稿图表、源码包和 PDF
```

默认读取 `results/experiments/`，也可用 `--experiment results/自定义实验目录` 选择另一套完整实验。图谱和图片的新输出分别在 `results/graph_rerun/` 和 `results/figures/`。

完整运行的投稿稿文件在本次实验目录的 `manuscript/`：`overleaf/` 为源码和 8 张图，`overleaf.zip` 可上传 Overleaf，`paper.pdf` 为编译结果，`data_sources.json` 记录输入和生成程序的哈希。图表生成代码位于 `src/diesel_kg/visualization/manuscript.py`，不再依赖其他工作目录。它保留当前作者、正文、表格列宽和左对齐样式，从实测结果重新计算 5 张数据表，保留没有真实评分的专家评价表。若新结果与正文中的数值不一致，生成程序会报出差异。

## 目录

```text
diesel-fault-kg/
├── run / run.py                 # 统一命令行入口
├── setup.sh                     # 安装环境
├── configs/                     # 默认路径、论文目标和参考目录
├── data/
│   ├── kg_build/                # 原始语料、NER/RE 划分、原算法与原检查点
│   ├── paper/                   # 原论文源文件及引用图
│   ├── manuscript/              # 当前投稿稿模板、出版社样式及原理图输入
│   ├── topic_dictionary_entities.jsonl
│   └── manifest.json            # 固定输入校验清单
├── assets/                      # 固定版本 BERT、RoBERTa 与 tokenizer
├── src/diesel_kg/
│   ├── data/                    # 数据处理和资源准备
│   ├── models/                  # 模型定义与加载
│   ├── training/                # NER/RE 训练
│   ├── inference/              # 实体和关系推理
│   ├── graph/                  # 主题、图谱、路径和消融
│   ├── evaluation/             # 评价与实测汇总
│   ├── validation/             # 数据、模型和计算一致性校验
│   ├── visualization/          # 实验图片及投稿稿图表生成
│   └── pipelines/              # 完整实验流程调度
├── results/
│   ├── experiments/             # 已完成的一套完整实验：权重、预测和评价
│   ├── run/                     # 本次完整运行的预测、评价、图表及投稿稿
│   ├── measured_metrics.csv     # 指向完整实验中的同一份实测汇总
│   ├── simulated/               # SIMULATED 仿真输入与唯一一份模拟 Excel
│   └── .provenance/              # 历史诊断、文件迁移与清理记录
├── requirements.txt             # 直接依赖
├── requirements.lock.txt        # 固定版本的完整依赖
├── .venv/                       # 当前 Mac 的运行环境
└── .cache/                      # 域适配训练与校验需要的特征分片
```

原算法 `EntityExtraction.py`、`RelationExtraction.py` 作为固定输入保存在 `data/kg_build/`；可执行实现统一在 `src/diesel_kg/`。本项目使用自己的数据、资源和环境。

## 权重和结果

最新完整运行位于 `results/run/`。`results/experiments/` 保留原有训练权重和实测记录，本次通过项目内部链接复用其中的权重。两套实验均按以下功能存放结果：

| 位置 | 内容 |
| --- | --- |
| `ner/`、`ner_domain/adaptation/` | NER 权重、训练记录与逐例实体预测 |
| `recorded_RE_training/`、`re_clean/` | RE 权重、固定划分、训练记录和测试预测 |
| `topics*`、`graph_k7/`、`graph_k10/` | 主题扫描、图谱与传播分析 |
| `graph_ablation/`、`adapted_model_graph/` | 不同模型的图谱与消融实验 |
| `cross_model_silver_evaluation/` | 使用同一冻结银标参考的模型对比 |
| `engineering/`、`system_validation/`、`temporal/` | 工程评价及输入验证 |
| `measured_metrics.csv` | 实测计算结果汇总 |

新完整运行结束时会自动生成包含 25 项指标的 `measured_metrics.csv` 和 `metrics_provenance.json`，后者逐项记录指标的来源文件、字段和 SHA256。历史数据及模型记录保留原执行路径和哈希，由 `configs/recorded_paths.json` 映射到当前项目；历史诊断记录集中在 `results/.provenance/experiments/`。已移除的历史脚本只保留指纹记录，校验报告明确披露这一点；本次新运行的源码按实际文件严格校验。

完全相同的固定文件通过项目内部相对符号链接共享。复制项目时应保留整个目录结构和链接。原始输入与实验固定输入独立保存，模型权重、数据划分和逐例预测均保留。

## 模拟数据

`results/simulated/` 保存明确标注 **SIMULATED** 的 7 系统、84 虚拟案例数据包，以及 `论文验证模拟数据_SIMULATED.xlsx`。内容包括关系正负标注、根因与传播链、物理标签与假定关联强度、虚拟 SIS 评分、虚拟工程师计时和事件时间戳。

```bash
./run simulate --seed 20261008 --cases-per-system 12 --output results/simulated_rerun
```

该命令在新空目录重建 CSV/JSON；交付 Excel 是现有数据的带公式查看副本，命令不重新生成 Excel。生成假设、随机种子和校验值见 `manifest.json`。这些数据用于仿真流程与计算验证；真实专家评分、工业根因、人工诊断效率和真实时序结论仍需要对应的真实证据。完整实测实验流程不会将模拟数据作为真实标注。

## 安装环境

当前电脑可以直接使用项目内 `.venv`。换电脑或重新安装时，安装 Python 3.12，然后执行：

```bash
./setup.sh
./run doctor
./run assets
```

依赖使用 `requirements.lock.txt` 固定版本。可用 `DIESEL_KG_PYTHON=/Python3.12解释器路径 ./setup.sh` 指定解释器。预训练模型已包含在项目中；缺少时可执行 `./run assets --download` 下载固定版本。

配置路径相对于项目根目录解析。训练设备自动选择可用的 MPS，否则使用 CPU，也可显式传入 `--device cpu`。`.venv` 换操作系统或 Python 安装位置时需要重建。`.cache/ner_domain_features/` 的特征分片参与现有 NER 校验，需保留。

编译投稿稿需要安装包含 `latexmk` 的 TeX 环境。当前 Mac 已有该环境；`./run plot` 可以只生成图表和源码，`./run manuscript --compile` 和完整运行会编译 PDF。

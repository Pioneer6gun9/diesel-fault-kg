import os

# 设置Hugging Face国内镜像（优先下载）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
import json
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, RobertaModel, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, f1_score, confusion_matrix
import warnings
# 可视化依赖
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

warnings.filterwarnings('ignore')

# ===================== 配置项（适配新数据目录） =====================
# 基础配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOG_DIR = "./logs_roberta_ner"
MODEL_SAVE_DIR = "./best_roberta_ner_model"
VISUALIZATION_DIR = "./visualizations_roberta_ner"
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

# 数据配置
DATA_DIR = "./ner_data_all_generate"
TRAIN_FILE = os.path.join(DATA_DIR, "train.txt")
DEV_FILE = os.path.join(DATA_DIR, "dev.txt")
TEST_FILE = os.path.join(DATA_DIR, "test.txt")

# 模型配置
ROBERTA_MODEL_NAME = "hfl/chinese-roberta-wwm-ext"  # 中文RoBERTa-WWM-Ext（效果更优）
MAX_SEQ_LEN = 512
DROPOUT = 0.2  # 防止纯RoBERTa过拟合
HIDDEN_SIZE = 768  # RoBERTa默认隐藏层维度

# 训练配置
BATCH_SIZE = 8  # 显存不足改为4，充足可改为16
LEARNING_RATE = 2e-5  # 纯RoBERTa最优学习率区间
EPOCHS = 15
WARMUP_RATIO = 0.1  # 学习率预热比例
PATIENCE = 5  # 早停耐心值（连续5轮验证集F1不提升则停止）
VERBOSE_STEP = 10  # 每10步打印一次训练进度

# 可视化配置（确保中文正常显示）
plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows
# plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']  # MacOS
# plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # Linux
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10

# ===================== 日志配置 =====================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "roberta_ner_train.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ===================== 数据加载与预处理（适配BIO格式） =====================
class NERDataset(Dataset):
    """NER数据集类，加载BIO格式数据"""

    def __init__(self, file_path, tokenizer, max_seq_len, label2id):
        self.data = self.load_data(file_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.label2id = label2id
        self.id2label = {v: k for k, v in label2id.items()}

    def load_data(self, file_path):
        """加载BIO格式数据，添加格式校验"""
        sentences = []
        labels = []
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            sent = []
            lab = []
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    if sent and lab:
                        if len(sent) != len(lab):
                            logger.warning(f"警告：第{line_num}行附近句子与标签长度不一致，跳过")
                        else:
                            sentences.append(sent)
                            labels.append(lab)
                    sent = []
                    lab = []
                else:
                    parts = line.split("\t", 1)
                    if len(parts) != 2:
                        logger.warning(f"警告：第{line_num}行格式错误，跳过：{line}")
                        continue
                    char, label = parts
                    sent.append(char)
                    lab.append(label)
            if sent and lab and len(sent) == len(lab):
                sentences.append(sent)
                labels.append(lab)
        logger.info(f"加载 {file_path}：{len(sentences)} 个句子")
        return list(zip(sentences, labels))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """编码单个样本，实现标签与Token对齐"""
        sentence, labels = self.data[idx]
        full_sentence = "".join(sentence)
        sentence_len = len(full_sentence)

        # Token编码（返回offset_mapping用于标签对齐）
        encoding = self.tokenizer(
            full_sentence,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_offsets_mapping=True
        )

        input_ids = torch.tensor(encoding["input_ids"], dtype=torch.long)
        attention_mask = torch.tensor(encoding["attention_mask"], dtype=torch.long)

        # 初始化标签（默认O标签）
        label_ids = torch.full((self.max_seq_len,), self.label2id["O"], dtype=torch.long)
        offsets = encoding["offset_mapping"]  # (max_seq_len, 2)：每个Token的字符起始/结束索引

        # 标签对齐：将字符级标签映射到Token级
        for token_idx in range(1, len(offsets) - 1):  # 跳过[CLS]和[SEP]
            start, end = offsets[token_idx]
            if start == end or start >= sentence_len:
                continue  # 跳过padding或无效Token
            char_idx = start  # Token的第一个字符对应原始句子的索引
            if char_idx < len(labels):
                label_ids[token_idx] = self.label2id.get(labels[char_idx], self.label2id["O"])

        # 调试输出（仅第一个短样本，验证对齐效果）
        if idx == 0 and len(sentence) <= 50:
            logger.debug(f"原始句子：{full_sentence}")
            logger.debug(f"原始标签：{labels}")
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[:sentence_len + 2])
            token_labels = [self.id2label[label_ids[i].item()] for i in range(len(tokens))]
            logger.debug(f"Token序列：{tokens}")
            logger.debug(f"Token标签：{token_labels}")

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label_ids": label_ids
        }


def build_label_mapping(train_file):
    """从训练集构建标签映射（BIO格式自动识别）"""
    labels = set()
    with open(train_file, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if line:
                parts = line.split("\t", 1)
                if len(parts) != 2:
                    logger.warning(f"警告：第{line_num}行标签提取失败，跳过：{line}")
                    continue
                _, label = parts
                labels.add(label)
    # 按BIO规则排序（O→B-xxx→I-xxx）
    label_list = sorted(labels, key=lambda x: (
        x.split("-")[-1] if "-" in x else "",
        0 if x == "O" else 1 if x.startswith("B-") else 2
    ))
    label2id = {label: i for i, label in enumerate(label_list)}
    id2label = {i: label for label, i in label2id.items()}
    logger.info(f"最终标签列表：{label_list}")
    logger.info(f"标签数量：{len(label_list)}")
    logger.info(f"标签映射：{label2id}")
    return label2id, id2label, label_list


# ===================== 纯RoBERTa模型定义（无BiLSTM+CRF） =====================
class RoBERTaOnlyNER(nn.Module):
    def __init__(self, roberta_model_name, num_labels, dropout):
        super().__init__()
        # 加载RoBERTa-WWM预训练模型
        self.roberta = RobertaModel.from_pretrained(roberta_model_name)
        # 解冻最后20层（纯RoBERTa需更多可训练参数弥补无LSTM/CRF的不足）
        for param in list(self.roberta.parameters())[:-20]:
            param.requires_grad = False
        logger.info("RoBERTa解冻策略：保留最后20层可训练")

        self.dropout = nn.Dropout(dropout)
        # 线性层直接映射到标签空间（768维→标签数）
        self.classifier = nn.Linear(self.roberta.config.hidden_size, num_labels)

        # 初始化线性层权重（提升训练稳定性）
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask, label_ids=None):
        """
        前向传播：
        - 训练模式（label_ids≠None）：返回交叉熵损失
        - 预测模式（label_ids=None）：返回预测标签序列
        """
        # RoBERTa编码：(batch_size, seq_len, 768)
        roberta_output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = roberta_output.last_hidden_state  # 取最后一层隐藏状态
        last_hidden_state = self.dropout(last_hidden_state)

        # 分类预测：(batch_size, seq_len, num_labels)
        logits = self.classifier(last_hidden_state)

        if label_ids is not None:
            # 训练模式：计算交叉熵损失（忽略padding位置）
            loss_fct = nn.CrossEntropyLoss()
            active_loss = attention_mask.view(-1) == 1  # 过滤padding（attention_mask=0）
            active_logits = logits.view(-1, self.classifier.out_features)
            active_labels = label_ids.view(-1)[active_loss]
            loss = loss_fct(active_logits[active_loss], active_labels)
            return loss
        else:
            # 预测模式：取每个位置概率最大的标签
            pred_labels = torch.argmax(logits, dim=-1)  # (batch_size, seq_len)
            return pred_labels.cpu().numpy()


# ===================== 评估函数（适配纯RoBERTa输出） =====================
def evaluate(model, dataloader, id2label, label2id, return_details=False):
    model.eval()
    all_true_labels = []
    all_pred_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            label_ids = batch["label_ids"].to(DEVICE)

            # 预测标签序列
            pred_labels = model(input_ids=input_ids, attention_mask=attention_mask)

            # 转换为原始标签（过滤[CLS]、[SEP]和padding）
            for i in range(len(pred_labels)):
                mask = attention_mask[i].cpu().numpy()
                true_label = label_ids[i].cpu().numpy()
                pred_label = pred_labels[i]

                # 只保留有效位置（mask=1）的标签，跳过[CLS]（第0位）和[SEP]（最后一个有效位）
                valid_true = [id2label[t] for t, m in zip(true_label, mask) if m == 1][1:-1]
                valid_pred = [id2label[p] for p, m in zip(pred_label, mask) if m == 1][1:-1]

                # 确保长度一致
                min_len = min(len(valid_true), len(valid_pred))
                all_true_labels.extend(valid_true[:min_len])
                all_pred_labels.extend(valid_pred[:min_len])

    # 计算核心指标（排除O标签，聚焦实体识别）
    entity_labels = [l for l in label2id if l != "O"]
    f1_micro = f1_score(all_true_labels, all_pred_labels, average="micro", labels=entity_labels, zero_division=0)
    f1_macro = f1_score(all_true_labels, all_pred_labels, average="macro", labels=entity_labels, zero_division=0)
    precision_micro = f1_score(
        all_true_labels, all_pred_labels, average="micro", labels=entity_labels, pos_label=None, zero_division=0
    )
    recall_micro = f1_score(
        all_true_labels, all_pred_labels, average="micro", labels=entity_labels, pos_label=None, zero_division=0
    )

    # 打印分类报告
    logger.info("\n" + "=" * 50)
    logger.info("RoBERTa实体识别分类报告（排除O标签）")
    logger.info("=" * 50)
    logger.info(classification_report(
        all_true_labels, all_pred_labels,
        labels=entity_labels,
        target_names=entity_labels,
        zero_division=0
    ))

    # 打印整体指标
    logger.info("\n" + "=" * 50)
    logger.info("整体性能指标（排除O标签）")
    logger.info("=" * 50)
    logger.info(f"Micro-Precision: {precision_micro:.4f}")
    logger.info(f"Micro-Recall: {recall_micro:.4f}")
    logger.info(f"Micro-F1: {f1_micro:.4f}")
    logger.info(f"Macro-F1: {f1_macro:.4f}")

    overall_metrics = {"Precision": precision_micro, "Recall": recall_micro, "F1": f1_micro}

    if return_details:
        report = classification_report(
            all_true_labels, all_pred_labels,
            labels=entity_labels, target_names=entity_labels,
            zero_division=0, output_dict=True
        )
        return f1_micro, all_true_labels, all_pred_labels, report, overall_metrics
    return f1_micro


# ===================== 可视化函数（4类图表全覆盖） =====================
def plot_training_metrics(train_losses, dev_f1s, save_path):
    """训练损失+验证集F1曲线（直观监控训练过程）"""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    # 训练损失
    color1 = '#FF6B6B'
    ax1.set_xlabel('训练轮数（Epoch）', fontsize=12)
    ax1.set_ylabel('训练损失（Loss）', color=color1, fontsize=12)
    ax1.plot(range(1, len(train_losses) + 1), train_losses, color=color1, marker='o', linewidth=2, label='训练损失')
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.grid(alpha=0.3)

    # 验证集F1
    ax2 = ax1.twinx()
    color2 = '#4ECDC4'
    ax2.set_ylabel('验证集Micro-F1', color=color2, fontsize=12)
    ax2.plot(range(1, len(dev_f1s) + 1), dev_f1s, color=color2, marker='s', linewidth=2, label='验证集F1')
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.set_ylim(0, 1.05)  # F1值范围固定在0-1.05

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower right', fontsize=10)

    plt.title('RoBERTa实体识别模型训练损失与验证集F1变化', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ 训练曲线已保存：{save_path}")


def plot_classification_report(class_report, overall_metrics, entity_labels, save_path):
    """每个实体标签仅显示一次，展示其精确率、召回率、F1分数"""
    # 提取数据：精确率（蓝）、召回率（橙）、F1分数（绿）
    metrics = ['precision', 'recall', 'f1-score']
    class_data = []
    # 提取每个实体类别的指标（每个标签仅取一次）
    for label in entity_labels:
        # 仅保留实体类型名（如B-发动机型号 → 发动机型号），并去重
        entity_name = label.split('-')[-1]
        # 每个实体类型只保留一条数据（取B-开头的标签对应的指标）
        if label.startswith('B-'):
            class_data.append([class_report[label][m] for m in metrics])
    # 提取去重后的实体标签名
    unique_labels = [label.split('-')[-1] for label in entity_labels if label.startswith('B-')]
    # 添加“整体性能”行
    class_data.append([
        class_report["macro avg"]["precision"],
        class_report["macro avg"]["recall"],
        class_report["macro avg"]["f1-score"]
    ])
    unique_labels.append('整体性能')

    # 绘图配置：适配去重后的标签数量
    plt.figure(figsize=(16, 9))
    x = np.arange(len(unique_labels))
    width = 0.28  # 宽柱子保证数值显示

    # 绘制并列柱状图
    bar1 = plt.bar(x - width, [d[0] for d in class_data], width,
                   label='精确率', color='#1f77b4', edgecolor='black', alpha=0.9)
    bar2 = plt.bar(x, [d[1] for d in class_data], width,
                   label='召回率', color='#ff7f0e', edgecolor='black', alpha=0.9)
    bar3 = plt.bar(x + width, [d[2] for d in class_data], width,
                   label='F1分数', color='#2ca02c', edgecolor='black', alpha=0.9)

    # 数值标签：清晰显示在柱子上方
    for i in range(len(x)):
        plt.text(x[i] - width, class_data[i][0] + 0.02,
                 f"{class_data[i][0]:.3f}", ha='center', va='bottom',
                 fontsize=10, fontweight='bold')
        plt.text(x[i], class_data[i][1] + 0.02,
                 f"{class_data[i][1]:.3f}", ha='center', va='bottom',
                 fontsize=10, fontweight='bold')
        plt.text(x[i] + width, class_data[i][2] + 0.02,
                 f"{class_data[i][2]:.3f}", ha='center', va='bottom',
                 fontsize=10, fontweight='bold')

    # 图例与样式配置
    plt.legend(loc='upper right', fontsize=12, frameon=True)
    plt.title('RoBERTa模型实体性能对比', fontsize=16, pad=25, fontweight='bold')
    plt.xlabel('实体类型', fontsize=14, fontweight='bold')
    plt.ylabel('指标值（0-1）', fontsize=14, fontweight='bold')
    plt.xticks(x, unique_labels, rotation=0, ha='center', fontsize=12)
    plt.ylim(0, 1.2)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ 去重标签后的实体性能对比图已保存：{save_path}")


def plot_category_heatmap(report, entity_labels, save_path):
    """类别性能热力图（直观查看各类别强弱）"""
    metrics = ['precision', 'recall', 'f1-score']
    heatmap_data = []
    for label in entity_labels:
        heatmap_data.append([report[label][m] for m in metrics])

    # 转换为DataFrame
    df = pd.DataFrame(
        heatmap_data,
        index=[label.replace('-', '\n') for label in entity_labels],
        columns=['精确率', '召回率', 'F1分数']
    )

    plt.figure(figsize=(10, len(entity_labels) * 0.8))
    sns.heatmap(
        df,
        annot=True, fmt='.3f',  # 保留3位小数
        cmap='RdYlBu_r',  # 红黄绿蓝配色（红色差，蓝色优）
        vmin=0, vmax=1,  # 颜色范围0-1
        cbar_kws={'label': '指标值'},
        linewidths=0.5,
        annot_kws={'fontsize': 9}
    )

    plt.title('RoBERTa模型各类别实体识别性能热力图', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ 类别热力图已保存：{save_path}")


def plot_confusion_matrix(true_labels, pred_labels, entity_labels, save_path):
    """混淆矩阵（分析预测错误分布）"""
    cm = confusion_matrix(true_labels, pred_labels, labels=entity_labels)
    df_cm = pd.DataFrame(
        cm,
        index=[label.replace('-', '\n') for label in entity_labels],
        columns=[label.replace('-', '\n') for label in entity_labels]
    )

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        df_cm,
        annot=True, fmt='d',  # 显示样本数量
        cmap='Blues',  # 蓝色系配色
        cbar_kws={'label': '样本数量'},
        linewidths=0.5,
        annot_kws={'fontsize': 8}  # 小字体避免拥挤
    )

    plt.xlabel('预测标签', fontsize=12)
    plt.ylabel('真实标签', fontsize=12)
    plt.title('RoBERTa实体识别混淆矩阵（排除O标签）', fontsize=14, fontweight='bold', pad=20)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✅ 混淆矩阵已保存：{save_path}")


# ===================== 训练主函数 =====================
def train_roberta_ner():
    logger.info(f"使用设备：{DEVICE}")
    logger.info("=" * 60)
    logger.info("开始RoBERTa实体识别模型训练（适配新数据目录）")
    logger.info("=" * 60)

    # 1. 校验数据文件是否存在
    for file_path in [TRAIN_FILE, DEV_FILE, TEST_FILE]:
        if not os.path.exists(file_path):
            logger.error(f"❌ 数据文件不存在：{file_path}")
            raise FileNotFoundError(f"请确保数据文件在目录 {DATA_DIR} 中")

    # 2. 构建标签映射
    logger.info("\n1. 构建标签映射...")
    label2id, id2label, label_list = build_label_mapping(TRAIN_FILE)
    NUM_LABELS = len(label_list)
    entity_labels = [l for l in label_list if l != "O"]
    logger.info(f"实体类别数量：{len(entity_labels)}")

    # 3. 加载RoBERTa分词器
    logger.info(f"\n2. 加载分词器：{ROBERTA_MODEL_NAME}")
    tokenizer = BertTokenizerFast.from_pretrained(ROBERTA_MODEL_NAME)
    logger.info("✅ 分词器加载完成")

    # 4. 构建数据集和数据加载器
    logger.info(f"\n3. 加载数据集（目录：{DATA_DIR}）...")
    train_dataset = NERDataset(TRAIN_FILE, tokenizer, MAX_SEQ_LEN, label2id)
    dev_dataset = NERDataset(DEV_FILE, tokenizer, MAX_SEQ_LEN, label2id)
    test_dataset = NERDataset(TEST_FILE, tokenizer, MAX_SEQ_LEN, label2id)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

    logger.info(f"训练集：{len(train_dataset)}条样本，{len(train_loader)}个批次")
    logger.info(f"验证集：{len(dev_dataset)}条样本，{len(dev_loader)}个批次")
    logger.info(f"测试集：{len(test_dataset)}条样本，{len(test_loader)}个批次")

    # 5. 初始化模型
    logger.info(f"\n4. 初始化RoBERTa模型...")
    model = RoBERTaOnlyNER(
        roberta_model_name=ROBERTA_MODEL_NAME,
        num_labels=NUM_LABELS,
        dropout=DROPOUT
    ).to(DEVICE)
    logger.info(f"✅ 模型初始化完成")
    logger.info(f"总参数数：{sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"可训练参数数：{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 6. 配置优化器和学习率调度器
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01  # L2正则化，防止过拟合
    )

    total_steps = len(train_loader) * EPOCHS
    warmup_steps = int(total_steps * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
    )
    logger.info(f"✅ 优化器配置完成：")
    logger.info(f"  - 学习率：{LEARNING_RATE}")
    logger.info(f"  - 预热步数：{warmup_steps}/{total_steps}")
    logger.info(f"  - 早停耐心值：{PATIENCE}")

    # 7. 训练过程（带早停）
    logger.info(f"\n5. 开始训练（共{EPOCHS}轮）...")
    train_losses = []  # 记录每轮训练损失
    dev_f1s = []       # 记录每轮验证集F1
    best_f1 = 0.0      # 最优验证集F1
    patience_counter = 0  # 早停计数器

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0

        for step, batch in enumerate(train_loader, 1):
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            label_ids = batch["label_ids"].to(DEVICE)

            # 前向传播+反向传播
            optimizer.zero_grad()
            loss = model(input_ids=input_ids, attention_mask=attention_mask, label_ids=label_ids)
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # 梯度裁剪，避免爆炸
            optimizer.step()
            scheduler.step()

            # 打印训练进度
            if step % VERBOSE_STEP == 0:
                avg_loss = total_loss / step
                current_lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Epoch [{epoch}/{EPOCHS}], Step [{step}/{len(train_loader)}], "
                    f"Loss: {avg_loss:.4f}, LR: {current_lr:.6f}"
                )

        # 记录每轮平均损失
        avg_epoch_loss = total_loss / len(train_loader)
        train_losses.append(avg_epoch_loss)
        logger.info(f"\nEpoch [{epoch}/{EPOCHS}] 训练结束，平均损失：{avg_epoch_loss:.4f}")

        # 验证集评估
        logger.info(f"\nEpoch [{epoch}/{EPOCHS}] 验证集评估...")
        dev_f1 = evaluate(model, dev_loader, id2label, label2id)
        dev_f1s.append(dev_f1)

        # 保存最优模型
        if dev_f1 > best_f1:
            best_f1 = dev_f1
            patience_counter = 0
            # 保存模型权重、标签映射、训练记录
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "label2id": label2id,
                "id2label": id2label,
                "epoch": epoch,
                "best_f1": best_f1,
                "train_losses": train_losses,
                "dev_f1s": dev_f1s
            }, os.path.join(MODEL_SAVE_DIR, "best_roberta_ner_model.pt"))
            logger.info(f"✅ 保存最优模型（验证集F1: {best_f1:.4f}）")
        else:
            patience_counter += 1
            logger.info(f"⚠️  验证集F1未提升，早停计数器：{patience_counter}/{PATIENCE}")
            if patience_counter >= PATIENCE:
                logger.info("❌ 早停触发，停止训练")
                break

    # 8. 生成训练过程可视化图表
    logger.info(f"\n6. 生成训练可视化图表...")
    plot_training_metrics(
        train_losses=train_losses,
        dev_f1s=dev_f1s,
        save_path=os.path.join(VISUALIZATION_DIR, "roberta_training_curve.png")
    )

    # 9. 测试集最终评估（基于最优模型）
    logger.info(f"\n7. 测试集最终评估...")
    checkpoint = torch.load(os.path.join(MODEL_SAVE_DIR, "best_roberta_ner_model.pt"), map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    # 详细评估（返回真实标签、预测标签、分类报告）
    test_f1, test_true, test_pred, test_report, overall_metrics = evaluate(
        model, test_loader, id2label, label2id, return_details=True
    )

    logger.info(f"\n===== 测试集最终结果 =====")
    logger.info(f"最优模型训练轮数：{checkpoint['epoch']}")
    logger.info(f"最优验证集F1：{checkpoint['best_f1']:.4f}")
    logger.info(f"测试集Micro-F1：{test_f1:.4f}")
    logger.info("=" * 50)

    # 10. 生成测试集可视化图表
    logger.info(f"\n8. 生成测试集可视化图表...")
    # （1）各类别+整体性能对比图
    plot_classification_report(
        class_report=test_report,
        overall_metrics=overall_metrics,
        entity_labels=entity_labels,
        save_path=os.path.join(VISUALIZATION_DIR, "roberta_class_overall_perf.png")
    )
    # （2）类别性能热力图
    plot_category_heatmap(
        report=test_report,
        entity_labels=entity_labels,
        save_path=os.path.join(VISUALIZATION_DIR, "roberta_category_heatmap.png")
    )
    # （3）混淆矩阵
    plot_confusion_matrix(
        true_labels=test_true,
        pred_labels=test_pred,
        entity_labels=entity_labels,
        save_path=os.path.join(VISUALIZATION_DIR, "roberta_confusion_matrix.png")
    )

    logger.info("\n" + "=" * 60)
    logger.info("RoBERTa实体识别模型训练全部完成！")
    logger.info(f"📁 模型保存目录：{MODEL_SAVE_DIR}")
    logger.info(f"📁 可视化图表目录：{VISUALIZATION_DIR}")
    logger.info(f"📁 训练日志目录：{LOG_DIR}")
    logger.info("生成的可视化图表：")
    logger.info("1. roberta_training_curve.png → 训练损失+验证集F1曲线")
    logger.info("2. roberta_class_overall_perf.png → 各类别+整体性能对比")
    logger.info("3. roberta_category_heatmap.png → 类别性能热力图")
    logger.info("4. roberta_confusion_matrix.png → 实体识别混淆矩阵")
    logger.info("=" * 60)


if __name__ == "__main__":
    # 强制使用国内镜像，避免模型下载失败
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    # 启动训练
    train_roberta_ner()
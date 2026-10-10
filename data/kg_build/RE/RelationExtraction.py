import os
import random  # 新增：导入random模块，解决shuffle报错

# 镜像源配置（加速模型下载）
os.environ.update({
    "HF_ENDPOINT": "https://hf-mirror.com",
    "TRANSFORMERS_CACHE": "./bert_cache",
    "HF_HUB_DISABLE_SYMLINKS_WARNING": "1"
})

import warnings
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizerFast, BertModel
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.metrics import f1_score  # 新增：导入f1_score手动计算
import numpy as np
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib import font_manager

# ========== 基础配置 ==========
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 中文显示配置
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 训练参数配置
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BERT_MODEL_NAME = "bert-base-chinese"
MAX_SEQ_LEN = 128  # 适配短文本（可根据实际调整）
BATCH_SIZE = 32  # 显存充足时增大
LEARNING_RATE = 1e-5
EPOCHS = 15
WARMUP_RATIO = 0.05
WEIGHT_DECAY = 1e-4
DROPOUT_RATE = 0.2

# 目录配置
LOG_DIR = "./re_train_logs"
MODEL_SAVE_DIR = "./best_re_model"
VISUAL_SAVE_DIR = os.path.join(LOG_DIR, "visualizations")
DATA_DIR = "./re_data_generate"  # 数据目录（与生成数据的目录一致）
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
os.makedirs(VISUAL_SAVE_DIR, exist_ok=True)

# 日志配置
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(LOG_DIR, "re_train.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# ========== 工具函数：手动计算micro F1 ==========
def calculate_micro_f1(true_labels, pred_labels, num_classes):
    """
    手动计算micro F1分数，避免依赖classification_report的返回格式
    :param true_labels: 真实标签列表
    :param pred_labels: 预测标签列表
    :param num_classes: 类别总数
    :return: micro F1分数
    """
    if len(true_labels) == 0 or len(pred_labels) == 0:
        return 0.0
    # 计算每个类别的TP、FP、FN
    tp = np.zeros(num_classes)
    fp = np.zeros(num_classes)
    fn = np.zeros(num_classes)

    for true, pred in zip(true_labels, pred_labels):
        if true == pred:
            tp[true] += 1
        else:
            fp[pred] += 1
            fn[true] += 1

    # 计算micro F1
    total_tp = tp.sum()
    total_fp = fp.sum()
    total_fn = fn.sum()

    if total_tp == 0:
        return 0.0
    precision = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)
    micro_f1 = 2 * precision * recall / (precision + recall)
    return micro_f1


# ========== 可视化工具类 ==========
# ========== 可视化工具类 ==========
class Visualizer:
    def __init__(self, save_dir, id2rel):
        self.save_dir = save_dir
        self.id2rel = id2rel
        self.train_losses = []  # 每个Epoch的平均损失
        self.train_accs = []  # 每个Epoch的训练准确率
        self.train_f1s = []  # 每个Epoch的训练F1
        self.dev_accs = []  # 每个Epoch的验证准确率
        self.dev_f1s = []  # 每个Epoch的验证F1
        self.step_losses = []  # 每5步的损失（单独绘图）

    def add_step_metric(self, loss):
        """记录每5步的损失（用于单独绘制详细损失曲线）"""
        self.step_losses.append(loss)

    def add_epoch_metrics(self, train_loss, train_acc, train_f1, dev_acc, dev_f1):
        """记录每个Epoch的平均指标（用于训练/验证对比）"""
        self.train_losses.append(train_loss)
        self.train_accs.append(train_acc)
        self.train_f1s.append(train_f1)
        self.dev_accs.append(dev_acc)
        self.dev_f1s.append(dev_f1)

    def plot_step_loss_curve(self):
        """绘制每5步的详细损失曲线"""
        plt.figure(figsize=(10, 5))
        steps = [i * 5 + 5 for i in range(len(self.step_losses))]  # 步数标签（5,10,15...）
        plt.plot(steps, self.step_losses, label="训练损失（每5步）", color="#1f77b4")
        plt.xlabel("训练步数")
        plt.ylabel("损失值")
        plt.title("关系抽取训练损失曲线（详细）")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, "step_train_loss.png"), dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("详细训练损失曲线已保存")

    def plot_epoch_loss_curve(self):
        """绘制每个Epoch的平均损失曲线"""
        plt.figure(figsize=(10, 5))
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, label="训练平均损失", marker="o", color="#1f77b4")
        plt.xlabel("Epoch")
        plt.ylabel("平均损失值")
        plt.title("关系抽取训练平均损失曲线")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.savefig(os.path.join(self.save_dir, "epoch_train_loss.png"), dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Epoch平均损失曲线已保存")

    def plot_metrics_comparison(self):
        """绘制训练/验证指标对比曲线（维度一致）"""
        epochs = range(1, len(self.dev_accs) + 1)
        plt.figure(figsize=(12, 5))
        # 准确率对比
        plt.subplot(1, 2, 1)
        plt.plot(epochs, self.train_accs, label="训练准确率", marker="o", color="#1f77b4")
        plt.plot(epochs, self.dev_accs, label="验证准确率", marker="s", color="#ff7f0e")
        plt.xlabel("Epoch")
        plt.ylabel("准确率")
        plt.legend()
        plt.grid(alpha=0.3)
        # F1对比
        plt.subplot(1, 2, 2)
        plt.plot(epochs, self.train_f1s, label="训练F1", marker="o", color="#1f77b4")
        plt.plot(epochs, self.dev_f1s, label="验证F1", marker="s", color="#ff7f0e")
        plt.xlabel("Epoch")
        plt.ylabel("F1分数")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(self.save_dir, "metrics_comparison.png"), dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("指标对比曲线已保存")

    def plot_confusion_matrix(self, true_labels, pred_labels):
        cm = confusion_matrix(true_labels, pred_labels, labels=range(len(self.id2rel)))
        plt.figure(figsize=(12, 10))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[self.id2rel[i] for i in range(len(self.id2rel))],
            yticklabels=[self.id2rel[i] for i in range(len(self.id2rel))]
        )
        plt.xlabel("预测关系")
        plt.ylabel("真实关系")
        plt.title("关系抽取混淆矩阵（测试集）")
        plt.xticks(rotation=45, ha="right")
        plt.savefig(os.path.join(self.save_dir, "confusion_matrix.png"), dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("混淆矩阵已保存")

    # ========== 参考示例图修改后的标签指标柱状图 ==========
    def plot_label_metrics(self, true_labels, pred_labels, num_classes):
        """
        可视化每个标签的精确率、召回率、F1分数（同类别多指标并列柱状图）
        :param true_labels: 真实标签列表
        :param pred_labels: 预测标签列表
        :param num_classes: 类别总数
        """
        # 获取详细的分类报告（包含每个类别的指标）
        report_dict = classification_report(
            true_labels, pred_labels,
            labels=range(num_classes),
            target_names=[self.id2rel[i] for i in range(num_classes)],
            zero_division=0,
            output_dict=True
        )

        # 提取每个类别的指标
        labels = []
        precisions = []  # 精确率（对应示例图的Precision）
        recalls = []  # 召回率（对应示例图的Recall）
        f1_scores = []  # F1分数（对应示例图的F1-Score）

        for i in range(num_classes):
            label_name = self.id2rel[i]
            labels.append(label_name)
            precisions.append(report_dict[label_name]["precision"])
            recalls.append(report_dict[label_name]["recall"])
            f1_scores.append(report_dict[label_name]["f1-score"])

        # 增加平均值行（参考示例图的Average）
        labels.append("Average")
        precisions.append(report_dict["macro avg"]["precision"])
        recalls.append(report_dict["macro avg"]["recall"])
        f1_scores.append(report_dict["macro avg"]["f1-score"])

        # 创建柱状图（同类别下3个指标并列）
        plt.figure(figsize=(16, 8))
        x = np.arange(len(labels))  # 标签位置
        width = 0.25  # 每个指标柱子的宽度

        # 绘制三个指标的并列柱状图
        # 1. 精确率（黄色，对应示例图的Precision）
        plt.bar(x - width, precisions, width, label="Precision", color="#f1c40f", edgecolor="black")
        # 2. 召回率（蓝色，对应示例图的Recall）
        plt.bar(x, recalls, width, label="Recall", color="#3498db", edgecolor="black")
        # 3. F1分数（红色，对应示例图的F1-Score）
        plt.bar(x + width, f1_scores, width, label="F1-Score", color="#e74c3c", edgecolor="black")

        # 添加数值标签
        for i in range(len(x)):
            # 精确率标签
            plt.text(x[i] - width, precisions[i] + 0.01,
                     f"{precisions[i]:.2f}", ha="center", va="bottom", fontsize=9)
            # 召回率标签
            plt.text(x[i], recalls[i] + 0.01,
                     f"{recalls[i]:.2f}", ha="center", va="bottom", fontsize=9)
            # F1分数标签
            plt.text(x[i] + width, f1_scores[i] + 0.01,
                     f"{f1_scores[i]:.2f}", ha="center", va="bottom", fontsize=9)

        # 图表配置
        plt.xlabel("关系标签", fontsize=12)
        plt.ylabel("指标值（0-1）", fontsize=12)
        plt.title("BERT模型关系抽取性能对比", fontsize=15, pad=20)
        plt.xticks(x, labels, rotation=45, ha="right", fontsize=10)
        plt.ylim(0, 1.1)  # y轴范围0-1.1，预留标签空间
        plt.grid(axis="y", alpha=0.3, linestyle="--")
        plt.legend(fontsize=11, loc="upper right")
        plt.tight_layout()  # 自动调整布局，防止标签重叠

        # 保存图片
        plt.savefig(os.path.join(self.save_dir, "label_metrics_bar.png"), dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("各标签指标柱状图已保存")


# ========== 数据集类 ==========
class REDataset(Dataset):
    def __init__(self, data_path, tokenizer, rel2id, max_seq_len):
        self.data = self.load_data(data_path)
        self.tokenizer = tokenizer
        self.rel2id = rel2id
        self.max_seq_len = max_seq_len

    def load_data(self, data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # 过滤无效样本
        valid_data = [x for x in data if
                      all(k in x and x[k].strip() for k in ["text", "subject", "object", "relation"])]
        logger.info(f"加载数据：{data_path}，有效样本数：{len(valid_data)}/{len(data)}")
        return valid_data

    def _locate_entity(self, text, entity, offsets):
        """定位实体在token序列中的位置"""
        entity_len = len(entity)
        for i, (start, end) in enumerate(offsets):
            if text[start:end] == entity:
                return (i, i + len(self.tokenizer.tokenize(entity)) - 1)
        # 模糊匹配（实体可能被分词拆分）
        entity_tokens = self.tokenizer.tokenize(entity)
        token_len = len(entity_tokens)
        for i in range(len(offsets) - token_len + 1):
            if offsets[i][0] >= offsets[i + token_len - 1][1]:
                continue
            if self.tokenizer.convert_tokens_to_string(
                    self.tokenizer.convert_ids_to_tokens(
                        self.tokenizer(text)["input_ids"][i + 1:i + token_len + 1]  # 跳过CLS
                    )
            ).strip() == entity:
                return (i + 1, i + token_len)  # +1 跳过CLS
        #  fallback：返回文本中间位置
        mid = len(offsets) // 2
        return (mid, mid)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        text = sample["text"]
        subject = sample["subject"]
        object_ = sample["object"]
        relation = sample["relation"]

        # 文本编码
        encoding = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_seq_len,
            return_offsets_mapping=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        offsets = encoding["offset_mapping"].squeeze(0).tolist()

        # 定位实体位置
        sub_pos = self._locate_entity(text, subject, offsets)
        obj_pos = self._locate_entity(text, object_, offsets)

        # 转换为tensor
        sub_pos = torch.tensor(sub_pos, dtype=torch.long)
        obj_pos = torch.tensor(obj_pos, dtype=torch.long)
        rel_id = torch.tensor(self.rel2id[relation], dtype=torch.long)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "sub_pos": sub_pos,
            "obj_pos": obj_pos,
            "rel_id": rel_id
        }


# ========== 模型定义 ==========
class RelationModel(nn.Module):
    def __init__(self, bert_name, num_relations, dropout_rate=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(bert_name, output_hidden_states=True)
        # 冻结前6层BERT
        for param in list(self.bert.parameters())[:-12]:
            param.requires_grad = False
        self.hidden_size = self.bert.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)

        # 实体交互层
        self.entity_interact = nn.Sequential(
            nn.Linear(self.hidden_size * 2, self.hidden_size),
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )

        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_size * 3, self.hidden_size),  # CLS + 主体 + 客体（含交互）
            nn.LayerNorm(self.hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(self.hidden_size, num_relations)
        )

    def _get_entity_feat(self, hidden, pos):
        """提取实体特征（平均池化）"""
        batch_size = hidden.shape[0]
        feats = []
        for i in range(batch_size):
            start, end = pos[i, 0].item(), pos[i, 1].item()
            start = max(0, min(start, hidden.shape[1] - 1))
            end = max(start, min(end, hidden.shape[1] - 1))
            feat = hidden[i, start:end + 1, :].mean(dim=0)
            feats.append(feat)
        return torch.stack(feats, dim=0)

    def forward(self, input_ids, attention_mask, sub_pos, obj_pos):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state  # (batch, seq_len, hidden)

        # 提取特征
        cls_feat = last_hidden[:, 0, :]  # CLS特征
        sub_feat = self._get_entity_feat(last_hidden, sub_pos)  # 主体特征
        obj_feat = self._get_entity_feat(last_hidden, obj_pos)  # 客体特征

        # 实体交互
        interact_feat = self.entity_interact(torch.cat([sub_feat, obj_feat], dim=-1))
        obj_feat = obj_feat + interact_feat  # 融合交互信息

        # 拼接分类
        combined_feat = torch.cat([cls_feat, sub_feat, obj_feat], dim=-1)
        combined_feat = self.dropout(combined_feat)
        logits = self.classifier(combined_feat)

        return logits


# ========== 数据平衡工具 ==========
def compute_class_weights(data, rel2id):
    """计算类别权重（解决不平衡）"""
    rel_counts = {}
    for sample in data:
        rel = sample["relation"]
        rel_counts[rel] = rel_counts.get(rel, 0) + 1
    total = len(data)
    weights = []
    for rel in rel2id.keys():
        count = rel_counts.get(rel, 1)
        weights.append(total / (len(rel2id) * count))
    return torch.tensor(weights, dtype=torch.float32).to(DEVICE)


def oversample_minority(data, rel2id):
    """过采样少数类"""
    rel_groups = {rel: [] for rel in rel2id.keys()}
    for sample in data:
        rel_groups[sample["relation"]].append(sample)
    max_count = max(len(group) for group in rel_groups.values())
    oversampled = []
    for rel, group in rel_groups.items():
        if len(group) == 0:
            continue
        # 重复采样至最大数量
        oversampled.extend(group * (max_count // len(group)) + group[:max_count % len(group)])
    random.shuffle(oversampled)  # 现在random已导入，可正常使用
    logger.info(f"过采样后样本数：{len(oversampled)}（原{len(data)}）")
    return oversampled


# ========== 评估函数 ==========
def evaluate(model, dataloader, id2rel, num_relations):
    model.eval()
    all_true = []
    all_pred = []
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            sub_pos = batch["sub_pos"].to(DEVICE)
            obj_pos = batch["obj_pos"].to(DEVICE)
            rel_id = batch["rel_id"].to(DEVICE)

            logits = model(input_ids, attention_mask, sub_pos, obj_pos)
            pred = torch.argmax(logits, dim=-1).cpu().numpy()
            true = rel_id.cpu().numpy()

            all_true.extend(true)
            all_pred.extend(pred)

    # 计算指标
    acc = accuracy_score(all_true, all_pred)
    # 手动计算micro F1（避免依赖classification_report）
    f1 = calculate_micro_f1(all_true, all_pred, num_relations)
    # 生成报告（仅用于日志展示）
    report = classification_report(
        all_true, all_pred,
        labels=range(num_relations),
        target_names=[id2rel[i] for i in range(num_relations)],
        zero_division=0
    )
    return acc, f1, report, all_true, all_pred


# ========== 训练主函数 ==========
def train():
    # 1. 加载分词器和数据
    tokenizer = BertTokenizerFast.from_pretrained(BERT_MODEL_NAME)

    # 构建关系映射
    all_data = []
    for split in ["train", "dev", "test"]:
        with open(os.path.join(DATA_DIR, f"{split}.json"), "r", encoding="utf-8") as f:
            all_data.extend(json.load(f))
    relations = sorted(list({x["relation"] for x in all_data}))
    rel2id = {rel: i for i, rel in enumerate(relations)}
    id2rel = {i: rel for rel, i in rel2id.items()}
    num_relations = len(relations)
    logger.info(f"关系类型：{relations}（共{num_relations}种）")
    json.dump(rel2id, open(os.path.join(MODEL_SAVE_DIR, "rel2id.json"), "w", encoding="utf-8"), ensure_ascii=False)

    # 2. 构建数据集
    train_dataset = REDataset(os.path.join(DATA_DIR, "train.json"), tokenizer, rel2id, MAX_SEQ_LEN)
    dev_dataset = REDataset(os.path.join(DATA_DIR, "dev.json"), tokenizer, rel2id, MAX_SEQ_LEN)
    test_dataset = REDataset(os.path.join(DATA_DIR, "test.json"), tokenizer, rel2id, MAX_SEQ_LEN)

    # 过采样少数类（仅训练集）
    train_dataset.data = oversample_minority(train_dataset.data, rel2id)

    # 构建DataLoader
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=0, pin_memory=True
    )

    # 3. 初始化模型、优化器、损失函数
    model = RelationModel(BERT_MODEL_NAME, num_relations, DROPOUT_RATE).to(DEVICE)
    logger.info(f"模型总参数：{sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"可训练参数：{sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    optimizer = optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY, eps=1e-8
    )
    total_steps = len(train_loader) * EPOCHS
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=1e-7)
    class_weights = compute_class_weights(train_dataset.data, rel2id)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 4. 早停机制和可视化工具
    early_stop_patience = 3
    early_stop_counter = 0
    best_dev_f1 = 0.0
    visualizer = Visualizer(VISUAL_SAVE_DIR, id2rel)

    # 5. 训练循环
    logger.info("开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        train_true = []
        train_pred = []

        for step, batch in enumerate(train_loader):
            # 加载数据
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)
            sub_pos = batch["sub_pos"].to(DEVICE)
            obj_pos = batch["obj_pos"].to(DEVICE)
            rel_id = batch["rel_id"].to(DEVICE)

            # 前向传播
            logits = model(input_ids, attention_mask, sub_pos, obj_pos)
            loss = criterion(logits, rel_id)

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            # 记录损失和预测结果
            total_loss += loss.item()
            pred = torch.argmax(logits, dim=-1).cpu().numpy()
            train_true.extend(rel_id.cpu().numpy())
            train_pred.extend(pred)

            # 每5步记录详细损失（用于单独绘图）和日志输出
            if (step + 1) % 5 == 0:
                avg_step_loss = total_loss / (step + 1)
                step_acc = accuracy_score(train_true, train_pred)
                step_f1 = calculate_micro_f1(train_true, train_pred, num_relations)
                visualizer.add_step_metric(avg_step_loss)  # 记录每5步的损失
                logger.info(
                    f"Epoch [{epoch + 1}/{EPOCHS}], Step [{step + 1}/{len(train_loader)}], "
                    f"Loss: {avg_step_loss:.4f}, Acc: {step_acc:.4f}, F1: {step_f1:.4f}"
                )

        # Epoch结束后计算训练集平均指标
        epoch_train_loss = total_loss / len(train_loader)
        epoch_train_acc = accuracy_score(train_true, train_pred)
        epoch_train_f1 = calculate_micro_f1(train_true, train_pred, num_relations)

        # 验证集评估
        dev_acc, dev_f1, dev_report, _, _ = evaluate(model, dev_loader, id2rel, num_relations)
        logger.info(f"\nEpoch [{epoch + 1}/{EPOCHS}] 训练集平均指标：")
        logger.info(f"Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:.4f}, F1: {epoch_train_f1:.4f}")
        logger.info(f"Epoch [{epoch + 1}/{EPOCHS}] 验证集结果：")
        logger.info(f"Accuracy: {dev_acc:.4f}, F1: {dev_f1:.4f}")
        logger.info(dev_report)

        # 记录当前Epoch的训练/验证指标（维度一致）
        visualizer.add_epoch_metrics(epoch_train_loss, epoch_train_acc, epoch_train_f1, dev_acc, dev_f1)

        # 保存最优模型
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "rel2id": rel2id
            }, os.path.join(MODEL_SAVE_DIR, "best_model.pt"))
            logger.info(f"✅ 保存最优模型（F1: {best_dev_f1:.4f}）")
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            logger.info(f"⚠️  早停计数器：{early_stop_counter}/{early_stop_patience}")
            if early_stop_counter >= early_stop_patience:
                logger.info("❌ 早停触发，停止训练")
                break

    # 6. 可视化训练结果（修复维度不匹配问题）
    visualizer.plot_step_loss_curve()  # 每5步详细损失曲线
    visualizer.plot_epoch_loss_curve()  # Epoch平均损失曲线
    visualizer.plot_metrics_comparison()  # 训练/验证指标对比（维度一致）

    # 7. 测试集最终评估
    logger.info("\n===== 测试集最终评估 =====")
    checkpoint = torch.load(os.path.join(MODEL_SAVE_DIR, "best_model.pt"), map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_acc, test_f1, test_report, test_true, test_pred = evaluate(model, test_loader, id2rel, num_relations)
    logger.info(f"测试集 Accuracy: {test_acc:.4f}, F1: {test_f1:.4f}")
    logger.info(test_report)

    # 绘制测试集混淆矩阵和标签指标柱状图
    visualizer.plot_confusion_matrix(test_true, test_pred)
    visualizer.plot_label_metrics(test_true, test_pred, num_relations)

    # 保存测试报告
    with open(os.path.join(LOG_DIR, "test_report.txt"), "w", encoding="utf-8") as f:
        f.write(f"测试集最终结果：\nAccuracy: {test_acc:.4f}\nF1: {test_f1:.4f}\n\n")
        f.write(test_report)
    logger.info("✅ 测试报告已保存")


if __name__ == "__main__":
    train()
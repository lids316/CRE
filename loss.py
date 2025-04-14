from typing import Any

import torch
from torch import nn
import torch.nn.functional as F


class SupConLoss(nn.Module):
    """监督对比损失，增强特征空间的区分度

    用于对比学习任务，目标是通过最小化同类样本之间的距离，
    并最大化不同类样本之间的距离来增强特征表示的区分度

    Attributes:
        temperature (float): 温度参数，控制相似度的放大/缩小，通常取较小的值，譬如0.07
    """

    def __init__(self, config):
        """初始化损失函数的参数
        """
        super().__init__()
        self.temperature = config.temperature  # 设置温度系数

    def forward(self, features, labels):
        """计算对比损失

        Args:
            features (Tensor): 特征向量，形状为 (batch_size, feature_dim)，
                                通常是神经网络的输出特征，已经经过归一化
            labels (Tensor): 每个样本的标签，形状为 (batch_size,)，
                             用于确定同类样本对

        Returns:
            Tensor: 计算得到的对比损失
        """
        device = features.device  # 获取输入数据所在的设备
        features = F.normalize(features, p=2, dim=1)  # 归一化特征向量，按行归一化，L2范数
        batch_size = features.shape[0]  # 获取批次大小

        # 构建标签mask，确定哪些样本属于同一类别
        labels = labels.view(-1, 1)  # 将标签转换为列向量，形状为 (batch_size, 1)
        mask = torch.eq(labels, labels.T).float().to(device)  # 计算标签相等的掩码，1表示同类，0表示不同类

        # 计算相似度矩阵：通过矩阵乘法计算每对样本之间的余弦相似度
        similarity = torch.matmul(features, features.T) / self.temperature  # 使用温度系数进行缩放

        # 排除自身对比，构建一个掩码，确保不会把每个样本与自身进行对比
        logits_mask = torch.ones_like(mask) - torch.eye(batch_size, device=device)  # 构建单位矩阵掩码
        mask = mask * logits_mask  # 将掩码与 logits_mask 相乘，排除对角线元素（即自身）

        # 计算对比损失
        exp_logits = torch.exp(similarity) * logits_mask  # 计算每对样本的指数相似度
        log_prob = similarity - torch.log(exp_logits.sum(dim=1, keepdim=True))  # 计算每个样本的对数概率
        mean_log_prob = (mask * log_prob).sum(1) / mask.sum(1)  # 计算同类样本的平均对数概率
        return -mean_log_prob.mean()  # 返回损失值，取负平均值


def generate_adversarial_example(config, value):
    """
    生成对抗样本，通过添加噪声来进行攻击。

    Args:
        config: 配置对象，包含噪声的参数。
        value: 输入的张量，表示模型的输入数据。

    Returns:
        x: 对抗样本，原始输入 `x` 添加噪声后的结果。
    """
    value = value.float()  # 将输入转换为浮点型
    noise = torch.randn_like(value)  # 生成与输入形状相同的标准正态噪声

    # 计算噪声的范数（按特定维度归一化）
    norm = torch.norm(noise, p=config.noise_p_norm, dim=list(range(1, value.dim())), keepdim=True)

    # 归一化噪声并根据噪声缩放因子调整
    noise = noise / norm * config.noise_scaling_factor

    # 将噪声加到输入上，生成对抗样本
    value = value + noise

    # 返回转换为长整型的对抗样本
    return value.long()


class AdversarialLoss(nn.Module):
    """
    AdversarialLoss 类实现了针对对抗训练的损失函数。
    该损失函数包含三个组成部分，用于优化模型在对抗样本和干净样本上的表现。
    """

    def __init__(self, config: Any):
        """
        初始化 AdversarialLoss 对象，设置相关的损失权重。

        Args:
            config: 配置对象，包含损失函数的权重和参数。
        """
        super(AdversarialLoss, self).__init__()
        self.margin_power = config.margin_power  # margin_power 参数用于调整 p-margin 损失中的比例
        self.robust_weight = config.robust_weight  # 对抗稳健性损失的权重
        self.margin_weight = config.margin_weight  # 对抗竞争求和 margin 损失的权重
        self.p_margin_weight = config.p_margin_weight  # p-margin 损失的权重

    @staticmethod
    def adversarial_robustness_loss(logits, labels):
        """
        计算对抗稳健性损失，使用交叉熵损失函数。

        Args:
            logits: 模型对抗样本的输出（未经激活的预测值）。
            labels: 真实标签。

        Returns:
            loss: 对抗稳健性损失。
        """
        return F.cross_entropy(logits, labels)  # 计算交叉熵损失

    @staticmethod
    def adversarial_comp_sum_p_margin_loss(logits, labels):
        """
        计算对抗竞争求和 p-margin 损失。目的是增强正确类别的对抗样本与最大类别之间的边距。

        Args:
            logits: 模型对抗样本的输出（未经激活的预测值）。
            labels: 真实标签。

        Returns:
            loss: 对抗竞争求和 p-margin 损失。
        """
        correct_class_logit = logits.gather(1, labels.unsqueeze(1)).squeeze(1)  # 获取正确类别的logit值
        max_logit, _ = logits.max(dim=1)  # 获取每个样本的最大 logit 值
        margin = 1.0 - (max_logit - correct_class_logit)  # 计算边距
        loss = torch.clamp(margin, min=0).mean()  # 对负边距进行截断并计算均值
        return loss

    def p_margin_loss(self, logits, labels):
        """
        计算 p-margin 损失。通过调整边距来增加模型的区分能力。

        Args:
            logits: 模型输出的对抗样本的 logits（未经激活的预测值）。
            labels: 真实标签。

        Returns:
            loss: p-margin 损失。
        """
        correct_class_logit = logits.gather(1, labels.unsqueeze(1)).squeeze(1)  # 获取正确类别的logit值
        max_logit, _ = logits.max(dim=1)  # 获取每个样本的最大 logit 值
        margin = 1.0 - (max_logit - correct_class_logit) / (self.margin_power - 1)  # 计算 p-margin
        loss = torch.clamp(margin, min=0).mean()  # 对负边距进行截断并计算均值
        return loss

    def forward(self, logits_adv, logits_clean, labels):
        """
        计算 AdversarialLoss，包含三个损失部分的加权和：
        1. 对抗稳健性损失（CrossEntropy损失）。
        2. 对抗竞争求和 p-margin 损失。
        3. p-margin 损失。

        Args:
            logits_adv: 模型在对抗样本上的输出。
            logits_clean: 模型在干净样本上的输出。
            labels: 真实标签。

        Returns:
            total_loss: 总损失，由三个损失组成的加权和。
        """
        # 计算每个损失
        loss1 = self.adversarial_robustness_loss(logits_adv, labels)  # 对抗稳健性损失
        loss2 = self.adversarial_comp_sum_p_margin_loss(logits_adv, labels)  # 对抗竞争求和 p-margin 损失
        loss3 = self.p_margin_loss(logits_clean, labels)  # p-margin 损失

        # 返回加权和的总损失
        total_loss = self.robust_weight * loss1 + self.margin_weight * loss2 + self.p_margin_weight * loss3
        return total_loss

def combined_loss(task_id: int, anchor: torch.Tensor, positive: torch.Tensor, negatives: torch.Tensor,
                  previous_logits: torch.Tensor, current_logits: torch.Tensor, labels: torch.Tensor,
                  alpha: float = 0.5, beta: float = 0.5, temperature: float = 2.0,
                  contrastive_temperature: float = 0.5) -> torch.Tensor:
    """
    计算综合损失，包括自监督对比损失和焦点知识蒸馏损失。

    自监督对比损失部分基于正负样本对的余弦相似度，焦点知识蒸馏部分则根据模型的输出和目标模型的输出计算KL散度。

    Args:
        task_id (int): 当前任务的 ID，决定是否计算蒸馏损失。
        anchor (torch.Tensor): 锚点样本特征，形状为 (batch_size, feature_dim)。
        positive (torch.Tensor): 正样本特征，形状为 (batch_size, feature_dim)。
        negatives (torch.Tensor): 负样本特征，形状为 (batch_size, num_negatives, feature_dim)。
        previous_logits (torch.Tensor): 先前模型的输出 logits，形状为 (batch_size, num_classes)。
        current_logits (torch.Tensor): 当前模型的输出 logits，形状为 (batch_size, num_classes)。
        labels (torch.Tensor): 真实标签，形状为 (batch_size,)。
        alpha (float, optional): 蒸馏损失中焦点损失和交叉熵损失的权重，默认值为 0.5。
        beta (float, optional): 对比损失和蒸馏损失的权重，默认值为 0.5。
        temperature (float, optional): 温度系数，用于计算蒸馏损失时的平滑度，默认值为 2.0。
        contrastive_temperature (float, optional): 对比损失中的温度系数，默认值为 0.5。

    Returns:
        torch.Tensor: 计算得到的总损失。
    """

    # 获取批量大小、负样本数量和特征维度
    batch_size, num_negatives, feature_dim = negatives.shape

    # 计算锚点和正样本之间的相似度（余弦相似度）
    positive_similarity = F.cosine_similarity(anchor, positive, dim=-1)
    positive_similarity = positive_similarity.unsqueeze(1)  # 增加维度以便广播

    # 计算锚点与每个负样本之间的相似度
    anchor_expanded = anchor.unsqueeze(1).expand(-1, num_negatives, -1)  # 扩展锚点维度以匹配负样本数量
    negative_similarities = F.cosine_similarity(anchor_expanded, negatives, dim=-1)

    # 合并正样本和负样本的相似度，并计算概率（使用对比温度）
    all_similarities = torch.cat([positive_similarity, negative_similarities], dim=1)
    probabilities = F.log_softmax(all_similarities / contrastive_temperature, dim=1)

    # 对比损失是第一个元素的负对数概率（锚点与正样本）
    contrastive_loss = -probabilities[:, 0].mean()

    if task_id > 0:
        # 计算焦点知识蒸馏损失（Knowledge Distillation Loss）
        soft_log_probs = F.log_softmax(current_logits / temperature, dim=1)  # 当前模型的 softmax 输出（对数）
        soft_targets = F.softmax(previous_logits / temperature, dim=1)  # 先前模型的 softmax 输出
        kd_loss = F.kl_div(soft_log_probs, soft_targets.detach(), reduction='batchmean') * (temperature ** 2)  # KL 散度损失

        # 计算交叉熵损失
        ce_loss = F.cross_entropy(current_logits, labels)

        # 蒸馏损失是 KD 损失和交叉熵损失的加权和
        distillation_loss = alpha * kd_loss + (1 - alpha) * ce_loss
    else:
        # 如果没有任务需要计算蒸馏损失
        distillation_loss = 0

    # 结合对比损失和蒸馏损失
    combined_total_loss = beta * contrastive_loss + (1 - beta) * distillation_loss

    return combined_total_loss

import random
from typing import Any
import numpy as np
from sklearn.cluster import KMeans
import torch
from torch import nn
from transformers import BertTokenizer
from typing import List
from data_loader import get_data_loader


def set_seed(seed: int) -> None:
    """
    设置随机种子以确保实验的可复现性。

    参数:
    seed (int) -- 随机种子值

    返回:
    None
    """
    # 设置 PyTorch CPU 随机种子
    torch.manual_seed(seed)

    # 设置 GPU 的随机种子，以支持多 GPU 设置
    torch.cuda.manual_seed_all(seed)

    # 设置 CUDA 的单 GPU 随机种子
    torch.cuda.manual_seed(seed)

    # 设置 NumPy 随机种子
    np.random.seed(seed)

    # 设置 Python 原生随机模块的种子
    random.seed(seed)

    # 设置 cudnn 来优化卷积算法，同时保证可复现性
    torch.backends.cudnn.benchmark = False  # 禁止 cuDNN 选择最合适的算法
    torch.backends.cudnn.deterministic = True  # 保证每次运行时算法是确定的，支持复现性


def eliminate_experiences(config, experiences:List[Any]):
    """
    从经验池中筛选高质量和频繁使用的经验，并去除重复的经验。

    通过阈值选择高质量经验，并通过使用频率选择最常使用的经验，合并这两部分经验，去重后更新经验池。

    Args:
        config: 配置对象，包含 threshold 参数，用于选择最频繁的经验的比例。
        experiences: 包含经验的对象，提供获取高质量经验和频繁使用经验的功能。

    Returns:
        experiences: 更新后的经验池对象，包含去重后的高质量和频繁使用经验。
    """
    # 获取配置中的经验筛选阈值
    threshold = config.threshold  # 用于决定频繁经验的筛选比例

    # 获取高质量经验
    high_quality_experiences = experiences.get_high_quality_experiences()

    # 使用频率计算，选择频繁使用的经验
    usage_threshold = int(len(experiences.experiences) * threshold)  # 计算要选择的经验数量
    # 按照使用频率对经验索引排序，选择前 `usage_threshold` 个频繁经验
    frequent_experiences_indices = sorted(
        range(len(experiences.usage_count)), key=lambda x: experiences.usage_count[x], reverse=True
    )[:usage_threshold]

    # 获取频繁使用的经验
    frequent_experiences = [experiences.experiences[i] for i in frequent_experiences_indices]

    # 合并高质量经验和频繁使用经验，并去重
    combined_experiences = high_quality_experiences + frequent_experiences

    # 去除重复经验，使用 set() 去重后再转换为 list
    unique_experiences = list(set(combined_experiences))

    # 更新经验池，重置使用计数
    experiences.experiences = unique_experiences
    experiences.usage_count = [0] * len(unique_experiences)  # 重置所有经验的使用计数

    return experiences


def select_data(config: Any, encoder: nn.Module, sample_set: list,
                select_num: int = None) -> list:
    """
    从样本集中选择一定数量的样本，使用 KMeans 聚类算法来选择距离聚类中心最近的样本。

    参数:
    config (object) -- 配置对象，包含模型的配置和超参数信息
    encoder (nn.Module) -- 句子编码器模型，用于生成输入样本的特征向量
    sample_set (torch.utils.data.Dataset) -- 包含样本的完整数据集
    select_num (int, optional) -- 要选择的样本数，默认为 None（此时选择记忆大小大小样本）

    返回:
    list -- 选择的样本集，包含距离聚类中心最近的样本
    """

    # 获取数据加载器，加载样本集，批次大小为 1，不打乱样本，避免丢失最后一个样本
    data_loader = get_data_loader(config, sample_set, shuffle=False, drop_last=False, batch_size=1)

    # 存储特征向量的列表
    features = []

    # 将模型设置为评估模式
    encoder.eval()

    # 遍历数据加载器中的每个批次
    for step, (_, _, tokens, _) in enumerate(data_loader):
        tokens = tokens.to(config.device)  # 将 token 转移到指定设备
        with torch.no_grad():  # 禁用梯度计算
            try:
                # 如果模型支持返回 `rel_hidden_states`，使用它
                feature = encoder(tokens).rel_hidden_states.cpu()
            except:
                # 如果不支持，则使用模型的默认输出
                feature = encoder(tokens).cpu()

        # 将计算出的特征向量加入特征列表
        features.append(feature)

    # 将特征列表拼接成一个 NumPy 数组
    features = np.concatenate(features)

    # 如果未指定选择样本数，则选择最小值：配置中的记忆大小和样本集的大小
    if select_num is None:
        num_clusters = min(config.memory_size, len(sample_set))
    else:
        num_clusters = min(select_num, len(sample_set))  # 否则选择指定的样本数

    # 使用 KMeans 聚类算法将样本分成 `num_clusters` 个簇
    distances = KMeans(n_clusters=num_clusters, random_state=0).fit_transform(features)

    # 初始化存储选中样本的列表
    selected_samples = []

    # 对每个簇，选择距离该簇中心最近的样本
    for k in range(num_clusters):
        # 找到距离簇中心最近的样本索引
        sel_index = np.argmin(distances[:, k])
        # 根据索引从样本集中选择样本
        instance = sample_set[sel_index]
        # 将选中的样本加入列表
        selected_samples.append(instance)

    # 返回选中的样本集
    return selected_samples


def get_proto(config: Any, encoder: nn.Module, memory_set: list,
              reference_vector: torch.Tensor = None) -> torch.Tensor:
    """
    获取原型向量。通过计算给定记忆集的特征向量的平均值，得到原型向量。

    参数:
    config (object) -- 配置对象，包含模型的配置和超参数信息
    encoder (nn.Module) -- 句子编码器模型，用于生成输入的特征向量
    memory_set (torch.utils.data.Dataset) -- 存储记忆样本的数据集
    reference_vector (torch.Tensor, optional) -- 一个参考向量，如果提供，将与特征向量一起计算

    返回:
    torch.Tensor -- 计算得到的原型向量，形状为 (1, hidden_size)
    """

    # 获取数据加载器，用于加载记忆集的样本
    data_loader = get_data_loader(config, memory_set, shuffle=False, drop_last=False, batch_size=1)

    # 存储特征向量的列表
    features = []

    # 获取编码器的输出维度（假设encoder输出是固定维度的）
    encoder_output_dim = config.encoder_output_size  # 根据你的模型配置定义

    # 将模型设置为评估模式
    encoder.eval()

    # 遍历数据加载器中的每个批次
    for _, (_, _, tokens, _) in enumerate(data_loader):
        # 将输入 token 转移到指定设备
        tokens = torch.stack([x.to(config.device) for x in tokens], dim=0)

        # 使用 `torch.no_grad()` 禁用梯度计算
        with torch.no_grad():
            # 获取特征向量
            feature = encoder(tokens)

        # 将得到的特征向量加入列表
        features.append(feature)

    # 如果提供了参考向量，将其添加到特征列表中并进行归一化
    if reference_vector is not None:
        features.append(reference_vector.unsqueeze(0))  # 将参考向量添加到特征列表
        features = [x / x.norm() for x in features]  # 对每个特征向量进行L2归一化

    # 将所有特征向量拼接在一起
    features = torch.cat(features, dim=0)

    # 计算特征向量的平均值作为原型向量
    proto = torch.mean(features, dim=0, keepdim=True)

    return proto  # 返回计算得到的原型向量


def get_aca_data(config: Any, training_data: dict, current_relations: list, tokenizer: BertTokenizer) -> list:
    """
    从训练数据中生成 ACA 数据。该函数通过给定的 `current_relations` 和训练数据生成特定格式的输入样本，适用于增量学习场景。

    参数:
    config (object) -- 配置对象，包含模型的配置和超参数信息
    training_data (dict) -- 包含关系和对应样本的训练数据，格式为 {relation: [samples]}
    current_relations (list) -- 当前任务中涉及的关系列表
    tokenizer (object) -- 用于将 token 转换为字符串的分词器对象

    返回:
    list -- 生成的 ACA 数据列表，每个元素是一个包含 'tokens' 和 'string' 的字典
    """

    relation_id = config.num_of_relations  # 获取初始关系 ID
    aca_data = []  # 存储生成的 ACA 数据

    # 生成对比样本数据
    for relation_1, relation_2 in zip(current_relations[:config.relations_per_task // 2],
                                      current_relations[config.relations_per_task // 2:]):
        samples_relation_1 = training_data[relation_1]  # 获取关系 relation_1 的数据
        samples_relation_2 = training_data[relation_2]  # 获取关系 relation_2 的数据

        l = 5

        for sample_1, sample_2 in zip(samples_relation_1, samples_relation_2):
            # 获取第一个数据的 tokens 并进行实体索引
            tokens_1 = sample_1['tokens'][1:-1][:]  # 去除 [CLS] 和 [SEP] 标记
            entity_1_start = tokens_1.index(30522)  # 实体 1 开始
            entity_1_end = tokens_1.index(30523)  # 实体 1 结束
            entity_2_start = tokens_1.index(30524)  # 实体 2 开始
            entity_2_end = tokens_1.index(30525)  # 实体 2 结束

            if entity_2_start <= entity_1_start:
                continue  # 如果实体顺序不对，跳过该样本
            # 截取 token 片段
            token_1_sub = tokens_1[max(0, entity_1_start - l): min(entity_1_end + l + 1, entity_2_start)]

            # 获取第二个数据的 tokens 并进行实体索引
            tokens_2 = sample_2['tokens'][1:-1][:]  # 去除 [CLS] 和 [SEP] 标记
            entity_1_start = tokens_2.index(30522)
            entity_1_end = tokens_2.index(30523)
            entity_2_start = tokens_2.index(30524)
            entity_2_end = tokens_2.index(30525)

            if entity_2_start <= entity_1_start:
                continue  # 如果实体顺序不对，跳过该样本
            # 截取 token 片段
            token_2_sub = tokens_2[max(entity_1_end + 1, entity_2_start - l): min(entity_2_end + l + 1, len(tokens_2))]

            # 合并两个片段并添加 [CLS] 和 [SEP] 标记
            token = [101] + token_1_sub + token_2_sub + [102]

            # 将该样本加入 ACA 数据
            aca_data.append({
                'relation': relation_id,
                'tokens': token,
                'string': tokenizer.decode(token)  # 解码 token 为字符串
            })

            # 确保每个实体标记只出现一次
            for index in [30522, 30523, 30524, 30525]:
                assert index in token and token.count(index) == 1

        relation_id += 1  # 更新关系 ID

    # 生成替换实体样本数据
    for relation in current_relations:
        # 排除某些特定的关系
        if relation in ['P26', 'P3373', 'per:siblings', 'org:alternate_names', 'per:spous', 'per:alternate_names',
                        'per:other_family']:
            continue

        # 遍历该关系的数据
        for sample in training_data[relation]:
            tokens = sample['tokens'][:]  # 获取样本的 tokens
            entity_1_start = tokens.index(30522)  # 实体 1 开始
            entity_1_end = tokens.index(30523)  # 实体 1 结束
            entity_2_start = tokens.index(30524)  # 实体 2 开始
            entity_2_end = tokens.index(30525)  # 实体 2 结束

            # 交换实体 1 和实体 2 的位置
            tokens[entity_1_start] = 30524  # 实体 1 -> 实体 2 开始
            tokens[entity_1_end] = 30525  # 实体 1 -> 实体 2 结束
            tokens[entity_2_start] = 30522  # 实体 2 -> 实体 1 开始
            tokens[entity_2_end] = 30523  # 实体 2 -> 实体 1 结束

            # 将该样本加入 ACA 数据
            aca_data.append({
                'relation': relation_id,
                'tokens': tokens,
                'string': tokenizer.decode(tokens)
            })

            # 确保每个实体标记只出现一次
            for index in [30522, 30523, 30524, 30525]:
                assert index in tokens and tokens.count(index) == 1

        relation_id += 1  # 更新关系 ID

    return aca_data  # 返回生成的 ACA 数据

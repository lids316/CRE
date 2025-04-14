import os
import json
import random
from typing import Dict, List, Tuple, Iterator, Any
import torch
from tqdm import tqdm
from transformers import PreTrainedTokenizer


class DataSampler:
    """
    数据采样器类，用于根据给定的配置、种子和分词器加载和处理数据集，并按任务批次返回训练、验证和测试数据。

    参数:
    - config (Any): 配置对象，包含数据文件路径、关系文件路径等配置。
    - seed (int): 随机种子，用于数据的随机化操作。
    - tokenizer (PreTrainedTokenizer): 用于将文本数据编码为模型可以接受的格式的分词器。
    """

    def __init__(self, config: Any, seed: int, tokenizer: PreTrainedTokenizer):
        """
        初始化数据采样器类。

        参数:
        - config (Any): 配置对象，包含数据路径、关系文件路径等。
        - seed (int): 随机种子。
        - tokenizer (PreTrainedTokenizer): 用于数据编码的分词器。
        """
        self.config = config
        self.tokenizer = tokenizer
        self.id2rel, self.rel2id = self._read_relations(config.relation_file)  # 加载关系文件，获取 id 和关系映射
        self.seed = seed
        self.rng = random.Random(seed) if seed is not None else random.Random()  # 初始化随机数生成器

        self._init_task_order()  # 初始化任务顺序（打乱关系）

        # 加载数据集（训练集、验证集、测试集）
        self.training_dataset, self.valid_dataset, self.test_dataset = self._load_datasets()

        self.batch = 0  # 当前批次
        self.task_length = len(self.id2rel) // self.config.relations_per_task  # 每个任务的关系数
        self.seen_relations: List[str] = []  # 已经处理过的关系
        self.history_test_data: Dict[str, List[Dict]] = {}  # 存储每个关系的历史测试数据

    def _init_task_order(self) -> None:
        """
        初始化任务顺序，将关系打乱顺序，准备任务处理。
        """
        self.shuffled_relations = list(range(len(self.id2rel)))  # 获取所有关系的索引
        self.rng.shuffle(self.shuffled_relations)  # 随机打乱索引顺序

    def _load_datasets(self) -> Tuple[List[List[Dict]], List[List[Dict]], List[List[Dict]]]:
        """
        加载数据集，如果缓存存在且有效，直接加载缓存，否则重新处理并保存数据集。

        返回:
        - Tuple[List[List[Dict]], List[List[Dict]], List[List[Dict]]]: 返回处理后的训练集、验证集和测试集。
        """
        cache_file = self.config.cache_file  # 缓存文件路径

        if os.path.exists(cache_file):
            # 如果缓存文件存在，则直接加载
            return torch.load(cache_file, weights_only=True)

        # 如果缓存无效，重新处理数据
        dataset = self._process_data(self.config.data_file)
        torch.save(dataset, cache_file)  # 保存缓存
        return dataset

    def _process_data(self, data_file: str) -> Tuple[List[List[Dict]], List[List[Dict]], List[List[Dict]]]:
        """
        处理原始数据文件，将其转化为适合模型的数据格式，并进行训练、验证、测试集的划分。

        参数:
        - data_file (str): 原始数据文件路径

        返回:
        - Tuple[List[List[Dict]], List[List[Dict]], List[List[Dict]]]: 返回训练集、验证集和测试集的样本。
        """
        with open(data_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)  # 加载原始数据

        num_relations = len(self.id2rel)  # 获取关系的总数
        train = [[] for _ in range(num_relations)]  # 初始化训练集
        valid = [[] for _ in range(num_relations)]  # 初始化验证集
        test = [[] for _ in range(num_relations)]  # 初始化测试集

        # 遍历每一个关系，处理每个关系的数据
        for rel_name, samples in tqdm(raw_data.items(), desc="Processing relations"):
            rel_id = self.rel2id[rel_name]  # 获取当前关系的ID
            if self.seed is not None:
                random.seed(self.seed)  # 设置随机种子
            texts = [' '.join(s['tokens']) for s in samples]  # 获取每个样本的文本
            encoded = self.tokenizer.batch_encode_plus(
                texts,
                add_special_tokens=True,
                truncation=True,
                padding=True
            )  # 使用分词器对文本进行编码

            # 创建数据样本
            tokenized_samples = [
                {
                    'relation': rel_id,
                    'tokens': encoded['input_ids'][i],
                    'string': f"[RelData] {rel_name} {text}"
                }
                for i, text in enumerate(texts)
            ]

            # 按照任务类型进行数据划分
            if self.config.task_name == 'FewRel':
                test_split = self.config.num_of_train_samples + self.config.num_of_val_samples
                train[rel_id] = tokenized_samples[:self.config.num_of_train_samples]  # 训练集划分
                valid[rel_id] = tokenized_samples[self.config.num_of_train_samples:test_split]  # 验证集划分
                test[rel_id] = tokenized_samples[test_split:]  # 测试集划分
            else:
                test_size = min(len(samples) // 5, 40)
                test[rel_id] = tokenized_samples[:test_size]  # 测试集划分
                train[rel_id] = tokenized_samples[test_size:test_size + 320]  # 训练集划分

        return train, valid, test

    def __iter__(self) -> Iterator:
        """
        迭代器初始化方法，用于开始一个新的训练周期。

        返回:
        - Iterator: 返回当前数据采样器对象自身作为迭代器。
        """
        self.batch = 0
        self.seen_relations = []  # 清空已见关系
        self.history_test_data = {}  # 清空历史测试数据
        return self

    def __next__(self) -> Tuple[Dict, Dict, Dict, List[str], Dict, List[str]]:
        """
        获取下一个批次的数据。

        返回:
        - Tuple[Dict, Dict, Dict, List[str], Dict, List[str]]: 返回当前批次的训练集、验证集、测试集，当前批次的关系列表，以及历史和已见的测试数据。
        """
        if self.batch >= self.task_length:
            raise StopIteration  # 如果当前批次已经超过总批次数，停止迭代

        start = self.batch * self.config.relations_per_task  # 当前批次的起始关系索引
        end = (self.batch + 1) * self.config.relations_per_task  # 当前批次的结束关系索引
        indices = self.shuffled_relations[start:end]  # 获取当前批次的关系索引
        self.batch += 1  # 更新批次计数

        current_relations = []  # 当前批次的关系列表
        cur_train, cur_valid, cur_test = {}, {}, {}  # 当前批次的训练集、验证集、测试集

        # 遍历当前批次的所有关系，获取其对应的数据
        for idx in indices:
            rel_name = self.id2rel[idx]
            current_relations.append(rel_name)
            self.seen_relations.append(rel_name)

            cur_train[rel_name] = self.training_dataset[idx]
            cur_valid[rel_name] = self.valid_dataset[idx]
            cur_test[rel_name] = self.test_dataset[idx]
            self.history_test_data[rel_name] = self.test_dataset[idx]

        return cur_train, cur_valid, cur_test, current_relations, self.history_test_data, self.seen_relations

    @staticmethod
    def _read_relations(relation_file: str) -> Tuple[List[str], Dict[str, int]]:
        """
        读取关系文件并返回关系ID与关系名称的映射。

        参数:
        - relation_file (str): 关系文件路径

        返回:
        - Tuple[List[str], Dict[str, int]]: 返回关系ID列表和关系名称到ID的映射字典
        """
        try:
            with open(relation_file, 'r', encoding='utf-8') as f:
                id2rel = json.load(f)  # 加载关系文件
            return id2rel, {rel: idx for idx, rel in enumerate(id2rel)}  # 返回ID到关系的映射
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Error loading relation file: {str(e)}") from e

    def reset(self) -> None:
        """
        重置数据采样器，准备开始新一轮的任务采样。

        重置状态包括批次计数器、已见关系和历史测试数据，并重新初始化任务顺序。
        """
        self.batch = 0
        self.seen_relations = []  # 清空已见关系
        self.history_test_data = {}  # 清空历史测试数据
        self._init_task_order()  # 重新初始化任务顺序

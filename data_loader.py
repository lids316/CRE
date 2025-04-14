import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Optional, Any


class ReDataset(Dataset):
    """
    数据集类，封装了数据与配置，并提供了对数据的访问方式。

    参数：
    - data (List[Dict]): 输入数据列表，每个字典包含了 token、标签、关系等信息。
    - config (Optional[Any]): 配置对象，默认为 None。
    """

    def __init__(self, data: List[dict], config: Optional[Any] = None):
        self.data = data  # 存储数据
        self.config = config  # 存储配置

    def __len__(self) -> int:
        """返回数据集的大小"""
        return len(self.data)

    def __getitem__(self, idx: int) -> dict:
        """根据索引获取单条数据"""
        return self.data[idx]

    @staticmethod
    def collate_fn(data: List[dict]) -> Tuple[List[int], torch.Tensor, torch.Tensor, List[str]]:
        """
        合并一个批次的数据，并进行必要的预处理。

        参数：
        - data (List[dict]): 输入的数据列表，每个数据项是一个字典，包含了 tokens、relation 等信息。

        返回：
        - idxs (List[int]): 当前批次的索引列表。
        - label (torch.Tensor): 当前批次的标签（关系）。
        - tokens (torch.Tensor): 当前批次的 tokens，经过 padding 后的张量。
        - strings (List[str]): 当前批次的字符串表示。
        """
        # 提取索引、标签、tokens 和字符串
        idxs = [item.get('idx', 0) for item in data]  # 默认索引为 0
        label = torch.tensor([item['relation'] for item in data])  # 关系标签
        tokens = [torch.tensor(item['tokens'], dtype=torch.long) for item in data]  # 每个样本的 token 张量

        # 使用 pad_sequence 对 tokens 进行填充
        tokens = nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0)  # padding_value=0

        # 提取每个样本的字符串表示，若没有则默认为 'None'
        strings = [item.get('string', 'None') for item in data]

        # 返回处理后的批次数据
        return idxs, label, tokens, strings


def get_data_loader(config: Any, data, shuffle: bool = False,
                    drop_last: bool = False, batch_size: Optional[int] = None,
                    sampler=None, num_workers: int = 0, pin_memory: bool = True) -> DataLoader:
    """
    创建并返回一个数据加载器（DataLoader）对象，用于加载训练、验证或测试数据。

    参数:
    - config: 配置对象，包含批量大小等相关配置。
    - data: 数据，通常是样本的列表或字典。
    - shuffle (bool): 是否打乱数据，默认为 False。
    - drop_last (bool): 如果为 True，最后一个批次如果小于 batch_size 将被丢弃，默认为 False。
    - batch_size (Optional[int]): 每个批次的大小，默认为 None，自动从配置中选择。
    - sampler: 数据采样器，可以用于自定义采样方式，默认为 None。
    - num_workers (int): 用于加载数据的子进程数，默认为 0，表示不使用多进程。
    - pin_memory (bool): 如果为 True，会将数据加载到固定内存，提升 GPU 加载速度，默认为 True。

    返回:
    - DataLoader: 生成的 DataLoader 对象。
    """

    # 创建数据集对象
    dataset = ReDataset(data, config)

    # 处理 batch_size
    if batch_size is None:
        batch_size = min(config.batch_size, len(data))
    else:
        batch_size = min(batch_size, len(data))

    # 创建 DataLoader
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
        sampler=sampler,
        collate_fn=dataset.collate_fn,
        drop_last=drop_last
    )

    return data_loader

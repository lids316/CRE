from typing import Any
from transformers import AutoModel, AutoConfig
import torch
import torch.nn as nn
from typing import List, Tuple
import torch.nn.functional as F


class ProtoSoftmaxLayer(nn.Module):
    def __init__(self, config: Any, sentence_encoder: nn.Module, num_classes: int):
        """
        初始化ProtoSoftmaxLayer。

        参数:
        config (object) -- 配置对象，包含模型的超参数和配置信息
        sentence_encoder (nn.Module) -- 句子编码器（通常是BERT或其他文本编码模型）
        num_classes (int) -- 类别数，定义输出层的大小
        """
        super(ProtoSoftmaxLayer, self).__init__()

        self.prototypes = None  # 初始化原型（在增量学习中会更新）
        self.config = config  # 配置对象
        self.sentence_encoder = sentence_encoder  # 句子编码器
        self.num_classes = num_classes  # 类别数
        self.hidden_size = self.sentence_encoder.get_output_size()  # 获取句子编码器的输出维度（隐藏层大小）
        self.classifier = nn.Linear(self.hidden_size, self.num_classes, bias=False)  # 全连接层，无偏置
        self.static_prototypes = torch.empty(0, self.hidden_size)

    @staticmethod
    def __calculate_distance__(representation: torch.Tensor, prototypes: torch.Tensor,
                               alpha: float = 0.5) -> torch.Tensor:
        """
        混合点积相似度与余弦相似度

        参数：
        alpha : 混合比例 (0.0=纯点积, 1.0=纯余弦)
        """
        dot_product = torch.matmul(representation, prototypes.T)  # (B, C)

        norm_rep = torch.norm(representation, p=2, dim=-1, keepdim=True)  # (B, 1)
        norm_proto = torch.norm(prototypes, p=2, dim=-1)  # (C,)

        cosine_sim = dot_product / (norm_rep * norm_proto + 1e-8)  # (B, C)

        return (1 - alpha) * dot_product + alpha * cosine_sim

    def memory_forward(self, representation: torch.Tensor) -> torch.Tensor:
        """
        通过内存原型计算距离。

        参数:
        representation (torch.Tensor) -- 输入表示，形状为(batch_size, hidden_size)

        返回:
        torch.Tensor -- 距离矩阵，形状为(batch_size, num_classes)
        """
        distance_memory = self.__calculate_distance__(representation, self.prototypes)  # 计算输入与原型的距离
        return distance_memory

    def set_memorized_prototypes(self, prototypes: torch.Tensor) -> None:
        """
        设置和存储原型。

        参数:
        prototypes (torch.Tensor) -- 存储的原型，形状为(num_classes, hidden_size)
        """
        # 存储原型并将其转移到指定设备上
        # self.prototypes = 0.1*self.static_prototypes + 0.9*prototypes.detach().to(self.config.device)
        self.prototypes = prototypes.detach().to(self.config.device)

    def set_static_prototypes(self, prototypes: torch.Tensor) -> None:
        # 存储原型并将其转移到指定设备上
        # 确保 self.static_prototypes 和 prototypes 都在相同设备上
        self.static_prototypes = torch.cat(
            [self.static_prototypes.to(self.config.device), prototypes.detach().to(self.config.device)], dim=0)

    def adjust_static_prototypes(self, prototypes: torch.Tensor) -> None:
        # 存储原型并将其转移到指定设备上
        self.static_prototypes = 0.99*self.static_prototypes + 0.01*prototypes.to(self.config.device)


    def get_feature(self, sentences: torch.Tensor) -> torch.Tensor:
        """
        获取句子的表示特征。

        参数:
        sentences (torch.Tensor) -- 输入的句子，形状为(batch_size, sequence_length)

        返回:
        torch.Tensor -- 获取到的句子特征，形状为(batch_size, hidden_size)
        """
        representation = self.sentence_encoder(sentences)  # 使用句子编码器获取句子特征
        return representation.cpu().data.numpy()  # 返回CPU上的特征，转换为numpy数组

    def get_mem_feature(self, representation: torch.Tensor) -> torch.Tensor:
        """
        获取表示与原型的距离特征。

        参数:
        representation (torch.Tensor) -- 输入的表示，形状为(batch_size, hidden_size)

        返回:
        torch.Tensor -- 计算得到的距离特征，形状为(batch_size, num_classes)
        """
        distance = self.memory_forward(representation)  # 计算输入表示与原型的距离
        return distance

    def incremental_learning(self, old_class_count: int, new_class_count: int) -> None:
        """
        执行增量学习，增加类别数并调整全连接层的权重。

        参数:
        old_class_count (int) -- 旧的类别数
        new_class_count (int) -- 新增的类别数
        """
        # 获取当前全连接层的权重
        weight = self.classifier.weight.data
        # 更新全连接层，新的类别数为原类别数 + 新增类别数
        self.classifier = nn.Linear(768, old_class_count + new_class_count, bias=False).to(self.config.device)

        with torch.no_grad():
            # 将旧类别的权重保留到新的全连接层中
            self.classifier.weight.data[:old_class_count] = weight[:old_class_count]

    @staticmethod
    def _init_weights(module: nn.Module):
        nn.init.xavier_normal_(module.weight)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

    def generate_negative_samples(self, batch_representations: torch.Tensor) -> torch.Tensor:
        """
        生成负样本。

        对于每个输入样本，从其余样本中随机选择 `num_negatives` 个样本作为负样本。

        Args:
            batch_representations (torch.Tensor): 输入的批次特征矩阵，形状为 (batch_size, feature_dim)。

        Returns:
            torch.Tensor: 生成的负样本，形状为 (batch_size, num_negatives, feature_dim)。
        """
        batch_size, feature_dim = batch_representations.shape  # 获取批次大小和特征维度
        negatives = []  # 用于存储负样本的列表

        # 对每个样本生成负样本
        for i in range(batch_size):
            # 选择除当前样本外的其他所有样本
            selection = torch.cat([batch_representations[:i], batch_representations[i + 1:]])

            # 随机选择 `num_negatives` 个样本作为负样本
            indices = torch.randperm(selection.size(0))[:self.config.num_negatives]  # 随机选取索引
            negative_samples = selection[indices]  # 根据索引选择负样本

            negatives.append(negative_samples)  # 将生成的负样本添加到列表中

        # 将负样本列表堆叠成一个张量，形状为 (batch_size, num_negatives, feature_dim)
        negatives = torch.stack(negatives, dim=0)

        return negatives

    def forward(self, sentences: torch.Tensor) -> tuple[Any, Any]:
        """
        前向传播，通过句子编码器获取句子表示，并通过全连接层计算logits。

        参数:
        sentences (torch.Tensor) -- 输入的句子，形状为(batch_size, sequence_length)

        返回:
        torch.Tensor -- logits和句子表示，形状为(batch_size, num_classes)和(batch_size, hidden_size)
        """
        # 获取句子表示
        representation = self.sentence_encoder(sentences)  # (B, H)
        logits = self.classifier(representation)  # 通过全连接层计算logits
        return logits, representation  # 返回logits和表示


class BertEncoder(nn.Module):
    def __init__(self, config: Any):
        """
        初始化BertEncoder模型。

        参数:
        config (object) -- 配置对象，包含模型的超参数和配置信息
        """
        super(BertEncoder, self).__init__()

        # 从预训练模型路径加载配置，并设置mean_resizing为False
        self.bert_config = AutoConfig.from_pretrained(config.bert_model_path, mean_resizing=False)

        # 加载预训练BERT模型
        self.encoder = AutoModel.from_pretrained(config.bert_model_path, config=self.bert_config)

        # 设置编码模式（'standard'或'entity_marker'）
        self._setup_encoding_pattern(config)

    def _setup_encoding_pattern(self, config: Any) -> None:
        """
        根据配置设置编码模式。

        参数:
        config (object) -- 配置对象，包含编码模式及其他设置

        返回:
        None
        """
        # 验证传入的编码模式是否合法
        if config.encoding_mode not in ['standard', 'entity_marker']:
            raise ValueError(f"Invalid pattern: {config.encoding_mode}")

        # 设置编码模式和输出维度
        self.pattern = config.encoding_mode
        self.output_size = 768  # 输出维度，通常为BERT的hidden_size

        # 如果编码模式为'entity_marker'，调整BERT词汇表大小并设置线性变换
        if self.pattern == 'entity_marker':
            self.encoder.resize_token_embeddings(config.vocab_size + 4)  # 增加4个标记（实体标记）
            self.linear_transform = nn.Sequential(
                nn.Linear(self.bert_config.hidden_size * 2, self.bert_config.hidden_size, bias=True),  # 合并两个实体向量后的线性变换
                nn.GELU(),  # GELU激活函数
                nn.LayerNorm([self.bert_config.hidden_size])  # 层归一化
            )
        else:
            # 如果编码模式为'标准'，只进行一个简单的线性变换
            self.linear_transform = nn.Linear(self.bert_config.hidden_size, self.output_size)

        # 初始化权重
        self._init_weights()

        # Dropout层用于防止过拟合
        self.drop = nn.Dropout(config.dropout_rate)

    def _init_weights(self) -> None:
        """
        初始化线性层的权重和偏置。

        返回:
        None
        """
        for module in self.linear_transform.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)  # 使用Xavier初始化
                nn.init.constant_(module.bias, 0)  # 偏置初始化为0

    def get_output_size(self) -> int:
        """
        获取输出的维度大小。

        返回:
        int -- 输出维度大小
        """
        return self.output_size

    def _standard_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        使用标准的BERT输出方式进行前向传播。

        参数:
        input_ids (torch.Tensor) -- 输入的token IDs，形状为(batch_size, sequence_length)

        返回:
        torch.Tensor -- BERT池化后的输出，形状为(batch_size, hidden_size)
        """
        outputs = self.encoder(input_ids)  # BERT模型的前向输出
        return outputs.pooler_output  # 返回池化后的输出（CLS标记的表示）

    def _entity_marker_forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        使用实体标记编码方式进行前向传播。

        参数:
        input_ids (torch.Tensor) -- 输入的token IDs，形状为(batch_size, sequence_length)

        返回:
        torch.Tensor -- 融合后的实体向量，形状为(batch_size, hidden_size)
        """
        # 创建注意力掩码（非零位置为有效）
        attention_mask = input_ids != 0

        # 进行BERT前向传播
        outputs = self.encoder(input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state  # 获取BERT最后一层的隐藏状态

        # 获取实体标记的位置，30522和30524是特殊的实体标记ID
        e11_pos = (input_ids == 30522).int().argmax(dim=1)  # 获取第一个实体标记的位置
        e21_pos = (input_ids == 30524).int().argmax(dim=1)  # 获取第二个实体标记的位置

        # 创建批次索引，用于从序列输出中获取对应位置的向量
        batch_indices = torch.arange(input_ids.size(0), device=input_ids.device)

        # 获取对应位置的实体向量
        e11_vectors = sequence_output[batch_indices, e11_pos]
        e21_vectors = sequence_output[batch_indices, e21_pos]

        # 将两个实体向量连接（拼接）
        combined = torch.cat([e11_vectors, e21_vectors], dim=1)

        # 通过线性变换和Dropout层进行处理
        hidden = self.linear_transform(self.drop(combined))

        return hidden

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        根据编码模式选择对应的前向传播方式。

        参数:
        input_ids (torch.Tensor) -- 输入的token IDs，形状为(batch_size, sequence_length)

        返回:
        torch.Tensor -- 模型的最终输出，形状为(batch_size, output_size)
        """
        if self.pattern == 'standard':
            return self._standard_forward(input_ids)
        return self._entity_marker_forward(input_ids)


class Experience:
    """
    用于管理训练中的经验。每个经验包含一个句子、标签、模型输出和损失值。可以根据损失值选择高质量经验并进行管理。

    Attributes:
        top_k (float): 选择高质量经验的比例，取值范围为 [0, 1]。
        experiences (List[Tuple[str, torch.Tensor, torch.Tensor, float]]): 存储经验的列表，每个经验包含句子、标签、模型输出和损失。
        usage_count (List[int]): 存储每个经验的使用次数。
    """

    def __init__(self, config):
        """
        初始化 Experience 对象。

        Args:
            config: 配置对象，包含 top_k 参数来指定选择的高质量经验比例。
        """
        self.top_k = config.top_k  # 设置选择高质量经验的比例
        self.experiences: List[Tuple[str, torch.Tensor, torch.Tensor, float]] = []  # 存储经验数据
        self.usage_count: List[int] = []  # 存储每个经验的使用次数

    def add(self, sentence: str, labels: torch.Tensor, outputs: torch.Tensor):
        """
        向经验池中添加新的经验。

        Args:
            sentence (str): 输入的句子或文本。
            labels (torch.Tensor): 对应的标签，通常是目标类别的张量。
            outputs (torch.Tensor): 模型的预测输出，通常是 logits。
        """
        # 计算交叉熵损失，用于衡量模型输出与标签的差异
        loss = F.cross_entropy(outputs, labels)

        # 将经验（句子、标签、模型输出、损失值）添加到经验池中
        self.experiences.append((sentence, labels, outputs, loss.item()))

        # 为新经验初始化使用次数
        self.usage_count.append(0)

    def get_high_quality_experiences(self) -> List[Tuple[str, torch.Tensor, torch.Tensor, float]]:
        """
        获取高质量经验，通过按损失值排序并选择前 top_k 比例的经验。

        Returns:
            List[Tuple[str, torch.Tensor, torch.Tensor, float]]: 高质量经验列表，每个经验包含句子、标签、模型输出和损失值。
        """
        # 选择 top_k 比例的高质量经验
        num_high_quality = int(len(self.experiences) * self.top_k)

        # 按损失值升序排序经验，损失越小的经验质量越高
        sorted_experiences = sorted(self.experiences, key=lambda x: x[3])

        # 返回前 top_k 个高质量经验
        high_quality_experiences = sorted_experiences[:num_high_quality]
        return high_quality_experiences

    def increment_usage(self, index: int):
        """
        增加指定经验的使用次数。

        Args:
            index (int): 要增加使用次数的经验的索引。
        """
        self.usage_count[index] += 1


def eliminate_experiences(config, experiences):
    # 获取高质量经验
    threshold = config.threshold
    high_quality_experiences = experiences.get_high_quality_experiences()

    # 使用频率计算
    usage_threshold = int(len(experiences.experiences) * threshold)
    frequent_experiences_indices = sorted(range(len(experiences.usage_count)), key=lambda x: experiences.usage_count[x],
                                          reverse=True)[:usage_threshold]

    # 获取频繁使用的经验
    frequent_experiences = [experiences.experiences[i] for i in frequent_experiences_indices]

    # 合并高质量经验和频繁使用经验，并去重
    combined_experiences = high_quality_experiences + frequent_experiences
    unique_experiences = list(set(combined_experiences))

    # 更新经验池
    experiences.experiences = unique_experiences
    experiences.usage_count = [0] * len(unique_experiences)

    return experiences

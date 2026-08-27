# -*- coding: utf-8 -*-
"""
仅此文件允许考生修改：
- 请在下列函数的函数体内完成实现。
- 不要改动函数名与参数签名。
- 你可以新增少量辅助函数。
"""

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


def _last_token_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """
    Last-token pooling
    
    参数：
        last_hidden_states: 模型输出的最后一层隐藏状态 (batch_size, seq_len, hidden_dim)
        attention_mask: 注意力掩码 (batch_size, seq_len)
    
    返回：
        句向量 (batch_size, hidden_dim)
    """
    # 检查是否使用了左侧 padding
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    
    if left_padding:
        # 左侧 padding：直接取最后一个 token
        return last_hidden_states[:, -1]
    else:
        # 右侧 padding：需要找到每个样本的最后一个有效 token
        seq_lens = attention_mask.sum(dim=1) - 1  # 最后一个有效 token 的索引
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), seq_lens]


def compute_similarity(text1: str, text2: str, model: AutoModel, tokenizer: AutoTokenizer) -> float:
    """
    参考实现：计算两个文本之间的相似度
    
    参数：
        text1: 第一个文本字符串
        text2: 第二个文本字符串
        model: 预加载的 AutoModel
        tokenizer: 预加载的 AutoTokenizer
    
    返回：
        相似度值（0.0 到 1.0 之间的浮点数）
    
    步骤：
    1. 对文本进行分词和编码
    2. 获取模型输出
    3. 使用 last-token pooling 提取句向量
    4. L2 归一化
    5. 计算余弦相似度（点积）
    """
    device = next(model.parameters()).device
    
    # 1. 对文本进行分词和编码
    inputs = tokenizer(
        [text1, text2],
        padding=True,           # 自动 padding 到最长序列
        truncation=True,        # 超过最大长度时截断
        max_length=8192,        # Qwen3-Embedding 支持的最大长度
        return_tensors="pt",    # 返回 PyTorch 张量
    )
    
    # 将输入迁移到模型所在设备
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # 2. 获取模型输出
    with torch.no_grad():
        outputs = model(**inputs)
    
    # 3. 使用 last-token pooling 提取句向量
    last_hidden_states = outputs.last_hidden_state  # (2, seq_len, hidden_dim)
    sentence_embeddings = _last_token_pool(last_hidden_states, inputs["attention_mask"])  # (2, hidden_dim)
    
    # 4. L2 归一化（将向量归一化为单位向量）
    sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
    
    # 5. 计算余弦相似度（归一化后的向量点积就是余弦相似度）
    similarity = (sentence_embeddings[0] @ sentence_embeddings[1]).item()
    
    return similarity
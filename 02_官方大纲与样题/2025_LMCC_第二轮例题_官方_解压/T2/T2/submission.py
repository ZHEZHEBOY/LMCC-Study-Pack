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


# ============================================================
# 相似度计算函数
# ============================================================
def student_compute_similarity(text1: str, text2: str, model: AutoModel, tokenizer: AutoTokenizer) -> float:
    """
    考生实现：计算两个文本之间的相似度
    
    参数：
        text1: 第一个文本字符串
        text2: 第二个文本字符串
        model: 预加载的 AutoModel（评测程序提供）
        tokenizer: 预加载的 AutoTokenizer（评测程序提供）
    
    返回：
        相似度值（0.0 到 1.0 之间的浮点数）
    
    要求：
        - 使用传入的 model 和 tokenizer，不要自己加载模型
        - 实现 last-token pooling
        - 必须使用左侧 padding
        - L2 归一化
        - 计算余弦相似度（点积）
        - 不得使用 sentence_transformers
    """
    # ======== 考生实现区域（可修改） ========
    
    # TODO: 在这里实现相似度计算  
    pass
    
    # ======== 考生实现区域（可修改） ========

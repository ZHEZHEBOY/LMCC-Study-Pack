# -*- coding: utf-8 -*-
"""
仅此文件允许考生修改：
- 请在下列函数的函数体内完成实现。
- 不要改动函数名与参数签名。
- 你可以新增少量辅助函数。
"""

from typing import List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# ============================================================
# 第一部分：Prompt 定义
# ============================================================
def build_system_prompt() -> str:
    """
    考生实现：定义 system prompt
    - 返回一个 system prompt，要求模型以"[Answer]: xxxx"的格式给出最终数值。
    """
    # ======== 考生实现区域（可修改） ========
    
    # TODO: 在这里编写你的 system prompt
    pass
    
    # ======== 考生实现区域（可修改） ========


# ============================================================
# 第二部分：模板拼装
# ============================================================
def apply_chat_template_single(
    tokenizer: AutoTokenizer,
    system_prompt: str,
    problem: str,
) -> str:
    """
    考生实现：将单个问题转换为模型输入文本
    - 使用 tokenizer.apply_chat_template 构造对话
    - 返回拼装好的文本字符串
    """
    # ======== 考生实现区域（可修改） ========
    
    # TODO: 在这里实现对话模板的构造
    pass
    
    # ======== 考生实现区域（可修改） ========


# ============================================================
# 第三部分：核心推理实现
# ============================================================
def generate_single(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    rendered_text: str,
    max_new_tokens: int,
    do_sample: bool,
) -> torch.Tensor:
    """
    考生实现：单条推理
    - 将文本 tokenize 后送入模型生成
    - 返回包含输入和输出的完整 token 序列
    """
    # ======== 考生实现区域（可修改） ========
    
    # TODO: 在这里实现单条推理
    pass
    
    # ======== 考生实现区域（可修改） ========


def generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    rendered_texts: List[str],
    max_new_tokens: int,
    do_sample: bool,
) -> List[torch.Tensor]:
    """
    考生实现：批量推理
    - 一次处理多个问题，提高效率
    - 返回所有批次的输出列表
    """
    # ======== 考生实现区域（可修改） ========
    
    # TODO: 在这里实现批量推理（以下代码均可修改）
    all_outputs = []
    batch_size: int = 2

    # 分批处理
    for i in range(0, len(rendered_texts), batch_size):
        pass

    return all_outputs
    
    # ======== 考生实现区域（可修改） ========


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
    - 返回一个 system prompt，要求模型以"[Answer]: "的单行格式给出最终数值。
    """
    # ======== 考生实现区域（可修改） ========
# 用于非思考模式
#     return """你是一个精确的数学计算助手。请仔细计算并输出最终数值。

# 【输出格式】
# - 使用格式：[Answer]: 数值
# - 例如：[Answer]: 42
# - 整数直接输出，确保计算准确
# - **千万不要输出任何计算过程**

# """
    return """你是一个精确的数学计算助手。请按照以下步骤严格计算并返回准确的最终数值。

【计算步骤】
步骤1 - 符号转换：
   - 将问题中的所有符号转换为标准数学符号
   - ＋, ➕, ⊕, 加 → +
   - －, 减 → -
   - ×, *, 乘 → ×
   - ÷, /, 除 → ÷
   - ⊕ 不是异或运算，就是普通加法 +
   - 输出转换后的标准计算公式

步骤2 - 分步计算：
   - 按照运算优先级逐步计算
   - 每步只写出计算结果，不要过多解释

步骤3 - 输出答案：
   - 使用格式：[Answer]: 数值
   - 例如：[Answer]: 42
   - 整数直接输出，确保计算准确

"""
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
    - 使用 tokenizer.apply_chat_template
    - 返回拼装好的文本字符串
    """
    # ======== 考生实现区域（可修改） ========
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": problem},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=True,
    )
    return rendered
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
    # ======== 考生实现区域（可修改） ========
    
    # 1. tokenize 输入
    inputs = tokenizer(rendered_text, return_tensors="pt", padding=True).to(model.device)
    
    # 2. 生成输出
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        attention_mask=inputs.get("attention_mask"),
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    
    return outputs
    
    # ======== 考生实现区域（可修改） ========


def generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    rendered_texts: List[str],
    max_new_tokens: int,
    do_sample: bool,
) -> List[torch.Tensor]:
    # ======== 考生实现区域（可修改） ========
    
    all_outputs = []
    batch_size: int = 8

    # 分批处理
    for i in range(0, len(rendered_texts), batch_size):
        batch_texts = rendered_texts[i:i + batch_size]
        
        # 1. 批量 tokenize
        batch_inputs = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        ).to(model.device)
        
        # 2. 批量生成
        batch_outputs = model.generate(
            input_ids=batch_inputs["input_ids"],
            attention_mask=batch_inputs.get("attention_mask"),
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        
        all_outputs.append(batch_outputs)
    
    # 返回所有批次的输出列表
    return all_outputs
    
    # ======== 考生实现区域（可修改） ========

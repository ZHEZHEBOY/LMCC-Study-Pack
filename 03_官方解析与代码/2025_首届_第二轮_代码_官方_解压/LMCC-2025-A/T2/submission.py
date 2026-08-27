# -*- coding: utf-8 -*-
"""
仅此文件允许考生修改：
- 请在下列函数的函数体内完成实现。
- 不要改动函数名与参数签名。
"""

from typing import List, Dict
import torch


# ============================================================
# 相似度计算函数（考生实现 - 20分）
# ============================================================
def compute_similarity(query_embedding: torch.Tensor, doc_embedding: torch.Tensor) -> float:
    """
    考生实现：计算两个向量的余弦相似度
    
    参数：
        query_embedding: 查询的 embedding 向量，shape: (embedding_dim,)
        doc_embedding: 文档的 embedding 向量，shape: (embedding_dim,)
    
    返回：
        float: 余弦相似度，范围 [-1, 1]，值越大表示越相似
    """
    # ======== 考生实现区域（可修改） ========
    
    # TODO: 实现余弦相似度计算
    
    return 0.0
    
    # ======== 考生实现区域（可修改） ========


# ============================================================
# 检索函数（考生实现）
# ============================================================
def retrieve_relevant_problems(
    query: str,
    problem_bank: List[Dict],
    embedding_model,
    tokenizer,
    top_k: int = 3,
    get_embedding_func=None
) -> List[Dict]:
    """
    考生实现：从题库中检索与查询最相关的题目
    
    参数：
        query: 查询文本（用户的问题）
        problem_bank: 题库列表，每个元素包含 id, problem, answer, explanation 字段
        embedding_model: 预加载的 embedding 模型
        tokenizer: embedding 模型的 tokenizer
        top_k: 返回最相关的前 k 个题目
        get_embedding_func: 获取文本 embedding 的函数（如果提供）
    
    返回：
        包含最相关的 top_k 个题目的列表
    """
    # ======== 考生实现区域（可修改） ========
    
    retrieved = []
    # TODO: 实现基于 embedding 的检索逻辑
    
    return retrieved[:top_k]
    
    # ======== 考生实现区域（可修改） ========


# ============================================================
# Prompt 定义（考生实现）
# ============================================================
def build_user_message(problem: str, retrieved_problems: List[Dict]) -> str:
    """
    考生实现：组装用户消息（包含检索内容和题目）
    
    参数：
        problem: 当前要解答的题目
        retrieved_problems: 检索到的相似题目列表，每个元素包含 id, problem, answer, explanation 字段
    
    返回：
        完整的用户消息字符串
    """
    # ======== 考生实现区域（可修改） ========
    
    # TODO: 实现用户消息组装逻辑
    
    return problem
    
    # ======== 考生实现区域（可修改） ========


def build_system_prompt() -> str:
    """
    考生实现：定义 system prompt（RAG 版本）
    
    要求：
    1. 引导模型理解数学填空题的任务
    2. 说明会提供检索到的相似题目作为参考
    3. 要求模型以 JSON 格式输出，包含 reasoning 和 answer 两个字段
    4. answer 字段只包含数字，不包含单位
    
    返回：system prompt 字符串
    """
    # ======== 考生实现区域（可修改） ========
    
    return """你是小学数学助手。"""
    
    # ======== 考生实现区域（可修改） ========


def build_generation_parameters() -> dict:
    """
    考生实现：定义生成参数
    
    可配置的参数：
    - max_new_tokens: 最大生成 token 数
    - do_sample: 是否使用采样
    - temperature: 温度参数
    - top_p: nucleus sampling 参数
    - enable_thinking: 是否开启 Thinking 模式
    
    返回：生成参数字典
    """
    # ======== 考生实现区域（可修改） ========
    
    return {
        "max_new_tokens": 256,
        "do_sample": False,
        "enable_thinking": False,
    }
    
    # ======== 考生实现区域（可修改） ========



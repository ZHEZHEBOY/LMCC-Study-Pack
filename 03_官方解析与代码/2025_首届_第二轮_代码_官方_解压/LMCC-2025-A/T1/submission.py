# -*- coding: utf-8 -*-
"""
仅此文件允许考生修改：
- 请在下列函数的函数体内完成实现。
- 不要改动函数名与参数签名。
"""


# ============================================================
# Prompt 定义（考生实现）
# ============================================================
def build_system_prompt() -> str:
    """
    考生实现：定义 system prompt
    
    要求：
    1. 引导模型理解作业批改任务，读取学生作业内容，判断答案是否正确
    2. 说明输入是学生作业文件内容，输出是标准JSON
    3. 引导模型从作业文件中正确识别出：
       - 每道题的学生答案
       - 每道题的正确性判断（true/false）
    4. 要求模型严格按照标准JSON格式输出（包含student_id、judgements两个字段）
    5. 强调要识别答案的核心内容，忽略无关的文字说明
    6. 提供多个示例帮助模型理解任务
    
    标准JSON格式：
    {
      "student_id": "学生ID",
      "judgements": [true, false, true, true, false]
    }
    
    返回：system prompt 字符串
    """
    # ======== 考生实现区域（可修改） ========
    
    # TODO: 实现 system prompt
    return ""
    
    # ======== 考生实现区域（可修改） ========


def build_generation_parameters() -> dict:
    """
    考生实现：定义生成参数
    
    可配置的参数：
    - max_new_tokens: 最大生成 token 数（默认 512）
    - do_sample: 是否使用采样（默认 False，使用贪心解码）
    - temperature: 温度参数（仅当 do_sample=True 时有效）
    - top_p: nucleus sampling 参数（仅当 do_sample=True 时有效）
    - enable_thinking: 是否开启 Thinking 模式（默认 False）
    
    返回：生成参数字典
    """
    # ======== 考生实现区域（可修改） ========
    
    # TODO: 配置生成参数
    return {
        "max_new_tokens": 512,
        "do_sample": False,
        "enable_thinking": False,
    }
    
    # ======== 考生实现区域（可修改） ========




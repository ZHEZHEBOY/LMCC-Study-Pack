# -*- coding: utf-8 -*-
"""
T1 数学作业批改助手 - 评测主程序
本文件不可修改

环境要求：
    在运行本程序前，请确保已激活正确的 conda 环境：
    $ conda activate mindspore
    
    必须使用 MindSpore 环境，否则可能缺少必要的依赖包或版本不匹配。
"""

import json
import re
import time
import argparse
import os
import warnings
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# 过滤 torch_npu 相关的警告
warnings.filterwarnings("ignore", category=UserWarning, module="torch_npu")
# 过滤 numpy 相关的警告（包括 getlimits 模块的警告）
warnings.filterwarnings("ignore", category=UserWarning, module="numpy")
warnings.filterwarnings("ignore", message=".*smallest subnormal.*")

import mindnlp
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import random

# 设置随机种子以确保结果可复现
SEED = 20251115
random.seed(SEED)
torch.manual_seed(SEED)

# 设置 transformers 日志级别，隐藏警告信息
transformers.logging.set_verbosity_error()

# 导入考生实现的函数
from submission import build_system_prompt, build_generation_parameters


# ============================================================
# 工具函数
# ============================================================
def print_memory_info(model: AutoModelForCausalLM):
    """显示模型显存使用情况"""
    print("\n【显存使用情况】")
    try:
        # 获取模型所在的设备
        device = next(model.parameters()).device
        
        # 尝试使用 torch.cuda（如果可用）
        if torch.cuda.is_available() and device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
            max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3  # GB
            print(f"  设备: {device}")
            print(f"  已分配显存: {allocated:.2f} GB")
            print(f"  已保留显存: {reserved:.2f} GB")
            print(f"  峰值显存: {max_allocated:.2f} GB")
        # 尝试使用 torch_npu（如果可用）
        elif hasattr(torch, 'npu') and hasattr(torch.npu, 'is_available') and torch.npu.is_available():
            try:
                # NPU API 可能不接受 device 参数，尝试两种方式
                try:
                    allocated = torch.npu.memory_allocated(device) / 1024**3
                    reserved = torch.npu.memory_reserved(device) / 1024**3
                    max_allocated = torch.npu.max_memory_allocated(device) / 1024**3
                except TypeError:
                    allocated = torch.npu.memory_allocated() / 1024**3
                    reserved = torch.npu.memory_reserved() / 1024**3
                    max_allocated = torch.npu.max_memory_allocated() / 1024**3
                print(f"  设备: {device}")
                print(f"  已分配显存: {allocated:.2f} GB")
                print(f"  已保留显存: {reserved:.2f} GB")
                print(f"  峰值显存: {max_allocated:.2f} GB")
            except Exception:
                # NPU 显存获取失败，使用通用方法
                raise
        else:
            # 通用方法：统计模型参数占用的显存
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            # 假设使用 float16，每个参数2字节
            model_size_gb = total_params * 2 / 1024**3
            print(f"  设备: {device}")
            print(f"  模型参数量: {total_params:,}")
            print(f"  可训练参数: {trainable_params:,}")
            print(f"  模型大小（估算）: {model_size_gb:.2f} GB (float16)")
    except Exception as e:
        # 如果所有方法都失败，至少显示设备信息
        try:
            device = next(model.parameters()).device
            print(f"  设备: {device}")
            print(f"  无法获取详细显存信息: {e}")
        except:
            print(f"  无法获取显存信息: {e}")


def load_jsonl(file_path: str) -> List[Dict]:
    """加载 JSONL 文件"""
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def extract_json(text: str) -> Optional[Dict]:
    """
    从模型输出中提取JSON格式的批改结果
    - 提取JSON对象
    - 返回解析后的字典或None
    """
    # 移除可能的markdown代码块标记
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    # 方法1: 使用正则匹配完整的JSON对象（支持嵌套）
    # 找到第一个 { 的位置
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
    # 从这个位置开始，匹配平衡的括号
    brace_count = 0
    end_idx = start_idx
    
    for i in range(start_idx, len(text)):
        if text[i] == '{':
            brace_count += 1
        elif text[i] == '}':
            brace_count -= 1
            if brace_count == 0:
                end_idx = i
                break
    
    if brace_count == 0 and end_idx > start_idx:
        json_str = text[start_idx:end_idx+1]
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            # 尝试清理并重试
            # 移除可能的控制字符
            json_str_clean = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
            try:
                return json.loads(json_str_clean)
            except json.JSONDecodeError:
                pass
    
    # 方法2: 尝试使用更宽松的正则
    patterns = [
        r'\{[^{}]*"student_id"[^{}]*"judgements"[^{}]*\}',  # 简单模式
        r'\{.*?"student_id".*?"judgements".*?\}',  # 贪婪模式
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                json_str = match.group(0)
                return json.loads(json_str)
            except json.JSONDecodeError:
                continue
    
    return None


def remove_thinking_tags(text: str) -> str:
    """移除 <think>...</think> 标签及其内容"""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)


def save_evaluation_record(task_name: str, correct: int, total: int, score: int, total_time: float):
    """
    保存评测记录到用户目录的 .lmcc 文件夹
    """
    try:
        # 创建记录目录
        home_dir = Path.home()
        lmcc_dir = home_dir / ".lmcc"
        lmcc_dir.mkdir(exist_ok=True)
        
        # 记录文件路径
        record_file = lmcc_dir / "evaluation_records.jsonl"
        
        # 准备记录数据
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task": task_name,
            "correct": correct,
            "total": total,
            "score": score,
            "total_time": round(total_time, 2)
        }
        
        # 追加写入记录
        with open(record_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print(f"\n✅ 评测记录已保存到: {record_file}")
    except Exception as e:
        print(f"\n⚠ 保存评测记录失败: {e}")


def apply_chat_template(
    tokenizer: AutoTokenizer,
    system_prompt: str,
    user_content: str,
    enable_thinking: bool = False,
) -> str:
    """
    将问题转换为模型输入文本
    - 使用 tokenizer.apply_chat_template
    - 返回拼装好的文本字符串
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )
    return rendered


def generate_single(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    rendered_text: str,
    generation_params: Dict,
) -> Tuple[torch.Tensor, int]:
    """
    单条推理（半精度、推理模式）
    - 输入：拼装好的文本字符串
    - 输出：(模型生成的 token 序列, 输入长度)
    """
    # 确保模型处于推理模式
    model.eval()
    
    # 1. tokenize 输入
    inputs = tokenizer(rendered_text, return_tensors="pt", padding=True).to(model.device)
    input_length = inputs["input_ids"].shape[1]
    
    # 2. 准备生成参数
    gen_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask"),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    # 添加用户配置的参数
    gen_kwargs.update(generation_params)
    
    # 移除 enable_thinking（这个参数不是 generate 的参数）
    gen_kwargs.pop("enable_thinking", None)
    
    # 3. 生成输出（使用 no_grad 禁用梯度计算，推理模式）
    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)
    
    return outputs, input_length


# ============================================================
# 评测函数
# ============================================================
def evaluate(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    system_prompt: str,
    test_data: List[Dict],
    generation_params: Dict,
    mode: str,
) -> Tuple[int, int, float]:
    """
    评测函数（5个学生）- 作业批改任务
    返回：(正确题数, 总题数, 总耗时)
    """
    print("\n" + "="*60)
    print("开始评测（5个学生，每个学生5道题）")
    print("="*60)
    
    # 显示评测开始前的显存信息
    print("\n【评测开始前显存状态】")
    print_memory_info(model)
    
    correct = 0
    total = 0
    total_time = 0.0
    
    enable_thinking = generation_params.get("enable_thinking", False)
    
    for idx, item in enumerate(test_data, 1):
        student_id = item["student_id"]
        content = item["content"]
        expected_judgements = item["expected_judgements"]
        
        total += len(expected_judgements)
        
        # 构造用户输入
        user_content = f"请批改以下学生作业，判断每道题的答案是否正确：\n\n学生ID: {student_id}\n\n作业内容：\n{content}"
        
        # 拼装输入
        rendered_text = apply_chat_template(tokenizer, system_prompt, user_content, enable_thinking)
        
        # 推理
        start_time = time.time()
        output_ids, input_length = generate_single(
            model=model,
            tokenizer=tokenizer,
            rendered_text=rendered_text,
            generation_params=generation_params,
        )
        elapsed = time.time() - start_time
        total_time += elapsed
        
        # 解码（只解码新生成的部分）
        generated_ids = output_ids[0][input_length:]
        full_output = tokenizer.decode(generated_ids, skip_special_tokens=False)
        
        # 移除思考标签
        non_thinking_text = remove_thinking_tags(full_output)
        
        # 提取JSON
        predicted_json = extract_json(non_thinking_text)
        
        # 判断正确性
        student_correct = 0
        is_valid = False
        if predicted_json is not None:
            # 检查必需字段
            has_student_id = "student_id" in predicted_json
            has_judgements = "judgements" in predicted_json
            
            if has_student_id and has_judgements:
                predicted_judgements = predicted_json.get("judgements", [])
                if isinstance(predicted_judgements, list) and len(predicted_judgements) == len(expected_judgements):
                    is_valid = True
                    # 逐题判断
                    for i in range(len(expected_judgements)):
                        if predicted_judgements[i] == expected_judgements[i]:
                            student_correct += 1
        
        correct += student_correct
        
        # 输出
        if mode == "demo":
            print(f"\n【学生 {idx}/{len(test_data)}】")
            print(f"学生ID：{student_id}")
            print(f"\n作业内容：")
            print(content)
            
            # 显示模型原始输出（包含特殊token）
            print(f"\n模型原始输出（包含特殊token）：")
            print("-" * 60)
            print(full_output)
            print("-" * 60)
            
            # 如果有thinking标签，显示移除后的文本
            if full_output != non_thinking_text:
                print(f"\n移除 thinking 标签后：")
                print("-" * 60)
                print(non_thinking_text)
                print("-" * 60)
            
            print(f"\n模型输出JSON：")
            if predicted_json:
                print(json.dumps(predicted_json, ensure_ascii=False, indent=2))
            else:
                print("❌ 无法提取JSON")
                print("\n调试信息：")
                print(f"  - 输出长度: {len(non_thinking_text)} 字符")
                print(f"  - 是否包含'{{': {'是' if '{' in non_thinking_text else '否'}")
                print(f"  - 是否包含'}}': {'是' if '}' in non_thinking_text else '否'}")
            
            print(f"\n期望JSON：")
            print(json.dumps({
                "student_id": student_id,
                "judgements": expected_judgements
            }, ensure_ascii=False, indent=2))
            
            # 显示对比信息
            if is_valid:
                print(f"\n题目判断对比：")
                predicted_judgements = predicted_json.get("judgements", [])
                for i in range(len(expected_judgements)):
                    status = "✅ 正确" if predicted_judgements[i] == expected_judgements[i] else "❌ 错误"
                    print(f"  题目{i+1}: 期望={expected_judgements[i]}, 预测={predicted_judgements[i]} - {status}")
            
            print(f"\n判定：{student_correct}/{len(expected_judgements)} 题正确")
            print(f"耗时：{elapsed:.3f}s")
        elif mode == "grading":
            status = f"{student_correct}/{len(expected_judgements)}"
            print(f"学生 {idx}/{len(test_data)}: {status} 题正确", flush=True)
    
    if mode == "grading":
        print()  # 换行
    
    # 显示评测结束后的显存信息
    print("\n【评测结束后显存状态】")
    print_memory_info(model)
    
    return correct, total, total_time


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="T1 小学数学作业批改助手 - 评测程序")
    parser.add_argument(
        "--mode",
        type=str,
        default="demo",
        choices=["demo", "grading"],
        help="运行模式：demo（详细输出）或 grading（简洁输出）",
    )
    args = parser.parse_args()
    
    mode = args.mode
    
    print("="*60)
    print("T1 小学数学作业批改助手 - 评测程序")
    print("="*60)
    print(f"运行模式：{mode}")
    
    # 加载模型
    print("\n正在加载模型...")
    model_path = os.environ.get("MODEL_NAME", "/home/ma-user/Qwen/Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    # 加载模型（半精度）
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # 半精度
        device_map="auto",
        trust_remote_code=True,
    )
    # 设置为推理模式
    model.eval()
    # 禁用梯度计算以节省显存和加速推理
    for param in model.parameters():
        param.requires_grad = False
    print("模型加载完成！（半精度、推理模式）")
    
    # 显示显存信息
    print_memory_info(model)
    
    # 构建 system prompt
    system_prompt = build_system_prompt()
    
    # 检查 system_prompt token 数量
    system_prompt_tokens = tokenizer.encode(system_prompt, add_special_tokens=False)
    system_prompt_token_count = len(system_prompt_tokens)
    max_system_prompt_tokens = 1024 * 2  # 2048
    
    if mode == "demo":
        print("\n【System Prompt】")
        print(system_prompt)
        print(f"\nSystem Prompt Token 数量：{system_prompt_token_count}/{max_system_prompt_tokens}")
    
    # 验证 token 数量限制
    if system_prompt_token_count > max_system_prompt_tokens:
        print(f"\n❌ 错误：System Prompt 超过限制！")
        print(f"   当前 token 数：{system_prompt_token_count}")
        print(f"   限制：{max_system_prompt_tokens}")
        print(f"   超出：{system_prompt_token_count - max_system_prompt_tokens} tokens")
        return
    
    # 获取生成参数
    generation_params = build_generation_parameters()
    if mode == "demo":
        print("\n【Generation Parameters】")
        for key, value in generation_params.items():
            print(f"  {key}: {value}")
    
    # 加载测试数据
    data_dir = Path(__file__).parent.parent / "data"
    test_data = load_jsonl(data_dir / "test_data_T1.jsonl")
    
    # 评测
    correct, total, total_time = evaluate(
        model, tokenizer, system_prompt, test_data, generation_params, mode
    )
    
    # 计算分数：每题2分，满分50分
    score = correct * 2
    
    # 输出总结
    print("\n" + "="*60)
    print("评测总结")
    print("="*60)
    print(f"正确题数：{correct}/{total}")
    print(f"总耗时：{total_time:.2f}s")
    print(f"得分：{score}/50")
    print("="*60)
    
    # 保存评测记录
    save_evaluation_record(
        task_name="T1-小学数学作业批改助手",
        correct=correct,
        total=total,
        score=score,
        total_time=total_time
    )


if __name__ == "__main__":
    main()


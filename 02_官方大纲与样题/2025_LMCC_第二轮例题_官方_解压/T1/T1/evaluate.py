"""
评测程序 - 支持两种运行模式：

1. 演示模式（demo）：
   - 详细展示每个题目的推理过程和结果
   - 显示错误样例的详细信息
   - 适合调试和查看模型表现
   使用方法：
     python evaluate.py
     python evaluate.py --mode demo

2. 评分模式（grading）：
   - 简洁的进度输出
   - 不展示错误详情
   - 最终输出详细评分
   使用方法：
     python evaluate.py --mode grading

评分规则：
  - 阶段一-正确性（逐条推理）：8题，每题10分，答对≥6题给满分60分
  - 阶段二-正确性（批量推理）：32题，每题1.5分，答对题数×1.5分，满分40分
  - 总分：100分
"""
import os
import sys
import time
import re
import json
import argparse
from typing import List, Tuple

import torch

from transformers import AutoTokenizer, AutoModelForCausalLM

from submission import build_system_prompt, apply_chat_template_single, generate_single, generate_batch


# =========================
# 配置
# =========================
MODEL_NAME = os.environ.get("MODEL_NAME", "../Qwen/Qwen3-0.6B")
MAX_NEW_TOKENS = 512
DO_SAMPLE = True


# ============================================================
# 统一后处理（考题自带）
# ============================================================
_think_pat = re.compile(r"<think>.*?</think>", re.DOTALL)

def remove_thinking(text: str) -> str:
    """移除文本中的 <think></think> 标签及其内容"""
    return _think_pat.sub("", text).strip()

def remove_answer_prefix(text: str) -> str:
    """移除文本中的 [Answer]: 前缀"""
    text = text.strip()
    # 匹配 [Answer]: 或 [Answer] : 等变体
    pattern = r"^\s*\[?\s*Answer\s*\]?\s*:?\s*"
    text = re.sub(pattern, "", text, flags=re.IGNORECASE)
    return text.strip()


# ============================================================
# 评测工具
# ============================================================
def _normalize_num(s: str) -> str:
    def to_half(c):
        code = ord(c)
        if 0xFF10 <= code <= 0xFF19:
            return chr(code - 0xFF10 + ord('0'))
        if code == 0xFF0E:
            return '.'
        if code == 0xFF0C:
            return ','
        return c
    s = "".join(to_half(c) for c in s).strip()
    s = s.replace(",", "")
    s = s.rstrip("。．.，,；;、 ")
    return s.strip()


def check_format_and_value(output: str, correct_value: str):
    output = output.strip()
    
    # 检查是否为空
    if not output:
        return False, None, "❌ 输出为空"
    
    # 直接将输出当作答案处理（新格式）
    pred_raw = output
    pred = _normalize_num(pred_raw)
    gold = _normalize_num(correct_value)

    def to_num(x):
        try:
            if "." in x:
                return float(x)
            return int(x)
        except:
            try:
                return float(x)
            except:
                return None

    p, g = to_num(pred), to_num(gold)
    if p is not None and g is not None:
        equal = (p == g) if (isinstance(p, int) and isinstance(g, int)) else (abs(float(p) - float(g)) < 1e-9)
    else:
        equal = (pred == gold)

    if equal:
        return True, pred, "✅ 答案正确"
    else:
        return False, pred, f"❌ 答案错误，应为 {gold}（模型输出: {pred_raw}）"


def score_report(stage_name: str, n_total: int, n_correct: int, elapsed: float, mode: str = "demo"):
    acc = n_correct / n_total if n_total else 0.0
    if mode == "demo":
        print(f"—— {stage_name} 评分 ——")
        print(f"正确题数/总题数：{n_correct}/{n_total}")
        print(f"准确率：{acc*100:.2f}%")
        print(f"总耗时：{elapsed:.3f} s\n")
    return acc

# ============================================================
# 推理封装（调用考生实现 + decode）
# ============================================================
def solve_single(
    model,
    tokenizer,
    problem: str,
    max_new_tokens: int,
    do_sample: bool,
) -> str:
    """
    单条推理完整流程：
    1. 拼装模板 (调用考生实现)
    2. 推理生成 (调用考生实现)
    3. 解码输出 (评测程序)
    """
    # 1. 拼装模板 (调用考生实现)
    system_prompt = build_system_prompt()
    rendered_text = apply_chat_template_single(tokenizer, system_prompt, problem)
    
    # 2. 推理生成 (调用考生实现)
    outputs = generate_single(model, tokenizer, rendered_text, max_new_tokens, do_sample)
    
    # 3. 解码新生成的 token (评测程序)
    inputs = tokenizer(rendered_text, return_tensors="pt", padding=True).to(model.device)
    if "attention_mask" in inputs:
        inp_len = int(inputs["attention_mask"].sum(dim=1).item())
    else:
        inp_len = inputs["input_ids"].shape[1]
    
    # 仅解码新生成的部分 (评测程序)
    gen_only = outputs[0][inp_len:]
    text = tokenizer.decode(gen_only, skip_special_tokens=True).strip()
    
    return text


def solve_batch(
    model,
    tokenizer,
    problems: List[str],
    max_new_tokens: int,
    do_sample: bool,
) -> List[str]:
    """
    批量推理完整流程：
    1. 拼装模板 (调用考生实现)
    2. 推理生成 (调用考生实现)
    3. 批量解码 (评测程序)
    """
    # 1. 批量拼装模板 (调用考生实现)
    system_prompt = build_system_prompt()
    rendered_texts = [
        apply_chat_template_single(tokenizer, system_prompt, p) 
        for p in problems
    ]
    
    # 边界情况：空列表
    if len(rendered_texts) == 0:
        return []
    
    # 2. 批量推理生成 (调用考生实现)
    all_outputs = generate_batch(model, tokenizer, rendered_texts, max_new_tokens, do_sample)
    
    # 边界情况：生成结果为空
    if len(all_outputs) == 0:
        return [""] * len(problems)
    
    # 3. 拼接所有批次的结果（评测程序处理）
    if len(all_outputs) == 1:
        batch_outputs = all_outputs[0]
    else:
        # 找到最大序列长度
        max_len = max(output.shape[1] for output in all_outputs)
        
        # 对每个批次的输出进行填充
        padded_outputs = []
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        
        for output in all_outputs:
            if output.shape[1] < max_len:
                # 需要在右侧填充
                padding_size = max_len - output.shape[1]
                padding = torch.full(
                    (output.shape[0], padding_size),
                    pad_token_id,
                    dtype=output.dtype,
                    device=output.device
                )
                padded_output = torch.cat([output, padding], dim=1)
                padded_outputs.append(padded_output)
            else:
                padded_outputs.append(output)
        
        batch_outputs = torch.cat(padded_outputs, dim=0)
    
    # 边界情况：生成结果为空
    if batch_outputs.numel() == 0:
        return [""] * len(problems)
    
    # 4. 批量解码（评测程序）
    # 直接解码完整输出，然后提取 assistant 回复部分
    out_texts: List[str] = []
    for out in batch_outputs:
        full_text = tokenizer.decode(out, skip_special_tokens=True).strip()
        
        # 找到最后一个 "assistant" 标记，提取其后的内容
        # 这样可以跳过 system 和 user 部分
        if "assistant" in full_text:
            # 找到最后一个 assistant 的位置
            last_assistant_pos = full_text.rfind("assistant")
            # 提取 assistant 之后的内容
            text = full_text[last_assistant_pos + len("assistant"):].strip()
        else:
            # 如果没有 assistant 标记，直接使用全文
            text = full_text
        
        out_texts.append(text)
    
    return out_texts


def run_pipeline(mode: str = "demo", model_name: str = None):
    """
    运行评测流程
    
    Args:
        mode: 运行模式，"demo" 或 "grading"
        model_name: 模型路径，如不指定则使用默认值
    """
    if mode not in ["demo", "grading"]:
        print(f"⚠️  警告：未知的运行模式 '{mode}'，使用默认模式 'demo'")
        mode = "demo"
    
    # 确定模型路径
    if model_name is None:
        model_name = MODEL_NAME
    
    if mode == "demo":
        print(f"========== 演示模式 ==========")
        print(f"Loading model: {model_name}\n")
    else:
        print(f"========== 评分模式 ==========")
        print(f"Loading model: {model_name}\n")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        padding_side="left",
        truncation_side="left",
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.float16,  # 使用半精度，显存占用减半
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = tokenizer.pad_token_id
    
    # 显示设备和显存占用信息
    device = next(model.parameters()).device
    print(f"✅ 模型已加载到设备：{device}")
    try:
        if device.type == "cuda":
            # CUDA 显存信息
            allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
            reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
            total = torch.cuda.get_device_properties(device).total_memory / 1024**3  # GB
            print(f"   显存占用：{allocated:.2f} GB / {total:.2f} GB ({allocated/total*100:.1f}%)")
            print(f"   已预留：{reserved:.2f} GB")
        elif device.type == "npu":
            # NPU 显存信息
            if hasattr(torch.npu, 'memory_allocated') and hasattr(torch.npu, 'memory_reserved'):
                allocated = torch.npu.memory_allocated(device) / 1024**3  # GB
                reserved = torch.npu.memory_reserved(device) / 1024**3  # GB
                print(f"   显存占用：{allocated:.2f} GB")
                print(f"   已预留：{reserved:.2f} GB")
            else:
                print(f"   (NPU 显存信息不可用)")
        elif device.type == "cpu":
            # CPU 内存信息（仅显示模型参数大小）
            param_size = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**3
            print(f"   模型参数大小：{param_size:.2f} GB")
    except Exception as e:
        print(f"   (无法获取显存信息: {e})")
    print()

    # 预检 system_prompt（来源于考生实现）
    sp = build_system_prompt().strip()
    assert len(sp) > 0, "system_prompt 为空，请在 submission.py 中实现 build_system_prompt()"

    # ===== 阶段一：逐条推理 =====
    # 从 data/test_data.jsonl 读取测试数据
    stage1_items: List[Tuple[str, str]] = []
    test_data_path = os.path.join(os.path.dirname(__file__), "data", "test_data_1.jsonl")
    with open(test_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                stage1_items.append((item["problem"], item["answer"]))

    if mode == "demo":
        print("==== 阶段一：逐条推理（chat template） ====")
    else:
        print("==== 阶段一：逐条推理 ====")
    
    stage1_correct = 0
    stage1_start = time.time()
    with torch.inference_mode():
        for idx, (problem, gold) in enumerate(stage1_items, 1):
            t0 = time.time()
            raw_text = solve_single(model, tokenizer, problem, MAX_NEW_TOKENS, DO_SAMPLE)
            no_think_text = remove_thinking(raw_text)
            cleaned_text = remove_answer_prefix(no_think_text)
            t1 = time.time()
            ok, _, msg = check_format_and_value(cleaned_text, gold)
            
            if mode == "demo":
                print(f"【题目】{problem}")
                print(f"【模型原始输出】{raw_text}")
                print(f"【非思考部分文本】{no_think_text}")
                print(f"【提取答案】{cleaned_text}")
                print(msg)
                print(f"【耗时】{t1 - t0:.6f} 秒\n")
            else:
                # 评分模式：简洁输出进度
                status = "✓" if ok else "✗"
                print(f"  [{idx}/{len(stage1_items)}] {status}")
            
            if ok:
                stage1_correct += 1
    stage1_elapsed = time.time() - stage1_start
    score_report("阶段一（逐条）", len(stage1_items), stage1_correct, stage1_elapsed, mode)

    # 计算第一阶段单条平均耗时
    stage1_avg_time = stage1_elapsed / len(stage1_items) if stage1_items else float("inf")

    # ===== 阶段二：批量推理 =====
    print("==== 阶段二：批量推理 ====")
    # 从 data/test_data_stage2.jsonl 读取测试数据
    stage2_items: List[Tuple[str, str]] = []
    test_data_2_path = os.path.join(os.path.dirname(__file__), "data", "test_data_2.jsonl")
    with open(test_data_2_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                item = json.loads(line)
                stage2_items.append((item["problem"], item["answer"]))

    problems_batch = [p for p, _ in stage2_items]

    if mode == "grading":
        print("  正在进行批量推理...")

    with torch.inference_mode():
        stage2_start = time.time()
        raw_list = solve_batch(model, tokenizer, problems_batch, MAX_NEW_TOKENS, DO_SAMPLE)
        no_think_list = [remove_thinking(t) for t in raw_list]
        cleaned_list = [remove_answer_prefix(t) for t in no_think_list]
        stage2_elapsed = time.time() - stage2_start
    
    if mode == "grading":
        print(f"  批量推理完成，耗时：{stage2_elapsed:.3f} s")

    # 统计并展示结果
    stage2_correct = 0
    wrong_samples = []  # 存储错误样例
    
    # 先统计所有结果
    for idx, ((prob, gold), raw_text, no_think_text, cleaned_text) in enumerate(
        zip(stage2_items, raw_list, no_think_list, cleaned_list), 1
    ):
        ok, _, msg = check_format_and_value(cleaned_text, gold)
        
        if ok:
            stage2_correct += 1
        else:
            # 记录错误样例
            wrong_samples.append((idx, prob, gold, raw_text, no_think_text, cleaned_text, msg))
    
    # 演示模式：仅展示错误样例
    if mode == "demo":
        if wrong_samples:
            print("—— 阶段二错误样例（仅展示出错条目）——")
            for idx, prob, gold, raw_text, no_think_text, cleaned_text, msg in wrong_samples:
                print(f"【题目 {idx}/{len(stage2_items)}】{prob}")
                print(f"【模型原始输出】{raw_text}")
                print(f"【非思考部分文本】{no_think_text}")
                print(f"【提取答案】{cleaned_text}")
                print(msg)
                print()
        else:
            print("—— 阶段二全部正确，无错误样例。——\n")

    score_report("阶段二（批量32题）", len(stage2_items), stage2_correct, stage2_elapsed, mode)

    # 计算阶段二单条平均耗时
    stage2_avg_time = stage2_elapsed / len(stage2_items) if stage2_items else float("inf")
    
    if mode == "demo":
        print("==== 性能对比 ====")
        print(f"阶段一单条平均耗时：{stage1_avg_time*1000:.1f} ms/样本 ({stage1_avg_time:.3f} s)")
        print(f"阶段二单条平均耗时：{stage2_avg_time*1000:.1f} ms/样本 ({stage2_avg_time:.3f} s)")
        
        # 显示如果按评分模式的得分
        print(f"\n==== 如果按评分模式计分 ====")
        
        # 阶段一正确性：每题10分，≥6题满分60分
        if stage1_correct >= 6:
            stage1_accuracy_score = 60
        else:
            stage1_accuracy_score = stage1_correct * 10
        
        # 阶段二正确性：每题1.5分，满分40分
        stage2_accuracy_score = min(stage2_correct * 1.5, 40)
        
        # 总分
        total_score = stage1_accuracy_score + stage2_accuracy_score
        
        print(f"阶段一正确性：{stage1_accuracy_score:.1f}/60 ({stage1_correct}/8题，每题10分，≥6题满分)")
        print(f"阶段二正确性：{stage2_accuracy_score:.1f}/40 ({stage2_correct}/32题，每题1.5分，满分40分)")
        print(f"总分：{total_score:.1f}/100\n")
    else:
        # 评分模式：计算最终分数
        print("\n" + "="*50)
        print("========== 最终评分 ==========")
        print("="*50)
        
        # 阶段一正确性评分：每题10分，达到6题及以上给满分60分
        if stage1_correct >= 6:
            stage1_accuracy_score = 60
        else:
            stage1_accuracy_score = stage1_correct * 10
        
        # 阶段二正确性评分：每题1.5分，满分40分
        stage2_accuracy_score = min(stage2_correct * 1.5, 40)
        
        # 总分
        total_score = stage1_accuracy_score + stage2_accuracy_score
        
        print(f"\n【阶段一：逐条推理 - 正确性】")
        print(f"  正确题数：{stage1_correct}/{len(stage1_items)}")
        print(f"  单条平均耗时：{stage1_avg_time:.3f} s/样本")
        print(f"  得分：{stage1_accuracy_score:.1f}/60  (每题10分，≥6题满分)")
        
        print(f"\n【阶段二：批量推理 - 正确性】")
        print(f"  正确题数：{stage2_correct}/{len(stage2_items)}")
        print(f"  批量平均耗时：{stage2_avg_time:.3f} s/样本")
        print(f"  得分：{stage2_accuracy_score:.1f}/40  (每题1.5分，满分40分)")
        
        print(f"\n{'='*50}")
        print(f"【总分汇总】")
        print(f"  阶段一正确性：{stage1_accuracy_score:.1f}/60")
        print(f"  阶段二正确性：{stage2_accuracy_score:.1f}/40")
        print(f"  " + "-"*46)
        print(f"  总分：{total_score:.1f}/100")
        print(f"{'='*50}\n")
    

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="T1 评测程序 - 支持演示模式和评分模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["demo", "grading"],
        help="运行模式：demo（演示模式，详细输出）或 grading（评分模式，仅输出分数）"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(mode=args.mode if args.mode else "demo")
# -*- coding: utf-8 -*-
"""
T2 评测程序 - 支持两种运行模式

1. 演示模式（demo）：
   - 详细展示每个测试用例的结果
   - 显示标准答案和考生答案的对比
   - 显示数据多样性的详细统计
   - 适合调试和查看实现效果
   使用方法：
     python evaluate.py
     python evaluate.py --mode demo

2. 评分模式（grading）：
   - 简洁的进度输出
   - 不展示每个测试用例的详情
   - 最终输出详细评分
   - 适合正式评测
   使用方法：
     python evaluate.py --mode grading

评分规则：
  - 第一部分（相似度计算准确性）：50分
    * 10组测试用例，每组5分
    * 允许误差≤0.03
    * 必须满分才能进行第二部分
  - 第二部分（数据多样性）：50分
    * 使用考生模型计算数据集相似度
    * 数据质量扣分（每条问题扣1分）：
      - 重复题目：每条重复扣1分
      - 空题目：每条空题目扣1分
      - 数字超过3个：每条扣1分
      - 答案超过3位数：每条扣1分
    * 数量要求：
      - 标准数量：1024条
      - 不足或超出：每条扣1分
    * 平均相似度≤0.5：满分50分
    * 平均相似度0.5~0.7：线性给分
    * 平均相似度>0.7：0分
  - 总分：100分
"""

import os
import json
import time
import argparse
from typing import List, Dict, Tuple

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from submission import compute_similarity


# =========================
# 配置
# =========================
DATA_PATH = "./data/dataset.jsonl"
TEST_CASES_PATH = "./data/test_data_1.jsonl"
N_SAMPLES = 1024
TOLERANCE = 0.03  # 相似度允许的绝对误差
STUDENT_MODEL = "../Qwen/Qwen3-Embedding-0.6B"


# =========================
# 工具函数 - 加载测试用例
# =========================
def load_test_cases(path: str = TEST_CASES_PATH) -> List[Dict]:
    """
    从文件加载测试用例
    返回：[{"text1": ..., "text2": ..., "description": ..., "standard_similarity": ...}, ...]
    """
    
    test_cases = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            text1 = obj.get("text1", "")
            text2 = obj.get("text2", "")
            desc = obj.get("description", "")
            std_sim = obj.get("standard_similarity")  # 必须包含标准相似度
            
            if text1 and text2 and std_sim is not None:
                test_cases.append({
                    "text1": text1,
                    "text2": text2,
                    "description": desc,
                    "standard_similarity": std_sim
                })
    
    return test_cases


# =========================
# 工具函数
# =========================
def load_student_model():
    """加载考生使用的模型（transformers）"""
    tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL, padding_side="left", trust_remote_code=True)
    try:
        model = AutoModel.from_pretrained(STUDENT_MODEL, trust_remote_code=True)
    except TypeError:
        model = AutoModel.from_pretrained(STUDENT_MODEL)
    
    # 获取设备
    if hasattr(torch, "npu") and hasattr(torch.npu, "is_available") and torch.npu.is_available():
        device = torch.device("npu:0")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    
    model.to(device)
    model.eval()
    
    return model, tokenizer


def count_numbers_in_text(text: str) -> int:
    """统计文本中的数字个数（连续的数字算作一个）"""
    import re
    # 匹配所有数字（整数和小数）
    numbers = re.findall(r'\d+\.?\d*', text)
    return len(numbers)


def is_valid_answer(answer: str) -> bool:
    """检查答案是否符合要求（最多3位数）"""
    import re
    # 提取答案中的数字部分
    numbers = re.findall(r'-?\d+\.?\d*', answer)
    if not numbers:
        return False
    
    # 检查主要数字是否为3位数以内
    main_number = numbers[0]
    # 去掉小数点和负号
    digits = main_number.replace('.', '').replace('-', '')
    # 小数的整数部分不超过3位
    if '.' in main_number:
        integer_part = main_number.split('.')[0].replace('-', '')
        return len(integer_part) <= 3
    else:
        return len(digits) <= 3


def load_dataset(data_path: str, n_required: int) -> List[Dict[str, str]]:
    """加载数据集"""
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在：{data_path}")
    
    items = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if "problem" not in obj:
                raise ValueError("数据格式错误：缺少 'problem' 字段")
            items.append({
                "problem": str(obj["problem"]).strip(),
                "answer": str(obj.get("answer", "")).strip()
            })
    
    # 只保留非空题面
    items = [it for it in items if it["problem"]]
    
    return items


def compute_pairwise_similarity_batch(model: AutoModel, tokenizer: AutoTokenizer, texts: List[str], batch_size: int = 32) -> Tuple[float, float, float]:
    """
    使用考生模型计算所有文本两两之间的平均相似度（仅上三角 i < j）
    使用批量编码以提高效率
    
    Args:
        model: 考生的 transformers 模型
        tokenizer: 考生的 tokenizer
        texts: 文本列表
        batch_size: 批处理大小
    
    Returns:
        (avg_similarity, min_similarity, max_similarity)
    """
    # 获取设备
    device = next(model.parameters()).device
    
    # 批量编码所有文本
    all_embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(device)
        
        # 获取嵌入
        with torch.no_grad():
            outputs = model(**inputs)
            # Last token pooling
            last_hidden_state = outputs.last_hidden_state
            batch_embeddings = last_hidden_state[:, -1, :]
            # L2 normalization
            batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
            all_embeddings.append(batch_embeddings.cpu())
    
    # 合并所有批次的嵌入
    embeddings = torch.cat(all_embeddings, dim=0)
    
    # 计算相似度矩阵
    sim_matrix = embeddings @ embeddings.T
    
    # 只取上三角
    n = len(texts)
    i_idx, j_idx = torch.triu_indices(n, n, offset=1)
    similarities = sim_matrix[i_idx, j_idx]
    
    avg_similarity = float(similarities.mean().item())
    min_similarity = float(similarities.min().item())
    max_similarity = float(similarities.max().item())
    
    return avg_similarity, min_similarity, max_similarity


# =========================
# 评测函数
# =========================
def test_similarity_accuracy(stu_model: AutoModel, stu_tokenizer: AutoTokenizer, mode: str = "demo") -> Tuple[int, List[Dict]]:
    """
    第一部分：测试相似度计算的准确性
    返回：(得分, 测试结果列表)
    
    注意：标准相似度从测试用例文件中读取，不需要传入标准模型
    """
    if mode == "demo":
        print("=" * 60)
        print("第一部分：相似度计算准确性测试")
        print("=" * 60)
        print()
    else:
        print("==== 第一部分：相似度计算准确性 ====")
    
    # 加载测试用例
    test_cases = load_test_cases()
    if mode == "demo":
        print(f"加载了 {len(test_cases)} 个测试用例")
        print()
    
    results = []
    passed = 0
    
    for i, case in enumerate(test_cases, 1):
        text1 = case["text1"]
        text2 = case["text2"]
        desc = case["description"]
        
        # 标准答案：直接使用预计算的标准相似度
        std_sim = case["standard_similarity"]
        
        # 考生答案
        try:
            stu_sim = compute_similarity(text1, text2, stu_model, stu_tokenizer)
            if not isinstance(stu_sim, (int, float)):
                raise TypeError(f"返回值类型错误：期望 float，实际 {type(stu_sim)}")
            # 允许浮点数精度误差（1e-6）
            if not (-1e-6 <= stu_sim <= 1.0 + 1e-6):
                raise ValueError(f"返回值超出范围：{stu_sim}（应在 0.0 到 1.0 之间）")
        except Exception as e:
            if mode == "demo":
                print(f"测试用例 {i:2d}: ❌ 失败 - 执行出错")
                print(f"  描述: {desc}")
                print(f"  text1: {text1}")
                print(f"  text2: {text2}")
                print(f"  错误: {e}")
                print()
            else:
                print(f"  [{i}/{len(test_cases)}] ✗ ({desc}) - 执行出错")
            
            results.append({
                "case": i,
                "description": desc,
                "text1": text1,
                "text2": text2,
                "passed": False,
                "error": str(e)
            })
            continue
        
        # 计算误差
        diff = abs(std_sim - stu_sim)
        is_passed = diff <= TOLERANCE
        
        if is_passed:
            passed += 1
            status_symbol = "✅" if mode == "demo" else "✓"
        else:
            status_symbol = "❌" if mode == "demo" else "✗"
        
        if mode == "demo":
            print(f"测试用例 {i:2d}: {status_symbol} {'通过' if is_passed else '失败'} ({desc})")
            print(f"  text1: {text1}")
            print(f"  text2: {text2}")
            print(f"  标准答案: {std_sim:.6f}")
            print(f"  考生答案: {stu_sim:.6f}")
            print(f"  误差: {diff:.6f} (容差: {TOLERANCE})")
            print()
        else:
            print(f"  [{i}/{len(test_cases)}] {status_symbol} ({desc})")
        
        results.append({
            "case": i,
            "description": desc,
            "text1": text1,
            "text2": text2,
            "std_sim": std_sim,
            "stu_sim": stu_sim,
            "diff": diff,
            "passed": is_passed
        })
    
    # 计算得分
    # 每个测试用例 5 分，通过 >= 8 个得满分
    total_cases = len(test_cases)
    score = min(50, passed * 5)
    
    if mode == "demo":
        print("-" * 60)
        print(f"通过：{passed} / {total_cases}")
        print(f"得分：{score} / 50")
        print()
    
    return score, results


def test_data_diversity(stu_model: AutoModel, stu_tokenizer: AutoTokenizer, mode: str = "demo") -> Tuple[int, Dict]:
    """
    第二部分：测试数据多样性
    返回：(得分, 统计信息)
    
    注意：使用考生的模型计算数据集相似度
    """
    if mode == "demo":
        print("=" * 60)
        print("第二部分：数据多样性测试")
        print("=" * 60)
        print()
    else:
        print("==== 第二部分：数据多样性 ====")
    
    # 加载数据
    try:
        items = load_dataset(DATA_PATH, N_SAMPLES)
    except Exception as e:
        print(f"❌ 数据加载失败：{e}")
        return 0, {"error": str(e)}
    
    # 检查数据数量（按缺少数量扣分，每少/多1条扣1分）
    quantity_penalty = 0
    if len(items) != N_SAMPLES:
        shortage = N_SAMPLES - len(items)
        quantity_penalty = abs(shortage)  # 每条扣1分，不设上限
        
        if mode == "demo":
            print(f"数据数量：{len(items)} 条（期望 {N_SAMPLES} 条）")
            if shortage > 0:
                print(f"⚠️  数据不足 {shortage} 条，将扣 {quantity_penalty} 分")
            else:
                print(f"⚠️  数据超出 {abs(shortage)} 条，将扣 {quantity_penalty} 分")
            print()
        else:
            if shortage > 0:
                print(f"  数据不足 {shortage} 条，将扣 {quantity_penalty} 分")
            else:
                print(f"  数据超出 {abs(shortage)} 条，将扣 {quantity_penalty} 分")
    else:
        if mode == "demo":
            print(f"数据数量：{len(items)} 条")
            print("✅ 数据数量符合要求")
            print()
    
    # 提取题目列表
    problems = [it["problem"] for it in items]
    unique_count = len(set(problems))
    
    # 初始化数据质量扣分
    quality_penalty = 0
    quality_issues = []
    
    # 检查唯一性（每条重复扣1分）
    if unique_count != len(problems):
        duplicate_count = len(problems) - unique_count
        quality_penalty += duplicate_count
        quality_issues.append(f"重复题目：{duplicate_count}条")
        
        if mode == "demo":
            print(f"⚠️  存在重复题目（{duplicate_count} 条），将扣 {duplicate_count} 分")
            # 找出重复的题目
            from collections import Counter
            problem_counter = Counter(problems)
            duplicates = [(prob, count) for prob, count in problem_counter.items() if count > 1]
            if duplicates:
                print("重复题目示例（前5条）：")
                for prob, count in duplicates[:5]:
                    print(f"  '{prob}' 出现了 {count} 次")
            print()
        else:
            print(f"  ⚠️  {duplicate_count} 条重复题目，将扣 {duplicate_count} 分")
    else:
        if mode == "demo":
            print(f"✅ 所有题目唯一（{unique_count}/{len(problems)}）")
            print()
    
    # 检查非空（每条空题目扣1分）
    empty_count = sum(1 for p in problems if not p)
    if empty_count > 0:
        quality_penalty += empty_count
        quality_issues.append(f"空题目：{empty_count}条")
        
        if mode == "demo":
            print(f"⚠️  存在空题目（{empty_count} 条），将扣 {empty_count} 分")
            print()
        else:
            print(f"  ⚠️  {empty_count} 条空题目，将扣 {empty_count} 分")
    
    # 检查数字个数（每个问题不超过3个数字，每条超标扣1分）
    invalid_problems = []
    for idx, item in enumerate(items, 1):
        num_count = count_numbers_in_text(item["problem"])
        if num_count > 3:
            invalid_problems.append((idx, item["problem"], num_count))
    
    if invalid_problems:
        quality_penalty += len(invalid_problems)
        quality_issues.append(f"数字超过3个：{len(invalid_problems)}条")
        
        if mode == "demo":
            print(f"⚠️  存在问题中数字超过3个的题目（{len(invalid_problems)} 条），将扣 {len(invalid_problems)} 分")
            print("示例（前5条）：")
            for idx, prob, count in invalid_problems[:5]:
                print(f"  {idx}. {prob} (包含{count}个数字)")
            print()
        else:
            print(f"  ⚠️  {len(invalid_problems)} 条题目数字超过3个，将扣 {len(invalid_problems)} 分")
    
    # 检查答案位数（答案最多3位数，每条超标扣1分）
    invalid_answers = []
    for idx, item in enumerate(items, 1):
        if not is_valid_answer(item["answer"]):
            invalid_answers.append((idx, item["problem"], item["answer"]))
    
    if invalid_answers:
        quality_penalty += len(invalid_answers)
        quality_issues.append(f"答案超过3位数：{len(invalid_answers)}条")
        
        if mode == "demo":
            print(f"⚠️  存在答案超过3位数的题目（{len(invalid_answers)} 条），将扣 {len(invalid_answers)} 分")
            print("示例（前5条）：")
            for idx, prob, ans in invalid_answers[:5]:
                print(f"  {idx}. {prob} = {ans}")
            print()
        else:
            print(f"  ⚠️  {len(invalid_answers)} 条题目答案超过3位数，将扣 {len(invalid_answers)} 分")
    
    # 显示数据质量总扣分
    if quality_penalty > 0:
        if mode == "demo":
            print(f"数据质量总扣分：{quality_penalty} 分")
            print(f"  - " + "\n  - ".join(quality_issues))
            print()
    
    # 计算平均相似度
    if mode == "demo":
        print("正在计算平均相似度...")
        print("（这可能需要几分钟时间）")
        print()
    else:
        print("  正在计算平均相似度...")
    
    start_time = time.time()
    avg_sim, min_sim, max_sim = compute_pairwise_similarity_batch(stu_model, stu_tokenizer, problems)
    elapsed = time.time() - start_time
    
    if mode == "grading":
        print(f"  计算完成，耗时：{elapsed:.2f} 秒")
    
    if mode == "demo":
        print(f"平均相似度：{avg_sim:.6f}")
        print(f"最小相似度：{min_sim:.6f}")
        print(f"最大相似度：{max_sim:.6f}")
        print(f"计算耗时：{elapsed:.2f} 秒")
        print()
    
    # 计算相似度得分
    # 平均相似度 <= 0.5: 满分 50
    # 0.5 < 平均相似度 <= 0.7: 线性给分
    # 平均相似度 > 0.7: 0 分
    if avg_sim <= 0.5:
        similarity_score = 50
        if mode == "demo":
            print(f"✅ 优秀！平均相似度 ≤ 0.5，相似度得分 50")
    elif avg_sim <= 0.7:
        similarity_score = int(50 * (0.7 - avg_sim) / 0.2)
        if mode == "demo":
            print(f"⚠️  平均相似度在 0.5~0.7 之间，相似度得分 {similarity_score}")
    else:
        similarity_score = 0
        if mode == "demo":
            print(f"❌ 平均相似度 > 0.7，相似度得分 0")
    
    # 最终得分 = 相似度得分 - 数量扣分 - 数据质量扣分（最低为0）
    total_penalty = quantity_penalty + quality_penalty
    score = max(0, similarity_score - total_penalty)
    
    if mode == "demo":
        if total_penalty > 0:
            print(f"总扣分明细：")
            if quantity_penalty > 0:
                print(f"  - 数量扣分：{quantity_penalty} 分")
            if quality_penalty > 0:
                print(f"  - 数据质量扣分：{quality_penalty} 分")
            print(f"  - 总计：{total_penalty} 分")
            print(f"最终得分：{similarity_score} - {total_penalty} = {score} / 50")
        else:
            print(f"得分：{score} / 50")
        print()
    
    stats = {
        "count": len(items),
        "unique": unique_count,
        "quantity_penalty": quantity_penalty,
        "quality_penalty": quality_penalty,
        "total_penalty": total_penalty,
        "quality_issues": quality_issues,
        "similarity_score": similarity_score,
        "avg_similarity": avg_sim,
        "min_similarity": min_sim,
        "max_similarity": max_sim,
        "elapsed": elapsed
    }
    
    return score, stats


def run_evaluation(mode: str = "demo"):
    """
    运行完整评测
    
    Args:
        mode: 运行模式，"demo" 或 "grading"
    """
    if mode not in ["demo", "grading"]:
        print(f"⚠️  警告：未知的运行模式 '{mode}'，使用默认模式 'demo'")
        mode = "demo"
    
    print()
    if mode == "demo":
        print("=" * 60)
        print("演示模式 - T2 多样性数据生成与相似度计算")
        print("=" * 60)
    else:
        print("=" * 60)
        print("评分模式 - T2 多样性数据生成与相似度计算")
        print("=" * 60)
    print()
    
    # 加载考生模型
    print("正在加载考生模型（transformers）...")
    stu_model, stu_tokenizer = load_student_model()
    print("✅ 考生模型加载完成")
    print()
    
    # 第一部分：相似度计算准确性（使用预计算的标准相似度）
    score1, results1 = test_similarity_accuracy(stu_model, stu_tokenizer, mode)
    
    # 第二部分：数据多样性（只有第一部分满分才进行）
    if score1 == 50:
        if mode == "demo":
            print()
            print("✅ 第一部分满分，继续进行第二部分测试")
            print()
        else:
            print("  ✅ 第一部分满分，继续第二部分")
        
        score2, results2 = test_data_diversity(stu_model, stu_tokenizer, mode)
    else:
        if mode == "demo":
            print()
            print("=" * 60)
            print(f"❌ 第一部分未满分（{score1}/50），跳过第二部分测试")
            print("=" * 60)
            print()
        else:
            print(f"  ❌ 第一部分未满分（{score1}/50），跳过第二部分")
        
        score2 = 0
        results2 = {"skipped": True, "reason": "第一部分未满分"}
    
    # 总分
    total_score = score1 + score2
    
    # 输出最终结果
    if mode == "demo":
        print()
        print("=" * 60)
        print("评测结果汇总")
        print("=" * 60)
        print(f"第一部分（相似度计算准确性）：{score1} / 50")
        print(f"第二部分（数据多样性）：      {score2} / 50")
        print(f"总分：                        {total_score} / 100")
        print("=" * 60)
        print()
    else:
        # 评分模式：详细输出
        print()
        print("=" * 60)
        print("========== 最终评分 ==========")
        print("=" * 60)
        
        # 第一部分详情
        print("\n【第一部分：相似度计算准确性】")
        passed = sum(1 for r in results1 if r.get("passed", False))
        total_cases = len(results1)
        print(f"  通过测试用例：{passed} / {total_cases}")
        print(f"  得分：{score1} / 50  (每组5分，≥8组满分)")
        
        # 第二部分详情
        print("\n【第二部分：数据多样性】")
        if results2.get("skipped"):
            print(f"  ⚠️  跳过：{results2.get('reason', '未知原因')}")
            print(f"  得分：{score2} / 50")
        elif "error" not in results2:
            print(f"  数据数量：{results2['count']} 条（要求：{N_SAMPLES}）")
            print(f"  唯一性：{results2['unique']} / {results2['count']}")
            
            # 显示扣分明细
            quantity_pen = results2.get('quantity_penalty', 0)
            quality_pen = results2.get('quality_penalty', 0)
            total_pen = results2.get('total_penalty', 0)
            
            if quantity_pen > 0:
                shortage = N_SAMPLES - results2['count']
                print(f"  数量扣分：-{quantity_pen} 分（{'不足' if shortage > 0 else '超出'} {abs(shortage)} 条）")
            
            if quality_pen > 0:
                print(f"  数据质量扣分：-{quality_pen} 分")
                if results2.get('quality_issues'):
                    for issue in results2['quality_issues']:
                        print(f"    · {issue}")
            
            print(f"  平均相似度：{results2['avg_similarity']:.6f}")
            print(f"  最小相似度：{results2['min_similarity']:.6f}")
            print(f"  最大相似度：{results2['max_similarity']:.6f}")
            
            sim_score = results2.get('similarity_score', 0)
            print(f"  相似度得分：{sim_score} / 50")
            
            if total_pen > 0:
                print(f"  总扣分：-{total_pen} 分")
                print(f"  最终得分：{sim_score} - {total_pen} = {score2} / 50")
            else:
                print(f"  最终得分：{score2} / 50")
        elif results2.get("error"):
            print(f"  错误：{results2['error']}")
            print(f"  得分：{score2} / 50")
        else:
            print(f"  得分：{score2} / 50")
        
        # 总分
        print(f"\n{'='*60}")
        print(f"【总分汇总】")
        print(f"  第一部分（相似度计算准确性）：{score1} / 50")
        print(f"  第二部分（数据多样性）：      {score2} / 50")
        print(f"  " + "-" * 56)
        print(f"  总分：                        {total_score} / 100")
        print(f"{'='*60}\n")


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="T2 评测程序 - 支持演示模式和评分模式",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        choices=["demo", "grading"],
        default="demo",
        help="运行模式：demo（演示模式，详细输出）或 grading（评分模式，简洁输出）"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_evaluation(mode=args.mode)


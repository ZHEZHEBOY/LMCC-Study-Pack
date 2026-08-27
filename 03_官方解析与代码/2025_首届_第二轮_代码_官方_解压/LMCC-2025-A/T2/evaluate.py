# -*- coding: utf-8 -*-
"""
T2 数学作业解答助手 - 评测主程序
本文件不可修改

环境要求：
    在运行本程序前，请确保已激活正确的 MindSpore 环境。
    
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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel
import transformers
import random

# 设置随机种子以确保结果可复现
SEED = 20251115
random.seed(SEED)
torch.manual_seed(SEED)

# 设置 transformers 日志级别，隐藏警告信息
transformers.logging.set_verbosity_error()

# 导入考生实现的函数
from submission import (
    build_system_prompt, 
    build_generation_parameters, 
    retrieve_relevant_problems,
    build_user_message,
    compute_similarity
)


# ============================================================
# 工具函数
# ============================================================
def print_memory_info(model: AutoModelForCausalLM):
    """显示模型显存使用情况"""
    print("\n【显存使用情况】")
    try:
        device = next(model.parameters()).device
        
        if torch.cuda.is_available() and device.type == 'cuda':
            allocated = torch.cuda.memory_allocated(device) / 1024**3
            reserved = torch.cuda.memory_reserved(device) / 1024**3
            max_allocated = torch.cuda.max_memory_allocated(device) / 1024**3
            print(f"  设备: {device}")
            print(f"  已分配显存: {allocated:.2f} GB")
            print(f"  已保留显存: {reserved:.2f} GB")
            print(f"  峰值显存: {max_allocated:.2f} GB")
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
            model_size_gb = total_params * 2 / 1024**3
            print(f"  设备: {device}")
            print(f"  模型参数量: {total_params:,}")
            print(f"  可训练参数: {trainable_params:,}")
            print(f"  模型大小（估算）: {model_size_gb:.2f} GB (float16)")
    except Exception as e:
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


def get_embedding(text: str, model, tokenizer) -> torch.Tensor:
    """使用 embedding 模型获取文本的向量表示"""
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        # 使用 mean pooling（按照 Qwen3-Embedding 官方推荐）
        attention_mask = inputs['attention_mask']
        token_embeddings = outputs.last_hidden_state
        # Mean pooling: 对所有token取平均（考虑attention mask）
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        embedding = (sum_embeddings / sum_mask).squeeze()
    
    # 保持在模型设备上（NPU/GPU），提高性能
    return embedding


def test_similarity_function(
    embedding_model,
    embedding_tokenizer,
    similarity_test_data: List[Dict],
    problem_bank: List[Dict],
    mode: str
) -> Tuple[int, int]:
    """
    测试相似度计算函数（20分）
    
    参数：
        embedding_model: embedding 模型
        embedding_tokenizer: embedding tokenizer
        similarity_test_data: 相似度测试数据
        problem_bank: 题库
        mode: 运行模式
    
    返回：
        (正确数, 总题数)
    """
    print("\n" + "="*60)
    print("第一部分：相似度计算测试（20分）")
    print("="*60)
    
    correct = 0
    total = len(similarity_test_data)
    
    # 预先计算所有题库中题目的 embedding
    if mode == "demo":
        print("\n正在计算题库中所有题目的 embedding...")
    
    problem_embeddings = []
    for prob in problem_bank:
        emb = get_embedding(prob['problem'], embedding_model, embedding_tokenizer)
        problem_embeddings.append(emb)
    
    if mode == "demo":
        print(f"题库 embedding 计算完成！共 {len(problem_embeddings)} 个题目")
    
    for idx, item in enumerate(similarity_test_data, 1):
        query = item['query']
        expected_id = item['expected_id']
        
        # 计算 query 的 embedding
        query_emb = get_embedding(query, embedding_model, embedding_tokenizer)
        
        # 使用考生的相似度函数计算与所有题目的相似度
        similarities = []
        try:
            for prob_idx, prob_emb in enumerate(problem_embeddings):
                sim = compute_similarity(query_emb, prob_emb)
                similarities.append((sim, prob_idx))
        except Exception as e:
            if mode == "demo":
                print(f"\n【相似度测试 {idx}/{total}】")
                print(f"查询: {query}")
                print(f"❌ 相似度计算出错: {e}")
            continue
        
        # 按相似度降序排序
        similarities.sort(key=lambda x: x[0], reverse=True)
        
        # 获取最相似的题目ID
        if similarities:
            predicted_id = problem_bank[similarities[0][1]]['id']
            is_correct = (predicted_id == expected_id)
            
            if is_correct:
                correct += 1
            
            if mode == "demo":
                print(f"\n【相似度测试 {idx}/{total}】")
                print(f"查询: {query}")
                print(f"期望找到的题目ID: {expected_id}")
                print(f"实际找到的题目ID: {predicted_id}")
                print(f"最高相似度: {similarities[0][0]:.4f}")
                if not is_correct:
                    print(f"期望题目: {problem_bank[expected_id-1]['problem']}")
                    print(f"实际题目: {problem_bank[predicted_id-1]['problem']}")
                print(f"判定: {'✅ 正确' if is_correct else '❌ 错误'}")
        elif mode == "demo":
            print(f"\n【相似度测试 {idx}/{total}】")
            print(f"查询: {query}")
            print(f"❌ 未找到相似题目")
    
    # 计算相似度测试得分
    similarity_score = correct
    
    print(f"\n{'='*60}")
    print(f"相似度测试结果：{correct}/{total} 正确")
    print(f"相似度测试得分：{similarity_score}/20 分")
    print(f"{'='*60}")
    
    return correct, total


def extract_json(text: str) -> Optional[Dict]:
    """从模型输出中提取JSON格式的答案"""
    text = re.sub(r'```json\s*', '', text)
    text = re.sub(r'```\s*', '', text)
    
    start_idx = text.find('{')
    if start_idx == -1:
        return None
    
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
        except json.JSONDecodeError:
            json_str_clean = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', json_str)
            try:
                return json.loads(json_str_clean)
            except json.JSONDecodeError:
                pass
    
    patterns = [
        r'\{[^{}]*"reasoning"[^{}]*"answer"[^{}]*\}',
        r'\{.*?"reasoning".*?"answer".*?\}',
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


def extract_answer(text: str) -> str:
    """从模型输出中提取答案"""
    result_json = extract_json(text)
    if result_json and "answer" in result_json:
        return str(result_json["answer"]).strip()
    return ""


def remove_thinking_tags(text: str) -> str:
    """移除 <think>...</think> 标签及其内容"""
    return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)


def save_evaluation_record(task_name: str, correct: int, total: int, score: int, total_time: float):
    """保存评测记录到用户目录的 .lmcc 文件夹"""
    try:
        home_dir = Path.home()
        lmcc_dir = home_dir / ".lmcc"
        lmcc_dir.mkdir(exist_ok=True)
        
        record_file = lmcc_dir / "evaluation_records.jsonl"
        
        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "task": task_name,
            "correct": correct,
            "total": total,
            "score": score,
            "total_time": round(total_time, 2)
        }
        
        with open(record_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
        print(f"\n✅ 评测记录已保存到: {record_file}")
    except Exception as e:
        print(f"\n⚠ 保存评测记录失败: {e}")


def format_problem(item: Dict) -> str:
    """将题目格式化为字符串"""
    return item["problem"]


def apply_chat_template(
    tokenizer: AutoTokenizer,
    system_prompt: str,
    user_message: str,
    enable_thinking: bool = False,
) -> str:
    """将问题转换为模型输入文本"""
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_message},
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
    """单条推理（半精度、推理模式）"""
    model.eval()
    
    inputs = tokenizer(rendered_text, return_tensors="pt", padding=True).to(model.device)
    input_length = inputs["input_ids"].shape[1]
    
    gen_kwargs = {
        "input_ids": inputs["input_ids"],
        "attention_mask": inputs.get("attention_mask"),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    
    gen_kwargs.update(generation_params)
    gen_kwargs.pop("enable_thinking", None)
    
    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)
    
    return outputs, input_length


# ============================================================
# RAG 评测函数
# ============================================================
def evaluate_rag(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    embedding_model,
    embedding_tokenizer,
    system_prompt: str,
    test_data: List[Dict],
    problem_bank: List[Dict],
    generation_params: Dict,
    mode: str,
) -> Tuple[int, int, float]:
    """
    RAG 问答评测（6题，每题5分，共30分）
    """
    print("\n" + "="*60)
    print("第二部分：RAG 问答测试（30分）")
    print("="*60)
    
    correct = 0
    total = len(test_data)
    total_time = 0.0
    
    enable_thinking = generation_params.get("enable_thinking", False)
    
    for idx, item in enumerate(test_data, 1):
        problem = format_problem(item)
        expected = item["answer"]
        
        # 步骤1：检索相关题目
        try:
            retrieved_problems = retrieve_relevant_problems(
                problem, problem_bank, embedding_model, embedding_tokenizer, top_k=3,
                get_embedding_func=get_embedding  # 传入 get_embedding 函数供考生使用
            )
        except Exception as e:
            if mode == "demo":
                print(f"\n【题目 {idx}/{total}】")
                print(f"题面：{problem}")
                print(f"❌ 检索出错: {e}")
            retrieved_problems = []
        
        # 步骤2：组装用户消息
        user_message = build_user_message(problem, retrieved_problems)
        
        # 步骤3：拼装输入
        rendered_text = apply_chat_template(
            tokenizer, system_prompt, user_message, enable_thinking
        )
        
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
        
        generated_ids = output_ids[0, input_length:]
        generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
        
        non_thinking_text = remove_thinking_tags(generated_text)
        
        # 提取答案
        predicted = extract_answer(non_thinking_text)
        predicted_json = extract_json(non_thinking_text)
        
        # 判断正确性
        is_correct = (predicted == expected)
        if is_correct:
            correct += 1
        
        # 输出
        if mode == "demo":
            print(f"\n【题目 {idx}/{total}】")
            print(f"题面：{problem}")
            print(f"\n检索到的相似题目数量：{len(retrieved_problems)}")
            if retrieved_problems:
                print("检索结果：")
                for i, prob in enumerate(retrieved_problems, 1):
                    print(f"  {i}. ID={prob['id']}: {prob['problem'][:50]}...")
                    print(f"     答案：{prob['answer']}")
            
            print(f"\n{'='*70}")
            print("【用户消息（部分）】")
            print(f"{'='*70}")
            print(user_message[:500] + "..." if len(user_message) > 500 else user_message)
            print(f"{'='*70}")
            
            print(f"\n提取的JSON：")
            if predicted_json:
                print(json.dumps(predicted_json, ensure_ascii=False, indent=2))
            else:
                print("❌ 无法提取JSON")
            
            print(f"\n正确答案：{expected}")
            print(f"提取的答案：{predicted if predicted else '(未提取到)'}")
            print(f"判定：{'✅ 正确' if is_correct else '❌ 错误'}")
            print(f"耗时：{elapsed:.3f}s")
        elif mode == "grading":
            status = "✅" if is_correct else "❌"
            print(f"题目 {idx}/{total}: {status}", flush=True)
    
    if mode == "grading":
        print()
    
    return correct, total, total_time


# ============================================================
# 主函数
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="T2 数学作业解答助手- 评测程序")
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
    print("T2 数学作业解答助手 - 评测程序")
    print("="*60)
    print(f"运行模式：{mode}")
    
    # 加载题库
    print("\n正在加载题库...")
    data_dir = Path(__file__).parent.parent / "data"
    with open(data_dir / "problem_bank.json", "r", encoding="utf-8") as f:
        problem_bank = json.load(f)
    print(f"题库加载完成！共 {len(problem_bank)} 道题")
    
    # 加载 Embedding 模型
    print("\n正在加载 Embedding 模型...")
    embedding_model_path = os.environ.get("EMBEDDING_MODEL_NAME", "/home/ma-user/Qwen/Qwen/Qwen3-Embedding-0.6B")
    embedding_tokenizer = AutoTokenizer.from_pretrained(embedding_model_path, trust_remote_code=True)
    embedding_model = AutoModel.from_pretrained(
        embedding_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    embedding_model.eval()
    for param in embedding_model.parameters():
        param.requires_grad = False
    print("Embedding 模型加载完成！")
    
    # 加载 LLM 模型
    print("\n正在加载 LLM 模型...")
    model_path = os.environ.get("MODEL_NAME", "/home/ma-user/Qwen/Qwen/Qwen3-0.6B")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    print("LLM 模型加载完成！")
    
    # 显示显存信息
    print_memory_info(model)
    
    # 测试1：相似度计算（20分）
    similarity_test_data = load_jsonl(data_dir / "similarity_test_T2.jsonl")
    sim_correct, sim_total = test_similarity_function(
        embedding_model, embedding_tokenizer, similarity_test_data, problem_bank, mode
    )
    similarity_score = sim_correct  # 每题1分，共20题
    
    # 测试2：RAG 问答（30分）
    # 构建 system prompt
    system_prompt = build_system_prompt()
    
    # 检查 system_prompt token 数量
    system_prompt_tokens = tokenizer.encode(system_prompt, add_special_tokens=False)
    system_prompt_token_count = len(system_prompt_tokens)
    max_system_prompt_tokens = 2048
    
    if mode == "demo":
        print(f"\nSystem Prompt Token 数量：{system_prompt_token_count}/{max_system_prompt_tokens}")
    
    if system_prompt_token_count > max_system_prompt_tokens:
        print(f"\n❌ 错误：System Prompt 超过限制！")
        print(f"   当前 token 数：{system_prompt_token_count}")
        print(f"   限制：{max_system_prompt_tokens}")
        return
    
    # 获取生成参数
    generation_params = build_generation_parameters()
    
    # 加载测试数据（6题）
    test_data = load_jsonl(data_dir / "test_data_T2.jsonl")
    
    # RAG 评测
    rag_correct, rag_total, total_time = evaluate_rag(
        model, tokenizer, embedding_model, embedding_tokenizer,
        system_prompt, test_data, problem_bank, generation_params, mode
    )
    
    # 计算 RAG 得分：每题5分
    rag_score = rag_correct * 5
    
    # 总分
    total_score = similarity_score + rag_score
    
    # 输出总结
    print("\n" + "="*60)
    print("评测总结")
    print("="*60)
    print(f"第一部分（相似度计算）：{sim_correct}/{sim_total} 正确，得分 {similarity_score}/20")
    print(f"第二部分（RAG问答）：{rag_correct}/{rag_total} 正确，得分 {rag_score}/30")
    print(f"总耗时：{total_time:.2f}s")
    print(f"总得分：{total_score}/50")
    print("="*60)
    
    # 保存评测记录
    save_evaluation_record(
        task_name="T2-数学作业解答助手",
        correct=sim_correct + rag_correct,
        total=sim_total + rag_total,
        score=total_score,
        total_time=total_time
    )


if __name__ == "__main__":
    main()

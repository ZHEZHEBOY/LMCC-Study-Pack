# 05 第二轮上机环境

## 最重要的结论

**截至 2026-08-27，CCF 尚未公开一份可确认的“2026 成人组第二轮最终软件版本表 / 考场镜像说明”。** 已公开资料存在 2025 MindSpore 路线和 2025 PyTorch 路线，不能把任一旧版本直接宣称为 2026 最终环境。

## 证据矩阵

| 资料 | 明确写了什么 | 可用于什么 | 不能据此声称什么 |
|---|---|---|---|
| [2025 第二轮模拟练习平台](https://lmcc.ccf.org.cn/101/1002/10145.html) | Python ≥ 3.9；MindSpore 2.7.0；MindNLP 0.5.0；Qwen3-0.6B、Qwen3-Embedding-0.6B | 复现 2025 官方例题 / 模拟环境 | 不能称为 2026 正式考场环境 |
| [2025 正式第二轮代码](../03_官方解析与代码/2025_首届_第二轮_代码_官方_解压/LMCC-2025-A/) | `考试必读.md` 要求激活名为 `MindSpore` 的 conda 环境；代码实际导入 PyTorch / Transformers / MindNLP 组件 | 理解 2025 正式工作流、评测器、提交接口 | 不能推导出 2026 精确包版本 |
| [2026 官方培训讲义](../04_官方课程与回放/2026_线上培训_第二轮上机环境配置与真题选讲_徐保毅_官方.pdf) | 讲义第 3 页明确写“**2025 必须使用 PyTorch 2.6.0**”；本地练习建议 Python ≥ 3.9、PyTorch、Transformers、VS Code 和两款 Qwen 模型 | 按 2025 真题路线练习 | 课程在 2026 发布，但这句话仍是 2025 历史要求，不是 2026 最终通知 |
| [最新考点规范](https://lmcc.ccf.org.cn/101/1003/10107.html) | 第二轮建议 CPU i5+、内存 8GB（2026 年 5 月更新）、硬盘 500GB；基于第一轮 Windows 10+ 配置 | 判断考点硬件级别 | 不是 Python / CUDA / 框架 / 模型版本清单 |

## 官方模型入口

- Generation Model（生成模型）：[Qwen3-0.6B · Hugging Face](https://huggingface.co/Qwen/Qwen3-0.6B) · [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-0.6B)
- Embedding Model（向量模型）：[Qwen3-Embedding-0.6B · Hugging Face](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B) · [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen3-Embedding-0.6B)
- 官方培训讲义中的 ModelScope 下载示例：

```text
git clone https://www.modelscope.cn/Qwen/Qwen3-0.6B.git
git clone https://www.modelscope.cn/Qwen/Qwen3-Embedding-0.6B.git
```

模型权重没有放进资料包：文件体积大，而且 CCF 尚未确认它们仍是 2026 正式第二轮指定模型。先读题和代码；等 2026 最终通知后再下载、锁版本。

## 推荐的练习顺序

1. 用官方第二轮例题包跑通 T1 / T2，理解 `submission.py` 与 `evaluate.py` 的边界。
2. 阅读 2025 正式第二轮题面和评分点，再完成正式代码包中的 TODO。
3. 分别记录 MindSpore 例题环境与 PyTorch 2.6.0 历史正式环境，不要混成一个“官方 2026 环境”。
4. 2026 第二轮前以 CCF 新发布的考生手册 / 环境通知覆盖本文件中的历史版本。

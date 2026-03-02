# RAG 评测体系与 Ragas

## 学习

**核心思想**

依靠"体感"调优必定会撞南墙，必须基于评测分推进敏捷开发。

**学习目标**

- Agent CI/CD Baseline 黄金验证测试集构建方式。

**实战与验收**

- 引入 Ragas 或 TruLens 第三方严苛打分器；集成算例对上周 RAG 系统全面打分。要求后续任意微调（Chunk变小/模型更换）都可执行自动跑批打分并直接观测到三大指标（上下文相关度、准确性、有助作答率）的起伏漂移。

---

## 第一部分学习：为什么需要评测体系？

### "感觉还不错"的陷阱

你调整了 Chunk 大小从 500 改成 300，手动试了三个问题，感觉回答质量提升了。于是你信心满满地上线——结果用户投诉暴增。

**原因**：你试的三个问题恰好变好了，但其他 97 个问题变差了。

### 生活类比：体检 vs 感觉

**没有评测**：你"感觉"身体不错，跑步也不喘。（可能血脂已经超标）

**有了评测**：年度体检，血压、血糖、血脂全部量化。每项有标准值，超标就发警报。

**RAG 评测就是给 AI 系统做"体检"，每次改动前后都有量化数据对比。**

### 评测驱动开发的流程

```
修改参数（Chunk大小/模型/Prompt）
        ↓
运行自动评测（跑100题验证集）
        ↓
对比三大指标的漂移
        ↓
指标上升 → 接受改动
指标下降 → 回滚改动
```

---

## 第二部分学习：Ragas 三大核心指标

### 指标一：Context Relevancy（上下文相关度）

**衡量什么**：检索到的文档段落和用户问题是否相关？

**直觉**：你问的是"退货期限"，检索到的是"退货政策"→ 相关；检索到的是"会员权益"→ 不相关。

```
Context Relevancy = 相关句子数 / 检索到的总句子数
```

**低分意味着**：检索召回了太多无关内容，浪费 Token 并可能引入噪音。

### 指标二：Faithfulness（忠实度/准确性）

**衡量什么**：模型的回答是否忠实于检索到的上下文？有没有"编造"？

**直觉**：文档里写的是"30天退货"，模型回答"30天退货"→ 忠实；模型回答"60天退货"→ 不忠实（幻觉）。

```
Faithfulness = 能在上下文中找到依据的陈述数 / 回答中的总陈述数
```

**低分意味着**：模型在"脑补"，生成了文档中没有的信息。

### 指标三：Answer Relevancy（有助作答率）

**衡量什么**：模型的回答是否真正回答了用户的问题？

**直觉**：你问"退货期限多少天？"，回答"退货需保持原包装"→ 虽然和退货相关，但没回答"多少天"。

```
Answer Relevancy = 回答与问题的语义相似度
```

**低分意味着**：模型"答非所问"，虽然用了正确的文档，但没有提取到关键信息。

### 三大指标的协作关系

```
Context Relevancy（检索质量）
        ↓ 影响
Faithfulness（生成忠实度）
        ↓ 影响
Answer Relevancy（最终回答质量）
```

如果 Context Relevancy 低，后面两个指标大概率也低——**垃圾进，垃圾出**。

---

## 第三部分学习：Ragas vs TruLens 对比

| 维度 | Ragas | TruLens |
| --- | --- | --- |
| 定位 | RAG 专用评测框架 | 通用 LLM 应用评测 |
| 核心指标 | Context Relevancy、Faithfulness、Answer Relevancy | 同类指标 + 自定义评测函数 |
| 评测方式 | 用 LLM 做裁判打分 | 用 LLM 做裁判打分 |
| 可视化 | 基础（命令行输出） | 内置 Dashboard（浏览器） |
| 易用性 | API 简洁，上手快 | 功能更全，但配置稍复杂 |
| 适用场景 | 纯 RAG 系统评测 | RAG + Agent + 对话系统 |

**推荐**：先用 Ragas 快速上手，后续 Agent 系统复杂后再引入 TruLens。

---

## 第四部分学习：黄金验证测试集构建

### 什么是黄金测试集？

一组**人工标注**的"问题 + 标准答案 + 期望召回文档"三元组，作为评测的基准。

### 构建原则

1. **覆盖多种查询类型**：简单事实、多跳推理、模糊查询、否定查询
2. **包含边界案例**：知识库中没有答案的问题（期望模型回答"不知道"）
3. **数量适中**：30-100 题。太少不稳定，太多维护成本高
4. **定期更新**：知识库更新后，测试集也要同步更新

### 测试集格式

```python
GOLDEN_TEST_SET = [
    {
        "question": "退货期限是多少天？",
        "ground_truth": "30天",
        "expected_chunks": ["c001"],
        "category": "simple_fact",
    },
    {
        "question": "公司CEO毕业于哪所大学？",
        "ground_truth": "清华大学",
        "expected_chunks": ["c005"],
        "category": "multi_hop",
    },
    {
        "question": "产品支持防水吗？",
        "ground_truth": "知识库中无相关信息",
        "expected_chunks": [],
        "category": "unanswerable",
    },
]
```

---

## 实战：Ragas 自动评测 Pipeline

```python
"""
RAG 自动评测管道：用 Ragas 对 RAG 系统做全面打分。
依赖：pip install ragas datasets openai
"""
from dataclasses import dataclass
import json

# ===== 评测数据结构 =====
@dataclass
class EvalSample:
    question: str
    ground_truth: str
    answer: str           # RAG 系统的实际回答
    contexts: list[str]   # RAG 系统实际检索到的文档段落

# ===== 模拟 RAG 系统输出（替换为你的真实系统）=====
def mock_rag_system(question: str) -> tuple[str, list[str]]:
    """返回 (answer, contexts)"""
    mock_outputs = {
        "退货期限是多少天？": (
            "退货期限为30天。[ref:c001]",
            ["商品支持购买后30天内无理由退货，退货时须保持原包装完好。"]
        ),
        "黄金会员有什么快递权益？": (
            "黄金会员每月享有10次免费快递权益。[ref:c003]",
            ["黄金会员每月享有10次免费快递，白银会员每月3次。"]
        ),
        "产品支持防水吗？": (
            "根据现有资料，未找到关于防水功能的说明。",
            ["产品保修期为一年，人为损坏不在保修范围内。"]
        ),
    }
    return mock_outputs.get(question, ("无法回答", []))

# ===== 评测集 =====
GOLDEN_SET = [
    {"question": "退货期限是多少天？", "ground_truth": "30天"},
    {"question": "黄金会员有什么快递权益？", "ground_truth": "每月10次免费快递"},
    {"question": "产品支持防水吗？", "ground_truth": "知识库中无相关信息"},
]

# ===== 用 Ragas 评测 =====
def run_ragas_evaluation():
    """
    使用 Ragas 框架对 RAG 系统评测。
    """
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_relevancy
    from datasets import Dataset

    # 收集 RAG 输出
    eval_data = {"question": [], "answer": [], "contexts": [], "ground_truth": []}

    for item in GOLDEN_SET:
        answer, contexts = mock_rag_system(item["question"])
        eval_data["question"].append(item["question"])
        eval_data["answer"].append(answer)
        eval_data["contexts"].append(contexts)
        eval_data["ground_truth"].append(item["ground_truth"])

    dataset = Dataset.from_dict(eval_data)

    # 运行评测
    result = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_relevancy],
    )

    print("\n" + "=" * 50)
    print("Ragas 评测结果")
    print("=" * 50)
    for metric, score in result.items():
        print(f"  {metric}: {score:.4f}")

    return result


# ===== 简化版自研评测（不依赖 Ragas 库）=====
def simple_evaluation():
    """
    不依赖第三方库的简化评测，用 LLM 做裁判。
    适合快速验证或 Ragas 安装困难时使用。
    """
    from openai import OpenAI
    client = OpenAI()

    JUDGE_PROMPT = """你是一个严格的 RAG 系统评测裁判。请对以下问答做评分。

问题：{question}
标准答案：{ground_truth}
系统回答：{answer}
系统使用的参考文档：{contexts}

请用 JSON 格式评分（每项 0-100 分）：
{{
    "context_relevancy": <参考文档与问题的相关性>,
    "faithfulness": <系统回答是否忠实于参考文档，没有编造>,
    "answer_relevancy": <系统回答是否真正回答了问题>
}}"""

    all_scores = {"context_relevancy": [], "faithfulness": [], "answer_relevancy": []}

    for item in GOLDEN_SET:
        answer, contexts = mock_rag_system(item["question"])

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(
                question=item["question"],
                ground_truth=item["ground_truth"],
                answer=answer,
                contexts="\n".join(contexts),
            )}],
            temperature=0,
        )

        try:
            scores = json.loads(resp.choices[0].message.content)
            for key in all_scores:
                all_scores[key].append(scores.get(key, 50))
        except (json.JSONDecodeError, KeyError):
            for key in all_scores:
                all_scores[key].append(50)

    print("\n" + "=" * 50)
    print("自研简化评测结果")
    print("=" * 50)
    for metric, scores in all_scores.items():
        avg = sum(scores) / len(scores)
        print(f"  {metric}: {avg:.1f} / 100")

    return all_scores


# ===== A/B 对比：参数调整前后 =====
def ab_eval_report(baseline_scores: dict, experiment_scores: dict, change_desc: str):
    """输出参数调整前后的指标对比报告"""
    print(f"\n{'='*60}")
    print(f"A/B 评测对比报告")
    print(f"变更说明：{change_desc}")
    print(f"{'='*60}")
    print(f"{'指标':<25} {'Baseline':>10} {'实验组':>10} {'变化':>10}")
    print("-" * 60)
    for metric in baseline_scores:
        base = sum(baseline_scores[metric]) / len(baseline_scores[metric])
        exp = sum(experiment_scores[metric]) / len(experiment_scores[metric])
        delta = exp - base
        arrow = "↑" if delta > 0 else "↓" if delta < 0 else "→"
        print(f"  {metric:<23} {base:>9.1f} {exp:>9.1f} {delta:>+8.1f} {arrow}")


if __name__ == "__main__":
    print("方案1：使用 Ragas 官方库评测")
    print("（需要安装 ragas 库，取消下方注释即可运行）")
    # run_ragas_evaluation()

    print("\n方案2：使用自研简化评测（无额外依赖）")
    scores = simple_evaluation()
```

---

## 验收交付：CI/CD 自动评测报告模板

> 根据验收标准，后续任意微调都可执行自动跑批打分，观测三大指标的起伏漂移。

```
============================================================
RAG 评测报告 — 2024-06-15 14:30
============================================================
变更说明：Chunk 大小从 500 改为 300
测试集：golden_test_v2.json（30题）
模型：GPT-4o-mini

指标                       Baseline    实验组      变化
------------------------------------------------------------
  context_relevancy           82.3       86.7     +4.4 ↑
  faithfulness                91.0       89.5     -1.5 ↓
  answer_relevancy            85.6       87.2     +1.6 ↑

结论：
  ✅ context_relevancy 提升，说明更小的 Chunk 减少了无关噪音
  ⚠️ faithfulness 微降，可能是 Chunk 切断了关键上下文
  ✅ answer_relevancy 提升，整体正向

决策：接受变更，但需监控 faithfulness 是否持续下降。
```

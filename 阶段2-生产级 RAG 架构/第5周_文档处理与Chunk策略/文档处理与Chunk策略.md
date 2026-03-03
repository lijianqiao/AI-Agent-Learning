# 文档处理与 Chunk 策略

## 学习

**学习目标**

- Fixed chunking vs Semantic chunking 算法选型、高频元数据注入 (Metadata tagging)

**实战**

- 构建多文档可调整摄入器 (Ingestion Pipeline)
- 框架拓宽：对比 LangChain 与专注检索增强的 LlamaIndex 在复杂结构文档应对上的异同。

**验收标准**

- 实现一个能应对长篇 PDF 与多级表头 Excel 解析方案且支持 A/B 切换的效果对比实验台。

---

## 第一部分学习：为什么需要 Chunking？

### 大模型的"胃容量"问题

大模型有上下文窗口限制（128K tokens 已经很大了），但企业知识库动辄几百份 PDF、几千页文档。全部塞进去既放不下、也浪费钱、还会引发注意力分散（第一周已学过）。

**Chunking 就是把大文档切成小块，只把最相关的几块喂给模型。**

### 生活类比：图书馆找答案

**没有 Chunking**：把整个图书馆搬到你桌上，让你从中找一句话。

**有了 Chunking**：图书管理员帮你翻好书、折好页、标好段落，你只需要看被标记的几段就好。

### Chunking 对 RAG 的影响

| 切得太大 | 切得太小 |
| --- | --- |
| 召回段落含大量无关内容 | 丢失上下文，语义不完整 |
| 消耗过多 Token 预算 | 需要召回更多 Chunk 才能凑够信息 |
| 向量化时语义被稀释 | 向量化时语义过于碎片 |

**核心目标：每个 Chunk 恰好包含一个完整的语义单元。**

---

## 第二部分学习：Fixed Chunking（固定切分）

### 原理

最简单的策略：按固定字符数/Token 数切分，相邻 Chunk 之间有一定重叠（overlap）防止在边界处切断句子。

```
文档总长：10,000 tokens
Chunk 大小：500 tokens
Overlap：50 tokens

Chunk 1: token[0:500]
Chunk 2: token[450:950]     ← 前50个与 Chunk 1 重叠
Chunk 3: token[900:1400]
...
```

### 优缺点

| 优点 | 缺点 |
| --- | --- |
| 实现简单、速度快 | 不关心语义边界，可能把一句话切成两半 |
| 确定性强，结果可复现 | 表格、代码块可能被拦腰截断 |
| 适合结构均匀的纯文本 | 不适合结构复杂的文档（PDF、合同） |

### 代码实现

```python
def fixed_chunking(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    """固定大小切分，带重叠窗口"""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks
```

---

## 第三部分学习：Semantic Chunking（语义切分）

### 原理

不按固定长度，而是按**语义边界**切分。核心思路：相邻句子的 Embedding 相似度高就合并，一旦相似度骤降就在此处切断。

```
句子1: "公司成立于2020年" ─┐
句子2: "总部位于北京"     ─┤ 相似度高 → 合为一个 Chunk
句子3: "员工约500人"     ─┘
                          ← 相似度骤降（话题切换）
句子4: "产品支持退货"     ─┐
句子5: "退货需30天内申请" ─┤ 合为另一个 Chunk
```

### 与 Fixed Chunking 的对比

| 维度 | Fixed Chunking | Semantic Chunking |
| --- | --- | --- |
| 切分依据 | 固定字符/Token 数 | 语义相似度阈值 |
| 语义完整性 | 低（可能切断句子） | 高（按话题边界切） |
| 计算成本 | 几乎为零 | 需要调用 Embedding 模型 |
| Chunk 大小 | 固定 | 不固定，受阈值影响 |
| 适用场景 | 结构均匀的纯文本 | 话题丰富的复杂文档 |

### 代码实现

```python
import numpy as np
from openai import OpenAI

client = OpenAI()

def get_embedding(text: str) -> np.ndarray:
    resp = client.embeddings.create(model="text-embedding-3-small", input=text)
    return np.array(resp.data[0].embedding)

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

def semantic_chunking(sentences: list[str], threshold: float = 0.75) -> list[str]:
    """
    基于相邻句子 Embedding 相似度的语义切分。
    相似度低于 threshold 时切断。
    """
    if not sentences:
        return []

    embeddings = [get_embedding(s) for s in sentences]
    chunks = []
    current_chunk = [sentences[0]]

    for i in range(1, len(sentences)):
        sim = cosine_sim(embeddings[i - 1], embeddings[i])
        if sim >= threshold:
            current_chunk.append(sentences[i])
        else:
            chunks.append("\n".join(current_chunk))
            current_chunk = [sentences[i]]

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks
```

---

## 第四部分学习：Metadata Tagging（元数据注入）

### 为什么 Chunk 不能只有正文？

纯文本 Chunk 缺乏结构信息。假设两个 Chunk 都提到"退货期限"，一个来自"中国区政策"，一个来自"美国区政策"。向量相似度几乎一样，但答案完全不同。

**元数据就是给每个 Chunk 贴标签，帮检索系统做精确过滤。**

### 常见元数据字段

| 字段 | 用途 | 示例 |
| --- | --- | --- |
| `source_file` | 文档来源 | "退换货政策_v3.pdf" |
| `page_number` | 所在页码 | 12 |
| `section_title` | 章节标题 | "第三章 退换货规定" |
| `doc_type` | 文档类型 | "policy" / "technical" |
| `created_date` | 创建日期 | "2024-06-15" |
| `chunk_index` | 当前 Chunk 序号 | 5 |

### 元数据增强检索

```python
# 检索时可以加 filter 条件
results = vector_db.search(
    query_embedding=query_emb,
    top_k=5,
    filter={"doc_type": "policy", "created_date": {"$gte": "2024-01-01"}},
)
# 只在"政策类"且"2024年之后"的文档中搜索
```

---

## 第五部分学习：LangChain vs LlamaIndex 框架对比

### 定位差异

| 维度 | LangChain | LlamaIndex |
| --- | --- | --- |
| 核心定位 | 通用 LLM 应用编排框架 | 专注数据索引与检索增强 |
| 擅长场景 | Agent 编排、Chain 组合、多工具协作 | 复杂文档摄入、索引构建、检索优化 |
| 文档加载器 | 有，种类多但较薄 | 有，深度适配多种格式 |
| Chunking | 基础支持 | 丰富策略（句子窗口、层级节点等） |
| 向量存储 | 集成多种，但索引管理较浅 | 深度集成，支持多种索引类型 |
| 学习曲线 | 中等（API 经常变动） | 中等（概念模型清晰） |

### 选型建议

- **如果你的核心是 RAG 系统**：优先 LlamaIndex，它的 `IngestionPipeline`、`SentenceWindowNodeParser` 等组件开箱即用
- **如果你的核心是 Agent + 工具编排**：优先 LangChain / LangGraph，RAG 只是其中一个工具
- **生产环境推荐**：两者组合使用——LlamaIndex 负责索引和检索，LangChain 负责上层编排

---

## 实战：多文档 Ingestion Pipeline

```python
"""
多文档摄入管道：支持 PDF 和 Excel，A/B 切换 Chunking 策略。
依赖：pip install langchain langchain-community pypdf openpyxl tiktoken
"""
import os
import tiktoken
from dataclasses import dataclass, field
from typing import Literal

# ===== 数据结构 =====
@dataclass
class Chunk:
    content: str
    metadata: dict = field(default_factory=dict)
    token_count: int = 0

@dataclass
class IngestionResult:
    source_file: str
    strategy: str
    total_chunks: int
    avg_chunk_tokens: float
    min_chunk_tokens: int
    max_chunk_tokens: int
    chunks: list[Chunk]

# ===== 文档加载器 =====
def load_pdf(path: str) -> list[dict]:
    """加载 PDF，返回按页切分的文本列表"""
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(path)
    pages = loader.load()
    return [{"text": p.page_content, "page": p.metadata.get("page", i)}
            for i, p in enumerate(pages)]

def load_excel(path: str) -> list[dict]:
    """加载 Excel，每行作为一条记录"""
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True)
    records = []
    for sheet in wb.sheetnames:
        ws = wb[sheet]
        headers = [cell.value for cell in next(ws.iter_rows(min_row=1, max_row=1))]
        for row_idx, row in enumerate(ws.iter_rows(min_row=2, values_only=True), start=2):
            row_text = " | ".join(
                f"{h}: {v}" for h, v in zip(headers, row) if v is not None
            )
            records.append({"text": row_text, "page": f"{sheet}_row{row_idx}"})
    return records

# ===== Chunking 策略 =====
def chunk_fixed(texts: list[dict], chunk_size: int = 500, overlap: int = 50) -> list[Chunk]:
    """Fixed Chunking：按字符数切分"""
    enc = tiktoken.encoding_for_model("gpt-4o")
    chunks = []
    for item in texts:
        text = item["text"]
        start = 0
        idx = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            content = text[start:end]
            chunks.append(Chunk(
                content=content,
                metadata={"page": item["page"], "chunk_index": idx, "strategy": "fixed"},
                token_count=len(enc.encode(content)),
            ))
            start += chunk_size - overlap
            idx += 1
    return chunks

def chunk_semantic_simple(texts: list[dict], max_chunk_tokens: int = 300) -> list[Chunk]:
    """简化版语义切分：按句号/换行切句，再按 Token 预算合并"""
    import re
    enc = tiktoken.encoding_for_model("gpt-4o")
    chunks = []

    for item in texts:
        sentences = re.split(r'(?<=[。！？\n])', item["text"])
        sentences = [s.strip() for s in sentences if s.strip()]

        current_content = ""
        current_tokens = 0
        idx = 0

        for sent in sentences:
            sent_tokens = len(enc.encode(sent))
            if current_tokens + sent_tokens > max_chunk_tokens and current_content:
                chunks.append(Chunk(
                    content=current_content,
                    metadata={"page": item["page"], "chunk_index": idx, "strategy": "semantic"},
                    token_count=current_tokens,
                ))
                current_content = sent
                current_tokens = sent_tokens
                idx += 1
            else:
                current_content += sent
                current_tokens += sent_tokens

        if current_content:
            chunks.append(Chunk(
                content=current_content,
                metadata={"page": item["page"], "chunk_index": idx, "strategy": "semantic"},
                token_count=current_tokens,
            ))

    return chunks

# ===== 主管道 =====
def ingest(
    file_path: str,
    strategy: Literal["fixed", "semantic"] = "fixed",
) -> IngestionResult:
    """
    统一摄入管道：自动识别文件格式，按指定策略切分。
    """
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        texts = load_pdf(file_path)
    elif ext in (".xlsx", ".xls"):
        texts = load_excel(file_path)
    else:
        with open(file_path, "r", encoding="utf-8") as f:
            texts = [{"text": f.read(), "page": 0}]

    if strategy == "fixed":
        chunks = chunk_fixed(texts)
    else:
        chunks = chunk_semantic_simple(texts)

    token_counts = [c.token_count for c in chunks]

    return IngestionResult(
        source_file=file_path,
        strategy=strategy,
        total_chunks=len(chunks),
        avg_chunk_tokens=round(sum(token_counts) / max(len(token_counts), 1), 1),
        min_chunk_tokens=min(token_counts) if token_counts else 0,
        max_chunk_tokens=max(token_counts) if token_counts else 0,
        chunks=chunks,
    )


# ===== A/B 对比实验 =====
def ab_compare(file_path: str):
    """对同一文件分别用两种策略切分，输出对比报告"""
    result_a = ingest(file_path, strategy="fixed")
    result_b = ingest(file_path, strategy="semantic")

    print(f"\n{'='*60}")
    print(f"A/B 对比：{file_path}")
    print(f"{'='*60}")
    for label, r in [("A (Fixed)", result_a), ("B (Semantic)", result_b)]:
        print(f"\n策略 {label}:")
        print(f"  Chunk 总数：{r.total_chunks}")
        print(f"  平均 Token/Chunk：{r.avg_chunk_tokens}")
        print(f"  最小/最大 Token：{r.min_chunk_tokens} / {r.max_chunk_tokens}")
        print(f"  首个 Chunk 预览：{r.chunks[0].content[:80]}...")


if __name__ == "__main__":
    # 用一个示例文本文件测试
    test_file = "test_doc.txt"
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("这是一份关于AI技术的文档。" * 200)
    ab_compare(test_file)
```

---

## 验收交付：A/B 对比实验台输出模板

> 根据验收标准，需实现一个支持长篇 PDF 与多级表头 Excel 的 A/B 切换效果对比实验台。

```
============================================================
A/B 对比：company_policy.pdf
============================================================

策略 A (Fixed):
  Chunk 总数：45
  平均 Token/Chunk：312.5
  最小/最大 Token：298 / 500
  首个 Chunk 预览：第一章 公司简介 本公司成立于2020年...

策略 B (Semantic):
  Chunk 总数：38
  平均 Token/Chunk：287.3
  最小/最大 Token：52 / 486
  首个 Chunk 预览：第一章 公司简介。本公司成立于2020年，总部位于北京...

分析结论：
  - Semantic 策略 Chunk 数量少 15%（38 vs 45），说明合并效果好
  - Semantic 策略的最小 Chunk 只有 52 tokens，存在极短段落
  - 建议：对 Semantic 策略增加 min_chunk_tokens 下限过滤
```

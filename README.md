# AI Agent 架构师 — 26 周系统学习路线

> 从 LLM 底层原理到企业级多智能体生产系统，一条完整的 AI Agent 架构师成长路径。

## 项目简介

本仓库是一份 **26 周 AI Agent 架构师学习计划**的全部学习笔记与实战代码。

每周对应一篇 Markdown 笔记，包含：

- **知识拆解**：核心概念用通俗类比讲透，配有对比表格
- **可运行代码**：Python 实战示例，可直接复制执行
- **验收交付**：量化验收标准与输出模板，确保学到的能落地

## 学习路线总览

```
阶段 1  ▸ LLM 原理与调度基石           W1 – W4
阶段 2  ▸ 生产级 RAG 架构              W5 – W8
缓冲期1 ▸ 架构复盘与评测体系搭建        W9
阶段 3  ▸ Agent 状态机与工作流编排       W10 – W13
阶段 4  ▸ Memory 系统重装抽象基建       W14 – W17
缓冲期2 ▸ 微服务解耦化重构              W18
阶段 5  ▸ 多智能体协作系统 (MAS)        W19 – W22
阶段 6  ▸ 生产化、安全风控与容灾降级     W23 – W26
```

## 核心技术栈

| 领域 | 技术 |
| --- | --- |
| LLM 调用 | OpenAI API, LiteLLM, 语义路由 |
| RAG | LangChain, LlamaIndex, Milvus/Weaviate, BM25, Cross-Encoder |
| Agent 编排 | LangGraph, DSPy, MCP 协议 |
| 多智能体 | CrewAI, MetaGPT, A2A 协议 |
| Memory | 滑动窗口, 向量归档, 递归摘要, 三层聚合器 |
| 安全 | NeMo Guardrails, E2B Sandbox, Prompt 防注入 |
| 生产化 | Kafka/RabbitMQ, Redis 语义缓存, 自动降级容灾 |
| 评测 | Ragas, TruLens |

## 项目结构

```
AI Agent架构师/
│
├── AI Agent架构师.md                          # 完整 26 周学习路线图
├── README.md
│
├── 阶段1-LLM原理与调度基石/
│   ├── 第1周_Transformer & Token 机制.md
│   ├── 第2周_推理模式与Prompt结构深度演进.md
│   ├── 第3周_模型网关建设与动态路由.md
│   └── 第4周_幻觉发生剖析与防御抑制策略.md
│
├── 阶段2-生产级 RAG 架构/
│   ├── 第5周_文档处理与Chunk策略.md
│   ├── 第6周_向量数据库与GraphRAG.md
│   ├── 第7周_混合检索与重排序.md
│   └── 第8周_生产级RAG与AgenticRAG.md
│
├── 缓冲期1-架构复盘与指标评测体系搭建/
│   └── 第9周_RAG评测体系Ragas.md
│
├── 阶段3-Agent 状态机演化与工作流编排/
│   ├── 第10周_LangGraph状态机与MCP协议.md
│   ├── 第11周_工作流分治与DSPy入门.md
│   ├── 第12周_DSPy编译优化与异常自愈.md
│   └── 第13周_链路监控与可观测性.md
│
├── 阶段4-Memory 系统重装抽象基建/
│   ├── 第14周_短期记忆与Token预算.md
│   ├── 第15周_长期记忆与用户画像.md
│   ├── 第16周_递归摘要压缩.md
│   └── 第17周_三层记忆聚合架构.md
│
├── 缓冲期2-系统服务彻底微服务解耦化重构/
│   └── 第18周_微服务解耦重构.md
│
├── 阶段5-多智能体协作系统网络 (Multi-Agent System)/
│   ├── 第19周_多Agent角色编排.md
│   ├── 第20周_异步通信与A2A协议.md
│   ├── 第21周_MAS闭环联动演练.md
│   └── 第22周_代码沙盒安全隔离.md
│
└── 阶段6-终极工程合规化升级与极客出师战/
    ├── 第23周_安全护栏与Prompt防注入.md
    ├── 第24周_高并发队列与消峰.md
    ├── 第25周_缓存计算与容灾降级.md
    └── 第26周_企业级系统封卷发布.md
```

> 各阶段文件夹后续会逐步补充实战代码项目。

## 各周速查索引

### 阶段 1 — LLM 原理与调度基石

| 周次 | 主题 | 关键词 |
| --- | --- | --- |
| W1 | Transformer & Token 机制 | Attention, KV-Cache, Context Window, Token 预算 |
| W2 | 推理模式与 Prompt 结构 | ReAct, CoT, Tool Calling, Vanilla Agent |
| W3 | 模型网关与动态路由 | LiteLLM, 语义路由, vLLM/TGI/Ollama, 熔断限流 |
| W4 | 幻觉剖析与防御 | Grounding, Temperature/Top-p, Confidence Score |

### 阶段 2 — 生产级 RAG 架构

| 周次 | 主题 | 关键词 |
| --- | --- | --- |
| W5 | 文档处理与 Chunk 策略 | Fixed/Semantic Chunking, Metadata, Ingestion Pipeline |
| W6 | 向量数据库与 GraphRAG | HNSW, Milvus/Weaviate, 知识图谱, Top-K 调参 |
| W7 | 混合检索与重排序 | BM25, Dense, RRF 融合, Cross-Encoder Reranker |
| W8 | 生产级 RAG 与 Agentic RAG | 自适应检索, 查询改写, 多跳问答, Reference Id |

### 缓冲期 1

| 周次 | 主题 | 关键词 |
| --- | --- | --- |
| W9 | RAG 评测体系 | Ragas, TruLens, CI/CD 自动评测, 黄金测试集 |

### 阶段 3 — Agent 状态机与工作流编排

| 周次 | 主题 | 关键词 |
| --- | --- | --- |
| W10 | LangGraph 状态机与 MCP 协议 | StateGraph, Conditional Edge, MCP Server/Client |
| W11 | 工作流分治与 DSPy 入门 | Planner/Executor, dspy.Signature, ChainOfThought |
| W12 | DSPy 编译优化与异常自愈 | Teleprompter, 指数退避, Human-in-the-loop |
| W13 | 链路监控与可观测性 | LangSmith, Arize Phoenix, Trace 树, RequestId |

### 阶段 4 — Memory 系统

| 周次 | 主题 | 关键词 |
| --- | --- | --- |
| W14 | 短期记忆与 Token 预算 | 滑动窗口, 提权算法, 80% 阈值释放 |
| W15 | 长期记忆与用户画像 | 向量归档, 定时 Job, 跨会话召回 |
| W16 | 递归摘要压缩 | Recursive Summarization, 实体抽取, 20:1 压缩 |
| W17 | 三层记忆聚合架构 | 短期/中期/长期, 统一聚合器, 50 轮压测 |

### 缓冲期 2

| 周次 | 主题 | 关键词 |
| --- | --- | --- |
| W18 | 微服务解耦重构 | IoC, 接口隔离, 插件化, 圈复杂度 |

### 阶段 5 — 多智能体协作系统

| 周次 | 主题 | 关键词 |
| --- | --- | --- |
| W19 | 多 Agent 角色编排 | CrewAI, MetaGPT, 角色预制舱, 产出物溯源 |
| W20 | 异步通信与 A2A 协议 | Pub/Sub, 黑板模式, A2A vs MCP, 冲突裁定 |
| W21 | MAS 闭环联动演练 | 五角色流水线, DAG 编排, 全自动博客生产 |
| W22 | 代码沙盒安全隔离 | E2B Sandbox, 受限 Docker, 恶意命令拦截 |

### 阶段 6 — 生产化与安全

| 周次 | 主题 | 关键词 |
| --- | --- | --- |
| W23 | 安全护栏与 Prompt 防注入 | NeMo Guardrails, 三层纵深防御, 95% 拦截率 |
| W24 | 高并发队列与消峰 | Kafka/RabbitMQ, 异步推理队列, QPS 5x 压测 |
| W25 | 缓存计算与容灾降级 | Prompt Cache, Redis 语义缓存, 30s 故障切换 |
| W26 | 企业级系统封卷发布 | 全链路集成测试, 架构白皮书, Ragas 评测报告 |

## 前置要求

- **Python**：熟悉异步编程（asyncio）、装饰器、类型注解
- **Docker**：能运行容器、编写简单 Dockerfile
- **REST API**：了解 HTTP 请求/响应、JSON 格式
- **基础数学**：线性代数向量运算、概率论基本概念（softmax、余弦相似度等）

## 使用方式

1. **按周阅读**：从第 1 周开始，每周消化一篇笔记
2. **动手实践**：每篇笔记的「实战」代码都可以直接运行（需配置 API Key）
3. **验收自检**：用每周的「验收交付」标准检验学习成果
4. **渐进构建**：后期周次的代码会复用前期组件，形成完整系统

## License

MIT

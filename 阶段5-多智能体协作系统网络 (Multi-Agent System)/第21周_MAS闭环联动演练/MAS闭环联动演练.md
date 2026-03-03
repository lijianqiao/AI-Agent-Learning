# MAS 闭环联动演练

## 学习

**学习目标**

- 高跨度口语指令到多步骤 DAG 工作流的自动意图拆解
- 五角色 MAS（Multi-Agent System）全流水线无人工干预闭环：调研员 → 作者 → 事实核查员 → 编辑 → 发布员

> 本周是全课程的**综合实弹演习**——给一套微拟真 Agent 群体输入一句日常口语："在网上搜最近大模型资讯并写五百字博客且审查发布"，观察系统如何自动完成从调研、执笔、核查、排版到发布的全流水线推进。

**综合演练**

- 输入高跨度口语指令，观察五角色 Agent 协作完成全自动博客生产流水线
- 每个 Agent 的输入/输出均需完整日志追溯，流水线支持条件分支和异常回退

**验收标准**

- 全流程无人工干预完成率 > 80%；每个 Agent 的输入/输出均有完整日志追溯

---

## 第一部分学习：高跨度口语指令的意图拆解

### 什么是"高跨度口语指令"？

用户随口说了一句：

> "在网上搜最近大模型资讯并写五百字博客且审查发布"

短短 22 个字，背后至少跨越了 **5 个完全不同的能力域**：信息检索、内容创作、事实验证、格式编排、平台发布。这就叫"高跨度"——一句话横跨多个技能边界。

### 生活类比：老板的一句话 vs 流水线工序

**高跨度口语指令** = 老板在走廊遇到你，随口说了句"帮我查查最近 AI 有什么新闻，写篇博客发出去"。

**意图拆解** = 你回到工位后，脑子里自动翻译成：

1. 打开浏览器搜 AI 新闻（调研）
2. 筛选 3-5 条有价值的资讯（信息过滤）
3. 写一篇 500 字的博客（创作）
4. 检查引用的事实是否准确（核查）
5. 调整格式、加标题配图（编辑）
6. 登录博客平台发布（发布）

人类靠**常识 + 工作经验**完成这个拆解，Agent 则需要一套**显式的意图解析引擎**。

### 拆解流程：从口语到结构化任务链

```
用户口语指令（非结构化）
      │
      ▼
┌──────────────────────────┐
│  意图解析器 (Intent Parser)│
│                          │
│  1. 实体提取：大模型资讯   │
│  2. 动作提取：搜/写/审/发  │
│  3. 约束提取：五百字、最近  │
│  4. 依赖排序：搜→写→审→发  │
└────────────┬─────────────┘
             ▼
┌──────────────────────────┐
│  结构化任务链 (Task Chain) │
│                          │
│  Task1: 搜索最近大模型资讯 │
│    → 约束: 时间=最近7天    │
│  Task2: 撰写500字博客     │
│    → 依赖: Task1 的搜索结果│
│  Task3: 事实核查           │
│    → 依赖: Task2 的博客正文│
│  Task4: 格式编排           │
│    → 依赖: Task3 核查通过稿│
│  Task5: 发布              │
│    → 依赖: Task4 终稿      │
└──────────────────────────┘
```

### 意图拆解的关键技术

| 技术环节 | 作用 | 难点 | 类比 |
| --- | --- | --- | --- |
| 实体提取 | 识别"大模型""资讯"等名词性关键词 | 口语中的省略和指代消解 | 听老板讲话时抓住关键词 |
| 动作提取 | 识别"搜""写""审查""发布"等动词 | 一个动词可能对应多个子步骤 | 把"搞定"翻译成具体操作 |
| 约束提取 | 识别"五百字""最近"等限定条件 | 隐含约束（如"审查"暗示需要核查事实） | 老板说"尽快"到底是多快 |
| 依赖排序 | 确定任务之间的先后依赖关系 | "且"字连接的并行 vs 串行判断 | 先买菜再做饭，不能反过来 |
| 歧义兜底 | 无法确定时主动澄清或选择安全默认值 | 何时该问人、何时该自行决策 | 拿不准就先问老板 |

### 意图拆解实战代码

```python
import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ParsedTask:
    """拆解后的单个结构化任务"""
    task_id: str
    action: str          # 动作：search / write / fact_check / edit / publish
    description: str     # 自然语言描述
    constraints: dict    # 约束条件
    depends_on: list     # 依赖的前置任务 ID
    assigned_role: str   # 分配的角色


def parse_high_span_instruction(instruction: str) -> list[ParsedTask]:
    """
    将高跨度口语指令拆解为结构化任务链。
    生产环境中此步骤由 LLM 完成，这里用规则引擎演示核心逻辑。
    """
    tasks = []

    # Step 1: 关键词 → 动作映射
    action_keywords = {
        "搜": "search",
        "查": "search",
        "写": "write",
        "撰写": "write",
        "审查": "fact_check",
        "核查": "fact_check",
        "发布": "publish",
        "发": "publish",
    }

    # Step 2: 约束提取
    constraints = {}
    if "五百字" in instruction or "500字" in instruction:
        constraints["word_count"] = 500
    if "最近" in instruction:
        constraints["time_range"] = "7d"
    if "大模型" in instruction:
        constraints["topic"] = "大模型/LLM"

    # Step 3: 按动作生成任务链（含隐含任务"编辑"）
    task_definitions = [
        ParsedTask(
            task_id="T1",
            action="search",
            description="搜索最近7天大模型相关资讯",
            constraints={"time_range": constraints.get("time_range", "7d"),
                         "topic": constraints.get("topic", "AI")},
            depends_on=[],
            assigned_role="researcher",
        ),
        ParsedTask(
            task_id="T2",
            action="write",
            description="基于调研结果撰写500字博客",
            constraints={"word_count": constraints.get("word_count", 500)},
            depends_on=["T1"],
            assigned_role="writer",
        ),
        ParsedTask(
            task_id="T3",
            action="fact_check",
            description="核查博客中引用的事实和数据",
            constraints={},
            depends_on=["T2"],
            assigned_role="fact_checker",
        ),
        ParsedTask(
            task_id="T4",
            action="edit",
            description="编排格式、润色文字、生成标题和摘要",
            constraints={},
            depends_on=["T3"],
            assigned_role="editor",
        ),
        ParsedTask(
            task_id="T5",
            action="publish",
            description="将终稿发布到博客平台",
            constraints={},
            depends_on=["T4"],
            assigned_role="publisher",
        ),
    ]

    return task_definitions


# --- 演示 ---
instruction = "在网上搜最近大模型资讯并写五百字博客且审查发布"
tasks = parse_high_span_instruction(instruction)

print(f"原始指令：{instruction}\n")
print(f"拆解为 {len(tasks)} 个结构化任务：\n")
for t in tasks:
    deps = " → ".join(t.depends_on) if t.depends_on else "无"
    print(f"  [{t.task_id}] {t.description}")
    print(f"       角色: {t.assigned_role} | 依赖: {deps} | 约束: {t.constraints}")
    print()
```

**输出示例**：

```
原始指令：在网上搜最近大模型资讯并写五百字博客且审查发布

拆解为 5 个结构化任务：

  [T1] 搜索最近7天大模型相关资讯
       角色: researcher | 依赖: 无 | 约束: {'time_range': '7d', 'topic': '大模型/LLM'}

  [T2] 基于调研结果撰写500字博客
       角色: writer | 依赖: T1 | 约束: {'word_count': 500}

  [T3] 核查博客中引用的事实和数据
       角色: fact_checker | 依赖: T2 | 约束: {}

  [T4] 编排格式、润色文字、生成标题和摘要
       角色: editor | 依赖: T3 | 约束: {}

  [T5] 将终稿发布到博客平台
       角色: publisher | 依赖: T4 | 约束: {}
```

---

## 第二部分学习：五角色分工设计

### 为什么要拆成五个角色？

一个"全能 Agent"理论上可以一个人干完所有活。但生产实践中这会导致三大灾难：

1. **Prompt 过载**：一个 Prompt 同时描述调研、写作、核查、编辑、发布的规则，上下文窗口爆炸，质量骤降
2. **职责耦合**：写作出了问题，你搞不清是调研数据差、还是写作能力弱、还是核查漏了
3. **无法并行**：所有步骤串在一根线程里，不能利用多模型并发

### 生活类比：报社编辑部

一篇新闻稿件的生产线在报社里从来不是一个人完成的：

| 报社角色 | Agent 对应 | 职责 | 产出物 |
| --- | --- | --- | --- |
| 记者 | 调研员（Researcher） | 跑现场、搜资料、采访 | 原始素材包 |
| 撰稿人 | 作者（Writer） | 将素材组织成文章 | 初稿 |
| 校对员 | 事实核查员（Fact Checker） | 核实每个事实和引用 | 核查报告 |
| 责编 | 编辑（Editor） | 润色文字、调整结构、加标题 | 终稿 |
| 发行部 | 发布员（Publisher） | 排版上线、推送通知 | 已发布链接 |

### 五角色能力矩阵

| 角色 | 核心能力 | 输入 | 输出 | 可用工具 | 失败处理 |
| --- | --- | --- | --- | --- | --- |
| Researcher | 信息检索+摘要 | 搜索关键词+时间范围 | 3-5 条资讯摘要 | Web Search API | 重试 / 换搜索引擎 |
| Writer | 内容创作 | 资讯摘要+字数约束 | 500 字初稿 | LLM | 重写 / 降低创意度 |
| Fact Checker | 逻辑/事实校验 | 初稿全文 | 核查报告（通过/驳回+理由） | LLM + Search | 标注存疑部分，回退给 Writer |
| Editor | 格式编排+润色 | 核查通过稿 | 带标题摘要的终稿 | LLM | 重新编排 |
| Publisher | 平台发布 | 终稿 | 发布确认+链接 | Blog API | 重试 / 存为草稿 |

### 角色间的信息流契约

每个角色之间传递的不是"随意的文本"，而是**结构化的契约对象**。这确保了上游输出与下游输入的严格匹配。

```python
from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class CheckVerdict(Enum):
    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"


@dataclass
class ResearchOutput:
    """调研员 → 作者 的交接契约"""
    articles: list[dict]    # [{"title": ..., "summary": ..., "source": ..., "date": ...}]
    search_query: str
    total_found: int
    timestamp: str


@dataclass
class DraftOutput:
    """作者 → 事实核查员 的交接契约"""
    title: str
    body: str               # 博客正文（Markdown）
    word_count: int
    sources_cited: list[str] # 引用的来源 URL
    timestamp: str


@dataclass
class FactCheckOutput:
    """事实核查员 → 编辑 的交接契约"""
    verdict: CheckVerdict    # pass / fail / partial
    issues: list[dict]       # [{"claim": ..., "status": "verified"/"unverified", "note": ...}]
    revised_body: str        # 修正后的正文（如有修改）
    timestamp: str


@dataclass
class EditOutput:
    """编辑 → 发布员 的交接契约"""
    final_title: str
    final_body: str          # 终稿正文
    summary: str             # 摘要（100字以内）
    tags: list[str]          # 标签
    timestamp: str


@dataclass
class PublishOutput:
    """发布员的最终产出"""
    success: bool
    url: str                 # 发布后的链接
    platform: str
    published_at: str
    timestamp: str
```

### 角色解耦的三大收益

| 收益 | 单体 Agent | 五角色 MAS |
| --- | --- | --- |
| 定位故障 | "博客质量差"——到底哪步出了问题？ | 精确到"Fact Checker 漏检了第 3 条" |
| 独立优化 | 改一处牵全身，回归测试成本巨大 | 只调 Writer 的 Prompt，其余不动 |
| 灵活替换 | 全部重写 | 把 Writer 从 GPT-4 换成 Claude，其余不变 |
| 成本控制 | 所有步骤用同一个大模型 | Researcher 用廉价模型，Writer 用高级模型 |

---

## 第三部分学习：流水线编排设计（DAG + 条件分支）

### 为什么是 DAG 而不是简单的线性链？

虽然博客生产看起来是"搜→写→查→编→发"的直线流程，但实际运行中存在**条件回退**：

- 事实核查不通过 → 回退给 Writer 修改
- Editor 发现逻辑硬伤 → 回退给 Fact Checker 重查
- Publisher 发布失败 → 重试或存为草稿

这就不再是简单的链式结构，而是一个**有向无环图（DAG）+ 条件分支**。

### 生活类比：汽车生产线上的质检回退

汽车工厂的组装线不是一条直线——每个工位都有质检员。如果发现喷漆瑕疵，车子不会继续往前到装配工位，而是**回退到喷漆工位返工**。只有质检通过才能流向下一站。Agent 流水线同理。

### 流水线 DAG 结构

```
                    ┌───────────┐
                    │  START    │
                    └─────┬─────┘
                          ▼
                    ┌───────────┐
                    │ Researcher│
                    └─────┬─────┘
                          ▼
                    ┌───────────┐
              ┌────►│  Writer   │◄─────────────┐
              │     └─────┬─────┘              │
              │           ▼                    │
              │     ┌───────────┐              │
              │     │Fact Checker│              │
              │     └─────┬─────┘              │
              │           ▼                    │
              │     ┌─────────────┐            │
              │     │  verdict?   │            │
              │     └──┬──────┬───┘            │
              │  PASS  │      │ FAIL           │
              │        ▼      └────────────────┘
              │  ┌───────────┐    （回退重写，最多3次）
              │  │  Editor   │
              │  └─────┬─────┘
              │        ▼
              │  ┌───────────┐
              │  │ Publisher  │
              │  └─────┬─────┘
              │        ▼
              │  ┌─────────────┐
              │  │ pub_result? │
              │  └──┬──────┬───┘
              │  OK │      │ FAIL（重试≤2次）
              │     ▼      └──► 存为草稿 ──► END
              │   END
              │
              └── 回退次数超限 ──► END（标记失败）
```

### 编排方案对比

| 编排方式 | 实现复杂度 | 条件分支 | 回退支持 | 并行支持 | 适用场景 |
| --- | --- | --- | --- | --- | --- |
| 简单 for 循环 | ★☆☆ | ❌ | ❌ | ❌ | Demo / 原型验证 |
| 责任链模式 | ★★☆ | ✅ | ❌ | ❌ | 线性流程、无回退 |
| DAG 引擎（自研） | ★★★ | ✅ | ✅ | ✅ | 本周实战选用 |
| LangGraph | ★★★ | ✅ | ✅ | ✅ | 生产级首选 |
| Temporal / Prefect | ★★★★ | ✅ | ✅ | ✅ | 大规模分布式编排 |

### DAG 引擎核心代码

```python
from enum import Enum
from typing import Callable, Any


class NodeStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


class DAGNode:
    """DAG 中的单个节点"""
    def __init__(self, node_id: str, execute_fn: Callable, role: str):
        self.node_id = node_id
        self.execute_fn = execute_fn
        self.role = role
        self.status = NodeStatus.PENDING
        self.result: Any = None

    def run(self, input_data: dict) -> Any:
        self.status = NodeStatus.RUNNING
        try:
            self.result = self.execute_fn(input_data)
            self.status = NodeStatus.SUCCESS
            return self.result
        except Exception as e:
            self.status = NodeStatus.FAILED
            raise


class PipelineDAG:
    """支持条件分支和回退的 DAG 流水线引擎"""

    def __init__(self, max_retries: int = 3):
        self.nodes: dict[str, DAGNode] = {}
        self.edges: list[tuple[str, str]] = []                    # (from, to)
        self.conditional_edges: dict[str, Callable] = {}          # node_id → router_fn
        self.max_retries = max_retries

    def add_node(self, node: DAGNode):
        self.nodes[node.node_id] = node

    def add_edge(self, from_id: str, to_id: str):
        self.edges.append((from_id, to_id))

    def add_conditional_edge(self, from_id: str, router_fn: Callable):
        """router_fn(result) → 下一个 node_id"""
        self.conditional_edges[from_id] = router_fn

    def get_next_nodes(self, current_id: str, result: Any) -> list[str]:
        if current_id in self.conditional_edges:
            next_id = self.conditional_edges[current_id](result)
            return [next_id] if next_id else []
        return [to_id for from_id, to_id in self.edges if from_id == current_id]

    def execute(self, start_node: str, initial_input: dict) -> dict:
        """执行整个 DAG 流水线"""
        current = start_node
        data = initial_input
        execution_log = []
        retry_counts: dict[str, int] = {}

        while current and current != "__END__":
            node = self.nodes[current]
            retry_counts.setdefault(current, 0)

            log_entry = {
                "node_id": current,
                "role": node.role,
                "input_snapshot": str(data)[:200],
                "status": None,
                "output_snapshot": None,
            }

            try:
                result = node.run(data)
                log_entry["status"] = "success"
                log_entry["output_snapshot"] = str(result)[:200]
                data = result if isinstance(result, dict) else {"result": result}
            except Exception as e:
                retry_counts[current] += 1
                log_entry["status"] = f"failed (attempt {retry_counts[current]})"

                if retry_counts[current] >= self.max_retries:
                    log_entry["status"] = "failed_final"
                    execution_log.append(log_entry)
                    break

                execution_log.append(log_entry)
                continue

            execution_log.append(log_entry)
            next_nodes = self.get_next_nodes(current, data)
            current = next_nodes[0] if next_nodes else None

        return {"final_output": data, "execution_log": execution_log}
```

---

## 第四部分学习：全流程日志追溯设计

### 为什么日志追溯是刚需？

MAS 系统最恐怖的 bug 不是"崩了"，而是"输出了一篇错误百出的博客但你不知道哪个 Agent 搞的鬼"。没有日志追溯的 MAS 就像没有黑匣子的飞机——出了事故只能猜。

### 生活类比：快递物流追踪

你在淘宝买了件衣服，物流信息显示：

> 03-01 10:00 已揽收（广州仓）→ 03-01 18:00 到达中转站（武汉）→ 03-02 09:00 派送中（北京）→ 03-02 14:00 已签收

每个环节都有**时间戳 + 操作人 + 状态 + 地点**。如果包裹丢了，你能精确定位到"武汉中转站到北京之间出了问题"。Agent 日志追溯完全一样——每个节点都要记录"谁在什么时间拿到了什么、做了什么、产出了什么"。

### 日志追溯的四层模型

| 层级 | 记录内容 | 粒度 | 用途 | 类比 |
| --- | --- | --- | --- | --- |
| Pipeline 层 | 整条流水线的开始/结束/总耗时/最终状态 | 一条 | 全局监控看板 | 包裹从发件到签收的总运单 |
| Node 层 | 每个节点的输入/输出/耗时/状态/重试次数 | 每个节点一条 | 故障定位 | 物流每个站点的扫描记录 |
| LLM Call 层 | 每次 LLM 调用的 Prompt/Response/Token 用量/延迟 | 每次调用一条 | 成本分析 + Prompt 调优 | 每次分拣员的操作录像 |
| Tool Call 层 | 每次外部工具调用的参数/返回/耗时 | 每次调用一条 | API 故障排查 | 每次快递员扫码枪的记录 |

### 结构化日志设计

```python
import uuid
import time
import json
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class TraceSpan:
    """单个追踪片段——对应一次节点执行或工具调用"""
    span_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    trace_id: str = ""           # 整条流水线共享的追踪 ID
    parent_span_id: str = ""     # 父级 span（用于嵌套，如 Node 下的 LLM Call）
    node_id: str = ""
    role: str = ""
    operation: str = ""          # execute_node / llm_call / tool_call
    status: str = "started"      # started / success / failed / retrying
    input_data: str = ""
    output_data: str = ""
    error_message: str = ""
    start_time: float = 0.0
    end_time: float = 0.0
    duration_ms: float = 0.0
    retry_attempt: int = 0
    metadata: dict = field(default_factory=dict)   # token_count, model_name 等

    def finish(self, status: str, output: str = "", error: str = ""):
        self.end_time = time.time()
        self.duration_ms = round((self.end_time - self.start_time) * 1000, 2)
        self.status = status
        self.output_data = output[:500]
        self.error_message = error


class PipelineTracer:
    """流水线全局追踪器"""

    def __init__(self):
        self.trace_id = str(uuid.uuid4())[:12]
        self.spans: list[TraceSpan] = []
        self.pipeline_start = time.time()

    def start_span(self, node_id: str, role: str, operation: str,
                   input_data: str, parent_span_id: str = "") -> TraceSpan:
        span = TraceSpan(
            trace_id=self.trace_id,
            parent_span_id=parent_span_id,
            node_id=node_id,
            role=role,
            operation=operation,
            input_data=input_data[:500],
            start_time=time.time(),
        )
        self.spans.append(span)
        return span

    def get_summary(self) -> dict:
        total_duration = round((time.time() - self.pipeline_start) * 1000, 2)
        node_spans = [s for s in self.spans if s.operation == "execute_node"]
        success_count = sum(1 for s in node_spans if s.status == "success")
        failed_count = sum(1 for s in node_spans if s.status == "failed")

        return {
            "trace_id": self.trace_id,
            "total_duration_ms": total_duration,
            "total_spans": len(self.spans),
            "node_executions": len(node_spans),
            "success": success_count,
            "failed": failed_count,
            "spans_detail": [asdict(s) for s in self.spans],
        }

    def print_timeline(self):
        """打印可视化时间线"""
        print(f"\n{'='*60}")
        print(f"  Pipeline Trace: {self.trace_id}")
        print(f"{'='*60}")
        for span in self.spans:
            status_icon = {"success": "✅", "failed": "❌", "started": "🔄",
                           "retrying": "🔁"}.get(span.status, "❓")
            indent = "    " if span.parent_span_id else "  "
            print(f"{indent}{status_icon} [{span.node_id}] {span.operation}"
                  f" | {span.role} | {span.duration_ms}ms | {span.status}")
            if span.error_message:
                print(f"{indent}   ⚠️  {span.error_message}")
        print(f"{'='*60}\n")
```

### 日志输出格式对比

| 格式 | 优点 | 缺点 | 适用场景 |
| --- | --- | --- | --- |
| JSON Lines | 机器可读、可直接导入 ELK/Loki | 人类阅读不便 | 生产环境 |
| 表格打印 | 人类友好、调试方便 | 不适合海量数据 | 开发调试 |
| OpenTelemetry | 标准协议、生态丰富 | 接入成本高 | 企业级监控 |
| 自定义 Markdown | 直观展示、可作为报告 | 不可程序化分析 | 演示交付 |

---

## 第五部分学习：无人工干预完成率统计方法

### 什么是"无人工干预完成率"？

**定义**：在 N 次端到端执行中，从接收口语指令到最终发布成功、全程没有任何人工介入的次数占比。

```
无人工干预完成率 = 全自动成功次数 / 总执行次数 × 100%
```

### 生活类比：自动贩卖机成功率

你往自动贩卖机里投了 100 次硬币买可乐：

- 82 次：硬币进去 → 按钮亮 → 可乐掉出来 → 完美
- 10 次：硬币卡住了，你拍了一下机器才行（人工干预）
- 5 次：选了可乐但出来了雪碧（结果错误）
- 3 次：吞了硬币啥也没出来（彻底失败）

无人工干预完成率 = 82 / 100 = 82%

### 完成率计算的细粒度拆解

| 统计维度 | 定义 | 计算方式 | 达标阈值 |
| --- | --- | --- | --- |
| 端到端完成率 | 最终成功发布的比例 | 发布成功次数 / 总执行次数 | > 80% |
| 零干预完成率 | 全程无任何人工介入 | 零人工介入次数 / 总执行次数 | > 80% |
| 单节点成功率 | 每个 Agent 单独的成功率 | 节点成功次数 / 节点执行次数 | > 90% |
| 回退率 | 因核查不通过触发回退的比例 | 回退次数 / 总执行次数 | < 30% |
| 平均重试次数 | 每次执行的平均重试总数 | 总重试次数 / 总执行次数 | < 2 |

### 统计引擎代码

```python
from dataclasses import dataclass, field


@dataclass
class SingleRunStats:
    """单次流水线执行的统计数据"""
    run_id: str
    success: bool = False
    human_intervention: bool = False
    total_nodes: int = 0
    nodes_succeeded: int = 0
    nodes_failed: int = 0
    retries: int = 0
    rollbacks: int = 0         # 回退次数
    total_duration_ms: float = 0.0


class CompletionRateTracker:
    """完成率统计引擎"""

    def __init__(self):
        self.runs: list[SingleRunStats] = []

    def record_run(self, stats: SingleRunStats):
        self.runs.append(stats)

    def compute_metrics(self) -> dict:
        if not self.runs:
            return {"error": "no runs recorded"}

        total = len(self.runs)
        success_count = sum(1 for r in self.runs if r.success)
        zero_intervention = sum(1 for r in self.runs
                                if r.success and not r.human_intervention)
        total_retries = sum(r.retries for r in self.runs)
        total_rollbacks = sum(r.rollbacks for r in self.runs)

        per_node_stats = {}
        for r in self.runs:
            per_node_stats.setdefault("total_nodes", 0)
            per_node_stats.setdefault("succeeded_nodes", 0)
            per_node_stats["total_nodes"] += r.total_nodes
            per_node_stats["succeeded_nodes"] += r.nodes_succeeded

        node_success_rate = (per_node_stats["succeeded_nodes"]
                             / max(per_node_stats["total_nodes"], 1) * 100)

        return {
            "total_runs": total,
            "end_to_end_completion_rate": f"{success_count / total * 100:.1f}%",
            "zero_intervention_rate": f"{zero_intervention / total * 100:.1f}%",
            "node_success_rate": f"{node_success_rate:.1f}%",
            "rollback_rate": f"{total_rollbacks / total * 100:.1f}%",
            "avg_retries_per_run": f"{total_retries / total:.2f}",
            "avg_duration_ms": f"{sum(r.total_duration_ms for r in self.runs) / total:.0f}",
        }

    def print_report(self):
        metrics = self.compute_metrics()
        print("\n" + "=" * 50)
        print("  MAS 流水线完成率统计报告")
        print("=" * 50)
        for k, v in metrics.items():
            label = {
                "total_runs": "总执行次数",
                "end_to_end_completion_rate": "端到端完成率",
                "zero_intervention_rate": "零干预完成率",
                "node_success_rate": "单节点成功率",
                "rollback_rate": "回退率",
                "avg_retries_per_run": "平均重试次数/次",
                "avg_duration_ms": "平均耗时(ms)",
            }.get(k, k)
            status = ""
            if k == "zero_intervention_rate":
                rate = float(v.strip('%'))
                status = " ✅" if rate > 80 else " ❌"
            print(f"  {label}: {v}{status}")
        print("=" * 50)


# --- 演示：模拟 10 次执行的统计 ---
tracker = CompletionRateTracker()
import random

for i in range(10):
    success = random.random() < 0.85
    intervention = random.random() < 0.1
    rollbacks = random.randint(0, 2) if success else random.randint(1, 3)
    tracker.record_run(SingleRunStats(
        run_id=f"run_{i+1:03d}",
        success=success,
        human_intervention=intervention and not success,
        total_nodes=5,
        nodes_succeeded=5 if success else random.randint(2, 4),
        nodes_failed=0 if success else 1,
        retries=random.randint(0, 3),
        rollbacks=rollbacks,
        total_duration_ms=random.uniform(3000, 12000),
    ))

tracker.print_report()
```

---

## 第六部分学习：实战代码——完整五角色全自动博客生产流水线

### 完整可运行代码

下面是一个自包含的、可直接运行的五角色 MAS 博客生产流水线。为了可独立执行（不依赖实际 API），所有 LLM 调用和搜索 API 均用模拟函数代替，但**架构、日志、条件分支、回退机制完全真实**。

```python
"""
MAS 闭环联动演练：五角色全自动博客生产流水线
==============================================

输入：高跨度口语指令 —— "在网上搜最近大模型资讯并写五百字博客且审查发布"
输出：已发布博客 + 全流程追溯日志 + 完成率统计

五角色：Researcher → Writer → Fact Checker → Editor → Publisher
"""

import uuid
import time
import json
import random
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Any, Optional, Callable
from enum import Enum


# ============================================================
# 第一层：结构化日志追溯系统
# ============================================================

class SpanStatus(Enum):
    STARTED = "started"
    SUCCESS = "success"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass
class TraceSpan:
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    trace_id: str = ""
    node_id: str = ""
    role: str = ""
    operation: str = ""
    status: str = "started"
    input_snapshot: str = ""
    output_snapshot: str = ""
    error_msg: str = ""
    start_ts: float = 0.0
    end_ts: float = 0.0
    duration_ms: float = 0.0
    retry_attempt: int = 0

    def finish(self, status: str, output: str = "", error: str = ""):
        self.end_ts = time.time()
        self.duration_ms = round((self.end_ts - self.start_ts) * 1000, 2)
        self.status = status
        self.output_snapshot = output[:300]
        self.error_msg = error


class PipelineTracer:
    def __init__(self):
        self.trace_id = uuid.uuid4().hex[:12]
        self.spans: list[TraceSpan] = []
        self.start_time = time.time()

    def begin(self, node_id: str, role: str, operation: str,
              input_data: str) -> TraceSpan:
        span = TraceSpan(
            trace_id=self.trace_id,
            node_id=node_id,
            role=role,
            operation=operation,
            input_snapshot=input_data[:300],
            start_ts=time.time(),
        )
        self.spans.append(span)
        return span

    def print_timeline(self):
        elapsed = round((time.time() - self.start_time) * 1000, 2)
        print(f"\n{'='*65}")
        print(f"  📋 Pipeline Trace  ID: {self.trace_id}  总耗时: {elapsed}ms")
        print(f"{'='*65}")
        for s in self.spans:
            icon = {"success": "✅", "failed": "❌", "retrying": "🔁"
                    }.get(s.status, "🔄")
            print(f"  {icon} [{s.node_id}] {s.role:<14} | {s.operation:<12}"
                  f" | {s.duration_ms:>8}ms | {s.status}")
            if s.error_msg:
                print(f"       ⚠️  {s.error_msg}")
        print(f"{'='*65}\n")

    def to_json_lines(self) -> str:
        return "\n".join(json.dumps(asdict(s), ensure_ascii=False)
                         for s in self.spans)


# ============================================================
# 第二层：五角色 Agent 实现
# ============================================================

class ResearcherAgent:
    """调研员：搜索最近大模型资讯"""

    ROLE = "Researcher"

    def execute(self, input_data: dict) -> dict:
        time.sleep(0.05)  # 模拟搜索延迟
        articles = [
            {"title": "GPT-5 发布：推理能力大幅跃升",
             "summary": "OpenAI 发布 GPT-5，在数学推理和代码生成上较 GPT-4 提升约 40%，"
                        "支持百万 Token 上下文窗口。",
             "source": "https://openai.com/blog/gpt5",
             "date": "2026-02-25"},
            {"title": "Claude 4 支持原生多模态 Agent",
             "summary": "Anthropic 发布 Claude 4，首次支持原生视觉-语言-动作（VLA）一体化，"
                        "可直接操控浏览器和桌面应用。",
             "source": "https://anthropic.com/claude4",
             "date": "2026-02-27"},
            {"title": "开源模型 Llama 4 登顶 LMSYS 竞技场",
             "summary": "Meta 开源 Llama 4 405B，首次在 LMSYS Chatbot Arena 综合排名"
                        "超越闭源模型，引发行业震动。",
             "source": "https://ai.meta.com/llama4",
             "date": "2026-02-28"},
        ]
        return {
            "articles": articles,
            "search_query": input_data.get("topic", "大模型"),
            "total_found": len(articles),
        }


class WriterAgent:
    """作者：基于调研结果撰写博客"""

    ROLE = "Writer"

    def execute(self, input_data: dict) -> dict:
        time.sleep(0.05)
        articles = input_data.get("articles", [])
        sources = [a["source"] for a in articles]
        summaries = "\n".join(f"- {a['title']}：{a['summary']}" for a in articles)

        body = f"""近期大模型领域迎来多项重磅更新，行业格局正在加速重塑。

首先，OpenAI 正式发布 GPT-5，该模型在数学推理和代码生成任务上较前代提升约 40%，并支持百万级别 Token 的超长上下文窗口，这意味着开发者可以一次性处理整本书籍或大型代码库，显著拓展了应用边界。

与此同时，Anthropic 推出了 Claude 4，其最大亮点是首次支持原生视觉-语言-动作（VLA）一体化架构。这使得 Claude 4 不再局限于文本对话，而是可以直接操控浏览器和桌面应用，在自动化办公和 Agent 构建领域展现出巨大潜力。

在开源阵营，Meta 的 Llama 4 405B 同样引发了行业震动。该模型首次在 LMSYS Chatbot Arena 综合排名中超越了所有闭源模型，标志着开源大模型在能力上已经具备与商业模型正面竞争的实力。

这三项进展共同勾勒出 2026 年初大模型发展的核心趋势：推理能力的持续跃升、多模态 Agent 的落地加速、以及开源与闭源的差距快速收窄。对于 AI 从业者而言，紧跟这些动态、及时调整技术路线，已经不再是"加分项"而是"必修课"。"""

        word_count = len(body.replace("\n", "").replace(" ", ""))

        return {
            "title": "2026年初大模型三大重磅更新：GPT-5、Claude 4、Llama 4 全解析",
            "body": body,
            "word_count": word_count,
            "sources_cited": sources,
        }


class FactCheckerAgent:
    """事实核查员：核查博客内容"""

    ROLE = "FactChecker"

    def execute(self, input_data: dict) -> dict:
        time.sleep(0.05)
        body = input_data.get("body", "")
        sources = input_data.get("sources_cited", [])

        checks = [
            {"claim": "GPT-5 推理能力较 GPT-4 提升约 40%",
             "status": "verified", "note": "与 OpenAI 官方博客一致"},
            {"claim": "Claude 4 支持原生 VLA 一体化",
             "status": "verified", "note": "与 Anthropic 发布公告一致"},
            {"claim": "Llama 4 在 LMSYS 排名超越闭源模型",
             "status": "verified", "note": "LMSYS 排行榜数据可查证"},
        ]

        all_verified = all(c["status"] == "verified" for c in checks)

        # 模拟 85% 的概率通过核查（演示回退机制）
        if random.random() < 0.85:
            verdict = "pass"
        else:
            verdict = "fail"
            checks[1]["status"] = "unverified"
            checks[1]["note"] = "需要补充 Anthropic 官方链接确认"

        return {
            "verdict": verdict,
            "issues": checks,
            "revised_body": body,
            "sources_cited": sources,
            "title": input_data.get("title", ""),
            "word_count": input_data.get("word_count", 0),
        }


class EditorAgent:
    """编辑：润色并生成摘要和标签"""

    ROLE = "Editor"

    def execute(self, input_data: dict) -> dict:
        time.sleep(0.05)
        return {
            "final_title": input_data.get("title", "无标题"),
            "final_body": input_data.get("revised_body", input_data.get("body", "")),
            "summary": "本文综述了 2026 年初 GPT-5、Claude 4 和 Llama 4 三大模型的"
                       "重要发布，分析了推理增强、多模态 Agent 和开源崛起三大趋势。",
            "tags": ["大模型", "GPT-5", "Claude 4", "Llama 4", "AI趋势"],
            "word_count": input_data.get("word_count", 0),
        }


class PublisherAgent:
    """发布员：将终稿发布到平台"""

    ROLE = "Publisher"

    def execute(self, input_data: dict) -> dict:
        time.sleep(0.05)
        # 模拟 90% 的发布成功率
        if random.random() < 0.9:
            return {
                "success": True,
                "url": f"https://blog.example.com/posts/{uuid.uuid4().hex[:8]}",
                "platform": "example_blog",
                "published_at": datetime.now().isoformat(),
                "final_title": input_data.get("final_title", ""),
            }
        else:
            raise ConnectionError("博客平台 API 超时，发布失败")


# ============================================================
# 第三层：DAG 流水线引擎（含条件分支 + 回退）
# ============================================================

class BlogPipeline:
    """五角色博客生产流水线"""

    def __init__(self):
        self.researcher = ResearcherAgent()
        self.writer = WriterAgent()
        self.fact_checker = FactCheckerAgent()
        self.editor = EditorAgent()
        self.publisher = PublisherAgent()
        self.tracer = PipelineTracer()
        self.max_fact_check_retries = 3
        self.max_publish_retries = 2

    def _run_node(self, node_id: str, agent, input_data: dict,
                  retry_attempt: int = 0) -> dict:
        span = self.tracer.begin(
            node_id=node_id,
            role=agent.ROLE,
            operation="execute_node",
            input_data=json.dumps(
                {k: str(v)[:100] for k, v in input_data.items()},
                ensure_ascii=False),
        )
        span.retry_attempt = retry_attempt

        try:
            result = agent.execute(input_data)
            span.finish("success",
                        output=json.dumps(
                            {k: str(v)[:100] for k, v in result.items()},
                            ensure_ascii=False))
            return result
        except Exception as e:
            span.finish("failed", error=str(e))
            raise

    def run(self, instruction: str) -> dict:
        """执行完整流水线"""
        print(f"\n🚀 启动流水线")
        print(f"   指令：{instruction}")
        print(f"   Trace ID：{self.tracer.trace_id}\n")

        run_stats = {
            "success": False,
            "human_intervention": False,
            "total_nodes": 5,
            "nodes_succeeded": 0,
            "retries": 0,
            "rollbacks": 0,
        }

        # --- Node 1: Researcher ---
        print("  📡 [1/5] Researcher 正在搜索资讯...")
        try:
            research_result = self._run_node(
                "N1_Research", self.researcher, {"topic": "大模型", "time_range": "7d"})
            run_stats["nodes_succeeded"] += 1
            print(f"       ✅ 找到 {research_result['total_found']} 条资讯")
        except Exception as e:
            print(f"       ❌ 调研失败: {e}")
            self.tracer.print_timeline()
            return {**run_stats, "final_output": None}

        # --- Node 2: Writer ---
        print("  ✍️  [2/5] Writer 正在撰写博客...")
        try:
            draft_result = self._run_node(
                "N2_Write", self.writer, research_result)
            run_stats["nodes_succeeded"] += 1
            print(f"       ✅ 初稿完成，{draft_result['word_count']} 字")
        except Exception as e:
            print(f"       ❌ 写作失败: {e}")
            self.tracer.print_timeline()
            return {**run_stats, "final_output": None}

        # --- Node 3: Fact Checker（含回退机制）---
        fact_check_input = draft_result
        for attempt in range(self.max_fact_check_retries):
            print(f"  🔍 [3/5] Fact Checker 正在核查"
                  f"{'（第 ' + str(attempt+1) + ' 次重审）' if attempt > 0 else ''}...")
            try:
                fc_result = self._run_node(
                    "N3_FactCheck", self.fact_checker,
                    fact_check_input, retry_attempt=attempt)
            except Exception as e:
                run_stats["retries"] += 1
                print(f"       ❌ 核查异常: {e}")
                continue

            if fc_result["verdict"] == "pass":
                run_stats["nodes_succeeded"] += 1
                verified_count = sum(1 for i in fc_result["issues"]
                                     if i["status"] == "verified")
                print(f"       ✅ 核查通过 ({verified_count}/{len(fc_result['issues'])} 条事实验证)")
                break
            else:
                run_stats["rollbacks"] += 1
                run_stats["retries"] += 1
                failed_claims = [i["claim"] for i in fc_result["issues"]
                                 if i["status"] != "verified"]
                print(f"       🔁 核查未通过，存疑: {failed_claims}")
                print(f"          → 回退给 Writer 修改...")

                # 回退给 Writer 重写
                fact_check_input = self._run_node(
                    "N2_Write_Retry", self.writer, research_result,
                    retry_attempt=attempt + 1)
                print(f"       ✍️  Writer 已修改初稿")
        else:
            print(f"       ❌ 核查 {self.max_fact_check_retries} 次仍未通过，流水线终止")
            self.tracer.print_timeline()
            return {**run_stats, "final_output": None}

        # --- Node 4: Editor ---
        print("  📝 [4/5] Editor 正在编排格式...")
        try:
            edit_result = self._run_node("N4_Edit", self.editor, fc_result)
            run_stats["nodes_succeeded"] += 1
            print(f"       ✅ 编排完成，标签: {edit_result['tags']}")
        except Exception as e:
            print(f"       ❌ 编辑失败: {e}")
            self.tracer.print_timeline()
            return {**run_stats, "final_output": None}

        # --- Node 5: Publisher（含重试）---
        for attempt in range(self.max_publish_retries):
            print(f"  🚀 [5/5] Publisher 正在发布"
                  f"{'（重试 ' + str(attempt+1) + '）' if attempt > 0 else ''}...")
            try:
                pub_result = self._run_node(
                    "N5_Publish", self.publisher,
                    edit_result, retry_attempt=attempt)
                if pub_result["success"]:
                    run_stats["nodes_succeeded"] += 1
                    run_stats["success"] = True
                    print(f"       ✅ 发布成功！URL: {pub_result['url']}")
                    break
            except Exception as e:
                run_stats["retries"] += 1
                print(f"       🔁 发布失败: {e}，准备重试...")
        else:
            print("       ⚠️  发布多次失败，保存为草稿")
            pub_result = {"success": False, "url": "draft://local",
                          "platform": "local_draft",
                          "published_at": datetime.now().isoformat()}

        # --- 输出追溯日志 ---
        self.tracer.print_timeline()

        return {**run_stats, "final_output": pub_result}


# ============================================================
# 第四层：批量执行 + 完成率统计
# ============================================================

@dataclass
class RunStats:
    run_id: str
    success: bool = False
    human_intervention: bool = False
    total_nodes: int = 0
    nodes_succeeded: int = 0
    nodes_failed: int = 0
    retries: int = 0
    rollbacks: int = 0
    total_duration_ms: float = 0.0


class CompletionTracker:
    def __init__(self):
        self.runs: list[RunStats] = []

    def record(self, stats: RunStats):
        self.runs.append(stats)

    def report(self) -> dict:
        n = len(self.runs)
        if n == 0:
            return {}
        succ = sum(1 for r in self.runs if r.success)
        zero = sum(1 for r in self.runs if r.success and not r.human_intervention)
        tot_nodes = sum(r.total_nodes for r in self.runs)
        succ_nodes = sum(r.nodes_succeeded for r in self.runs)
        tot_retries = sum(r.retries for r in self.runs)
        tot_rollbacks = sum(r.rollbacks for r in self.runs)
        avg_dur = sum(r.total_duration_ms for r in self.runs) / n

        return {
            "总执行次数": n,
            "端到端完成率": f"{succ/n*100:.1f}%",
            "零干预完成率": f"{zero/n*100:.1f}%",
            "单节点成功率": f"{succ_nodes/max(tot_nodes,1)*100:.1f}%",
            "回退率": f"{tot_rollbacks/n*100:.1f}%",
            "平均重试次数": f"{tot_retries/n:.2f}",
            "平均耗时(ms)": f"{avg_dur:.0f}",
        }

    def print_report(self):
        m = self.report()
        print("\n" + "=" * 55)
        print("  📊 MAS 博客流水线 · 完成率统计报告")
        print("=" * 55)
        for k, v in m.items():
            flag = ""
            if k == "零干预完成率":
                rate = float(v.strip('%'))
                flag = " ✅ 达标" if rate > 80 else " ❌ 未达标"
            print(f"    {k:<16} : {v}{flag}")
        print("=" * 55)


# ============================================================
# 主入口：执行演练
# ============================================================

def main():
    instruction = "在网上搜最近大模型资讯并写五百字博客且审查发布"
    num_runs = 10

    print("╔" + "═"*58 + "╗")
    print("║  MAS 闭环联动演练：五角色全自动博客生产流水线        ║")
    print("╚" + "═"*58 + "╝")
    print(f"\n📌 口语指令：「{instruction}」")
    print(f"📌 批量执行：{num_runs} 次\n")

    tracker = CompletionTracker()

    for i in range(num_runs):
        print(f"\n{'─'*55}")
        print(f"  🔄 第 {i+1}/{num_runs} 次执行")
        print(f"{'─'*55}")

        pipeline = BlogPipeline()
        start = time.time()
        result = pipeline.run(instruction)
        elapsed = round((time.time() - start) * 1000, 2)

        tracker.record(RunStats(
            run_id=f"run_{i+1:03d}",
            success=result.get("success", False),
            human_intervention=result.get("human_intervention", False),
            total_nodes=result.get("total_nodes", 5),
            nodes_succeeded=result.get("nodes_succeeded", 0),
            nodes_failed=5 - result.get("nodes_succeeded", 0),
            retries=result.get("retries", 0),
            rollbacks=result.get("rollbacks", 0),
            total_duration_ms=elapsed,
        ))

    # 最终统计报告
    tracker.print_report()

    # 输出最后一次执行的完整 JSON Lines 日志样本
    print("\n📋 最后一次执行的追溯日志样本 (JSON Lines):")
    print("─" * 55)
    print(pipeline.tracer.to_json_lines())
    print("─" * 55)


if __name__ == "__main__":
    main()
```

### 代码架构总览

```
┌──────────────────────────────────────────────────────┐
│                      main()                          │
│  ┌─────────────────────────────────────────────────┐ │
│  │            CompletionTracker                    │ │
│  │  ┌───────────────────────────────────────────┐  │ │
│  │  │           BlogPipeline                    │  │ │
│  │  │  ┌─────────────────────────────────────┐  │  │ │
│  │  │  │         PipelineTracer              │  │  │ │
│  │  │  │  (每个节点执行生成 TraceSpan)        │  │  │ │
│  │  │  └─────────────────────────────────────┘  │  │ │
│  │  │                                           │  │ │
│  │  │  N1: ResearcherAgent ──────────┐          │  │ │
│  │  │  N2: WriterAgent ◄─────────┐   │          │  │ │
│  │  │  N3: FactCheckerAgent ─┬───┘   │ DAG      │  │ │
│  │  │                  PASS  │ FAIL → 回退      │  │ │
│  │  │  N4: EditorAgent ◄─────┘                  │  │ │
│  │  │  N5: PublisherAgent (重试≤2次)             │  │ │
│  │  └───────────────────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────┘
```

### 运行效果示例

```
╔══════════════════════════════════════════════════════════╗
║  MAS 闭环联动演练：五角色全自动博客生产流水线            ║
╚══════════════════════════════════════════════════════════╝

📌 口语指令：「在网上搜最近大模型资讯并写五百字博客且审查发布」
📌 批量执行：10 次

───────────────────────────────────────────────────────
  🔄 第 1/10 次执行
───────────────────────────────────────────────────────

🚀 启动流水线
   指令：在网上搜最近大模型资讯并写五百字博客且审查发布
   Trace ID：a1b2c3d4e5f6

  📡 [1/5] Researcher 正在搜索资讯...
       ✅ 找到 3 条资讯
  ✍️  [2/5] Writer 正在撰写博客...
       ✅ 初稿完成，462 字
  🔍 [3/5] Fact Checker 正在核查...
       ✅ 核查通过 (3/3 条事实验证)
  📝 [4/5] Editor 正在编排格式...
       ✅ 编排完成，标签: ['大模型', 'GPT-5', 'Claude 4', 'Llama 4', 'AI趋势']
  🚀 [5/5] Publisher 正在发布...
       ✅ 发布成功！URL: https://blog.example.com/posts/f7a8b9c0

=================================================================
  📋 Pipeline Trace  ID: a1b2c3d4e5f6  总耗时: 267ms
=================================================================
  ✅ [N1_Research] Researcher     | execute_node |    52ms | success
  ✅ [N2_Write]    Writer         | execute_node |    51ms | success
  ✅ [N3_FactCheck] FactChecker   | execute_node |    53ms | success
  ✅ [N4_Edit]     Editor         | execute_node |    52ms | success
  ✅ [N5_Publish]  Publisher      | execute_node |    51ms | success
=================================================================

... (后续 9 次执行省略) ...

=======================================================
  📊 MAS 博客流水线 · 完成率统计报告
=======================================================
    总执行次数         : 10
    端到端完成率       : 90.0%
    零干预完成率       : 90.0% ✅ 达标
    单节点成功率       : 96.0%
    回退率             : 20.0%
    平均重试次数       : 0.40
    平均耗时(ms)       : 285
=======================================================
```

---

## 验收交付

### 交付一：全流程闭环验证

**验证清单**：

| 验证项 | 验证方法 | 预期结果 | 状态 |
| --- | --- | --- | --- |
| 意图拆解 | 输入口语指令，检查输出的任务链 | 拆解为 5 个有依赖关系的结构化任务 | ✅ |
| 五角色协作 | 执行流水线，检查各角色是否按序执行 | Researcher→Writer→FactChecker→Editor→Publisher | ✅ |
| 条件回退 | 多次执行观察核查失败时的回退行为 | 核查失败 → 回退 Writer → 重新核查 | ✅ |
| 发布重试 | 模拟发布失败 | 自动重试最多 2 次，最终降级为草稿 | ✅ |
| 日志追溯 | 检查每次执行的 TraceSpan | 每个节点有完整的 input/output/duration/status | ✅ |
| 完成率统计 | 批量执行 10 次后查看报告 | 零干预完成率 > 80% | ✅ |

### 交付二：追溯日志样本

**JSON Lines 格式日志样本**：

```json
{"span_id":"a1b2c3d4","trace_id":"f0e1d2c3b4a5","node_id":"N1_Research","role":"Researcher","operation":"execute_node","status":"success","input_snapshot":"{\"topic\":\"大模型\",\"time_range\":\"7d\"}","output_snapshot":"{\"total_found\":3,\"search_query\":\"大模型\"}","duration_ms":52.31}
{"span_id":"e5f6a7b8","trace_id":"f0e1d2c3b4a5","node_id":"N2_Write","role":"Writer","operation":"execute_node","status":"success","input_snapshot":"{\"articles\":[...]}","output_snapshot":"{\"word_count\":462,\"title\":\"2026年初大模型三大重磅更新\"}","duration_ms":51.44}
{"span_id":"c9d0e1f2","trace_id":"f0e1d2c3b4a5","node_id":"N3_FactCheck","role":"FactChecker","operation":"execute_node","status":"success","input_snapshot":"{\"body\":\"近期大模型领域...\"}","output_snapshot":"{\"verdict\":\"pass\",\"issues\":[...]}","duration_ms":53.12}
{"span_id":"a3b4c5d6","trace_id":"f0e1d2c3b4a5","node_id":"N4_Edit","role":"Editor","operation":"execute_node","status":"success","input_snapshot":"{\"revised_body\":\"近期大模型领域...\"}","output_snapshot":"{\"final_title\":\"2026年初大模型三大重磅更新\",\"tags\":[...]}","duration_ms":52.08}
{"span_id":"e7f8a9b0","trace_id":"f0e1d2c3b4a5","node_id":"N5_Publish","role":"Publisher","operation":"execute_node","status":"success","input_snapshot":"{\"final_title\":\"2026年初大模型三大重磅更新\"}","output_snapshot":"{\"success\":true,\"url\":\"https://blog.example.com/posts/f7a8b9c0\"}","duration_ms":51.77}
```

### 交付三：完成率统计报告（参考）

```
=======================================================
  📊 MAS 博客流水线 · 完成率统计报告
=======================================================
    总执行次数         : 10
    端到端完成率       : 90.0%
    零干预完成率       : 90.0% ✅ 达标
    单节点成功率       : 96.0%
    回退率             : 20.0%
    平均重试次数       : 0.40
    平均耗时(ms)       : 285
=======================================================

结论：零干预完成率 90% > 80% 阈值，验收通过 ✅
```

### 核心知识点自检表

| 知识点 | 一句话检验 | 掌握程度 |
| --- | --- | --- |
| 高跨度口语指令拆解 | 能否说出从口语到任务链的四个提取步骤？ | ☐ |
| 五角色分工设计 | 能否解释为什么不用单体 Agent 而要拆五个角色？ | ☐ |
| 角色间信息契约 | 能否为任意两个相邻角色定义结构化交接对象？ | ☐ |
| DAG + 条件分支 | 能否画出含回退箭头的完整流水线 DAG？ | ☐ |
| 全流程日志追溯 | 能否说出日志追溯的四个层级及各自用途？ | ☐ |
| TraceSpan 设计 | 能否解释 trace_id 和 span_id 的关系？ | ☐ |
| 无人工干预完成率 | 能否手算 10 次执行中的零干预完成率？ | ☐ |
| 完成率细粒度指标 | 能否区分端到端完成率和零干预完成率？ | ☐ |

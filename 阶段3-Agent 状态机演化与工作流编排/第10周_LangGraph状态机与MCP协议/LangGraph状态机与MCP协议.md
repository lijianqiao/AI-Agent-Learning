# LangGraph 状态机与 MCP 协议

## 学习

**学习目标**

- 自主图计算拓扑流转模型、节点定义与边条件转换控制理论 (Conditional Edges)
- MCP 协议 (Model Context Protocol)：Anthropic 主导的工具/资源集成标准，已被 LangGraph、Claude、Cursor 等主流 Agent 框架广泛采纳；掌握其 Server/Client 交互规范、Resources/Tools/Prompts 三类原语定义

**实战**

- 应用 LangGraph 或类似前沿网格化引擎绘制并替换早期的硬编码暴力 Agent 行为循环
- 基于 MCP 协议封装一个自定义工具服务（如封装本地文件检索能力），并接入 Agent 工具调用链路

**验收标准**

- 对图状有向无环或条件有步骤流转机制不迷航，杜绝长逻辑状态机发版导致的死锁溢出；能独立实现并部署一个可用的 MCP Server

---

## 第一部分学习：为什么需要状态机编排

### 硬编码 Agent 的痛苦

回忆第二周我们手写的 Vanilla Agent——一个 `while` 循环加上一堆 `if/else` 判断。当时处理的场景很简单：调用搜索、调用计算器、返回结果。但如果业务变复杂了呢？

想象你要搭建一个**客服退款 Agent**，流程如下：

1. 先判断用户意图（咨询 / 投诉 / 退款）
2. 如果是退款 → 查询订单状态
3. 如果订单已发货 → 需要用户提供快递单号 → 审核
4. 如果订单未发货 → 直接退款
5. 退款金额超过 500 → 需要主管审批
6. 审批通过 → 执行退款；审批拒绝 → 通知用户

用 if/else 硬编码写出来是什么样？

```python
# ❌ 硬编码版本（噩梦级代码）
def agent_loop(user_input):
    intent = classify_intent(user_input)
    if intent == "退款":
        order = query_order(user_input)
        if order["status"] == "已发货":
            tracking = ask_user_for_tracking()
            if tracking:
                review = audit_review(tracking)
                if review["passed"]:
                    if order["amount"] > 500:
                        approval = manager_approve(order)
                        if approval:
                            execute_refund(order)
                        else:
                            notify_user("主管拒绝退款")
                    else:
                        execute_refund(order)
                else:
                    notify_user("审核未通过")
        elif order["status"] == "未发货":
            if order["amount"] > 500:
                # 又一层嵌套...
                ...
    elif intent == "咨询":
        ...
    elif intent == "投诉":
        ...
```

### 生活类比：地铁线路图 vs 手写导航指令

**硬编码 if/else** 就像在纸上写：

> "出站后左转，走 200 米，看到红色大楼右转，再走 50 米，如果路口有施工就绕道……"

每增加一个分支都要重新写一遍所有路径，任何一个条件变了就要改几十行代码。

**状态机编排** 就像画一张**地铁线路图**：

> 每个站是一个"节点"，站与站之间的线路是"边"，换乘站有"条件分叉"（往浦东还是虹桥？看你要去哪）。新增一条线只要加节点和边，不用改已有的线路。

### 硬编码 vs 状态机编排对比

| 维度 | 硬编码 if/else | 状态机（LangGraph） |
| --- | --- | --- |
| 可读性 | 嵌套深度 5+ 层，人眼跟踪极难 | 图结构一目了然，节点即步骤 |
| 可维护性 | 改一个分支可能影响全局 | 新增/修改节点独立隔离 |
| 可测试性 | 难以对单个路径单元测试 | 每个节点可单独测试 |
| 死锁风险 | 容易出现遗漏的 else 分支导致卡死 | 框架内置最大步数保护 |
| 可视化 | 无法可视化 | 可导出为流程图 |
| 复用性 | 基本为零 | 节点函数可跨图复用 |
| 并行执行 | 需要手动管理线程 | 框架原生支持并行分支 |

### 核心结论

当 Agent 的决策路径超过 **3 个分叉点**时，就应该从硬编码切换到状态机编排。这不是"锦上添花"，而是**工程生存的必要条件**——否则你将面对的是一坨不可维护、不可调试、不可扩展的"意大利面条代码"。

---

## 第二部分学习：LangGraph 核心概念

### LangGraph 是什么？

LangGraph 是 LangChain 团队推出的**图状态机编排引擎**，专门用于构建复杂的、有状态的 Agent 工作流。它的核心思想是：**把 Agent 的行为建模为一张有向图，节点是操作，边是流转条件**。

### 生活类比：自动贩卖机

一台自动贩卖机就是一个典型的状态机：

- **状态**：等待投币 → 已投币 → 选择商品 → 出货中 → 完成
- **节点（操作）**：检测硬币、显示商品列表、弹出商品、找零
- **边（转换条件）**：投币金额够了 → 进入选择界面；金额不够 → 继续等待投币
- **全局状态**：当前余额、已选商品、库存数量

LangGraph 的工作方式与此完全一致，只是操作对象从硬币和商品变成了 LLM 调用和工具执行。

### 四大核心概念

#### 1. StateGraph（状态图）

**定义**：整个工作流的容器，就像自动贩卖机的"机箱"。它持有一个全局共享的状态对象（TypedDict），所有节点都可以读写这个状态。

```python
from langgraph.graph import StateGraph
from typing import TypedDict, Annotated
from operator import add

class AgentState(TypedDict):
    messages: Annotated[list, add]  # 消息历史（追加模式）
    current_step: str               # 当前步骤标记
    retry_count: int                # 重试计数
```

**关键点**：`Annotated[list, add]` 表示每个节点返回的 messages 会**追加**到已有列表，而不是覆盖。这是 LangGraph 的"归约器"（Reducer）机制。

#### 2. Node（节点）

**定义**：图中的每个"站点"，对应一个具体操作。可以是 LLM 调用、工具执行、数据处理等。每个节点是一个普通 Python 函数，接收状态、返回状态更新。

```python
def call_llm(state: AgentState) -> dict:
    """节点：调用大模型"""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def use_tool(state: AgentState) -> dict:
    """节点：执行工具"""
    last_msg = state["messages"][-1]
    result = execute_tool(last_msg.tool_calls[0])
    return {"messages": [result]}
```

#### 3. Edge（边）

**定义**：连接两个节点的"线路"，表示执行完 A 后无条件跳转到 B。

```python
graph.add_edge("call_llm", "use_tool")  # call_llm 完成后必定执行 use_tool
```

#### 4. Conditional Edge（条件边）

**定义**：最核心也最强大的概念——根据当前状态动态决定下一步走哪个节点。就像地铁的换乘站，根据你的目的地决定换哪条线。

```python
def should_continue(state: AgentState) -> str:
    """路由函数：决定下一步去哪"""
    last_msg = state["messages"][-1]
    if last_msg.tool_calls:
        return "use_tool"     # 模型想调用工具 → 去工具节点
    else:
        return "end"          # 模型给出最终答案 → 结束

graph.add_conditional_edges(
    "call_llm",               # 从哪个节点出发
    should_continue,           # 路由判断函数
    {
        "use_tool": "use_tool",  # 返回值 → 目标节点的映射
        "end": END,
    }
)
```

### 四大概念关系图

```
┌──────────────────────────────────────────┐
│            StateGraph（机箱）              │
│                                          │
│   AgentState: {messages, step, ...}      │
│                                          │
│   ┌─────────┐   Edge    ┌──────────┐    │
│   │  START   │─────────→│ call_llm │    │
│   └─────────┘           └────┬─────┘    │
│                              │           │
│                    Conditional Edge      │
│                      ┌───────┴───────┐   │
│                      ↓               ↓   │
│               ┌──────────┐     ┌─────┐   │
│               │ use_tool │     │ END │   │
│               └────┬─────┘     └─────┘   │
│                    │                     │
│                    │   Edge              │
│                    └──→ call_llm         │
│                                          │
└──────────────────────────────────────────┘
```

### 概念速查表

| 概念 | 类比 | 作用 | 代码关键字 |
| --- | --- | --- | --- |
| StateGraph | 地铁线路网总图 | 定义整个工作流骨架和全局状态 | `StateGraph(AgentState)` |
| Node | 地铁站 | 执行具体操作（LLM/工具/逻辑） | `graph.add_node("name", fn)` |
| Edge | 单行线路 | 无条件从 A 跳到 B | `graph.add_edge("A", "B")` |
| Conditional Edge | 换乘站 | 根据状态动态选择下一站 | `graph.add_conditional_edges(...)` |
| START | 起点站 | 图的入口 | `from langgraph.graph import START` |
| END | 终点站 | 图的出口，流程结束 | `from langgraph.graph import END` |

---

## 第三部分学习：死锁防御与最大步数限制

### 什么是状态机死锁？

**生活类比：两个人互相让路**

> 你和对面的人在狭窄走廊相遇。你往左让，他也往左让；你往右让，他也往右让。两个人不停地"礼让"，但谁也走不了——这就是**死锁**。

在状态机中，死锁通常发生在：

1. **循环依赖**：节点 A 等节点 B 完成，节点 B 等节点 A 完成
2. **条件永不满足**：条件边的判断条件永远走不到 END，图一直在打转
3. **无限重试**：工具调用失败 → 重试 → 继续失败 → 继续重试……

### 三种典型危险场景

| 危险模式 | 描述 | 后果 |
| --- | --- | --- |
| 无限循环 | LLM 反复决定调用工具，永远不给最终答案 | CPU 占满，API 费用爆炸 |
| 条件黑洞 | 条件边遗漏了某个返回值的映射 | 运行时抛异常或挂死 |
| 雪崩重试 | 工具失败触发重试，重试又失败，指数膨胀 | 下游服务被压垮 |

### 防御策略一：最大步数限制（recursion_limit）

LangGraph 内置了 `recursion_limit` 参数，限制图的最大执行步数：

```python
app = graph.compile()

result = app.invoke(
    {"messages": [HumanMessage(content="帮我查询天气")]},
    config={"recursion_limit": 25}  # 最多执行 25 步，超出则强制终止
)
```

**如何设定合理值？**

| 场景复杂度 | 建议 recursion_limit | 理由 |
| --- | --- | --- |
| 简单问答（1-2 次工具调用） | 10 | 留足裕量即可 |
| 中等流程（3-5 个节点流转） | 25 | 默认值，适合大多数场景 |
| 复杂多步（子图嵌套、并行） | 50-100 | 需要配合监控告警 |

### 防御策略二：节点内重试计数

在状态中维护重试计数器，超过阈值时强制走向降级路径：

```python
class AgentState(TypedDict):
    messages: Annotated[list, add]
    retry_count: int

def use_tool(state: AgentState) -> dict:
    try:
        result = call_external_api()
        return {"messages": [ToolMessage(content=result)], "retry_count": 0}
    except Exception as e:
        new_count = state["retry_count"] + 1
        return {
            "messages": [ToolMessage(content=f"工具失败(第{new_count}次): {e}")],
            "retry_count": new_count,
        }

def should_retry_or_fallback(state: AgentState) -> str:
    if state["retry_count"] >= 3:
        return "fallback"   # 超过 3 次 → 降级
    return "call_llm"       # 否则让 LLM 重新决策
```

### 防御策略三：超时熔断

对整个图执行设置超时，防止因外部服务卡死导致的无限等待：

```python
import asyncio

async def run_with_timeout(app, inputs, timeout_seconds=30):
    try:
        result = await asyncio.wait_for(
            app.ainvoke(inputs),
            timeout=timeout_seconds,
        )
        return result
    except asyncio.TimeoutError:
        return {"error": f"执行超时({timeout_seconds}s)，已强制终止"}
```

### 防御清单速查

| 防御层 | 手段 | 实现方式 |
| --- | --- | --- |
| 框架层 | 最大步数限制 | `recursion_limit` 参数 |
| 节点层 | 重试计数 + 降级 | 状态字段 `retry_count` + 条件边 |
| 系统层 | 超时熔断 | `asyncio.wait_for` 包装 |
| 监控层 | 步数/耗时告警 | LangSmith 或自定义 Trace |

---

## 第四部分学习：MCP 协议三类原语

### MCP 是什么？

**MCP（Model Context Protocol）** 是 Anthropic 于 2024 年底主导发布的开放标准协议，目标是为大模型提供一种**统一的方式**来访问外部工具、数据源和提示模板。

### 生活类比：USB 接口标准

在 USB 出现之前，每种设备都有自己的接口——打印机用并口、键盘用 PS/2、相机用 FireWire……每接一个新设备就要装一种新驱动。

**USB 的诞生** 统一了这一切：不管是鼠标、U 盘还是手机，插上就能用。

MCP 就是 AI Agent 世界的"USB 标准"：

- **没有 MCP**：每个工具对接都要写定制代码（LangChain 的 Tool 一套格式、AutoGPT 一套格式、Claude 又一套）
- **有了 MCP**：所有工具服务遵循统一协议，任何兼容 MCP 的 Agent 都能即插即用

### 三类原语（Primitives）

MCP 定义了三种核心能力原语，覆盖了 Agent 与外部世界交互的全部场景：

#### 1. Resources（资源）

**是什么**：向模型暴露**数据和内容**，类似于 REST API 的 GET 请求——只读、不产生副作用。

**类比**：图书馆的藏书目录。你可以查阅（读取），但不能修改书的内容。

**典型场景**：

- 读取本地文件内容
- 查询数据库中的记录
- 获取 API 返回的结构化数据
- 读取系统配置信息

```python
@server.resource("file://{path}")
async def read_file(path: str) -> str:
    """暴露本地文件为可读资源"""
    with open(path, "r", encoding="utf-8") as f:
        return f.read()
```

#### 2. Tools（工具）

**是什么**：让模型执行**有副作用的操作**，类似于 REST API 的 POST/PUT/DELETE——可以改变外部世界的状态。

**类比**：遥控器的按钮。按下去会产生实际效果（开灯、调温度、换频道）。

**典型场景**：

- 发送邮件
- 写入数据库
- 创建文件
- 调用第三方 API 执行操作

```python
@server.tool("send_email")
async def send_email(to: str, subject: str, body: str) -> str:
    """发送邮件（有副作用的工具）"""
    result = email_client.send(to=to, subject=subject, body=body)
    return f"邮件已发送至 {to}"
```

#### 3. Prompts（提示模板）

**是什么**：预定义的、可复用的**提示词模板**，帮助用户或 Agent 以标准化的方式与模型交互。

**类比**：公司的公文模板。你不用每次从头写报告，直接往模板里填内容就行。

**典型场景**：

- 代码审查提示模板
- 数据分析报告模板
- 翻译任务专用提示词

```python
@server.prompt("code_review")
async def code_review_prompt(code: str, language: str) -> str:
    """代码审查提示模板"""
    return f"""请审查以下 {language} 代码，从以下维度评估：
1. 安全性漏洞
2. 性能问题
3. 代码规范

```{language}
{code}
```

请逐条列出发现的问题和改进建议。"""
```

### 三类原语对比表

| 维度 | Resources（资源） | Tools（工具） | Prompts（提示模板） |
| --- | --- | --- | --- |
| 核心用途 | 暴露数据/内容 | 执行操作/产生副作用 | 提供标准化交互模板 |
| REST 类比 | GET（只读） | POST/PUT/DELETE（写入） | 预制请求模板 |
| 生活类比 | 图书馆藏书 | 遥控器按钮 | 公文模板 |
| 控制权 | 应用程序控制 | 模型自主调用 | 用户选择使用 |
| 副作用 | 无 | 有 | 无 |
| 典型例子 | 读文件、查数据库 | 发邮件、写文件 | 代码审查模板、分析模板 |

### 为什么是这三种？

这三类原语恰好覆盖了 Agent 与外部世界交互的完整闭环：

```
Agent 需要了解世界    →  Resources（读取数据）
Agent 需要改变世界    →  Tools（执行操作）
Agent 需要标准化表达  →  Prompts（复用模板）
```

---

## 第五部分学习：MCP Server/Client 交互流程

### 架构总览

MCP 采用经典的 **Client-Server 架构**，通信模型是请求-响应式的 JSON-RPC 2.0：

```
┌─────────────────────┐          ┌─────────────────────┐
│     MCP Client      │          │     MCP Server      │
│  (Agent / IDE)      │          │  (工具服务提供方)    │
│                     │  JSON-RPC│                     │
│  ┌───────────────┐  │ ◄──────► │  ┌───────────────┐  │
│  │ 发现可用资源  │  │  2.0    │  │ Resources     │  │
│  │ 调用工具      │  │  over   │  │ Tools         │  │
│  │ 获取提示模板  │  │  stdio/ │  │ Prompts       │  │
│  └───────────────┘  │  SSE    │  └───────────────┘  │
└─────────────────────┘          └─────────────────────┘
```

### 生活类比：点餐系统

把 MCP 的交互过程想象成在餐厅点餐：

1. **服务发现**（看菜单）：客人（Client）坐下后，先问服务员要菜单（Server 返回可用的 Resources/Tools/Prompts 列表）
2. **能力协商**（确认忌口）：客人告诉服务员自己的要求（Client 声明自己支持的协议版本和能力）
3. **调用工具**（点菜）：客人选好菜品下单（Client 发送 `tools/call` 请求）
4. **返回结果**（上菜）：厨房做好菜端上来（Server 执行工具并返回结果）

### 完整交互生命周期

#### 第一阶段：初始化握手

```
Client                          Server
  │                               │
  │── initialize(capabilities) ──→│  客户端发起连接，声明能力
  │←── capabilities + info ───────│  服务端返回自身能力和信息
  │── initialized ───────────────→│  客户端确认初始化完成
  │                               │
```

#### 第二阶段：服务发现

```
Client                          Server
  │                               │
  │── resources/list ────────────→│  "你有哪些可读资源？"
  │←── [file://*, db://orders] ──│  "这些文件和数据库"
  │                               │
  │── tools/list ────────────────→│  "你能执行哪些操作？"
  │←── [send_email, search...] ──│  "发邮件、搜索等"
  │                               │
  │── prompts/list ──────────────→│  "你有哪些提示模板？"
  │←── [code_review, summary] ───│  "代码审查、摘要等"
  │                               │
```

#### 第三阶段：运行时调用

```
Client                          Server
  │                               │
  │── tools/call(send_email,     │
  │     {to, subject, body})  ───→│  "帮我发一封邮件"
  │←── {result: "发送成功"} ──────│  "好的，已发送"
  │                               │
  │── resources/read             │
  │     ("file:///data.csv") ────→│  "给我这个文件内容"
  │←── {content: "..."} ─────────│  "这是文件内容"
  │                               │
```

### 两种传输方式

MCP 支持两种通信传输方式，适用于不同部署场景：

| 传输方式 | 通信机制 | 适用场景 | 生活类比 |
| --- | --- | --- | --- |
| **stdio** | 标准输入输出管道 | 本地进程间通信（IDE 插件、CLI 工具） | 面对面交谈 |
| **SSE + HTTP** | Server-Sent Events | 远程服务、云端部署、多客户端共享 | 电话通话 |

**stdio 方式**：Client 以子进程方式启动 Server，通过 stdin/stdout 通信。适合 Cursor、VS Code 等 IDE 场景。

**SSE 方式**：Server 独立运行在某个端口，Client 通过 HTTP 连接。适合微服务部署、多个 Agent 共享同一个工具服务。

### 一个请求的完整旅程

以 Agent 调用"搜索本地文件"工具为例：

```
1. Agent(LLM) 决定需要搜索文件
       ↓
2. MCP Client 构造 JSON-RPC 请求：
   {"jsonrpc":"2.0","method":"tools/call","params":{"name":"search_files","arguments":{"query":"报告"}}}
       ↓
3. 通过 stdio/SSE 发送给 MCP Server
       ↓
4. Server 解析请求，执行 search_files 函数
       ↓
5. Server 返回 JSON-RPC 响应：
   {"jsonrpc":"2.0","result":{"content":[{"type":"text","text":"找到3个匹配文件..."}]}}
       ↓
6. MCP Client 将结果转为 ToolMessage，放回 Agent 上下文
       ↓
7. Agent(LLM) 根据工具结果继续推理
```

---

## 第六部分学习：实战代码

### 实战一：用 LangGraph 构建多步骤 Agent

这个 Agent 能够根据用户问题，自主决定是直接回答、调用搜索工具还是调用计算器，并通过条件边实现流程流转。

```python
# 安装依赖：pip install langgraph langchain-openai langchain-core

from typing import TypedDict, Annotated, Literal
from operator import add
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
import json

# ===== 1. 定义全局状态 =====
class AgentState(TypedDict):
    messages: Annotated[list, add]
    retry_count: int

# ===== 2. 定义工具 =====
def calculator(expression: str) -> str:
    """安全计算数学表达式"""
    try:
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return "错误：表达式包含非法字符"
        return str(eval(expression))
    except Exception as e:
        return f"计算错误：{e}"

def search_knowledge(query: str) -> str:
    """模拟知识库搜索"""
    knowledge = {
        "langgraph": "LangGraph 是 LangChain 团队推出的图状态机编排引擎，用于构建复杂的有状态 Agent 工作流。",
        "mcp": "MCP (Model Context Protocol) 是 Anthropic 发布的开放协议，统一了大模型访问外部工具和数据源的方式。",
        "react": "ReAct 是一种让大模型交替进行推理(Reasoning)和行动(Acting)的提示范式。",
    }
    for key, val in knowledge.items():
        if key in query.lower():
            return val
    return f"未找到关于 '{query}' 的相关知识。"

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式，支持加减乘除和括号",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式"}
                },
                "required": ["expression"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": "搜索知识库获取技术概念解释",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"],
            },
        },
    },
]

TOOL_MAP = {"calculator": calculator, "search_knowledge": search_knowledge}

# ===== 3. 定义节点函数 =====
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def call_llm_node(state: AgentState) -> dict:
    """节点：调用大模型进行推理决策"""
    messages = state["messages"]
    sys_msg = SystemMessage(content="你是一个智能助手，可以使用工具回答问题。请用中文思考和回答。")
    response = llm.invoke(
        [sys_msg] + messages,
        tools=TOOLS_SCHEMA,
        tool_choice="auto",
    )
    return {"messages": [response]}

def execute_tool_node(state: AgentState) -> dict:
    """节点：执行工具调用"""
    last_msg = state["messages"][-1]
    tool_results = []

    for tool_call in last_msg.tool_calls:
        fn_name = tool_call["name"]
        fn_args = tool_call["args"]

        if fn_name in TOOL_MAP:
            result = TOOL_MAP[fn_name](**fn_args)
        else:
            result = f"错误：未知工具 '{fn_name}'"

        tool_results.append(
            ToolMessage(content=result, tool_call_id=tool_call["id"])
        )

    return {"messages": tool_results}

# ===== 4. 定义条件路由 =====
def should_continue(state: AgentState) -> Literal["execute_tool", "end"]:
    """条件边：判断是调用工具还是结束"""
    last_msg = state["messages"][-1]

    if hasattr(last_msg, "tool_calls") and last_msg.tool_calls:
        return "execute_tool"
    return "end"

# ===== 5. 构建图 =====
graph = StateGraph(AgentState)

graph.add_node("call_llm", call_llm_node)
graph.add_node("execute_tool", execute_tool_node)

graph.add_edge(START, "call_llm")

graph.add_conditional_edges(
    "call_llm",
    should_continue,
    {
        "execute_tool": "execute_tool",
        "end": END,
    },
)

graph.add_edge("execute_tool", "call_llm")

app = graph.compile()

# ===== 6. 运行测试 =====
if __name__ == "__main__":
    test_queries = [
        "LangGraph 是什么？请帮我查一下。",
        "请计算 (15 * 8 + 20) / 4 的结果。",
        "MCP 协议是什么？另外帮我算一下 100 / 3 保留两位小数。",
    ]

    for query in test_queries:
        print(f"\n{'='*60}")
        print(f"用户提问：{query}")
        print(f"{'='*60}")

        result = app.invoke(
            {"messages": [HumanMessage(content=query)], "retry_count": 0},
            config={"recursion_limit": 15},
        )

        final_msg = result["messages"][-1]
        print(f"\nAgent 回答：{final_msg.content}")

        print(f"\n--- 完整流转轨迹（共 {len(result['messages'])} 条消息）---")
        for i, msg in enumerate(result["messages"]):
            role = type(msg).__name__
            content = str(msg.content)[:80]
            extra = ""
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                tools = [tc["name"] for tc in msg.tool_calls]
                extra = f" → 调用工具: {tools}"
            print(f"  [{i+1}] {role}: {content}{extra}")
```

**预期输出（参考）**：

```
============================================================
用户提问：MCP 协议是什么？另外帮我算一下 100 / 3 保留两位小数。
============================================================

Agent 回答：MCP (Model Context Protocol) 是 Anthropic 发布的开放协议，统一了大模型访问外部工具和数据源的方式。另外，100 ÷ 3 ≈ 33.33。

--- 完整流转轨迹（共 5 条消息）---
  [1] HumanMessage: MCP 协议是什么？另外帮我算一下 100 / 3 保留两位小数。
  [2] AIMessage:  → 调用工具: ['search_knowledge', 'calculator']
  [3] ToolMessage: MCP (Model Context Protocol) 是 Anthropic 发布的开放协议...
  [4] ToolMessage: 33.333333333333336
  [5] AIMessage: MCP (Model Context Protocol) 是 Anthropic 发布的开放协议，统一了大模型...
```

> **关键观察**：LangGraph 自动编排了"LLM 决策 → 并行调工具 → 工具结果回传 → LLM 汇总回答"的完整闭环，所有流转由图结构驱动而非硬编码 if/else。

### 实战二：构建一个简单的 MCP Server（本地文件检索）

这个 MCP Server 暴露本地文件系统的检索能力，任何兼容 MCP 的客户端（如 Cursor、Claude Desktop）都可以直接调用。

```python
# 安装依赖：pip install mcp

import os
import glob
from mcp.server.fastmcp import FastMCP

mcp_server = FastMCP(
    name="local-file-search",
    version="1.0.0",
)

# ===== Resource：读取指定文件内容 =====
@mcp_server.resource("file://local/{path}")
async def read_local_file(path: str) -> str:
    """读取本地文件内容（Resource：只读，无副作用）"""
    safe_base = os.path.expanduser("~/documents")
    full_path = os.path.normpath(os.path.join(safe_base, path))

    if not full_path.startswith(safe_base):
        raise ValueError("禁止访问指定目录之外的文件")

    if not os.path.exists(full_path):
        return f"文件不存在：{path}"

    with open(full_path, "r", encoding="utf-8") as f:
        content = f.read()

    if len(content) > 10000:
        content = content[:10000] + "\n...(文件过长，已截断)"

    return content

# ===== Tool：搜索文件 =====
@mcp_server.tool()
async def search_files(
    keyword: str,
    file_extension: str = "*.txt",
    max_results: int = 10,
) -> str:
    """
    在本地文档目录中搜索包含关键词的文件。
    返回匹配的文件名和匹配行。

    Args:
        keyword: 要搜索的关键词
        file_extension: 文件扩展名过滤，默认 *.txt
        max_results: 最大返回结果数
    """
    search_dir = os.path.expanduser("~/documents")
    pattern = os.path.join(search_dir, "**", file_extension)
    matches = []

    for filepath in glob.glob(pattern, recursive=True):
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    if keyword.lower() in line.lower():
                        rel_path = os.path.relpath(filepath, search_dir)
                        matches.append(f"{rel_path}:{line_num}: {line.strip()}")
                        if len(matches) >= max_results:
                            break
        except (UnicodeDecodeError, PermissionError):
            continue

        if len(matches) >= max_results:
            break

    if not matches:
        return f"未找到包含 '{keyword}' 的文件"

    header = f"找到 {len(matches)} 条匹配结果：\n"
    return header + "\n".join(matches)

# ===== Tool：列出目录结构 =====
@mcp_server.tool()
async def list_directory(
    subpath: str = "",
    max_depth: int = 2,
) -> str:
    """
    列出本地文档目录的结构。

    Args:
        subpath: 子目录路径（相对于文档根目录）
        max_depth: 最大递归深度
    """
    base_dir = os.path.expanduser("~/documents")
    target = os.path.normpath(os.path.join(base_dir, subpath))

    if not target.startswith(base_dir):
        raise ValueError("禁止访问指定目录之外的路径")

    if not os.path.isdir(target):
        return f"目录不存在：{subpath}"

    lines = []

    def walk(path, prefix, depth):
        if depth > max_depth:
            return
        try:
            entries = sorted(os.listdir(path))
        except PermissionError:
            return

        dirs = [e for e in entries if os.path.isdir(os.path.join(path, e))]
        files = [e for e in entries if os.path.isfile(os.path.join(path, e))]

        for f in files:
            lines.append(f"{prefix}📄 {f}")
        for d in dirs:
            lines.append(f"{prefix}📁 {d}/")
            walk(os.path.join(path, d), prefix + "  ", depth + 1)

    walk(target, "", 0)

    if not lines:
        return "目录为空"

    return "\n".join(lines[:100])

# ===== Prompt：文件分析模板 =====
@mcp_server.prompt()
async def analyze_file(filepath: str) -> str:
    """文件内容分析提示模板"""
    return f"""请分析以下文件的内容，从这些维度进行评估：

1. **文件概要**：文件的主要用途和核心内容
2. **结构质量**：内容组织是否清晰、逻辑是否连贯
3. **关键发现**：文件中最值得注意的信息点
4. **改进建议**：有哪些可以优化的地方

请先使用文件读取工具获取 `{filepath}` 的内容，然后进行分析。"""

# ===== 启动服务器 =====
if __name__ == "__main__":
    print("启动 MCP Server: local-file-search")
    print("传输方式: stdio")
    print("可用能力:")
    print("  Resources: file://local/{path}")
    print("  Tools: search_files, list_directory")
    print("  Prompts: analyze_file")
    mcp_server.run(transport="stdio")
```

### 在 Cursor 中接入 MCP Server

创建或编辑项目根目录下的 `.cursor/mcp.json` 配置文件：

```json
{
  "mcpServers": {
    "local-file-search": {
      "command": "python",
      "args": ["path/to/mcp_file_search_server.py"],
      "env": {}
    }
  }
}
```

配置完成后，Cursor 中的 AI 助手将能够：

- 使用 `search_files` 工具在本地文档中搜索内容
- 使用 `list_directory` 工具查看目录结构
- 通过 `file://local/{path}` 资源直接读取文件
- 使用 `analyze_file` 提示模板对文件进行标准化分析

### 两个实战的架构关系

```
┌─────────────────────────────────────────────────────┐
│               LangGraph Agent（实战一）               │
│                                                     │
│   START → call_llm ←→ execute_tool → END            │
│               ↓                                     │
│         需要访问外部工具？                            │
│               ↓                                     │
│   ┌───────────────────────────────────────┐         │
│   │     MCP Client（协议适配层）           │         │
│   └──────────────┬────────────────────────┘         │
│                  │ JSON-RPC 2.0 (stdio/SSE)         │
└──────────────────┼──────────────────────────────────┘
                   │
┌──────────────────┼──────────────────────────────────┐
│   MCP Server：local-file-search（实战二）             │
│                  │                                   │
│   Resources: file://local/*                          │
│   Tools: search_files, list_directory                │
│   Prompts: analyze_file                              │
└──────────────────────────────────────────────────────┘
```

---

## 验收交付

### 交付一：LangGraph 状态机设计能力验证

绘制一个包含条件分支的完整图结构，证明对有向图流转机制的理解：

```
                    ┌──────────┐
                    │  START   │
                    └────┬─────┘
                         │
                    ┌────▼─────┐
              ┌─────┤ call_llm ├─────┐
              │     └──────────┘     │
              │                      │
        有工具调用               无工具调用
              │                      │
        ┌─────▼──────┐         ┌─────▼─────┐
        │execute_tool│         │    END     │
        └─────┬──────┘         └───────────┘
              │
        重试次数 < 3？
        ┌─────┴──────┐
        │            │
       是            否
        │            │
   ┌────▼─────┐  ┌───▼─────┐
   │ call_llm │  │fallback │──→ END
   └──────────┘  └─────────┘
```

**核心检查清单**：

| 检查项 | 要求 | 状态 |
| --- | --- | --- |
| StateGraph 定义 | 包含明确的 TypedDict 状态类型 | ✅ |
| Node 节点 | 至少 3 个功能节点（LLM/工具/降级） | ✅ |
| Edge 普通边 | 至少 1 条无条件边 | ✅ |
| Conditional Edge | 至少 1 个条件路由函数 | ✅ |
| 死锁防御 | recursion_limit + retry_count | ✅ |
| 可运行 | 代码可直接执行并产出正确结果 | ✅ |

### 交付二：MCP Server 部署验证

**部署清单**：

| 验证项 | 验证方法 | 预期结果 |
| --- | --- | --- |
| Server 启动 | `python mcp_file_search_server.py` | 进程正常运行，无报错 |
| Resources 注册 | Client 调用 `resources/list` | 返回 `file://local/{path}` |
| Tools 注册 | Client 调用 `tools/list` | 返回 `search_files`、`list_directory` |
| Prompts 注册 | Client 调用 `prompts/list` | 返回 `analyze_file` |
| 工具调用 | 调用 `search_files(keyword="测试")` | 返回匹配文件列表 |
| 安全边界 | 尝试访问 `../../etc/passwd` | 被路径校验拦截 |

### 核心知识点自检表

| 知识点 | 一句话检验 | 掌握程度 |
| --- | --- | --- |
| 状态机 vs 硬编码 | 能否说出 3 个以上硬编码的致命缺陷？ | ☐ |
| StateGraph | 能否解释 Annotated + Reducer 的作用？ | ☐ |
| Conditional Edge | 能否手写一个带 3 个分支的路由函数？ | ☐ |
| 死锁防御 | 能否说出至少 3 种防御手段？ | ☐ |
| MCP 三类原语 | 能否区分 Resources 和 Tools 的本质差异？ | ☐ |
| MCP 交互流程 | 能否画出 Client-Server 的完整生命周期？ | ☐ |
| MCP Server 实现 | 能否独立从零搭建一个可用的 MCP Server？ | ☐ |

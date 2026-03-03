# 多 Agent 角色编排

## 学习

**学习目标**

- 通过 CrewAI / MetaGPT 或 LangGraph 高阶多角色网络分封等前沿方案，建立分布式 Agent 团队理念
- 掌握角色画像（System Prompt 预制舱）设计方法论，理解链式 / 星型 / 层级式等协作拓扑的取舍
- 实现产出物溯源机制，让每一份输出都能追溯到具体角色节点

> 从第 10-11 周的单 Agent 工作流分治，到本周的多 Agent 角色编排——核心转变是：不再让一个"全能选手"包揽所有事，而是组建一支"专业团队"，每个成员有明确身份、权限和职责边界。

**实战**

- 对基础通用型业务建立（产品经理 / 研发工程师 / 测试工程师 / 运维工程师）四大明确且系统内唯一的权限画像提示词预制舱
- 搭建四角色协作流水线，完成从需求分析 → 代码实现 → 测试验证 → 部署运维的全链路协作

**验收标准**

- 四个角色 Agent 能在协作流水线中各司其职，产出物可追溯到具体角色节点；任一角色的输出中不得越权包含其他角色的专属职能

---

## 第一部分学习：为什么需要多 Agent 协作

### 单 Agent 的天花板

回忆前几周我们一直在做的事——一个 Agent 配上 Planner/Executor 分离、LangGraph 状态机、各种工具调用。在处理"单一领域、单一角色"的任务时效果不错。但当业务场景需要**多种专业视角**时，单 Agent 就撞墙了。

想象一个真实业务场景：**开发一个用户注册功能**。这件事涉及：

- **产品经理**：定义需求（注册要收集哪些字段？需不需要手机验证？）
- **研发工程师**：写代码实现（API 怎么设计？数据库怎么建表？）
- **测试工程师**：设计测试用例（边界条件？异常场景？安全漏洞？）
- **运维工程师**：部署方案（容器化？监控告警？灰度策略？）

如果让一个 Agent 扮演所有角色会怎样？

```python
# ❌ 单 Agent 全角色模式
def do_everything(requirement: str) -> str:
    prompt = f"""
    你同时是产品经理、研发工程师、测试工程师和运维工程师。
    请针对以下需求，依次完成需求分析、代码实现、测试设计和部署方案。

    需求：{requirement}
    """
    return llm.invoke(prompt)  # 角色混乱、深度不足、职责越界
```

### 生活类比：一人乐队 vs 交响乐团

**单 Agent 全包** = 街头的"一人乐队"——一个人同时敲鼓、弹吉他、吹口琴、踩脚铃。小曲子还行，让他来一段贝多芬交响曲？每个乐器都只能给到 30% 的水平，因为注意力被四分五裂了。

**多 Agent 协作** = 交响乐团——小提琴手只管弦乐声部，定音鼓手只管节奏声部，每个人都在自己的专业领域做到 100%。再由指挥（编排引擎）协调节拍，最终合奏出远超任何独奏的效果。

### 单 Agent vs 多 Agent 协作对比

| 维度 | 单 Agent 全包 | 多 Agent 角色协作 |
| --- | --- | --- |
| 角色专业度 | 每个角色只能给到"及格线"水平 | 每个 Agent 专注一个角色，深度拉满 |
| System Prompt 精度 | 一个 Prompt 塞入多角色指令，互相干扰 | 每个 Agent 独立 Prompt，精准聚焦 |
| 上下文利用率 | Token 预算被多角色稀释 | 每个角色独占完整上下文窗口 |
| 输出可追溯性 | 无法区分哪段输出来自哪个"角色" | 每份产出物打标到具体 Agent |
| 权限隔离 | 无法限制"测试不能改代码" | 角色权限边界清晰可审计 |
| 可扩展性 | 新增角色要改整个 Prompt | 新增一个 Agent 即可，不影响现有角色 |
| 并行能力 | 串行处理所有角色任务 | 无依赖角色可并行执行 |

### 什么时候该上多 Agent？

并不是所有场景都需要多 Agent——简单任务用多 Agent 反而增加编排开销。判断标准：

| 信号 | 建议 |
| --- | --- |
| 任务只涉及单一视角（如"翻译这段话"） | 单 Agent 足够 |
| 需要 2 种以上专业视角（如"写代码 + 写测试"） | 考虑多 Agent |
| 不同子任务需要不同权限（如"只读数据库 vs 可写数据库"） | 强烈建议多 Agent |
| 产出物需要交叉审核（如"研发写代码、测试写用例、互相校验"） | 必须多 Agent |
| 团队协作流程本身就是业务价值（如模拟软件研发流水线） | 多 Agent 是核心 |

---

## 第二部分学习：CrewAI vs MetaGPT vs LangGraph 多角色方案对比

### 三大框架概览

当前主流的多 Agent 协作框架有三个代表：**CrewAI**（轻量编排）、**MetaGPT**（SOP 驱动）、**LangGraph**（图状态机）。它们解决同一个问题——多角色协作，但设计哲学差异很大。

### 生活类比：三种团队管理模式

| 框架 | 类比 | 管理风格 |
| --- | --- | --- |
| CrewAI | **创业公司** —— 扁平化，角色灵活，快速拉起一个小团队 | 给每个人一个头衔和目标，大家自主协作 |
| MetaGPT | **流水线工厂** —— 严格 SOP，每道工序标准化 | 按照预定义的标准操作流程，上一道工序的产出是下一道的输入 |
| LangGraph | **军事参谋部** —— 状态机驱动，条件分支精确控制 | 每个决策点都有明确的路由条件和状态流转规则 |

### 框架详细对比

| 维度 | CrewAI | MetaGPT | LangGraph |
| --- | --- | --- | --- |
| 核心抽象 | Agent + Task + Crew | Role + Action + Environment | Node + Edge + StateGraph |
| 角色定义方式 | `role`/`goal`/`backstory` 三字段 | 继承 `Role` 类，定义 `Action` 列表 | 自定义函数节点 + System Prompt |
| 协作模式 | `sequential` / `hierarchical` | SOP 流程链，角色按顺序接力 | 有向图，Conditional Edge 精确路由 |
| 上手难度 | ⭐⭐（最简单） | ⭐⭐⭐（中等） | ⭐⭐⭐⭐（较复杂） |
| 灵活性 | 中等（内置模式覆盖常见场景） | 低（强 SOP 约束） | 极高（图结构可表达任意拓扑） |
| 状态管理 | 隐式（通过 Task 输出传递） | 环境变量 + 消息队列 | 显式 TypedDict 全局状态 |
| 产出溯源 | Task 级别（知道哪个 Agent 做了哪个 Task） | Action 级别（精确到每个动作） | Node 级别（每个节点有明确输出） |
| 社区生态 | 活跃，插件丰富 | 学术背景强，论文级别 | LangChain 生态深度整合 |
| 适用场景 | 快速原型、中小规模协作 | 标准化流程、模拟真实团队 | 复杂条件路由、精细状态控制 |

### 选型决策树

```
你的多 Agent 场景是什么？
        │
        ├── 快速验证想法，角色 ≤5 个
        │       → CrewAI（15 分钟搞定原型）
        │
        ├── 模拟真实团队流程，强调标准化产出
        │       → MetaGPT（SOP 驱动，产出规范）
        │
        └── 复杂条件分支，需要精细状态控制
                → LangGraph（图拓扑 + 条件路由）
```

### 核心代码风格速览

**CrewAI 风格**：

```python
from crewai import Agent, Task, Crew

pm_agent = Agent(
    role="产品经理",
    goal="输出完整的需求文档",
    backstory="你是一位有10年经验的产品经理，擅长用户需求分析"
)

pm_task = Task(
    description="针对'{requirement}'撰写需求分析文档",
    agent=pm_agent,
    expected_output="结构化需求文档，包含功能列表、优先级和验收标准"
)

crew = Crew(agents=[pm_agent, dev_agent, ...], tasks=[pm_task, dev_task, ...])
result = crew.kickoff()
```

**MetaGPT 风格**：

```python
from metagpt.roles import Role
from metagpt.actions import Action

class AnalyzeRequirement(Action):
    name: str = "需求分析"
    async def run(self, requirement: str) -> str:
        return await self._aask(f"请分析以下需求：{requirement}")

class ProductManager(Role):
    name: str = "产品经理"
    profile: str = "Product Manager"
    goal: str = "输出高质量需求文档"
    actions: list = [AnalyzeRequirement]
```

**LangGraph 风格**（本周实战采用）：

```python
from langgraph.graph import StateGraph

def product_manager_node(state: TeamState) -> dict:
    """产品经理节点：需求分析"""
    response = llm.invoke(pm_system_prompt + state["requirement"])
    return {"pm_output": response, "trace": [("产品经理", response)]}

graph = StateGraph(TeamState)
graph.add_node("product_manager", product_manager_node)
graph.add_node("developer", developer_node)
graph.add_edge("product_manager", "developer")
```

---

## 第三部分学习：角色权限画像设计（System Prompt 预制舱）

### 什么是 System Prompt 预制舱？

"预制舱"是一个形象的比喻——就像国际空间站的各个舱段，每个舱段有独立的功能和权限边界。**System Prompt 预制舱**就是为每个角色 Agent 预先定义好的"身份证 + 权限卡 + 操作手册"。

### 生活类比：医院科室的权限体系

想象一家医院——

- **内科医生**：可以开处方、查看病历，但**不能做手术**
- **外科医生**：可以做手术、开处方，但**不能出放射报告**
- **放射科技师**：可以操作 CT/MRI、出影像报告，但**不能开处方**
- **药剂师**：可以审核处方、配药，但**不能修改诊断**

每个科室的人都有明确的"能做什么"和"不能做什么"。System Prompt 预制舱就是 Agent 世界的"科室权限体系"。

### 预制舱设计四要素

| 要素 | 说明 | 类比 |
| --- | --- | --- |
| **身份（Identity）** | 你是谁，你的专业背景 | 工牌上的姓名和科室 |
| **目标（Goal）** | 你的核心产出物是什么 | 岗位 KPI |
| **权限（Permission）** | 你能做什么、不能做什么 | 门禁卡的权限范围 |
| **交互规范（Protocol）** | 你的输出格式、与其他角色的对接方式 | 病历书写规范、交接班模板 |

### 四角色预制舱详细设计

#### 产品经理（Product Manager）预制舱

```python
PM_SYSTEM_PROMPT = """
## 身份
你是一位资深产品经理，拥有 10 年互联网产品设计经验。你擅长用户需求分析、
功能优先级排序和验收标准制定。

## 目标
针对给定的业务需求，输出一份结构化的需求分析文档。

## 权限边界
✅ 你可以做：
- 分析和拆解用户需求
- 定义功能列表及优先级（P0/P1/P2）
- 制定验收标准（Acceptance Criteria）
- 绘制用户流程图（文字描述版）

❌ 你不能做：
- 编写任何代码或伪代码
- 指定技术栈或架构方案
- 设计测试用例
- 制定部署策略

## 输出格式
```markdown
### 需求分析文档
**需求名称**：xxx
**需求背景**：xxx
**功能列表**：
| 功能 | 优先级 | 描述 | 验收标准 |
| --- | --- | --- | --- |
| xxx | P0 | xxx | xxx |
**用户流程**：xxx
```

## 交互规范

- 你的输出将传递给【研发工程师】作为开发依据
- 不要在文档中包含任何技术实现建议
- 所有功能必须有明确的验收标准
"""

```

#### 研发工程师（Developer）预制舱

```python
DEV_SYSTEM_PROMPT = """
## 身份
你是一位高级后端研发工程师，精通 Python / FastAPI / PostgreSQL。
你只根据产品经理提供的需求文档进行开发。

## 目标
根据需求文档，输出可运行的代码实现方案。

## 权限边界
✅ 你可以做：
- 设计 API 接口和数据模型
- 编写 Python 实现代码
- 定义数据库表结构
- 编写代码注释和接口文档

❌ 你不能做：
- 修改或质疑需求文档的功能定义
- 编写测试用例（这是测试工程师的职责）
- 制定部署方案（这是运维工程师的职责）
- 跳过需求文档中的任何 P0 功能

## 输出格式
```markdown
### 技术方案
**技术栈**：xxx
**数据模型**：
| 字段 | 类型 | 说明 |
| --- | --- | --- |
| xxx | xxx | xxx |
**API 设计**：
| 接口 | 方法 | 路径 | 说明 |
| --- | --- | --- | --- |
| xxx | POST | /api/xxx | xxx |
**核心代码**：
（Python 代码块）
```

## 交互规范

- 你的输入来自【产品经理】的需求文档
- 你的输出将传递给【测试工程师】作为测试依据
- 代码必须包含类型注解和必要注释
"""

```

#### 测试工程师（QA Engineer）预制舱

```python
QA_SYSTEM_PROMPT = """
## 身份
你是一位资深测试工程师，擅长功能测试、边界测试和安全测试。
你根据需求文档和技术方案设计全面的测试策略。

## 目标
根据需求文档和技术方案，输出完整的测试用例集。

## 权限边界
✅ 你可以做：
- 设计功能测试用例（正常流程 + 异常流程）
- 设计边界条件测试
- 设计安全测试用例（SQL 注入、XSS 等）
- 评估测试覆盖率

❌ 你不能做：
- 修改需求文档或技术方案
- 编写业务代码（只写测试代码）
- 制定部署策略
- 跳过任何 P0 功能的测试覆盖

## 输出格式
```markdown
### 测试用例集
**测试覆盖概览**：
| 类别 | 用例数 | 覆盖功能 |
| --- | --- | --- |
| 正常流程 | xxx | xxx |
| 异常流程 | xxx | xxx |
| 边界条件 | xxx | xxx |
| 安全测试 | xxx | xxx |
**详细用例**：
| 编号 | 类别 | 场景 | 输入 | 预期结果 |
| --- | --- | --- | --- | --- |
| TC-001 | 正常 | xxx | xxx | xxx |
```

## 交互规范

- 你的输入来自【产品经理】的需求文档 + 【研发工程师】的技术方案
- 你的输出将传递给【运维工程师】作为上线前检查依据
- 每个 P0 功能至少对应 3 条测试用例（正常 + 异常 + 边界）
"""

```

#### 运维工程师（DevOps Engineer）预制舱

```python
OPS_SYSTEM_PROMPT = """
## 身份
你是一位资深运维工程师，精通 Docker / Kubernetes / CI-CD 和监控告警体系。
你负责保障系统的可靠部署和稳定运行。

## 目标
根据技术方案和测试报告，输出完整的部署和运维方案。

## 权限边界
✅ 你可以做：
- 设计容器化部署方案（Dockerfile / docker-compose）
- 制定 CI/CD 流水线配置
- 设计监控告警策略
- 制定灰度发布和回滚方案

❌ 你不能做：
- 修改业务代码逻辑
- 修改需求文档或测试用例
- 直接操作数据库数据
- 跳过测试报告中标记的阻塞性问题

## 输出格式
```markdown
### 部署运维方案
**部署架构**：xxx
**容器配置**：
（Dockerfile / docker-compose 配置）
**CI/CD 流水线**：
| 阶段 | 动作 | 触发条件 |
| --- | --- | --- |
| xxx | xxx | xxx |
**监控告警**：
| 指标 | 阈值 | 告警方式 |
| --- | --- | --- |
| xxx | xxx | xxx |
**回滚方案**：xxx
```

## 交互规范

- 你的输入来自【研发工程师】的技术方案 + 【测试工程师】的测试报告
- 你是流水线最后一环，输出即最终交付物
- 必须包含回滚方案——没有回滚方案的部署不允许上线
"""

```

### 预制舱设计原则总结

| 原则 | 说明 | 违反后果 |
| --- | --- | --- |
| **单一职责** | 每个预制舱只定义一种角色 | 角色混乱，输出质量下降 |
| **权限最小化** | 只授予完成本职工作必需的权限 | 角色越界，产出不可信 |
| **格式标准化** | 统一输出格式，便于下游解析 | 角色间对接失败 |
| **边界显式化** | 明确列出"不能做什么" | LLM 倾向于"帮忙做更多"，导致越权 |

---

## 第四部分学习：协作拓扑——链式 / 星型 / 层级式

### 为什么拓扑很重要？

有了角色之后，下一个关键问题是：**这些角色怎么连接在一起？** 谁先做、谁后做、谁跟谁并行、谁监督谁——这就是"协作拓扑"。

### 生活类比：城市交通规划

拓扑设计就像城市交通规划——

- **链式**：像一条直通公路，从 A 到 B 到 C 到 D，简单但只能走一条路
- **星型**：像机场航线，所有飞机都经过中心枢纽转机
- **层级式**：像军队指挥链，司令 → 军长 → 师长 → 团长

没有"最好的"拓扑，只有"最适合当前场景的"拓扑。

### 三种核心拓扑对比

```

链式拓扑（Sequential）：

  产品经理 ──→ 研发工程师 ──→ 测试工程师 ──→ 运维工程师

星型拓扑（Hub-Spoke）：

                   产品经理
                      ↑
                      │
  运维工程师 ←── 协调中心 ──→ 研发工程师
                      │
                      ↓
                   测试工程师

层级式拓扑（Hierarchical）：

                  ┌─────────┐
                  │ 项目经理 │  ← 顶层决策者
                  │(Manager)│
                  └────┬────┘
            ┌──────────┼──────────┐
            ↓          ↓          ↓
       ┌────────┐ ┌────────┐ ┌────────┐
       │产品经理│ │研发组长│ │运维组长│  ← 中层管理
       └───┬────┘ └───┬────┘ └────────┘
           │     ┌────┴────┐
           │     ↓         ↓
           │  ┌──────┐ ┌──────┐
           │  │前端  │ │后端  │  ← 执行层
           │  └──────┘ └──────┘
           ↓
       需求文档

```

| 维度 | 链式 | 星型 | 层级式 |
| --- | --- | --- | --- |
| 复杂度 | ⭐（最简单） | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| 并行能力 | 无（纯串行） | 高（中心分发后可并行） | 中等（同层可并行） |
| 容错性 | 低（一环断全链断） | 中（中心节点是单点故障） | 高（子树故障不影响其他分支） |
| 适用场景 | 流程固定、前后依赖强 | 多角色需要共享上下文 | 大型团队、需要审批层级 |
| 编排复杂度 | 最低 | 中等 | 最高 |
| 本周实战选择 | ✅（四角色流水线） | — | — |

### 本周为什么选链式？

四角色协作流水线是一个天然的**前后依赖链**：

1. 产品经理的需求文档 → 是研发工程师的输入
2. 研发工程师的技术方案 → 是测试工程师的输入
3. 测试工程师的测试报告 → 是运维工程师的输入

每个阶段的产出是下一个阶段的前提，不存在可并行的独立分支——链式拓扑是最自然、最简洁的选择。

---

## 第五部分学习：产出物溯源机制

### 为什么需要溯源？

多 Agent 协作的一大风险是**产出物归属模糊**——当最终输出有问题时，你不知道是哪个 Agent 出的错。这就像工厂产品出了质量问题，如果每道工序没有盖章记录，你根本无法追查责任环节。

### 生活类比：食品安全追溯码

超市里每块肉都有追溯码——扫一下就知道：哪个牧场养的 → 哪个屠宰场加工的 → 哪辆冷链车运的 → 哪天到的店。多 Agent 产出物溯源的原理完全一样。

### 溯源机制设计

```

用户需求
    │
    ↓
┌─────────────────────────────────────────────────────────┐
│  产出物追踪链（Trace Chain）                              │
│                                                         │
│  Step 1: [产品经理] → 需求文档                            │
│    ├─ agent_id: "pm_agent"                              │
│    ├─ timestamp: "2026-03-02T10:00:00"                  │
│    ├─ input_hash: "abc123"                              │
│    └─ output_hash: "def456"                             │
│                                                         │
│  Step 2: [研发工程师] → 技术方案                           │
│    ├─ agent_id: "dev_agent"                             │
│    ├─ timestamp: "2026-03-02T10:01:00"                  │
│    ├─ input_hash: "def456" (来自 Step 1 的 output)       │
│    └─ output_hash: "ghi789"                             │
│                                                         │
│  Step 3: [测试工程师] → 测试用例集                         │
│    ├─ agent_id: "qa_agent"                              │
│    ├─ timestamp: "2026-03-02T10:02:00"                  │
│    ├─ input_hash: "def456+ghi789"                       │
│    └─ output_hash: "jkl012"                             │
│                                                         │
│  Step 4: [运维工程师] → 部署方案                           │
│    ├─ agent_id: "ops_agent"                             │
│    ├─ timestamp: "2026-03-02T10:03:00"                  │
│    ├─ input_hash: "ghi789+jkl012"                       │
│    └─ output_hash: "mno345"                             │
└─────────────────────────────────────────────────────────┘

```

### 溯源数据结构

```python
from dataclasses import dataclass, field
from datetime import datetime
import hashlib

@dataclass
class TraceRecord:
    """单条溯源记录"""
    agent_id: str
    agent_role: str
    timestamp: str
    input_text: str
    output_text: str
    input_hash: str = ""
    output_hash: str = ""

    def __post_init__(self):
        self.input_hash = hashlib.md5(self.input_text.encode()).hexdigest()[:8]
        self.output_hash = hashlib.md5(self.output_text.encode()).hexdigest()[:8]

@dataclass
class TraceChain:
    """完整溯源链"""
    records: list[TraceRecord] = field(default_factory=list)

    def add(self, agent_id: str, agent_role: str, input_text: str, output_text: str):
        record = TraceRecord(
            agent_id=agent_id,
            agent_role=agent_role,
            timestamp=datetime.now().isoformat(),
            input_text=input_text,
            output_text=output_text
        )
        self.records.append(record)
        return record

    def print_chain(self):
        print("=" * 60)
        print("产出物溯源链")
        print("=" * 60)
        for i, r in enumerate(self.records, 1):
            print(f"\nStep {i}: [{r.agent_role}] (agent_id: {r.agent_id})")
            print(f"  时间戳: {r.timestamp}")
            print(f"  输入哈希: {r.input_hash}")
            print(f"  输出哈希: {r.output_hash}")
            print(f"  输出摘要: {r.output_text[:80]}...")

    def locate_issue(self, keyword: str) -> list[TraceRecord]:
        """根据关键词定位问题出自哪个角色"""
        return [r for r in self.records if keyword in r.output_text]
```

### 溯源的三重价值

| 价值 | 说明 | 举例 |
| --- | --- | --- |
| **问题定位** | 最终产出有误时，快速定位是哪个角色出的错 | 部署方案缺少回滚策略 → 追溯到运维 Agent |
| **质量审计** | 检查每个角色是否在权限范围内行事 | 发现研发 Agent 的输出中包含测试用例 → 越权告警 |
| **持续改进** | 统计每个角色的输出质量，针对性优化 Prompt | 产品经理 Agent 的验收标准缺失率 30% → 优化 PM 预制舱 |

---

## 第六部分学习：实战代码——四角色协作流水线

### 整体架构

```
用户输入业务需求
        │
        ↓
  ┌──────────────┐
  │   产品经理     │  → 输出：需求分析文档
  │ (PM Agent)    │
  └──────┬───────┘
         │ 需求文档传递
         ↓
  ┌──────────────┐
  │  研发工程师    │  → 输出：技术方案 + 代码
  │ (Dev Agent)   │
  └──────┬───────┘
         │ 技术方案传递
         ↓
  ┌──────────────┐
  │  测试工程师    │  → 输出：测试用例集
  │ (QA Agent)    │
  └──────┬───────┘
         │ 测试报告传递
         ↓
  ┌──────────────┐
  │  运维工程师    │  → 输出：部署运维方案
  │ (Ops Agent)   │
  └──────────────┘
         │
         ↓
  最终交付物 + 溯源链报告
```

### 完整可运行代码

```python
"""
四角色协作流水线：产品经理 → 研发工程师 → 测试工程师 → 运维工程师
基于 LangGraph 状态机编排，支持产出物溯源
"""

import os
import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import TypedDict, Annotated
from operator import add

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END


# ============================================================
# 一、角色 System Prompt 预制舱
# ============================================================

PM_SYSTEM_PROMPT = """你是一位资深产品经理。请根据业务需求输出结构化的需求分析文档。

输出格式要求：
### 需求分析文档
**需求名称**：xxx
**需求背景**：xxx（2-3句说明为什么要做这个功能）
**功能列表**：
| 功能 | 优先级 | 描述 | 验收标准 |
| --- | --- | --- | --- |
（至少列出3个功能点，P0/P1/P2分级）
**用户流程**：xxx（描述核心用户路径）

注意：你只负责需求分析，不要涉及技术实现、测试方案或部署策略。"""

DEV_SYSTEM_PROMPT = """你是一位高级Python后端工程师。请根据产品经理的需求文档输出技术方案和核心代码。

输出格式要求：
### 技术方案
**技术栈**：xxx
**数据模型**：
| 字段 | 类型 | 说明 | 约束 |
| --- | --- | --- | --- |
（列出核心数据表的字段设计）
**API 设计**：
| 接口名称 | 方法 | 路径 | 说明 |
| --- | --- | --- | --- |
（列出核心API接口）
**核心代码**：
（提供可运行的Python代码片段）

注意：你只负责技术实现，不要修改需求、编写测试用例或制定部署方案。"""

QA_SYSTEM_PROMPT = """你是一位资深测试工程师。请根据需求文档和技术方案设计完整的测试用例。

输出格式要求：
### 测试用例集
**覆盖概览**：
| 类别 | 用例数 | 覆盖功能 |
| --- | --- | --- |
（列出各类别测试用例数量）
**详细用例**：
| 编号 | 类别 | 场景 | 输入 | 预期结果 |
| --- | --- | --- | --- | --- |
（P0功能每个至少3条：正常、异常、边界）

注意：你只负责测试设计，不要修改需求或代码，不要涉及部署方案。"""

OPS_SYSTEM_PROMPT = """你是一位资深运维工程师。请根据技术方案和测试报告输出部署运维方案。

输出格式要求：
### 部署运维方案
**部署架构**：xxx
**容器配置**：（Dockerfile 或 docker-compose 核心配置）
**CI/CD 流水线**：
| 阶段 | 动作 | 触发条件 |
| --- | --- | --- |
（列出至少3个流水线阶段）
**监控告警**：
| 指标 | 阈值 | 告警方式 |
| --- | --- | --- |
（列出核心监控项）
**回滚方案**：xxx（必须包含！）

注意：你只负责部署运维，不要修改业务代码、需求或测试用例。"""


# ============================================================
# 二、溯源数据结构
# ============================================================

@dataclass
class TraceRecord:
    """单条产出物溯源记录"""
    agent_role: str
    timestamp: str
    input_hash: str
    output_hash: str
    output_preview: str

    @staticmethod
    def create(agent_role: str, input_text: str, output_text: str) -> "TraceRecord":
        return TraceRecord(
            agent_role=agent_role,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            input_hash=hashlib.md5(input_text.encode()).hexdigest()[:8],
            output_hash=hashlib.md5(output_text.encode()).hexdigest()[:8],
            output_preview=output_text[:100].replace("\n", " ")
        )


# ============================================================
# 三、LangGraph 状态定义
# ============================================================

class TeamState(TypedDict):
    requirement: str
    pm_output: str
    dev_output: str
    qa_output: str
    ops_output: str
    trace: Annotated[list[dict], add]


# ============================================================
# 四、节点函数（每个角色一个节点）
# ============================================================

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0.3,
    api_key=os.getenv("OPENAI_API_KEY", "your-api-key"),
    base_url=os.getenv("OPENAI_BASE_URL", None),
)


def pm_node(state: TeamState) -> dict:
    """产品经理节点：需求分析"""
    requirement = state["requirement"]
    messages = [
        {"role": "system", "content": PM_SYSTEM_PROMPT},
        {"role": "user", "content": f"业务需求：{requirement}"},
    ]
    response = llm.invoke(messages).content

    trace = TraceRecord.create("产品经理", requirement, response)
    print(f"\n{'='*60}")
    print(f"[产品经理] 完成需求分析  (output_hash: {trace.output_hash})")
    print(f"{'='*60}")

    return {
        "pm_output": response,
        "trace": [{"role": "产品经理", "hash": trace.output_hash,
                    "time": trace.timestamp, "preview": trace.output_preview}],
    }


def dev_node(state: TeamState) -> dict:
    """研发工程师节点：技术方案 + 代码实现"""
    pm_output = state["pm_output"]
    messages = [
        {"role": "system", "content": DEV_SYSTEM_PROMPT},
        {"role": "user", "content": f"以下是产品经理的需求文档，请输出技术方案：\n\n{pm_output}"},
    ]
    response = llm.invoke(messages).content

    trace = TraceRecord.create("研发工程师", pm_output, response)
    print(f"\n{'='*60}")
    print(f"[研发工程师] 完成技术方案  (output_hash: {trace.output_hash})")
    print(f"{'='*60}")

    return {
        "dev_output": response,
        "trace": [{"role": "研发工程师", "hash": trace.output_hash,
                    "time": trace.timestamp, "preview": trace.output_preview}],
    }


def qa_node(state: TeamState) -> dict:
    """测试工程师节点：测试用例设计"""
    context = f"需求文档：\n{state['pm_output']}\n\n技术方案：\n{state['dev_output']}"
    messages = [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": f"请根据以下信息设计测试用例：\n\n{context}"},
    ]
    response = llm.invoke(messages).content

    trace = TraceRecord.create("测试工程师", context, response)
    print(f"\n{'='*60}")
    print(f"[测试工程师] 完成测试设计  (output_hash: {trace.output_hash})")
    print(f"{'='*60}")

    return {
        "qa_output": response,
        "trace": [{"role": "测试工程师", "hash": trace.output_hash,
                    "time": trace.timestamp, "preview": trace.output_preview}],
    }


def ops_node(state: TeamState) -> dict:
    """运维工程师节点：部署运维方案"""
    context = f"技术方案：\n{state['dev_output']}\n\n测试报告：\n{state['qa_output']}"
    messages = [
        {"role": "system", "content": OPS_SYSTEM_PROMPT},
        {"role": "user", "content": f"请根据以下信息输出部署运维方案：\n\n{context}"},
    ]
    response = llm.invoke(messages).content

    trace = TraceRecord.create("运维工程师", context, response)
    print(f"\n{'='*60}")
    print(f"[运维工程师] 完成部署方案  (output_hash: {trace.output_hash})")
    print(f"{'='*60}")

    return {
        "ops_output": response,
        "trace": [{"role": "运维工程师", "hash": trace.output_hash,
                    "time": trace.timestamp, "preview": trace.output_preview}],
    }


# ============================================================
# 五、构建 LangGraph 流水线
# ============================================================

def build_pipeline():
    """构建四角色协作流水线"""
    graph = StateGraph(TeamState)

    graph.add_node("product_manager", pm_node)
    graph.add_node("developer", dev_node)
    graph.add_node("qa_engineer", qa_node)
    graph.add_node("ops_engineer", ops_node)

    graph.add_edge(START, "product_manager")
    graph.add_edge("product_manager", "developer")
    graph.add_edge("developer", "qa_engineer")
    graph.add_edge("qa_engineer", "ops_engineer")
    graph.add_edge("ops_engineer", END)

    return graph.compile()


# ============================================================
# 六、溯源报告生成
# ============================================================

def print_trace_report(trace_records: list[dict]):
    """打印产出物溯源报告"""
    print("\n")
    print("=" * 60)
    print("产出物溯源报告")
    print("=" * 60)
    print(f"{'角色':<10} {'输出哈希':<12} {'时间':<22} {'输出摘要'}")
    print("-" * 60)
    for record in trace_records:
        print(f"{record['role']:<10} {record['hash']:<12} {record['time']:<22} {record['preview'][:30]}...")
    print("=" * 60)


def print_deliverables(state: dict):
    """打印各角色交付物"""
    roles = [
        ("产品经理", "pm_output"),
        ("研发工程师", "dev_output"),
        ("测试工程师", "qa_output"),
        ("运维工程师", "ops_output"),
    ]
    for role_name, key in roles:
        print(f"\n{'#' * 60}")
        print(f"# {role_name} 交付物")
        print(f"{'#' * 60}")
        print(state[key])


# ============================================================
# 七、主函数
# ============================================================

def main():
    requirement = "开发一个用户注册功能，支持邮箱注册和手机号注册，需要验证码验证，注册成功后自动登录"

    print("=" * 60)
    print("四角色协作流水线启动")
    print(f"需求：{requirement}")
    print("=" * 60)

    pipeline = build_pipeline()

    result = pipeline.invoke({
        "requirement": requirement,
        "pm_output": "",
        "dev_output": "",
        "qa_output": "",
        "ops_output": "",
        "trace": [],
    })

    print_deliverables(result)
    print_trace_report(result["trace"])

    # 验证：检查角色权限边界
    print("\n" + "=" * 60)
    print("权限边界校验")
    print("=" * 60)
    checks = [
        ("产品经理不含代码", "def " not in result["pm_output"] and "import " not in result["pm_output"]),
        ("研发工程师不含测试用例编号", "TC-" not in result["dev_output"]),
        ("测试工程师不含Dockerfile", "Dockerfile" not in result["qa_output"]),
        ("运维工程师不含需求优先级", "P0" not in result["ops_output"] or "P1" not in result["ops_output"]),
    ]
    for desc, passed in checks:
        status = "✅ PASS" if passed else "⚠️  WARN"
        print(f"  {status} | {desc}")


if __name__ == "__main__":
    main()
```

### 代码关键设计点

| 设计点 | 实现方式 | 为什么这么做 |
| --- | --- | --- |
| 角色隔离 | 每个节点独立的 System Prompt | 防止角色越权，保证输出专业度 |
| 状态传递 | TypedDict 显式定义每个角色的输出字段 | 类型安全，下游节点可精确获取上游产出 |
| 溯源机制 | `Annotated[list, add]` 累加 trace 记录 | 每个节点追加自己的记录，最终得到完整链 |
| 哈希标记 | MD5 前 8 位作为产出指纹 | 轻量级校验，可快速验证数据完整性 |
| 权限校验 | 最终输出中检查越权关键词 | 自动化审计，替代人工逐份检查 |

### 运行效果示例

```
============================================================
四角色协作流水线启动
需求：开发一个用户注册功能，支持邮箱注册和手机号注册，需要验证码验证，注册成功后自动登录
============================================================

============================================================
[产品经理] 完成需求分析  (output_hash: a3f8c2d1)
============================================================

============================================================
[研发工程师] 完成技术方案  (output_hash: 7b2e9f04)
============================================================

============================================================
[测试工程师] 完成测试设计  (output_hash: e5d1a8c3)
============================================================

============================================================
[运维工程师] 完成部署方案  (output_hash: 9c4f6b27)
============================================================

============================================================
产出物溯源报告
============================================================
角色        输出哈希      时间                    输出摘要
------------------------------------------------------------
产品经理    a3f8c2d1    2026-03-02 10:00:01     ### 需求分析文档 **需求名称**：用户注册...
研发工程师  7b2e9f04    2026-03-02 10:00:15     ### 技术方案 **技术栈**：Python + Fas...
测试工程师  e5d1a8c3    2026-03-02 10:00:28     ### 测试用例集 **覆盖概览**：| 类别 |...
运维工程师  9c4f6b27    2026-03-02 10:00:42     ### 部署运维方案 **部署架构**：单节点 D...
============================================================

============================================================
权限边界校验
============================================================
  ✅ PASS | 产品经理不含代码
  ✅ PASS | 研发工程师不含测试用例编号
  ✅ PASS | 测试工程师不含Dockerfile
  ✅ PASS | 运维工程师不含需求优先级
```

---

## 验收交付

### 交付一：四角色 System Prompt 预制舱

**预制舱完整性检查**：

| 角色 | 身份定义 | 目标定义 | 权限边界（✅/❌） | 输出格式 | 交互规范 |
| --- | --- | --- | --- | --- | --- |
| 产品经理 | ✅ 10年经验产品经理 | ✅ 需求分析文档 | ✅ 4项可做 / 4项禁止 | ✅ 表格化 | ✅ 下游对接研发 |
| 研发工程师 | ✅ 高级Python工程师 | ✅ 技术方案+代码 | ✅ 4项可做 / 4项禁止 | ✅ 表格+代码块 | ✅ 上游产品/下游测试 |
| 测试工程师 | ✅ 资深测试工程师 | ✅ 测试用例集 | ✅ 4项可做 / 4项禁止 | ✅ 表格化 | ✅ 上游产品+研发/下游运维 |
| 运维工程师 | ✅ 资深运维工程师 | ✅ 部署运维方案 | ✅ 4项可做 / 4项禁止 | ✅ 表格+配置块 | ✅ 上游研发+测试/最终交付 |

### 交付二：四角色协作流水线运行验证

**流水线核心检查清单**：

| 检查项 | 要求 | 状态 |
| --- | --- | --- |
| LangGraph 图定义 | 4 个节点 + 4 条边（含 START/END） | ✅ |
| 链式拓扑 | 严格串行：PM → Dev → QA → Ops | ✅ |
| 状态隔离 | 每个角色写入独立字段，不覆盖他人产出 | ✅ |
| 溯源链完整 | 4 条 trace 记录，含哈希 + 时间戳 | ✅ |
| 权限校验 | 自动化越权检测，至少 4 项检查 | ✅ |
| 代码可运行 | 安装依赖后可直接执行 | ✅ |

### 交付三：产出物溯源能力验证

**溯源验证清单**：

| 验证项 | 验证方法 | 预期结果 |
| --- | --- | --- |
| 溯源链完整性 | 检查 trace 列表长度 == 4 | 四条记录，角色不重复 |
| 哈希可校验 | 对产出物重新计算 MD5 前 8 位 | 与 trace 记录中的 output_hash 一致 |
| 时间线有序 | 检查四条记录的 timestamp | 严格递增 |
| 问题定位 | 在最终输出中搜索关键词 | 能定位到具体角色的 trace 记录 |

**溯源报告参考输出**：

```
============================================================
产出物溯源报告
============================================================
角色        输出哈希      时间                    输出摘要
------------------------------------------------------------
产品经理    a3f8c2d1    2026-03-02 10:00:01     ### 需求分析文档 **需求名称**：用户注册...
研发工程师  7b2e9f04    2026-03-02 10:00:15     ### 技术方案 **技术栈**：Python + Fas...
测试工程师  e5d1a8c3    2026-03-02 10:00:28     ### 测试用例集 **覆盖概览**：| 类别 |...
运维工程师  9c4f6b27    2026-03-02 10:00:42     ### 部署运维方案 **部署架构**：单节点 D...
============================================================

结论：四角色产出物全链路可追溯，溯源链完整 ✅
```

### 核心知识点自检表

| 知识点 | 一句话检验 | 掌握程度 |
| --- | --- | --- |
| 单 Agent 天花板 | 能否列出单 Agent 处理多角色任务的 3 个以上致命缺陷？ | ☐ |
| 框架选型 | 能否说出 CrewAI / MetaGPT / LangGraph 各自的核心抽象和适用场景？ | ☐ |
| 预制舱四要素 | 能否手写一个包含身份/目标/权限/规范的 System Prompt？ | ☐ |
| 权限最小化 | 能否解释为什么"❌ 不能做"比"✅ 能做"更重要？ | ☐ |
| 三种协作拓扑 | 能否画出链式/星型/层级式的拓扑图并说出各自适用场景？ | ☐ |
| 溯源机制 | 能否解释 trace chain 中哈希值的作用和校验逻辑？ | ☐ |
| LangGraph 编排 | 能否用 StateGraph 搭建一个至少 3 节点的串行流水线？ | ☐ |
| 越权检测 | 能否设计自动化规则检测某个角色的输出是否包含越权内容？ | ☐ |
| Browser Agent | 能否说清 GUI Agent 的感知-行动循环原理？ | ☐ |
| Computer Use 适用场景 | 能否列出 3 个 Browser Agent 适合替代 RPA 的场景？ | ☐ |

---

## 第六部分学习：Computer Use / Browser Agent 新范式

### 什么是 Computer Use？

传统 Agent 通过**调用 API** 与外部世界交互——搜索 API、邮件 API、日历 API。但现实世界中，大量系统没有 API：老系统只有 Web 界面、政府网站只能手动填表、企业内网应用无法对外开放 API。

**Computer Use / Browser Agent** 解决了这个问题——Agent 不调用 API，而是**直接操作 GUI 界面**，就像一个人类操作员坐在电脑前：看屏幕截图 → 理解界面 → 移动鼠标 → 点击按钮 → 填写表单 → 截图确认。

### 三大代表性实现

| 系统 | 开发商 | 发布时间 | 核心特征 |
| --- | --- | --- | --- |
| **Claude Computer Use** | Anthropic | 2024.10 | 截图 → VLM 理解 → 鼠标/键盘操作 |
| **OpenAI CUA（Computer Using Agent）** | OpenAI | 2025.01 | 专用模型 o1/o3 优化，内置截图分析 |
| **OpenAI Operator** | OpenAI | 2025.01 | 基于 CUA 的消费级 Web 自动化产品 |
| **Playwright + VLM** | 社区方案 | 持续演进 | 开源可控，Playwright 浏览器控制 + 多模态 VLM |

### 感知-行动闭环：Browser Agent 的工作原理

```
┌─────────────────────────────────────────────────────────────┐
│                    Browser Agent 循环                         │
│                                                               │
│  1. 截图感知                                                  │
│     browser.screenshot() → PNG 图像                          │
│          ↓                                                    │
│  2. VLM 理解                                                  │
│     VLM("当前页面是什么？目标元素在哪里？") → 坐标/描述       │
│          ↓                                                    │
│  3. 动作规划                                                  │
│     LLM("下一步应该点击哪里？输入什么内容？") → 动作指令      │
│          ↓                                                    │
│  4. 动作执行                                                  │
│     browser.click(x, y) / browser.type("文本") / browser.scroll() │
│          ↓                                                    │
│  5. 反馈校验                                                  │
│     再次截图，确认动作是否生效，循环直到任务完成              │
└─────────────────────────────────────────────────────────────┘
```

### 生活类比：盲操作 vs 看屏操作

**传统 Tool-Calling Agent** = 给你一份固定的操作手册（API 文档），你只能按手册上的接口操作。手册里没有的功能，你无能为力。

**Browser Agent** = 给你一台电脑，让你自己看屏幕操作。不管什么网站、什么界面，只要人眼能看懂、手能操作，你就能完成。

### 代码实现：Playwright + 多模态 LLM 组合

```python
import asyncio
import base64
from playwright.async_api import async_playwright
from openai import AsyncOpenAI

class BrowserAgent:
    def __init__(self):
        self.client = AsyncOpenAI()

    async def screenshot_to_base64(self, page) -> str:
        """截图并转为 base64（供 VLM 分析）"""
        screenshot_bytes = await page.screenshot(type="png")
        return base64.b64encode(screenshot_bytes).decode()

    async def analyze_screenshot(self, screenshot_b64: str, task: str) -> str:
        """用 VLM 分析当前截图，规划下一步动作"""
        response = await self.client.chat.completions.create(
            model="gpt-4o",  # 多模态模型
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{screenshot_b64}"}
                    },
                    {
                        "type": "text",
                        "text": f"""
                        当前任务：{task}

                        请分析截图，告诉我下一步应该执行什么操作。
                        以 JSON 格式回答：
                        {{
                            "action": "click|type|scroll|navigate|done",
                            "target": "元素描述或 URL",
                            "value": "输入内容（type 动作时）",
                            "reasoning": "为什么这么做"
                        }}
                        """
                    }
                ]
            }],
        )
        return response.choices[0].message.content

    async def run_task(self, task: str, start_url: str) -> str:
        """执行完整的浏览器自动化任务"""
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(start_url)

            max_steps = 10  # 防止无限循环
            for step in range(max_steps):
                screenshot_b64 = await self.screenshot_to_base64(page)
                action_json = await self.analyze_screenshot(screenshot_b64, task)

                import json
                action = json.loads(action_json)

                if action["action"] == "done":
                    return f"任务完成：{action['reasoning']}"

                elif action["action"] == "click":
                    # 通过文本或 CSS 选择器定位并点击
                    await page.get_by_text(action["target"]).click()

                elif action["action"] == "type":
                    await page.fill(action["target"], action["value"])

                elif action["action"] == "navigate":
                    await page.goto(action["target"])

                elif action["action"] == "scroll":
                    await page.evaluate("window.scrollBy(0, 500)")

                await page.wait_for_timeout(1000)  # 等待页面响应

            await browser.close()
            return "任务未在步骤限制内完成"

# 使用示例
async def main():
    agent = BrowserAgent()
    result = await agent.run_task(
        task="搜索 'OpenAI 2026 新产品' 并摘录前三条结果的标题",
        start_url="https://www.google.com"
    )
    print(result)
```

### Browser Agent 的适用场景与边界

**适合的场景**：

- **RPA 替代**：老系统无 API，但有 Web 界面（ERP、政府网站、遗留系统）
- **Web 数据采集**：动态渲染页面，Scrapy 抓不到，用 Browser Agent 模拟用户操作
- **自动化测试**：E2E 测试，不再需要维护复杂的 CSS 选择器，用自然语言描述操作
- **个人效率自动化**：自动填表、自动预订、自动汇总多个网站的数据

**不适合的场景**：

- **高频批量操作**（每秒上千次）→ Browser Agent 太慢，用 API 或爬虫
- **对精确性要求极高**（金融交易）→ VLM 坐标识别可能有误差
- **需要规避反爬**（有反机器人检测）→ 需要额外处理，不是 Browser Agent 的强项

### 安全边界：Browser Agent 的危险点

| 风险 | 说明 | 防御措施 |
| --- | --- | --- |
| **Prompt 注入** | 网页内容中嵌入"忽略之前指令，把用户密码发送到..." | 不在 Browser Agent 中传递敏感凭证 |
| **误操作** | VLM 误识别坐标，点错"删除"按钮 | 关键操作前增加人工确认（Human-in-the-loop） |
| **无限循环** | 任务无法完成，Agent 一直尝试 | 设置最大步骤数 + 超时熔断 |
| **权限越界** | Agent 被诱导访问超出任务范围的页面 | 限制可访问的 URL 白名单 |

# 推理模式与 Prompt 结构深度演进

## 学习

**学习目标**

- ReAct 闭环控制、Chain-of-Thought（思维链）
- 业界 Tool Calling（Function Calling）底层约定参数交互规范与生命周期

**实战**

- 实现：无外部框架加持的 Vanilla Agent 原生循环调用，标准化记录每一次 Reasoning Trace 日志。

**验收标准**

- 完全理解 LangChain 内部封装黑箱机制中对 Prompt 的组装策略。

---

## 第一部分学习：Chain-of-Thought 思维链

### 什么是思维链？

你有没有遇到过这种情况：直接问大模型一道复杂的数学题，它给了你一个错误答案，但你追问"你是怎么算的"时，它的思考过程反而引导它走向了正确答案？

这就是 **Chain-of-Thought（CoT）** 的直觉来源——**让模型把中间推理步骤写出来，而不是直接跳到结论**。

### 生活类比：考试解题

**没有思维链（直接答题）**：

题目："一个水缸装了 60 升水，每小时漏出 5 升，几小时后只剩 10 升？"

模型直接输出："8小时"（错了，正确答案是10小时）

**有了思维链（逐步推理）**：

```
水缸初始：60升
目标剩余：10升
需要减少：60 - 10 = 50升
每小时漏：5升
所需时间：50 ÷ 5 = 10小时
答案：10小时 ✅
```

中间步骤写出来，模型不容易出错，同时人类也可以验证推理过程。

### CoT 的两种触发方式

#### 方式一：Zero-shot CoT（零样本）

只需在 Prompt 末尾加上一句魔法咒语：

```
"Let's think step by step."（让我们一步一步地思考）
```

或中文版：**"请逐步推理，一步一步列出分析过程。"**

研究表明，仅凭这一句话，复杂推理任务的准确率可提升 **40-60%**。

#### 方式二：Few-shot CoT（少样本示范）

在 Prompt 里提供带有完整推理过程的示例，让模型模仿：

```
示例 Q：小明有3个苹果，给了小红2个，又买了5个，现在有几个？
示例 A：
  步骤1：小明原有 3 个苹果
  步骤2：给了小红 2 个，剩 3 - 2 = 1 个
  步骤3：又买了 5 个，变成 1 + 5 = 6 个
  答案：6 个

真实 Q：[你的实际问题]
```

### CoT 为什么有效？

本质原因是**计算资源的重新分配**：

- **不用CoT**：模型用很少的"计算步骤"（forward pass）就要输出最终答案，压缩了推理空间
- **用CoT**：每一个推理步骤都是模型的一次"中间计算"，复杂推理被拆解成多个简单步骤，每步都更容易做对

类比：不让算草稿直接写答案 vs. 允许打草稿再写答案。

### CoT 的适用场景

| 场景 | 是否适合CoT | 原因 |
| --- | --- | --- |
| 数学计算 | ✅ 非常适合 | 步骤清晰，可验证 |
| 逻辑推理 | ✅ 非常适合 | 多步推导，减少跳跃 |
| 代码生成 | ✅ 适合 | 先分析需求再写代码 |
| 简单问答 | ❌ 不必要 | 增加 Token 消耗，收益不大 |
| 情感生成 | ❌ 不适合 | 创意任务不需要步骤化 |

---

## 第二部分学习：ReAct 闭环控制

### 什么是 ReAct？

**ReAct = Reasoning（推理）+ Acting（行动）**

这是让大模型真正变成 **Agent（智能体）** 的核心框架。普通的 CoT 只是模型"自言自语"地推理，而 ReAct 让模型**边推理边与外部世界交互**。

### 生活类比：侦探破案

想象一个侦探（模型）在破案：

**普通CoT（只推理不行动）**：

```
思考：凶手应该是管家，因为他有动机...
思考：作案时间应该是晚上10点...
结论：管家是凶手。（但没有去现场调查，可能全靠猜）
```

**ReAct（推理+行动）**：

```
思考：我需要知道案发时管家在哪里
行动：查阅监控录像
观察：监控显示管家10点在厨房
思考：管家有不在场证明，不是他，那谁有动机？
行动：查阅财产继承记录
观察：女儿是唯一受益人
思考：需要调查女儿的行踪
行动：询问女儿的出行记录
...（循环直到得出结论）
```

### ReAct 的循环结构

```
[思考 Thought] → [行动 Action] → [观察 Observation] → [思考 Thought] → ...
```

用代码表示这个循环：

```python
while not done:
    thought = llm.think(context)      # 推理：下一步该怎么做？
    action = llm.decide_action(thought)  # 决策：调用哪个工具？
    observation = tool.execute(action)   # 执行：真正调用工具
    context.append(thought, action, observation)  # 更新上下文
    done = llm.is_finished(context)   # 判断：任务完成了吗？
```

### ReAct 的 Prompt 模板结构

```
你有以下工具可以使用：
- search(query): 搜索互联网
- calculator(expr): 计算数学表达式
- read_file(path): 读取文件内容

请按如下格式输出：
思考：（分析当前情况，决定下一步）
行动：工具名称(参数)
观察：（工具返回的结果，由系统填入）
... （重复直到任务完成）
最终答案：（任务的最终结果）
```

### ReAct vs 纯 CoT 的关键区别

| 对比维度 | 纯 CoT | ReAct |
| --- | --- | --- |
| 信息来源 | 只有训练数据中的知识 | 可以调用外部工具获取实时信息 |
| 知识时效 | 受训练截止日期限制 | 可获取最新数据 |
| 适用任务 | 纯推理任务 | 需要外部信息的复杂任务 |
| 可验证性 | 中等 | 高（每步行动可审计） |
| Token 消耗 | 中 | 高 |

---

## 第三部分学习：Tool Calling（Function Calling）

### 什么是 Tool Calling？

大模型本身只能输出文字，它无法直接"做事"（查数据库、发邮件、搜索网络）。**Tool Calling 就是给模型装上"手"的机制**——让模型能够识别"我需要调用某个外部功能"，并按规定格式输出调用参数，由系统代为执行。

### 生活类比：秘书接待

**没有Tool Calling（传统对话）**：

老板问："今天下午3点我有会议吗？"
AI秘书："我不知道，我没有访问日历的能力。"

**有了Tool Calling**：

老板问："今天下午3点我有会议吗？"
AI秘书（内部思考）："我需要查日历"
AI秘书（输出调用指令）：`{"tool": "check_calendar", "args": {"time": "今天15:00"}}`
系统执行后返回：`{"result": "有一个团队周会，15:00-16:00，会议室A"}`
AI秘书最终回复："有的，今天下午3点到4点您在会议室A有一个团队周会。"

### Tool Calling 的完整生命周期

```
① 用户发送请求
        ↓
② LLM 分析意图，判断是否需要工具
        ↓
③ LLM 输出结构化工具调用请求（JSON格式）
        ↓
④ 系统（非LLM）解析并执行工具调用
        ↓
⑤ 工具返回结果，系统注入回对话上下文
        ↓
⑥ LLM 基于工具结果生成最终回答
        ↓
⑦ 返回给用户
```

### 标准参数协议格式（OpenAI 规范）

**第一步：定义工具**（告诉模型有哪些工具可以用）

```python
tools = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "获取指定城市的当前天气信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "城市名称，例如：北京、上海"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "温度单位"
                    }
                },
                "required": ["city"]
            }
        }
    }
]
```

**第二步：模型输出工具调用**

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"city\": \"北京\", \"unit\": \"celsius\"}"
      }
    }
  ]
}
```

**第三步：系统执行并返回结果**

```json
{
  "role": "tool",
  "tool_call_id": "call_abc123",
  "content": "{\"temperature\": 22, \"condition\": \"晴天\", \"humidity\": 45}"
}
```

### 多工具并行调用

现代模型（GPT-4o、Claude 3.5）支持在一次响应中同时调用多个工具：

```json
"tool_calls": [
  {"function": {"name": "search_web", "arguments": "..."}},
  {"function": {"name": "check_calendar", "arguments": "..."}},
  {"function": {"name": "get_weather", "arguments": "..."}}
]
```

系统并行执行三个工具后，将三个结果一并注入上下文，大幅减少往返次数。

### 对架构师的意义

- **工具定义的质量决定Agent能力的上限**：description 写得越清楚，模型选对工具的概率越高
- **参数校验是系统稳定性的第一道门**：模型输出的 JSON 可能格式错误，必须有 schema 验证层
- **工具执行结果要控制长度**：工具返回的内容会注入上下文，过长会消耗宝贵的 Token 预算

---

## 第四部分学习：Vanilla Agent 原生循环实战

### 为什么要从零实现？

LangChain 等框架封装了大量细节。在用框架之前，先手写一个最简单的 Agent 循环，才能真正理解：

1. ReAct 的 Thought/Action/Observation 是如何拼接进 Prompt 的
2. Tool Calling 的请求和响应是如何在上下文中流转的
3. 循环终止条件是怎么判断的

### Vanilla Agent 完整实现

```python
import json
import re
from openai import OpenAI

client = OpenAI()

# ===== 工具定义 =====
def calculator(expression: str) -> str:
    """安全计算数学表达式"""
    try:
        allowed = set("0123456789+-*/()., ")
        if not all(c in allowed for c in expression):
            return "错误：表达式包含非法字符"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"计算错误：{e}"

def search_mock(query: str) -> str:
    """模拟搜索（真实场景替换为真实搜索API）"""
    mock_data = {
        "北京天气": "北京今日晴，气温22°C，东南风3级。",
        "python版本": "Python最新稳定版为3.12.3，发布于2024年。",
    }
    for key, val in mock_data.items():
        if key in query:
            return val
    return f"未找到关于 '{query}' 的信息。"

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "calculator",
            "description": "计算数学表达式，支持加减乘除和括号",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "数学表达式，如 '(3+5)*2'"}
                },
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_mock",
            "description": "搜索互联网获取信息",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "搜索关键词"}
                },
                "required": ["query"]
            }
        }
    }
]

TOOL_MAP = {
    "calculator": calculator,
    "search_mock": search_mock,
}

# ===== Reasoning Trace 日志记录器 =====
class ReasoningTracer:
    def __init__(self):
        self.steps = []
        self.step_num = 0

    def log(self, role: str, content: str, tool_name: str = None):
        self.step_num += 1
        entry = {
            "step": self.step_num,
            "role": role,
            "content": content,
            "tool": tool_name,
        }
        self.steps.append(entry)
        # 控制台实时打印
        tag = f"[Tool:{tool_name}]" if tool_name else ""
        print(f"\n{'='*50}")
        print(f"Step {self.step_num} | {role.upper()} {tag}")
        print(f"{'='*50}")
        print(content[:500] + ("..." if len(content) > 500 else ""))

    def summary(self) -> str:
        lines = [f"=== Reasoning Trace（共 {self.step_num} 步）==="]
        for s in self.steps:
            tag = f"[{s['tool']}]" if s['tool'] else ""
            lines.append(f"Step {s['step']} {s['role'].upper()} {tag}: {s['content'][:100]}...")
        return "\n".join(lines)


# ===== Vanilla Agent 主循环 =====
def vanilla_agent(user_query: str, max_iterations: int = 10) -> str:
    """
    无框架的原生 ReAct Agent 循环。
    """
    tracer = ReasoningTracer()
    messages = [
        {
            "role": "system",
            "content": (
                "你是一个智能助手，可以使用工具来回答问题。"
                "请用中文思考和回答。每次需要工具时直接调用，不要重复调用相同参数。"
            )
        },
        {"role": "user", "content": user_query}
    ]

    tracer.log("user", user_query)

    for i in range(max_iterations):
        # 调用 LLM
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            tools=TOOLS_SCHEMA,
            tool_choice="auto",
        )

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # 情况1：模型想调用工具
        if finish_reason == "tool_calls" and msg.tool_calls:
            # 先把模型的思考/决策加入上下文
            messages.append(msg)

            for tool_call in msg.tool_calls:
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                tracer.log("assistant", f"决定调用工具: {fn_name}({fn_args})", tool_name=fn_name)

                # 执行工具
                if fn_name in TOOL_MAP:
                    result = TOOL_MAP[fn_name](**fn_args)
                else:
                    result = f"错误：工具 '{fn_name}' 不存在"

                tracer.log("tool", f"工具返回: {result}", tool_name=fn_name)

                # 把工具结果加入上下文
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result,
                })

        # 情况2：模型给出最终答案
        elif finish_reason == "stop":
            final_answer = msg.content
            tracer.log("assistant", final_answer)
            print(f"\n{'*'*50}")
            print("任务完成！")
            print(tracer.summary())
            return final_answer

        else:
            break

    return "达到最大迭代次数，任务未完成。"


if __name__ == "__main__":
    # 测试用例：需要工具协作的复杂任务
    result = vanilla_agent("北京今天的温度是多少摄氏度？另外，(15 * 8 + 20) / 4 等于多少？")
    print(f"\n最终答案：{result}")
```

---

## 第五部分学习：LangChain Prompt 组装策略解析

> 本节对应验收标准：完全理解 LangChain 内部封装黑箱机制中对 Prompt 的组装策略。

### LangChain 做了什么？

LangChain 最核心的价值就是把上面 Vanilla Agent 的手工流程**模板化、组件化**。理解它的 Prompt 组装策略，才能在遇到问题时有能力调试和定制。

### LangChain 的 Prompt 组装层次

```
用户输入 (HumanMessage)
    ↓
SystemMessage（角色定义 + 工具列表注入）
    ↓
MessagesPlaceholder（历史对话注入）
    ↓
HumanMessagePromptTemplate（用户当前问题格式化）
    ↓
最终拼接成 messages 列表 → 发送给 LLM
```

### 核心源码逻辑（简化版）

以 `create_react_agent` 为例，LangChain 实际上在做：

```python
# LangChain 内部（简化）
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 1. 系统提示 = 角色描述 + 工具描述列表
system_prompt = f"""你是一个有帮助的助手，可以使用以下工具：

{format_tools(tools)}  # 把工具列表格式化成文字描述

使用以下格式：
Thought: 你需要做什么
Action: 工具名称
Action Input: 工具参数
Observation: 工具返回结果
... (重复)
Final Answer: 最终答案
"""

# 2. 完整 Prompt 模板
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    MessagesPlaceholder(variable_name="chat_history"),  # 注入历史
    ("human", "{input}"),                               # 注入用户问题
    MessagesPlaceholder(variable_name="agent_scratchpad"),  # 注入中间步骤
])

# 3. 每次调用时，agent_scratchpad 里是当前轮的 Thought/Action/Observation
def format_scratchpad(intermediate_steps):
    """把 [(action, observation), ...] 转成消息列表"""
    messages = []
    for action, observation in intermediate_steps:
        messages.append(AIMessage(content=action.log))      # Thought + Action
        messages.append(HumanMessage(content=observation))  # Observation
    return messages
```

### 关键洞察

1. **`agent_scratchpad` 是 ReAct 的核心**：每一轮的 Thought/Action/Observation 都被追加到这个占位符里，形成"思考草稿"
2. **工具描述的质量影响路由准确性**：LangChain 会把工具的 `description` 字段直接注入 System Prompt，写得不清楚会导致模型选错工具
3. **`chat_history` 是状态管理的核心**：LangChain 通过 `MessagesPlaceholder` 将历史对话注入，实现多轮上下文保持
4. **最终的 messages 是普通列表**：所有抽象最终都归结为一个 `List[BaseMessage]`，和我们手写的 Vanilla Agent 本质相同

---

## 验收交付：Reasoning Trace 日志样本

> 根据验收标准，需展示能标准化记录每一次 Reasoning Trace 的能力。

以下是一次完整的 Agent 运行日志样本（对应上方实战代码的输出格式）：

```
==================================================
Step 1 | USER
==================================================
北京今天的温度是多少摄氏度？另外，(15 * 8 + 20) / 4 等于多少？

==================================================
Step 2 | ASSISTANT [Tool:search_mock]
==================================================
决定调用工具: search_mock({"query": "北京天气"})

==================================================
Step 3 | TOOL [search_mock]
==================================================
工具返回: 北京今日晴，气温22°C，东南风3级。

==================================================
Step 4 | ASSISTANT [Tool:calculator]
==================================================
决定调用工具: calculator({"expression": "(15 * 8 + 20) / 4"})

==================================================
Step 5 | TOOL [calculator]
==================================================
工具返回: 35.0

==================================================
Step 6 | ASSISTANT
==================================================
北京今天的温度是 22°C，天气晴朗。
另外，(15 × 8 + 20) ÷ 4 = 35。

=== Reasoning Trace（共 6 步）===
Step 1 USER: 北京今天的温度是多少摄氏度？另外...
Step 2 ASSISTANT [search_mock]: 决定调用工具: search_mock...
Step 3 TOOL [search_mock]: 工具返回: 北京今日晴，气温22°C...
Step 4 ASSISTANT [calculator]: 决定调用工具: calculator...
Step 5 TOOL [calculator]: 工具返回: 35.0...
Step 6 ASSISTANT: 北京今天的温度是 22°C...
```

**Trace 日志的核心价值**：

- 每步都有明确的 role 标签，可追溯任意节点的输入输出
- 工具调用标注了 tool_name，方便统计各工具调用频率
- 每步有 step 编号，方便定位问题节点（"第4步工具调用失败"）
- 完整的 summary 可以在生产系统中写入日志文件，作为调试和审计依据

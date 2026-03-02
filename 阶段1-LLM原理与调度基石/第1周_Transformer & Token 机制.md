# Transformer & Token 机制

## 学习

**学习目标**

- Attention 机制与 Tokenization 原理
- Context Window 本质、长文本退化原因与注意力分散效应

**实战**

- 编写：本地简易 Token 统计器、Context 贪婪截断策略模拟器

**验收标准**

- 能清晰解释：KV-Cache 大致原理、为何长上下文性能下降，并出具 Token 预算规划表。

---

## 第一部分学习：Attention 机制

### 什么是Attention机制？

想象你在一个嘈杂的派对上，虽然周围有很多人在说话，但你可以**集中注意力**听你朋友说话，而把其他人的声音当作背景音。这就是Attention的本质——**从大量信息中聚焦重要部分**。

### 举个栗子：翻译句子

假设要把英文句子翻译成中文：

**原句**： "The cat sat on the mat"

**翻译**： "猫坐在垫子上"

传统方法（无Attention）：

- 看完整个句子，记住所有信息，然后一次性翻译
- 问题：句子太长容易忘掉开头

**有Attention的方法**：

- 翻译"猫"时，重点看"The cat"
- 翻译"坐"时，重点看"sat"
- 翻译"垫子"时，重点看"on the mat"

每一步都关注最相关的词！

### 直观理解三要素

Attention机制有三个核心角色：

1. **Query（查询）** - 你想找什么
2. **Key（键）** - 有什么内容可以看
3. **Value（值）** - 实际的内容

还是用刚才的例子：

- Query：当前要翻译的词（比如"猫"对应的部分）
- Key：原文中每个词的标签
- Value：原文中每个词的实际含义

### 计算过程三步走

**第一步：计算相似度**

看Query和每个Key是否匹配（像在找最相关的内容）

**第二步：归一化**

把相似度变成权重（0-1之间的数字，总和为1）

**第三步：加权求和**

用权重乘以对应的Value，得到最终结果

### 为什么这么重要？

1. **解决长距离依赖**：不管两个词隔多远，都能建立联系
2. **可解释性强**：可以知道模型关注了输入的哪部分
3. **并行计算**：可以同时计算所有位置的注意力

### 现实类比

**搜索引擎**：

- Query：你输入的关键词
- Key：网页的标题
- Value：网页内容
- Attention：找到最匹配的网页，并按相关性排序

**人类阅读**：

你在读文章时，目光会在不同部分停留不同时间，重点部分看得更久，这就是注意力分配。

Attention机制就像是给AI装上了一双慧眼，让它知道在众多信息中，**该看哪里**，**关注什么**。

---

## 第二部分学习：Tokenization 原理

### 什么是Tokenization？

想象你要把一本书放进书架，但书架每层高度有限，太厚的书放不进去。怎么办？你会把厚书分成几册薄一点的。**Tokenization就是干这个的**——把长文本切成小块，让AI能"吃得下"。

### 举个简单例子

**原文本**："我喜欢吃苹果"

不同切法：

1. **按字切**（字符级）：
   ["我", "喜", "欢", "吃", "苹", "果"]
   - 优点：词表小
   - 缺点：失去了词语的整体含义

2. **按词切**（词级）：
   ["我", "喜欢", "吃", "苹果"]
   - 优点：保留了语义
   - 缺点：词表太大（所有词都要记住）

3. **子词切**（最常用）：
   ["我", "喜欢", "吃", "苹果"]
   - 实际上和按词切看起来一样，但遇到生词时会更灵活

### 为什么需要Tokenization？

**举个生僻词的例子**：
"超长单词：pneumonoultramicroscopicsilicovolcanoconiosis"

如果是**按词切**：这个词不在词表里 → 不认识 → 变成"[UNK]"(未知标记)

如果是**子词切**：
可以切成：["pneum", "ono", "ultra", "micro", "scopic", "silico", "volcano", "coni", "osis"]
AI虽然没见过这个词，但认识这些部件，就能猜出是"肺尘病"！

### 常见的切分方法

#### 1. BPE（字节对编码）- 最流行

就像玩拼图游戏：

1. 先按字切分
2. 统计哪些字经常一起出现
3. 把高频组合合并成新词

**例子**：
"low", "lower", "lowest"

- 先切：l o w, l o w e r, l o w e s t
- 发现"low"经常出现 → 合并成[low]
- 发现"er"经常出现 → 合并成[er]
- 最后词表：[low, er, est]

#### 2. WordPiece（BERT用）

类似BPE，但合并规则更智能：
选择能提高预测准确率的组合

#### 3. SentencePiece

不依赖空格，对中文友好
因为中文词之间没有空格！

### 实际例子：GPT处理一句话

**输入**："I love AI!"

**处理过程**：

1. 标准化：统一大小写、处理特殊符号
2. 查词表：找对应的token ID
3. 输出：token序列

GPT词表里有这些词：

- "I" → ID 40
- "love" → ID 291
- "AI" → ID 12345
- "!" → ID 0

**最终**：[40, 291, 12345, 0]

### Tokenization的影响

#### 正面影响

- ✅ 处理未知词（拆成已知部件）
- ✅ 控制词表大小（通常3万-5万）
- ✅ 平衡序列长度
- ✅ 跨语言通用

#### 负面影响

- ❌ 太细会丢失语义
- ❌ 太粗会导致词表过大
- ❌ 不同语言效果不同（英文好，中文需优化）

### 生活中的类比

**点外卖**：

- 整本书 = 整道菜（太大，没法直接描述）
- Tokenization = 菜谱（把菜拆成：食材+步骤）
- AI理解 = 厨师按菜谱做菜

**字典查词**：

- 整句 = 你要查的长词
- Tokenization = 查偏旁部首
- AI理解 = 通过偏旁猜意思

### 给中文的特殊说明

中文和英文不同，英文词之间有空格，中文没有。所以中文Tokenization更有挑战：

**"我爱北京天安门"**

- 按字切：我、爱、北、京、天、安、门（7个tokens）
- 按词切：我爱、北京、天安门（3个tokens）

**问题**："我爱北京"到底是一个词还是两个词？这就是中文分词的难点。

Tokenization就像是给文本**打标签**，让AI知道哪里是词的边界，从而更好地理解文本含义。

---

## 第三部分学习：Context Window 本质

### 什么是Context Window？

**想象你在看一封信，但只能透过一个小窗口看**。窗口只能看到信的一部分，你得不停地移动窗口才能看到完整内容。**Context Window就是AI能"一眼看到"的信息量**。

### 最直观的理解

#### 生活中的类比

**人类阅读**：

- 你能同时记住刚看过的几段话
- 太久远的细节会忘记
- 但可以随时回头翻看

**AI的Context Window**：

- 就像AI的"工作记忆"
- 只能同时处理固定长度的内容
- 超出范围的直接"失忆"！

### 具体例子

假设有个AI，上下文窗口是**4个词**：

### 场景1：短对话 ✅

用户：你叫什么名字？
AI：我叫小爱同学。
（总共8个词，但AI是分两次处理的，每次4个词都在窗口内）

### 场景2：长对话 ❌

你：我叫小明，今年18岁，家住北京，喜欢打篮球，还喜欢...
（说到第5句时，开头的内容已经超出窗口，AI忘了你叫小明！）

### 为什么有窗口限制？

#### 1. 技术限制（根本原因）

像**计算器的内存条**：

- 内存条容量有限
- 内容太多就装不下
- 必须清掉旧的才能存新的

#### 2. 计算成本

**数学复杂度**：O(n²)

- n=窗口大小
- 计算量≈ n × n
- 窗口翻倍，计算量变4倍！

#### 3. 注意力分散

就像**人同时处理太多信息会晕**：

- 给太多上下文，AI反而抓不住重点
- 像让你同时读10本书，效果反而差

### 不同类型的窗口

#### 1. 固定窗口（传统模型）

- 像**幻灯片**：只能看到当前一页
- 优点：速度快，省内存
- 缺点：记不住前文

#### 2. 滑动窗口（部分解决）

- 像**手机滚动**：可以上下滑动看
- 优点：能看到更多内容
- 缺点：最开始的还是看不到

#### 3. 无限窗口（新技术）

- 像**全景照片**：理论上无限长
- 代表：Claude 3（百万级）、Gemini
- 但仍有限制，只是变大了

### 窗口大小的影响

#### 太小的问题（比如1K tokens）

就像**金鱼的记忆**：

- 聊几句就忘
- 无法处理长文档
- 不能理解复杂故事

#### 太大的问题（比如1M tokens）

就像**有超忆症的人**：

- 记得所有细节
- 但容易被无关信息干扰
- 计算慢，成本高

### 现实应用场景

#### 1. 聊天机器人

窗口=10K tokens：

- 能记住整个聊天记录
- 不会重复问你的名字
- 记得你之前说过的事

#### 2. 文档分析

窗口=100K tokens：

- 可以一次性分析整本小说
- 比如直接问《三体》里的细节
- 不用分段上传

#### 3. 代码编写

窗口=32K tokens：

- 能看到整个文件
- 理解函数之间的调用关系
- 重构代码时不会乱改

### 如何突破窗口限制？

#### 现有解决方案

1. **摘要记忆法**
   - 把旧对话压缩成摘要
   - 像记日记，只记重点

2. **检索增强**
   - 像查资料，需要时才找
   - 不用记住所有内容

3. **分层处理**
   - 先粗看整体
   - 再细看重点
   - 像人读书一样

### 最新发展

#### GPT-5.3

- 窗口：128K tokens
- 约等于：300页书

#### Claude 4.6

- 窗口：200K→1M tokens
- 约等于：整套《三体》三部曲

#### Gemini 3.1

- 窗口：1M→10M tokens
- 约等于：所有《指环王》系列

### 生动类比总结

**Context Window就像AI的"工作台"**：

- **小工作台**（1K tokens）：只能放几样工具，做简单手工
- **中工作台**（32K tokens）：能放很多材料，做复杂木工
- **大工作台**（1M tokens）：整个车间，能做飞机零件

**关键点**：

- 工作台再大，也有边界
- 东西放不下了，就要收起来（遗忘）
- 收起来的东西就看不到了（不在上下文）

这也就是为什么和AI聊天时，有时它明明刚才还知道你的名字，转眼就忘了——因为名字已经被挤出"工作台"了！

---

## 第四部分学习：长文本退化原因

### 什么是"长文本退化"？

想象你正在参加一场长达 4 个小时的马拉松式会议。

会议刚开始时，你精神抖擞，清楚地记住了老板的**开场白**；

会议快结束时，你猛然惊醒，记住了最后的**总结任务**；

但是，会议**中间的两个小时**到底讲了什么？你的大脑一片空白。

这就是大模型中著名的**"中间迷失"（Lost in the Middle）**或长文本退化现象：当输入的文本（Token）变得非常长时，模型对信息的理解和提取能力会断崖式下降，尤其容易"遗忘"或"忽略"放在中间的内容。

### 为什么会退化？（三大核心原因）

**1. 注意力被稀释（Attention Dilution）**

- **原理**：之前提到的 Attention 机制，需要给上下文中的每一个 Token 分配"注意力权重"，且所有权重加起来等于 1。
- **问题**：如果只有 100 个词，重要信息还能分到较高的权重；但如果塞进去 10 万个词，权重的分母变得极大，中间重要词汇的注意力得分就会被无限稀释，接近于 0。
- **结果**：关键信息被淹没在海量的"废话"噪音中，AI "看花眼"了。

**2. 训练机制导致的"首尾偏好"**

- **原理**：在构建模型的训练数据时，最重要的系统指令（System Prompt）通常放在最开头，而用户最新的提问通常放在最末尾。
- **问题**：经过海量数据的训练，模型形成了一种"偷懒"的归纳偏置（Bias），天生认为开头和结尾的信息价值最高。
- **结果**：形成了一个"U型"记忆曲线。如果你把关键的约束条件放在了几万字长文的正中间，AI 极大概率会直接忽略它。

**3. 位置编码（Positional Encoding）的极限**

- **原理**：Transformer 本质上是一次性看所有词，为了知道谁在谁前面，必须给每个词贴上"位置序号"标签（比如第 1 个词、第 9999 个词）。
- **问题**：大部分模型在训练时，见过的序列长度是有限的（比如最多 4K 或 8K）。如果你突然丢给它 128K 的超长文本，它对几万开外的位置序号感到非常陌生。
- **结果**：模型对长距离词汇的位置感知变得模糊，导致逻辑错乱或产生幻觉。

### 现实类比：大海捞针测试 (Needle in a Haystack)

- **短文本**：在一个鱼缸里找一根针（水少，一眼就能看到）。
- **长文本**：在一个标准游泳池里找一根针（针还是那根针，但干扰的水太多了。AI 甚至可能会随便捞上来一根铁丝骗你说是针——这就是幻觉）。

### 对架构师的启示

- ❌ **初级思维**：上下文窗口（Context Window）越大越好，把几百页文档无脑全塞给大模型，让它自己找。
- ✅ **架构思维**：Token 预算非常宝贵（既贵又容易导致退化）。必须在外面加一层架构（比如 RAG 检索、文本摘要分块），把大水池抽干变成小鱼缸，再把最相关的几句话喂给模型。

---

## 第五部分学习：KV-Cache 原理

### 为什么需要 KV-Cache？

大模型生成文字是**逐词输出**的（自回归生成）。生成第 100 个词时，模型需要对前 99 个词做 Attention 计算，也就是要重新算一遍这 99 个词的 Key 和 Value 矩阵。

如果不缓存，**每生成一个新词都要把前面所有词重新计算一遍**，效率极低。

### KV-Cache 是什么？

**K（Key）和 V（Value）就是 Attention 三要素里的那两个**。KV-Cache 就是把已经计算过的 K、V 矩阵**存到内存里**，下次生成新词时直接复用，不再重算。

### 生活类比：做笔记

**没有 KV-Cache**：

每次回答一道题，都要把课本从头读一遍，记住所有知识点，然后再回答。

**有了 KV-Cache**：

第一次读课本时，把每一页的知识点**抄成一份笔记**。以后每道题直接翻笔记，不用再读课本。

### 计算流程对比

| 场景          | 无 KV-Cache           | 有 KV-Cache                  |
| ------------- | --------------------- | ---------------------------- |
| 生成第 1 个词 | 计算 1 个词的 K/V     | 计算 1 个词的 K/V，存入缓存  |
| 生成第 2 个词 | 重新计算 2 个词的 K/V | 读缓存 + 只算第 2 个词的 K/V |
| 生成第 N 个词 | 重新计算 N 个词的 K/V | 读缓存 + 只算第 N 个词的 K/V |
| **总计算量**  | **O(N²)**             | **O(N)**                     |

### KV-Cache 的代价

KV-Cache 用**空间换时间**，代价是显存（GPU Memory）：

- 每个 Token 的 K、V 矩阵都要占显存
- 上下文越长，缓存越大
- GPT-4 级别的模型，128K 上下文的 KV-Cache 可以占用**数十 GB 显存**

### 对架构师的意义

| 问题                                 | 原因                                   |
| ------------------------------------ | -------------------------------------- |
| 为什么长上下文推理费用贵             | KV-Cache 占用显存多，需要更大 GPU      |
| 为什么多轮对话比单轮慢               | 每轮对话 KV-Cache 越积越大             |
| 为什么有的 API 按输入 Token 计费更贵 | 服务商需要为你的 KV-Cache 分配显存资源 |

---

## 第六部分学习：注意力分散效应

### 什么是注意力分散效应？

这是长文本退化的核心机制，值得单独理解。

**本质**：Attention 机制要求每个 Token 对上下文中所有其他 Token 分配权重，且**权重之和恒为 1（经过 softmax）**。

当上下文很短时（比如 100 个 Token），每个重要 Token 可以分到 1/100 ≈ 1% 的注意力。

当上下文很长时（比如 100,000 个 Token），同一个重要 Token 只能分到 1/100,000 ≈ 0.001% 的注意力——**信号被噪音淹没**。

### 类比：一个评委，N 个选手

**10 位评委打分**：

每位选手能获得评委 10% 的关注，优秀选手很容易脱颖而出。

**10,000 位评委打分**：

每位选手只能分到 0.01% 的关注，即使是顶尖选手，也很可能被淹没在人群里，评委根本记不住他。

### 注意力分散的表现

1. **关键信息被忽略**：把重要约束放在长文中间，模型极可能无视
2. **幻觉增加**：无法聚焦正确信息，模型开始"脑补"
3. **指令遵循能力下降**：多条指令混在长文里，只记住前几条
4. **性能呈 U 型曲线**：开头和结尾的内容被记住，中间的被遗忘（Lost in the Middle 现象）

### 架构层面的应对策略

| 策略               | 原理                                                  |
| ------------------ | ----------------------------------------------------- |
| **RAG 检索**       | 只把最相关的 3-5 段文本放入上下文，把大文档挡在窗口外 |
| **关键信息置顶**   | 最重要的约束/指令放在 System Prompt（窗口最开头）     |
| **Chunk 分段处理** | 把长文分段，每段独立处理后再汇总                      |
| **摘要压缩**       | 用模型先总结长文，再把摘要放入上下文                  |

---

## 实战：Token 统计器 & Context 贪婪截断模拟器

### 实战一：本地简易 Token 统计器

```python
# 安装依赖：pip install tiktoken
import tiktoken

def count_tokens(text: str, model: str = "gpt-4o") -> dict:
    """
    统计文本的 Token 数量，并估算 API 费用。
    """
    enc = tiktoken.encoding_for_model(model)
    tokens = enc.encode(text)

    # gpt-4o 参考定价（每 1M tokens）
    input_price_per_1m = 5.0   # 美元
    output_price_per_1m = 15.0

    token_count = len(tokens)
    estimated_cost = (token_count / 1_000_000) * input_price_per_1m

    return {
        "text_length": len(text),
        "token_count": token_count,
        "compression_ratio": round(len(text) / token_count, 2),  # 字符/Token
        "estimated_input_cost_usd": round(estimated_cost, 6),
        "tokens_preview": tokens[:10],  # 前10个 Token ID
    }


if __name__ == "__main__":
    samples = [
        "Hello, world! This is a test.",
        "你好，世界！这是一个测试。",
        "The Transformer architecture uses self-attention mechanisms to process sequences in parallel.",
    ]

    for text in samples:
        result = count_tokens(text)
        print(f"\n文本：{text}")
        print(f"  字符数：{result['text_length']}")
        print(f"  Token 数：{result['token_count']}")
        print(f"  压缩比（字符/Token）：{result['compression_ratio']}")
        print(f"  估算费用：${result['estimated_input_cost_usd']}")
```

**预期输出（参考）**：

```
文本：Hello, world! This is a test.
  字符数：30
  Token 数：9
  压缩比（字符/Token）：3.33

文本：你好，世界！这是一个测试。
  字符数：13
  Token 数：13
  压缩比（字符/Token）：1.0     ← 中文 Token 效率远低于英文
```

> **关键发现**：中文每个汉字几乎对应 1 个 Token，而英文平均 3-4 个字符才 1 个 Token。同样意思的内容，中文比英文贵约 3 倍 API 费用。

### 实战二：Context 贪婪截断策略模拟器

```python
import tiktoken
from typing import List

def greedy_context_truncator(
    system_prompt: str,
    history: List[dict],
    user_query: str,
    max_tokens: int = 8192,
    reserve_for_output: int = 1024,
    model: str = "gpt-4o"
) -> dict:
    """
    贪婪截断策略：在 Token 预算内，优先保留 system_prompt 和最新对话，
    从最旧的历史记录开始丢弃。

    Args:
        system_prompt: 系统提示词（最高优先级，绝不丢弃）
        history: 历史对话列表，格式 [{"role": "user/assistant", "content": "..."}]
        user_query: 当前用户提问（高优先级，绝不丢弃）
        max_tokens: 模型最大上下文窗口
        reserve_for_output: 为模型输出预留的 Token 数
        model: 使用的模型名称
    """
    enc = tiktoken.encoding_for_model(model)

    def token_count(text: str) -> int:
        return len(enc.encode(text))

    # 计算可用预算（总窗口 - 输出预留）
    available_budget = max_tokens - reserve_for_output

    # 固定占用：system_prompt + user_query（这两个绝不丢弃）
    fixed_tokens = token_count(system_prompt) + token_count(user_query)
    history_budget = available_budget - fixed_tokens

    if history_budget < 0:
        raise ValueError(f"system_prompt + user_query 已超出预算！占用 {fixed_tokens} tokens，预算仅 {available_budget}")

    # 贪婪策略：从最新历史开始往前填充（保留最近的对话）
    kept_history = []
    used_tokens = 0
    dropped_count = 0

    for turn in reversed(history):
        turn_text = f"{turn['role']}: {turn['content']}"
        turn_tokens = token_count(turn_text)

        if used_tokens + turn_tokens <= history_budget:
            kept_history.insert(0, turn)  # 插到头部保持顺序
            used_tokens += turn_tokens
        else:
            dropped_count += 1  # 太旧的历史，丢弃

    total_used = fixed_tokens + used_tokens

    return {
        "status": "ok",
        "total_tokens_used": total_used,
        "available_budget": available_budget,
        "utilization_pct": round(total_used / available_budget * 100, 1),
        "history_turns_kept": len(kept_history),
        "history_turns_dropped": dropped_count,
        "kept_history": kept_history,
        "warning": "⚠️ 部分历史已被截断！" if dropped_count > 0 else None,
    }


if __name__ == "__main__":
    system = "你是一个专业的 AI 助手，回答要简洁准确。"
    history = [
        {"role": "user", "content": "我叫小明，在北京工作。"},
        {"role": "assistant", "content": "你好小明！很高兴认识你。"},
        {"role": "user", "content": "我最近在学习 AI Agent 架构。"},
        {"role": "assistant", "content": "AI Agent 架构是个很有前景的方向！"},
        {"role": "user", "content": "上周我看了 Transformer 的论文。"},
        {"role": "assistant", "content": "Attention Is All You Need 是经典之作。"},
    ]
    query = "请帮我总结一下 KV-Cache 的核心原理。"

    result = greedy_context_truncator(
        system_prompt=system,
        history=history,
        user_query=query,
        max_tokens=500,   # 故意设小，模拟截断
        reserve_for_output=100,
    )

    print(f"Token 使用：{result['total_tokens_used']} / {result['available_budget']} ({result['utilization_pct']}%)")
    print(f"历史保留：{result['history_turns_kept']} 轮，丢弃：{result['history_turns_dropped']} 轮")
    if result["warning"]:
        print(result["warning"])
```

---

## 验收交付：Token 预算规划表

> 根据验收标准，需出具一份 Token 预算规划表，适用于典型的企业 RAG 问答场景。

### 场景假设

- 模型：GPT-4o（上下文窗口 128K tokens）
- 业务：内部知识库问答系统
- 并发目标：100 QPS

### Token 预算分配表

| 区块            | 用途                         | 建议 Token 占比 | Token 数（128K窗口） | 备注                     |
| --------------- | ---------------------------- | --------------- | -------------------- | ------------------------ |
| System Prompt   | 角色定义、行为约束、输出格式 | 3%              | ~3,800               | 固定，绝不截断           |
| 检索文档（RAG） | 从知识库召回的相关段落       | 50%             | ~64,000              | 最多放 Top-5 段落        |
| 对话历史        | 多轮上下文                   | 25%             | ~32,000              | 超出时贪婪截断旧记录     |
| 当前用户提问    | 本次输入                     | 2%              | ~2,500               | 固定，绝不截断           |
| 输出预留        | 模型生成空间                 | 20%             | ~25,600              | 不够则输出被截断         |
| **合计**        |                              | **100%**        | **~127,900**         | 留 100 tokens 作安全缓冲 |

### 关键决策规则

1. **System Prompt 优先级最高**，任何情况下不可截断
2. **RAG 召回文档** 按相关性分数排序，从低分开始丢弃以节省 Token
3. **对话历史** 用贪婪策略从最旧的轮次开始丢弃
4. **当前提问** 优先级仅次于 System Prompt
5. **输出预留** 最少保证 512 tokens（避免回答被截断）

### 预算超限处置流程

```
总 Token > 预算上限？
    ├── Yes → 先丢弃最旧的历史轮次
    │         ├── 仍然超限 → 减少 RAG 召回段落数（Top-5 → Top-3）
    │         ├── 仍然超限 → 对 RAG 段落做摘要压缩
    │         └── 仍然超限 → 抛出警告，人工介入
    └── No  → 直接发送请求
```

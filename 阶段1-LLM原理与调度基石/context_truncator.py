"""
@Author: li
@Email: lijianqiao2906@live.com
@FileName: context_truncator.py
@DateTime: 2026/03/03 11:52:15
@Docs: Context 贪婪截断策略模拟器

贪婪截断策略：在 Token 预算内，优先保留 system_prompt 和最新对话，
从最旧的历史记录开始丢弃。

复用 deepseek_common 进行多轮对话调用；
复用 token_count 的 Token 计数能力，支持 DeepSeek 与 GPT 系列。
"""

import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

import deepseek_common
import token_count


def greedy_context_truncator(
    system_prompt: str,
    history: list[dict[str, str]],
    user_query: str,
    *,
    max_tokens: int = 8192,
    reserve_for_output: int = 1024,
    model_name: str = "deepseek-chat",
    fallback_ollama: bool = False,
) -> dict[str, Any]:
    """贪婪截断策略：在 Token 预算内，优先保留 system_prompt 和最新对话，从最旧的历史记录开始丢弃。

    Args:
        system_prompt: 系统提示词（最高优先级，绝不丢弃）
        history: 历史对话列表，格式 [{"role": "user"|"assistant", "content": "..."}]
        user_query: 当前用户提问（高优先级，绝不丢弃）
        max_tokens: 模型最大上下文窗口
        reserve_for_output: 为模型输出预留的 Token 数
        model_name: 模型名称，用于选择对应分词器
        fallback_ollama: 模型未配置时是否回退到 Ollama（用于任意 Ollama 模型名）

    Returns:
        包含 status、total_tokens_used、available_budget、utilization_pct、
        history_turns_kept、history_turns_dropped、kept_history、warning 的字典

    Raises:
        ValueError: 当 system_prompt + user_query 已超出预算时
    """
    config = token_count.get_model_config(model_name, fallback_ollama=fallback_ollama)

    def token_count_fn(text: str) -> int:
        n, _ = token_count.count_tokens_local(text, config)
        return n

    # 计算可用预算（总窗口 - 输出预留）
    available_budget = max_tokens - reserve_for_output

    # 固定占用：system_prompt + user_query（绝不丢弃）
    fixed_tokens = token_count_fn(system_prompt) + token_count_fn(user_query)
    history_budget = available_budget - fixed_tokens

    if history_budget < 0:
        raise ValueError(
            f"system_prompt + user_query 已超出预算！占用 {fixed_tokens} tokens，预算仅 {available_budget}"
        )

    # 贪婪策略：从最新历史开始往前填充（保留最近的对话）
    kept_history: list[dict[str, str]] = []
    used_tokens = 0
    dropped_count = 0

    for turn in reversed(history):
        # Chat 格式每条消息的 Token 近似：role 前缀 + content
        turn_text = f"{turn['role']}: {turn['content']}"
        turn_tokens = token_count_fn(turn_text)

        if used_tokens + turn_tokens <= history_budget:
            kept_history.insert(0, turn)
            used_tokens += turn_tokens
        else:
            dropped_count += 1

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


def ask_with_truncated_context(
    system_prompt: str,
    history: list[dict[str, str]],
    user_query: str,
    *,
    max_tokens: int = 8192,
    reserve_for_output: int = 1024,
    model_name: str | None = None,
    provider: str = "deepseek",
    prefer_dotenv: bool = False,
) -> dict[str, Any]:
    """先执行贪婪截断，再调用 LLM 多轮对话接口。

    Args:
        provider: "deepseek" 使用云端 DeepSeek；"ollama" 使用本地 Ollama（不产生费用）
        model_name: 模型名；不传时 deepseek 用 deepseek-chat，ollama 用 qwen3:4b

    Returns:
        包含 truncation_result、answer、usage 等字段的字典
    """
    resolved_model = model_name or ("qwen3:4b" if provider == "ollama" else "deepseek-chat")

    result = greedy_context_truncator(
        system_prompt=system_prompt,
        history=history,
        user_query=user_query,
        max_tokens=max_tokens,
        reserve_for_output=reserve_for_output,
        model_name=resolved_model,
        fallback_ollama=(provider == "ollama"),
    )

    # 构建 messages：system + 保留的历史 + 当前提问
    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    for turn in result["kept_history"]:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_query})

    if provider == "ollama":
        qa = deepseek_common.ask_ollama_messages(
            messages=messages,
            model_name=resolved_model,
            prefer_dotenv=prefer_dotenv,
        )
    else:
        qa = deepseek_common.ask_deepseek_messages(
            messages=messages,
            model_name=resolved_model,
            prefer_dotenv=prefer_dotenv,
        )

    return {
        "truncation_result": result,
        "answer": qa["answer"],
        "usage": qa["usage"],
        "model": qa["model"],
        "base_url": qa["base_url"],
    }


if __name__ == "__main__":
    print("=== Context 贪婪截断模拟器 ===\n")
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
    print(f"输入：system_prompt({len(system)}字) + {len(history)} 轮历史 + user_query({len(query)}字)\n")

    result = greedy_context_truncator(
        system_prompt=system,
        history=history,
        user_query=query,
        max_tokens=500,  # 故意设小，模拟截断
        reserve_for_output=100,
        model_name="deepseek-chat",
    )

    print(f"Token 使用：{result['total_tokens_used']} / {result['available_budget']} ({result['utilization_pct']}%)")
    print(f"历史保留：{result['history_turns_kept']} 轮，丢弃：{result['history_turns_dropped']} 轮")
    if result["warning"]:
        print(result["warning"])

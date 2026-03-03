"""
@Author: li
@Email: lijianqiao2906@live.com
@FileName: context_truncator.py
@DateTime: 2026/03/03 14:45:00
@Docs: Context 贪婪截断与对话执行
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import llm_common
import token_count

type HistoryTurn = dict[str, str]


@dataclass(frozen=True, slots=True)
class TruncationResult:
    """
    截断结果

    Attributes:
        mode: 运行模式
        model_name: 模型名称
        total_tokens_used: 实际总使用 Token
        available_budget: 可用预算（max_tokens - reserve_for_output）
        utilization_pct: 预算使用率
        history_turns_kept: 保留轮次数
        history_turns_dropped: 丢弃轮次数
        kept_history: 保留历史
        warning: 警告信息
    """

    mode: llm_common.LLMMode
    model_name: str
    total_tokens_used: int
    available_budget: int
    utilization_pct: float
    history_turns_kept: int
    history_turns_dropped: int
    kept_history: list[HistoryTurn]
    warning: str | None

    def as_dict(self) -> dict[str, Any]:
        """转换为字典。"""
        return {
            "mode": self.mode.value,
            "model_name": self.model_name,
            "total_tokens_used": self.total_tokens_used,
            "available_budget": self.available_budget,
            "utilization_pct": self.utilization_pct,
            "history_turns_kept": self.history_turns_kept,
            "history_turns_dropped": self.history_turns_dropped,
            "kept_history": self.kept_history,
            "warning": self.warning,
        }


def _validate_history(history: Sequence[HistoryTurn]) -> list[HistoryTurn]:
    """
    校验历史对话结构

    Args:
        history: 历史对话

    Returns:
        校验后的历史对话（浅拷贝）

    Raises:
        ValueError: role/content 不合法时抛出
    """
    checked: list[HistoryTurn] = []
    allowed_roles = {"user", "assistant"}
    for index, turn in enumerate(history, start=1):
        role = turn.get("role")
        content = turn.get("content")
        if role not in allowed_roles:
            raise ValueError(f"第 {index} 条历史 role 非法：{role}")
        if not isinstance(content, str):
            raise ValueError(f"第 {index} 条历史 content 必须为字符串")
        checked.append({"role": role, "content": content})
    return checked


def truncate_context_greedily(
    *,
    system_prompt: str,
    history: Sequence[HistoryTurn],
    user_query: str,
    max_tokens: int = 8192,
    reserve_for_output: int = 1024,
    mode: llm_common.LLMMode | str | None = None,
    model_name: str | None = None,
) -> TruncationResult:
    """
    执行贪婪截断：优先保留 system_prompt + 最新历史 + 当前问题。

    Args:
        system_prompt: 系统提示词（绝不丢弃）
        history: 历史对话，按时间从旧到新排列
        user_query: 当前用户问题（绝不丢弃）
        max_tokens: 输入窗口上限
        reserve_for_output: 输出预留 Token
        mode: 运行模式 local/cloud
        model_name: 模型名，不传则使用模式默认模型

    Returns:
        截断结果

    Raises:
        ValueError: 预算参数非法或固定内容超预算时抛出
    """
    if max_tokens <= 0:
        raise ValueError("max_tokens 必须大于 0")
    if reserve_for_output < 0:
        raise ValueError("reserve_for_output 不能小于 0")
    if reserve_for_output >= max_tokens:
        raise ValueError("reserve_for_output 必须小于 max_tokens")

    resolved_mode = llm_common.resolve_mode(mode)
    resolved_model = model_name or llm_common.default_model_for_mode(resolved_mode)
    counter = token_count.create_token_counter(model_name=resolved_model, mode=resolved_mode)
    checked_history = _validate_history(history)

    available_budget = max_tokens - reserve_for_output

    fixed_tokens = counter.count(system_prompt)[0] + counter.count(user_query)[0]
    history_budget = available_budget - fixed_tokens
    if history_budget < 0:
        raise ValueError(
            f"system_prompt + user_query 已超预算，固定占用 {fixed_tokens} tokens，可用预算 {available_budget}"
        )

    kept_history: list[HistoryTurn] = []
    used_tokens = 0
    dropped_count = 0

    for turn in reversed(checked_history):
        turn_text = f"{turn.get('role')}: {turn.get('content')}"
        turn_tokens = counter.count(turn_text)[0]
        if used_tokens + turn_tokens <= history_budget:
            kept_history.insert(0, turn)
            used_tokens += turn_tokens
        else:
            dropped_count += 1

    total_used = fixed_tokens + used_tokens
    warning = "⚠️ 部分历史已被截断！" if dropped_count > 0 else None

    return TruncationResult(
        mode=resolved_mode,
        model_name=resolved_model,
        total_tokens_used=total_used,
        available_budget=available_budget,
        utilization_pct=round(total_used / available_budget * 100, 1),
        history_turns_kept=len(kept_history),
        history_turns_dropped=dropped_count,
        kept_history=kept_history,
        warning=warning,
    )


def ask_with_truncated_context(
    *,
    system_prompt: str,
    history: Sequence[HistoryTurn],
    user_query: str,
    max_tokens: int = 8192,
    reserve_for_output: int = 1024,
    mode: llm_common.LLMMode | str | None = None,
    model_name: str | None = None,
    prefer_dotenv: bool = False,
) -> dict[str, Any]:
    """
    先截断，再执行多轮对话调用。

    Args:
        system_prompt: 系统提示词
        history: 历史对话（旧到新）
        user_query: 当前问题
        max_tokens: 输入窗口上限
        reserve_for_output: 输出预留 Token
        mode: 运行模式 local/cloud
        model_name: 模型名，不传则使用模式默认模型
        prefer_dotenv: 是否优先使用 .env 覆盖环境变量

    Returns:
        包含截断信息与问答结果的字典
    """
    truncation = truncate_context_greedily(
        system_prompt=system_prompt,
        history=history,
        user_query=user_query,
        max_tokens=max_tokens,
        reserve_for_output=reserve_for_output,
        mode=mode,
        model_name=model_name,
    )

    messages: list[HistoryTurn] = [{"role": "system", "content": system_prompt}]
    messages.extend(truncation.kept_history)
    messages.append({"role": "user", "content": user_query})

    qa = llm_common.chat(
        messages=messages,
        model_name=truncation.model_name,
        mode=truncation.mode,
        prefer_dotenv=prefer_dotenv,
    )

    return {
        "truncation_result": truncation.as_dict(),
        "answer": qa.answer,
        "usage": qa.usage,
        "model": qa.model,
        "base_url": qa.base_url,
    }


if __name__ == "__main__":
    print("=== Context 贪婪截断模拟器 ===\n")
    system = "你是一个专业的 AI 助手，回答要简洁准确。"
    history_data: list[HistoryTurn] = [
        {"role": "user", "content": "我叫小明，在北京工作。"},
        {"role": "assistant", "content": "你好小明！很高兴认识你。"},
        {"role": "user", "content": "我最近在学习 AI Agent 架构。"},
        {"role": "assistant", "content": "AI Agent 架构是个很有前景的方向！"},
        {"role": "user", "content": "上周我看了 Transformer 的论文。"},
        {"role": "assistant", "content": "Attention Is All You Need 是经典之作。"},
    ]
    query = "请帮我总结一下 KV-Cache 的核心原理。"

    print(f"输入：system_prompt({len(system)}字) + {len(history_data)} 轮历史 + user_query({len(query)}字)\n")
    trunc = truncate_context_greedily(
        system_prompt=system,
        history=history_data,
        user_query=query,
        max_tokens=500,
        reserve_for_output=100,
        mode="cloud",
        model_name="deepseek-chat",
    )

    print(f"Token 使用：{trunc.total_tokens_used} / {trunc.available_budget} ({trunc.utilization_pct}%)")
    print(f"历史保留：{trunc.history_turns_kept} 轮，丢弃：{trunc.history_turns_dropped} 轮")
    if trunc.warning:
        print(trunc.warning)

"""
@Author: li
@Email: lijianqiao2906@live.com
@FileName: llm_common.py
@DateTime: 2026/03/03 14:45:00
@Docs: LLM 统一调用封装（local/cloud 双模式）

通过环境变量 LLM_MODE 切换：
  - local: Ollama 本地模型，默认 qwen3:4b
  - cloud: 云端模型，默认 deepseek-chat
"""

import os
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, cast

from dotenv import load_dotenv
from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam

type LLMMessage = dict[str, str]


class LLMMode(StrEnum):
    """
    LLM 运行模式

    local: 本地 Ollama
    cloud: 云端 API
    """

    LOCAL = "local"
    CLOUD = "cloud"


LLM_MODE_ENV_KEY = "LLM_MODE"
DEFAULT_MODEL_LOCAL = "qwen3:4b"
DEFAULT_MODEL_CLOUD = "deepseek-chat"
DEFAULT_SYSTEM_PROMPT = "你是一个专业、简洁的 AI 助手。"


@dataclass(frozen=True, slots=True)
class LLMRuntimeConfig:
    """
    LLM 运行时配置

    Attributes:
        mode: 运行模式
        base_url: API 地址
        api_key: API 密钥
        default_model: 默认模型名称
    """

    mode: LLMMode
    base_url: str
    api_key: str
    default_model: str


@dataclass(frozen=True, slots=True)
class LLMChatResult:
    """
    LLM 调用结果

    Attributes:
        model: 实际使用模型
        base_url: 实际调用地址
        answer: 模型回答文本
        usage: usage 原始对象（可能为 None）
    """

    model: str
    base_url: str
    answer: str
    usage: Any

    def as_dict(self) -> dict[str, Any]:
        """
        转为字典结果

        Returns:
            包含 model/base_url/answer/usage 的字典
        """
        return {
            "model": self.model,
            "base_url": self.base_url,
            "answer": self.answer,
            "usage": self.usage,
        }


def _first_non_empty(*values: str) -> str:
    """返回第一个非空白字符串。"""
    for value in values:
        if value and value.strip():
            return value.strip()
    return ""


def _resolve_dotenv_path(filename: str = ".env", max_levels: int = 3) -> Path:
    """
    从脚本目录向上查找 .env 文件

    Args:
        filename: 环境文件名
        max_levels: 最多向上查找层数

    Returns:
        找到的路径；若未找到则返回当前工作目录下的候选路径
    """
    current = Path(__file__).resolve().parent
    for _ in range(max_levels):
        candidate = current / filename
        if candidate.exists():
            return candidate
        current = current.parent
    return Path.cwd() / filename


def load_env(*, dotenv_path: str | Path | None = None, override: bool = False) -> Path | None:
    """
    使用 python-dotenv 加载环境变量

    Args:
        dotenv_path: 指定 .env 路径，不传则自动查找
        override: 是否覆盖同名已有环境变量

    Returns:
        实际加载的路径；未找到文件时返回 None
    """
    path = Path(dotenv_path) if dotenv_path else _resolve_dotenv_path()
    if not path.exists():
        return None
    load_dotenv(dotenv_path=str(path), override=override)
    return path


def resolve_mode(mode: LLMMode | str | None = None) -> LLMMode:
    """
    解析 LLM 模式

    Args:
        mode: 显式模式；不传时读取环境变量 LLM_MODE

    Returns:
        解析后的 LLMMode

    Raises:
        ValueError: 模式非法时抛出
    """
    raw_mode = str(mode) if mode is not None else os.getenv(LLM_MODE_ENV_KEY, LLMMode.LOCAL.value)
    normalized = raw_mode.strip().lower()
    try:
        return LLMMode(normalized)
    except ValueError as exc:
        raise ValueError(f"不支持的 LLM_MODE: {raw_mode}，仅支持 local 或 cloud") from exc


def default_model_for_mode(mode: LLMMode | str, *, prefer_env: bool = True) -> str:
    """
    获取某个模式下的默认模型

    Args:
        mode: 运行模式
        prefer_env: 是否优先读取环境变量中的默认模型

    Returns:
        默认模型名称
    """
    resolved_mode = resolve_mode(mode)
    if prefer_env:
        load_env()

    if resolved_mode == LLMMode.LOCAL:
        return _first_non_empty(os.getenv("OLLAMA_DEFAULT_MODEL", ""), DEFAULT_MODEL_LOCAL)
    return _first_non_empty(
        os.getenv("CLOUD_DEFAULT_MODEL", ""),
        os.getenv("DEEPSEEK_DEFAULT_MODEL", ""),
        DEFAULT_MODEL_CLOUD,
    )


def resolve_runtime_config(
    *,
    mode: LLMMode | str | None = None,
    prefer_dotenv: bool = False,
) -> LLMRuntimeConfig:
    """
    解析完整运行配置

    Args:
        mode: 运行模式，不传则从环境变量读取
        prefer_dotenv: 是否优先以 .env 覆盖当前环境变量

    Returns:
        运行时配置对象

    Raises:
        RuntimeError: cloud 模式下缺失 API Key 时抛出
    """
    load_env(override=prefer_dotenv)
    resolved_mode = resolve_mode(mode)

    if resolved_mode == LLMMode.LOCAL:
        return LLMRuntimeConfig(
            mode=resolved_mode,
            base_url=_first_non_empty(os.getenv("OLLAMA_BASE_URL", ""), "http://localhost:11434/v1"),
            api_key=_first_non_empty(os.getenv("OLLAMA_API_KEY", ""), "ollama"),
            default_model=default_model_for_mode(resolved_mode, prefer_env=False),
        )

    api_key = _first_non_empty(
        os.getenv("CLOUD_API_KEY", ""),
        os.getenv("DEEPSEEK_API_KEY", ""),
        os.getenv("OPENAI_API_KEY", ""),
    )
    if not api_key:
        raise RuntimeError("cloud 模式未配置 API_KEY，请设置 CLOUD_API_KEY / DEEPSEEK_API_KEY / OPENAI_API_KEY")

    return LLMRuntimeConfig(
        mode=resolved_mode,
        base_url=_first_non_empty(
            os.getenv("CLOUD_BASE_URL", ""),
            os.getenv("DEEPSEEK_BASE_URL", ""),
            os.getenv("BASE_URL", ""),
            "https://api.deepseek.com/v1",
        ),
        api_key=api_key,
        default_model=default_model_for_mode(resolved_mode, prefer_env=False),
    )


def _validate_messages(messages: Sequence[LLMMessage]) -> list[LLMMessage]:
    """
    校验消息列表

    Args:
        messages: 消息序列

    Returns:
        校验后的消息列表（浅拷贝）

    Raises:
        ValueError: 消息为空、字段缺失或类型错误时抛出
    """
    if not messages:
        raise ValueError("messages 不能为空")

    allowed_roles = {"system", "user", "assistant", "developer", "tool"}
    checked: list[LLMMessage] = []
    for index, message in enumerate(messages, start=1):
        role = message.get("role")
        content = message.get("content")
        if role not in allowed_roles:
            raise ValueError(f"第 {index} 条消息 role 非法：{role}")
        if not isinstance(content, str):
            raise ValueError(f"第 {index} 条消息 content 必须为字符串")
        checked.append({"role": role, "content": content})
    return checked


def _extract_answer(response: Any) -> str:
    """
    从响应对象中提取文本回答

    Args:
        response: OpenAI SDK 响应对象

    Returns:
        文本回答

    Raises:
        RuntimeError: 响应缺少 choices 时抛出
    """
    choices = getattr(response, "choices", None)
    if not choices:
        raise RuntimeError("模型未返回可用回答")

    content = getattr(choices[0].message, "content", "")
    if content is None:
        return ""
    if isinstance(content, str):
        return content

    # 针对少数返回分段内容的情况，尽量拼接文本块
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                parts.append(text)
                continue
            if isinstance(block, dict):
                block_text = block.get("text")
                if isinstance(block_text, str):
                    parts.append(block_text)
        return "".join(parts)
    return str(content)


def chat(
    messages: Sequence[LLMMessage],
    *,
    model_name: str | None = None,
    mode: LLMMode | str | None = None,
    prefer_dotenv: bool = False,
) -> LLMChatResult:
    """
    多轮对话统一入口

    Args:
        messages: 消息列表，格式 [{"role": "...", "content": "..."}]
        model_name: 指定模型名，不传则使用当前模式默认模型
        mode: local/cloud，不传则从 LLM_MODE 读取
        prefer_dotenv: 是否优先使用 .env 覆盖现有环境变量

    Returns:
        结构化调用结果
    """
    checked_messages = _validate_messages(messages)
    runtime = resolve_runtime_config(mode=mode, prefer_dotenv=prefer_dotenv)
    final_model = _first_non_empty(model_name or "", runtime.default_model)

    client = OpenAI(api_key=runtime.api_key, base_url=runtime.base_url)
    response = client.chat.completions.create(
        model=final_model,
        messages=cast(Iterable[ChatCompletionMessageParam], checked_messages),
        stream=False,
    )

    return LLMChatResult(
        model=final_model,
        base_url=runtime.base_url,
        answer=_extract_answer(response),
        usage=getattr(response, "usage", None),
    )


def ask(
    text: str,
    *,
    model_name: str | None = None,
    mode: LLMMode | str | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    prefer_dotenv: bool = False,
) -> LLMChatResult:
    """
    单轮问答统一入口

    Args:
        text: 用户输入文本
        model_name: 指定模型名，不传则使用当前模式默认模型
        mode: local/cloud，不传则从 LLM_MODE 读取
        system_prompt: 系统提示词
        prefer_dotenv: 是否优先使用 .env 覆盖现有环境变量

    Returns:
        结构化调用结果

    Raises:
        ValueError: 输入为空时抛出
    """
    if not text or not text.strip():
        raise ValueError("输入文本不能为空")

    return chat(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        model_name=model_name,
        mode=mode,
        prefer_dotenv=prefer_dotenv,
    )

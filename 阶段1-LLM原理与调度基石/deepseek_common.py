"""
@Author: li
@Email: lijianqiao2906@live.com
@FileName: deepseek_common.py
@DateTime: 2026/03/03 10:55:00
@Docs: DeepSeek 通用能力封装，仅包含环境变量加载与问答调用
"""

import os
from pathlib import Path
from typing import Any

from openai import OpenAI


def _find_dotenv(filename: str = ".env") -> Path:
    """从脚本所在目录向上逐级查找 `.env`，最多查找 2 层。

    这样无论从项目根目录还是子目录运行，都能找到正确的 `.env`。
    """
    current = Path(__file__).resolve().parent
    for _ in range(2):
        candidate = current / filename
        if candidate.exists():
            return candidate
        current = current.parent
    return Path(filename)


def load_dotenv_file(dotenv_path: str = ".env", overwrite: bool = False) -> None:
    """轻量加载 `.env` 到环境变量。

    Args:
        dotenv_path: `.env` 文件名或绝对路径；传入文件名时自动向上查找
        overwrite: 是否覆盖已有同名环境变量
    """

    p = Path(dotenv_path)
    env_file = p if p.is_absolute() else _find_dotenv(dotenv_path)
    if not env_file.exists():
        return

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")

        if overwrite or key not in os.environ:
            os.environ[key] = value


def ask_deepseek(
    text: str,
    *,
    model_name: str = "deepseek-chat",
    prefer_dotenv: bool = False,
    system_prompt: str = "你是一个专业、简洁的 AI 助手。",
) -> dict[str, Any]:
    """调用 DeepSeek 问答接口。

    Args:
        text: 用户输入文本
        model_name: 模型名称
        prefer_dotenv: 是否使用 `.env` 覆盖当前环境变量
        system_prompt: 系统提示词

    Returns:
        包含模型信息、接口地址、回答文本与 usage 的字典

    Raises:
        RuntimeError: 当缺少 API Key 时抛出
    """

    load_dotenv_file(overwrite=prefer_dotenv)

    base_url = os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("BASE_URL") or "https://api.deepseek.com/v1"
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY，请在 .env 或系统环境变量中配置")

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        stream=False,
    )

    answer = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)

    return {
        "model": model_name,
        "base_url": base_url,
        "answer": answer,
        "usage": usage,
    }


def ask_ollama(
    text: str,
    *,
    model_name: str = "qwen3:4b",
    prefer_dotenv: bool = False,
    system_prompt: str = "你是一个专业、简洁的 AI 助手。",
) -> dict[str, Any]:
    """调用本地 Ollama 问答接口，不产生费用。

    使用 OLLAMA_BASE_URL、OLLAMA_API_KEY，与 DeepSeek 配置互不干扰。

    Args:
        text: 用户输入文本
        model_name: Ollama 模型名，如 gemma3:4b、qwen3:4b、deepseek-r1:1.5b
        prefer_dotenv: 是否使用 .env 覆盖当前环境变量
        system_prompt: 系统提示词
    """
    load_dotenv_file(overwrite=prefer_dotenv)

    base_url = os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434/v1"
    api_key = os.environ.get("OLLAMA_API_KEY") or "ollama"

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": text},
        ],
        stream=False,
    )

    answer = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)

    return {
        "model": model_name,
        "base_url": base_url,
        "answer": answer,
        "usage": usage,
    }


def ask_ollama_messages(
    messages: list[dict[str, Any]],
    *,
    model_name: str = "qwen3:4b",
    prefer_dotenv: bool = False,
) -> dict[str, Any]:
    """支持多轮对话的 Ollama 调用，messages 格式同 OpenAI API。"""
    load_dotenv_file(overwrite=prefer_dotenv)

    base_url = os.environ.get("OLLAMA_BASE_URL") or "http://localhost:11434/v1"
    api_key = os.environ.get("OLLAMA_API_KEY") or "ollama"

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,  # type: ignore[arg-type]
        stream=False,
    )

    answer = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)

    return {
        "model": model_name,
        "base_url": base_url,
        "answer": answer,
        "usage": usage,
    }


def ask_deepseek_messages(
    messages: list[dict[str, Any]],
    *,
    model_name: str = "deepseek-chat",
    prefer_dotenv: bool = False,
) -> dict[str, Any]:
    """支持多轮对话的 DeepSeek 调用，messages 格式同 OpenAI API。

    Args:
        messages: 消息列表，每项需包含 role 与 content，如
                  [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}]
        model_name: 模型名称
        prefer_dotenv: 是否使用 `.env` 覆盖当前环境变量

    Returns:
        包含 model、base_url、answer、usage 的字典
    """
    load_dotenv_file(overwrite=prefer_dotenv)

    base_url = os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("BASE_URL") or "https://api.deepseek.com/v1"
    api_key = os.environ.get("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError("未设置 DEEPSEEK_API_KEY，请在 .env 或系统环境变量中配置")

    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model_name,
        messages=messages,  # type: ignore[arg-type]
        stream=False,
    )

    answer = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)

    return {
        "model": model_name,
        "base_url": base_url,
        "answer": answer,
        "usage": usage,
    }

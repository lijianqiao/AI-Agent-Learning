"""
@Author: li
@Email: lijianqiao2906@live.com
@FileName: token_count.py
@DateTime: 2026/03/03 10:30:00
@Docs: 通过 DeepSeek API 完成文本问答并统计输入输出 Token、压缩比与费用

Token 计数策略（优先级从高到低）：
  1. API 返回的 usage.prompt_tokens（精确，以此为准）
  2a. [GPT 模型]   tiktoken 本地精确计数
  2b. [DeepSeek]   官方分词器（transformers.AutoTokenizer，需安装 transformers）
  3.  [DeepSeek 兜底] 官方字符比例估算：中文字符 × 0.6，其余字符 × 0.3
"""

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

# 确保同级目录下的模块可导入，不依赖运行时 CWD
sys.path.insert(0, str(Path(__file__).parent))

import deepseek_common

# ── 分词器资源初始化 ──────────────────────────────────────────────────────────

# 官方 DeepSeek 分词器目录（项目根目录下）
_TOKENIZER_DIR = Path(__file__).resolve().parent.parent / "deepseek_v3_tokenizer"


def _load_deepseek_tokenizer():
    """尝试加载官方 DeepSeek 分词器，失败时静默返回 None。"""
    try:
        from transformers import AutoTokenizer  # type: ignore

        if _TOKENIZER_DIR.exists():
            return AutoTokenizer.from_pretrained(str(_TOKENIZER_DIR), trust_remote_code=True)
    except Exception:
        pass
    return None


# 模块加载时各初始化一次，避免每次调用重复加载
_deepseek_tokenizer = _load_deepseek_tokenizer()


def _get_tiktoken_encoder(encoding_name: str):
    """获取 tiktoken 编码器，失败时抛出清晰错误。"""
    try:
        import tiktoken  # type: ignore

        return tiktoken.get_encoding(encoding_name)
    except ImportError as e:
        raise ImportError("使用 GPT 模型需要安装 tiktoken：pip install tiktoken") from e


# ── 数据类定义 ────────────────────────────────────────────────────────────────

TokenizerType = Literal["tiktoken", "deepseek", "ollama"]


@dataclass(frozen=True)
class ModelPricing:
    """模型价格配置。

    Attributes:
        input_cache_hit_per_1m: 输入缓存命中单价（每百万 Token）
        input_cache_miss_per_1m: 输入缓存未命中单价（每百万 Token）
        output_per_1m: 输出单价（每百万 Token）
        currency: 计费货币（CNY / USD）
    """

    input_cache_hit_per_1m: float
    input_cache_miss_per_1m: float
    output_per_1m: float
    currency: str = "CNY"


@dataclass(frozen=True)
class ModelConfig:
    """模型配置。

    Attributes:
        model: 模型名称
        tokenizer_type: 本地计数策略，"tiktoken" 用于 GPT 系列，"deepseek" 用于 DeepSeek 系列
        pricing: 价格配置
        tiktoken_encoding: tiktoken 编码名称，tokenizer_type="tiktoken" 时必填
    """

    model: str
    tokenizer_type: TokenizerType
    pricing: ModelPricing
    tiktoken_encoding: str | None = field(default=None)


# ── 支持的模型列表 ────────────────────────────────────────────────────────────

SUPPORTED_MODELS: dict[str, ModelConfig] = {
    # ── DeepSeek 系列 ──────────────────────────────────────────────
    "deepseek-chat": ModelConfig(
        model="deepseek-chat",
        tokenizer_type="deepseek",
        pricing=ModelPricing(
            input_cache_hit_per_1m=0.2,
            input_cache_miss_per_1m=2.0,
            output_per_1m=3.0,
            currency="CNY",
        ),
    ),
    # ── OpenAI GPT 系列 ────────────────────────────────────────────
    # tiktoken 编码参考：https://platform.openai.com/tokenizer
    # cl100k_base：GPT-4 / GPT-3.5-turbo
    # o200k_base ：GPT-4o / GPT-4o-mini
    "gpt-4o": ModelConfig(
        model="gpt-4o",
        tokenizer_type="tiktoken",
        tiktoken_encoding="o200k_base",
        pricing=ModelPricing(
            input_cache_hit_per_1m=1.25,
            input_cache_miss_per_1m=2.50,
            output_per_1m=10.00,
            currency="USD",
        ),
    ),
    "gpt-4o-mini": ModelConfig(
        model="gpt-4o-mini",
        tokenizer_type="tiktoken",
        tiktoken_encoding="o200k_base",
        pricing=ModelPricing(
            input_cache_hit_per_1m=0.075,
            input_cache_miss_per_1m=0.15,
            output_per_1m=0.60,
            currency="USD",
        ),
    ),
    "gpt-4-turbo": ModelConfig(
        model="gpt-4-turbo",
        tokenizer_type="tiktoken",
        tiktoken_encoding="cl100k_base",
        pricing=ModelPricing(
            input_cache_hit_per_1m=10.0,
            input_cache_miss_per_1m=10.0,
            output_per_1m=30.0,
            currency="USD",
        ),
    ),
    # ── Ollama 本地模型（不产生费用）─────────────────────────────────────
    # Token 计数使用字符比例估算；Ollama API 若返回 usage 则以 API 为准
    "ollama": ModelConfig(
        model="ollama",
        tokenizer_type="ollama",
        pricing=ModelPricing(0.0, 0.0, 0.0, "N/A"),
    ),
    "qwen3:4b": ModelConfig(
        model="qwen3:4b",
        tokenizer_type="ollama",
        pricing=ModelPricing(0.0, 0.0, 0.0, "N/A"),
    ),
    "deepseek-r1:1.5b": ModelConfig(
        model="deepseek-r1:1.5b",
        tokenizer_type="ollama",
        pricing=ModelPricing(0.0, 0.0, 0.0, "N/A"),
    ),
    "gemma3:4b": ModelConfig(
        model="gemma3:4b",
        tokenizer_type="ollama",
        pricing=ModelPricing(0.0, 0.0, 0.0, "N/A"),
    ),
}


# ── 核心函数 ──────────────────────────────────────────────────────────────────


def get_model_config(model_name: str, fallback_ollama: bool = False) -> ModelConfig:
    """获取模型配置。

    Args:
        model_name: 模型名称
        fallback_ollama: 未命中时是否回退到 Ollama 配置（用于任意 Ollama 模型名）

    Raises:
        ValueError: 当模型不在支持列表中且 fallback_ollama=False 时抛出
    """
    if model_name in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[model_name]
    if fallback_ollama:
        return SUPPORTED_MODELS["ollama"]
    raise ValueError(f"不支持的模型: {model_name}，支持: {', '.join(SUPPORTED_MODELS.keys())}")


def count_tokens_local(text: str, config: ModelConfig) -> tuple[int, str]:
    """根据模型配置本地估算 Token 数，返回 (token_count, source)。

    source 说明计数来源：
      - "tiktoken"           ：GPT 系列，tiktoken 精确计数
      - "official_tokenizer" ：DeepSeek，官方分词器精确计数
      - "ratio_estimate"     ：DeepSeek 兜底，按官方比例估算

    注意：API 返回的 usage.prompt_tokens 比任何本地计数都准确，
    本函数仅用于 API 不返回 usage 时的兜底估算。
    """
    if config.tokenizer_type == "tiktoken":
        enc = _get_tiktoken_encoder(config.tiktoken_encoding or "cl100k_base")
        return len(enc.encode(text)), "tiktoken"

    if config.tokenizer_type == "ollama":
        chinese_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
        other_count = len(text) - chinese_count
        estimated = max(1, round(chinese_count * 0.6 + other_count * 0.3))
        return estimated, "ratio_estimate"

    # DeepSeek 策略
    if _deepseek_tokenizer is not None:
        return len(_deepseek_tokenizer.encode(text)), "official_tokenizer"

    # DeepSeek 官方比例兜底：1 中文字符 ≈ 0.6 token，1 其他字符 ≈ 0.3 token
    chinese_count = sum(1 for c in text if "\u4e00" <= c <= "\u9fff")
    other_count = len(text) - chinese_count
    estimated = max(1, round(chinese_count * 0.6 + other_count * 0.3))
    return estimated, "ratio_estimate"


def calculate_token_metrics(
    question: str,
    answer: str,
    usage: Any,
    *,
    model_name: str = "deepseek-chat",
    fallback_ollama: bool = False,
) -> dict[str, Any]:
    """计算输入输出 Token、压缩比和费用。

    Args:
        question: 用户输入文本
        answer: 模型回答文本
        usage: API 返回的 usage 对象（可能为 None）
        model_name: 模型名称
        fallback_ollama: 模型未配置时是否回退到 Ollama 配置（用于任意 Ollama 模型名）

    Returns:
        包含输入输出 token、压缩比、费用与 tokenizer 来源的字典
    """
    config = get_model_config(model_name, fallback_ollama=fallback_ollama)

    if usage is not None:
        # 优先使用 API 返回的精确值
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        prompt_cache_hit_tokens = int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
        prompt_cache_miss_tokens = int(getattr(usage, "prompt_cache_miss_tokens", 0) or 0)
        if prompt_cache_hit_tokens + prompt_cache_miss_tokens == 0:
            prompt_cache_miss_tokens = input_tokens
        tokenizer_source = "api_usage"
    else:
        # API 无 usage 时本地兜底
        input_tokens, tokenizer_source = count_tokens_local(question, config)
        output_tokens, _ = count_tokens_local(answer, config)
        prompt_cache_hit_tokens = 0
        prompt_cache_miss_tokens = input_tokens

    pricing = config.pricing
    input_cost = (prompt_cache_hit_tokens / 1_000_000) * pricing.input_cache_hit_per_1m + (
        prompt_cache_miss_tokens / 1_000_000
    ) * pricing.input_cache_miss_per_1m
    output_cost = (output_tokens / 1_000_000) * pricing.output_per_1m
    total_cost = input_cost + output_cost

    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "tokenizer_source": tokenizer_source,
        "compression_ratio": round(len(question) / input_tokens, 2) if input_tokens else None,
        "pricing_currency": pricing.currency,
        "estimated_input_cost": round(input_cost, 6),
        "estimated_output_cost": round(output_cost, 6),
        "estimated_total_cost": round(total_cost, 6),
    }


def run_deepseek_and_count(
    text: str,
    *,
    model_name: str = "deepseek-chat",
    prefer_dotenv: bool = False,
) -> dict[str, Any]:
    """执行 DeepSeek 问答并输出完整统计结果。"""
    qa_result = deepseek_common.ask_deepseek(
        text,
        model_name=model_name,
        prefer_dotenv=prefer_dotenv,
    )
    metrics = calculate_token_metrics(
        question=text,
        answer=qa_result["answer"],
        usage=qa_result["usage"],
        model_name=model_name,
    )

    return {
        "model": qa_result["model"],
        "base_url": qa_result["base_url"],
        "question": text,
        "answer": qa_result["answer"],
        **metrics,
    }


def run_ollama_and_count(
    text: str,
    *,
    model_name: str = "llama3.2",
    prefer_dotenv: bool = False,
) -> dict[str, Any]:
    """执行 Ollama 本地问答并输出完整统计结果，不产生费用。"""
    qa_result = deepseek_common.ask_ollama(
        text,
        model_name=model_name,
        prefer_dotenv=prefer_dotenv,
    )
    metrics = calculate_token_metrics(
        question=text,
        answer=qa_result["answer"],
        usage=qa_result["usage"],
        model_name=model_name,
        fallback_ollama=True,
    )

    return {
        "model": qa_result["model"],
        "base_url": qa_result["base_url"],
        "question": text,
        "answer": qa_result["answer"],
        **metrics,
    }


def main() -> None:
    """命令行入口函数。

    示例：
        # 默认使用 qwen3:4b
        uv run token_count.py --provider ollama --text "介绍一下 KV-Cache"

        # 指定其他本地模型
        uv run token_count.py --provider ollama --model deepseek-r1:1.5b --text "你好"
        uv run token_count.py --provider ollama --model gemma3:4b --text "你好"
    """
    import argparse

    parser = argparse.ArgumentParser(description="LLM 问答 + Token 统计（支持 DeepSeek / Ollama / GPT）")
    parser.add_argument("--provider", choices=["deepseek", "ollama"], default="deepseek")
    parser.add_argument("--model", default=None, help="模型名称；ollama 默认 qwen3:4b")
    parser.add_argument("--text", default=None, help="输入文本；不传则进入交互输入")
    parser.add_argument("--prefer-dotenv", action="store_true", help="使用 .env 覆盖当前环境变量")
    args = parser.parse_args()

    model = args.model or ("qwen3:4b" if args.provider == "ollama" else "deepseek-chat")

    text = (args.text or "").strip()
    if not text:
        text = input("请输入文本：").strip()
    if not text:
        raise ValueError("输入文本不能为空")

    if args.provider == "ollama":
        result = run_ollama_and_count(text, model_name=model, prefer_dotenv=args.prefer_dotenv)
    else:
        result = run_deepseek_and_count(text, model_name=model, prefer_dotenv=args.prefer_dotenv)

    print("\n=== 回答结果 ===")
    print(result["answer"])

    print("\n=== Token 统计 ===")
    print(f"模型：{result['model']}")
    print(f"接口地址：{result['base_url']}")
    print(f"Token 计数来源：{result['tokenizer_source']}")
    print(f"输入 Token：{result['input_tokens']}")
    print(f"输出 Token：{result['output_tokens']}")
    print(f"压缩比（字符/Token）：{result['compression_ratio']}")
    currency = result["pricing_currency"]
    print(f"预计输入费用（{currency}）：{result['estimated_input_cost']}")
    print(f"预计输出费用（{currency}）：{result['estimated_output_cost']}")
    print(f"预计总费用（{currency}）：{result['estimated_total_cost']}")


if __name__ == "__main__":
    main()

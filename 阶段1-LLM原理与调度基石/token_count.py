"""
@Author: li
@Email: lijianqiao2906@live.com
@FileName: token_count.py
@DateTime: 2026/03/03 14:45:00
@Docs: LLM 统一问答 + Token 统计 + 费用估算
"""

import argparse
from dataclasses import dataclass
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any

import llm_common


class TokenizerStrategy(StrEnum):
    """
    分词策略

    TIKTOKEN: GPT 系列本地精确计数
    DEEPSEEK: DeepSeek 官方分词器优先，失败时比例估算
    RATIO: 字符比例估算（常用于本地模型）
    """

    TIKTOKEN = "tiktoken"
    DEEPSEEK = "deepseek"
    RATIO = "ratio"


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """
    模型计费配置

    Attributes:
        input_cache_hit_per_1m: 输入缓存命中价格（每百万 Token）
        input_cache_miss_per_1m: 输入缓存未命中价格（每百万 Token）
        output_per_1m: 输出价格（每百万 Token）
        currency: 货币单位
    """

    input_cache_hit_per_1m: float
    input_cache_miss_per_1m: float
    output_per_1m: float
    currency: str

    @classmethod
    def free(cls) -> "ModelPricing":
        """构造免费模型计费配置。"""
        return cls(0.0, 0.0, 0.0, "N/A")


@dataclass(frozen=True, slots=True)
class ModelSpec:
    """
    模型配置

    Attributes:
        model: 模型名
        tokenizer_strategy: 分词策略
        pricing: 计费配置
        tiktoken_encoding: tiktoken 编码名（仅 TIKTOKEN 策略使用）
    """

    model: str
    tokenizer_strategy: TokenizerStrategy
    pricing: ModelPricing
    tiktoken_encoding: str | None = None


@dataclass(frozen=True, slots=True)
class TokenCounter:
    """
    Token 计数器

    Attributes:
        model_name: 实际模型名
        mode: 运行模式
        spec: 模型配置
    """

    model_name: str
    mode: llm_common.LLMMode
    spec: ModelSpec

    def count(self, text: str) -> tuple[int, str]:
        """
        统计文本 Token 数

        Args:
            text: 输入文本

        Returns:
            (token 数, 计数来源)
        """
        return count_tokens_with_spec(text, self.spec)


@dataclass(frozen=True, slots=True)
class TokenMetrics:
    """
    Token 统计结果

    Attributes:
        input_tokens: 输入 Token
        output_tokens: 输出 Token
        tokenizer_source: 计数来源
        compression_ratio: 字符/输入 Token 压缩比
        pricing_currency: 货币单位
        estimated_input_cost: 输入费用估算
        estimated_output_cost: 输出费用估算
        estimated_total_cost: 总费用估算
    """

    input_tokens: int
    output_tokens: int
    tokenizer_source: str
    compression_ratio: float | None
    pricing_currency: str
    estimated_input_cost: float
    estimated_output_cost: float
    estimated_total_cost: float

    def as_dict(self) -> dict[str, Any]:
        """转换为字典。"""
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "tokenizer_source": self.tokenizer_source,
            "compression_ratio": self.compression_ratio,
            "pricing_currency": self.pricing_currency,
            "estimated_input_cost": self.estimated_input_cost,
            "estimated_output_cost": self.estimated_output_cost,
            "estimated_total_cost": self.estimated_total_cost,
        }


@dataclass(frozen=True, slots=True)
class AskAndCountResult:
    """
    问答 + 统计聚合结果

    Attributes:
        mode: 运行模式
        model: 实际模型
        base_url: 调用地址
        question: 提问
        answer: 回答
        metrics: Token 统计结果
    """

    mode: llm_common.LLMMode
    model: str
    base_url: str
    question: str
    answer: str
    metrics: TokenMetrics

    def as_dict(self) -> dict[str, Any]:
        """转换为字典。"""
        data = {
            "mode": self.mode.value,
            "model": self.model,
            "base_url": self.base_url,
            "question": self.question,
            "answer": self.answer,
        }
        data.update(self.metrics.as_dict())
        return data


DEEPSEEK_TOKENIZER_DIR = Path(__file__).resolve().parent.parent / "deepseek_v3_tokenizer"

MODEL_SPECS: dict[str, ModelSpec] = {
    "deepseek-chat": ModelSpec(
        model="deepseek-chat",
        tokenizer_strategy=TokenizerStrategy.DEEPSEEK,
        pricing=ModelPricing(0.2, 2.0, 3.0, "CNY"),
    ),
    "gpt-4o": ModelSpec(
        model="gpt-4o",
        tokenizer_strategy=TokenizerStrategy.TIKTOKEN,
        pricing=ModelPricing(1.25, 2.50, 10.0, "USD"),
        tiktoken_encoding="o200k_base",
    ),
    "gpt-4o-mini": ModelSpec(
        model="gpt-4o-mini",
        tokenizer_strategy=TokenizerStrategy.TIKTOKEN,
        pricing=ModelPricing(0.075, 0.15, 0.60, "USD"),
        tiktoken_encoding="o200k_base",
    ),
    "gpt-4-turbo": ModelSpec(
        model="gpt-4-turbo",
        tokenizer_strategy=TokenizerStrategy.TIKTOKEN,
        pricing=ModelPricing(10.0, 10.0, 30.0, "USD"),
        tiktoken_encoding="cl100k_base",
    ),
    "qwen3:4b": ModelSpec("qwen3:4b", TokenizerStrategy.RATIO, ModelPricing.free()),
    "deepseek-r1:1.5b": ModelSpec("deepseek-r1:1.5b", TokenizerStrategy.RATIO, ModelPricing.free()),
    "gemma3:4b": ModelSpec("gemma3:4b", TokenizerStrategy.RATIO, ModelPricing.free()),
}

LOCAL_GENERIC_SPEC = ModelSpec("local-generic", TokenizerStrategy.RATIO, ModelPricing.free())


def _safe_int(value: Any) -> int:
    """安全转换为整数，非法值返回 0。"""
    try:
        return int(value or 0)
    except (TypeError, ValueError):
        return 0


def _ratio_token_count(text: str) -> int:
    """
    按字符比例估算 Token

    规则：
      - 中文字符约 0.6 Token
      - 其他字符约 0.3 Token
    """
    if not text:
        return 0
    chinese_count = sum(1 for char in text if "\u4e00" <= char <= "\u9fff")
    other_count = len(text) - chinese_count
    return max(1, round(chinese_count * 0.6 + other_count * 0.3))


@lru_cache(maxsize=4)
def _get_tiktoken_encoder(encoding_name: str):
    """获取并缓存 tiktoken 编码器。"""
    try:
        import tiktoken  # type: ignore
    except ImportError as exc:
        raise ImportError("使用 tiktoken 分词需安装依赖：pip install tiktoken") from exc
    return tiktoken.get_encoding(encoding_name)


@lru_cache(maxsize=1)
def _load_deepseek_tokenizer():
    """尝试加载 DeepSeek 官方分词器，失败返回 None。"""
    try:
        from transformers import AutoTokenizer  # type: ignore
    except ImportError:
        return None

    if not DEEPSEEK_TOKENIZER_DIR.exists():
        return None

    try:
        return AutoTokenizer.from_pretrained(str(DEEPSEEK_TOKENIZER_DIR), trust_remote_code=True)
    except Exception:
        return None


def resolve_model_spec(model_name: str, *, mode: llm_common.LLMMode) -> ModelSpec:
    """
    解析模型配置

    Args:
        model_name: 模型名称
        mode: 运行模式

    Returns:
        模型配置

    Raises:
        ValueError: cloud 模式下模型未注册时抛出
    """
    spec = MODEL_SPECS.get(model_name)
    if spec is not None:
        return spec
    if mode == llm_common.LLMMode.LOCAL:
        return LOCAL_GENERIC_SPEC
    raise ValueError(f"cloud 模式下未注册模型: {model_name}")


def count_tokens_with_spec(text: str, spec: ModelSpec) -> tuple[int, str]:
    """
    根据模型配置统计 Token

    Args:
        text: 输入文本
        spec: 模型配置

    Returns:
        (token 数量, 来源)
    """
    match spec.tokenizer_strategy:
        case TokenizerStrategy.TIKTOKEN:
            encoding_name = spec.tiktoken_encoding or "cl100k_base"
            encoder = _get_tiktoken_encoder(encoding_name)
            return len(encoder.encode(text)), "tiktoken"
        case TokenizerStrategy.DEEPSEEK:
            tokenizer = _load_deepseek_tokenizer()
            if tokenizer is not None:
                return len(tokenizer.encode(text)), "official_tokenizer"
            return _ratio_token_count(text), "ratio_estimate"
        case TokenizerStrategy.RATIO:
            return _ratio_token_count(text), "ratio_estimate"


def create_token_counter(
    *,
    model_name: str | None = None,
    mode: llm_common.LLMMode | str | None = None,
) -> TokenCounter:
    """
    创建可复用 Token 计数器

    Args:
        model_name: 模型名，不传则使用模式默认模型
        mode: local/cloud，不传则读取 LLM_MODE

    Returns:
        TokenCounter 实例
    """
    resolved_mode = llm_common.resolve_mode(mode)
    resolved_model = model_name or llm_common.default_model_for_mode(resolved_mode)
    spec = resolve_model_spec(resolved_model, mode=resolved_mode)
    return TokenCounter(model_name=resolved_model, mode=resolved_mode, spec=spec)


def calculate_token_metrics(
    *,
    question: str,
    answer: str,
    usage: Any,
    counter: TokenCounter,
) -> TokenMetrics:
    """
    计算 Token 指标与费用

    Args:
        question: 输入文本
        answer: 输出文本
        usage: API usage（可能为 None）
        counter: 计数器

    Returns:
        结构化指标对象
    """
    if usage is not None:
        input_tokens = _safe_int(getattr(usage, "prompt_tokens", 0))
        output_tokens = _safe_int(getattr(usage, "completion_tokens", 0))
        cache_hit = _safe_int(getattr(usage, "prompt_cache_hit_tokens", 0))
        cache_miss = _safe_int(getattr(usage, "prompt_cache_miss_tokens", 0))
        if cache_hit + cache_miss == 0:
            cache_miss = input_tokens
        source = "api_usage"
    else:
        input_tokens, source = counter.count(question)
        output_tokens, _ = counter.count(answer)
        cache_hit = 0
        cache_miss = input_tokens

    pricing = counter.spec.pricing
    input_cost = (cache_hit / 1_000_000) * pricing.input_cache_hit_per_1m + (
        cache_miss / 1_000_000
    ) * pricing.input_cache_miss_per_1m
    output_cost = (output_tokens / 1_000_000) * pricing.output_per_1m
    total_cost = input_cost + output_cost

    return TokenMetrics(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        tokenizer_source=source,
        compression_ratio=round(len(question) / input_tokens, 2) if input_tokens else None,
        pricing_currency=pricing.currency,
        estimated_input_cost=round(input_cost, 6),
        estimated_output_cost=round(output_cost, 6),
        estimated_total_cost=round(total_cost, 6),
    )


def ask_and_count(
    text: str,
    *,
    mode: llm_common.LLMMode | str | None = None,
    model_name: str | None = None,
    prefer_dotenv: bool = False,
) -> AskAndCountResult:
    """
    执行问答并输出 Token 统计

    Args:
        text: 输入文本
        mode: local/cloud，不传则读取 LLM_MODE
        model_name: 模型名，不传则使用模式默认模型
        prefer_dotenv: 是否优先使用 .env 覆盖环境变量

    Returns:
        聚合结果对象
    """
    counter = create_token_counter(model_name=model_name, mode=mode)
    qa = llm_common.ask(
        text,
        model_name=counter.model_name,
        mode=counter.mode,
        prefer_dotenv=prefer_dotenv,
    )
    metrics = calculate_token_metrics(
        question=text,
        answer=qa.answer,
        usage=qa.usage,
        counter=counter,
    )
    return AskAndCountResult(
        mode=counter.mode,
        model=qa.model,
        base_url=qa.base_url,
        question=text,
        answer=qa.answer,
        metrics=metrics,
    )


def _print_result(result: AskAndCountResult) -> None:
    """打印结果到终端。"""
    print("\n=== 回答结果 ===")
    print(result.answer)

    print("\n=== Token 统计 ===")
    print(f"模式：{result.mode.value}")
    print(f"模型：{result.model}")
    print(f"接口地址：{result.base_url}")
    print(f"Token 计数来源：{result.metrics.tokenizer_source}")
    print(f"输入 Token：{result.metrics.input_tokens}")
    print(f"输出 Token：{result.metrics.output_tokens}")
    print(f"压缩比（字符/Token）：{result.metrics.compression_ratio}")
    print(f"预计输入费用（{result.metrics.pricing_currency}）：{result.metrics.estimated_input_cost}")
    print(f"预计输出费用（{result.metrics.pricing_currency}）：{result.metrics.estimated_output_cost}")
    print(f"预计总费用（{result.metrics.pricing_currency}）：{result.metrics.estimated_total_cost}")


def main() -> None:
    """
    命令行入口

    示例:
        uv run token_count.py --mode local --text "你好"
        uv run token_count.py --mode cloud --model deepseek-chat --text "介绍注意力机制"
    """
    parser = argparse.ArgumentParser(description="LLM 问答 + Token 统计（local/cloud）")
    parser.add_argument("--mode", choices=["local", "cloud"], default=None, help="运行模式，默认读取 LLM_MODE")
    parser.add_argument("--model", default=None, help="模型名称，不传时按模式使用默认模型")
    parser.add_argument("--text", default=None, help="输入文本，不传则进入交互输入")
    parser.add_argument("--prefer-dotenv", action="store_true", help="是否优先使用 .env 覆盖环境变量")
    args = parser.parse_args()

    text = (args.text or "").strip()
    if not text:
        text = input("请输入文本：").strip()
    if not text:
        raise ValueError("输入文本不能为空")

    result = ask_and_count(
        text,
        mode=args.mode,
        model_name=args.model,
        prefer_dotenv=args.prefer_dotenv,
    )
    _print_result(result)


if __name__ == "__main__":
    main()

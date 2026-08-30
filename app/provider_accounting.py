"""Provider-neutral generation usage, latency, and cost accounting.

The provider response owns token counts; AtlasRAG does not pretend whitespace or
character counts are tokenizer-equivalent. Cost is stored in integer micro-USD
to avoid floating-point accounting drift.
"""

from collections.abc import Callable
from dataclasses import dataclass
from time import perf_counter
from typing import Protocol

from app.models import EvidenceChunk


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    text: str
    model: str
    input_tokens: int
    output_tokens: int

    def __post_init__(self) -> None:
        if not self.model.strip():
            raise ValueError("provider model must not be empty")
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("provider token usage cannot be negative")


class ProviderClient(Protocol):
    def generate(self, question: str, evidence: list[EvidenceChunk]) -> ProviderResponse: ...


@dataclass(frozen=True, slots=True)
class ModelPricing:
    """Price per one million tokens, expressed in integer micro-USD."""

    input_micro_usd_per_million_tokens: int
    output_micro_usd_per_million_tokens: int

    def __post_init__(self) -> None:
        if self.input_micro_usd_per_million_tokens < 0:
            raise ValueError("input token price cannot be negative")
        if self.output_micro_usd_per_million_tokens < 0:
            raise ValueError("output token price cannot be negative")

    def estimate_micro_usd(self, *, input_tokens: int, output_tokens: int) -> int:
        if input_tokens < 0 or output_tokens < 0:
            raise ValueError("token usage cannot be negative")
        numerator = (
            input_tokens * self.input_micro_usd_per_million_tokens
            + output_tokens * self.output_micro_usd_per_million_tokens
        )
        return (numerator + 999_999) // 1_000_000


@dataclass(frozen=True, slots=True)
class GenerationUsage:
    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    estimated_cost_micro_usd: int | None
    succeeded: bool


class UsageSink(Protocol):
    def record(self, usage: GenerationUsage) -> None: ...


class InMemoryUsageSink:
    def __init__(self) -> None:
        self.records: list[GenerationUsage] = []

    def record(self, usage: GenerationUsage) -> None:
        self.records.append(usage)


class MeteredAnswerGenerator:
    """AnswerGenerator-compatible adapter with explicit provider accounting."""

    def __init__(
        self,
        *,
        provider_name: str,
        client: ProviderClient,
        sink: UsageSink,
        pricing_by_model: dict[str, ModelPricing] | None = None,
        clock: Callable[[], float] = perf_counter,
    ) -> None:
        if not provider_name.strip():
            raise ValueError("provider_name must not be empty")
        self._provider_name = provider_name
        self._client = client
        self._sink = sink
        self._pricing_by_model = pricing_by_model or {}
        self._clock = clock

    def generate(self, question: str, evidence: list[EvidenceChunk]) -> str:
        started = self._clock()
        try:
            response = self._client.generate(question, evidence)
        except Exception:
            elapsed_ms = max((self._clock() - started) * 1000.0, 0.0)
            self._sink.record(
                GenerationUsage(
                    provider=self._provider_name,
                    model="unknown",
                    input_tokens=0,
                    output_tokens=0,
                    latency_ms=elapsed_ms,
                    estimated_cost_micro_usd=None,
                    succeeded=False,
                )
            )
            raise

        elapsed_ms = max((self._clock() - started) * 1000.0, 0.0)
        pricing = self._pricing_by_model.get(response.model)
        estimated_cost = (
            pricing.estimate_micro_usd(
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
            )
            if pricing is not None
            else None
        )
        self._sink.record(
            GenerationUsage(
                provider=self._provider_name,
                model=response.model,
                input_tokens=response.input_tokens,
                output_tokens=response.output_tokens,
                latency_ms=elapsed_ms,
                estimated_cost_micro_usd=estimated_cost,
                succeeded=True,
            )
        )
        return response.text

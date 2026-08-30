from dataclasses import dataclass

import pytest

from app.models import EvidenceChunk
from app.provider_accounting import (
    InMemoryUsageSink,
    MeteredAnswerGenerator,
    ModelPricing,
    ProviderResponse,
)


@dataclass
class StaticProvider:
    response: ProviderResponse

    def generate(self, question: str, evidence: list[EvidenceChunk]) -> ProviderResponse:
        assert question
        assert evidence
        return self.response


class FailingProvider:
    def generate(self, question: str, evidence: list[EvidenceChunk]) -> ProviderResponse:
        raise RuntimeError("provider unavailable")


def evidence() -> list[EvidenceChunk]:
    return [EvidenceChunk(id="c1", text="grounded evidence", source="doc.md", score=1.0)]


def test_metered_generator_records_provider_reported_tokens_latency_and_cost() -> None:
    sink = InMemoryUsageSink()
    ticks = iter([10.0, 10.125])
    generator = MeteredAnswerGenerator(
        provider_name="example-provider",
        client=StaticProvider(ProviderResponse("answer", "model-a", 800, 200)),
        sink=sink,
        pricing_by_model={
            "model-a": ModelPricing(
                input_micro_usd_per_million_tokens=500_000,
                output_micro_usd_per_million_tokens=2_000_000,
            )
        },
        clock=lambda: next(ticks),
    )

    assert generator.generate("question", evidence()) == "answer"
    usage = sink.records[0]
    assert usage.input_tokens == 800
    assert usage.output_tokens == 200
    assert usage.latency_ms == pytest.approx(125.0)
    assert usage.estimated_cost_micro_usd == 800
    assert usage.succeeded is True


def test_unknown_model_keeps_cost_unknown_instead_of_inventing_price() -> None:
    sink = InMemoryUsageSink()
    ticks = iter([0.0, 0.01])
    generator = MeteredAnswerGenerator(
        provider_name="example-provider",
        client=StaticProvider(ProviderResponse("answer", "unpriced-model", 10, 5)),
        sink=sink,
        pricing_by_model={},
        clock=lambda: next(ticks),
    )

    generator.generate("question", evidence())

    assert sink.records[0].estimated_cost_micro_usd is None


def test_provider_failure_records_failure_without_fake_usage() -> None:
    sink = InMemoryUsageSink()
    ticks = iter([5.0, 5.05])
    generator = MeteredAnswerGenerator(
        provider_name="example-provider",
        client=FailingProvider(),
        sink=sink,
        clock=lambda: next(ticks),
    )

    with pytest.raises(RuntimeError, match="provider unavailable"):
        generator.generate("question", evidence())

    usage = sink.records[0]
    assert usage.succeeded is False
    assert usage.model == "unknown"
    assert usage.input_tokens == 0
    assert usage.output_tokens == 0
    assert usage.estimated_cost_micro_usd is None


def test_pricing_rounds_up_to_micro_usd_and_rejects_negative_values() -> None:
    pricing = ModelPricing(1, 1)
    assert pricing.estimate_micro_usd(input_tokens=1, output_tokens=0) == 1

    with pytest.raises(ValueError):
        ModelPricing(-1, 0)
    with pytest.raises(ValueError):
        ProviderResponse("x", "model", -1, 0)

from pathlib import Path

from app.demo_report import build_demo_report_html, write_demo_report


def test_demo_report_contains_regression_metrics_and_cases() -> None:
    html = build_demo_report_html()

    assert "RAG evaluation" in html
    assert "Citation precision" in html
    assert "Citation recall" in html
    assert "Abstention accuracy" in html
    assert "Supported answers" in html
    assert "What is the passport office opening time?" in html
    assert "abstained" in html
    assert "application regression metrics" in html
    assert "not model-based semantic groundedness scores" in html


def test_demo_report_is_standalone_html(tmp_path: Path) -> None:
    output = write_demo_report(tmp_path / "atlasrag.html")

    assert output.exists()
    content = output.read_text(encoding="utf-8")
    assert content.startswith("<!doctype html>")
    assert "<style>" in content
    assert "rag-regression-v1" in content

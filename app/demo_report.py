from __future__ import annotations

import argparse
from html import escape
from pathlib import Path

from app.models import QueryRequest
from app.query_service import ExtractiveAnswerGenerator, QueryService
from app.rag_evaluation import evaluate_rag_regression
from app.regression_dataset import (
    RAG_REGRESSION_DATASET_PROVENANCE,
    RAG_REGRESSION_DATASET_VERSION,
    RAG_REGRESSION_DOCUMENTS,
    RAG_REGRESSION_EXAMPLES,
)
from app.retrieval import InMemoryRetriever

STYLES = """
:root {
  font-family: Inter, ui-sans-serif, system-ui, sans-serif;
  color: #172033;
  background: #f5f6f8;
}
* { box-sizing: border-box; }
body { margin: 0; }
main { max-width: 1120px; margin: 0 auto; padding: 40px 24px 64px; }
h1 { margin: 5px 0 8px; font-size: clamp(2rem, 6vw, 4rem); }
h2 { margin: 0 0 15px; font-size: 1.15rem; }
.sub { max-width: 820px; color: #647083; line-height: 1.6; }
.note { color: #707a89; font-size: .82rem; line-height: 1.5; }
.cards {
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 12px;
  margin: 24px 0;
}
.card, .panel {
  background: #fff;
  border: 1px solid #dfe4ea;
  border-radius: 14px;
  box-shadow: 0 8px 24px rgba(20, 30, 50, .05);
}
.card { padding: 16px; }
.card span { display: block; color: #707a89; font-size: .75rem; text-transform: uppercase; }
.card strong { display: block; margin-top: 8px; font-size: 1.45rem; }
.panel { padding: 20px; margin-top: 18px; overflow: auto; }
table { width: 100%; border-collapse: collapse; font-size: .88rem; }
th, td {
  padding: 11px 8px;
  border-bottom: 1px solid #edf0f3;
  text-align: left;
  vertical-align: top;
}
th { color: #707a89; font-weight: 600; }
.badge { display: inline-block; padding: 3px 8px; border-radius: 999px; background: #eef2f6; }
.badge-yes { background: #dcfce7; }
.badge-no { background: #fef3c7; }
.answer { max-width: 420px; line-height: 1.45; }
.citations { min-width: 210px; }
.citation { display: block; margin-bottom: 5px; }
@media (max-width: 900px) { .cards { grid-template-columns: 1fr 1fr; } }
@media (max-width: 520px) { .cards { grid-template-columns: 1fr; } }
"""


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def _service() -> QueryService:
    return QueryService(
        InMemoryRetriever(RAG_REGRESSION_DOCUMENTS),
        ExtractiveAnswerGenerator(),
        minimum_evidence_score=0.25,
    )


def build_demo_report_html() -> str:
    service = _service()
    metrics = evaluate_rag_regression(
        service,
        RAG_REGRESSION_DOCUMENTS,
        RAG_REGRESSION_EXAMPLES,
    )

    case_rows: list[str] = []
    for case in RAG_REGRESSION_EXAMPLES:
        response = service.query(QueryRequest(question=case.question, top_k=4))
        citations = (
            "".join(
                f'<span class="citation">{escape(citation.chunk_id)} · {citation.score:.2f}</span>'
                for citation in response.citations
            )
            or "—"
        )
        expected = "abstain" if case.should_abstain else ", ".join(sorted(case.relevant_ids))
        badge_class = "badge-yes" if response.grounded else "badge-no"
        case_rows.append(
            "<tr>"
            f"<td>{escape(case.question)}</td>"
            f"<td>{escape(expected)}</td>"
            f'<td><span class="badge {badge_class}">'
            f"{'grounded' if response.grounded else 'abstained'}</span></td>"
            f'<td class="citations">{citations}</td>'
            f'<td class="answer">{escape(response.answer)}</td>'
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AtlasRAG — Evaluation Demo</title>
<style>{STYLES}</style>
</head>
<body>
<main>
<header>
  <div class="note">{escape(RAG_REGRESSION_DATASET_VERSION)} · DETERMINISTIC REGRESSION DATA</div>
  <h1>RAG evaluation</h1>
  <p class="sub">
    A credential-free regression run showing citation behavior, abstention and answer support.
    Retrieval and generation are deterministic so the same cases can run in CI without an LLM key.
  </p>
</header>
<section class="cards">
  <div class="card"><span>Cases</span><strong>{metrics.evaluated}</strong></div>
  <div class="card">
    <span>Citation precision</span><strong>{_pct(metrics.citation_precision)}</strong>
  </div>
  <div class="card">
    <span>Citation recall</span><strong>{_pct(metrics.citation_recall)}</strong>
  </div>
  <div class="card">
    <span>Abstention accuracy</span><strong>{_pct(metrics.abstention_accuracy)}</strong>
  </div>
  <div class="card">
    <span>Supported answers</span><strong>{_pct(metrics.supported_answer_rate)}</strong>
  </div>
</section>
<section class="panel">
  <h2>Regression cases</h2>
  <table>
    <thead>
      <tr>
        <th>Question</th><th>Expected evidence</th><th>Result</th>
        <th>Citations</th><th>Answer</th>
      </tr>
    </thead>
    <tbody>{"".join(case_rows)}</tbody>
  </table>
</section>
<section class="panel">
  <h2>What these numbers mean</h2>
  <p class="sub">
    Citation metrics compare returned chunk IDs with hand-authored expected evidence.
    Supported-answer rate checks whether the deterministic extractive answer is contained
    in cited evidence. These are application regression metrics, not model-based semantic groundedness scores.
  </p>
  <p class="note">{escape(RAG_REGRESSION_DATASET_PROVENANCE)}</p>
</section>
</main>
</body>
</html>"""


def write_demo_report(path: str | Path) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_demo_report_html(), encoding="utf-8")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate the AtlasRAG evaluation demo")
    parser.add_argument("--output", default="build/atlasrag-evaluation.html")
    args = parser.parse_args()
    print(write_demo_report(args.output))


if __name__ == "__main__":
    main()

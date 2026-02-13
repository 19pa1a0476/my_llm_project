import argparse
import json
import os
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import requests
from sqlalchemy import create_engine, text

from api.app.config import settings

try:
    from deepeval import evaluate
    from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
    from deepeval.test_case import LLMTestCase
except Exception as exc:
    raise RuntimeError("deepeval is required to run this benchmark script.") from exc


@dataclass
class EvalRow:
    sample_id: str
    input_text: str
    expected_output: str
    actual_output: str
    faithfulness_score: float | None
    answer_relevancy_score: float | None
    correctness_score: float | None
    hallucination_flag: bool | None
    latency_ms: int
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float
    metadata: dict


def load_jsonl(path: str) -> list[dict]:
    records: list[dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run_eval(dataset_path: str, api_base_url: str, model_name: str, prompt_version: str, retriever_version: str) -> None:
    run_uuid = uuid.uuid4()
    dataset_name = os.path.basename(dataset_path)

    engine = create_engine(settings.database_url, pool_pre_ping=True, future=True)
    with engine.begin() as conn:
        conn.execute(
            text(
                """
                INSERT INTO eval_runs (
                    run_id, source, dataset_name, model_name, model_version,
                    prompt_version, retriever_version, started_at, status
                ) VALUES (
                    :run_id, 'deepeval', :dataset_name, :model_name, :model_version,
                    :prompt_version, :retriever_version, :started_at, 'running'
                )
                """
            ),
            {
                "run_id": run_uuid,
                "dataset_name": dataset_name,
                "model_name": model_name,
                "model_version": "v0.2",
                "prompt_version": prompt_version,
                "retriever_version": retriever_version,
                "started_at": datetime.now(timezone.utc),
            },
        )

    rows: list[EvalRow] = []
    for idx, row in enumerate(load_jsonl(dataset_path)):
        query = row.get("input") or row.get("prompt") or ""
        expected = row.get("expected_output") or row.get("response") or ""
        context = row.get("context", [])

        started = time.perf_counter()
        response = requests.get(f"{api_base_url}/query", params={"q": query}, timeout=120)
        response.raise_for_status()
        payload = response.json()
        latency_ms = int((time.perf_counter() - started) * 1000)

        actual_output = payload.get("answer", "")
        test_case = LLMTestCase(
            input=query,
            expected_output=expected,
            actual_output=actual_output,
            retrieval_context=context if isinstance(context, list) else [str(context)],
        )
        faithfulness_metric = FaithfulnessMetric()
        answer_relevancy_metric = AnswerRelevancyMetric()
        evaluate([test_case], [faithfulness_metric, answer_relevancy_metric], print_results=False)

        faithfulness = getattr(faithfulness_metric, "score", None)
        relevancy = getattr(answer_relevancy_metric, "score", None)
        correctness = relevancy
        hallucination = bool(faithfulness is not None and faithfulness < 0.5)

        rows.append(
            EvalRow(
                sample_id=str(row.get("id", idx)),
                input_text=query,
                expected_output=expected,
                actual_output=actual_output,
                faithfulness_score=faithfulness,
                answer_relevancy_score=relevancy,
                correctness_score=correctness,
                hallucination_flag=hallucination,
                latency_ms=payload.get("latency_ms", latency_ms),
                input_tokens=payload.get("input_tokens", 0),
                output_tokens=payload.get("output_tokens", 0),
                estimated_cost_usd=payload.get("estimated_cost_usd", 0.0),
                metadata={"cache_hit": payload.get("cache_hit", False)},
            )
        )

    with engine.begin() as conn:
        run_db_id = conn.execute(
            text("SELECT id FROM eval_runs WHERE run_id = :run_id"),
            {"run_id": run_uuid},
        ).scalar_one()

        insert_stmt = text(
            """
            INSERT INTO eval_samples (
                eval_run_id, sample_id, input_text, expected_output, actual_output,
                faithfulness_score, answer_relevancy_score, correctness_score,
                hallucination_flag, latency_ms, input_tokens, output_tokens,
                estimated_cost_usd, metadata, created_at
            ) VALUES (
                :eval_run_id, :sample_id, :input_text, :expected_output, :actual_output,
                :faithfulness_score, :answer_relevancy_score, :correctness_score,
                :hallucination_flag, :latency_ms, :input_tokens, :output_tokens,
                :estimated_cost_usd, CAST(:metadata AS JSONB), :created_at
            )
            """
        )
        for row in rows:
            conn.execute(
                insert_stmt,
                {
                    "eval_run_id": run_db_id,
                    "sample_id": row.sample_id,
                    "input_text": row.input_text,
                    "expected_output": row.expected_output,
                    "actual_output": row.actual_output,
                    "faithfulness_score": row.faithfulness_score,
                    "answer_relevancy_score": row.answer_relevancy_score,
                    "correctness_score": row.correctness_score,
                    "hallucination_flag": row.hallucination_flag,
                    "latency_ms": row.latency_ms,
                    "input_tokens": row.input_tokens,
                    "output_tokens": row.output_tokens,
                    "estimated_cost_usd": row.estimated_cost_usd,
                    "metadata": json.dumps(row.metadata),
                    "created_at": datetime.now(timezone.utc),
                },
            )

        conn.execute(
            text(
                """
                UPDATE eval_runs
                SET completed_at = :completed_at, status = 'completed'
                WHERE run_id = :run_id
                """
            ),
            {"completed_at": datetime.now(timezone.utc), "run_id": run_uuid},
        )

    print(f"DeepEval run complete. run_id={run_uuid} samples={len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DeepEval benchmark and persist results.")
    parser.add_argument("--dataset", default="training/eval.jsonl", help="Path to JSONL dataset.")
    parser.add_argument("--api-url", default="http://localhost:8000", help="Base API URL.")
    parser.add_argument("--model-name", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--prompt-version", default="v1")
    parser.add_argument("--retriever-version", default="chroma-minilm-v1")
    args = parser.parse_args()

    run_eval(
        dataset_path=args.dataset,
        api_base_url=args.api_url,
        model_name=args.model_name,
        prompt_version=args.prompt_version,
        retriever_version=args.retriever_version,
    )


if __name__ == "__main__":
    main()

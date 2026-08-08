from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from aic_retrieval.contracts import Task
from aic_retrieval.evaluation import (
    EvaluationError,
    FrameInterval,
    KisGroundTruth,
    KisResponse,
    QaGroundTruth,
    QaResponse,
    TrakeGroundTruth,
    TrakeResponse,
    evaluate_ranked,
    score_kis,
    score_qa,
    score_trake,
)
from aic_retrieval.experiment import (
    ExperimentRecord,
    ResourceMetrics,
    compare_experiments,
)
from aic_retrieval.submission import parse_query

FIXTURES = Path(__file__).parent / "fixtures/official-examples.json"


class EvaluationTests(unittest.TestCase):
    def test_official_examples(self) -> None:
        fixtures = json.loads(FIXTURES.read_text(encoding="utf-8"))

        task, ground_truth, responses = parse_query(fixtures["kis"])
        self.assertEqual(task, Task.TEXTUAL_KIS)
        self.assertEqual(
            evaluate_ranked(task, ground_truth, responses).response_scores,
            (1.0, 0.0, 0.0),
        )

        task, ground_truth, responses = parse_query(fixtures["qa"])
        self.assertEqual(
            evaluate_ranked(task, ground_truth, responses).response_scores,
            (1.0, 0.0, 0.0),
        )

        task, ground_truth, responses = parse_query(fixtures["trake"])
        metrics = evaluate_ranked(task, ground_truth, responses)
        self.assertEqual(metrics.response_scores, (0.75,))
        self.assertEqual(metrics.final_score, 0.75)

    def test_official_final_score_example(self) -> None:
        ground_truth = KisGroundTruth("video", FrameInterval(10, 20))
        responses = [KisResponse("wrong", rank) for rank in range(100)]
        responses[0] = KisResponse("video", 5)
        responses[2] = KisResponse("video", 15)
        metrics = evaluate_ranked(Task.TEXTUAL_KIS, ground_truth, responses)
        synthetic_scores = list(metrics.response_scores)
        synthetic_scores[0] = 0.5
        synthetic_scores[2] = 0.8
        recall = [max(synthetic_scores[:cutoff]) for cutoff in (1, 5, 20, 50, 100)]
        self.assertAlmostEqual(sum(recall) / 5, 0.74)

    def test_kis_interval_boundaries_are_inclusive(self) -> None:
        ground_truth = KisGroundTruth("video", FrameInterval(10, 20))
        self.assertEqual(score_kis(ground_truth, KisResponse("video", 10)), 1.0)
        self.assertEqual(score_kis(ground_truth, KisResponse("video", 20)), 1.0)
        self.assertEqual(score_kis(ground_truth, KisResponse("video", 21)), 0.0)

    def test_qa_normalizes_unicode_case_and_whitespace(self) -> None:
        ground_truth = QaGroundTruth("video", FrameInterval(1, 2), ("Màu Xanh",))
        response = QaResponse("video", 1, "  MÀU   XANH ")
        self.assertEqual(score_qa(ground_truth, response), 1.0)

    def test_qa_semantic_proxy_is_explicitly_labeled(self) -> None:
        ground_truth = QaGroundTruth("video", FrameInterval(1, 2), ("năm",))
        response = QaResponse("video", 1, "5")
        metrics = evaluate_ranked(
            Task.QA,
            ground_truth,
            (response,),
            answer_matcher=lambda answer, expected: answer == "5" and "năm" in expected,
            matcher_name="local-semantic-proxy",
        )
        self.assertEqual(metrics.matcher, "local-semantic-proxy")
        self.assertEqual(metrics.final_score, 1.0)

    def test_trake_partial_credit_and_wrong_video(self) -> None:
        ground_truth = TrakeGroundTruth(
            "video",
            (FrameInterval(1, 2), FrameInterval(4, 5)),
        )
        self.assertEqual(score_trake(ground_truth, TrakeResponse("video", (1, 6))), 0.5)
        self.assertEqual(score_trake(ground_truth, TrakeResponse("wrong", (1, 4))), 0.0)

    def test_trake_rejects_event_count_and_non_monotonic_frames(self) -> None:
        ground_truth = TrakeGroundTruth(
            "video",
            (FrameInterval(1, 2), FrameInterval(4, 5)),
        )
        with self.assertRaisesRegex(EvaluationError, "event count"):
            score_trake(ground_truth, TrakeResponse("video", (1,)))
        with self.assertRaisesRegex(EvaluationError, "strictly increasing"):
            TrakeResponse("video", (4, 2))

    def test_empty_responses_score_zero(self) -> None:
        metrics = evaluate_ranked(
            Task.TEXTUAL_KIS,
            KisGroundTruth("video", FrameInterval(1, 2)),
            (),
        )
        self.assertEqual(metrics.final_score, 0.0)
        self.assertEqual(dict(metrics.recall_at), {1: 0.0, 5: 0.0, 20: 0.0, 50: 0.0, 100: 0.0})

    def test_rejects_over_100_and_duplicate_responses(self) -> None:
        ground_truth = KisGroundTruth("video", FrameInterval(1, 2))
        with self.assertRaisesRegex(EvaluationError, "exceeds maximum"):
            evaluate_ranked(
                Task.TEXTUAL_KIS,
                ground_truth,
                tuple(KisResponse("video", frame_id) for frame_id in range(101)),
            )
        with self.assertRaisesRegex(EvaluationError, "duplicate"):
            evaluate_ranked(
                Task.TEXTUAL_KIS,
                ground_truth,
                (KisResponse("video", 1), KisResponse("video", 1)),
            )

    def test_submission_schema_errors_are_actionable(self) -> None:
        with self.assertRaisesRegex(EvaluationError, "frame_id must be an integer"):
            parse_query(
                {
                    "task": "textual-kis",
                    "ground_truth": {"video_id": "video", "start": 1, "end": 2},
                    "responses": [{"video_id": "video", "frame_id": "1"}],
                }
            )

    def test_experiment_json_is_stable_and_comparison_has_all_cutoffs(self) -> None:
        baseline_metric = evaluate_ranked(
            Task.TEXTUAL_KIS,
            KisGroundTruth("video", FrameInterval(1, 2)),
            (KisResponse("wrong", 1),),
            query_id="query-1",
        )
        candidate_metric = evaluate_ranked(
            Task.TEXTUAL_KIS,
            KisGroundTruth("video", FrameInterval(1, 2)),
            (KisResponse("video", 1),),
            query_id="query-1",
        )
        resources = ResourceMetrics(1.0, 2.0, 3.0, 4.0)
        baseline = ExperimentRecord("base", "abc", "rev", {"b": 2, "a": 1}, (baseline_metric,), resources)
        candidate = ExperimentRecord("candidate", "abc", "rev", {"a": 1}, (candidate_metric,), resources)
        self.assertEqual(baseline.to_json(), baseline.to_json())
        comparison = compare_experiments(baseline, candidate)
        self.assertEqual(tuple(cutoff for cutoff, _ in comparison.recall_at_delta), (1, 5, 20, 50, 100))
        self.assertEqual(comparison.final_score_delta, 1.0)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "experiment.json"
            baseline.save(path)
            first = path.read_bytes()
            baseline.save(path)
            self.assertEqual(path.read_bytes(), first)

    def test_experiment_comparison_requires_same_queries_and_manifest(self) -> None:
        metric = evaluate_ranked(
            Task.TEXTUAL_KIS,
            KisGroundTruth("video", FrameInterval(1, 2)),
            (),
            query_id="query-1",
        )
        other_metric = evaluate_ranked(
            Task.TEXTUAL_KIS,
            KisGroundTruth("video", FrameInterval(1, 2)),
            (),
            query_id="query-2",
        )
        resources = ResourceMetrics(1.0, 2.0, 3.0, 4.0)
        baseline = ExperimentRecord("base", "manifest-a", "rev", {}, (metric,), resources)
        with self.assertRaisesRegex(ValueError, "manifest"):
            compare_experiments(
                baseline,
                ExperimentRecord("candidate", "manifest-b", "rev", {}, (metric,), resources),
            )
        with self.assertRaisesRegex(ValueError, "query IDs"):
            compare_experiments(
                baseline,
                ExperimentRecord(
                    "candidate", "manifest-a", "rev", {}, (other_metric,), resources
                ),
            )


if __name__ == "__main__":
    unittest.main()

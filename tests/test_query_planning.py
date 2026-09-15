from __future__ import annotations

import json
import unittest
from types import SimpleNamespace
from unittest import mock

from aic_retrieval.query_planning import (
    MAX_NEW_TOKENS,
    OrderedEvent,
    QueryPlan,
    QueryPlanningError,
    QwenQueryPlanner,
    VisualQuery,
    baseline_query_plan,
    parse_query_plan,
)
from aic_retrieval.reranking import PLANNER_MODEL_ID, PLANNER_REVISION


def valid_payload() -> dict[str, object]:
    return {
        "holistic_visual": "a woman in a purple ao dai beside a white tuk-tuk",
        "visual_clauses": [
            "a woman posing in a purple traditional dress",
            "a white tuk-tuk parked beside a woman",
        ],
        "ordered_events": [
            {
                "visual": "a woman posing beside a white tuk-tuk",
                "exact_text": "79H-6072",
            }
        ],
        "exact_texts": ["79H-6072"],
    }


class QueryPlanningTests(unittest.TestCase):
    def test_baseline_preserves_normalized_raw_query(self) -> None:
        plan = baseline_query_plan("  người\tđàn ông không đứng  ")
        self.assertEqual(
            plan,
            QueryPlan(
                (VisualQuery("người đàn ông không đứng", "raw"),),
                (),
                (),
            ),
        )

    def test_parser_builds_bounded_plan_and_preserves_exact_text(self) -> None:
        raw = "Người phụ nữ cạnh xe lam có biển số 79H-6072."
        plan = parse_query_plan(json.dumps(valid_payload(), ensure_ascii=False), raw)
        self.assertEqual(plan.visual_queries[0], VisualQuery(raw, "raw"))
        self.assertEqual(plan.visual_queries[1].kind, "holistic")
        self.assertEqual(sum(item.kind == "clause" for item in plan.visual_queries), 2)
        self.assertEqual(plan.exact_texts, ("79H-6072",))
        self.assertEqual(plan.ordered_events[0].exact_text, "79H-6072")

    def test_parser_accepts_one_json_fence(self) -> None:
        content = json.dumps(valid_payload(), ensure_ascii=False)
        self.assertEqual(
            parse_query_plan(f"```json\n{content}\n```", "raw query 79H-6072"),
            parse_query_plan(content, "raw query 79H-6072"),
        )

    def test_parser_rejects_schema_bounds_duplicates_and_bad_event_reference(self) -> None:
        payload = valid_payload()
        cases = (
            ({**payload, "extra": True}, "invalid fields"),
            ({**payload, "visual_clauses": ["x"] * 5}, "exceeds"),
            ({**payload, "visual_clauses": ["same", "same"]}, "duplicates"),
            (
                {
                    **payload,
                    "ordered_events": [
                        {"visual": "street sign", "exact_text": "different"}
                    ],
                },
                "must appear",
            ),
            ({**payload, "exact_texts": ["x\x00y"]}, "controls"),
            (
                {
                    **payload,
                    "ordered_events": [],
                    "exact_texts": ["invented sign"],
                },
                "verbatim",
            ),
        )
        for value, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(QueryPlanningError, message):
                    parse_query_plan(json.dumps(value), "raw query 79H-6072")

    def test_contract_rejects_invalid_kinds_counts_and_missing_text_reference(self) -> None:
        with self.assertRaisesRegex(QueryPlanningError, "kind"):
            VisualQuery("text", "unknown")
        with self.assertRaisesRegex(QueryPlanningError, "begin"):
            QueryPlan((VisualQuery("english", "holistic"),), (), ())
        with self.assertRaisesRegex(QueryPlanningError, "appear"):
            QueryPlan(
                (VisualQuery("raw", "raw"),),
                (OrderedEvent("sign", "private text"),),
                (),
            )
        with self.assertRaisesRegex(QueryPlanningError, "VisualQuery tuple"):
            QueryPlan([VisualQuery("raw", "raw")], (), ())  # type: ignore[arg-type]
        with self.assertRaisesRegex(QueryPlanningError, "exact texts must be a tuple"):
            QueryPlan(
                (VisualQuery("raw", "raw"),),
                (),
                ["raw"],  # type: ignore[arg-type]
            )

    def test_qwen_wrapper_reuses_pinned_runtime_and_strict_parser(self) -> None:
        content = json.dumps(valid_payload(), ensure_ascii=False)
        runtime = SimpleNamespace(
            last_generated_tokens=27,
            generate_json=mock.Mock(return_value=content),
        )
        with mock.patch(
            "aic_retrieval.query_planning.QwenPlanner",
            return_value=runtime,
        ) as loader:
            planner = QwenQueryPlanner(device="cuda")
            plan = planner.plan("query có biển 79H-6072")
        loader.assert_called_once_with(
            PLANNER_MODEL_ID,
            PLANNER_REVISION,
            device="cuda",
        )
        prompt = runtime.generate_json.call_args.args[0]
        self.assertIn("exact_texts", prompt)
        self.assertIn("Do not translate exact text", prompt)
        self.assertIn("query có biển 79H-6072", prompt)
        self.assertEqual(
            runtime.generate_json.call_args.kwargs,
            {"max_new_tokens": MAX_NEW_TOKENS},
        )
        self.assertEqual(plan.exact_texts, ("79H-6072",))
        self.assertEqual(planner.last_generated_tokens, 27)

    def test_generated_content_has_no_serialization_api(self) -> None:
        plan = parse_query_plan(
            json.dumps(valid_payload()),
            "private raw query 79H-6072",
        )
        self.assertFalse(hasattr(plan, "to_dict"))
        self.assertFalse(hasattr(plan, "to_json"))


if __name__ == "__main__":
    unittest.main()

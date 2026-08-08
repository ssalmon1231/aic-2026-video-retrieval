from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import numpy as np

from aic_retrieval.contracts import KeyframeRecord
from aic_retrieval.data import ObjectDetection, load_object_detections, parse_object_detections
from aic_retrieval.index import ExactIndex, IndexMetadata
from aic_retrieval.reranking import (
    MAX_NEW_TOKENS,
    PLANNER_MODEL_ID,
    PLANNER_REVISION,
    ContrastivePlan,
    ContrastiveReranker,
    QwenPlanner,
    RerankerConfig,
    RerankingError,
    build_object_vocabulary,
    object_consistency,
    parse_plan,
    score_shortlist,
)
from aic_retrieval.retrieval import RetrievalConfig, retrieve_kis


class FakeEncoder:
    def __init__(self, vectors: np.ndarray) -> None:
        self.vectors = vectors
        self.calls: list[tuple[str, ...]] = []

    def encode(self, texts: tuple[str, ...]) -> SimpleNamespace:
        self.calls.append(texts)
        return SimpleNamespace(vectors=self.vectors)


def test_index() -> ExactIndex:
    vectors = np.eye(2, dtype=np.float32)
    keyframes = (
        KeyframeRecord("video-a", "001", "a.jpg", 0, 10, 0, "objects/a.json"),
        KeyframeRecord("video-b", "001", "b.jpg", 0, 20, 0, "objects/b.json"),
    )
    return ExactIndex(
        vectors,
        keyframes,
        IndexMetadata(1, "numpy-flat-ip", "test", "test", 2, 2, "float32", True, "abc"),
    )


class RerankingTests(unittest.TestCase):
    def test_object_loader_supports_both_observed_schemas_and_neutral_failures(self) -> None:
        parallel = {
            "detection_scores": ["0.9"],
            "detection_class_names": ["Motorcycle"],
            "detection_class_entities": ["motorbike"],
            "detection_boxes": [["0", 1, 2, 3]],
            "detection_class_labels": [7],
        }
        listed = [{"confidence": "0.8", "category": "Bicycle", "box": [0, 1, 2, 3]}]
        self.assertEqual(parse_object_detections(parallel)[0].labels, ("motorcycle", "motorbike", "7"))
        self.assertEqual(parse_object_detections(listed), (ObjectDetection(0.8, ("bicycle",)),))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "valid.json").write_text(json.dumps(parallel), encoding="utf-8")
            (root / "bad.json").write_text("not-json", encoding="utf-8")
            self.assertEqual(load_object_detections(root, "valid.json")[0].score, 0.9)
            self.assertEqual(load_object_detections(root, "missing.json"), ())
            self.assertEqual(load_object_detections(root, "bad.json"), ())
            self.assertEqual(load_object_detections(root, "../outside.json"), ())

    def test_plan_parser_is_strict_and_private(self) -> None:
        content = json.dumps(
            {
                "positive": "one man riding a motorcycle",
                "negatives": ["group bicycle race"],
                "required_objects": ["motorcycle"],
                "excluded_objects": ["bicycle"],
            }
        )
        plan = parse_plan(content, ("motorcycle", "bicycle"))
        self.assertEqual(plan.negatives, ("group bicycle race",))
        contrastive_only_plan = parse_plan(content, ())
        self.assertEqual(contrastive_only_plan.required_objects, ())
        self.assertEqual(contrastive_only_plan.excluded_objects, ())
        fenced = f"```json\n{content}\n```"
        fenced_plan = parse_plan(fenced, ("motorcycle", "bicycle"))
        self.assertEqual(fenced_plan, plan)
        for invalid in (
            "```json\n{}\n``` trailing text",
            "```text\n" + content + "\n```",
            "```json\n" + content + "\n```\n```json\n{}\n```",
            json.dumps({**json.loads(content), "extra": True}),
            json.dumps({**json.loads(content), "negatives": ["a", "b", "c", "d"]}),
            json.dumps({**json.loads(content), "required_objects": ["car"]}),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(RerankingError):
                    parse_plan(invalid, ("motorcycle", "bicycle"))

    def test_object_scoring_and_vocabulary_are_deterministic(self) -> None:
        detections = (
            ObjectDetection(0.9, ("motorcycle",)),
            ObjectDetection(0.8, ("bicycle",)),
        )
        plan = ContrastivePlan("positive", (), ("motorcycle",), ("bicycle",))
        self.assertAlmostEqual(object_consistency(detections, plan), 0.1)
        vocabulary = build_object_vocabulary(((detections[1],), (detections[0],)))
        self.assertEqual(vocabulary, ("motorcycle", "bicycle"))

    def test_contrastive_negative_penalty_and_stable_ties(self) -> None:
        index = test_index()
        hits = index.search(np.array([1.0, 1.0], dtype=np.float32), limit=2)
        encoder = FakeEncoder(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
        plan = ContrastivePlan("motorcycle", ("bicycle",), (), ())
        ranked = score_shortlist(
            index,
            hits,
            ((), ()),
            plan,
            encoder,  # type: ignore[arg-type]
            RerankerConfig(enabled=True, positive_weight=1.0, negative_weight=1.0),
        )
        self.assertEqual(ranked[0].row, 0)
        self.assertEqual(encoder.calls, [("motorcycle", "bicycle")])

        object_only_encoder = FakeEncoder(np.array([[1.0, 1.0]], dtype=np.float32))
        tied = score_shortlist(
            index,
            hits,
            ((), ()),
            ContrastivePlan("same", (), (), ()),
            object_only_encoder,  # type: ignore[arg-type]
            RerankerConfig(enabled=True, object_weight=1.0),
        )
        self.assertEqual(tuple(hit.row for hit in tied), tuple(hit.row for hit in hits))
        self.assertEqual(object_only_encoder.calls, [])

    def test_qwen_planner_pins_loader_and_generation_contract(self) -> None:
        calls: dict[str, object] = {}
        generated = json.dumps(
            {
                "positive": "one motorcycle rider",
                "negatives": ["bicycle race"],
                "required_objects": ["motorcycle"],
                "excluded_objects": ["bicycle"],
            }
        )

        class TensorBatch(dict):
            def to(self, device: str):
                calls["batch_device"] = device
                return self

        class Tokenizer:
            @classmethod
            def from_pretrained(cls, model_id: str, **keywords: object):
                calls["tokenizer_load"] = (model_id, keywords)
                return cls()

            def apply_chat_template(self, messages, **keywords: object) -> str:
                calls["messages"] = messages
                calls["chat_keywords"] = keywords
                return "rendered prompt"

            def __call__(self, rendered: str, **keywords: object) -> TensorBatch:
                calls["tokenizer_call"] = (rendered, keywords)
                return TensorBatch(input_ids=np.array([[10, 11, 12]]))

            def decode(self, tokens, **keywords: object) -> str:
                calls["decoded_tokens"] = np.asarray(tokens).tolist()
                calls["decode_keywords"] = keywords
                return generated

        class Model:
            config = SimpleNamespace(_commit_hash=PLANNER_REVISION)

            @classmethod
            def from_pretrained(cls, model_id: str, **keywords: object):
                calls["model_load"] = (model_id, keywords)
                return cls()

            def to(self, device: str):
                calls["model_device"] = device
                return self

            def eval(self) -> None:
                calls["eval"] = True

            def generate(self, **keywords: object):
                calls["generate"] = keywords
                return np.array([[10, 11, 12, 21, 22]])

        torch_module = ModuleType("torch")
        torch_module.float16 = "float16"  # type: ignore[attr-defined]
        torch_module.inference_mode = nullcontext  # type: ignore[attr-defined]
        transformers_module = ModuleType("transformers")
        transformers_module.AutoTokenizer = Tokenizer  # type: ignore[attr-defined]
        transformers_module.AutoModelForCausalLM = Model  # type: ignore[attr-defined]
        with mock.patch.dict(
            sys.modules,
            {"torch": torch_module, "transformers": transformers_module},
        ):
            planner = QwenPlanner(device="cuda")
            plan = planner.plan(
                "private query must remain internal",
                ("motorcycle", "bicycle"),
            )

        expected_tokenizer_loader = {
            "revision": PLANNER_REVISION,
            "trust_remote_code": False,
        }
        expected_model_loader = {
            **expected_tokenizer_loader,
            "torch_dtype": torch_module.float16,
        }
        self.assertEqual(
            calls["tokenizer_load"],
            (PLANNER_MODEL_ID, expected_tokenizer_loader),
        )
        self.assertEqual(calls["model_load"], (PLANNER_MODEL_ID, expected_model_loader))
        self.assertEqual(calls["model_device"], "cuda")
        self.assertTrue(calls["eval"])
        self.assertEqual(calls["batch_device"], "cuda")
        messages = calls["messages"]
        self.assertIsInstance(messages, list)
        planner_prompt = messages[0]["content"]
        self.assertIn("one minified JSON object only", planner_prompt)
        self.assertIn(
            'Valid example: {"positive":"person outdoors",'
            '"negatives":[],"required_objects":[],"excluded_objects":[]}',
            planner_prompt,
        )
        self.assertIn(
            "negatives must be a JSON array of 0 to 3 unique strings",
            planner_prompt,
        )
        self.assertIn(
            "After each key write a colon and its JSON value",
            planner_prompt,
        )
        self.assertIn(
            "When vocabulary is empty, both object arrays must be []",
            planner_prompt,
        )
        self.assertIn("one short visual string", planner_prompt)
        self.assertIn("No explanation or markdown", planner_prompt)
        generate_call = calls["generate"]
        self.assertIsInstance(generate_call, dict)
        self.assertEqual(
            set(generate_call), {"input_ids", "do_sample", "max_new_tokens"}
        )
        np.testing.assert_array_equal(generate_call["input_ids"], [[10, 11, 12]])
        self.assertIs(generate_call["do_sample"], False)
        self.assertEqual(generate_call["max_new_tokens"], MAX_NEW_TOKENS)
        self.assertEqual(calls["decoded_tokens"], [21, 22])
        self.assertEqual(calls["decode_keywords"], {"skip_special_tokens": True})
        self.assertEqual(planner.last_generated_tokens, 2)
        self.assertEqual(plan.negatives, ("bicycle race",))

    def test_qwen_planner_resets_generated_token_count_before_failure(self) -> None:
        planner = object.__new__(QwenPlanner)
        planner._device = "cuda"
        planner._torch = SimpleNamespace(inference_mode=nullcontext)
        planner._last_generated_tokens = 12

        class BrokenTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                raise RuntimeError("private generated content")

        planner._tokenizer = BrokenTokenizer()
        planner._model = SimpleNamespace()
        with self.assertRaisesRegex(RerankingError, "planner generation failed"):
            planner.plan("private query", ())
        self.assertEqual(planner.last_generated_tokens, 0)

    def test_qwen_planner_sanitizes_generation_failure(self) -> None:
        planner = object.__new__(QwenPlanner)
        planner._device = "cuda"
        planner._torch = SimpleNamespace(inference_mode=nullcontext)
        planner._last_generated_tokens = 0
        planner._tokenizer = SimpleNamespace(
            apply_chat_template=lambda *args, **kwargs: "rendered",
            __call__=lambda *args, **kwargs: None,
        )

        class BrokenTokenizer:
            def apply_chat_template(self, *args, **kwargs):
                return "rendered"

            def __call__(self, *args, **kwargs):
                raise RuntimeError("private generated content")

        planner._tokenizer = BrokenTokenizer()
        planner._model = SimpleNamespace()
        with self.assertRaisesRegex(RerankingError, "planner generation failed") as raised:
            planner.plan("private query", ())
        self.assertNotIn("private generated content", str(raised.exception))

    def test_failed_and_over_budget_reranking_falls_back_to_exact_baseline(self) -> None:
        index = test_index()
        query = np.array([1.0, 0.5], dtype=np.float32)
        baseline = retrieve_kis(index, query, RetrievalConfig(candidate_depth=2, result_limit=2))

        class FailingPlanner:
            def plan(self, query: str, vocabulary: tuple[str, ...]) -> ContrastivePlan:
                raise RuntimeError("private generated content")

        reranker = ContrastiveReranker(
            RerankerConfig(enabled=True, object_weight=1.0),
            FailingPlanner(),
            FakeEncoder(np.ones((1, 2), dtype=np.float32)),  # type: ignore[arg-type]
            ".",
        )
        failed = retrieve_kis(
            index,
            query,
            RetrievalConfig(candidate_depth=2, result_limit=2),
            rerank=lambda idx, hits: reranker.rerank("private query", idx, hits),
        )
        self.assertEqual(failed, baseline)
        self.assertTrue(reranker.last_status.fallback)

        class Planner:
            def plan(self, query: str, vocabulary: tuple[str, ...]) -> ContrastivePlan:
                return ContrastivePlan("positive", (), (), ())

        ticks = iter((0.0, 2.0))
        slow = ContrastiveReranker(
            RerankerConfig(enabled=True, object_weight=1.0, planner_budget_seconds=1.5),
            Planner(),
            FakeEncoder(np.ones((1, 2), dtype=np.float32)),  # type: ignore[arg-type]
            ".",
            clock=lambda: next(ticks),
        )
        first = slow.rerank("private", index, index.search(query, 2))
        second = slow.rerank("private", index, index.search(query, 2))
        self.assertEqual(first, index.search(query, 2))
        self.assertEqual(second, index.search(query, 2))
        self.assertTrue(slow.circuit_open)


if __name__ == "__main__":
    unittest.main()

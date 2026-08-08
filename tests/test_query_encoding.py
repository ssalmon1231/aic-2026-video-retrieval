from __future__ import annotations

import builtins
import json
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from aic_retrieval.query import (
    IMAGE_PROVENANCE,
    MULTILINGUAL_IMAGE_PROVENANCE,
    NORMALIZATION,
    SENTENCE_TRANSFORMERS_BACKEND,
    QueryEncoderConfig,
    QueryEncoderRuntime,
    QueryEncodingError,
    _load_multilingual_runtime,
    _load_runtime,
    _normalize_vector,
    encode_text,
    encode_texts,
    load_config,
    normalize_query,
)

REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
MULTILINGUAL_REVISION = "58edf8cada9e398793dca955574a48cbb7f18be2"


class FakeTensor:
    def __init__(self, value: object) -> None:
        self.value = np.asarray(value)
        self.devices: list[str] = []

    def to(self, device: str) -> FakeTensor:
        self.devices.append(device)
        return self

    def float(self) -> FakeTensor:
        self.value = self.value.astype(np.float32)
        return self

    def __getitem__(self, index: int) -> FakeTensor:
        return FakeTensor(self.value[index])

    def detach(self) -> FakeTensor:
        return self

    def cpu(self) -> FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self.value


class FakeTorch:
    @staticmethod
    @contextmanager
    def inference_mode():
        yield


class FakeTokenizer:
    def __init__(self) -> None:
        self.calls: list[tuple[list[str], dict[str, object]]] = []
        self.input_ids = FakeTensor([[1, 2]])
        self.attention_mask = FakeTensor([[1, 1]])

    def __call__(self, texts: list[str], **kwargs: object) -> dict[str, FakeTensor]:
        self.calls.append((texts, kwargs))
        return {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
        }


class FakeModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(max_position_embeddings=77)
        )
        self.text_calls: list[dict[str, object]] = []
        self.projected: object | None = None

    def text_model(self, **kwargs: object) -> SimpleNamespace:
        self.text_calls.append(kwargs)
        return SimpleNamespace(pooler_output=FakeTensor([[1.0, 2.0]]))

    def text_projection(self, pooled: object) -> FakeTensor:
        self.projected = pooled
        return FakeTensor([[3.0, 4.0]])


def make_config(**changes: object) -> QueryEncoderConfig:
    values: dict[str, object] = {
        "model_id": "openai/clip-vit-base-patch32",
        "revision": REVISION,
        "tokenizer_use_fast": False,
        "device": "cpu",
    }
    values.update(changes)
    return QueryEncoderConfig(**values)  # type: ignore[arg-type]


def make_multilingual_config(**changes: object) -> QueryEncoderConfig:
    values: dict[str, object] = {
        "backend": SENTENCE_TRANSFORMERS_BACKEND,
        "model_id": "sentence-transformers/clip-ViT-B-32-multilingual-v1",
        "revision": MULTILINGUAL_REVISION,
        "tokenizer_use_fast": None,
        "device": "cpu",
    }
    values.update(changes)
    return QueryEncoderConfig(**values)  # type: ignore[arg-type]


class QueryEncodingTests(unittest.TestCase):
    def test_normalization_preserves_vietnamese_and_negation(self) -> None:
        self.assertEqual(
            normalize_query("  không\tchạy  nhanh \n"),
            "không chạy nhanh",
        )

    def test_rejects_empty_text_and_untrusted_config(self) -> None:
        with self.assertRaisesRegex(QueryEncodingError, "must not be empty"):
            normalize_query(" \t\n")
        for changes, message in (
            ({"model_id": ""}, "model_id"),
            ({"revision": "revision-unverified"}, "unverified"),
            ({"revision": "abc"}, "40-character"),
            ({"revision": REVISION.upper()}, "lowercase hexadecimal"),
            ({"device": ""}, "device"),
            ({"tokenizer_use_fast": True}, "must be false"),
            ({"backend": "unknown"}, "backend must be one of"),
        ):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(QueryEncodingError, message):
                    make_config(**changes)

    def test_config_loader_is_strict(self) -> None:
        valid = {
            "model_id": "openai/clip-vit-base-patch32",
            "revision": REVISION,
            "tokenizer_use_fast": False,
            "device": "cuda",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "encoder.json"
            path.write_text(json.dumps(valid), encoding="utf-8")
            self.assertEqual(load_config(path).revision, REVISION)
            cases = (
                ({"unexpected": True}, "missing fields"),
                ({**valid, "unexpected": True}, "unknown"),
                ([], "root must be an object"),
            )
            for payload, message in cases:
                with self.subTest(message=message):
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(QueryEncodingError, message):
                        load_config(path)

    def test_multilingual_config_loader_is_strict(self) -> None:
        valid = {
            "backend": SENTENCE_TRANSFORMERS_BACKEND,
            "model_id": "sentence-transformers/clip-ViT-B-32-multilingual-v1",
            "revision": MULTILINGUAL_REVISION,
            "device": "cuda",
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "encoder.json"
            path.write_text(json.dumps(valid), encoding="utf-8")
            config = load_config(path)
            self.assertEqual(config.backend, SENTENCE_TRANSFORMERS_BACKEND)
            self.assertIsNone(config.tokenizer_use_fast)
            for payload, message in (
                ({**valid, "tokenizer_use_fast": False}, "unknown"),
                ({key: value for key, value in valid.items() if key != "backend"}, "tokenizer_use_fast"),
                ({**valid, "backend": "unknown"}, "backend must be one of"),
            ):
                with self.subTest(message=message):
                    path.write_text(json.dumps(payload), encoding="utf-8")
                    with self.assertRaisesRegex(QueryEncodingError, message):
                        load_config(path)
        with self.assertRaisesRegex(QueryEncodingError, "not configurable"):
            make_multilingual_config(tokenizer_use_fast=False)

    def test_missing_optional_dependencies_has_actionable_error(self) -> None:
        real_import = builtins.__import__

        def blocked_import(name: str, *args: object, **kwargs: object) -> object:
            if name in {"torch", "transformers"}:
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=blocked_import):
            with self.assertRaisesRegex(QueryEncodingError, r"\[clip\]"):
                _load_runtime(make_config())

        def blocked_multilingual(name: str, *args: object, **kwargs: object) -> object:
            if name == "sentence_transformers":
                raise ImportError("blocked for test")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=blocked_multilingual):
            with self.assertRaisesRegex(QueryEncodingError, r"\[multilingual\]"):
                _load_multilingual_runtime(make_multilingual_config())

    def test_runtime_requires_configured_checkpoint_revision(self) -> None:
        class AutoTokenizer:
            @staticmethod
            def from_pretrained(*args: object, **kwargs: object) -> object:
                return object()

        class LoadedModel:
            def __init__(self) -> None:
                self.config = SimpleNamespace(_commit_hash="0" * 40)

        model_kwargs: dict[str, object] = {}

        class CLIPModel:
            @staticmethod
            def from_pretrained(*args: object, **kwargs: object) -> LoadedModel:
                model_kwargs.update(kwargs)
                return LoadedModel()

        fake_transformers = SimpleNamespace(
            AutoTokenizer=AutoTokenizer,
            CLIPModel=CLIPModel,
        )
        with mock.patch.dict(
            "sys.modules",
            {"torch": SimpleNamespace(), "transformers": fake_transformers},
        ):
            with self.assertRaisesRegex(
                QueryEncodingError,
                "loaded checkpoint revision",
            ):
                _load_runtime(make_config())
        self.assertIs(model_kwargs["use_safetensors"], False)
        self.assertEqual(model_kwargs["revision"], REVISION)

    def test_encode_projects_and_normalizes_float32_vector(self) -> None:
        tokenizer = FakeTokenizer()
        model = FakeModel()
        with mock.patch(
            "aic_retrieval.query._load_runtime",
            return_value=(FakeTorch(), tokenizer, model),
        ):
            encoded = encode_text(
                "  a   motorcycle  ",
                make_config(),
                expected_dimension=2,
            )

        np.testing.assert_allclose(encoded.vector, [0.6, 0.8], rtol=1e-6)
        self.assertEqual(encoded.vector.dtype, np.float32)
        self.assertEqual(tokenizer.calls[0][0], ["a motorcycle"])
        self.assertEqual(
            tokenizer.calls[0][1],
            {
                "padding": True,
                "truncation": True,
                "max_length": 77,
                "return_tensors": "pt",
            },
        )
        self.assertEqual(tokenizer.input_ids.devices, ["cpu"])
        self.assertEqual(tokenizer.attention_mask.devices, ["cpu"])
        self.assertIsNotNone(model.projected)
        self.assertEqual(encoded.provenance.revision, REVISION)
        self.assertEqual(encoded.provenance.tokenizer_class, "FakeTokenizer")
        self.assertEqual(encoded.provenance.normalization, NORMALIZATION)
        self.assertEqual(encoded.provenance.image_features, IMAGE_PROVENANCE)
        self.assertEqual(encoded.provenance.backend, "openai-clip")
        self.assertEqual(encoded.provenance.runtime_class, "FakeModel")
        self.assertTrue(encoded.provenance.normalized)

    def test_multilingual_runtime_and_encoding_are_pinned(self) -> None:
        constructor: dict[str, object] = {}

        class SentenceTransformer:
            def __init__(self, model_id: str, **kwargs: object) -> None:
                constructor["model_id"] = model_id
                constructor.update(kwargs)
                self.config = SimpleNamespace(_commit_hash=MULTILINGUAL_REVISION)
                self.calls: list[tuple[list[str], dict[str, object]]] = []

            def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
                self.calls.append((texts, kwargs))
                return np.array([[3.0, 4.0]], dtype=np.float64)

        fake_module = SimpleNamespace(SentenceTransformer=SentenceTransformer)
        with mock.patch.dict("sys.modules", {"sentence_transformers": fake_module}):
            model = _load_multilingual_runtime(make_multilingual_config())
        self.assertEqual(
            constructor,
            {
                "model_id": "sentence-transformers/clip-ViT-B-32-multilingual-v1",
                "revision": MULTILINGUAL_REVISION,
                "device": "cpu",
                "trust_remote_code": False,
            },
        )

        with mock.patch(
            "aic_retrieval.query._load_multilingual_runtime",
            return_value=model,
        ):
            encoded = encode_text(
                "  một   người đàn ông  ",
                make_multilingual_config(),
                expected_dimension=2,
            )
        np.testing.assert_allclose(encoded.vector, [0.6, 0.8], rtol=1e-6)
        self.assertEqual(encoded.vector.dtype, np.float32)
        self.assertEqual(
            model.calls,
            [
                (
                    ["một người đàn ông"],
                    {
                        "convert_to_numpy": True,
                        "normalize_embeddings": False,
                        "show_progress_bar": False,
                    },
                )
            ],
        )
        self.assertEqual(encoded.provenance.backend, SENTENCE_TRANSFORMERS_BACKEND)
        self.assertEqual(encoded.provenance.revision, MULTILINGUAL_REVISION)
        self.assertIsNone(encoded.provenance.tokenizer_class)
        self.assertIsNone(encoded.provenance.tokenizer_use_fast)
        self.assertEqual(encoded.provenance.runtime_class, "SentenceTransformer")
        self.assertEqual(
            encoded.provenance.image_features,
            MULTILINGUAL_IMAGE_PROVENANCE,
        )

    def test_batch_encoding_reuses_warm_multilingual_runtime(self) -> None:
        class SentenceTransformer:
            def __init__(self) -> None:
                self.calls: list[list[str]] = []

            def encode(self, texts: list[str], **kwargs: object) -> np.ndarray:
                self.calls.append(texts)
                return np.array([[3.0, 4.0], [0.0, 2.0]], dtype=np.float64)

        model = SentenceTransformer()
        with mock.patch(
            "aic_retrieval.query._load_multilingual_runtime",
            return_value=model,
        ) as loader:
            config = make_multilingual_config()
            runtime = QueryEncoderRuntime(config, expected_dimension=2)
            encoded = encode_texts(
                ["  một   người  ", " xe máy "],
                config,
                expected_dimension=2,
                runtime=runtime,
            )

        loader.assert_called_once_with(config)
        self.assertEqual(model.calls, [["một người", "xe máy"]])
        self.assertEqual(encoded.vectors.shape, (2, 2))
        self.assertEqual(encoded.vectors.dtype, np.float32)
        np.testing.assert_allclose(encoded.vectors, [[0.6, 0.8], [0.0, 1.0]])

    def test_batch_encoding_rejects_empty_mismatched_and_invalid_outputs(self) -> None:
        config = make_multilingual_config()
        model = SimpleNamespace(
            encode=lambda texts, **kwargs: np.ones((len(texts), 2), dtype=np.float32)
        )
        with mock.patch(
            "aic_retrieval.query._load_multilingual_runtime", return_value=model
        ):
            runtime = QueryEncoderRuntime(config, expected_dimension=2)
        with self.assertRaisesRegex(QueryEncodingError, "non-empty sequence"):
            runtime.encode([])
        with self.assertRaisesRegex(QueryEncodingError, "non-empty sequence"):
            runtime.encode("query")  # type: ignore[arg-type]
        with self.assertRaisesRegex(QueryEncodingError, "does not match"):
            encode_texts(
                ["query"],
                config,
                expected_dimension=3,
                runtime=runtime,
            )

        for output, message in (
            (np.ones((1, 3), dtype=np.float32), "batch shape"),
            (np.array([[np.nan, 1.0]], dtype=np.float32), "NaN or infinity"),
            (np.zeros((1, 2), dtype=np.float32), "norm must be positive"),
        ):
            with self.subTest(message=message):
                bad_model = SimpleNamespace(encode=lambda *args, value=output, **kwargs: value)
                with mock.patch(
                    "aic_retrieval.query._load_multilingual_runtime",
                    return_value=bad_model,
                ):
                    with self.assertRaisesRegex(QueryEncodingError, message):
                        encode_texts(
                            ["query"],
                            config,
                            expected_dimension=2,
                        )

    def test_multilingual_runtime_rejects_wrong_revision(self) -> None:
        class SentenceTransformer:
            def __init__(self, *args: object, **kwargs: object) -> None:
                self.config = SimpleNamespace(_commit_hash="0" * 40)

        fake_module = SimpleNamespace(SentenceTransformer=SentenceTransformer)
        with mock.patch.dict("sys.modules", {"sentence_transformers": fake_module}):
            with self.assertRaisesRegex(QueryEncodingError, "loaded checkpoint revision"):
                _load_multilingual_runtime(make_multilingual_config())

    def test_multilingual_encoding_rejects_bad_batch_shape(self) -> None:
        model = SimpleNamespace(
            encode=lambda *args, **kwargs: np.array([3.0, 4.0], dtype=np.float32)
        )
        with mock.patch(
            "aic_retrieval.query._load_multilingual_runtime",
            return_value=model,
        ):
            with self.assertRaisesRegex(QueryEncodingError, "batch shape"):
                encode_text(
                    "query",
                    make_multilingual_config(),
                    expected_dimension=2,
                )

    def test_vector_validation_rejects_bad_outputs(self) -> None:
        for vector, dimension, message in (
            ([1.0], 2, "shape"),
            ([np.nan, 1.0], 2, "NaN or infinity"),
            ([np.inf, 1.0], 2, "NaN or infinity"),
            ([0.0, 0.0], 2, "norm must be positive"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(QueryEncodingError, message):
                    _normalize_vector(vector, dimension)
        for dimension in (0, -1, True, 2.0):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(QueryEncodingError, "positive integer"):
                    encode_text("query", make_config(), expected_dimension=dimension)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()

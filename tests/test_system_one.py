import copy
import io
import json
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.error import HTTPError, URLError

import numpy as np
import pytest
import torch
from PIL import Image

from methods.system_one.backends import ClefFlashBackend, JevBackend
from methods.system_one.cache import ChoiceCache
from methods.system_one.retrieval import TrainingRetriever
from methods.system_one.schema import build_choice_request, parse_choice


def request(**kwargs):
    return build_choice_request(
        "test-decision-model",
        ["candidate-a", "candidate-b"],
        "new query",
        ["training query"],
        [[0.25, 0.75]],
        **kwargs,
    )


def response(values=None, choice="model_1"):
    return {
        "answers": {
            "selected_model": {
                "choice": choice,
                "probabilities": values if values is not None else {"model_1": 0.8, "model_0": 0.2},
            }
        }
    }


def test_retrieval_excludes_target_and_duplicate_text_and_images():
    retriever = TrainingRetriever(
        [[1, 0], [1, 0], [0.9, 0.1], [0.8, 0.2], [0, 1]],
        [" TARGET ", "first", " FIRST  ", "shared image", "second"],
        ["", "", "", "same-image", ""],
        k=2,
    )
    np.testing.assert_array_equal(retriever.retrieve([1, 0], "target", "same-image"), [1, 4])


def test_retrieval_requires_full_history_and_stable_ties():
    retriever = TrainingRetriever([[1, 0]] * 3, ["a", "b", "c"], k=2)
    np.testing.assert_array_equal(retriever.retrieve([1, 0], "new"), [0, 1])
    with pytest.raises(ValueError, match="found 1"):
        TrainingRetriever([[1, 0]] * 3, ["a", "a", "c"], k=2).retrieve([1, 0], "c")


@pytest.mark.parametrize("values", [[[0, 0]], [[np.nan, 1]]])
def test_retrieval_rejects_undefined_cosine(values):
    with pytest.raises(ValueError):
        TrainingRetriever(values, ["a"])


def test_choice_uses_only_training_utilities_and_preserves_candidate_order():
    payload = request()
    state = payload["state"]
    assert state["shared_training_history"] == [
        {
            "query": "training query",
            "observed_utility": {"model_0": 0.25, "model_1": 0.75},
        }
    ]
    assert set(state) == {"target_query", "candidates", "shared_training_history"}
    np.testing.assert_allclose(parse_choice(response(), ["model_0", "model_1"]), [0.2, 0.8])


@pytest.mark.parametrize(
    "bad_response",
    [
        {},
        response({"model_0": 1}),
        response({"model_0": -0.1, "model_1": 1.1}),
        response({"model_0": np.nan, "model_1": 0.5}),
        response({"model_0": 0, "model_1": 0}),
        response({"model_0": 0.1, "model_1": 0.1}),
        response({"model_0": 0.9, "model_1": 0.1}),
        response(choice="unknown"),
        response({"model_0": [0.2], "model_1": [0.8]}),
    ],
)
def test_choice_rejects_invalid_or_incomplete_probabilities(bad_response):
    with pytest.raises(ValueError):
        parse_choice(bad_response, ["model_0", "model_1"])


def test_choice_normalizes_rounded_api_probabilities():
    reply = response({"model_0": 0.33, "model_1": 0.66})
    np.testing.assert_allclose(parse_choice(reply, ["model_0", "model_1"]), [1 / 3, 2 / 3])


def test_cache_reuses_valid_scores_and_changes_with_evidence_and_backend(tmp_path):
    backend = SimpleNamespace(cache_identity={"model": "v1"}, predict=Mock(return_value=response()))
    payload = request()
    cache = ChoiceCache(tmp_path)
    scores = cache.scores(backend, payload)
    scores[0] = 999
    np.testing.assert_allclose(cache.scores(backend, payload), [0.2, 0.8])
    np.testing.assert_allclose(ChoiceCache(tmp_path).scores(backend, payload), [0.2, 0.8])
    assert backend.predict.call_count == 1
    payload["state"]["shared_training_history"][0]["observed_utility"]["model_0"] = 1.0
    cache.scores(backend, payload)
    backend.cache_identity = {"model": "v2"}
    cache.scores(backend, payload)
    assert backend.predict.call_count == 3
    for file in tmp_path.glob("*.json"):
        assert "training query" not in file.read_text()


def test_invalid_decisions_are_never_cached(tmp_path):
    backend = SimpleNamespace(cache_identity="v1", predict=Mock(return_value={}))
    with pytest.raises(ValueError):
        ChoiceCache(tmp_path).scores(backend, request())
    assert not list(tmp_path.iterdir())


def test_multimodal_request_maps_images_and_hides_local_paths(tmp_path):
    paths = []
    for index, color in enumerate(["red", "blue"]):
        path = tmp_path / f"image-{index}.png"
        Image.new("RGB", (32, 32), color).save(path)
        paths.append(path)
    payload = request(target_image=paths[0], history_images=paths[1:])
    original = copy.deepcopy(payload)
    assert payload["state"]["target_image"] == {"image_index": 0}
    assert payload["state"]["shared_training_history"][0]["image"] == {"image_index": 1}
    assert str(tmp_path) not in json.dumps(payload)
    wire = JevBackend._wire_request(payload)
    assert payload == original
    assert "images" not in wire and len(wire["assets"]) == 2
    assert wire["state"]["target_image"] in wire["assets"]
    assert wire["state"]["shared_training_history"][0]["image"] in wire["assets"]


def test_jev_requires_environment_key_without_making_a_call(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    call = Mock()
    monkeypatch.setattr("methods.system_one.backends._https_urlopen", call)
    with pytest.raises(RuntimeError, match="TYPESAFE_API_KEY"):
        JevBackend().predict(request())
    call.assert_not_called()


def test_jev_http_request_and_bounded_retries(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    call = Mock(side_effect=[URLError("temporary"), io.StringIO(json.dumps(response()))])
    sleep = Mock()
    monkeypatch.setattr("methods.system_one.backends._https_urlopen", call)
    monkeypatch.setattr("methods.system_one.backends.time.sleep", sleep)
    assert JevBackend(retries=1).predict(request()) == response()
    assert call.call_count == 2
    sleep.assert_called_once()
    http_request = call.call_args.args[0]
    assert http_request.get_header("Authorization") == "Bearer test-key"
    assert "test-key" not in http_request.data.decode()


def test_jev_auth_failure_is_not_retried_or_exposed(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "test-key")
    call = Mock(side_effect=HTTPError("https://invalid.example", 401, "test-key", {}, None))
    monkeypatch.setattr("methods.system_one.backends._https_urlopen", call)
    with pytest.raises(RuntimeError, match="HTTP 401") as error:
        JevBackend().predict(request())
    assert "test-key" not in str(error.value)
    assert call.call_count == 1


def fake_clef(max_length=16):
    backend = ClefFlashBackend(device="cpu")
    question = SimpleNamespace(question_id="selected_model", option_ids=["model_0", "model_1"])
    backend._runtime = SimpleNamespace(
        encode_record=Mock(return_value=SimpleNamespace(input_ids=[1] * 8, questions=[question])),
        collate_records=Mock(return_value="batch"),
    )
    backend._processor = SimpleNamespace(tokenizer=SimpleNamespace(pad_token_id=0))
    backend._network = Mock(return_value=[[torch.tensor([0.0, 1.0])]])
    backend._context_limit = max_length
    return backend


def test_native_clef_scores_schema_logits_and_preserves_complete_evidence():
    backend = fake_clef()
    reply = backend.predict(request())
    values = parse_choice(reply, ["model_0", "model_1"])
    np.testing.assert_allclose(values, torch.tensor([0.0, 1.0]).softmax(-1).numpy())
    assert backend._runtime.encode_record.call_args.kwargs["max_length"] == 2**31 - 1


def test_native_clef_rejects_context_overflow_before_forward():
    backend = fake_clef(max_length=4)
    with pytest.raises(ValueError, match="no truncation"):
        backend.predict(request())
    backend._network.assert_not_called()


def test_native_clef_decodes_images_without_changing_the_input_request(tmp_path):
    path = tmp_path / "image.png"
    Image.new("RGB", (32, 32), "green").save(path)
    payload = request(target_image=path, history_images=[path])
    before = copy.deepcopy(payload)
    backend = fake_clef()
    backend.predict(payload)
    record = backend._runtime.encode_record.call_args.args[1]
    assert all(isinstance(image, Image.Image) for image in record["images"])
    assert payload == before

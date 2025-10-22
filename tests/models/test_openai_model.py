from __future__ import annotations

from types import MethodType, SimpleNamespace
from typing import Iterable, Iterator

import pytest

from models.model import ModelInput, SpeechStructure
from models.openai_model import EmptyModelOutputError, OpenAIModel
from prompts import RoleType


def _make_completion(message: str, completion_tokens: int) -> SimpleNamespace:
    return SimpleNamespace(
        usage=SimpleNamespace(completion_tokens=completion_tokens),
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=message),
                logprobs=None,
            )
        ],
    )


def _attach_stubbed_call_openai(model: OpenAIModel, responses: Iterable[SimpleNamespace]) -> None:
    response_iter: Iterator[SimpleNamespace] = iter(responses)

    def _stub_call_openai(
        self: OpenAIModel,
        messages: list[dict[str, str]],
        speech_structure: SpeechStructure,
        max_new_tokens: int,
    ) -> SimpleNamespace:
        return next(response_iter)

    model.call_openai = MethodType(_stub_call_openai, model)


def _make_model(monkeypatch: pytest.MonkeyPatch) -> OpenAIModel:
    class _FakeOpenAIClient:
        pass

    monkeypatch.setattr("models.openai_model.openai.OpenAI", lambda: _FakeOpenAIClient())
    return OpenAIModel(alias="test")


def _single_model_input(message: str) -> list[ModelInput]:
    return [ModelInput(role=RoleType.USER, content=message)]


def test_predict_single_input_returns_failed_response_after_first_empty_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    model: OpenAIModel = _make_model(monkeypatch)
    _attach_stubbed_call_openai(model, [_make_completion("", 0)])

    response = model.predict_single_input(_single_model_input("hi"))

    assert response.failed is True
    assert model._empty_output_streak == 1


def test_predict_single_input_raises_after_thirteen_consecutive_empty_completions(monkeypatch: pytest.MonkeyPatch) -> None:
    model: OpenAIModel = _make_model(monkeypatch)
    _attach_stubbed_call_openai(model, [_make_completion("", 0) for _ in range(13)])

    for _ in range(12):
        response = model.predict_single_input(_single_model_input("hi"))
        assert response.failed is True

    with pytest.raises(EmptyModelOutputError):
        model.predict_single_input(_single_model_input("hi"))

    assert model._empty_output_streak == 13


def test_predict_single_input_resets_streak_after_non_empty_completion(monkeypatch: pytest.MonkeyPatch) -> None:
    model: OpenAIModel = _make_model(monkeypatch)
    _attach_stubbed_call_openai(
        model,
        [
            _make_completion("", 0),
            _make_completion("hello world", 5),
        ],
    )

    first_response = model.predict_single_input(_single_model_input("hi"))
    assert first_response.failed is True
    assert model._empty_output_streak == 1

    second_response = model.predict_single_input(_single_model_input("hi"))
    assert second_response.failed is False
    assert second_response.speech == "hello world"
    assert model._empty_output_streak == 0

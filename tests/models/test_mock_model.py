from __future__ import annotations

import pytest

from models.mock_model import MockModel
from models.model import ModelInput, SpeechStructure
from prompts import RoleType


def make_input(content: str) -> list[list[ModelInput]]:
    return [[ModelInput(role=RoleType.USER, content=content)]]


def test_predict_returns_mock_response():
    model = MockModel(alias="mock")
    responses = model.predict(make_input("hello world"))

    assert len(responses) == 1
    assert responses[0].speech == "<mock response for hello world>"
    assert responses[0].prompt == "hello world"


def test_predict_truncates_to_250_chars():
    model = MockModel(alias="mock")
    content = "a" * 300
    responses = model.predict(make_input(content))

    assert responses[0].speech == f"<mock response for {content[:250]}>"


def test_predict_respects_num_return_sequences():
    model = MockModel(alias="mock")
    responses = model.predict(make_input("foo"), num_return_sequences=2)

    assert len(responses) == 2
    assert all(r.speech == "<mock response for foo>" for r in responses)


def test_predict_handles_multi_batch_inputs():
    model = MockModel(alias="mock")
    inputs = [
        [ModelInput(role=RoleType.USER, content="first")],
        [
            ModelInput(role=RoleType.SYSTEM, content="system message"),
            ModelInput(role=RoleType.USER, content="second user message"),
        ],
    ]

    responses = model.predict(inputs)

    assert len(responses) == 2
    assert responses[0].speech == "<mock response for first>"
    assert responses[1].speech == "<mock response for second user message>"
    assert responses[1].prompt == "system message\nsecond user message"


def test_predict_raises_for_conflicting_batch_and_num_sequences():
    model = MockModel(alias="mock")
    inputs = [
        [ModelInput(role=RoleType.USER, content="first")],
        [ModelInput(role=RoleType.USER, content="second")],
    ]

    with pytest.raises(ValueError):
        model.predict(inputs, num_return_sequences=2)


def test_predict_raises_for_decision_speech_structure():
    model = MockModel(alias="mock")
    with pytest.raises(ValueError):
        model.predict(make_input("hello"), speech_structure=SpeechStructure.DECISION)


def test_predict_handles_empty_inputs():
    model = MockModel(alias="mock")
    responses = model.predict([], num_return_sequences=1)

    assert len(responses) == 1
    assert responses[0].speech == "<mock response for >"
    assert responses[0].prompt == ""

from __future__ import annotations

import pytest

from models.mock_model import MockModel
from models.model import ModelInput, SpeechStructure
from prompts import RoleType
from utils import constants


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


def test_predict_returns_decision_response_for_decision_speech_structure() -> None:
    model = MockModel(alias="mock")
    message_content: str = (
        "Debater_B establishes point one. Debater_B strengthens with point two. "
        "Debater_B closes stronger than Debater_A."
    )
    responses = model.predict(
        make_input(message_content),
        speech_structure=SpeechStructure.DECISION,
        num_return_sequences=2,
    )

    assert len(responses) == 2
    for response in responses:
        assert response.decision == constants.DEFAULT_DEBATER_B_NAME
        assert response.probabilistic_decision is not None
        assert response.probabilistic_decision[constants.DEFAULT_DEBATER_A_NAME] == pytest.approx(1.0 / 3.0, rel=1e-6)
        assert response.probabilistic_decision[constants.DEFAULT_DEBATER_B_NAME] == pytest.approx(2.0 / 3.0, rel=1e-6)
        assert response.prompt == message_content


def test_predict_handles_empty_inputs():
    model = MockModel(alias="mock")
    responses = model.predict([], num_return_sequences=1)

    assert len(responses) == 1
    assert responses[0].speech == "<mock response for >"
    assert responses[0].prompt == ""

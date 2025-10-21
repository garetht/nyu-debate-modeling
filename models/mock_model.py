from __future__ import annotations

from typing import Dict, List

import utils.constants as constants

from models.model import Model, ModelInput, ModelResponse, SpeechStructure


class MockModel(Model):
    """A simple mock implementation of the Model interface used for testing."""

    def predict(
        self,
        inputs: List[List[ModelInput]],
        max_new_tokens: int = 250,
        speech_structure: SpeechStructure = SpeechStructure.OPEN_ENDED,
        num_return_sequences: int = 1,
        **kwargs,
    ) -> List[ModelResponse]:
        """Return a canned response that echoes the first 250 characters of the latest input."""
        if len(inputs) > 1 and num_return_sequences > 1:
            raise ValueError(
                f"Cannot handle len(inputs) ({len(inputs)}) > 1 together with num_return_sequences ({num_return_sequences}) > 1."
            )

        def get_batch_inputs(batch_index: int) -> List[ModelInput]:
            if not inputs:
                return []
            if batch_index < len(inputs):
                return inputs[batch_index]
            return inputs[-1]

        def create_open_ended_response(batch_index: int) -> ModelResponse:
            batch_inputs: List[ModelInput] = get_batch_inputs(batch_index)
            latest_content: str = batch_inputs[-1].content if batch_inputs else ""
            truncated_content: str = latest_content[:250]
            speech: str = f"<mock response for {truncated_content}>"
            prompt: str = "\n".join(model_input.content for model_input in batch_inputs)
            return ModelResponse(speech=speech, prompt=prompt)

        def create_decision_response(batch_index: int) -> ModelResponse:
            batch_inputs: List[ModelInput] = get_batch_inputs(batch_index)
            prompt: str = "\n".join(model_input.content for model_input in batch_inputs)
            prompt_text: str = prompt.lower()
            debater_a_mentions: int = prompt_text.count(constants.DEFAULT_DEBATER_A_NAME.lower())
            debater_b_mentions: int = prompt_text.count(constants.DEFAULT_DEBATER_B_NAME.lower())
            total_mentions: int = debater_a_mentions + debater_b_mentions

            if total_mentions > 0:
                debater_a_score: float = float(debater_a_mentions + 1)
                debater_b_score: float = float(debater_b_mentions + 1)
            else:
                prompt_hash_seed: int = sum(ord(character) for character in prompt)
                normalized_hash: float = (prompt_hash_seed % 1000) / 1000.0
                debater_a_score = 1.0 + normalized_hash
                debater_b_score = 1.0 + (1.0 - normalized_hash)

            total_score: float = debater_a_score + debater_b_score

            if total_score == 0.0:
                debater_a_probability: float = 0.5
                debater_b_probability: float = 0.5
            else:
                debater_a_probability = debater_a_score / total_score
                debater_b_probability = debater_b_score / total_score

            decision: str = (
                constants.DEFAULT_DEBATER_A_NAME
                if debater_a_probability >= debater_b_probability
                else constants.DEFAULT_DEBATER_B_NAME
            )
            probabilistic_decision: Dict[str, float] = {
                constants.DEFAULT_DEBATER_A_NAME: debater_a_probability,
                constants.DEFAULT_DEBATER_B_NAME: debater_b_probability,
            }

            return ModelResponse(decision=decision, probabilistic_decision=probabilistic_decision, prompt=prompt)

        if len(inputs) > 1:
            if speech_structure == SpeechStructure.DECISION:
                return [create_decision_response(i) for i in range(len(inputs))]
            return [create_open_ended_response(i) for i in range(len(inputs))]

        num_return_sequences = max(num_return_sequences, 1)
        if speech_structure == SpeechStructure.DECISION:
            return [create_decision_response(0) for _ in range(num_return_sequences)]

        return [create_open_ended_response(0) for _ in range(num_return_sequences)]

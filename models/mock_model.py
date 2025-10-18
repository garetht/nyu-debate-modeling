from __future__ import annotations

from typing import List

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

        if speech_structure == SpeechStructure.DECISION:
            raise ValueError("MockModel only supports open-ended speech generation.")

        def create_response(batch_index: int) -> ModelResponse:
            if not inputs:
                batch_inputs: List[ModelInput] = []
            elif batch_index < len(inputs):
                batch_inputs = inputs[batch_index]
            else:
                batch_inputs = inputs[-1]
            latest_content = batch_inputs[-1].content if batch_inputs else ""
            truncated = latest_content[:250]
            speech = f"<mock response for {truncated}>"
            prompt = "\n".join([model_input.content for model_input in batch_inputs])
            return ModelResponse(speech=speech, prompt=prompt)

        if len(inputs) > 1:
            return [create_response(i) for i in range(len(inputs))]

        num_return_sequences = max(num_return_sequences, 1)
        return [create_response(0) for _ in range(num_return_sequences)]

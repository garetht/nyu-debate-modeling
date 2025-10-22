from __future__ import annotations

from types import SimpleNamespace
from typing import Dict, List, cast

import io
import json
import logging
import uuid

import openai
import pytest

from batch_utilities.batch_runner import OpenAIBatchRunner
from models.model import RoleType, SpeechStructure


class FakeFiles:
    def __init__(self) -> None:
        self.created_inputs: Dict[str, bytes] = {}
        self.output_payloads: Dict[str, bytes] = {}

    def create(self, *, file: tuple[str, io.BytesIO], purpose: str) -> SimpleNamespace:
        file_name, buffer = file
        file_id = f"{file_name}-{uuid.uuid4().hex}"
        self.created_inputs[file_id] = buffer.getvalue()
        return SimpleNamespace(id=file_id)

    def content(self, file_id: str) -> io.BytesIO:
        return io.BytesIO(self.output_payloads[file_id])


class FakeBatches:
    def __init__(self, files: FakeFiles) -> None:
        self._files = files
        self._batch_calls: Dict[str, int] = {}
        self._input_to_output: Dict[str, str] = {}

    def create(self, *, input_file_id: str, endpoint: str, completion_window: str) -> SimpleNamespace:
        batch_id = f"batch-{uuid.uuid4().hex}"
        output_file_id = f"output-{uuid.uuid4().hex}"
        self._batch_calls[batch_id] = 0
        self._input_to_output[batch_id] = output_file_id
        self._files.output_payloads[output_file_id] = self._build_output_bytes(input_file_id)
        return SimpleNamespace(id=batch_id, status="in_progress", output_file_id=None)

    def retrieve(self, batch_id: str) -> SimpleNamespace:
        call_count = self._batch_calls.get(batch_id, 0)
        output_file_id = self._input_to_output[batch_id]
        if call_count == 0:
            self._batch_calls[batch_id] = 1
            return SimpleNamespace(id=batch_id, status="in_progress", output_file_id=None)
        return SimpleNamespace(id=batch_id, status="completed", output_file_id=output_file_id)

    def _build_output_bytes(self, input_file_id: str) -> bytes:
        input_bytes = self._files.created_inputs[input_file_id]
        lines = input_bytes.decode("utf-8").splitlines()
        output_lines: List[str] = []
        for line in lines:
            request = json.loads(line)
            body = {
                "id": f"chatcmpl-{uuid.uuid4().hex}",
                "object": "chat.completion",
                "created": 0,
                "model": "gpt-4-0125-preview",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello world"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"completion_tokens": 1, "prompt_tokens": 1, "total_tokens": 2},
            }
            payload = {
                "custom_id": request["custom_id"],
                "response": {"body": body},
            }
            output_lines.append(json.dumps(payload))
        return ("\n".join(output_lines)).encode("utf-8")


class FakeOpenAI:
    def __init__(self) -> None:
        self.files = FakeFiles()
        self.batches = FakeBatches(self.files)


def make_message(content: str) -> Dict[str, str]:
    return {"role": RoleType.USER.name.lower(), "content": content}


def test_single_request_batch_execution() -> None:
    client = FakeOpenAI()
    logger = logging.getLogger("test_single_request_batch_execution")
    runner = OpenAIBatchRunner(
        client=cast(openai.OpenAI, client),
        endpoint="gpt-4-0125-preview",
        logger=logger,
        batch_size=1,
        poll_interval_seconds=0.0,
        queue_timeout_seconds=0.0,
        sleep_interval_seconds=0.0,
    )

    response = runner.execute(
        messages=[make_message("Hello")],
        max_new_tokens=10,
        speech_structure=SpeechStructure.OPEN_ENDED,
    )

    assert response.choices[0].message.content == "Hello world"


def test_failed_batch_request_sets_exception() -> None:
    class ErrorFiles(FakeFiles):
        def content(self, file_id: str) -> io.BytesIO:
            payload = json.dumps(
                {
                    "custom_id": "custom-id",
                    "response": {"error": {"message": "boom"}},
                }
            ).encode("utf-8")
            return io.BytesIO(payload)

    class ErrorBatches(FakeBatches):
        def create(self, *, input_file_id: str, endpoint: str, completion_window: str) -> SimpleNamespace:
            batch_id = f"batch-{uuid.uuid4().hex}"
            self._batch_calls[batch_id] = 0
            self._input_to_output[batch_id] = "error-output"
            return SimpleNamespace(id=batch_id, status="in_progress", output_file_id=None)

        def retrieve(self, batch_id: str) -> SimpleNamespace:
            if self._batch_calls.get(batch_id, 0) == 0:
                self._batch_calls[batch_id] = 1
                return SimpleNamespace(id=batch_id, status="in_progress", output_file_id=None)
            return SimpleNamespace(id=batch_id, status="completed", output_file_id="error-output")

    class ErrorOpenAI(FakeOpenAI):
        def __init__(self) -> None:
            self.files = ErrorFiles()
            self.batches = ErrorBatches(self.files)

    client = ErrorOpenAI()
    logger = logging.getLogger("test_failed_batch_request_sets_exception")
    runner = OpenAIBatchRunner(
        client=cast(openai.OpenAI, client),
        endpoint="gpt-4-0125-preview",
        logger=logger,
        batch_size=1,
        poll_interval_seconds=0.0,
        queue_timeout_seconds=0.0,
        sleep_interval_seconds=0.0,
    )

    with pytest.raises(RuntimeError):
        runner.execute(
            messages=[make_message("Hello")],
            max_new_tokens=10,
            speech_structure=SpeechStructure.OPEN_ENDED,
        )

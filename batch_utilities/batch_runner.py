from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import io
import json
import logging
import time
import uuid

import openai
from openai.types import Batch
from openai.types.chat import ChatCompletion

from models.model import SpeechStructure


@dataclass
class PendingBatchRequest:
    custom_id: str
    messages: List[Dict[str, str]]
    max_new_tokens: int
    speech_structure: SpeechStructure
    result: Optional[ChatCompletion] = None
    error: Optional[Exception] = None
    created_at: float = field(default_factory=time.monotonic)


class OpenAIBatchRunner:
    """Collects chat completions and executes them through the OpenAI Batch API synchronously."""

    def __init__(
        self,
        *,
        client: openai.OpenAI,
        endpoint: str,
        logger: logging.Logger,
        batch_size: int = 8,
        poll_interval_seconds: float = 1.0,
        queue_timeout_seconds: float = 0.25,
        sleep_interval_seconds: float = 0.01,
    ) -> None:
        self._client = client
        self._endpoint = endpoint
        self._logger = logger
        self._batch_size = batch_size
        self._poll_interval_seconds = poll_interval_seconds
        self._queue_timeout_seconds = queue_timeout_seconds
        self._sleep_interval_seconds = sleep_interval_seconds
        self._buffer: List[PendingBatchRequest] = []
        self._buffer_created_at: Optional[float] = None

    def execute(
        self,
        *,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        speech_structure: SpeechStructure,
    ) -> ChatCompletion:
        request = PendingBatchRequest(
            custom_id=f"debate-{uuid.uuid4().hex}",
            messages=messages,
            max_new_tokens=max_new_tokens,
            speech_structure=speech_structure,
        )
        self._enqueue(request)
        self._wait_for_completion(request)
        if request.error:
            raise request.error
        if request.result is None:
            raise RuntimeError("Request completed without a result.")
        return request.result

    def _enqueue(self, request: PendingBatchRequest) -> None:
        self._buffer.append(request)
        if self._buffer_created_at is None:
            self._buffer_created_at = request.created_at
        self._flush_if_ready()

    def _wait_for_completion(self, request: PendingBatchRequest) -> None:
        while request.result is None and request.error is None:
            if request in self._buffer:
                if time.monotonic() - (self._buffer_created_at or request.created_at) >= self._queue_timeout_seconds:
                    self._flush_buffer()
                else:
                    time.sleep(self._sleep_interval_seconds)
            else:
                time.sleep(self._sleep_interval_seconds)

    def _flush_if_ready(self) -> None:
        if len(self._buffer) >= self._batch_size:
            self._flush_buffer()
        elif self._buffer_created_at is not None:
            if time.monotonic() - self._buffer_created_at >= self._queue_timeout_seconds:
                self._flush_buffer()

    def _flush_buffer(self) -> None:
        if not self._buffer:
            return
        requests = list(self._buffer)
        self._buffer.clear()
        self._buffer_created_at = None
        self._launch_batch(requests)

    def _launch_batch(self, requests: List[PendingBatchRequest]) -> None:
        self._logger.debug("Launching OpenAI batch with %s requests.", len(requests))
        input_bytes = self._render_requests(requests)
        input_file = self._client.files.create(
            file=(f"batch-{uuid.uuid4().hex}.jsonl", io.BytesIO(input_bytes)),
            purpose="batch",
        )
        batch_job = self._client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
        )
        batch_job = self._poll_batch(batch_job.id)
        if not getattr(batch_job, "output_file_id", None):
            error_message = f"OpenAI batch {batch_job.id} completed without an output file."
            self._fail_requests(requests, RuntimeError(error_message))
            return
        output_content = self._client.files.content(batch_job.output_file_id).read().decode("utf-8")
        responses_by_id = self._parse_output_lines(output_content)
        for request in requests:
            response = responses_by_id.get(request.custom_id)
            if response is None:
                self._record_failure(request, RuntimeError(f"No response found for custom_id={request.custom_id}"))
                continue
            if "error" in response:
                self._record_failure(request, RuntimeError(str(response["error"])))
                continue
            chat_completion = ChatCompletion.model_validate(response["body"])
            request.result = chat_completion
        self._logger.debug("Finished OpenAI batch %s with %s responses.", batch_job.id, len(responses_by_id))

    def _render_requests(self, requests: Iterable[PendingBatchRequest]) -> bytes:
        rendered_lines = []
        for request in requests:
            body = self._build_request_body(request)
            rendered_lines.append(
                json.dumps(
                    {
                        "custom_id": request.custom_id,
                        "method": "POST",
                        "url": "/v1/chat/completions",
                        "body": body,
                    }
                )
            )
        return ("\n".join(rendered_lines)).encode("utf-8")

    def _build_request_body(self, request: PendingBatchRequest) -> Dict[str, object]:
        body: Dict[str, object] = {
            "model": self._endpoint,
            "messages": request.messages,
            "max_completion_tokens": request.max_new_tokens,
            "logprobs": request.speech_structure != SpeechStructure.OPEN_ENDED,
            "top_logprobs": 5 if request.speech_structure != SpeechStructure.OPEN_ENDED else None,
        }
        if "o4" in self._endpoint:
            body["reasoning_effort"] = "low"
        return {key: value for key, value in body.items() if value is not None}

    def _poll_batch(self, batch_id: str) -> Batch:
        batch_job = self._client.batches.retrieve(batch_id)
        while getattr(batch_job, "status", "") in {"validating", "in_progress", "queued"}:
            time.sleep(self._poll_interval_seconds)
            batch_job = self._client.batches.retrieve(batch_id)
        if getattr(batch_job, "status", "") != "completed":
            raise RuntimeError(f"OpenAI batch {batch_id} failed with status={getattr(batch_job, 'status', 'unknown')}")
        return batch_job

    @staticmethod
    def _parse_output_lines(content: str) -> Dict[str, Dict[str, object]]:
        responses: Dict[str, Dict[str, object]] = {}
        for line in content.splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            responses[payload["custom_id"]] = payload["response"]
        return responses

    def _fail_requests(self, requests: Iterable[PendingBatchRequest], error: Exception) -> None:
        for request in requests:
            self._record_failure(request, error)

    @staticmethod
    def _record_failure(request: PendingBatchRequest, error: Exception) -> None:
        request.error = error

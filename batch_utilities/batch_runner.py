from __future__ import annotations

from concurrent.futures import Future
from dataclasses import dataclass
from threading import Lock, Thread
from types import SimpleNamespace
from typing import Any, Dict, Iterable, List, Optional

import io
import json
import logging
import time
import uuid

import openai

from models.model import SpeechStructure


@dataclass(frozen=True)
class PendingBatchRequest:
    """Container for a single pending batch request entry."""

    custom_id: str
    messages: List[Dict[str, str]]
    max_new_tokens: int
    speech_structure: SpeechStructure
    future: Future[SimpleNamespace]


class OpenAIBatchRunner:
    """Collects chat completions and executes them through the OpenAI Batch API."""

    def __init__(
        self,
        *,
        client: openai.OpenAI,
        endpoint: str,
        logger: logging.Logger,
        batch_size: int = 8,
        flush_interval_seconds: float = 0.25,
        poll_interval_seconds: float = 1.0,
    ) -> None:
        self._client = client
        self._endpoint = endpoint
        self._logger = logger
        self._batch_size = batch_size
        self._flush_interval_seconds = flush_interval_seconds
        self._poll_interval_seconds = poll_interval_seconds
        self._lock = Lock()
        self._pending: List[PendingBatchRequest] = []
        self._flush_thread: Optional[Thread] = None

    def execute(
        self,
        *,
        messages: List[Dict[str, str]],
        max_new_tokens: int,
        speech_structure: SpeechStructure,
    ) -> SimpleNamespace:
        future: Future[SimpleNamespace] = Future()
        request = PendingBatchRequest(
            custom_id=f"debate-{uuid.uuid4().hex}",
            messages=messages,
            max_new_tokens=max_new_tokens,
            speech_structure=speech_structure,
            future=future,
        )
        batch = self._enqueue_request(request)
        if batch:
            self._launch_batch(batch)
        return future.result()

    def _enqueue_request(self, request: PendingBatchRequest) -> Optional[List[PendingBatchRequest]]:
        with self._lock:
            self._pending.append(request)
            if len(self._pending) >= self._batch_size:
                return self._drain_pending()
            if self._flush_thread is None:
                self._flush_thread = Thread(target=self._delayed_flush, daemon=True)
                self._flush_thread.start()
        return None

    def _delayed_flush(self) -> None:
        time.sleep(self._flush_interval_seconds)
        batch = self._drain_pending()
        if batch:
            self._launch_batch(batch)

    def _drain_pending(self) -> Optional[List[PendingBatchRequest]]:
        with self._lock:
            if not self._pending:
                self._flush_thread = None
                return None
            pending = list(self._pending)
            self._pending.clear()
            self._flush_thread = None
            return pending

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
                self._fail_request(
                    request,
                    RuntimeError(f"No response found for custom_id={request.custom_id} in batch {batch_job.id}"),
                )
                continue
            if "error" in response:
                self._fail_request(request, RuntimeError(str(response["error"])))
                continue
            chat_completion = self._to_namespace(response["body"])
            request.future.set_result(chat_completion)
        self._logger.debug(
            "Finished OpenAI batch %s with %s successful responses.", batch_job.id, len(responses_by_id)
        )

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

    def _build_request_body(self, request: PendingBatchRequest) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": self._endpoint,
            "messages": request.messages,
            "max_completion_tokens": request.max_new_tokens,
            "logprobs": request.speech_structure != SpeechStructure.OPEN_ENDED,
            "top_logprobs": 5 if request.speech_structure != SpeechStructure.OPEN_ENDED else None,
        }
        reasoning_effort: Optional[str] = "low" if "o4" in self._endpoint else None
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        return {key: value for key, value in body.items() if value is not None}

    def _poll_batch(self, batch_id: str) -> Any:
        batch_job = self._client.batches.retrieve(batch_id)
        while getattr(batch_job, "status", "") in {"validating", "in_progress", "queued"}:
            time.sleep(self._poll_interval_seconds)
            batch_job = self._client.batches.retrieve(batch_id)
        if getattr(batch_job, "status", "") != "completed":
            raise RuntimeError(f"OpenAI batch {batch_id} failed with status={getattr(batch_job, 'status', 'unknown')}")
        return batch_job

    def _parse_output_lines(self, content: str) -> Dict[str, Dict[str, Any]]:
        responses: Dict[str, Dict[str, Any]] = {}
        for line in content.splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            responses[payload["custom_id"]] = payload["response"]
        return responses

    def _fail_requests(self, requests: Iterable[PendingBatchRequest], error: Exception) -> None:
        for request in requests:
            self._fail_request(request, error)

    @staticmethod
    def _fail_request(request: PendingBatchRequest, error: Exception) -> None:
        request.future.set_exception(error)

    def _to_namespace(self, value: Any) -> Any:
        if isinstance(value, dict):
            return SimpleNamespace(**{key: self._to_namespace(item) for key, item in value.items()})
        if isinstance(value, list):
            return [self._to_namespace(item) for item in value]
        return value

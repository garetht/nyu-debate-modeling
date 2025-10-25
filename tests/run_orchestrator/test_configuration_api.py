from __future__ import annotations

from typing import Any, Dict

from fastapi.testclient import TestClient

from run_orchestrator.configuration_api import app


def _build_standard_payload() -> Dict[str, Any]:
    return {
        "configurations": [
            {
                "instance_ip": "127.0.0.1",
                "configurations": {
                    "name": "Integration Config",
                    "num_iters": 1,
                    "count": 1,
                },
            }
        ]
    }


def test_configuration_api_allows_cors_preflight() -> None:
    client = TestClient(app)
    response = client.options(
        "/configurations/start",
        headers={
            "origin": "https://example.com",
            "access-control-request-method": "POST",
        },
    )
    assert response.status_code == 200
    allow_origin = response.headers.get("access-control-allow-origin")
    assert allow_origin == "*"
    allow_methods = response.headers.get("access-control-allow-methods", "")
    assert "POST" in allow_methods


def test_configuration_api_response_contains_cors_headers() -> None:
    client = TestClient(app)
    response = client.post(
        "/configurations/start",
        headers={"origin": "https://example.com"},
        json=_build_standard_payload(),
    )
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == "*"

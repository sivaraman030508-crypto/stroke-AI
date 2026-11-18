# Tests for the Eka Care httpx client (mocked transport — no network).
# Safety: no hard-coded credentials, tokens never logged, safe error types.

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import httpx
import pytest

from services.eka_care import (
    AUTH_LOGIN_PATH,
    AUTH_REFRESH_PATH,
    EkaAPIError,
    EkaAuthError,
    EkaCareClient,
    EkaConfigurationError,
)


def make_client(handler, **kwargs):
    kwargs.setdefault("client_id", "test-client-id")
    kwargs.setdefault("client_secret", "test-client-secret")
    kwargs.setdefault("api_key", "test-api-key")
    kwargs.setdefault("base_url", "https://eka.test")
    kwargs.setdefault("transport", httpx.MockTransport(handler))
    return EkaCareClient(**kwargs)


def login_ok(request: httpx.Request) -> httpx.Response:
    return httpx.Response(
        200,
        json={"access_token": "tok-1", "refresh_token": "ref-1",
              "expires_in": 3600},
    )


def test_missing_credentials_raise_configuration_error(monkeypatch):
    monkeypatch.delenv("EKA_CLIENT_ID", raising=False)
    monkeypatch.delenv("EKA_CLIENT_SECRET", raising=False)
    with pytest.raises(EkaConfigurationError):
        EkaCareClient(client_id="", client_secret="")


def test_login_and_start_assessment():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == AUTH_LOGIN_PATH:
            return login_ok(request)
        seen[request.url.path] = request
        if request.url.path == "/assessment/api/v1/init/":
            return httpx.Response(200, json={"assessment_id": "A-42"})
        if request.url.path == "/assessment/api/v1/start/A-42":
            return httpx.Response(
                200,
                json={
                    "questions": [
                        {"qid": 1, "question_text": "Any weakness?",
                         "component_code": "yesno",
                         "component_data": {"options": ["Yes", "No"]},
                         "is_mandatory": True}
                    ],
                    "progress": 0,
                },
            )
        return httpx.Response(404)

    client = make_client(handler)
    result = client.start_assessment(
        patient_pseudonym_id="PT-ABC123", age=70, sex="male",
        presenting_symptoms=["weakness"],
    )
    assert result["assessment_id"] == "A-42"
    start = result["start_response"]
    assert start["questions"][0]["qid"] == 1

    init_request = seen["/assessment/api/v1/init/"]
    body = init_request.content.decode()
    assert "PT-ABC123" in body  # pseudonymous ID only
    assert init_request.headers.get("auth") == "tok-1"
    assert init_request.headers.get("client-id") == "test-client-id"
    client.close()


def test_continue_and_submit_payloads():
    paths = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == AUTH_LOGIN_PATH:
            return login_ok(request)
        paths.append(request.url.path)
        if request.url.path.endswith("/continue/A-1"):
            assert b'"qid"' in request.content
            assert b'"user_response"' in request.content
            return httpx.Response(
                200, json={"questions": [], "progress": 100}
            )
        if request.url.path.endswith("/submit/A-1"):
            return httpx.Response(
                200,
                json={"success": True,
                      "explain": [{"label": "Confirm medication list"}]},
            )
        return httpx.Response(404)

    client = make_client(handler)
    cont = client.continue_assessment("A-1", "1", "Yes")
    assert cont["questions"] == []
    sub = client.submit_assessment("A-1")
    assert sub["success"] is True
    client.close()


def test_401_triggers_single_refresh_retry():
    calls = {"login": 0, "refresh": 0, "data": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == AUTH_LOGIN_PATH:
            calls["login"] += 1
            return httpx.Response(
                200,
                json={"access_token": f"tok-{calls['login']}",
                      "refresh_token": "ref-1", "expires_in": 3600},
            )
        if request.url.path == AUTH_REFRESH_PATH:
            calls["refresh"] += 1
            return httpx.Response(
                200,
                json={"access_token": "tok-refreshed",
                      "refresh_token": "ref-2", "expires_in": 3600},
            )
        if request.url.path == "/assessment/api/v1/search":
            calls["data"] += 1
            if calls["data"] == 1:
                return httpx.Response(401)
            return httpx.Response(200, json={"results": []})
        return httpx.Response(404)

    client = make_client(handler)
    # Prime a token first.
    client.get_access_token()
    data = client.search_symptom_or_diagnosis("weakness")
    assert data == {"results": []}
    assert calls["refresh"] == 1
    assert calls["data"] == 2
    client.close()


def test_auth_failure_raises_auth_error():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == AUTH_LOGIN_PATH:
            return httpx.Response(401)
        return httpx.Response(500)

    client = make_client(handler)
    with pytest.raises(EkaAuthError):
        client.get_access_token()
    client.close()


def test_api_failure_raises_safe_api_error():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == AUTH_LOGIN_PATH:
            return login_ok(request)
        return httpx.Response(500, json={"error": "secret internals"})

    client = make_client(handler)
    with pytest.raises(EkaAPIError) as excinfo:
        client.get_document_status("D-1")
    assert excinfo.value.status_code == 500
    # Safe message: no response body internals leaked.
    assert "secret internals" not in str(excinfo.value)
    client.close()


def test_document_upload_and_status_flow():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == AUTH_LOGIN_PATH:
            return login_ok(request)
        if request.url.path == "/mr/api/v2/docs":
            assert request.url.params.get("dt") == "lr"
            assert request.url.params.get("task") == "smart"
            return httpx.Response(200, json={"document_id": "DOC-9"})
        if request.url.path == "/mr/api/v1/docs/DOC-9/result":
            return httpx.Response(
                200,
                json={"status": "completed",
                      "data": {"output": {"data": []}}},
            )
        return httpx.Response(404)

    client = make_client(handler)
    up = client.upload_medical_document(
        "labs.pdf", b"%PDF-1.4", "application/pdf"
    )
    assert up["document_id"] == "DOC-9"
    status = client.get_document_status("DOC-9")
    assert status["status"] == "completed"
    report = client.get_parsed_report("DOC-9")
    assert report["document_id"] == "DOC-9"
    client.close()


def test_tokens_stay_in_memory_only():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == AUTH_LOGIN_PATH:
            return login_ok(request)
        return httpx.Response(404)

    client = make_client(handler)
    client.get_access_token()
    # Token attributes exist on the client object, not on disk-backed state.
    assert client._access_token == "tok-1"
    assert not hasattr(client, "_token_file")
    client.close()

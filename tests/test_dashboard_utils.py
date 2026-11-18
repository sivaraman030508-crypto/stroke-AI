# Tests for dashboard helpers (pure functions + API client transport).
# Safety: no administration instructions, pseudonymous IDs, safe errors.

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pytest
import requests

from app.dashboard_utils import (
    STATUS_META,
    ApiClientError,
    StrokeApiClient,
    allowed_document,
    audit_export_json,
    build_audit_export,
    format_probability,
    friendly_error,
    patient_pseudonym,
    status_display,
    triage_recommendation,
    window_countdown,
)

# ---------------------------------------------------------------------------
# Status presentation
# ---------------------------------------------------------------------------


def test_status_colors_match_safety_palette():
    assert status_display("HEMORRHAGE_SUSPECTED")["color"] == "#c62828"
    assert status_display("URGENT_RADIOLOGY_REVIEW")["color"] == "#ef6c00"
    assert status_display("INSUFFICIENT_INFORMATION")["color"] == "#616161"
    assert status_display("POTENTIALLY_ELIGIBLE_PENDING_REVIEW")["color"] == "#2e7d32"


def test_unknown_status_defaults_to_review_color():
    meta = status_display("SOMETHING_NEW")
    assert meta["color"] == STATUS_META["URGENT_RADIOLOGY_REVIEW"]["color"]
    assert status_display(None)["label"] == "Urgent radiology review"


def test_recommendations_never_instruct_administration():
    for status in STATUS_META:
        text = triage_recommendation(status).lower()
        assert "administer" not in text
        assert "thrombolysis" not in text or "never" in text


# ---------------------------------------------------------------------------
# Formatting / pseudonymisation / timing
# ---------------------------------------------------------------------------


def test_patient_pseudonym_is_deterministic_and_not_raw_id():
    pseudo = patient_pseudonym("7f3c-study-uuid")
    assert pseudo.startswith("PT-")
    assert pseudo == patient_pseudonym("7f3c-study-uuid")
    assert "7f3c" not in pseudo.lower()
    assert patient_pseudonym("") == "—"


def test_format_probability():
    assert format_probability(0.852) == "85.2%"
    assert format_probability(None) == "—"
    assert format_probability("bad") == "—"


def test_window_countdown_inside_and_outside():
    inside = window_countdown(
        "2026-11-18T06:00:00", "2026-11-18T08:00:00"
    )
    assert inside["elapsed_hours"] == 2.0
    assert inside["within_window"] is True
    assert inside["remaining_hours"] == 2.5

    outside = window_countdown(
        "2026-11-18T00:00:00", "2026-11-18T08:00:00"
    )
    assert outside["within_window"] is False
    assert outside["remaining_hours"] is None

    missing = window_countdown(None, "2026-11-18T08:00:00")
    assert missing["elapsed_hours"] is None
    assert missing["within_window"] is None


# ---------------------------------------------------------------------------
# Document validation (delegated; DICOM must be blocked)
# ---------------------------------------------------------------------------


def test_allowed_document_blocks_dicom_and_allows_pdf():
    ok, reason = allowed_document("ct.dcm", "application/dicom", b"DICM")
    assert ok is False
    assert "never be sent to Eka" in reason

    ok, _ = allowed_document("labs.pdf", "application/pdf", b"%PDF")
    assert ok is True


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


def test_friendly_error_messages():
    assert "uvicorn" in friendly_error(ApiClientError(0, "x"))
    assert "not finished" in friendly_error(ApiClientError(409, "x"))
    assert "Eka Care" in friendly_error(ApiClientError(502, "x"))
    assert "EKA_CLIENT_ID" in friendly_error(ApiClientError(503, "x"))


# ---------------------------------------------------------------------------
# Audit export
# ---------------------------------------------------------------------------


def test_build_audit_export_is_safe_and_serializable():
    export = build_audit_export(
        "study-xyz",
        {
            "imaging_status": "HEMORRHAGE_SUSPECTED",
            "model_a_result": {
                "max_hemorrhagic_probability": 0.91,
                "predicted_class": "hemorrhagic",
            },
            "model_a_version": "stroke-type-triage-v1",
            "thresholds": {"hemorrhage_probability": 0.5},
            "timestamp": "2026-11-18T08:00:00+00:00",
        },
        {"final_status": "HEMORRHAGE_SUSPECTED", "rule_trace": "R001"},
        [{"id": 1, "event_type": "confirmation"}],
    )
    assert export["patient_pseudonym"].startswith("PT-")
    assert "study-xyz" not in export["patient_pseudonym"]
    blob = audit_export_json(export)
    assert "ADMINISTER_THROMBOLYSIS" not in blob
    parsed = json.loads(blob)
    assert parsed["eligibility"]["final_status"] == "HEMORRHAGE_SUSPECTED"
    assert parsed["audit_events"][0]["event_type"] == "confirmation"


# ---------------------------------------------------------------------------
# API client transport (fake session, no network)
# ---------------------------------------------------------------------------


class FakeResponse:
    def __init__(self, status_code=200, payload=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.content = b"{}"
        self.text = json.dumps(self._payload)

    def json(self):
        return self._payload


class FakeSession:
    def __init__(self, response=None, exc=None):
        self.response = response or FakeResponse()
        self.exc = exc
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        if self.exc:
            raise self.exc
        return self.response


def test_client_composes_urls_and_methods():
    session = FakeSession(FakeResponse(200, {"status": "ok"}))
    client = StrokeApiClient(base_url="http://api.test", session=session)
    assert client.health() == {"status": "ok"}
    method, url, _ = session.calls[0]
    assert (method, url) == ("GET", "http://api.test/health")


def test_client_raises_safe_error_on_http_failure():
    session = FakeSession(FakeResponse(404, {"detail": "Study not found"}))
    client = StrokeApiClient(base_url="http://api.test", session=session)
    with pytest.raises(ApiClientError) as excinfo:
        client.study_result("missing")
    assert excinfo.value.status_code == 404
    assert "Study not found" in excinfo.value.detail


def test_client_raises_status_zero_when_unreachable():
    session = FakeSession(exc=requests.ConnectionError("refused"))
    client = StrokeApiClient(base_url="http://api.test", session=session)
    with pytest.raises(ApiClientError) as excinfo:
        client.health()
    assert excinfo.value.status_code == 0


def test_client_eka_methods_call_expected_routes():
    session = FakeSession(FakeResponse(200, {"assessment_id": "A"}))
    client = StrokeApiClient(base_url="http://api.test", session=session)
    client.eka_start({"patient_pseudonym_id": "PT-1", "age": 60})
    client.eka_continue(
        {"assessment_id": "A", "question_id": "1", "answer": "Yes"}
    )
    client.eka_submit({"assessment_id": "A"})
    client.eka_context({"study_id": "S"})
    paths = [url.replace("http://api.test", "") for _, url, _ in session.calls]
    assert paths == [
        "/eka/assessment/start",
        "/eka/assessment/continue",
        "/eka/assessment/submit",
        "/eka/context",
    ]

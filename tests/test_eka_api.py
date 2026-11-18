# Tests for the Eka Care FastAPI endpoints (fake Eka service injected).
# Safety: DICOM rejected with 400; context requires a study; every response
# carries clinician-review flags and disclaimers.

import json
import os
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

_TMP_DIR = tempfile.mkdtemp(prefix="eka_api_test_")
_DB_PATH = os.path.join(_TMP_DIR, "eka_studies.db").replace("\\", "/")
os.environ["STUDIES_DB_URL"] = f"sqlite:///{_DB_PATH}"
os.environ["SKIP_MODEL_LOAD"] = "1"

from fastapi.testclient import TestClient

import app.api_eka as eka_module
from app.api import SessionLocal, StudyModel, app
from app.api_eka import get_eka_service

app.dependency_overrides[get_eka_service] = lambda: object()  # replaced below
client = TestClient(app)


class FakeEka:
    def start_assessment(self, patient_pseudonym_id, age, sex="unknown",
                         presenting_symptoms=None, symptom_onset_time=None):
        assert patient_pseudonym_id.startswith("PT-") or patient_pseudonym_id
        return {
            "assessment_id": "A-100",
            "start_response": {
                "questions": [
                    {
                        "qid": 7,
                        "question_text": "Is there facial droop?",
                        "component_code": "yesno",
                        "component_data": {"options": ["Yes", "No"]},
                        "is_mandatory": True,
                    }
                ],
                "progress": 10,
                "is_last_question": False,
            },
            "presenting_symptoms": list(presenting_symptoms or []),
            "symptom_onset_time": symptom_onset_time,
        }

    def continue_assessment(self, assessment_id, question_id, answer):
        return {"questions": [], "progress": 100, "is_last_question": True}

    def submit_assessment(self, assessment_id):
        return {
            "success": True,
            "assessment_id": assessment_id,
            "explain": [{"label": "Confirm anticoagulant history"}],
        }

    def upload_medical_document(self, filename, content, content_type="application/pdf",
                                document_type="lr", task="smart"):
        assert content  # never empty
        return {"document_id": "DOC-1"}

    def get_document_status(self, document_id):
        return {"document_id": document_id, "status": "completed", "raw": {}}

    def get_parsed_report(self, document_id):
        return {
            "document_id": document_id,
            "status": "completed",
            "data": {
                "output": {
                    "data": [
                        {"test_name": "INR", "data": {"value": 1.3}},
                    ]
                },
                "pii": {"medications": ["Apixaban 5mg"]},
            },
            "history": {"prior_intracranial_hemorrhage": False},
        }


app.dependency_overrides[get_eka_service] = lambda: FakeEka()

# ---------------------------------------------------------------------------
# Assessment flow
# ---------------------------------------------------------------------------


def test_assessment_start_returns_question():
    resp = client.post(
        "/eka/assessment/start",
        json={
            "patient_pseudonym_id": "PT-DEADBEEF01",
            "age": 72,
            "sex": "female",
            "presenting_symptoms": ["face droop"],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["assessment_id"] == "A-100"
    assert data["question"]["id"] == "7"
    assert "facial droop" in data["question"]["text"]
    assert data["question"]["options"] == ["Yes", "No"]
    assert data["requires_clinician_review"] is True
    assert "not a diagnosis" in data["disclaimer"].lower()


def test_assessment_continue_completes_without_next_question():
    resp = client.post(
        "/eka/assessment/continue",
        json={"assessment_id": "A-100", "question_id": "7", "answer": "Yes"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "completed"
    assert data["question"] is None


def test_assessment_submit_returns_unconfirmed_summary():
    resp = client.post(
        "/eka/assessment/submit", json={"assessment_id": "A-100"}
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["result"]["success"] is True
    assert data["requires_clinician_review"] is True
    assert "thrombolysis eligibility" in data["disclaimer"].lower()


# ---------------------------------------------------------------------------
# Document endpoints
# ---------------------------------------------------------------------------


def test_document_upload_rejects_dicom():
    resp = client.post(
        "/eka/documents/upload",
        files={"file": ("ct_slice.dcm", b"\x00" * 128 + b"DICM" + b"\x00" * 8,
                        "application/dicom")},
        data={"study_id": "S-1"},
    )
    assert resp.status_code == 400
    assert "never be sent to Eka" in resp.json()["detail"]


def test_document_upload_rejects_disallowed_type():
    resp = client.post(
        "/eka/documents/upload",
        files={"file": ("notes.txt", b"plain text", "text/plain")},
    )
    assert resp.status_code == 400


def test_document_upload_accepts_pdf():
    resp = client.post(
        "/eka/documents/upload",
        files={"file": ("labs.pdf", b"%PDF-1.4 content", "application/pdf")},
        data={"study_id": "S-1"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["document_id"] == "DOC-1"
    assert data["requires_clinician_confirmation"] is True
    assert data["disclaimer"]


def test_document_status_and_parsed():
    status = client.get("/eka/documents/DOC-1/status")
    assert status.status_code == 200
    assert status.json()["status"] == "completed"
    assert status.json()["requires_clinician_confirmation"] is True

    parsed = client.get("/eka/documents/DOC-1/parsed")
    assert parsed.status_code == 200
    data = parsed.json()
    assert data["fields"]["lab_values"]["inr"]["value"] == 1.3
    assert data["checklist_suggestions"]["anticoagulant_status"] == "positive"
    assert data["requires_clinician_confirmation"] is True


def test_document_unknown_id_returns_404():
    resp = client.get("/eka/documents/NOPE/status")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Combined context
# ---------------------------------------------------------------------------


def _seed_study(study_id: str, imaging_status: str = "NO_ACUTE_STROKE_EVIDENCE"):
    db = SessionLocal()
    try:
        existing = db.query(StudyModel).filter(StudyModel.id == study_id).first()
        if existing is None:
            db.add(
                StudyModel(
                    id=study_id,
                    status="completed",
                    result_json=json.dumps(
                        {
                            "study_id": study_id,
                            "imaging_status": imaging_status,
                            "model_a_result": {"max_hemorrhagic_probability": 0.1},
                            "model_b_result": None,
                            "top_slices": [],
                            "model_a_version": "stroke-type-triage-v1",
                            "timestamp": "2026-11-18T08:00:00+00:00",
                            "disclaimer": "test",
                        }
                    ),
                )
            )
            db.commit()
    finally:
        db.close()


def test_context_requires_study():
    resp = client.post(
        "/eka/context", json={"study_id": "does-not-exist"}
    )
    assert resp.status_code == 404


def test_context_builds_unconfirmed_engine_inputs():
    _seed_study("S-CTX-1")
    # Ensure a parsed document is registered.
    client.post(
        "/eka/documents/upload",
        files={"file": ("labs.pdf", b"%PDF-1.4", "application/pdf")},
    )
    client.get("/eka/documents/DOC-1/parsed")

    resp = client.post(
        "/eka/context",
        json={
            "study_id": "S-CTX-1",
            "assessment_id": None,
            "document_ids": ["DOC-1"],
            "clinician_confirmed": False,
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["study_id"] == "S-CTX-1"
    assert data["imaging_status"] == "NO_ACUTE_STROKE_EVIDENCE"
    assert data["requires_clinician_confirmation"] is True
    assert data["requires_clinician_review"] is True
    assert data["clinician_confirmed"] is False
    assert "final_status" not in data
    engine_inputs = data["eka_engine_inputs"]
    assert engine_inputs["eka_extracted_anticoagulant_use"] == "positive"
    assert engine_inputs["eka_extracted_lab_values"]["inr"] == 1.3
    assert engine_inputs["eka_extracted_prior_hemorrhage_history"] is False
    # Unknown/missing documents become missing, never invented facts.
    resp2 = client.post(
        "/eka/context",
        json={"study_id": "S-CTX-1", "document_ids": ["MISSING-DOC"]},
    )
    assert resp2.status_code == 200
    assert "MISSING-DOC" in resp2.json()["uncertain_or_missing"]


def test_context_with_assessment_completed_flag():
    _seed_study("S-CTX-2")
    resp = client.post(
        "/eka/context",
        json={
            "study_id": "S-CTX-2",
            "assessment_result": {
                "completed": True,
                "summary": "done",
                "suggested_considerations": ["verify BP"],
            },
            "document_ids": [],
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["eka_assessment"]["completed"] is True
    assert data["eka_engine_inputs"]["eka_assessment_completed"] is True
    assert data["eka_engine_inputs"]["eka_extracted_lab_values"] is None


# ---------------------------------------------------------------------------
# Unconfigured Eka -> 503 (no credentials in the environment)
# ---------------------------------------------------------------------------


def test_unconfigured_eka_returns_503(monkeypatch):
    monkeypatch.setattr(eka_module, "_eka_client", None)
    monkeypatch.delenv("EKA_CLIENT_ID", raising=False)
    monkeypatch.delenv("EKA_CLIENT_SECRET", raising=False)
    saved = app.dependency_overrides.pop(get_eka_service, None)
    try:
        resp = client.post(
            "/eka/assessment/start",
            json={"patient_pseudonym_id": "PT-X", "age": 60},
        )
        assert resp.status_code == 503
        assert "EKA_CLIENT_ID" in resp.json()["detail"]
    finally:
        if saved is not None:
            app.dependency_overrides[get_eka_service] = saved

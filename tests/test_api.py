# Pytest tests for the Stroke CT Triage Assistant FastAPI application.
# Uses TestClient with mocked model inference; no real checkpoints are loaded.

import json
import os
import sys
import tempfile
import uuid

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Point the API at a throwaway SQLite database BEFORE importing app.api.
_TMP_DIR = tempfile.mkdtemp(prefix="stroke_api_test_")
_DB_PATH = os.path.join(_TMP_DIR, "test_studies.db").replace("\\", "/")
os.environ["STUDIES_DB_URL"] = f"sqlite:///{_DB_PATH}"
os.environ["SKIP_MODEL_LOAD"] = "1"

import torch
from fastapi.testclient import TestClient

import app.api as api_module
from app.api import (
    AuditEventModel,
    SessionLocal,
    StudyModel,
    app,
    process_study,
    run_model_b_on_slices,
)

client = TestClient(app)


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class MockModelAService:
    model_version = "stroke-type-triage-v1"

    def predict(self, tensor):
        return {
            "predicted_class": "hemorrhagic",
            "confidence": 0.85,
            "class_probabilities": {
                "normal": 0.05,
                "ischemic": 0.10,
                "hemorrhagic": 0.85,
            },
            "model_version": self.model_version,
            "inference_timestamp": "2026-01-01T00:00:00Z",
            "disclaimer": api_module.DISCLAIMER,
        }


class MockModelBService:
    model_version = "STROKE_AI_Model_B_Final"

    def predict(self, tensor):
        return {
            "subtype_probabilities": {
                "epidural": 0.02,
                "intraparenchymal": 0.78,
                "intraventricular": 0.05,
                "subarachnoid": 0.05,
                "subdural": 0.10,
            },
            "highest_probability_subtype": "intraparenchymal",
            "model_version": self.model_version,
            "inference_timestamp": "2026-01-01T00:00:00Z",
            "disclaimer": api_module.DISCLAIMER,
        }


api_module._model_a_service = MockModelAService()
api_module._model_b_service = MockModelBService()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_result(study_id, imaging_status):
    hemorrhagic = imaging_status == "HEMORRHAGE_SUSPECTED"
    return {
        "study_id": study_id,
        "model_a_result": {
            "max_hemorrhagic_probability": 0.85 if hemorrhagic else 0.05,
            "max_ischemic_probability": 0.10,
            "max_normal_probability": 0.05,
            "predicted_class": "hemorrhagic" if hemorrhagic else "normal",
            "top_3_hemorrhagic_slices": [0, 1, 2] if hemorrhagic else [],
            "processed_slices": 3,
            "failed_slices": 0,
            "slice_details": [],
        },
        "model_b_result": None,
        "top_slices": [0, 1, 2] if hemorrhagic else [],
        "imaging_status": imaging_status,
        "model_a_version": "stroke-type-triage-v1",
        "model_b_version": "STROKE_AI_Model_B_Final",
        "thresholds": {"hemorrhage_probability": 0.5, "low_confidence": 0.5},
        "timestamp": "2026-01-01T00:00:00Z",
        "disclaimer": api_module.DISCLAIMER,
    }


def _insert_study(study_id=None, status="completed",
                  imaging_status="HEMORRHAGE_SUSPECTED"):
    study_id = study_id or str(uuid.uuid4())
    db = SessionLocal()
    try:
        result_json = None
        if status == "completed":
            result_json = json.dumps(_build_result(study_id, imaging_status))
        db.add(StudyModel(id=study_id, status=status, result_json=result_json))
        db.commit()
    finally:
        db.close()
    return study_id


def _eligibility_payload(study_id, **overrides):
    payload = {
        "study_id": study_id,
        "lkw_time": "2026-11-18T08:00:00",
        "current_time": "2026-11-18T10:00:00",
        "image_quality": "good",
        "severe_uncontrolled_hypertension": False,
        "prior_intracranial_hemorrhage": False,
        "active_bleeding": False,
        "recent_major_surgery_or_trauma": False,
        "anticoagulant_status": "negative",
        "recent_ischemic_stroke_or_head_injury": False,
    }
    payload.update(overrides)
    return payload


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:
    def test_health_ok(self):
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"
        assert data["model_a_loaded"] is True
        assert data["model_b_loaded"] is True
        assert data["model_a_version"] == "stroke-type-triage-v1"
        assert data["model_b_version"] == "STROKE_AI_Model_B_Final"
        assert "not clinically validated" in data["disclaimer"]
        assert "ADMINISTER_THROMBOLYSIS" not in json.dumps(data)


# ---------------------------------------------------------------------------
# Upload / status / result
# ---------------------------------------------------------------------------


class TestStudyEndpoints:
    def test_upload_requires_files(self):
        response = client.post("/studies/upload")
        assert response.status_code == 422

    def test_upload_returns_pseudonymous_study_id(self):
        files = [
            ("files", ("a.dcm", b"not-a-dicom", "application/dicom")),
            ("files", ("b.dcm", b"also-not-a-dicom", "application/dicom")),
        ]
        response = client.post("/studies/upload", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "queued"
        study_id = data["study_id"]
        assert study_id
        assert "patient" not in json.dumps(data).lower()

        # Background task runs synchronously under TestClient; unreadable
        # slices must NOT be reported as normal — the study fails safely.
        status_resp = client.get(f"/studies/{study_id}/status")
        assert status_resp.status_code == 200
        assert status_resp.json()["status"] == "failed"

        result_resp = client.get(f"/studies/{study_id}/result")
        assert result_resp.status_code == 400
        assert "ADMINISTER_THROMBOLYSIS" not in result_resp.text

    def test_status_unknown_study(self):
        response = client.get(f"/studies/{uuid.uuid4()}/status")
        assert response.status_code == 404

    def test_result_unknown_study(self):
        response = client.get(f"/studies/{uuid.uuid4()}/result")
        assert response.status_code == 404

    def test_result_not_completed(self):
        study_id = _insert_study(status="queued")
        response = client.get(f"/studies/{study_id}/result")
        assert response.status_code == 409

    def test_result_completed_returns_stored_payload(self):
        study_id = _insert_study(imaging_status="HEMORRHAGE_SUSPECTED")
        response = client.get(f"/studies/{study_id}/result")
        assert response.status_code == 200
        data = response.json()
        assert data["study_id"] == study_id
        assert data["imaging_status"] == "HEMORRHAGE_SUSPECTED"
        assert data["model_a_version"] == "stroke-type-triage-v1"
        assert data["model_b_version"] == "STROKE_AI_Model_B_Final"
        assert "timestamp" in data
        assert "not clinically validated" in data["disclaimer"]


# ---------------------------------------------------------------------------
# Eligibility (rules engine)
# ---------------------------------------------------------------------------


class TestEligibility:
    def test_unknown_study_returns_404(self):
        response = client.post(
            "/eligibility", json=_eligibility_payload(str(uuid.uuid4()))
        )
        assert response.status_code == 404

    def test_hemorrhage_study_maps_to_hemorrhage_suspected(self):
        study_id = _insert_study(imaging_status="HEMORRHAGE_SUSPECTED")
        response = client.post(
            "/eligibility", json=_eligibility_payload(study_id)
        )
        assert response.status_code == 200
        data = response.json()
        assert data["final_status"] == "HEMORRHAGE_SUSPECTED"
        assert data["rule_trace"] == "RULE_HEMORRHAGE_DETECTED"
        assert data["requires_radiologist_review"] is False

    def test_missing_lkw_maps_to_insufficient_information(self):
        study_id = _insert_study(imaging_status="NO_ACUTE_STROKE_EVIDENCE")
        response = client.post(
            "/eligibility",
            json=_eligibility_payload(study_id, lkw_time=None),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["final_status"] == "INSUFFICIENT_INFORMATION"
        assert data["rule_trace"] == "RULE_MISSING_LKW"
        assert "lkw_time" in data["missing_fields"]

    def test_poor_quality_maps_to_urgent_review(self):
        study_id = _insert_study(imaging_status="NO_ACUTE_STROKE_EVIDENCE")
        response = client.post(
            "/eligibility",
            json=_eligibility_payload(study_id, image_quality="poor"),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["final_status"] == "URGENT_RADIOLOGY_REVIEW"
        assert data["rule_trace"] == "RULE_POOR_IMAGE_QUALITY"
        assert data["requires_radiologist_review"] is True

    def test_outside_window(self):
        study_id = _insert_study(imaging_status="NO_ACUTE_STROKE_EVIDENCE")
        response = client.post(
            "/eligibility",
            json=_eligibility_payload(study_id, lkw_time="2026-11-13T10:00:00"),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["final_status"] == "OUTSIDE_STANDARD_WINDOW"
        assert data["within_standard_window"] is False
        assert data["elapsed_hours"] > 4.5

    def test_contraindication_pending_review(self):
        study_id = _insert_study(imaging_status="NO_ACUTE_STROKE_EVIDENCE")
        response = client.post(
            "/eligibility",
            json=_eligibility_payload(
                study_id, severe_uncontrolled_hypertension=True
            ),
        )
        assert response.status_code == 200
        data = response.json()
        assert data["final_status"] == "POTENTIALLY_ELIGIBLE_PENDING_REVIEW"
        assert data["rule_trace"] == "RULE_CONTRADICTION_PENDING_REVIEW"
        assert data["requires_radiologist_review"] is True

    def test_negative_ai_never_means_eligibility(self):
        study_id = _insert_study(imaging_status="NO_ACUTE_STROKE_EVIDENCE")
        response = client.post(
            "/eligibility", json=_eligibility_payload(study_id)
        )
        assert response.status_code == 200
        data = response.json()
        assert data["final_status"] == "NO_HEMORRHAGE_DETECTED"
        assert data["requires_radiologist_review"] is False

    def test_never_administer_thrombolysis(self):
        cases = [
            {"imaging_status": "HEMORRHAGE_SUSPECTED", "overrides": {}},
            {
                "imaging_status": "NO_ACUTE_STROKE_EVIDENCE",
                "overrides": {"image_quality": "poor"},
            },
            {
                "imaging_status": "NO_ACUTE_STROKE_EVIDENCE",
                "overrides": {"lkw_time": None},
            },
            {
                "imaging_status": "NO_ACUTE_STROKE_EVIDENCE",
                "overrides": {"active_bleeding": True},
            },
        ]
        for case in cases:
            study_id = _insert_study(imaging_status=case["imaging_status"])
            response = client.post(
                "/eligibility",
                json=_eligibility_payload(study_id, **case["overrides"]),
            )
            assert response.status_code == 200
            assert "ADMINISTER_THROMBOLYSIS" not in response.text.upper()


# ---------------------------------------------------------------------------
# Audit
# ---------------------------------------------------------------------------


class TestAudit:
    def test_audit_record_and_retrieve(self):
        study_id = _insert_study()
        response = client.post(
            "/audit",
            json={
                "study_id": study_id,
                "event_type": "confirmation",
                "clinician_notes": "Clinician reviewed and confirmed",
            },
        )
        assert response.status_code == 200
        assert response.json()["status"] == "recorded"

        events = client.get(f"/audit/{study_id}")
        assert events.status_code == 200
        data = events.json()
        assert len(data) == 1
        assert data[0]["event_type"] == "confirmation"
        assert data[0]["clinician_notes"] == "Clinician reviewed and confirmed"
        assert data[0]["created_at"] is not None

    def test_audit_invalid_event_type(self):
        study_id = _insert_study()
        response = client.post(
            "/audit",
            json={"study_id": study_id, "event_type": "diagnose"},
        )
        assert response.status_code == 400

    def test_audit_unknown_study(self):
        response = client.post(
            "/audit",
            json={"study_id": str(uuid.uuid4()), "event_type": "override"},
        )
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Processing pipeline internals (mocked preprocessing)
# ---------------------------------------------------------------------------


class TestProcessingPipeline:
    def test_process_study_completes_and_persists(self, monkeypatch):
        monkeypatch.setattr(
            api_module,
            "preprocess_for_model_a",
            lambda path, target_size=224: (torch.zeros(1, 1, 224, 224), None),
        )
        monkeypatch.setattr(
            api_module,
            "preprocess_for_model_b",
            lambda path, target_size=224: (torch.zeros(1, 1, 224, 224), None),
        )
        study_id = _insert_study(status="queued")

        process_study(["slice_0.dcm", "slice_1.dcm"], study_id)

        db = SessionLocal()
        try:
            study = db.get(StudyModel, study_id)
            assert study.status == "completed"
            result = json.loads(study.result_json)
        finally:
            db.close()

        assert result["imaging_status"] == "HEMORRHAGE_SUSPECTED"
        assert result["top_slices"] == [0, 1]
        assert result["model_b_result"] is not None
        assert result["model_b_result"]["highest_probability_subtype"] == "intraparenchymal"
        assert result["thresholds"]["hemorrhage_probability"] == 0.5

    def test_process_study_fails_when_all_slices_fail(self, monkeypatch):
        def _raise(*args, **kwargs):
            raise ValueError("unreadable image")

        monkeypatch.setattr(api_module, "preprocess_for_model_a", _raise)
        study_id = _insert_study(status="queued")

        process_study(["bad.dcm"], study_id)

        db = SessionLocal()
        try:
            study = db.get(StudyModel, study_id)
            assert study.status == "failed"
            assert study.result_json is None
        finally:
            db.close()

    def test_run_model_b_on_slices_aggregates_by_max(self, monkeypatch):
        calls = {"n": 0}

        def fake_preprocess(path, target_size=224):
            calls["n"] += 1
            return torch.zeros(1, 1, 224, 224), None

        monkeypatch.setattr(api_module, "preprocess_for_model_b", fake_preprocess)

        class VaryingModelB:
            model_version = "STROKE_AI_Model_B_Final"

            def predict(self, tensor):
                return {
                    "subtype_probabilities": {
                        "epidural": 0.1,
                        "intraparenchymal": 0.6,
                        "intraventricular": 0.1,
                        "subarachnoid": 0.1,
                        "subdural": 0.1,
                    },
                    "highest_probability_subtype": "intraparenchymal",
                    "model_version": self.model_version,
                    "inference_timestamp": "2026-01-01T00:00:00Z",
                    "disclaimer": api_module.DISCLAIMER,
                }

        result = run_model_b_on_slices(
            VaryingModelB(), ["a.dcm", "b.dcm"], [0, 1]
        )
        assert calls["n"] == 2
        assert result["highest_probability_subtype"] == "intraparenchymal"
        assert 0.0 <= result["subtype_probabilities"]["intraparenchymal"] <= 1.0

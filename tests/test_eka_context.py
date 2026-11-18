# Tests for rules.eka_context helpers.
# Safety: DICOM never allowed to Eka; extracted fields are unconfirmed;
# context output never contains eligibility or diagnosis fields.

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rules.eka_context import (
    DISCLAIMER,
    assessment_summary,
    build_clinical_context,
    extract_document_fields,
    is_dicom_upload,
    map_to_checklist,
    uncertain_or_missing_fields,
    validate_document_upload,
)


def test_is_dicom_upload_by_extension():
    assert is_dicom_upload("ct_slice.dcm", "", b"") is True
    assert is_dicom_upload("scan.DICOM", "", b"") is True
    assert is_dicom_upload("report.pdf", "", b"") is False


def test_is_dicom_upload_by_content_magic():
    content = b"\x00" * 128 + b"DICM" + b"\x00" * 16
    assert is_dicom_upload("unknown.bin", "", content) is True
    assert is_dicom_upload("unknown.bin", "application/dicom", b"x") is True


def test_validate_blocks_dicom():
    content = b"\x00" * 128 + b"DICM" + b"\x00" * 16
    ok, reason = validate_document_upload("ct.dcm", "application/dicom", content)
    assert ok is False
    assert "never be sent to Eka" in reason


def test_validate_blocks_empty_and_disallowed():
    ok, reason = validate_document_upload("report.pdf", "application/pdf", b"")
    assert ok is False
    assert "empty" in reason.lower()

    ok, reason = validate_document_upload("notes.txt", "text/plain", b"hello")
    assert ok is False
    assert "PDF or image" in reason


def test_validate_allows_pdf_and_image():
    ok, _ = validate_document_upload("report.pdf", "application/pdf", b"%PDF")
    assert ok is True
    ok, _ = validate_document_upload("labs.png", "image/png", b"\x89PNG")
    assert ok is True


SAMPLE_REPORT = {
    "status": "completed",
    "confidence": 0.9,
    "data": {
        "output": {
            "data": [
                {"test_name": "INR", "data": {"value": 1.4}},
                {"test_name": "Platelet count",
                 "data": {"value": 210, "unit": "10^3/uL"}},
                {"test_name": "Creatinine",
                 "data": {"value": 0.9, "unit": "mg/dL"}},
                {"test_name": "Systolic BP", "data": {"value": 190}},
            ]
        },
        "pii": {"medications": ["Warfarin 5mg", "Metformin"]},
    },
    "history": {
        "prior_stroke": False,
        "prior_intracranial_hemorrhage": True,
    },
}


def test_extract_document_fields_reports_values():
    fields = extract_document_fields(SAMPLE_REPORT)
    assert fields["lab_values"]["inr"]["value"] == 1.4
    assert fields["lab_values"]["inr"]["status"] == "reported"
    assert fields["lab_values"]["platelet_count"]["unit"] == "10^3/uL"
    assert fields["anticoagulant_use"]["value"] == "positive"
    assert fields["anticoagulant_use"]["status"] == "reported"
    assert fields["prior_hemorrhage_history"]["value"] is True
    assert fields["prior_stroke_history"]["value"] is False


def test_extract_marks_missing_and_uncertain():
    fields = extract_document_fields({"data": {"output": {"data": []}}})
    assert fields["lab_values"]["inr"]["status"] == "missing"
    assert fields["anticoagulant_use"]["status"] == "uncertain"
    assert fields["medications"]["status"] == "missing"

    missing = uncertain_or_missing_fields(fields)
    assert "lab_values.inr" in missing
    assert "anticoagulant_use" in missing


def test_map_to_checklist_suggestions():
    fields = extract_document_fields(SAMPLE_REPORT)
    checklist = map_to_checklist(fields)
    assert checklist["anticoagulant_status"] == "positive"
    assert checklist["prior_intracranial_hemorrhage"] is True
    assert checklist["severe_uncontrolled_hypertension"] is True  # SBP 190
    assert checklist["requires_clinician_confirmation"] is True
    assert checklist["source"] == "eka_document_unconfirmed"


def test_map_to_checklist_keeps_missing_as_none():
    fields = extract_document_fields({"data": {"output": {"data": []}}})
    checklist = map_to_checklist(fields)
    assert checklist["anticoagulant_status"] == "unknown"
    assert checklist["prior_intracranial_hemorrhage"] is None
    assert checklist["severe_uncontrolled_hypertension"] is None


def test_assessment_summary():
    summary = assessment_summary(
        {"success": True, "explain": [{"label": "Check BP"},
                                      {"label": "Review imaging"}]}
    )
    assert summary["completed"] is True
    assert "Check BP" in summary["summary"]
    assert summary["requires_clinician_confirmation"] is True

    empty = assessment_summary({})
    assert empty["completed"] is False


def test_build_clinical_context_has_no_eligibility_fields():
    fields = extract_document_fields(SAMPLE_REPORT)
    context = build_clinical_context(
        ct_result={"study_id": "S1", "imaging_status": "NO_ACUTE_STROKE_EVIDENCE"},
        assessment_result={"success": True, "explain": [{"label": "x"}]},
        document_findings={"D1": fields},
        clinician_confirmed=False,
    )
    assert context["requires_clinician_confirmation"] is True
    assert context["requires_clinician_review"] is True
    assert context["clinician_confirmed"] is False
    assert context["disclaimer"] == DISCLAIMER
    assert "final_status" not in context
    assert "eligibility" not in context
    assert "diagnosis" not in context
    assert "checklist_suggestions" in context
    assert context["checklist_suggestions"]["anticoagulant_status"] == "positive"


def test_build_clinical_context_without_eka():
    context = build_clinical_context(
        ct_result={"study_id": "S2"},
        assessment_result=None,
        document_findings={},
    )
    assert context["eka_assessment"] is None
    assert "assessment:no_eka_context" in context["uncertain_or_missing"]
    assert context["requires_clinician_confirmation"] is True

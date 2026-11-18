# Tests for Eka clinical-context support in the rules engine.
# Safety: Eka warnings never change final_status, never exclude a patient,
# never output ADMINISTER_THROMBOLYSIS, and always require confirmation.

import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rules.engine import (
    eka_context_present,
    evaluate_eka_context,
    run_engine,
)

BASE_INPUTS = {
    "imaging_status": "NO_ACUTE_STROKE_EVIDENCE",
    "lkw_time": "2026-11-18T06:00:00",
    "current_time": "2026-11-18T08:00:00",
    "image_quality": "good",
    "severe_uncontrolled_hypertension": False,
    "prior_intracranial_hemorrhage": False,
    "active_bleeding": False,
    "recent_major_surgery_or_trauma": False,
    "anticoagulant_status": "negative",
    "recent_ischemic_stroke_or_head_injury": False,
}


def _with(**eka_fields):
    return {**BASE_INPUTS, **eka_fields}


def test_no_eka_context_produces_no_warnings():
    out = run_engine(_with())
    assert out["eka_warnings"] == []
    assert out["eka_rule_traces"] == []
    assert out["eka_requires_clinician_confirmation"] is False
    assert eka_context_present(BASE_INPUTS) is False


def test_eka_context_present_detection():
    assert eka_context_present(_with(eka_assessment_completed=True)) is True
    assert eka_context_present(
        _with(eka_extracted_anticoagulant_use="positive")
    ) is True
    assert eka_context_present(BASE_INPUTS) is False


def test_anticoagulant_detected_adds_warning_not_exclusion():
    baseline = run_engine(_with())
    out = run_engine(
        _with(
            eka_assessment_completed=True,
            eka_extracted_anticoagulant_use="positive",
            eka_extracted_lab_values={"inr": 1.1},
        )
    )
    traces = [w["trace"] for w in out["eka_warnings"]]
    assert "EKA_ANTICOAGULANT_USE_WARNING" in traces
    # Final status must be identical to the run without Eka context.
    assert out["final_status"] == baseline["final_status"]
    assert out["eka_requires_clinician_confirmation"] is True
    assert out["requires_radiologist_review"] is True
    warning = next(
        w for w in out["eka_warnings"]
        if w["trace"] == "EKA_ANTICOAGULANT_USE_WARNING"
    )
    assert warning["requires_clinician_confirmation"] is True
    assert warning["field"] == "eka_extracted_anticoagulant_use"


def test_prior_hemorrhage_history_adds_warning():
    out = run_engine(
        _with(
            eka_assessment_completed=True,
            eka_extracted_prior_hemorrhage_history=True,
            eka_extracted_lab_values={"inr": 1.0},
        )
    )
    traces = [w["trace"] for w in out["eka_warnings"]]
    assert "EKA_PRIOR_HEMORRHAGE_HISTORY_WARNING" in traces
    assert out["final_status"] == "NO_HEMORRHAGE_DETECTED"


def test_missing_lab_values_adds_warning():
    out = run_engine(
        _with(
            eka_assessment_completed=True,
            eka_extracted_lab_values=None,
        )
    )
    traces = [w["trace"] for w in out["eka_warnings"]]
    assert "EKA_MISSING_LAB_VALUES_WARNING" in traces


def test_uncertain_extraction_adds_warning():
    out = run_engine(
        _with(
            eka_assessment_completed=True,
            eka_extracted_anticoagulant_use="uncertain",
            eka_extracted_lab_values={"inr": 1.0},
        )
    )
    traces = [w["trace"] for w in out["eka_warnings"]]
    assert "EKA_UNCERTAIN_EXTRACTION_WARNING" in traces


def test_uncertain_confidence_adds_warning():
    out = run_engine(
        _with(
            eka_assessment_completed=True,
            eka_context_confidence="low",
            eka_extracted_lab_values={"inr": 1.0},
        )
    )
    traces = [w["trace"] for w in out["eka_warnings"]]
    assert "EKA_UNCERTAIN_EXTRACTION_WARNING" in traces


def test_eka_never_changes_hemorrhage_status():
    inputs = _with(
        imaging_status="HEMORRHAGE_SUSPECTED",
        lkw_time=None,
        eka_assessment_completed=True,
        eka_extracted_anticoagulant_use="positive",
        eka_extracted_lab_values=None,
    )
    out = run_engine(inputs)
    assert out["final_status"] == "HEMORRHAGE_SUSPECTED"
    assert out["rule_trace"] == "RULE_HEMORRHAGE_DETECTED"


def test_eka_never_changes_insufficient_information_status():
    inputs = _with(
        lkw_time=None,
        eka_assessment_completed=True,
        eka_extracted_prior_hemorrhage_history=True,
    )
    out = run_engine(inputs)
    assert out["final_status"] == "INSUFFICIENT_INFORMATION"
    assert "lkw_time" in out["missing_fields"]


def test_no_administer_thrombolysis_in_eka_output():
    out = run_engine(
        _with(
            eka_assessment_completed=True,
            eka_extracted_anticoagulant_use="positive",
            eka_extracted_prior_hemorrhage_history=True,
            eka_extracted_lab_values=None,
            eka_context_confidence="low",
        )
    )
    blob = json.dumps(out, default=str)
    assert "ADMINISTER_THROMBOLYSIS" not in blob
    assert "administer" not in blob.lower()


def test_eka_warnings_always_flag_clinician_review():
    out = run_engine(
        _with(
            eka_assessment_completed=True,
            eka_extracted_anticoagulant_use="positive",
            eka_extracted_lab_values={"inr": 1.0},
        )
    )
    assert out["requires_radiologist_review"] is True
    for warning in out["eka_warnings"]:
        assert warning["requires_clinician_confirmation"] is True
    assert "unconfirmed" in out["rationale"].lower()


def test_evaluate_eka_context_direct():
    warnings = evaluate_eka_context(
        _with(
            eka_extracted_prior_hemorrhage_history=True,
            eka_extracted_lab_values={"inr": 1.2},
        )
    )
    assert any(
        w["trace"] == "EKA_PRIOR_HEMORRHAGE_HISTORY_WARNING"
        for w in warnings
    )
    # No context at all -> no warnings even with defaults present.
    assert evaluate_eka_context(BASE_INPUTS) == []

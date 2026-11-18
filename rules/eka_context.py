# Eka Care clinical-context helpers.
# Eka output is NEVER a diagnosis, never a definitive contraindication, and
# never an automatic exclusion. All Eka-derived findings require clinician
# confirmation. CT images are never sent to Eka.

DISCLAIMER = (
    "Eka Care output is unconfirmed clinical-context support only. It does not "
    "diagnose stroke, does not determine thrombolysis eligibility, and requires "
    "clinician confirmation."
)

EKA_CONTEXT_FIELDS = [
    "eka_assessment_completed",
    "eka_extracted_medications",
    "eka_extracted_anticoagulant_use",
    "eka_extracted_lab_values",
    "eka_extracted_prior_stroke_history",
    "eka_extracted_prior_hemorrhage_history",
    "eka_context_confidence",
]

ANTICOAGULANT_NAMES = (
    "warfarin", "apixaban", "rivaroxaban", "dabigatran", "edoxaban",
    "heparin", "enoxaparin", "clopidogrel", "aspirin",
)

DOC_ALLOWED_EXTENSIONS = (".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff")
DOC_BLOCKED_EXTENSIONS = (".dcm", ".dicom")


def _field_status(value):
    if value is None:
        return "missing"
    if value == "uncertain":
        return "uncertain"
    if isinstance(value, dict) and value.get("status") in ("missing", "uncertain"):
        return value["status"]
    return "reported"


def is_dicom_upload(filename, content_type, content=None):
    """True if the upload looks like a CT DICOM (must never go to Eka)."""
    name = (filename or "").lower()
    if name.endswith(DOC_BLOCKED_EXTENSIONS):
        return True
    if (content_type or "").lower() in ("application/dicom", "application/dicom+json"):
        return True
    if content and len(content) > 132 and content[128:132] == b"DICM":
        return True
    return False


def validate_document_upload(filename, content_type, content):
    """Return (ok, reason). Only PDF/images are allowed; DICOM is blocked."""
    if not content:
        return False, "The uploaded file is empty."
    if is_dicom_upload(filename, content_type, content):
        return False, "CT DICOM files must never be sent to Eka Care."
    name = (filename or "").lower()
    ctype = (content_type or "").lower()
    allowed = name.endswith(DOC_ALLOWED_EXTENSIONS) or ctype.startswith("image/") or ctype == "application/pdf"
    if not allowed:
        return False, "Only PDF or image documents may be sent to Eka Care."
    return True, ""


def _find_lab(reported_labs, *needles):
    for item in reported_labs or []:
        if not isinstance(item, dict):
            continue
        name = str(item.get("test_name", "")).lower()
        if any(n in name for n in needles):
            data = item.get("data") or {}
            return data.get("value"), data.get("unit")
    return None, None


def extract_document_fields(parsed_report):
    """Extract only checklist-relevant fields from an Eka parsed report.

    Every field is returned as {"value": ..., "status": reported|missing|uncertain}.
    Extracted values are never treated as verified clinical facts.
    """
    report = parsed_report or {}
    data = ((report.get("data") or {}).get("output") or {}).get("data") or []
    if not isinstance(data, list):
        data = []

    inr, _ = _find_lab(data, "inr")
    platelets, platelet_unit = _find_lab(data, "platelet")
    creatinine, creat_unit = _find_lab(data, "creatinine")
    bp_syst, _ = _find_lab(data, "systolic", "blood pressure", "bp")

    meds = report.get("medications")
    if meds is None:
        pii = ((report.get("data") or {}).get("pii") or {})
        meds = pii.get("medications") if isinstance(pii, dict) else None

    if isinstance(meds, list) and meds:
        meds_lower = [str(m).lower() for m in meds]
        anticoag = "positive" if any(a in " ".join(meds_lower) for a in ANTICOAGULANT_NAMES) else "negative"
        anticoag_status = "reported"
    elif isinstance(meds, list):
        meds, anticoag, anticoag_status = [], "negative", "reported"
    else:
        meds, anticoag, anticoag_status = None, "uncertain", "uncertain"

    history = report.get("history") or {}
    prior_stroke = history.get("prior_stroke") if isinstance(history, dict) else None
    prior_hem = history.get("prior_intracranial_hemorrhage") if isinstance(history, dict) else None

    def _status(v, explicit=None):
        if explicit:
            return explicit
        return "reported" if v is not None else "missing"

    fields = {
        "medications": {"value": meds, "status": _status(meds)},
        "anticoagulant_use": {"value": anticoag, "status": anticoag_status},
        "lab_values": {
            "inr": {"value": inr, "status": _status(inr)},
            "platelet_count": {"value": platelets, "unit": platelet_unit,
                               "status": _status(platelets)},
            "creatinine": {"value": creatinine, "unit": creat_unit,
                           "status": _status(creatinine)},
            "systolic_bp": {"value": bp_syst, "status": _status(bp_syst)},
        },
        "prior_stroke_history": {"value": prior_stroke, "status": _status(prior_stroke)},
        "prior_hemorrhage_history": {"value": prior_hem, "status": _status(prior_hem)},
        "confidence": {"value": report.get("confidence"), "status": _status(report.get("confidence"))},
    }
    return fields


def uncertain_or_missing_fields(fields):
    """List of field paths whose value is missing or uncertain."""
    out = []

    def walk(prefix, node):
        if isinstance(node, dict) and "status" in node:
            if node["status"] in ("missing", "uncertain"):
                out.append(prefix)
            return
        if isinstance(node, dict):
            for k, v in node.items():
                walk(f"{prefix}.{k}" if prefix else str(k), v)

    for key, val in (fields or {}).items():
        walk(key, val)
    return out


def map_to_checklist(fields):
    """Map extracted document fields into the eligibility checklist.

    Returns suggestions only — every value is unconfirmed and requires a
    clinician to verify before use. Missing/uncertain values stay None.
    """
    ac = (fields.get("anticoagulant_use") or {})
    meds = (fields.get("medications") or {})
    labs = (fields.get("lab_values") or {})
    prior_hem = (fields.get("prior_hemorrhage_history") or {})
    prior_stroke = (fields.get("prior_stroke_history") or {})
    systolic = (labs.get("systolic_bp") or {})

    def _reported(node, key="value"):
        return node.get(key) if node.get("status") == "reported" else None

    anticoagulant_status = "unknown"
    if ac.get("status") == "reported" and ac.get("value") in ("positive", "negative"):
        anticoagulant_status = ac["value"]

    systolic_value = _reported(systolic)
    hypertension = None
    if systolic_value is not None:
        try:
            hypertension = float(systolic_value) >= 180
        except (TypeError, ValueError):
            hypertension = None

    return {
        "anticoagulant_status": anticoagulant_status,
        "prior_intracranial_hemorrhage": _reported(prior_hem),
        "prior_ischemic_stroke_or_head_injury": _reported(prior_stroke),
        "severe_uncontrolled_hypertension": hypertension,
        "medications_reported": _reported(meds) is not None,
        "source": "eka_document_unconfirmed",
        "requires_clinician_confirmation": True,
    }


def assessment_summary(assessment_result):
    """Normalise an Eka submit result into a compact clinical-context dict."""
    raw = assessment_result or {}
    explain = raw.get("explain") or []
    labels = [str(e.get("label")) for e in explain if isinstance(e, dict) and e.get("label")]

    completed = raw.get("success")
    if completed is None:
        completed = raw.get("assessment_completed")
    if completed is None:
        completed = raw.get("completed")
    if completed is None:
        completed = bool(explain)

    summary_text = raw.get("summary")
    if not summary_text:
        summary_text = (
            "; ".join(labels[:5])
            if labels
            else "Assessment completed — review with the clinical team."
        )

    return {
        "assessment_id": raw.get("assessment_id"),
        "completed": bool(completed),
        "summary": summary_text,
        "suggested_considerations": labels
        or list(raw.get("suggested_considerations") or []),
        "requires_clinician_confirmation": True,
        "source": "eka_assessment_unconfirmed",
    }


def build_clinical_context(ct_result, assessment_result, document_findings,
                           clinician_confirmed=False):
    """Combine CT results, Eka assessment, and document findings into a
    structured clinical-context object for the rules engine.

    Never includes a final eligibility or diagnosis field — clinicians decide.
    """
    ct_result = ct_result or {}
    doc_findings = document_findings or {}

    checklist = {}
    uncertain = []
    for doc_id, fields in doc_findings.items():
        if isinstance(fields, dict) and fields.get("status") in ("missing", "uncertain"):
            # Document was requested but has no usable extraction.
            uncertain.append(doc_id)
            continue
        mapped = map_to_checklist(fields)
        for key, val in mapped.items():
            if key in ("source", "requires_clinician_confirmation"):
                continue
            if val is not None and checklist.get(key) in (None, "unknown"):
                checklist[key] = val
        uncertain.extend(f"{doc_id}:{p}" for p in uncertain_or_missing_fields(fields))

    eka_assessment = assessment_summary(assessment_result) if assessment_result else None
    if eka_assessment is None:
        uncertain.append("assessment:no_eka_context")

    return {
        "study_id": ct_result.get("study_id"),
        "imaging_status": ct_result.get("imaging_status"),
        "model_a_result": ct_result.get("model_a_result"),
        "model_b_result": ct_result.get("model_b_result"),
        "eka_assessment": eka_assessment,
        "document_findings": doc_findings,
        "checklist_suggestions": checklist,
        "uncertain_or_missing": uncertain,
        "requires_clinician_confirmation": True,
        "requires_clinician_review": True,
        "clinician_confirmed": bool(clinician_confirmed),
        "disclaimer": DISCLAIMER,
    }

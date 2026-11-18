# Eka Care clinical-context endpoints.
#
# Scope: symptom/risk assessment Q&A, medical-document parsing (PDF/images
# only), and structured clinical context. NEVER used for CT interpretation,
# stroke diagnosis, thrombolysis eligibility, or treatment recommendations.
# CT DICOM uploads are rejected here and must never be sent to Eka.
# All Eka-derived output requires clinician confirmation.

import json
import uuid
from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from rules.eka_context import (
    assessment_summary,
    build_clinical_context,
    extract_document_fields,
    map_to_checklist,
    uncertain_or_missing_fields,
    validate_document_upload,
)
from schemas.eka import (
    AssessmentContinueRequest,
    AssessmentContinueResponse,
    AssessmentQuestion,
    AssessmentResult,
    AssessmentStartRequest,
    AssessmentStartResponse,
    AssessmentSubmitRequest,
    AssessmentSubmitResponse,
    DocumentStatusResponse,
    DocumentUploadResponse,
    EkaContextRequest,
    EkaContextResponse,
    ParsedDocumentResponse,
)
from services.eka_care import (
    EkaAPIError,
    EkaAuthError,
    EkaCareClient,
    EkaConfigurationError,
)

router = APIRouter(prefix="/eka", tags=["eka"])

# In-memory registries (pseudonymous IDs only; tokens never stored here).
_DOCUMENTS: Dict[str, Dict[str, Any]] = {}
_ASSESSMENTS: Dict[str, Dict[str, Any]] = {}

_eka_client: Optional[EkaCareClient] = None


def get_eka_service() -> EkaCareClient:
    """Return the shared Eka Care client (overridable in tests)."""
    global _eka_client
    if _eka_client is None:
        try:
            _eka_client = EkaCareClient()
        except EkaConfigurationError as exc:
            raise HTTPException(status_code=503, detail=str(exc))
    return _eka_client


def reset_eka_state() -> None:
    """Clear in-memory Eka registries (tests / dashboard reset)."""
    _DOCUMENTS.clear()
    _ASSESSMENTS.clear()


def _eka_error(exc: Exception) -> HTTPException:
    """Map Eka client exceptions to safe HTTP errors (no tokens leaked)."""
    if isinstance(exc, EkaConfigurationError):
        return HTTPException(status_code=503, detail=str(exc))
    if isinstance(exc, EkaAuthError):
        return HTTPException(status_code=502, detail="Eka Care authentication failed.")
    if isinstance(exc, EkaAPIError):
        return HTTPException(status_code=502, detail="Eka Care request failed.")
    return HTTPException(status_code=502, detail="Eka Care is unavailable.")


def _normalize_question(data: Any) -> Optional[AssessmentQuestion]:
    """Map an Eka start/continue payload to an AssessmentQuestion."""
    if not isinstance(data, dict):
        return None
    raw = None
    questions = data.get("questions")
    if isinstance(questions, list) and questions:
        raw = questions[0]
    elif data.get("qid") is not None:
        raw = data
    if not isinstance(raw, dict):
        return None

    options = []
    component_data = raw.get("component_data")
    if isinstance(component_data, dict):
        for key in ("options", "choices", "values", "items"):
            values = component_data.get(key)
            if isinstance(values, list) and values:
                options = [
                    str(v.get("label", v)) if isinstance(v, dict) else str(v)
                    for v in values
                ]
                break
        tip = component_data.get("tip")
    else:
        tip = None

    qid = raw.get("qid", raw.get("id"))
    progress = data.get("progress")
    try:
        progress = float(progress) if progress is not None else None
    except (TypeError, ValueError):
        progress = None

    return AssessmentQuestion(
        id=str(qid) if qid is not None else "",
        text=str(raw.get("question_text") or raw.get("text") or ""),
        type=str(raw.get("component_code") or raw.get("type") or "unknown"),
        options=options,
        tip=str(tip) if tip is not None else None,
        is_mandatory=raw.get("is_mandatory"),
        progress=progress,
        is_last_question=data.get("is_last_question"),
    )


def _pick(fields: Dict[str, Any], key: str) -> Any:
    """Pull a reported/uncertain/missing value out of extracted doc fields."""
    node = fields.get(key)
    if not isinstance(node, dict):
        return None
    status = node.get("status")
    if status == "uncertain":
        return "uncertain"
    if status == "missing":
        return None
    return node.get("value")


def _engine_inputs_from_findings(
    document_findings: Dict[str, Any], assessment: Optional[Dict[str, Any]]
) -> Dict[str, Any]:
    """Derive optional engine Eka inputs from unconfirmed findings.

    Missing/uncertain values stay None/"uncertain" — never invented facts.
    """
    meds = anticoag = prior_stroke = prior_hem = conf = None
    labs = None

    for fields in document_findings.values():
        if not isinstance(fields, dict) or "anticoagulant_use" not in fields:
            continue
        if anticoag is None:
            anticoag = _pick(fields, "anticoagulant_use")
        if meds is None:
            meds = _pick(fields, "medications")
        if prior_stroke is None:
            prior_stroke = _pick(fields, "prior_stroke_history")
        if prior_hem is None:
            prior_hem = _pick(fields, "prior_hemorrhage_history")
        if conf is None:
            conf = _pick(fields, "confidence")
        if labs is None:
            labs_node = fields.get("lab_values")
            if isinstance(labs_node, dict):
                mapped = {}
                for name, node in labs_node.items():
                    if not isinstance(node, dict):
                        continue
                    status = node.get("status")
                    if status == "uncertain":
                        mapped[name] = "uncertain"
                    elif status == "missing" or node.get("value") is None:
                        mapped[name] = None
                    else:
                        mapped[name] = node.get("value")
                labs = mapped

    summary = assessment or {}
    return {
        "eka_assessment_completed": bool(summary.get("completed")),
        "eka_extracted_medications": meds,
        "eka_extracted_anticoagulant_use": anticoag,
        "eka_extracted_lab_values": labs,
        "eka_extracted_prior_stroke_history": prior_stroke,
        "eka_extracted_prior_hemorrhage_history": prior_hem,
        "eka_context_confidence": conf,
    }


# ---------------------------------------------------------------------------
# Assessment Q&A flow
# ---------------------------------------------------------------------------


@router.post("/assessment/start", response_model=AssessmentStartResponse)
def start_assessment(
    req: AssessmentStartRequest,
    eka: EkaCareClient = Depends(get_eka_service),
):
    """Start an Eka symptom/risk assessment (pseudonymous ID only)."""
    try:
        result = eka.start_assessment(
            patient_pseudonym_id=req.patient_pseudonym_id,
            age=req.age,
            sex=req.sex,
            presenting_symptoms=req.presenting_symptoms,
            symptom_onset_time=req.symptom_onset_time,
        )
    except (EkaConfigurationError, EkaAuthError, EkaAPIError) as exc:
        raise _eka_error(exc)

    assessment_id = str(result.get("assessment_id") or "")
    question = _normalize_question(result.get("start_response"))
    return AssessmentStartResponse(
        assessment_id=assessment_id,
        question=question,
        status="in_progress",
    )


@router.post("/assessment/continue", response_model=AssessmentContinueResponse)
def continue_assessment(
    req: AssessmentContinueRequest,
    eka: EkaCareClient = Depends(get_eka_service),
):
    """Submit one answer and return the next question (or completion)."""
    try:
        data = eka.continue_assessment(
            req.assessment_id, req.question_id, req.answer
        )
    except (EkaConfigurationError, EkaAuthError, EkaAPIError) as exc:
        raise _eka_error(exc)

    question = _normalize_question(data)
    progress = None
    if isinstance(data, dict):
        try:
            raw_progress = data.get("progress")
            progress = float(raw_progress) if raw_progress is not None else None
        except (TypeError, ValueError):
            progress = None

    return AssessmentContinueResponse(
        assessment_id=req.assessment_id,
        status="in_progress" if question is not None else "completed",
        question=question,
        progress=progress,
    )


@router.post("/assessment/submit", response_model=AssessmentSubmitResponse)
def submit_assessment(
    req: AssessmentSubmitRequest,
    eka: EkaCareClient = Depends(get_eka_service),
):
    """Submit a completed assessment; returns an unconfirmed summary."""
    try:
        data = eka.submit_assessment(req.assessment_id)
    except (EkaConfigurationError, EkaAuthError, EkaAPIError) as exc:
        raise _eka_error(exc)

    summary = assessment_summary(data)
    _ASSESSMENTS[req.assessment_id] = summary

    return AssessmentSubmitResponse(
        assessment_id=req.assessment_id,
        result=AssessmentResult(
            assessment_id=req.assessment_id,
            summary=summary.get("summary", ""),
            suggested_considerations=summary.get("suggested_considerations") or [],
            success=bool(summary.get("completed")),
        ),
    )


# ---------------------------------------------------------------------------
# Medical-document parsing (PDF/images only — CT DICOM is rejected)
# ---------------------------------------------------------------------------


@router.post("/documents/upload", response_model=DocumentUploadResponse)
async def upload_document(
    eka: EkaCareClient = Depends(get_eka_service),
    file: UploadFile = File(...),
    study_id: Optional[str] = Form(None),
):
    """Upload a PDF/image clinical document to Eka (never CT images)."""
    content = await file.read()
    filename = file.filename or "document"
    content_type = file.content_type or "application/pdf"

    ok, reason = validate_document_upload(filename, content_type, content)
    if not ok:
        raise HTTPException(status_code=400, detail=reason)

    try:
        data = eka.upload_medical_document(filename, content, content_type)
    except (EkaConfigurationError, EkaAuthError, EkaAPIError) as exc:
        raise _eka_error(exc)

    document_id = str(data.get("document_id") or data.get("id") or uuid.uuid4().hex)
    _DOCUMENTS[document_id] = {
        "document_id": document_id,
        "filename": filename,
        "study_id": study_id,
        "status": "queued",
        "parsed": None,
        "fields": None,
    }
    return DocumentUploadResponse(
        document_id=document_id, status="queued", study_id=study_id
    )


@router.get("/documents/{document_id}/status", response_model=DocumentStatusResponse)
def document_status(
    document_id: str,
    eka: EkaCareClient = Depends(get_eka_service),
):
    """Poll the parsing status of an uploaded document."""
    record = _DOCUMENTS.get(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        data = eka.get_document_status(document_id)
    except (EkaConfigurationError, EkaAuthError, EkaAPIError) as exc:
        raise _eka_error(exc)

    status = str(data.get("status", "unknown"))
    record["status"] = status
    return DocumentStatusResponse(document_id=document_id, status=status)


@router.get("/documents/{document_id}/parsed", response_model=ParsedDocumentResponse)
def document_parsed(
    document_id: str,
    eka: EkaCareClient = Depends(get_eka_service),
):
    """Fetch a parsed report; extract only checklist-relevant fields."""
    record = _DOCUMENTS.get(document_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")
    try:
        report = eka.get_parsed_report(document_id)
    except (EkaConfigurationError, EkaAuthError, EkaAPIError) as exc:
        raise _eka_error(exc)

    fields = extract_document_fields(report)
    record["parsed"] = report
    record["fields"] = fields
    record["status"] = str(report.get("status") or record.get("status") or "completed")

    return ParsedDocumentResponse(
        document_id=document_id,
        status=record["status"],
        fields=fields,
        uncertain_or_missing=uncertain_or_missing_fields(fields),
        checklist_suggestions=map_to_checklist(fields),
    )


# ---------------------------------------------------------------------------
# Combined clinical context (never a diagnosis, never eligibility)
# ---------------------------------------------------------------------------


@router.post("/context", response_model=EkaContextResponse)
def build_context(
    req: EkaContextRequest,
    eka: EkaCareClient = Depends(get_eka_service),
):
    """Combine CT result + Eka assessment + document findings into context."""
    from app.api import SessionLocal, StudyModel  # lazy — avoids circular import

    db = SessionLocal()
    try:
        study = db.query(StudyModel).filter(StudyModel.id == req.study_id).first()
    finally:
        db.close()
    if study is None:
        raise HTTPException(status_code=404, detail="Study not found")

    ct_result: Dict[str, Any] = {"study_id": req.study_id}
    if study.result_json:
        try:
            stored = json.loads(study.result_json)
            if isinstance(stored, dict):
                ct_result = stored
        except Exception:
            pass

    document_findings: Dict[str, Any] = {}
    for doc_id in req.document_ids:
        record = _DOCUMENTS.get(doc_id)
        if record is None:
            document_findings[doc_id] = {"status": "missing"}
        elif record.get("fields") is not None:
            document_findings[doc_id] = record["fields"]
        else:
            document_findings[doc_id] = {"status": "missing"}

    assessment = req.assessment_result
    if assessment is None and req.assessment_id:
        assessment = _ASSESSMENTS.get(req.assessment_id)

    context = build_clinical_context(
        ct_result=ct_result,
        assessment_result=assessment,
        document_findings=document_findings,
        clinician_confirmed=req.clinician_confirmed,
    )
    context["eka_engine_inputs"] = _engine_inputs_from_findings(
        document_findings, assessment
    )
    return EkaContextResponse(**context)

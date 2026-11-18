# FastAPI application for Stroke CT Triage Assistant
# Research prototype for clinical decision support.
# Never autonomous diagnosis; never recommends thrombolysis administration.

import json
import os
import shutil
import tempfile
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import torch
from fastapi import BackgroundTasks, Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy import Column, DateTime, Integer, String, Text, create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.model_a import ModelAService, create_model_a_service
from imaging.model_b import ModelBService, create_model_b_service
from imaging.preprocessing import preprocess_for_model_a, preprocess_for_model_b
from rules.engine import run_engine

DISCLAIMER = (
    "Research prototype; not clinically validated; "
    "not for autonomous diagnosis or treatment decisions."
)

HEMORRHAGE_THRESHOLD = 0.5
LOW_CONFIDENCE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# SQLite + SQLAlchemy setup (MVP — pseudonymous study IDs only)
# ---------------------------------------------------------------------------

DATABASE_URL = os.environ.get("STUDIES_DB_URL", "sqlite:///./studies.db")

Base = declarative_base()


class StudyModel(Base):
    __tablename__ = "studies"

    id = Column(String, primary_key=True, index=True)
    status = Column(String, default="queued")  # queued, processing, completed, failed
    result_json = Column(Text, nullable=True)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )
    updated_at = Column(
        DateTime,
        default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class AuditEventModel(Base):
    __tablename__ = "audit_events"

    id = Column(Integer, primary_key=True, index=True)
    study_id = Column(String, index=True)
    event_type = Column(String)  # "confirmation" or "override"
    clinician_notes = Column(Text, nullable=True)
    created_at = Column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )


connect_args = {"check_same_thread": False} if DATABASE_URL.startswith("sqlite") else {}
engine = create_engine(DATABASE_URL, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base.metadata.create_all(bind=engine)


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class HealthCheck(BaseModel):
    status: str
    model_a_loaded: bool
    model_b_loaded: bool
    model_a_version: str
    model_b_version: str
    disclaimer: str


class StudyUploadResponse(BaseModel):
    study_id: str
    status: str


class StudyStatus(BaseModel):
    status: str
    message: Optional[str] = None


class StudyResult(BaseModel):
    study_id: str
    model_a_result: Dict[str, Any]
    model_b_result: Optional[Dict[str, Any]] = None
    top_slices: List[int] = []
    imaging_status: str
    model_a_version: str
    model_b_version: Optional[str] = None
    thresholds: Optional[Dict[str, float]] = None
    timestamp: str
    disclaimer: str


class EligibilityRequest(BaseModel):
    study_id: str
    lkw_time: Optional[str] = None
    current_time: Optional[str] = None
    image_quality: str = "good"
    severe_uncontrolled_hypertension: bool = False
    prior_intracranial_hemorrhage: bool = False
    active_bleeding: bool = False
    recent_major_surgery_or_trauma: bool = False
    anticoagulant_status: str = "negative"
    recent_ischemic_stroke_or_head_injury: bool = False
    # Optional unconfirmed Eka clinical context (warnings only, never exclusions)
    eka_assessment_completed: bool = False
    eka_extracted_medications: Optional[Any] = None
    eka_extracted_anticoagulant_use: Optional[str] = None
    eka_extracted_lab_values: Optional[Any] = None
    eka_extracted_prior_stroke_history: Optional[Any] = None
    eka_extracted_prior_hemorrhage_history: Optional[Any] = None
    eka_context_confidence: Optional[str] = None


class EligibilityResponse(BaseModel):
    study_id: str
    final_status: str
    rule_trace: str
    elapsed_hours: Optional[float] = None
    within_standard_window: Optional[bool] = None
    requires_radiologist_review: bool = False
    missing_fields: List[str] = []
    rationale: str
    eka_context: Optional[Dict[str, Any]] = None
    eka_warnings: List[Dict[str, Any]] = []
    eka_rule_traces: List[str] = []
    eka_requires_clinician_confirmation: bool = False
    disclaimer: str = DISCLAIMER


class AuditCreate(BaseModel):
    study_id: str
    event_type: str  # "confirmation" or "override"
    clinician_notes: Optional[str] = None


# ---------------------------------------------------------------------------
# FastAPI app setup
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Stroke CT Triage Assistant API",
    description=(
        "Research prototype for clinical decision support — "
        "never autonomous diagnosis or treatment recommendations."
    ),
    version="0.1.0",
)

# CORS — local Streamlit development only
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Dependency — DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Model services (loaded once at startup; tests inject mocks)
_model_a_service: Optional[ModelAService] = None
_model_b_service: Optional[ModelBService] = None


@app.on_event("startup")
def load_models():
    global _model_a_service, _model_b_service
    if os.environ.get("SKIP_MODEL_LOAD") == "1":
        return
    try:
        _model_a_service = create_model_a_service()
    except Exception:
        _model_a_service = None
    try:
        _model_b_service = create_model_b_service()
    except Exception:
        _model_b_service = None


# ---------------------------------------------------------------------------
# Processing pipeline
# ---------------------------------------------------------------------------


def run_model_b_on_slices(
    model_b_svc: ModelBService, dcm_paths: List[str], slice_indices: List[int]
) -> Dict[str, Any]:
    """Run Model B on selected slices and aggregate subtype probabilities (max)."""
    subtype_probs_accumulator: Dict[str, float] = {}

    for idx in slice_indices:
        if idx >= len(dcm_paths):
            continue
        try:
            tensor_b, _ = preprocess_for_model_b(dcm_paths[idx], target_size=224)
            with torch.no_grad():
                b_result = model_b_svc.predict(tensor_b)
            for subtype, prob in b_result["subtype_probabilities"].items():
                subtype_probs_accumulator[subtype] = max(
                    subtype_probs_accumulator.get(subtype, 0.0), float(prob)
                )
        except Exception:
            continue

    if not subtype_probs_accumulator:
        return {
            "subtype_probabilities": None,
            "highest_probability_subtype": None,
            "model_version": getattr(model_b_svc, "model_version", "unknown"),
            "inference_timestamp": datetime.now(timezone.utc).isoformat(),
            "disclaimer": DISCLAIMER,
        }

    highest_subtype = max(subtype_probs_accumulator, key=subtype_probs_accumulator.get)
    return {
        "subtype_probabilities": {
            k: round(v, 4) for k, v in subtype_probs_accumulator.items()
        },
        "highest_probability_subtype": highest_subtype,
        "model_version": getattr(model_b_svc, "model_version", "unknown"),
        "inference_timestamp": datetime.now(timezone.utc).isoformat(),
        "disclaimer": DISCLAIMER,
    }


def process_study(dcm_paths: List[str], study_id: str) -> None:
    """Background task: preprocess slices, run Model A (+ Model B), persist result."""
    session = SessionLocal()
    study = session.get(StudyModel, study_id)
    try:
        if study is None:
            study = StudyModel(id=study_id, status="queued")
            session.add(study)
        study.status = "processing"
        session.commit()

        if _model_a_service is None:
            raise RuntimeError("Model A service is unavailable")

        slice_results: List[Dict[str, Any]] = []
        failed_slices = 0
        for i, path in enumerate(dcm_paths):
            try:
                tensor_a, _ = preprocess_for_model_a(path, target_size=224)
                a_result = _model_a_service.predict(tensor_a)
                a_result["slice_index"] = i
                slice_results.append(a_result)
            except Exception:
                # Do not fabricate a "normal" prediction for failed slices.
                failed_slices += 1

        if not slice_results:
            study.status = "failed"
            session.commit()
            return

        max_hem = max(
            float(r["class_probabilities"].get("hemorrhagic", 0.0))
            for r in slice_results
        )
        max_isk = max(
            float(r["class_probabilities"].get("ischemic", 0.0))
            for r in slice_results
        )
        max_norm = max(
            float(r["class_probabilities"].get("normal", 0.0))
            for r in slice_results
        )

        # Hemorrhage takes precedence; low confidence escalates to review.
        if max_hem >= HEMORRHAGE_THRESHOLD:
            predicted_class = "hemorrhagic"
            imaging_status = "HEMORRHAGE_SUSPECTED"
        elif max(max_isk, max_norm) < LOW_CONFIDENCE_THRESHOLD:
            predicted_class = "uncertain"
            imaging_status = "URGENT_RADIOLOGY_REVIEW"
        elif max_isk >= max_norm:
            predicted_class = "ischemic"
            imaging_status = "ISCHEMIC_PATTERN_POSSIBLE"
        else:
            predicted_class = "normal"
            imaging_status = "NO_ACUTE_STROKE_EVIDENCE"

        hem_sorted = sorted(
            slice_results,
            key=lambda r: float(r["class_probabilities"].get("hemorrhagic", 0.0)),
            reverse=True,
        )
        hem_slices = [
            r["slice_index"]
            for r in hem_sorted
            if float(r["class_probabilities"].get("hemorrhagic", 0.0)) > 0.0
        ][:3]

        if predicted_class == "hemorrhagic" and hem_slices:
            model_b_result = run_model_b_on_slices(
                _model_b_service, dcm_paths, hem_slices
            ) if _model_b_service is not None else None
        else:
            model_b_result = None

        result = {
            "study_id": study_id,
            "model_a_result": {
                "max_hemorrhagic_probability": round(max_hem, 4),
                "max_ischemic_probability": round(max_isk, 4),
                "max_normal_probability": round(max_norm, 4),
                "predicted_class": predicted_class,
                "top_3_hemorrhagic_slices": hem_slices,
                "processed_slices": len(slice_results),
                "failed_slices": failed_slices,
                "slice_details": [
                    {
                        "slice_index": r["slice_index"],
                        "predicted_class": r["predicted_class"],
                        "confidence": r["confidence"],
                    }
                    for r in slice_results
                ],
            },
            "model_b_result": model_b_result,
            "top_slices": hem_slices,
            "imaging_status": imaging_status,
            "model_a_version": getattr(_model_a_service, "model_version", "unknown"),
            "model_b_version": getattr(_model_b_service, "model_version", None)
            if _model_b_service is not None
            else None,
            "thresholds": {
                "hemorrhage_probability": HEMORRHAGE_THRESHOLD,
                "low_confidence": LOW_CONFIDENCE_THRESHOLD,
            },
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "disclaimer": DISCLAIMER,
        }

        study.result_json = json.dumps(result)
        study.status = "completed"
        session.commit()
    except Exception:
        try:
            if study is not None:
                study.status = "failed"
                session.commit()
        except Exception:
            pass
    finally:
        session.close()


def process_study_background(dcm_paths: List[str], study_id: str) -> None:
    """Run processing, then delete the temporary DICOM files."""
    try:
        process_study(dcm_paths, study_id)
    finally:
        if dcm_paths:
            shutil.rmtree(os.path.dirname(dcm_paths[0]), ignore_errors=True)


# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------


@app.get("/health", response_model=HealthCheck)
def health_check():
    """Health check endpoint."""
    return HealthCheck(
        status="ok",
        model_a_loaded=_model_a_service is not None,
        model_b_loaded=_model_b_service is not None,
        model_a_version=getattr(_model_a_service, "model_version", "stroke-type-triage-v1"),
        model_b_version=getattr(_model_b_service, "model_version", "STROKE_AI_Model_B_Final"),
        disclaimer=DISCLAIMER,
    )


@app.post("/studies/upload", response_model=StudyUploadResponse)
async def upload_study_files(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
):
    """Accept multiple DICOM files, create a pseudonymous study ID, save temporarily."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    study_id = str(uuid.uuid4())
    tmp_dir = tempfile.mkdtemp(prefix=f"study_{study_id}_")
    saved_paths: List[str] = []

    try:
        for upload_file in files:
            content = await upload_file.read()
            if not content:
                raise HTTPException(status_code=400, detail="Uploaded file is empty")
            filepath = os.path.join(tmp_dir, f"{uuid.uuid4()}.dcm")
            with open(filepath, "wb") as f:
                f.write(content)
            saved_paths.append(filepath)
    except HTTPException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    except Exception:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail="Failed to store uploaded files")

    db = SessionLocal()
    try:
        db.add(StudyModel(id=study_id, status="queued"))
        db.commit()
    finally:
        db.close()

    background_tasks.add_task(process_study_background, saved_paths, study_id)

    return StudyUploadResponse(study_id=study_id, status="queued")


@app.get("/studies/{study_id}/status", response_model=StudyStatus)
def study_status(study_id: str, db: Session = Depends(get_db)):
    """Return the processing status of a study."""
    study = db.query(StudyModel).filter(StudyModel.id == study_id).first()
    if not study:
        raise HTTPException(status_code=404, detail="Study not found")
    return StudyStatus(status=study.status)


@app.get("/studies/{study_id}/result", response_model=StudyResult)
def study_result(study_id: str, db: Session = Depends(get_db)):
    """Return the full result for a completed study."""
    study = db.query(StudyModel).filter(StudyModel.id == study_id).first()
    if not study:
        raise HTTPException(status_code=404, detail="Study not found")
    if study.status == "failed":
        raise HTTPException(
            status_code=400, detail="Processing failed for this study; no result available"
        )
    if study.status != "completed" or not study.result_json:
        raise HTTPException(
            status_code=409, detail="Study has not completed processing"
        )
    try:
        return json.loads(study.result_json)
    except Exception:
        raise HTTPException(status_code=500, detail="Stored result is unavailable")


@app.post("/eligibility", response_model=EligibilityResponse)
def check_eligibility(req: EligibilityRequest, db: Session = Depends(get_db)):
    """Run the rule-based thrombolysis review engine for a completed study."""
    study = db.query(StudyModel).filter(StudyModel.id == req.study_id).first()
    if not study:
        raise HTTPException(status_code=404, detail="Study not found")
    if study.status != "completed" or not study.result_json:
        raise HTTPException(
            status_code=409, detail="Study has not completed processing"
        )
    try:
        stored_result = json.loads(study.result_json)
    except Exception:
        raise HTTPException(status_code=500, detail="Stored result is unavailable")

    inputs = {
        "imaging_status": stored_result.get("imaging_status", ""),
        "lkw_time": req.lkw_time,
        "current_time": req.current_time
        or datetime.now(timezone.utc).isoformat(),
        "image_quality": req.image_quality,
        "severe_uncontrolled_hypertension": req.severe_uncontrolled_hypertension,
        "prior_intracranial_hemorrhage": req.prior_intracranial_hemorrhage,
        "active_bleeding": req.active_bleeding,
        "recent_major_surgery_or_trauma": req.recent_major_surgery_or_trauma,
        "anticoagulant_status": req.anticoagulant_status,
        "recent_ischemic_stroke_or_head_injury": req.recent_ischemic_stroke_or_head_injury,
        "eka_assessment_completed": req.eka_assessment_completed,
        "eka_extracted_medications": req.eka_extracted_medications,
        "eka_extracted_anticoagulant_use": req.eka_extracted_anticoagulant_use,
        "eka_extracted_lab_values": req.eka_extracted_lab_values,
        "eka_extracted_prior_stroke_history": req.eka_extracted_prior_stroke_history,
        "eka_extracted_prior_hemorrhage_history": req.eka_extracted_prior_hemorrhage_history,
        "eka_context_confidence": req.eka_context_confidence,
    }

    result = run_engine(inputs)

    return EligibilityResponse(
        study_id=req.study_id,
        final_status=result["final_status"],
        rule_trace=result["rule_trace"],
        elapsed_hours=result.get("elapsed_hours"),
        within_standard_window=result.get("within_standard_window"),
        requires_radiologist_review=result.get("requires_radiologist_review", False),
        missing_fields=result.get("missing_fields") or [],
        rationale=result.get("rationale", ""),
        eka_context=result.get("eka_context"),
        eka_warnings=result.get("eka_warnings") or [],
        eka_rule_traces=result.get("eka_rule_traces") or [],
        eka_requires_clinician_confirmation=result.get(
            "eka_requires_clinician_confirmation", False
        ),
    )


@app.post("/audit", response_model=Dict[str, str])
def record_audit(audit: AuditCreate, db: Session = Depends(get_db)):
    """Record a clinician confirmation or override."""
    if audit.event_type not in ("confirmation", "override"):
        raise HTTPException(
            status_code=400, detail="event_type must be 'confirmation' or 'override'"
        )
    study = db.query(StudyModel).filter(StudyModel.id == audit.study_id).first()
    if not study:
        raise HTTPException(status_code=404, detail="Study not found")

    db.add(
        AuditEventModel(
            study_id=audit.study_id,
            event_type=audit.event_type,
            clinician_notes=audit.clinician_notes,
        )
    )
    db.commit()
    return {"status": "recorded"}


@app.get("/audit/{study_id}", response_model=List[Dict[str, Any]])
def get_audit_events(study_id: str, db: Session = Depends(get_db)):
    """Return audit events for a study."""
    events = (
        db.query(AuditEventModel)
        .filter(AuditEventModel.study_id == study_id)
        .order_by(AuditEventModel.created_at)
        .all()
    )
    return [
        {
            "id": e.id,
            "study_id": e.study_id,
            "event_type": e.event_type,
            "clinician_notes": e.clinician_notes,
            "created_at": e.created_at.isoformat() if e.created_at else None,
        }
        for e in events
    ]


# Eka Care clinical-context routes (assessment/documents/context only)
from app.api_eka import router as eka_router  # noqa: E402

app.include_router(eka_router)


# ---------------------------------------------------------------------------
# Run: uvicorn app.api:app --reload --port 8000
# ---------------------------------------------------------------------------

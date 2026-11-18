# Pydantic schemas for the Eka Care clinical-context integration.
# Eka output is clinical-context support only — never a diagnosis, never a
# treatment recommendation, always requires clinician confirmation.

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

EKA_DISCLAIMER = (
    "Eka Care output is clinical-context support only — unconfirmed and "
    "requires clinician review. It is not a diagnosis and never determines "
    "thrombolysis eligibility."
)


class AssessmentStartRequest(BaseModel):
    patient_pseudonym_id: str = Field(min_length=1, max_length=64)
    age: int = Field(ge=0, le=130)
    sex: Literal["male", "female", "other", "unknown"] = "unknown"
    presenting_symptoms: List[str] = Field(default_factory=list)
    symptom_onset_time: Optional[str] = None


class AssessmentQuestion(BaseModel):
    id: str
    text: str
    type: str = "unknown"
    options: List[str] = Field(default_factory=list)
    tip: Optional[str] = None
    is_mandatory: Optional[bool] = None
    progress: Optional[float] = None
    is_last_question: Optional[bool] = None


class AssessmentStartResponse(BaseModel):
    assessment_id: str
    question: Optional[AssessmentQuestion] = None
    status: str = "in_progress"
    requires_clinician_review: bool = True
    disclaimer: str = EKA_DISCLAIMER


class AssessmentContinueRequest(BaseModel):
    assessment_id: str
    question_id: str
    answer: Any


class AssessmentContinueResponse(BaseModel):
    assessment_id: str
    status: Literal["in_progress", "completed"] = "in_progress"
    question: Optional[AssessmentQuestion] = None
    progress: Optional[float] = None
    requires_clinician_review: bool = True
    disclaimer: str = EKA_DISCLAIMER


class AssessmentSubmitRequest(BaseModel):
    assessment_id: str


class AssessmentResult(BaseModel):
    assessment_id: Optional[str] = None
    summary: str = ""
    suggested_considerations: List[str] = Field(default_factory=list)
    success: bool = False


class AssessmentSubmitResponse(BaseModel):
    assessment_id: str
    result: AssessmentResult
    requires_clinician_review: bool = True
    disclaimer: str = EKA_DISCLAIMER


class EkaContextRequest(BaseModel):
    study_id: str
    assessment_id: Optional[str] = None
    assessment_result: Optional[Dict[str, Any]] = None
    document_ids: List[str] = Field(default_factory=list)
    clinician_confirmed: bool = False


class DocumentField(BaseModel):
    value: Any = None
    status: Literal["reported", "missing", "uncertain"] = "missing"


class DocumentUploadResponse(BaseModel):
    document_id: str
    status: str = "queued"
    study_id: Optional[str] = None
    requires_clinician_confirmation: bool = True
    disclaimer: str = EKA_DISCLAIMER


class DocumentStatusResponse(BaseModel):
    document_id: str
    status: str
    requires_clinician_confirmation: bool = True
    disclaimer: str = EKA_DISCLAIMER


class ParsedDocumentResponse(BaseModel):
    document_id: str
    status: str
    fields: Dict[str, Any] = Field(default_factory=dict)
    uncertain_or_missing: List[str] = Field(default_factory=list)
    checklist_suggestions: Dict[str, Any] = Field(default_factory=dict)
    requires_clinician_confirmation: bool = True
    disclaimer: str = EKA_DISCLAIMER


class EkaContextResponse(BaseModel):
    study_id: Optional[str] = None
    imaging_status: Optional[str] = None
    model_a_result: Optional[Dict[str, Any]] = None
    model_b_result: Optional[Dict[str, Any]] = None
    eka_assessment: Optional[Dict[str, Any]] = None
    document_findings: Dict[str, Any] = Field(default_factory=dict)
    checklist_suggestions: Dict[str, Any] = Field(default_factory=dict)
    uncertain_or_missing: List[str] = Field(default_factory=list)
    eka_engine_inputs: Dict[str, Any] = Field(default_factory=dict)
    requires_clinician_confirmation: bool = True
    requires_clinician_review: bool = True
    clinician_confirmed: bool = False
    disclaimer: str = EKA_DISCLAIMER


class SymptomSearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=200)


class SymptomSearchResponse(BaseModel):
    query: str
    results: List[Dict[str, Any]] = Field(default_factory=list)
    disclaimer: str = EKA_DISCLAIMER

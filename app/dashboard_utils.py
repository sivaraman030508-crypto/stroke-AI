# Pure helpers for the Streamlit dashboard (no streamlit import).
# Safety: never displays "administer thrombolysis"; never implies diagnosis;
# pseudonymous study IDs only; conservative language everywhere.

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import requests

from rules.eka_context import validate_document_upload

API_BASE_URL_DEFAULT = "http://127.0.0.1:8000"

DISCLAIMER = (
    "Research prototype; not clinically validated; "
    "not for autonomous diagnosis or treatment decisions."
)

EKA_DISCLAIMER = (
    "Eka Care output is unconfirmed clinical-context support only. It does not "
    "diagnose stroke, does not determine thrombolysis eligibility, and requires "
    "clinician confirmation."
)

# ---------------------------------------------------------------------------
# Status presentation
# ---------------------------------------------------------------------------

STATUS_META: Dict[str, Dict[str, str]] = {
    "HEMORRHAGE_SUSPECTED": {
        "label": "Hemorrhage suspected",
        "color": "#c62828",
        "recommendation": (
            "Urgent radiologist review. A hemorrhage pattern on CT requires "
            "specialist assessment. Decision support only."
        ),
    },
    "URGENT_RADIOLOGY_REVIEW": {
        "label": "Urgent radiology review",
        "color": "#ef6c00",
        "recommendation": (
            "Prompt radiologist review recommended. Imaging or timing requires "
            "specialist assessment before any pathway decision."
        ),
    },
    "INSUFFICIENT_INFORMATION": {
        "label": "Insufficient information",
        "color": "#616161",
        "recommendation": (
            "Required information is missing (e.g., last-known-well time). "
            "Obtain details before any pathway decision."
        ),
    },
    "OUTSIDE_STANDARD_WINDOW": {
        "label": "Outside standard window",
        "color": "#ef6c00",
        "recommendation": (
            "The recorded time window may be exceeded — specialist review "
            "required. This is not a treatment decision."
        ),
    },
    "POTENTIALLY_ELIGIBLE_PENDING_REVIEW": {
        "label": "Pending clinician review",
        "color": "#2e7d32",
        "recommendation": (
            "Checklist complete pending clinician confirmation. Eligibility is "
            "always a clinician decision — never automatic."
        ),
    },
    "NO_HEMORRHAGE_DETECTED": {
        "label": "No hemorrhage detected",
        "color": "#1565c0",
        "recommendation": (
            "No hemorrhage pattern detected by the model. A negative result "
            "never implies treatment eligibility."
        ),
    },
    "NO_ACUTE_STROKE_EVIDENCE": {
        "label": "No acute stroke evidence",
        "color": "#1565c0",
        "recommendation": (
            "No acute stroke pattern detected by the model. A negative result "
            "never implies treatment eligibility."
        ),
    },
    "ISCHEMIC_PATTERN_POSSIBLE": {
        "label": "Ischemic pattern possible",
        "color": "#1565c0",
        "recommendation": (
            "An ischemic pattern may be present — radiologist review required. "
            "This is not a diagnosis."
        ),
    },
    "failed": {
        "label": "Processing failed",
        "color": "#c62828",
        "recommendation": (
            "No result is available. Nothing was fabricated — retry the upload "
            "or escalate to radiology."
        ),
    },
}

TOOLTIPS = {
    "lkw": (
        "Last known well: the last time the patient was seen in their normal "
        "state. Missing LKW triggers INSUFFICIENT_INFORMATION."
    ),
    "hemorrhage_probability": (
        "Highest Model A hemorrhage probability across processed CT slices. "
        "At or above the 0.50 threshold the result is flagged for urgent "
        "radiology review."
    ),
    "imaging_status": (
        "Raw Model A imaging classification. The rule engine maps this to a "
        "conservative final triage status."
    ),
    "rule_trace": (
        "Identifier of the declarative rule that produced the final status. "
        "Full trace is stored for audit."
    ),
    "clinician_confirmation": (
        "Every output requires a clinician to confirm or override it before "
        "it informs any pathway decision."
    ),
    "eka_context": (
        "Unconfirmed clinical context extracted from Eka Care. Never a "
        "diagnosis, never a contraindication, never an automatic exclusion."
    ),
    "model_version": (
        "Version string stored with every prediction for auditability."
    ),
}


def status_display(status: Optional[str]) -> Dict[str, str]:
    """Return label/color/recommendation for a status (unknown → review)."""
    key = status or "URGENT_RADIOLOGY_REVIEW"
    meta = STATUS_META.get(key)
    if meta is None:
        meta = STATUS_META["URGENT_RADIOLOGY_REVIEW"]
        return {
            "label": key.replace("_", " ").title(),
            "color": meta["color"],
            "recommendation": meta["recommendation"],
        }
    return dict(meta)


def triage_recommendation(status: Optional[str]) -> str:
    """Conservative recommendation text — never a treatment instruction."""
    return status_display(status)["recommendation"]


def patient_pseudonym(study_id: str) -> str:
    """Deterministic pseudonymous patient label (never a real identifier)."""
    if not study_id:
        return "—"
    digest = hashlib.sha256(str(study_id).encode("utf-8")).hexdigest()[:10]
    return f"PT-{digest.upper()}"


def format_probability(value: Any) -> str:
    """Format a 0-1 probability as a percentage string."""
    try:
        p = float(value)
    except (TypeError, ValueError):
        return "—"
    if 0.0 <= p <= 1.0:
        return f"{p * 100:.1f}%"
    return f"{p:.1f}%"


def _parse_time(value: Any) -> Optional[datetime]:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if isinstance(value, str) and value:
        try:
            parsed = datetime.fromisoformat(value)
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    return None


def window_countdown(lkw_time: Any, current_time: Any) -> Dict[str, Any]:
    """Elapsed hours since LKW, window flag, and remaining hours (or None)."""
    lkw = _parse_time(lkw_time)
    now = _parse_time(current_time) or datetime.now(timezone.utc)
    if lkw is None:
        return {
            "elapsed_hours": None,
            "within_window": None,
            "remaining_hours": None,
        }
    elapsed = round((now - lkw).total_seconds() / 3600.0, 1)
    within = elapsed <= 4.5
    remaining = round(4.5 - elapsed, 1)
    return {
        "elapsed_hours": elapsed,
        "within_window": within,
        "remaining_hours": remaining if within else None,
    }


def allowed_document(filename: str, content_type: str, content: bytes):
    """Delegate to the shared Eka document validator (DICOM blocked)."""
    return validate_document_upload(filename, content_type, content)


def is_dicom_upload(filename: str, content_type: str, content: bytes) -> bool:
    from rules.eka_context import is_dicom_upload as _is_dicom

    return _is_dicom(filename, content_type, content)


def friendly_error(err: Any) -> str:
    """Turn API/client exceptions into safe, actionable messages."""
    status = getattr(err, "status_code", None)
    detail = getattr(err, "detail", None) or str(err)
    if status == 0:
        return (
            "API unreachable — start the backend with "
            "`uvicorn app.api:app --port 8000`."
        )
    if status == 404:
        return "Not found — check the study ID or reset the session."
    if status == 409:
        return "Study has not finished processing yet."
    if status == 502:
        return "Eka Care request failed — clinical context is unavailable."
    if status == 503:
        return "Eka Care is not configured (set EKA_CLIENT_ID / EKA_CLIENT_SECRET)."
    return str(detail)


def build_audit_export(
    study_id: str,
    result: Optional[Dict[str, Any]],
    eligibility: Optional[Dict[str, Any]],
    audit_events: Optional[List[Dict[str, Any]]],
) -> Dict[str, Any]:
    """Assemble a JSON-safe audit export (no file bytes, no identifiers)."""
    model_a = (result or {}).get("model_a_result") or {}
    return {
        "study_id": study_id,
        "patient_pseudonym": patient_pseudonym(study_id),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "imaging_status": (result or {}).get("imaging_status"),
        "model_a_summary": {
            k: model_a.get(k)
            for k in (
                "max_hemorrhagic_probability",
                "max_ischemic_probability",
                "max_normal_probability",
                "predicted_class",
                "processed_slices",
                "failed_slices",
            )
        },
        "model_a_version": (result or {}).get("model_a_version"),
        "model_b_version": (result or {}).get("model_b_version"),
        "thresholds": (result or {}).get("thresholds"),
        "prediction_timestamp": (result or {}).get("timestamp"),
        "eligibility": {
            "final_status": (eligibility or {}).get("final_status"),
            "rule_trace": (eligibility or {}).get("rule_trace"),
            "elapsed_hours": (eligibility or {}).get("elapsed_hours"),
            "within_standard_window": (eligibility or {}).get(
                "within_standard_window"
            ),
            "requires_radiologist_review": (eligibility or {}).get(
                "requires_radiologist_review"
            ),
            "rationale": (eligibility or {}).get("rationale"),
            "eka_rule_traces": (eligibility or {}).get("eka_rule_traces") or [],
        },
        "audit_events": audit_events or [],
        "disclaimer": DISCLAIMER,
    }


def audit_export_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, default=str)


# ---------------------------------------------------------------------------
# API client (requests-based)
# ---------------------------------------------------------------------------


class ApiClientError(Exception):
    """Safe API error — never contains tokens or patient data."""

    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


class StrokeApiClient:
    """Thin requests wrapper around the local triage API."""

    def __init__(self, base_url: Optional[str] = None, session=None,
                 timeout: float = 30.0):
        self.base_url = (
            base_url or os.environ.get("API_BASE_URL") or API_BASE_URL_DEFAULT
        ).rstrip("/")
        self._session = session or requests.Session()
        self.timeout = timeout

    def _request(self, method: str, path: str, **kwargs) -> Any:
        try:
            resp = self._session.request(
                method, self.base_url + path, timeout=self.timeout, **kwargs
            )
        except requests.RequestException:
            raise ApiClientError(
                0, "API unreachable — start uvicorn app.api:app --port 8000"
            )
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text or "Request failed"
            raise ApiClientError(resp.status_code, str(detail))
        if not resp.content:
            return {}
        try:
            return resp.json()
        except Exception:
            raise ApiClientError(resp.status_code, "Invalid JSON from API")

    # Core triage ---------------------------------------------------------
    def health(self) -> Dict[str, Any]:
        return self._request("GET", "/health")

    def upload_studies(self, files: List[Dict[str, Any]]) -> Dict[str, Any]:
        multipart = [
            ("files", (f["name"], f["bytes"], f.get("content_type",
                                                     "application/dicom")))
            for f in files
        ]
        return self._request("POST", "/studies/upload", files=multipart)

    def study_status(self, study_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/studies/{study_id}/status")

    def study_result(self, study_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/studies/{study_id}/result")

    def eligibility(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/eligibility", json=payload)

    def audit_create(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/audit", json=payload)

    def audit_list(self, study_id: str) -> List[Dict[str, Any]]:
        return self._request("GET", f"/audit/{study_id}")

    # Eka clinical context ------------------------------------------------
    def eka_start(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/eka/assessment/start", json=payload)

    def eka_continue(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/eka/assessment/continue", json=payload)

    def eka_submit(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/eka/assessment/submit", json=payload)

    def eka_context(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._request("POST", "/eka/context", json=payload)

    def eka_upload_document(
        self, filename: str, content: bytes, content_type: str,
        study_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        files = {"file": (filename, content, content_type)}
        data = {"study_id": study_id} if study_id else {}
        return self._request(
            "POST", "/eka/documents/upload", files=files, data=data
        )

    def eka_document_status(self, document_id: str) -> Dict[str, Any]:
        return self._request(
            "GET", f"/eka/documents/{document_id}/status"
        )

    def eka_document_parsed(self, document_id: str) -> Dict[str, Any]:
        return self._request("GET", f"/eka/documents/{document_id}/parsed")

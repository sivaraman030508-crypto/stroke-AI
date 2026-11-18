# Eka Care API client (clinical-context layer only).
#
# Scope: symptom/risk assessment, medical-document parsing, symptom and
# diagnosis search. NEVER used for CT interpretation, stroke diagnosis,
# thrombolysis eligibility decisions, or treatment recommendations.
#
# Credentials come from environment variables (never hard-coded):
#   EKA_CLIENT_ID, EKA_CLIENT_SECRET, EKA_API_KEY, EKA_BASE_URL
# Tokens are held in memory only; nothing sensitive is ever logged.

import time
import uuid
from typing import Any, Dict, Optional

import httpx

DEFAULT_BASE_URL = "https://api.eka.care"
DEFAULT_TIMEOUT = 20.0

AUTH_LOGIN_PATH = "/connect-auth/v1/account/login"
AUTH_REFRESH_PATH = "/connect-auth/v1/account/refresh-token"
ASSESSMENT_INIT_PATH = "/assessment/api/v1/init/"
ASSESSMENT_START_PATH = "/assessment/api/v1/start/{assessment_id}"
ASSESSMENT_CONTINUE_PATH = "/assessment/api/v1/continue/{assessment_id}"
ASSESSMENT_SUBMIT_PATH = "/assessment/api/v1/submit/{assessment_id}"
DOCUMENT_UPLOAD_PATH = "/mr/api/v2/docs"
DOCUMENT_RESULT_PATH = "/mr/api/v1/docs/{document_id}/result"
SYMPTOM_SEARCH_PATH = "/assessment/api/v1/search"

DEFAULT_WORKFLOW_ID = 1000


class EkaConfigurationError(RuntimeError):
    """Eka Care credentials/base URL are not configured."""


class EkaAuthError(RuntimeError):
    """Eka Care authentication failed."""


class EkaAPIError(RuntimeError):
    """Eka Care API request failed (safe, token-free message)."""

    def __init__(self, status_code: int, message: str = ""):
        self.status_code = status_code
        super().__init__(message or f"Eka Care request failed (HTTP {status_code}).")


class EkaCareClient:
    """Synchronous httpx client with in-memory token management."""

    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        transport: Optional[httpx.BaseTransport] = None,
        timeout: float = DEFAULT_TIMEOUT,
        workflow_id: int = DEFAULT_WORKFLOW_ID,
        http_client: Optional[httpx.Client] = None,
    ):
        import os

        self._client_id = client_id or os.environ.get("EKA_CLIENT_ID", "")
        self._client_secret = client_secret or os.environ.get("EKA_CLIENT_SECRET", "")
        self._api_key = api_key or os.environ.get("EKA_API_KEY", "")
        self._base_url = (base_url or os.environ.get("EKA_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.workflow_id = workflow_id

        if not self._client_id or not self._client_secret:
            raise EkaConfigurationError(
                "Eka Care is not configured. Set EKA_CLIENT_ID and "
                "EKA_CLIENT_SECRET environment variables."
            )

        # Tokens live in memory only — never written to disk.
        self._access_token: Optional[str] = None
        self._refresh_token: Optional[str] = None
        self._expires_at: float = 0.0

        if http_client is not None:
            self._client = http_client
        else:
            self._client = httpx.Client(
                base_url=self._base_url,
                timeout=httpx.Timeout(timeout, connect=5.0),
                transport=transport,
            )

    # ------------------------------------------------------------------
    # Token management (memory only, auto-refresh on 401)
    # ------------------------------------------------------------------

    @property
    def base_url(self) -> str:
        return self._base_url

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "EkaCareClient":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def get_access_token(self) -> str:
        """Return a valid access token, authenticating if needed."""
        if self._access_token and time.time() < self._expires_at:
            return self._access_token
        self._authenticate()
        return self._access_token or ""

    def refresh_access_token(self) -> str:
        """Refresh the access token using the in-memory refresh token."""
        if not self._refresh_token:
            self._authenticate()
            return self._access_token or ""

        response = self._client.post(
            AUTH_REFRESH_PATH,
            headers={
                "Authorization": f"Bearer {self._access_token or ''}",
                "Client-Id": self._client_id,
            },
            json={"refresh_token": self._refresh_token,
                  "access_token": self._access_token},
        )
        if response.status_code >= 400:
            # Fall back to a full re-authentication.
            self._access_token = None
            self._refresh_token = None
            self._authenticate()
            return self._access_token or ""

        data = response.json()
        self._store_tokens(data)
        return self._access_token or ""

    def _store_tokens(self, data: Dict[str, Any]) -> None:
        self._access_token = data.get("access_token")
        self._refresh_token = data.get("refresh_token", self._refresh_token)
        expires_in = int(data.get("expires_in") or 3600)
        self._expires_at = time.time() + max(expires_in - 30, 30)

    def _authenticate(self) -> None:
        response = self._client.post(
            AUTH_LOGIN_PATH,
            json={"client_id": self._client_id,
                  "client_secret": self._client_secret},
        )
        if response.status_code >= 400:
            raise EkaAuthError("Eka Care authentication failed.")
        data = response.json()
        if not data.get("access_token"):
            raise EkaAuthError("Eka Care authentication failed.")
        self._store_tokens(data)

    def _headers(self) -> Dict[str, str]:
        token = self._access_token or self._api_key or ""
        return {
            "auth": token,
            "client-id": self._client_id,
            "Accept": "application/json",
        }

    def _request(self, method: str, path: str, **kwargs) -> Any:
        """Perform an authenticated request; refresh once on 401."""
        self.get_access_token()

        response = self._client.request(method, path, headers=self._headers(), **kwargs)
        if response.status_code == 401:
            self.refresh_access_token()
            response = self._client.request(
                method, path, headers=self._headers(), **kwargs
            )
        if response.status_code >= 400:
            raise EkaAPIError(response.status_code)
        if not response.content:
            return {}
        try:
            return response.json()
        except Exception:
            raise EkaAPIError(response.status_code,
                              "Invalid JSON response from Eka Care.")

    # ------------------------------------------------------------------
    # Assessment flow
    # ------------------------------------------------------------------

    def start_assessment(
        self,
        patient_pseudonym_id: str,
        age: int,
        sex: str = "unknown",
        presenting_symptoms=None,
        symptom_onset_time: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Initialize an assessment and fetch the first question.

        Uses only a pseudonymous patient identifier — never real identifiers.
        """
        gender_map = {"male": "m", "female": "f", "male ": "m"}
        gender = gender_map.get((sex or "").strip().lower(), "o")

        init_payload = {
            "user_info": {"age": int(age), "gender": gender},
            "workflow_id": self.workflow_id,
            "unique_identifier": patient_pseudonym_id,
            "transaction_id": uuid.uuid4().hex,
        }
        init_data = self._request("POST", ASSESSMENT_INIT_PATH, json=init_payload)
        assessment_id = init_data.get("assessment_id") or init_data.get("id")
        if not assessment_id:
            raise EkaAPIError(200, "Eka Care did not return an assessment id.")

        start_path = ASSESSMENT_START_PATH.format(assessment_id=assessment_id)
        start_data = self._request("PUT", start_path)
        return {
            "assessment_id": assessment_id,
            "start_response": start_data,
            "presenting_symptoms": list(presenting_symptoms or []),
            "symptom_onset_time": symptom_onset_time,
        }

    def continue_assessment(self, assessment_id: str, question_id: str,
                            answer: Any) -> Dict[str, Any]:
        """Submit an answer and get the next question (or completion)."""
        path = ASSESSMENT_CONTINUE_PATH.format(assessment_id=assessment_id)
        try:
            qid: Any = int(question_id)
        except (TypeError, ValueError):
            qid = question_id
        payload = {"qid": qid, "user_response": answer}
        return self._request("PUT", path, json=payload)

    def submit_assessment(self, assessment_id: str) -> Dict[str, Any]:
        """Submit a completed assessment and get the summary."""
        path = ASSESSMENT_SUBMIT_PATH.format(assessment_id=assessment_id)
        return self._request("PUT", path)

    # ------------------------------------------------------------------
    # Medical-document parsing (PDF/images only — never CT DICOM)
    # ------------------------------------------------------------------

    def upload_medical_document(
        self,
        filename: str,
        content: bytes,
        content_type: str = "application/pdf",
        document_type: str = "lr",
        task: str = "smart",
    ) -> Dict[str, Any]:
        """Upload a PDF/image medical document; returns document_id."""
        files = {"file": (filename, content, content_type or "application/pdf")}
        return self._request(
            "POST", DOCUMENT_UPLOAD_PATH,
            params={"dt": document_type, "task": task},
            files=files,
        )

    def get_document_status(self, document_id: str) -> Dict[str, Any]:
        """Check parsing status of an uploaded document."""
        path = DOCUMENT_RESULT_PATH.format(document_id=document_id)
        data = self._request("GET", path)
        return {
            "document_id": document_id,
            "status": data.get("status", "unknown"),
            "raw": data,
        }

    def get_parsed_report(self, document_id: str) -> Dict[str, Any]:
        """Fetch the parsed report for an uploaded document."""
        path = DOCUMENT_RESULT_PATH.format(document_id=document_id)
        data = self._request("GET", path)
        data.setdefault("document_id", document_id)
        return data

    # ------------------------------------------------------------------
    # Symptom / diagnosis search
    # ------------------------------------------------------------------

    def search_symptom_or_diagnosis(self, query: str) -> Dict[str, Any]:
        """Search symptoms/diagnoses (ICD-10 style coding support).

        Adjust SYMPTOM_SEARCH_PATH to match your Eka contract if needed.
        """
        data = self._request("GET", SYMPTOM_SEARCH_PATH, params={"q": query})
        if isinstance(data, dict):
            return data
        return {"results": data if isinstance(data, list) else []}

# Streamlit clinical dashboard — Stroke CT Triage Assistant.
# Decision support only: never autonomous diagnosis, never a treatment
# recommendation, never a thrombolysis administration instruction.
# UI: white background, light gray panels, blue accent; red = hemorrhage,
# amber = review, green only for completed review / pending-review status.

import io
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np
import plotly.graph_objects as go
import pydicom
import streamlit as st
from PIL import Image

from app.dashboard_utils import (
    DISCLAIMER,
    EKA_DISCLAIMER,
    TOOLTIPS,
    ApiClientError,
    StrokeApiClient,
    allowed_document,
    audit_export_json,
    build_audit_export,
    friendly_error,
    format_probability,
    patient_pseudonym,
    status_display,
    triage_recommendation,
    window_countdown,
)

st.set_page_config(
    page_title="Stroke CT Triage Assistant",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

_STATE_DEFAULTS: Dict[str, Any] = {
    "files": [],               # [{"name": str, "bytes": bytes}]
    "pixel_arrays": None,      # List[Optional[np.ndarray]]
    "pixel_key": None,
    "study_id": None,
    "study_status": None,
    "result": None,
    "eligibility": None,
    "audit_events": None,
    "eka_assessment": None,    # {assessment_id, question, history, result}
    "eka_docs": [],            # [{document_id, filename, status, parsed}]
    "last_context": None,
    "sb_lkw": None,
    "sb_now": datetime.now().replace(microsecond=0),
    "sb_quality": "good",
}

for _key, _default in _STATE_DEFAULTS.items():
    if _key not in st.session_state:
        st.session_state[_key] = _default

client = StrokeApiClient()

# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
    .stApp { background-color: #ffffff; }
    section[data-testid="stSidebar"] { background-color: #f5f6f8; }
    div[data-testid="stVerticalBlock"] > div:has(> div.stMarkdown) {
        background-color: #f7f8fa;
        border: 1px solid #e3e6ea;
        border-radius: 8px;
        padding: 12px 16px;
    }
    .triage-banner {
        border-radius: 8px;
        padding: 14px 18px;
        margin: 6px 0 12px 0;
    }
    .demo-badge {
        background-color: #e8f0fe;
        color: #1a56c4;
        border: 1px solid #c9dcfb;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
    }
    .header-sub { color: #5f6368; font-size: 0.9rem; margin-top: 2px; }
    .model-chip {
        display: inline-block;
        background-color: #f1f3f4;
        border-radius: 12px;
        padding: 2px 10px;
        font-size: 0.78rem;
        margin-left: 6px;
    }
</style>
""",
    unsafe_allow_html=True,
)


def render_status_banner(status: Optional[str], tooltip_key: Optional[str] = None):
    """Colored triage banner: red hemorrhage, amber review, gray insufficient,
    blue ischemic pattern, green completed-review pending status."""
    meta = status_display(status)
    tip = ""
    if tooltip_key and tooltip_key in TOOLTIPS:
        tip = f' <span title="{TOOLTIPS[tooltip_key]}">?</span>'
    st.markdown(
        f'<div class="triage-banner" style="background: {meta["color"]}14; '
        f'border-left: 6px solid {meta["color"]};">'
        f'<span style="color: {meta["color"]}; font-weight: 700;">'
        f'{meta["label"]}</span>{tip}<br>'
        f'<span style="color: #3c4043;">{meta["recommendation"]}</span></div>',
        unsafe_allow_html=True,
    )


# ---------------------------------------------------------------------------
# Header (persistent)
# ---------------------------------------------------------------------------


def fetch_health() -> Optional[Dict[str, Any]]:
    try:
        return client.health()
    except ApiClientError:
        return None


def render_header():
    health = fetch_health()
    left, right = st.columns([7, 3])
    with left:
        st.markdown("## Stroke CT Triage Assistant")
        st.markdown(
            '<div class="header-sub">Clinical decision support research '
            "prototype — never autonomous diagnosis</div>",
            unsafe_allow_html=True,
        )
    with right:
        st.markdown(
            '<div style="text-align: right;">'
            '<span class="demo-badge">LOCAL DEMO</span><br>'
            f'<span class="model-chip">Model A: '
            f'{"loaded" if (health or {}).get("model_a_loaded") else "not loaded"}</span>'
            f'<span class="model-chip">Model B: '
            f'{"loaded" if (health or {}).get("model_b_loaded") else "not loaded"}</span>'
            "</div>",
            unsafe_allow_html=True,
        )
        if health is None:
            st.caption("API unreachable — start uvicorn on port 8000.")


# ---------------------------------------------------------------------------
# Sidebar (persistent)
# ---------------------------------------------------------------------------


def render_sidebar():
    with st.sidebar:
        st.markdown("### Study")

        study_input = st.text_input(
            "Study ID (pseudonymous)",
            value=st.session_state.study_id or "",
            help="UUID assigned at upload — never a real patient identifier.",
        )
        if study_input and study_input != st.session_state.study_id:
            st.session_state.study_id = study_input

        pseudonym = patient_pseudonym(st.session_state.study_id or "")
        st.markdown(f"**Patient pseudonym:** `{pseudonym}`")

        if st.session_state.study_id and st.button("Load study"):
            try:
                st.session_state.study_status = client.study_status(
                    st.session_state.study_id
                )["status"]
                if st.session_state.study_status == "completed":
                    st.session_state.result = client.study_result(
                        st.session_state.study_id
                    )
            except ApiClientError as exc:
                st.error(friendly_error(exc))

        st.markdown("### Timing & quality")
        st.session_state.sb_lkw = st.datetime_input(
            "Last known well (LKW)",
            value=st.session_state.sb_lkw,
            help=TOOLTIPS["lkw"],
        )
        st.session_state.sb_now = st.datetime_input(
            "Current time",
            value=st.session_state.sb_now,
            help="Update to the time of this review.",
        )
        st.session_state.sb_quality = st.selectbox(
            "Image quality",
            ["good", "poor", "low_confidence"],
            index=["good", "poor", "low_confidence"].index(
                st.session_state.sb_quality
            ),
            help="Poor quality triggers urgent radiology review.",
        )

        countdown = window_countdown(st.session_state.sb_lkw, st.session_state.sb_now)
        if countdown["elapsed_hours"] is not None:
            st.caption(
                f"Elapsed since LKW: {countdown['elapsed_hours']} h"
                + (
                    f" — {countdown['remaining_hours']} h remaining in 4.5 h window"
                    if countdown["within_window"]
                    else " — outside standard window"
                )
            )

        st.markdown("### Checklist status")
        st.caption(
            f"- Study loaded: {'yes' if st.session_state.study_id else 'no'}\n"
            f"- LKW provided: {'yes' if st.session_state.sb_lkw else 'no'}\n"
            f"- Image quality: {st.session_state.sb_quality}\n"
            f"- Result available: {'yes' if st.session_state.result else 'no'}\n"
            f"- Eligibility run: "
            f"{'yes' if st.session_state.eligibility else 'no'}"
        )

        st.markdown("### Session")
        if st.button("Reset session"):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()


# ---------------------------------------------------------------------------
# Triage Dashboard tab
# ---------------------------------------------------------------------------


def poll_study(study_id: str, timeout_s: float = 60.0) -> Optional[str]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            status = client.study_status(study_id)["status"]
        except ApiClientError as exc:
            st.error(friendly_error(exc))
            return None
        if status in ("completed", "failed"):
            return status
        time.sleep(1.5)
    return "timeout"


def render_triage_tab():
    st.markdown("### Triage dashboard")

    if not st.session_state.files:
        st.info(
            "Empty state — upload CT DICOM slices below to start a triage "
            "review. No output is shown until a study completes."
        )

    uploads = st.file_uploader(
        "CT DICOM slices",
        type=["dcm", "dicom"],
        accept_multiple_files=True,
        help="Multiple slices of one study. Stored ephemerally; never sent "
        "to Eka Care.",
    )
    if uploads:
        names = [u.name for u in uploads]
        current = [f["name"] for f in st.session_state.files]
        if names != current:
            st.session_state.files = [
                {"name": u.name, "bytes": u.getvalue()} for u in uploads
            ]
            st.session_state.pixel_arrays = None
            st.session_state.pixel_key = None

    col_run, col_status = st.columns([1, 2])
    with col_run:
        run_clicked = st.button(
            "Run triage analysis",
            type="primary",
            disabled=not st.session_state.files,
        )
    with col_status:
        if st.session_state.study_id:
            st.caption(
                f"Study `{st.session_state.study_id}` — status: "
                f"**{st.session_state.study_status or 'unknown'}**"
            )

    if run_clicked and st.session_state.files:
        with st.spinner("Uploading and queueing study…"):
            try:
                resp = client.upload_studies(st.session_state.files)
                st.session_state.study_id = resp["study_id"]
                st.session_state.study_status = resp.get("status", "queued")
                st.session_state.result = None
                st.session_state.eligibility = None
                st.session_state.audit_events = None
            except ApiClientError as exc:
                st.error(friendly_error(exc))
                return

    study_id = st.session_state.study_id
    if not study_id:
        return

    status = st.session_state.study_status
    if status in ("queued", "processing") and st.session_state.result is None:
        with st.spinner("Processing CT slices with Model A / Model B…"):
            status = poll_study(study_id)
            st.session_state.study_status = status
            if status == "completed":
                try:
                    st.session_state.result = client.study_result(study_id)
                except ApiClientError as exc:
                    st.error(friendly_error(exc))
                    return

    if status == "timeout":
        st.warning(
            "Still processing — refresh the page in a moment to load the "
            "result."
        )
        return

    if status == "failed":
        render_status_banner("failed")
        st.error(
            "Processing failed for this study; no result available. "
            "Nothing was fabricated — retry or escalate to radiology."
        )
        return

    result = st.session_state.result
    if not result:
        st.info("Empty state — no result yet for this study.")
        return

    imaging_status = result.get("imaging_status")
    render_status_banner(imaging_status, tooltip_key="imaging_status")

    model_a = result.get("model_a_result") or {}
    st.markdown(
        f'<span title="{TOOLTIPS["hemorrhage_probability"]}">'
        "<b>Model A — slice classification</b></span>",
        unsafe_allow_html=True,
    )
    m1, m2, m3, m4 = st.columns(4)
    m1.metric(
        "Hemorrhage probability",
        format_probability(model_a.get("max_hemorrhagic_probability")),
        help=TOOLTIPS["hemorrhage_probability"],
    )
    m2.metric(
        "Ischemic probability",
        format_probability(model_a.get("max_ischemic_probability")),
        help=TOOLTIPS["imaging_status"],
    )
    m3.metric(
        "Normal probability",
        format_probability(model_a.get("max_normal_probability")),
        help=TOOLTIPS["imaging_status"],
    )
    m4.metric(
        "Slices processed",
        f"{model_a.get('processed_slices', 0)} "
        f"(failed: {model_a.get('failed_slices', 0)})",
        help="Slices that could be decoded and scored; failed slices are "
        "never treated as normal.",
    )

    probs = {
        "normal": float(model_a.get("max_normal_probability") or 0),
        "ischemic": float(model_a.get("max_ischemic_probability") or 0),
        "hemorrhagic": float(model_a.get("max_hemorrhagic_probability") or 0),
    }
    colors = {"normal": "#2e7d32", "ischemic": "#1565c0",
              "hemorrhagic": "#c62828"}
    fig = go.Figure(
        go.Bar(
            x=list(probs.keys()),
            y=list(probs.values()),
            marker_color=[colors[k] for k in probs.keys()],
            text=[format_probability(v) for v in probs.values()],
            textposition="outside",
        )
    )
    fig.update_layout(
        height=300,
        yaxis_title="Max probability across slices",
        yaxis_range=[0, 1],
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig)

    top_slices = result.get("top_slices") or []
    if top_slices:
        st.caption(
            f"Top hemorrhage-risk slices (Model A ranking): "
            f"{', '.join(str(s) for s in top_slices)}"
        )

    model_b = result.get("model_b_result")
    if model_b and model_b.get("highest_probability_subtype"):
        st.markdown("**Model B — hemorrhage subtype (unconfirmed)**")
        st.caption(
            f"Highest probability subtype: "
            f"**{model_b['highest_probability_subtype']}**"
        )
        subtype_probs = model_b.get("subtype_probabilities") or {}
        if subtype_probs:
            st.bar_chart(subtype_probs)

    with st.expander("Model versions, thresholds & provenance"):
        st.caption(
            f"Model A version: {result.get('model_a_version')} — "
            f"{TOOLTIPS['model_version']}"
        )
        st.caption(
            f"Model B version: {result.get('model_b_version') or 'n/a'}"
        )
        st.json(result.get("thresholds") or {})
        st.caption(f"Prediction timestamp: {result.get('timestamp')}")

    st.caption(DISCLAIMER)


# ---------------------------------------------------------------------------
# CT Review tab
# ---------------------------------------------------------------------------


def _load_pixel_arrays(files: List[Dict[str, Any]]) -> List[Optional[np.ndarray]]:
    arrays: List[Optional[np.ndarray]] = []
    for item in files:
        try:
            ds = pydicom.dcmread(io.BytesIO(item["bytes"]))
            arr = np.asarray(ds.pixel_array, dtype=np.float32)
            slope = float(getattr(ds, "RescaleSlope", 1) or 1)
            intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
            arrays.append(arr * slope + intercept)
        except Exception:
            arrays.append(None)
    return arrays


def _display_normalize(arr: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(arr, 2), np.percentile(arr, 98)
    if hi <= lo:
        hi = lo + 1.0
    scaled = (np.clip(arr, lo, hi) - lo) / (hi - lo) * 255.0
    return scaled.astype(np.uint8)


def _brain_window(arr: np.ndarray, wl: float = 40.0, ww: float = 80.0) -> np.ndarray:
    lo, hi = wl - ww / 2.0, wl + ww / 2.0
    scaled = (np.clip(arr, lo, hi) - lo) / max(hi - lo, 1e-6) * 255.0
    return scaled.astype(np.uint8)


def render_ct_tab():
    st.markdown("### CT review")

    if not st.session_state.files:
        st.info(
            "Empty state — upload DICOM files on the Triage Dashboard tab "
            "to review slices."
        )
        return

    file_key = tuple(
        (f["name"], len(f["bytes"])) for f in st.session_state.files
    )
    if st.session_state.pixel_key != file_key:
        with st.spinner("Loading slices…"):
            st.session_state.pixel_arrays = _load_pixel_arrays(st.session_state.files)
        st.session_state.pixel_key = file_key

    arrays = st.session_state.pixel_arrays or []
    valid_indices = [i for i, a in enumerate(arrays) if a is not None]
    if not valid_indices:
        st.error(
            "No readable slices — processing would be unsafe. "
            "No result was fabricated."
        )
        return

    result = st.session_state.result or {}
    top_slices = result.get("top_slices") or []
    default_idx = top_slices[0] if top_slices else valid_indices[0]
    if not (0 <= default_idx < len(arrays)) or arrays[default_idx] is None:
        default_idx = valid_indices[0]

    if len(arrays) <= 1:
        idx = int(default_idx)
        st.caption("Single slice loaded — slice navigation disabled.")
    else:
        idx = st.slider(
            "Slice index",
            0,
            len(arrays) - 1,
            value=int(default_idx),
            help="Top Model A hemorrhage-risk slices are pre-selected.",
        )
    if top_slices:
        st.caption(
            f"Top-risk slices: {', '.join(str(s) for s in top_slices)} "
            f"(current: {idx})"
        )

    arr = arrays[idx]
    if arr is None:
        st.warning(f"Slice {idx} could not be decoded.")
        return

    col_orig, col_win = st.columns(2)
    with col_orig:
        st.caption("Original (percentile-normalized)")
        st.image(Image.fromarray(_display_normalize(arr), mode="L"), width="stretch")
    with col_win:
        st.caption("Brain window (WL 40 / WW 80)")
        st.image(Image.fromarray(_brain_window(arr), mode="L"), width="stretch")

    slice_details = (result.get("model_a_result") or {}).get("slice_details") or []
    if slice_details:
        class_colors = {
            "hemorrhagic": "#c62828",
            "ischemic": "#1565c0",
            "normal": "#2e7d32",
            "uncertain": "#ef6c00",
        }
        fig = go.Figure(
            go.Bar(
                x=[d["slice_index"] for d in slice_details],
                y=[d["confidence"] for d in slice_details],
                marker_color=[
                    class_colors.get(d.get("predicted_class"), "#9aa0a6")
                    for d in slice_details
                ],
                hovertemplate=(
                    "Slice %{x}<br>Confidence %{y:.2f}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            height=260,
            xaxis_title="Slice index",
            yaxis_title="Confidence",
            margin=dict(l=10, r=10, t=20, b=10),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig)
        st.caption(
            "Red = hemorrhagic, blue = ischemic, green = normal, "
            "amber = uncertain (Model A per-slice output)."
        )

    st.caption(DISCLAIMER)


# ---------------------------------------------------------------------------
# Eligibility Assistant tab
# ---------------------------------------------------------------------------


def render_eligibility_tab():
    st.markdown("### Eligibility assistant")
    st.caption(
        "Runs the declarative rule engine. Output is a conservative triage "
        "status for clinician review — never a treatment instruction."
    )

    study_id = st.session_state.study_id
    if not study_id or not st.session_state.result:
        st.info("Empty state — complete a triage analysis first.")
        return

    with st.form("eligibility_form"):
        st.markdown(
            f'<span title="{TOOLTIPS["lkw"]}"><b>Timing (from sidebar)</b></span>',
            unsafe_allow_html=True,
        )
        lkw = st.session_state.sb_lkw
        now = st.session_state.sb_now
        st.caption(
            f"LKW: {lkw if lkw else 'not provided'} | Current: {now} | "
            f"Image quality: {st.session_state.sb_quality}"
        )
        countdown = window_countdown(lkw, now)
        if countdown["within_window"] is False:
            st.warning("LKW is outside the standard 4.5-hour window.")
        elif lkw is None:
            st.warning(
                "LKW missing — the engine will return "
                "INSUFFICIENT_INFORMATION."
            )

        c1, c2 = st.columns(2)
        with c1:
            htn = st.checkbox("Severe uncontrolled hypertension")
            prior_hem = st.checkbox("Prior intracranial hemorrhage")
            bleeding = st.checkbox("Active bleeding")
        with c2:
            surgery = st.checkbox("Recent major surgery or trauma")
            recent_stroke = st.checkbox(
                "Recent ischemic stroke or head injury"
            )
        anticoag = st.selectbox(
            "Anticoagulant status",
            ["negative", "positive", "unknown"],
            help="Clinician-entered; Eka suggestions are never auto-applied.",
        )
        include_eka = st.checkbox(
            "Include unconfirmed Eka clinical context",
            help=TOOLTIPS["eka_context"],
        )
        submitted = st.form_submit_button("Run eligibility review")

    if submitted:
        payload: Dict[str, Any] = {
            "study_id": study_id,
            "lkw_time": lkw.isoformat() if lkw else None,
            "current_time": now.isoformat() if now else None,
            "image_quality": st.session_state.sb_quality,
            "severe_uncontrolled_hypertension": htn,
            "prior_intracranial_hemorrhage": prior_hem,
            "active_bleeding": bleeding,
            "recent_major_surgery_or_trauma": surgery,
            "anticoagulant_status": anticoag,
            "recent_ischemic_stroke_or_head_injury": recent_stroke,
        }
        if include_eka:
            ctx = st.session_state.last_context or {}
            eka_inputs = ctx.get("eka_engine_inputs") or {}
            if eka_inputs:
                payload.update(eka_inputs)
            else:
                st.warning(
                    "No Eka context available yet — build it on the "
                    "Eka Context tab first."
                )
        try:
            st.session_state.eligibility = client.eligibility(payload)
        except ApiClientError as exc:
            st.error(friendly_error(exc))

    eligibility = st.session_state.eligibility
    if not eligibility:
        st.caption("No eligibility run recorded yet for this session.")
        return

    render_status_banner(
        eligibility.get("final_status"), tooltip_key="rule_trace"
    )

    e1, e2, e3 = st.columns(3)
    e1.metric("Elapsed since LKW",
              f"{eligibility.get('elapsed_hours')} h"
              if eligibility.get("elapsed_hours") is not None else "—",
              help=TOOLTIPS["lkw"])
    e2.metric(
        "Within 4.5 h window",
        {True: "yes", False: "no", None: "unknown"}[
            eligibility.get("within_standard_window")
        ],
        help="Standard thrombolysis review window is 4.5 hours from LKW. "
        "This is a review flag, not a treatment decision.",
    )
    e3.metric(
        "Radiologist review required",
        "yes" if eligibility.get("requires_radiologist_review") else "no",
        help=TOOLTIPS["clinician_confirmation"],
    )

    with st.expander("Rule trace & rationale"):
        st.code(eligibility.get("rule_trace", ""), language="text")
        st.caption(TOOLTIPS["rule_trace"])
        st.write(eligibility.get("rationale", ""))

    if eligibility.get("eka_warnings"):
        with st.expander(
            f"Eka clinical-context warnings "
            f"({len(eligibility['eka_warnings'])}) — unconfirmed"
        ):
            for warning in eligibility["eka_warnings"]:
                st.warning(
                    f"`{warning.get('trace')}` — {warning.get('message')}"
                )
            for trace in eligibility.get("eka_rule_traces") or []:
                st.caption(trace)
            st.caption(TOOLTIPS["eka_context"])

    if eligibility.get("missing_fields"):
        st.info(
            "Missing fields: "
            + ", ".join(eligibility["missing_fields"])
        )

    st.markdown("#### Clinician confirmation")
    st.caption(TOOLTIPS["clinician_confirmation"])
    confirmed = st.checkbox(
        "I have reviewed this output as a clinician",
        key="elig_confirm",
    )
    if confirmed:
        event_col, notes_col = st.columns([1, 2])
        with event_col:
            event_type = st.radio(
                "Record as", ["confirmation", "override"], horizontal=True
            )
        with notes_col:
            notes = st.text_area("Clinician notes (optional)", height=68)
        if st.button("Record audit event"):
            try:
                client.audit_create(
                    {
                        "study_id": study_id,
                        "event_type": event_type,
                        "clinician_notes": notes or None,
                    }
                )
                st.session_state.audit_events = None
                st.success("Audit event recorded.")
            except ApiClientError as exc:
                st.error(friendly_error(exc))

    st.caption(DISCLAIMER)


# ---------------------------------------------------------------------------
# Eka Context tab
# ---------------------------------------------------------------------------


def render_eka_tab():
    st.markdown("### Eka clinical context")
    st.info(EKA_DISCLAIMER)

    study_id = st.session_state.study_id

    # --- Assessment Q&A ---------------------------------------------------
    st.markdown("#### Symptom / risk assessment")
    eka = st.session_state.eka_assessment

    if eka is None:
        with st.form("eka_start_form"):
            age = st.number_input("Age", min_value=0, max_value=130, value=65)
            sex = st.selectbox(
                "Sex", ["unknown", "male", "female", "other"]
            )
            symptoms = st.text_input(
                "Presenting symptoms (comma-separated)"
            )
            onset = st.text_input(
                "Symptom onset time (ISO format, optional)"
            )
            if st.form_submit_button("Start assessment"):
                try:
                    resp = client.eka_start(
                        {
                            "patient_pseudonym_id": patient_pseudonym(
                                study_id or "local-demo"
                            ),
                            "age": int(age),
                            "sex": sex,
                            "presenting_symptoms": [
                                s.strip()
                                for s in symptoms.split(",")
                                if s.strip()
                            ],
                            "symptom_onset_time": onset or None,
                        }
                    )
                    st.session_state.eka_assessment = {
                        "assessment_id": resp.get("assessment_id"),
                        "question": resp.get("question"),
                        "history": [],
                        "result": None,
                    }
                    st.rerun()
                except ApiClientError as exc:
                    st.error(friendly_error(exc))
    else:
        if eka.get("result"):
            result = eka["result"]
            render_status_banner("POTENTIALLY_ELIGIBLE_PENDING_REVIEW")
            st.markdown(f"**Assessment summary:** {result.get('summary', '')}")
            considerations = result.get("suggested_considerations") or []
            if considerations:
                st.markdown("**Suggested considerations (unconfirmed)**")
                for item in considerations:
                    st.caption(f"- {item}")
            st.caption(TOOLTIPS["eka_context"])
            if st.button("Start a new assessment"):
                st.session_state.eka_assessment = None
                st.rerun()
        elif eka.get("question"):
            q = eka["question"]
            st.markdown(
                f"**Question {len(eka['history']) + 1}** "
                f"(`{q.get('id')}`): {q.get('text', '')}"
            )
            answer_key = f"eka_ans_{q.get('id')}_{len(eka['history'])}"
            if q.get("options"):
                answer = st.radio(
                    "Answer", q["options"], key=answer_key, horizontal=True
                )
            else:
                answer = st.text_input("Answer", key=answer_key)
            if st.button("Submit answer"):
                try:
                    resp = client.eka_continue(
                        {
                            "assessment_id": eka["assessment_id"],
                            "question_id": q.get("id", ""),
                            "answer": answer,
                        }
                    )
                    eka["history"].append(
                        {"question": q.get("text"), "answer": answer}
                    )
                    next_q = resp.get("question")
                    if resp.get("status") == "completed" or not next_q:
                        submission = client.eka_submit(
                            {"assessment_id": eka["assessment_id"]}
                        )
                        eka["result"] = submission.get("result") or {}
                        eka["question"] = None
                    else:
                        eka["question"] = next_q
                    st.rerun()
                except ApiClientError as exc:
                    st.error(friendly_error(exc))
            if st.button("Submit assessment now"):
                try:
                    submission = client.eka_submit(
                        {"assessment_id": eka["assessment_id"]}
                    )
                    eka["result"] = submission.get("result") or {}
                    eka["question"] = None
                    st.rerun()
                except ApiClientError as exc:
                    st.error(friendly_error(exc))
        else:
            st.warning("Assessment has no active question.")
            if st.button("Reset assessment"):
                st.session_state.eka_assessment = None
                st.rerun()

    st.divider()

    # --- Documents --------------------------------------------------------
    st.markdown("#### Medical documents (PDF/images — never CT)")
    st.caption(
        "CT DICOM files are rejected here and are never sent to Eka Care."
    )
    doc_uploads = st.file_uploader(
        "Clinical documents",
        type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
        key="eka_doc_uploader",
    )
    if st.button(
        "Send documents to Eka",
        disabled=not doc_uploads,
    ):
        existing = {d["filename"] for d in st.session_state.eka_docs}
        for upload in doc_uploads or []:
            if upload.name in existing:
                continue
            content = upload.getvalue()
            content_type = upload.type or "application/pdf"
            ok, reason = allowed_document(upload.name, content_type, content)
            if not ok:
                st.error(f"{upload.name}: {reason}")
                continue
            try:
                resp = client.eka_upload_document(
                    upload.name, content, content_type, study_id
                )
                st.session_state.eka_docs.append(
                    {
                        "document_id": resp.get("document_id"),
                        "filename": upload.name,
                        "status": resp.get("status", "queued"),
                        "parsed": None,
                    }
                )
            except ApiClientError as exc:
                st.error(f"{upload.name}: {friendly_error(exc)}")

    for doc in st.session_state.eka_docs:
        with st.expander(f"{doc['filename']} — {doc.get('status', 'unknown')}"):
            b1, b2 = st.columns(2)
            if b1.button("Check parsing status", key=f"st_{doc['document_id']}"):
                try:
                    doc["status"] = client.eka_document_status(
                        doc["document_id"]
                    ).get("status", doc["status"])
                    st.rerun()
                except ApiClientError as exc:
                    st.error(friendly_error(exc))
            if b2.button("Fetch parsed fields", key=f"pr_{doc['document_id']}"):
                try:
                    doc["parsed"] = client.eka_document_parsed(
                        doc["document_id"]
                    )
                    st.rerun()
                except ApiClientError as exc:
                    st.error(friendly_error(exc))
            parsed = doc.get("parsed")
            if parsed:
                st.markdown("**Checklist suggestions (unconfirmed)**")
                st.json(parsed.get("checklist_suggestions") or {})
                uncertain = parsed.get("uncertain_or_missing") or []
                if uncertain:
                    st.warning(
                        "Missing or uncertain: " + ", ".join(uncertain)
                    )
                st.caption(TOOLTIPS["eka_context"])

    st.divider()

    # --- Combined context -------------------------------------------------
    st.markdown("#### Combined clinical context")
    if st.button("Build clinical context for current study"):
        if not study_id:
            st.warning("Load or run a study first (see sidebar).")
        else:
            try:
                ctx = client.eka_context(
                    {
                        "study_id": study_id,
                        "assessment_id": (eka or {}).get("assessment_id"),
                        "document_ids": [
                            d["document_id"]
                            for d in st.session_state.eka_docs
                            if d.get("document_id")
                        ],
                        "clinician_confirmed": False,
                    }
                )
                st.session_state.last_context = ctx
            except ApiClientError as exc:
                st.error(friendly_error(exc))

    context = st.session_state.last_context
    if context:
        st.caption(
            f"Imaging status: {context.get('imaging_status') or 'n/a'}"
        )
        st.markdown("**Checklist suggestions**")
        st.json(context.get("checklist_suggestions") or {})
        st.markdown("**Engine inputs (unconfirmed)**")
        st.json(context.get("eka_engine_inputs") or {})
        uncertain = context.get("uncertain_or_missing") or []
        if uncertain:
            st.warning("Missing or uncertain: " + ", ".join(uncertain))
        st.caption(TOOLTIPS["eka_context"])
        st.caption(context.get("disclaimer") or EKA_DISCLAIMER)
    else:
        st.caption("No context built yet for this session.")


# ---------------------------------------------------------------------------
# Pathway Simulator tab (placeholder only)
# ---------------------------------------------------------------------------


def render_simulator_tab():
    st.markdown("### Pathway simulator")
    st.info(
        "Placeholder only — the pathway simulator is not implemented in "
        "this MVP. No simulated clinical output is produced."
    )


# ---------------------------------------------------------------------------
# Audit tab
# ---------------------------------------------------------------------------


def render_audit_tab():
    st.markdown("### Audit")
    study_id = st.session_state.study_id
    if not study_id:
        st.info("Empty state — no study loaded.")
        return

    if st.button("Load audit events"):
        try:
            st.session_state.audit_events = client.audit_list(study_id)
        except ApiClientError as exc:
            st.error(friendly_error(exc))

    events = st.session_state.audit_events
    if events:
        st.dataframe(events)
    elif events == []:
        st.caption("No audit events recorded for this study.")
    else:
        st.caption("Click 'Load audit events' to fetch the audit trail.")

    export = build_audit_export(
        study_id,
        st.session_state.result,
        st.session_state.eligibility,
        events,
    )
    st.download_button(
        "Download audit JSON",
        data=audit_export_json(export),
        file_name=f"audit_{study_id}.json",
        mime="application/json",
    )
    st.caption(DISCLAIMER)


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

render_header()
render_sidebar()

(
    tab_triage,
    tab_ct,
    tab_eligibility,
    tab_eka,
    tab_simulator,
    tab_audit,
) = st.tabs(
    [
        "Triage Dashboard",
        "CT Review",
        "Eligibility Assistant",
        "Eka Context",
        "Pathway Simulator",
        "Audit",
    ]
)

with tab_triage:
    render_triage_tab()
with tab_ct:
    render_ct_tab()
with tab_eligibility:
    render_eligibility_tab()
with tab_eka:
    render_eka_tab()
with tab_simulator:
    render_simulator_tab()
with tab_audit:
    render_audit_tab()

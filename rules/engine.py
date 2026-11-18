# Thrombolysis Review Engine - Rule Engine
# Evaluates declarative rules from thrombolysis_rules.yaml.
# Never outputs ADMINISTER_THROMBOLYSIS; always conservative.
# Optional Eka clinical-context fields add warnings only — they never
# exclude a patient and never change the final status on their own.

import yaml
from datetime import datetime


ALLOWED_FINAL_STATUSES = [
    "HEMORRHAGE_SUSPECTED",
    "NO_HEMORRHAGE_DETECTED",
    "OUTSIDE_STANDARD_WINDOW",
    "INSUFFICIENT_INFORMATION",
    "POTENTIALLY_ELIGIBLE_PENDING_REVIEW",
    "URGENT_RADIOLOGY_REVIEW",
]

STATUS_MAP = {
    "HEMORRHAGE_SUSPECTED": "HEMORRHAGE_SUSPECTED",
    "NO_ACUTE_STROKE_EVIDENCE": "NO_HEMORRHAGE_DETECTED",
    "ISCHEMIC_PATTERN_POSSIBLE": "URGENT_RADIOLOGY_REVIEW",
    "OUTSIDE_STANDARD_WINDOW": "OUTSIDE_STANDARD_WINDOW",
    "INSUFFICIENT_INFORMATION": "INSUFFICIENT_INFORMATION",
    "POTENTIALLY_ELIGIBLE_PENDING_REVIEW": "POTENTIALLY_ELIGIBLE_PENDING_REVIEW",
    "URGENT_RADIOLOGY_REVIEW": "URGENT_RADIOLOGY_REVIEW",
}

REVIEW_REQUIRED_STATUSES = {
    "URGENT_RADIOLOGY_REVIEW",
    "POTENTIALLY_ELIGIBLE_PENDING_REVIEW",
}

# Optional Eka clinical-context inputs (all unconfirmed until a clinician verifies)
EKA_CONTEXT_DEFAULTS = {
    "eka_assessment_completed": False,
    "eka_extracted_medications": None,
    "eka_extracted_anticoagulant_use": None,
    "eka_extracted_lab_values": None,
    "eka_extracted_prior_stroke_history": None,
    "eka_extracted_prior_hemorrhage_history": None,
    "eka_context_confidence": None,
}


def load_rules(rules_path="rules/thrombolysis_rules.yaml"):
    """Load rules from YAML file."""
    with open(rules_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("rules", [])


def compute_elapsed_hours(lkw_time, current_time):
    """Compute elapsed hours from LKW to current time, or None."""
    if not lkw_time or not current_time:
        return None
    try:
        lkw = datetime.fromisoformat(lkw_time)
        now = datetime.fromisoformat(current_time)
        return round((now - lkw).total_seconds() / 3600, 1)
    except Exception:
        return None


def check_standard_window(elapsed_hours):
    """True if elapsed time is within the 4.5-hour standard window."""
    return elapsed_hours is not None and elapsed_hours <= 4.5


def detect_missing_fields(lkw_time, image_quality):
    """List of missing/concerning field names."""
    missing = []
    if not lkw_time or lkw_time == "":
        missing.append("lkw_time")
    if image_quality in ("poor", "low_confidence"):
        missing.append("image_quality")
    return missing


def _resolve_output_value(value, elapsed_hours, within_standard_window, inputs):
    """Resolve a YAML {{placeholder}} output value to a native value."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("{{") and stripped.endswith("}}"):
            ref = stripped[2:-2].strip()
            if ref == "elapsed_hours":
                return elapsed_hours
            if ref == "within_standard_window":
                return within_standard_window
            if ref == "imaging_status":
                return inputs.get("imaging_status", "")
            if ref == "final_status":
                return None
            return inputs.get(ref)
    return value


def _evaluate_condition(condition, context):
    """Safely evaluate a rule condition against the input context."""
    try:
        return bool(eval(condition, {"__builtins__": {}}, context))
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Eka clinical-context evaluation (warnings only — never exclusions)
# ---------------------------------------------------------------------------


def _is_uncertain(value):
    if value == "uncertain":
        return True
    if isinstance(value, dict):
        return value.get("status") == "uncertain" or value.get("value") == "uncertain"
    return False


def _is_positive(value):
    if value == "positive":
        return True
    if isinstance(value, dict):
        return value.get("value") == "positive"
    return False


def _is_true(value):
    if value is True:
        return True
    if isinstance(value, dict):
        return value.get("value") is True
    return False


def eka_context_present(inputs):
    """True when any Eka clinical-context field has been supplied."""
    if inputs.get("eka_assessment_completed"):
        return True
    for field, default in EKA_CONTEXT_DEFAULTS.items():
        if inputs.get(field, default) is not None and inputs.get(field, default) is not False:
            return True
    return False


def evaluate_eka_context(inputs):
    """Return warning dicts for Eka-derived clinical context.

    Safety rules:
    - Warnings never change final_status.
    - Eka output is never a definitive contraindication.
    - No patient is excluded based only on Eka output.
    - Every finding requires clinician confirmation.
    """
    warnings = []
    if not eka_context_present(inputs):
        return warnings

    def add(field, value, trace, message):
        warnings.append({
            "field": field,
            "value": value,
            "trace": trace,
            "message": message,
            "requires_clinician_confirmation": True,
        })

    anticoag = inputs.get("eka_extracted_anticoagulant_use")
    if _is_positive(anticoag):
        add(
            "eka_extracted_anticoagulant_use", anticoag,
            "EKA_ANTICOAGULANT_USE_WARNING",
            "Eka clinical context reports anticoagulant use (unconfirmed). "
            "A clinician must verify this before it informs the thrombolysis "
            "review; Eka output alone never determines eligibility.",
        )
    elif _is_uncertain(anticoag):
        add(
            "eka_extracted_anticoagulant_use", anticoag,
            "EKA_UNCERTAIN_EXTRACTION_WARNING",
            "Eka anticoagulant-use extraction is uncertain (unconfirmed). "
            "Clinician verification required.",
        )

    prior_hem = inputs.get("eka_extracted_prior_hemorrhage_history")
    if _is_true(prior_hem):
        add(
            "eka_extracted_prior_hemorrhage_history", prior_hem,
            "EKA_PRIOR_HEMORRHAGE_HISTORY_WARNING",
            "Eka clinical context reports prior intracranial hemorrhage "
            "(unconfirmed). A clinician must verify this history; Eka output "
            "alone never determines eligibility.",
        )
    elif _is_uncertain(prior_hem):
        add(
            "eka_extracted_prior_hemorrhage_history", prior_hem,
            "EKA_UNCERTAIN_EXTRACTION_WARNING",
            "Eka prior-hemorrhage extraction is uncertain (unconfirmed). "
            "Clinician verification required.",
        )

    labs = inputs.get("eka_extracted_lab_values")
    if labs is None or labs in ("missing", "uncertain") or _is_uncertain(labs):
        labs_missing = True
    elif isinstance(labs, dict):
        labs_missing = not labs or any(
            v is None or _is_uncertain(v) for v in labs.values()
        )
    else:
        labs_missing = False
    if labs_missing:
        add(
            "eka_extracted_lab_values", labs,
            "EKA_MISSING_LAB_VALUES_WARNING",
            "Eka laboratory values are missing or uncertain (unconfirmed). "
            "Missing-data warning — obtain labs and clinician confirmation "
            "before relying on the checklist.",
        )

    meds = inputs.get("eka_extracted_medications")
    if _is_uncertain(meds):
        add(
            "eka_extracted_medications", meds,
            "EKA_UNCERTAIN_EXTRACTION_WARNING",
            "Eka medication extraction is uncertain (unconfirmed). "
            "Clinician verification required.",
        )

    confidence = inputs.get("eka_context_confidence")
    if confidence in ("low", "uncertain") or _is_uncertain(confidence):
        add(
            "eka_context_confidence", confidence,
            "EKA_UNCERTAIN_EXTRACTION_WARNING",
            "Eka clinical-context confidence is low or uncertain "
            "(unconfirmed). Clinician verification required.",
        )

    return warnings


def attach_eka_context(output, inputs):
    """Attach Eka context, warnings, and rule traces to an engine output.

    Eka findings only add clinician-review requirements — final_status is
    never changed and no patient is excluded based only on Eka output.
    """
    eka_context = {
        field: inputs.get(field, default)
        for field, default in EKA_CONTEXT_DEFAULTS.items()
    }
    output["eka_context"] = eka_context

    warnings = evaluate_eka_context(inputs)
    output["eka_warnings"] = warnings
    output["eka_rule_traces"] = [
        f'{w["trace"]} (field={w["field"]}, value={w["value"]!r})'
        for w in warnings
    ]
    output["eka_requires_clinician_confirmation"] = (
        True if warnings else eka_context_present(inputs)
    )

    if warnings:
        output["requires_radiologist_review"] = True
        note = " ".join(w["message"] for w in warnings)
        base = output.get("rationale") or ""
        output["rationale"] = (base + " " + note).strip()

    return output


def evaluate_rules(rules, inputs):
    """Evaluate rules sequentially; the first matching rule wins."""
    elapsed_hours = compute_elapsed_hours(
        inputs.get("lkw_time"), inputs.get("current_time")
    )
    within_standard_window = check_standard_window(elapsed_hours)
    missing_fields = detect_missing_fields(
        inputs.get("lkw_time"), inputs.get("image_quality")
    )

    context = {
        **inputs,
        "elapsed_hours": elapsed_hours,
        "within_standard_window": within_standard_window,
        "missing_fields": missing_fields,
        **EKA_CONTEXT_DEFAULTS,
        **{k: v for k, v in inputs.items() if k.startswith("eka_")},
    }

    for rule in rules:
        condition = rule.get("condition", "")
        if not _evaluate_condition(condition, context):
            continue

        result_section = rule.get("result", {}) or {}
        merged_fields = dict(result_section.get("output_fields", {}) or {})
        for key, val in rule.items():
            if key.startswith("output_"):
                merged_fields[key[len("output_"):]] = val

        output = {
            "elapsed_hours": elapsed_hours,
            "within_standard_window": within_standard_window,
            "imaging_status": inputs.get("imaging_status", ""),
            "final_status": None,
            "rule_trace": rule.get("id", "UNKNOWN"),
            "requires_radiologist_review": False,
            "missing_fields": missing_fields,
            "rationale": rule.get(
                "rationale", result_section.get("rationale", "")
            ),
        }

        for field_name, field_val in merged_fields.items():
            resolved = _resolve_output_value(
                field_val, elapsed_hours, within_standard_window, inputs
            )
            if field_name in ("elapsed_hours", "within_standard_window"):
                if resolved is not None:
                    output[field_name] = resolved
            elif field_name == "final_status":
                if resolved is not None:
                    output["final_status"] = resolved
            elif field_name == "imaging_status":
                if resolved is not None:
                    output["imaging_status"] = resolved
            elif field_name == "missing_fields":
                if resolved is not None:
                    output["missing_fields"] = resolved
            else:
                output[field_name] = resolved

        if output["final_status"] is None:
            output["final_status"] = STATUS_MAP.get(
                output.get("imaging_status", ""), "URGENT_RADIOLOGY_REVIEW"
            )

        if output.get("missing_fields") is None:
            output["missing_fields"] = missing_fields

        output["requires_radiologist_review"] = (
            output.get("final_status") in REVIEW_REQUIRED_STATUSES
        )

        return attach_eka_context(output, inputs)

    # No rule matched - fall back to the conservative status mapping.
    final_status = STATUS_MAP.get(
        inputs.get("imaging_status", ""), "URGENT_RADIOLOGY_REVIEW"
    )

    output = {
        "elapsed_hours": elapsed_hours,
        "within_standard_window": within_standard_window,
        "imaging_status": inputs.get("imaging_status", ""),
        "final_status": final_status,
        "rule_trace": "RULE_DEFAULT_STATUS_MAPPING",
        "requires_radiologist_review": final_status in REVIEW_REQUIRED_STATUSES,
        "missing_fields": missing_fields,
        "rationale": "No specific rule matched; conservative default status mapping applied.",
    }
    return attach_eka_context(output, inputs)


def run_engine(inputs, rules_path="rules/thrombolysis_rules.yaml"):
    """Convenient wrapper: load rules and evaluate."""
    rules = load_rules(rules_path)
    return evaluate_rules(rules, inputs)


if __name__ == "__main__":
    example_inputs = {
        "imaging_status": "HEMORRHAGE_SUSPECTED",
        "lkw_time": None,
        "current_time": "2026-11-18T10:00:00",
        "image_quality": "good",
        "anticoagulant_status": "negative",
    }
    result = run_engine(example_inputs)
    for k, v in result.items():
        print(f"  {k}: {v}")

# Smoke test: the Streamlit dashboard renders without exceptions.
# Uses Streamlit's AppTest (headless). The API is unreachable in this test,
# so the dashboard must degrade gracefully to its empty/error states.

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from streamlit.testing.v1 import AppTest

DASHBOARD = os.path.join(ROOT, "app", "dashboard.py")


def _run():
    at = AppTest.from_file(DASHBOARD, default_timeout=60)
    at.run()
    return at


def test_dashboard_renders_without_exceptions():
    at = _run()
    assert not at.exception, at.exception
    assert len(at.tabs) == 6
    tab_labels = [t.label for t in at.tabs]
    assert tab_labels == [
        "Triage Dashboard",
        "CT Review",
        "Eligibility Assistant",
        "Eka Context",
        "Pathway Simulator",
        "Audit",
    ]


def test_dashboard_has_no_administration_instruction():
    at = _run()
    blob = "\n".join(
        str(el.value) for el in at.markdown
    ).lower()
    assert "administer thrombolysis" not in blob
    assert "ADMINISTER_THROMBOLYSIS" not in blob


def test_dashboard_shows_local_demo_and_disclaimer_context():
    at = _run()
    blob = "\n".join(str(el.value) for el in at.markdown)
    assert "LOCAL DEMO" in blob
    assert "Stroke CT Triage Assistant" in blob

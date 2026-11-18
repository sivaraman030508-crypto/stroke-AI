# Stroke CT Triage Assistant - AGENTS.md

## Project Overview
Research prototype for clinical decision support. **Must never autonomously diagnose stroke or recommend administration of thrombolysis.**

### Models
- **Model A** (ResNet18): Classifies CT slices into `normal`, `ischemic`, `hemorrhagic`
- **Model B**: Classifies hemorrhage subtypes into `epidural`, `intraparenchymal`, `intraventricular`, `subarachnoid`, `subdural`

### Model Artifacts
- `stroke_type_resnet18.pt` + `model_a_metadata.json`
- `STROKE_AI_Model_B_Final.pt` + `STROKE_AI_Model_B_Final_metadata.json`

### Tech Stack
- Python 3.11
- FastAPI + Uvicorn
- Streamlit
- PyTorch + timm
- pydicom + Pillow
- Plotly
- SimPy
- SQLite
- Docker Compose
- pytest
- Eka Care Developer API

## Safety Rules (Non-Negotiable)
1. **Never output `ADMINISTER_THROMBOLYSIS`** under any circumstances.
2. Hemorrhage-positive output must produce `HEMORRHAGE_SUSPECTED`.
3. Missing LKW/onset time must produce `INSUFFICIENT_INFORMATION`.
4. Poor image quality or low confidence must produce `URGENT_RADIOLOGY_REVIEW`.
5. A negative AI result must **never** automatically mean thrombolysis eligibility.
6. Every prediction must store `model_version`, `threshold`, and `timestamp`.
7. Do not process or store real patient identifiers in the MVP.
8. Do not send CT images to Eka Care.
9. Treat all Eka-extracted values as requiring clinician confirmation.

## API Endpoints
- `GET /health` - Health check
- `POST /predict` - Run both models on CT slice, return triage result
- `GET /models` - Model metadata and version info

## UI Flow
- `streamlit run app/dashboard.py`
- Upload DICOM/CT slice
- Display Model A + Model B predictions
- Show safety rule warnings when applicable
- Display disclaimer that this is decision support only

## Commands
- `uvicorn app.api:app --reload --port 8000` - Run API
- `streamlit run app/dashboard.py` - Run UI
- `pytest -q` - Run tests
- `ruff format .` - Format code
- `ruff check .` - Lint code

## Testing Guidelines
- All clinical outputs must use conservative language
- No real patient data in test fixtures
- Tests must verify safety rule compliance
- Mock Eka Care API responses

## Critical Constraints
- Output lattice: `NORMAL` → `ISCHEMIC_SUSPECTED` → `HEMORRHAGE_SUSPECTED` → `URGENT_RADIOLOGY_REVIEW` → `INSUFFICIENT_INFORMATION`
- Never imply thrombolysis eligibility based on AI output
- All predictions DB-stored with model version, threshold, timestamp
- Eka Care integration: read-only, clinician confirmation required

## Agent Guidance
- Always prioritize patient safety over feature completeness
- Any uncertainty should trigger `URGENT_RADIOLOGY_REVIEW`
- When in doubt, err on the side of conservative triage
- Review `app/api.py` and `app/dashboard.py` before making changes that affect clinical outputs
# Stroke Inference Service
# Orchestrates Model A (stroke type) and Model B (hemorrhage subtype) inference
# on CT DICOM slices with proper preprocessing and aggregation.

import numpy as np
import torch
from pathlib import Path
from datetime import datetime

_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


class StrokeInferenceService:
    """Service class for running stroke triage inference on CT DICOM slices.

    Coordinates Model A (normal/ischemic/hemorrhagic classification) and
    Model B (hemorrhage subtype classification) with proper preprocessing
    and study-level aggregation.
    """

    def __init__(
        self,
        model_a_service,
        model_b_service,
        device: str | None = None,
    ):
        """Initialize the inference service.

        Args:
            model_a_service: Initialized ModelAService instance.
            model_b_service: Initialized ModelBService instance.
            device: Optional device override ('cuda' or 'cpu').
        """
        self.model_a_service = model_a_service
        self.model_b_service = model_b_service

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

    def preprocess_dicom(self, dicom_path: str, target_size: int = 224) -> torch.Tensor:
        """Preprocess a single DICOM file.

        Args:
            dicom_path: Path to the DICOM file.
            target_size: Target image size for model input.

        Returns:
            Preprocessed tensor of shape (1, 1, target_size, target_size).

        Raises:
            FileNotFoundError: If DICOM file does not exist.
            ValueError: If DICOM file is invalid or not CT modality.
        """
        from imaging.preprocessing import preprocess_for_model_a

        tensor, _ = preprocess_for_model_a(
            filepath=dicom_path, target_size=target_size
        )
        return tensor.to(self.device)

    def run_model_a_on_slice(self, tensor: torch.Tensor) -> dict:
        """Run Model A inference on a single CT slice tensor.

        Args:
            tensor: Preprocessed CT slice tensor (1, 1, H, W).

        Returns:
            Model A prediction dict with predicted_class, confidence,
            class_probabilities, model_version, inference_timestamp, disclaimer.
        """
        with torch.no_grad():
            result = self.model_a_service.predict(tensor)
        return result

    def aggregate_study_level_model_a(
        self, slice_results: list[dict],
    ) -> dict:
        """Aggregate Model A results across all slices in a study.

        Computes max probability per class and determines study-level prediction.

        Args:
            slice_results: List of Model A prediction dicts, one per slice.

        Returns:
            Dict with study-level aggregation:
            - max_hemorrhagic_probability
            - max_ischemic_probability
            - max_normal_probability
            - predicted_class (study-level)
            - top_3_hemorrhagic_slices (indices with highest hemorrhagic prob)
        """
        if not slice_results:
            return {
                "max_hemorrhagic_probability": 0.0,
                "max_ischemic_probability": 0.0,
                "max_normal_probability": 0.0,
                "predicted_class": "normal",
                "top_3_hemorrhagic_slices": [],
            }

        # Extract probabilities from all slices
        hem_probabilities = [
            r["confidence"] if r["predicted_class"] == "hemorrhagic" else 0.0
            for r in slice_results
        ]
        ischemic_probabilities = [
            r["confidence"]
            if r["predicted_class"] == "ischemic"
            else 0.0
            for r in slice_results
        ]
        normal_probabilities = [
            r["confidence"]
            if r["predicted_class"] == "normal"
            else 0.0
            for r in slice_results
        ]

        # Get max probabilities
        max_hemorrhagic = max(hem_probabilities) if hem_probabilities else 0.0
        max_ischemic = max(ischemic_probabilities) if ischemic_probabilities else 0.0
        max_normal = max(normal_probabilities) if normal_probabilities else 0.0

        # Determine study-level predicted class
        max_prob = max(max_hemorrhagic, max_ischemic, max_normal)

        if max_prob == max_hemorrhagic:
            predicted_class = "hemorrhagic"
        elif max_prob == max_ischemic:
            predicted_class = "ischemic"
        else:
            predicted_class = "normal"

        # Find top 3 slices by hemorrhagic probability
        hemorrhagic_slice_indices = [
            i for i, r in enumerate(slice_results) if r["predicted_class"] == "hemorrhagic"
        ]
        # Sort by confidence and take top 3
        top_3_indices = sorted(
            hemorrhagic_slice_indices,
            key=lambda i: slice_results[i]["confidence"],
            reverse=True,
        )[:3]

        return {
            "max_hemorrhagic_probability": round(max_hemorrhagic, 4),
            "max_ischemic_probability": round(max_ischemic, 4),
            "max_normal_probability": round(max_normal, 4),
            "predicted_class": predicted_class,
            "top_3_hemorrhagic_slices": top_3_indices,
        }

    def run_model_b_on_slices(
        self, slice_indices: list[int], dcm_paths: list[str],
    ) -> dict:
        """Run Model B on selected slices and aggregate subtype probabilities.

        Args:
            slice_indices: Indices of slices to run Model B on.
            dcm_paths: List of DICOM file paths corresponding to all slices.

        Returns:
            Dict with aggregated Model B results:
            - subtype_probabilities: dict of subtype -> max probability
            - highest_probability_subtype: subtype name
            - model_version
            - inference_timestamp
            - disclaimer
        """
        if not slice_indices:
            return {
                "subtype_probabilities": None,
                "highest_probability_subtype": None,
                "model_version": self.model_b_service.model_version,
                "inference_timestamp": datetime.utcnow().isoformat() + "Z",
                "disclaimer": (
                    "Research prototype; not clinically validated; "
                    "not for autonomous diagnosis or treatment decisions."
                ),
            }

        # Run Model B on each selected slice and collect probabilities
        subtype_probs_accumulator = {}

        for idx in slice_indices:
            dcm_path = dcm_paths[idx] if idx < len(dcm_paths) else None
            if dcm_path is None:
                continue

            try:
                tensor = self.preprocess_dicom(dcm_path)
                with torch.no_grad():
                    b_result = self.model_b_service.predict(tensor)

                # Aggregate using max probability per subtype
                for subtype, prob in b_result["subtype_probabilities"].items():
                    if subtype not in subtype_probs_accumulator:
                        subtype_probs_accumulator[subtype] = prob
                    else:
                        subtype_probs_accumulator[subtype] = max(
                            subtype_probs_accumulator[subtype], prob
                        )
            except Exception as e:
                # Skip slices that fail preprocessing rather than failing entire study
                continue

        # If no slices produced probabilities, return null
        if not subtype_probs_accumulator:
            return {
                "subtype_probabilities": None,
                "highest_probability_subtype": None,
                "model_version": self.model_b_service.model_version,
                "inference_timestamp": datetime.utcnow().isoformat() + "Z",
                "disclaimer": (
                    "Research prototype; not clinically validated; "
                    "not for autonomous diagnosis or treatment decisions."
                ),
            }

        # Find highest probability subtype
        highest_subtype = max(
            subtype_probs_accumulator, key=subtype_probs_accumulator.get
        )
        highest_prob = subtype_probs_accumulator[highest_subtype]

        return {
            "subtype_probabilities": {
                k: round(v, 4) for k, v in subtype_probs_accumulator.items()
            },
            "highest_probability_subtype": highest_subtype,
            "model_version": self.model_b_service.model_version,
            "inference_timestamp": datetime.utcnow().isoformat() + "Z",
            "disclaimer": (
                "Research prototype; not clinically validated; "
                "not for autonomous diagnosis or treatment decisions."
            ),
        }

    def infer_study(self, dcm_paths: list[str]) -> dict:
        """Run full study inference: Model A on all slices, then Model B if needed.

        Args:
            dcm_paths: List of DICOM file paths for the study.

        Returns:
            Dict with complete study inference results:
            - study_id: generated from paths
            - model_a_result: aggregated Model A results
            - model_b_result: Model B results (or null if not hemorrhagic)
            - top_slices: top hemorrhagic slice indices
            - imaging_status: triage status string
            - model_a_version: Model A version string
            - model_b_version: Model B version string
            - inference_timestamp: UTC timestamp
            - disclaimer: research prototype disclaimer
        """
        # Generate study ID from paths
        study_id = (
            "_".join(Path(p).stem for p in dcm_paths[:5])
            if dcm_paths
            else "unknown_study"
        )

        # Preprocess all slices and run Model A
        slice_results = []
        for i, dcm_path in enumerate(dcm_paths):
            try:
                tensor = self.preprocess_dicom(dcm_path)
                a_result = self.run_model_a_on_slice(tensor)
                a_result["slice_index"] = i
                slice_results.append(a_result)
            except Exception as e:
                # Skip slices that fail preprocessing
                # Add a minimal result indicating insufficient information
                slice_results.append(
                    {
                        "predicted_class": "normal",
                        "confidence": 0.0,
                        "class_probabilities": {
                            "normal": 1.0,
                            "ischemic": 0.0,
                            "hemorrhagic": 0.0,
                        },
                        "model_version": self.model_a_service.model_version,
                        "inference_timestamp": datetime.utcnow().isoformat() + "Z",
                        "disclaimer": (
                            "Research prototype; not clinically validated; "
                            "not for autonomous diagnosis or treatment decisions."
                        ),
                        "slice_index": i,
                        "error": str(e),
                    }
                )

        # Aggregate Model A results at study level
        model_a_aggregated = self.aggregate_study_level_model_a(slice_results)

        # Determine imaging status and whether to run Model B
        predicted_class = model_a_aggregated["predicted_class"]
        max_hem_prob = model_a_aggregated["max_hemorrhagic_probability"]

        # Mapping of study-level predictions to imaging status
        status_mapping = {
            "hemorrhagic": "HEMORRHAGE_SUSPECTED",
            "ischemic": "ISCHEMIC_PATTERN_POSSIBLE",
            "normal": "NO_ACUTE_STROKE_EVIDENCE",
        }

        imaging_status = status_mapping.get(predicted_class, "UNKNOWN")

        # Model B only runs if study-level prediction is hemorrhagic
        if predicted_class == "hemorrhagic" and max_hem_prob > 0:
            # Get top 3 hemorrhagic slice indices from aggregation
            top_3_indices = model_a_aggregated.get("top_3_hemorrhagic_slices", [])

            # Run Model B on top hemorrhagic slices
            model_b_result = self.run_model_b_on_slices(
                top_3_indices, dcm_paths
            )
        else:
            # Do not run Model B if not hemorrhagic
            model_b_result = {
                "subtype_probabilities": None,
                "highest_probability_subtype": None,
                "model_version": self.model_b_service.model_version,
                "inference_timestamp": datetime.utcnow().isoformat() + "Z",
                "disclaimer": (
                    "Research prototype; not clinically validated; "
                    "not for autonomous diagnosis or treatment decisions."
                ),
            }

        return {
            "study_id": study_id,
            "model_a_result": model_a_aggregated,
            "model_b_result": model_b_result,
            "top_slices": model_a_aggregated.get("top_3_hemorrhagic_slices", []),
            "imaging_status": imaging_status,
            "model_a_version": self.model_a_service.model_version,
            "model_b_version": self.model_b_service.model_version,
            "inference_timestamp": datetime.utcnow().isoformat() + "Z",
            "disclaimer": (
                "Research prototype; not clinically validated; "
                "not for autonomous diagnosis or treatment decisions."
            ),
        }


# --- Factory function ---

def create_stroke_inference_service(
    model_a_checkpoint: str = str(_ARTIFACTS_DIR / "stroke_type_resnet18.pt"),
    model_a_metadata: str = str(_ARTIFACTS_DIR / "model_a_metadata.json"),
    model_b_checkpoint: str = str(_ARTIFACTS_DIR / "STROKE_AI_Model_B_Final.pt"),
    model_b_metadata: str = str(_ARTIFACTS_DIR / "STROKE_AI_Model_B_Final_metadata.json"),
    device: str | None = None,
) -> StrokeInferenceService:
    """Factory function to create a StrokeInferenceService instance.

    Args:
        model_a_checkpoint: Path to Model A checkpoint file.
        model_a_metadata: Path to Model A metadata JSON file.
        model_b_checkpoint: Path to Model B checkpoint file.
        model_b_metadata: Path to Model B metadata JSON file.
        device: Optional device override ('cuda' or 'cpu').

    Returns:
        Configured StrokeInferenceService instance.

    Raises:
        FileNotFoundError: If checkpoint or metadata paths do not exist.
    """
    from imaging.model_a import ModelAService
    from imaging.model_b import ModelBService

    model_a = ModelAService(
        checkpoint_path=model_a_checkpoint,
        metadata_path=model_a_metadata,
        device=device,
    )
    model_b = ModelBService(
        checkpoint_path=model_b_checkpoint,
        metadata_path=model_b_metadata,
        device=device,
    )

    return StrokeInferenceService(
        model_a_service=model_a,
        model_b_service=model_b,
        device=device,
    )
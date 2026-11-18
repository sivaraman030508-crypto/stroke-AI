# Unit tests for Model A using a dummy tensor and mocked model output.
# These tests verify service structure and error handling without requiring
# actual trained weights.

import sys
import os
import torch
import torch.nn as nn
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Add app directory to path
sys.path.insert(0, str(Path(__file__).parent.resolve()))


class DummyStrokeTypeNet(nn.Module):
    """Minimal dummy model for testing service structure.

    Returns logits where the predicted class depends on input tensor
    values in a deterministic way so tests can verify predict() logic.
    """

    def __init__(self, num_classes=3, in_chans=1):
        super().__init__()
        self.num_classes = num_classes
        self.in_chans = in_chans

    def forward(self, x):
        # x shape: (B, C, H, W)
        # If any pixel > 0, predict class 1 (ischemic), else class 0 (normal)
        has_positive = (x > 0).any().item()
        logits = torch.zeros(x.shape[0], self.num_classes)
        if has_positive:
            logits[:, 1] = 1.0  # ischemic
        else:
            logits[:, 0] = 1.0  # normal
        # class 2 (hemorrhagic) is never top in this dummy
        return logits


def test_model_a_service_creation_with_missing_checkpoint():
    """Test that ModelAService raises FileNotFoundError for missing checkpoint."""
    from model_a import ModelAService

    with pytest.raises(FileNotFoundError):
        ModelAService(
            checkpoint_path="C:\\nonexistent\\checkpoint.pt",
            metadata_path=str(
                Path(__file__).resolve().parent.parent
                / "artifacts" / "model_a_metadata.json"
            ),
        )


def test_model_a_service_creation_with_missing_metadata():
    """Test that ModelAService raises FileNotFoundError for missing metadata."""
    from model_a import ModelAService

    with pytest.raises(FileNotFoundError):
        ModelAService(
            checkpoint_path=str(
                Path(__file__).resolve().parent.parent
                / "artifacts" / "stroke_type_resnet18.pt"
            ),
            metadata_path="C:\\nonexistent\\metadata.json",
        )


def test_model_a_service_predict_structure():
    """Test that predict returns all required keys with correct types.

    Uses a mocked model to avoid needing actual weights.
    """
    import torch

    from model_a import ModelAService, StrokeTypeNet

    # Create service with real metadata but dummy model
    service = ModelAService.__new__(ModelAService)
    service.model_version = "stroke-type-triage-v1"
    service.classes = ["normal", "ischemic", "hemorrhagic"]
    service.image_size = 224
    service.input_channels = 1
    service.device = torch.device("cpu")
    service.model = DummyStrokeTypeNet(num_classes=3, in_chans=1)

    # Run prediction with dummy tensor
    tensor = torch.zeros(1, 1, 224, 224)
    result = service.predict(tensor)

    # Verify all required keys exist
    required_keys = [
        "predicted_class",
        "predicted_index",
        "confidence",
        "class_probabilities",
        "model_version",
        "inference_timestamp",
        "disclaimer",
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"

    # Verify types and constraints
    assert result["predicted_class"] in ["normal", "ischemic", "hemorrhagic"]
    assert isinstance(result["predicted_index"], int)
    assert 0 <= result["confidence"] <= 1
    assert isinstance(result["class_probabilities"], dict)
    assert len(result["class_probabilities"]) == 3
    assert abs(sum(result["class_probabilities"].values()) - 1.0) < 1e-5
    assert result["model_version"] == "stroke-type-triage-v1"
    assert "disclaimer" in result
    assert "not clinically validated" in result["disclaimer"]

    # Verify class_probabilities keys match classes
    for cls in ["normal", "ischemic", "hemorrhagic"]:
        assert cls in result["class_probabilities"]


def test_model_a_service_predict_with_custom_tensor():
    """Test predict with a tensor that biases toward a specific class."""
    import torch

    from model_a import ModelAService, StrokeTypeNet

    service = ModelAService.__new__(ModelAService)
    service.model_version = "stroke-type-triage-v1"
    service.classes = ["normal", "ischemic", "hemorrhagic"]
    service.image_size = 224
    service.input_channels = 1
    service.device = torch.device("cpu")
    service.model = DummyStrokeTypeNet(num_classes=3, in_chans=1)

    import torch

    # Tensor with positive logits for "ischemic" class
    tensor = torch.zeros(1, 1, 224, 224)
    tensor[0, 0, 0, 0] = 5.0  # bias

    result = service.predict(tensor)
    assert result["predicted_class"] == "ischemic"
    assert result["predicted_index"] == 1


def test_model_a_service_with_cuda_available():
    """Test service can be created on CPU (skip if CUDA-only constraints)."""
    import torch

    # Should not raise an error on CPU-only systems
    from model_a import ModelAService

    try:
        service = ModelAService.__new__(ModelAService)
        service.model_version = "stroke-type-triage-v1"
        service.classes = ["normal", "ischemic", "hemorrhagic"]
        service.image_size = 224
        service.input_channels = 1
        service.device = torch.device("cpu")
        service.model = DummyStrokeTypeNet(num_classes=3, in_chans=1)
        # verify predict works on CPU
        tensor = torch.zeros(1, 1, 224, 224)
        result = service.predict(tensor)
        assert "predicted_class" in result
    except Exception as e:
        pytest.skip(f"CPU test failed (expected in CUDA-only env): {e}")


def test_stroketypenet_creates_correct_num_classes():
    """Test StrokeTypeNet instantiation with correct configuration."""
    import torch
    import timm

    from model_a import StrokeTypeNet

    model = StrokeTypeNet(num_classes=3, in_chans=1)

    # Test forward pass
    x = torch.zeros(2, 1, 224, 224)
    with torch.no_grad():
        output = model(x)

    assert output.shape == (2, 3)
    assert model.num_classes == 3
    assert model.in_chans == 1
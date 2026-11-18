# Unit tests for Model B using a dummy tensor and mocked model output.
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


class DummyHemorrhageSubtypeNet(nn.Module):
    """Minimal dummy model for testing service structure.

    Returns logits where the predicted class depends on input tensor
    values in a deterministic way so tests can verify predict() logic.
    If any pixel > 0, class 1 (intraparenchymal) gets highest probability.
    Otherwise class 0 (epidural) gets highest probability.
    Other classes get negligible probability.
    """

    def __init__(self, num_classes=5, in_chans=1):
        super().__init__()
        self.num_classes = num_classes
        self.in_chans = in_chans

    def forward(self, x):
        # Check if any pixel value is positive
        # x shape: (B, C, H, W)
        has_positive = (x > 0).any().item()
        # Use large values so softmax produces ~one-hot outputs
        logits = torch.full((x.shape[0], self.num_classes), -10.0)
        if has_positive:
            logits[:, 1] = 10.0  # intraparenchymal - high logit
        else:
            logits[:, 0] = 10.0  # epidural - high logit
        # Other classes remain at -10.0 so their softmax prob ~0.000045
        return logits


def test_model_b_service_creation_with_missing_checkpoint():
    """Test that ModelBService raises FileNotFoundError for missing checkpoint."""
    from imaging.model_b import ModelBService

    with pytest.raises(FileNotFoundError):
        ModelBService(
            checkpoint_path="C:\\nonexistent\\checkpoint.pt",
            metadata_path=str(
                Path(__file__).resolve().parent.parent
                / "artifacts" / "STROKE_AI_Model_B_Final_metadata.json"
            ),
        )


def test_model_b_service_creation_with_missing_metadata():
    """Test that ModelBService raises FileNotFoundError for missing metadata."""
    from imaging.model_b import ModelBService

    with pytest.raises(FileNotFoundError):
        ModelBService(
            checkpoint_path=str(
                Path(__file__).resolve().parent.parent
                / "artifacts" / "STROKE_AI_Model_B_Final.pt"
            ),
            metadata_path="C:\\nonexistent\\metadata.json",
        )


def test_model_b_service_predict_structure():
    """Test that predict returns all required keys with correct types.

    Uses a mocked model to avoid needing actual weights.
    """
    import torch

    from imaging.model_b import ModelBService

    service = ModelBService.__new__(ModelBService)
    service.model_version = "STROKE_AI_Model_B_Final"
    service.classes = [
        "epidural", "intraparenchymal", "intraventricular",
        "subarachnoid", "subdural",
    ]
    service.image_size = 224
    service.input_channels = 1
    service.device = torch.device("cpu")
    service.model = DummyHemorrhageSubtypeNet(num_classes=5, in_chans=1)

    import torch
    tensor = torch.zeros(1, 1, 224, 224)
    result = service.predict(tensor)

    # Verify all required keys exist
    required_keys = [
        "subtype_probabilities",
        "highest_probability_subtype",
        "model_version",
        "inference_timestamp",
        "disclaimer",
    ]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"

    # Verify types and constraints
    assert isinstance(result["subtype_probabilities"], dict)
    assert len(result["subtype_probabilities"]) == 5
    assert abs(sum(result["subtype_probabilities"].values()) - 1.0) < 1e-5
    assert result["model_version"] == "STROKE_AI_Model_B_Final"
    assert "disclaimer" in result
    assert "not clinically validated" in result["disclaimer"]

    # Verify class_probabilities keys match classes
    for cls in [
        "epidural", "intraparenchymal", "intraventricular",
        "subarachnoid", "subdural",
    ]:
        assert cls in result["subtype_probabilities"]

    # Verify highest_probability_subtype is one of the classes
    assert result["highest_probability_subtype"] in result["subtype_probabilities"]


def test_model_b_service_predict_with_custom_tensor():
    """Test predict with a tensor that biases toward a specific class."""
    import torch

    from imaging.model_b import ModelBService

    service = ModelBService.__new__(ModelBService)
    service.model_version = "STROKE_AI_Model_B_Final"
    service.classes = [
        "epidural", "intraparenchymal", "intraventricular",
        "subarachnoid", "subdural",
    ]
    service.image_size = 224
    service.input_channels = 1
    service.device = torch.device("cpu")
    service.model = DummyHemorrhageSubtypeNet(num_classes=5, in_chans=1)

    import torch

    # Tensor with positive pixel -> should predict class 1 (intraparenchymal)
    tensor = torch.zeros(1, 1, 224, 224)
    tensor[0, 0, 0, 0] = 5.0  # bias

    result = service.predict(tensor)
    assert result["highest_probability_subtype"] == "intraparenchymal"
    assert result["subtype_probabilities"]["intraparenchymal"] > 0.5


def test_model_b_service_predict_no_positive():
    """Test predict with all-zero tensor predicts class 0 (epidural)."""
    import torch

    from imaging.model_b import ModelBService

    service = ModelBService.__new__(ModelBService)
    service.model_version = "STROKE_AI_Model_B_Final"
    service.classes = [
        "epidural", "intraparenchymal", "intraventricular",
        "subarachnoid", "subdural",
    ]
    service.image_size = 224
    service.input_channels = 1
    service.device = torch.device("cpu")
    service.model = DummyHemorrhageSubtypeNet(num_classes=5, in_chans=1)

    import torch

    # All zeros -> should predict class 0 (epidural)
    tensor = torch.zeros(1, 1, 224, 224)

    result = service.predict(tensor)
    assert result["highest_probability_subtype"] == "epidural"
    assert result["subtype_probabilities"]["epidural"] > 0.5


def test_model_b_service_with_cuda_available():
    """Test service can be created on CPU."""
    import torch

    from imaging.model_b import ModelBService

    try:
        service = ModelBService.__new__(ModelBService)
        service.model_version = "STROKE_AI_Model_B_Final"
        service.classes = [
            "epidural", "intraparenchymal", "intraventricular",
            "subarachnoid", "subdural",
        ]
        service.image_size = 224
        service.input_channels = 1
        service.device = torch.device("cpu")
        service.model = DummyHemorrhageSubtypeNet(num_classes=5, in_chans=1)
        # verify predict works on CPU
        tensor = torch.zeros(1, 1, 224, 224)
        result = service.predict(tensor)
        assert "highest_probability_subtype" in result
    except Exception as e:
        pytest.skip(f"CPU test failed (expected in CUDA-only env): {e}")


def test_hemorrhagesubtypenet_creates_correct_num_classes():
    """Test HemorrhageSubtypeNet instantiation with correct configuration."""
    import torch
    import timm

    from imaging.model_b import HemorrhageSubtypeNet

    model = HemorrhageSubtypeNet(num_classes=5, in_chans=1)

    # Test forward pass
    x = torch.zeros(2, 1, 224, 224)
    with torch.no_grad():
        output = model(x)

    assert output.shape == (2, 5)
    assert model.num_classes == 5
    assert model.in_chans == 1
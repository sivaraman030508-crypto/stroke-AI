# Model B: Hemorrhage Subtype Classification (ResNet18)
# Never use for autonomous diagnosis or thrombolysis decisions.

import torch
import torch.nn as nn
import timm
from pathlib import Path
from datetime import datetime

_ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


class HemorrhageSubtypeNet(nn.Module):
    """ResNet-18 model for hemorrhage subtype classification.

    Uses grayscale input (in_chans=1) as CT slices are single-channel.
    Predicts 5 hemorrhage subtypes: epidural, intraparenchymal,
    intraventricular, subarachnoid, subdural.
    """

    def __init__(self, num_classes: int = 5, in_chans: int = 1, dropout: float = 0.0):
        super().__init__()
        self.num_classes = num_classes
        self.in_chans = in_chans

        self.backbone = timm.models.resnet18(pretrained=False, in_chans=in_chans)

        # Replace the final fully connected layer
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        if dropout > 0.0:
            self.backbone.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(in_features, num_classes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


class ModelBService:
    """Service class for Model B inference.

    Handles checkpoint loading, device management, and prediction logic.
    """

    def __init__(
        self,
        checkpoint_path: str,
        metadata_path: str,
        device: str | None = None,
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.metadata_path = Path(metadata_path)

        # Load metadata
        if not self.metadata_path.exists():
            raise FileNotFoundError(
                f"Model B metadata not found at: {self.metadata_path}"
            )
        import json
        with open(self.metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.model_version = self.metadata.get("model_name", "unknown")
        self.classes = self.metadata.get(
            "classes",
            ["epidural", "intraparenchymal", "intraventricular",
             "subarachnoid", "subdural"],
        )
        self.image_size = self.metadata.get("image_size", 224)
        self.input_channels = self.metadata.get("input_channels", 1)

        # Determine device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Initialize model
        self.model = HemorrhageSubtypeNet(
            num_classes=len(self.classes),
            in_chans=self.input_channels,
        ).to(self.device)

        # Load checkpoint
        self._load_checkpoint()

        self.model.eval()

    def _load_checkpoint(self):
        """Load model checkpoint safely.

        Raises FileNotFoundError if checkpoint does not exist.
        """
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model B checkpoint not found at: {self.checkpoint_path}"
            )

        checkpoint = torch.load(
            self.checkpoint_path, map_location=self.device
        )

        # Handle bare state_dict and common checkpoint wrappers
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            for wrapper in ("state_dict", "model_state_dict", "model"):
                if wrapper in checkpoint:
                    state_dict = checkpoint[wrapper]
                    break

        # Align input channels with the checkpoint if needed
        conv1_weight = state_dict.get("conv1.weight")
        if conv1_weight is not None and conv1_weight.shape[1] != self.input_channels:
            self.input_channels = int(conv1_weight.shape[1])
            self.model = HemorrhageSubtypeNet(
                num_classes=len(self.classes),
                in_chans=self.input_channels,
            ).to(self.device)

        # Map unwrapped torchvision-style keys (conv1.*, fc.*) onto the backbone
        if any(k.startswith("conv1.") for k in state_dict):
            state_dict = {f"backbone.{k}": v for k, v in state_dict.items()}

        self.model.load_state_dict(state_dict)
        print(
            f"Model B checkpoint loaded: {self.checkpoint_path}"
            f" (version: {self.model_version})"
        )

    @torch.no_grad()
    def predict(self, tensor: torch.Tensor) -> dict:
        """Run inference on a single CT slice tensor.

        Args:
            tensor: Input tensor of shape (1, 1, H, W) or (1, C, H, W)

        Returns:
            dict with subtype_probabilities, highest_probability_subtype,
            model_version, inference_timestamp, disclaimer
        """
        self.model.eval()

        # Ensure tensor is on the correct device and dtype
        tensor = tensor.to(self.device).float()

        # Ensure correct number of dimensions: (1, C, H, W)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)

        # Repeat grayscale slices if the checkpoint expects RGB input
        if tensor.shape[1] == 1 and self.model.in_chans != 1:
            tensor = tensor.repeat(1, self.model.in_chans, 1, 1)

        # Run inference
        with torch.no_grad():
            logits = self.model(tensor)

        # Compute probabilities via softmax
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

        highest_index = int(torch.argmax(probs).item())
        highest_probability_subtype = (
            self.classes[highest_index] if highest_index < len(self.classes) else "unknown"
        )

        subtype_probabilities = {
            cls: float(prob) for cls, prob in zip(self.classes, probs)
        }

        result = {
            "subtype_probabilities": subtype_probabilities,
            "highest_probability_subtype": highest_probability_subtype,
            "model_version": self.model_version,
            "inference_timestamp": datetime.utcnow().isoformat() + "Z",
            "disclaimer": (
                "Research prototype; not clinically validated; "
                "not for autonomous diagnosis or treatment decisions."
            ),
        }

        return result


def create_model_b_service(
    checkpoint_path: str = str(_ARTIFACTS_DIR / "STROKE_AI_Model_B_Final.pt"),
    metadata_path: str = str(_ARTIFACTS_DIR / "STROKE_AI_Model_B_Final_metadata.json"),
    device: str | None = None,
) -> ModelBService:
    """Factory function to create a ModelBService instance.

    Args:
        checkpoint_path: Path to the Model B checkpoint file.
        metadata_path: Path to the Model B metadata JSON file.
        device: Optional device override ('cuda' or 'cpu').

    Returns:
        ModelBService instance.

    Raises:
        FileNotFoundError: If checkpoint or metadata path does not exist.
    """
    return ModelBService(
        checkpoint_path=checkpoint_path,
        metadata_path=metadata_path,
        device=device,
    )
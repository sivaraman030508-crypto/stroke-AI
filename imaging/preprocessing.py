# Preprocessing module for Stroke CT Triage Assistant
# Reads DICOM files, converts to Hounsfield units, applies windowing,
# and prepares tensors for Model A and Model B inference.

import numpy as np
import pydicom
import torch
from pathlib import Path


# Window settings for brain CT
BRAIN_WINDOW_WIDTH = 80
BRAIN_WINDOW_LEVEL = 40


class _SyntheticDicomTag:
    """Helper to simulate a DICOM tag."""

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return str(self.value)


class _SyntheticDicom:
    """Minimal synthetic DICOM-like object for testing preprocessing.

    Simulates a CT DICOM dataset with required tags for HU conversion
    and windowing, without requiring pydicom or actual DICOM files.
    """

    def __init__(
        self,
        pixel_array: np.ndarray,
        modality: str = "CT",
        rescale_slope: float = 0.04,
        rescale_intercept: float = -1000,
        rows: int = 64,
        cols: int = 64,
        photometric_interpretation: str = "MONOCHROME2",
    ):
        # Do NOT set self.PixelData here - it conflicts with the @property below
        # Instead, store pixel array internally and expose via PixelArray property
        self._pixel_array = pixel_array.astype(np.float32)
        self.Rows = rows
        self.Cols = cols
        self.Modality = modality
        self.PhotometricInterpretation = photometric_interpretation
        self.RescaleSlope = _SyntheticDicomTag(rescale_slope)
        self.RescaleIntercept = _SyntheticDicomTag(rescale_intercept)

    @property
    def PixelArray(self):
        return self._pixel_array

    @PixelArray.setter
    def PixelArray(self, value):
        self._pixel_array = value

    @property
    def pixel_array(self):
        # Lowercase version for compatibility with preprocessing code
        return self._pixel_array

    @property
    def PixelData(self):
        # Return a mock bytes-like object for pydicom compatibility
        # In real usage, this would be actual DICOM PixelData
        return b"\x00" * (self.Rows * self.Cols * 2)


def read_dicom_file(filepath: str) -> pydicom.Dataset:
    """Read a DICOM file and return the dataset.

    Args:
        filepath: Path to the DICOM file.

    Returns:
        pydicom.Dataset containing the DICOM metadata.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not a valid DICOM file or is not a CT modality.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"DICOM file not found: {filepath}")

    try:
        ds = pydicom.dcmread(filepath)
    except Exception as e:
        raise ValueError(f"Failed to read DICOM file: {filepath}") from e

    # Check modality
    modality = getattr(ds, "Modality", None)
    if modality != "CT":
        raise ValueError(
            f"Non-CT modality detected: {modality}. "
            "Only CT images are supported for stroke triage."
        )

    return ds


def apply_rescale_slope_intercept(ds: pydicom.Dataset, pixel_array: np.ndarray) -> np.ndarray:
    """Convert pixel data to Hounsfield units (HU).

    Uses RescaleSlope and RescaleIntercept if available in DICOM metadata.
    Falls back to linear conversion if tags are missing.

    Args:
        ds: pydicom Dataset object.
        pixel_array: Raw pixel array from DICOM.

    Returns:
        Hounsfield unit-scaled numpy array.
    """
    slope = 1.0
    intercept = 0.0

    if hasattr(ds, "RescaleSlope"):
        try:
            slope = float(ds.RescaleSlope)
        except (ValueError, TypeError):
            slope = 1.0

    if hasattr(ds, "RescaleIntercept"):
        try:
            intercept = float(ds.RescaleIntercept)
        except (ValueError, TypeError):
            intercept = 0.0

    hu_image = pixel_array * slope + intercept
    return hu_image


def apply_brain_window(
    hu_image: np.ndarray, window_width: int = BRAIN_WINDOW_WIDTH, window_level: int = BRAIN_WINDOW_LEVEL
) -> np.ndarray:
    """Apply brain windowing to Hounsfield unit image.

    Maps HU values to 0-255 grayscale range for brain window:
    - Lower bound = window_level - window_width/2
    - Upper bound = window_level + window_width/2

    Args:
        hu_image: Hounsfield unit numpy array.
        window_width: Width of the window (default 80).
        window_level: Level of the window (default 40).

    Returns:
        Windowed numpy array with values in 0-255 range.
    """
    lower_bound = window_level - window_width // 2
    upper_bound = window_level + window_width // 2

    # Clip to window range
    windowed = np.clip(hu_image, lower_bound, upper_bound)

    # Normalize to 0-255
    windowed = windowed - lower_bound
    if upper_bound > lower_bound:
        windowed = windowed / (upper_bound - lower_bound) * 255.0

    # Convert to uint8
    windowed = np.uint8(windowed)
    return windowed


def preprocess_dicom(
    filepath: str,
    target_size: int = 224,
    dataset: pydicom.Dataset | None = None,
) -> tuple[torch.Tensor, np.ndarray]:
    """Preprocess a DICOM CT slice for model inference.

    Steps:
    1. Read DICOM file (or use provided dataset)
    2. Convert pixel data to Hounsfield units
    3. Apply brain windowing
    4. Resize to target_size x target_size
    5. Convert to grayscale PyTorch tensor with shape (1, 1, H, W)

    Args:
        filepath: Path to the DICOM file.
        target_size: Image size for model input (default 224 for both models).
        dataset: Optional pre-loaded pydicom Dataset. If provided,
            skips the file reading step. Useful for testing.

    Returns:
        tuple of (tensor, display_array):
        - tensor: torch.Tensor of shape (1, 1, target_size, target_size), dtype float32, normalized to [0, 1]
        - display_array: numpy.uint8 array of shape (target_size, target_size) for visualization

    Raises:
        ValueError: If DICOM file is unreadable or not a CT modality.
        FileNotFoundError: If the file does not exist and no dataset provided.
    """
    # Use provided dataset if available (for testing)
    if dataset is not None:
        ds = dataset
    else:
        # Read DICOM
        ds = read_dicom_file(filepath)

    # Extract pixel data
    if not hasattr(ds, "pixel_array") and not hasattr(ds, "PixelArray"):
        raise ValueError(f"DICOM file has no pixel data: {filepath}")

    # Get pixel array - try pixel_array first, then PixelArray
    if hasattr(ds, "pixel_array"):
        pixel_array = ds.pixel_array.astype(np.float32)
    elif hasattr(ds, "PixelArray"):
        pixel_array = ds.PixelArray.astype(np.float32)
    else:
        raise ValueError(f"DICOM file has no pixel data: {filepath}")

    # Convert to Hounsfield units
    hu_image = apply_rescale_slope_intercept(ds, pixel_array)

    # Apply brain windowing
    windowed = apply_brain_window(hu_image)

    # Resize to target size
    # Use PIL for resizing - no random augmentation
    from PIL import Image

    # Convert numpy array to PIL Image, resize, convert back
    pil_image = Image.fromarray(windowed.astype(np.uint8))
    pil_image = pil_image.resize((target_size, target_size), Image.BILINEAR)
    resized = np.array(pil_image)

    # Ensure uint8
    if resized.dtype != np.uint8:
        resized = np.uint8(resized)

    # Create PyTorch tensor with shape (1, 1, H, W)
    # Add channel dimensions: (H, W) -> (1, 1, H, W)
    tensor = torch.from_numpy(resized).float() / 255.0  # normalize to [0, 1]
    tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Return tensor and display array (same as resized for now)
    # display_array is the windowed resized array for UI display
    display_array = resized

    return tensor, display_array


def preprocess_for_model_a(
    filepath: str,
    target_size: int = 224,
    dataset: pydicom.Dataset | None = None,
) -> tuple[torch.Tensor, np.ndarray]:
    """Preprocess DICOM specifically for Model A (stroke type classification).

    Model A uses grayscale 1-channel input with ResNet18.

    Args:
        filepath: Path to the DICOM file.
        target_size: Image size (default 224).

    Returns:
        (tensor, display_array) tuple as per base preprocessing.
    """
    return preprocess_dicom(filepath, target_size=target_size, dataset=dataset)


def preprocess_for_model_b(
    filepath: str,
    target_size: int = 224,
    dataset: pydicom.Dataset | None = None,
) -> tuple[torch.Tensor, np.ndarray]:
    """Preprocess DICOM specifically for Model B (hemorrhage subtype classification).

    Model B uses grayscale 1-channel input with ResNet18.

    Args:
        filepath: Path to the DICOM file.
        target_size: Image size (default 224).

    Returns:
        (tensor, display_array) tuple as per base preprocessing.
    """
    return preprocess_dicom(filepath, target_size=target_size, dataset=dataset)


# --- Synthetic test data helpers ---

def test_preprocessing_with_synthetic_dicom():
    """Test preprocessing pipeline using a synthetic DICOM-like object.

    This test creates a minimal synthetic DICOM dataset and verifies
    that the preprocessing functions work correctly without requiring
    an actual DICOM file.

    Returns:
        None
    """
    from imaging.preprocessing import preprocess_for_model_a, _SyntheticDicom
    import torch

    # Create synthetic CT-like data HU values around brain tissue (~40 HU)
    rows, cols = 64, 64
    synthetic_pixel_data = np.full((rows, cols), 40, dtype=np.float32)

    # Create synthetic DICOM object
    synthetic_ds = _SyntheticDicom(
        pixel_array=synthetic_pixel_data,
        modality="CT",
        rescale_slope=0.04,
        rescale_intercept=-1000,
        rows=rows,
        cols=cols,
    )

    # Test preprocessing with synthetic dataset (pass dataset parameter)
    try:
        tensor, display_array = preprocess_for_model_a(
            filepath="synthetic", target_size=224, dataset=synthetic_ds
        )
        # Verify tensor shape
        assert tensor.shape == (1, 1, 224, 224), f"Expected shape (1,1,224,224), got {tensor.shape}"
        # Verify tensor values are normalized [0, 1]
        assert tensor.min().item() >= 0.0, "Tensor min should be >= 0"
        assert tensor.max().item() <= 1.0, "Tensor max should be <= 1"
        # Verify display array shape
        assert display_array.shape == (224, 224), f"Expected display shape (224,224), got {display_array.shape}"
        # Verify display array is uint8
        assert display_array.dtype == np.uint8, f"Expected uint8, got {display_array.dtype}"

        print("Synthetic DICOM preprocessing test PASSED")
    except Exception as e:
        print(f"Synthetic DICOM preprocessing test FAILED: {e}")
        raise


def test_preprocessing_error_handling():
    """Test that preprocessing raises clear errors for invalid inputs.

    Verifies:
    - FileNotFoundError for missing files
    - ValueError for non-CT modalities
    - ValueError for missing pixel data
    """
    from imaging.preprocessing import read_dicom_file, preprocess_for_model_a

    # Test 1: Missing file
    try:
        read_dicom_file(filepath="/nonexistent/path.dcm")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("Test 1 PASSED: FileNotFoundError for missing file")

    # Test 2: Non-existent file in preprocessing
    try:
        preprocess_for_model_a(filepath="/nonexistent/path.dcm", target_size=224)
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        print(f"Test 2 PASSED: FileNotFoundError in preprocessing: {e}")

    # Test 3: Synthetic dataset with missing pixel data
    from imaging.preprocessing import _SyntheticDicom
    synthetic_ds = _SyntheticDicom(
        pixel_array=np.full((64, 64), 40, dtype=np.float32),
        modality="CT",
    )
    # Remove pixel array to test error handling
    synthetic_ds._pixel_array = None
    try:
        preprocess_for_model_a(filepath="synthetic", target_size=224, dataset=synthetic_ds)
        assert False, "Should have raised ValueError for missing pixel data"
    except ValueError as e:
        print(f"Test 3 PASSED: ValueError for missing pixel data: {e}")

    print("Error handling tests complete")


if __name__ == "__main__":
    # Run tests when script is executed directly
    test_preprocessing_with_synthetic_dicom()
    print()
    test_preprocessing_error_handling()
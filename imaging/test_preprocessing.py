# Unit tests for preprocessing module using synthetic DICOM-like objects.
# These tests verify preprocessing pipeline without requiring actual DICOM files.

import sys
import os
import numpy as np
import pytest

# Add imaging directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from imaging.preprocessing import (
    preprocess_for_model_a,
    preprocess_for_model_b,
    _SyntheticDicom,
    read_dicom_file,
)


class TestPreprocessingSynthetic:
    """Tests for preprocessing using synthetic DICOM objects."""

    def setup_method(self):
        """Set up synthetic DICOM dataset for each test."""
        self.rows, self.cols = 64, 64
        # Brain tissue ~40 HU, air -1000 HU
        self.synthetic_pixel_data = np.full(
            (self.rows, self.cols), 40, dtype=np.float32
        )

        self.synthetic_ds = _SyntheticDicom(
            pixel_array=self.synthetic_pixel_data,
            modality="CT",
            rescale_slope=0.04,
            rescale_intercept=-1000,
            rows=self.rows,
            cols=self.cols,
        )

    def test_preprocess_returns_correct_tensor_shape(self):
        """Test that preprocess_for_model_a returns tensor of shape (1, 1, 224, 224)."""
        tensor, display_array = preprocess_for_model_a(
            filepath="synthetic", target_size=224, dataset=self.synthetic_ds
        )
        assert tensor.shape == (1, 1, 224, 224), (
            f"Expected shape (1,1,224,224), got {tensor.shape}"
        )
        assert display_array.shape == (224, 224), (
            f"Expected display shape (224,224), got {display_array.shape}"
        )
        assert display_array.dtype == np.uint8, (
            f"Expected uint8, got {display_array.dtype}"
        )

    def test_tensor_values_normalized(self):
        """Test that tensor values are normalized to [0, 1] range."""
        tensor, _ = preprocess_for_model_a(
            filepath="synthetic", target_size=224, dataset=self.synthetic_ds
        )
        assert tensor.min().item() >= 0.0, "Tensor min should be >= 0"
        assert tensor.max().item() <= 1.0, "Tensor max should be <= 1"

    def test_display_array_uint8(self):
        """Test that display array is uint8."""
        _, display_array = preprocess_for_model_a(
            filepath="synthetic", target_size=224, dataset=self.synthetic_ds
        )
        assert display_array.dtype == np.uint8

    def test_both_models_same_preprocessing(self):
        """Test that both Model A and Model B preprocessing produce identical output."""
        tensor_a, display_a = preprocess_for_model_a(
            filepath="synthetic", target_size=224, dataset=self.synthetic_ds
        )
        tensor_b, display_b = preprocess_for_model_b(
            filepath="synthetic", target_size=224, dataset=self.synthetic_ds
        )

        assert tensor_a.shape == tensor_b.shape, (
            f"Tensor shapes differ: {tensor_a.shape} vs {tensor_b.shape}"
        )
        assert display_a.shape == display_b.shape, (
            f"Display shapes differ: {display_a.shape} vs {display_b.shape}"
        )
        assert np.array_equal(display_a, display_b), "Display arrays should be identical"


class TestPreprocessingErrorHandling:
    """Tests for error handling in preprocessing."""

    def test_missing_file_raises_error(self):
        """Test that preprocessing raises FileNotFoundError for missing file."""
        from imaging.preprocessing import preprocess_for_model_a

        with pytest.raises(FileNotFoundError):
            preprocess_for_model_a(filepath="/nonexistent/path.dcm", target_size=224)

    def test_read_dicom_missing_file(self):
        """Test that read_dicom_file raises FileNotFoundError for missing file."""
        from imaging.preprocessing import read_dicom_file

        with pytest.raises(FileNotFoundError):
            read_dicom_file(filepath="/nonexistent/path.dcm")


class TestPreprocessingImportantEdgeCases:
    """Tests for important edge cases in preprocessing."""

    def setup_method(self):
        """Set up synthetic DICOM dataset for each test."""
        self.rows, self.cols = 64, 64
        self.synthetic_pixel_data = np.full(
            (self.rows, self.cols), 40, dtype=np.float32
        )
        self.synthetic_ds = _SyntheticDicom(
            pixel_array=self.synthetic_pixel_data,
            modality="CT",
            rescale_slope=0.04,
            rescale_intercept=-1000,
            rows=self.rows,
            cols=self.cols,
        )

    def test_synthetic_dicom_creation(self):
        """Test that _SyntheticDicom can be created and used with explicit rows/cols."""
        from imaging.preprocessing import _SyntheticDicom
        import numpy as np

        # Pass rows and cols explicitly
        ds = _SyntheticDicom(
            pixel_array=np.full((32, 32), 50, dtype=np.float32),
            modality="CT",
            rows=32,
            cols=32,
        )
        assert ds.Modality == "CT"
        assert ds.Rows == 32
        assert ds.Cols == 32

    def test_preprocessing_with_different_target_sizes(self):
        """Test preprocessing with different target sizes using synthetic dataset."""
        from imaging.preprocessing import preprocess_for_model_a

        for size in [64, 128, 224, 256]:
            tensor, display = preprocess_for_model_a(
                filepath="synthetic", target_size=size, dataset=self.synthetic_ds
            )
            assert tensor.shape == (1, 1, size, size), (
                f"Failed for target_size={size}: got {tensor.shape}"
            )
            assert display.shape == (size, size), (
                f"Failed display for target_size={size}: got {display.shape}"
            )


if __name__ == "__main__":
    # Run tests when script is executed directly
    import os

    # Test 1: Synthetic DICOM preprocessing
    print("Testing synthetic DICOM preprocessing...")
    rows, cols = 64, 64
    synthetic_pixel_data = np.full((rows, cols), 40, dtype=np.float32)
    synthetic_ds = __import__("imaging.preprocessing", fromlist=["_SyntheticDicom"])._SyntheticDicom(
        pixel_array=synthetic_pixel_data,
        modality="CT",
        rows=rows,
        cols=cols,
    )

    tensor, display = __import__("imaging.preprocessing", fromlist=["preprocess_for_model_a"]).preprocess_for_model_a(
        filepath="synthetic", target_size=224, dataset=synthetic_ds
    )
    assert tensor.shape == (1, 1, 224, 224)
    assert display.shape == (224, 224)
    assert display.dtype == np.uint8
    print("  Synthetic DICOM preprocessing: PASSED")

    # Test 2: Error handling - missing file
    print("Testing error handling...")
    try:
        from imaging.preprocessing import read_dicom_file
        read_dicom_file(filepath="/nonexistent/path.dcm")
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        print("  FileNotFoundError for missing file: PASSED")

    print()
    print("All manual tests PASSED")
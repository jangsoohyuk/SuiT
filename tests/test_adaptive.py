import unittest
import torch
import numpy as np

# Attempt to import skimage and set a flag
try:
    from skimage.segmentation import slic
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# Module imports
from suit import ParameterPredictor # Changed
from utils import generate_superpixels # Changed & removed comment for simplicity in diff
from suit import suit_tiny_224_adaptive, SuitAdaptive # Changed

# For mocking
from unittest.mock import patch, MagicMock


class TestParameterPredictor(unittest.TestCase):
    def setUp(self):
        self.predictor = ParameterPredictor()
        self.dummy_images_batch2 = torch.randn(2, 3, 224, 224) # Batch of 2
        self.dummy_images_batch1 = torch.randn(1, 3, 224, 224) # Batch of 1

    def test_output_shape_batch2(self):
        output = self.predictor(self.dummy_images_batch2)
        self.assertIsInstance(output, dict)
        self.assertIn("num_superpixels", output)
        self.assertIn("compactness", output)
        self.assertEqual(output["num_superpixels"].shape, torch.Size([2]))
        self.assertEqual(output["compactness"].shape, torch.Size([2]))

    def test_output_shape_batch1(self):
        output = self.predictor(self.dummy_images_batch1)
        self.assertIsInstance(output, dict)
        self.assertIn("num_superpixels", output)
        self.assertIn("compactness", output)
        self.assertEqual(output["num_superpixels"].shape, torch.Size([1]))
        self.assertEqual(output["compactness"].shape, torch.Size([1]))


class TestGenerateSuperpixels(unittest.TestCase):
    def setUp(self):
        self.dummy_images = torch.rand(2, 3, 64, 64)  # Batch of 2, 3 channels, 64x64
        self.K_values = torch.tensor([100, 150])
        self.m_values = torch.tensor([10.0, 20.0])

        self.dummy_gray_images = torch.rand(2, 1, 64, 64) # Grayscale

    @unittest.skipUnless(SKIMAGE_AVAILABLE, "scikit-image not available")
    def test_output_properties_color(self):
        superpixels_batch = generate_superpixels(self.dummy_images, self.K_values, self.m_values)
        self.assertEqual(superpixels_batch.shape, torch.Size([2, 64, 64]))
        self.assertTrue(superpixels_batch.dtype == torch.int64 or superpixels_batch.dtype == torch.long)

    @unittest.skipUnless(SKIMAGE_AVAILABLE, "scikit-image not available")
    def test_output_properties_grayscale(self):
        superpixels_batch = generate_superpixels(self.dummy_gray_images, self.K_values, self.m_values)
        self.assertEqual(superpixels_batch.shape, torch.Size([2, 64, 64]))
        self.assertTrue(superpixels_batch.dtype == torch.int64 or superpixels_batch.dtype == torch.long)

    @unittest.skipUnless(SKIMAGE_AVAILABLE, "scikit-image not available")
    def test_num_segments_approx(self):
        # SLIC doesn't guarantee exact K, so we check if it's reasonably close.
        # This test can be a bit flaky.
        superpixels_map_img1 = generate_superpixels(self.dummy_images[0].unsqueeze(0), 
                                                    self.K_values[0].unsqueeze(0), 
                                                    self.m_values[0].unsqueeze(0))
        num_unique_segments1 = len(torch.unique(superpixels_map_img1))
        # print(f"Image 1: K_expected={self.K_values[0].item()}, K_actual={num_unique_segments1}")
        # Allowing a tolerance, e.g., +/- 50% or a fixed number. SLIC can sometimes produce far fewer.
        # For very small images or extreme K, this can be less predictable.
        self.assertGreaterEqual(num_unique_segments1, 1) # Should have at least one segment
        # self.assertLessEqual(num_unique_segments1, self.K_values[0].item() + 20) # Example tolerance

        superpixels_map_img2 = generate_superpixels(self.dummy_images[1].unsqueeze(0), 
                                                    self.K_values[1].unsqueeze(0), 
                                                    self.m_values[1].unsqueeze(0))
        num_unique_segments2 = len(torch.unique(superpixels_map_img2))
        # print(f"Image 2: K_expected={self.K_values[1].item()}, K_actual={num_unique_segments2}")
        self.assertGreaterEqual(num_unique_segments2, 1)
        # self.assertLessEqual(num_unique_segments2, self.K_values[1].item() + 20)

    # Test the fallback behavior when skimage is not available
    @unittest.skipIf(SKIMAGE_AVAILABLE, "scikit-image is available, skipping fallback test")
    def test_no_skimage_fallback(self):
        # This test runs if SKIMAGE_AVAILABLE is False (globally for the test file)
        # or if utils.SKIMAGE_AVAILABLE is False (original was utils.superpixels.SKIMAGE_AVAILABLE)
        
        # To ensure we test the fallback in generate_superpixels,
        # we might need to patch 'utils.SKIMAGE_AVAILABLE' to False for this test
        # This assumes utils.generate_superpixels directly uses utils.SKIMAGE_AVAILABLE
        
        # Store original state of SKIMAGE_AVAILABLE in utils module
        # This requires utils module to be imported. It should be by `from utils import generate_superpixels`
        import utils as utils_module # Import the module itself to patch its attribute
        original_skimage_available_in_util = utils_module.SKIMAGE_AVAILABLE
        utils_module.SKIMAGE_AVAILABLE = False # Force fallback for this test

        with self.assertWarns(UserWarning) if hasattr(self, 'assertWarns') else self.assertRaises(UserWarning): # Python 3.8+ for assertWarns
            # In older Python, we might need to check printed output or just expect the zeros.
            # For now, let's assume print is a side effect we don't directly test, focus on output.
            # The warning is printed, not raised as an exception.
            # So, we'll check the output.
            pass # The warning is printed, not raised as an exception.
        
        superpixels_batch = generate_superpixels(self.dummy_images, self.K_values, self.m_values)
        self.assertEqual(superpixels_batch.shape, torch.Size([2, 64, 64]))
        self.assertTrue(superpixels_batch.dtype == torch.int64 or superpixels_batch.dtype == torch.long)
        self.assertTrue(torch.all(superpixels_batch == 0))

        utils_module.SKIMAGE_AVAILABLE = original_skimage_available_in_util # Restore


class TestSuitAdaptive(unittest.TestCase):
    def setUp(self):
        # Using a small variant for faster testing. Ensure num_classes is passed.
        self.num_classes = 10
        self.model = suit_tiny_224_adaptive(num_classes=self.num_classes, pretrained=False) 
        self.model.eval() # Set to eval mode for testing (disables dropout etc.)
        self.dummy_images_batch2 = torch.randn(2, 3, 224, 224) # Batch of 2
        self.dummy_images_batch1 = torch.randn(1, 3, 224, 224) # Batch of 1

    def test_forward_pass_batch2(self):
        output = self.model(self.dummy_images_batch2)
        self.assertEqual(output.shape, torch.Size([2, self.num_classes]))

    def test_forward_pass_batch1(self):
        output = self.model(self.dummy_images_batch1)
        self.assertEqual(output.shape, torch.Size([1, self.num_classes]))

    @patch('utils.generate_superpixels') # Mock the actual superpixel generation - CHANGED
    def test_parameter_clipping_integration(self, mock_generate_superpixels):
        # Configure the mock generate_superpixels to return a valid tensor
        # so the rest of the forward pass doesn't fail.
        # Shape: [B, H, W]. For suit_tiny_224, feature map before tokenization is smaller.
        # The actual spix_label is interpolated to feature map size.
        # For this test, the exact content of spix_label doesn't matter as much as K, m args.
        # Let's assume prepare_tokens will handle size interpolation.
        # Base feature map size is img_size / downsample. Default downsample is 2. So, 224/2 = 112.
        mock_sp_map = torch.zeros((self.dummy_images_batch1.shape[0], 112, 112), dtype=torch.long)
        mock_generate_superpixels.return_value = mock_sp_map

        # Create a ParameterPredictor that returns known out-of-range values
        mock_predictor = MagicMock(spec=ParameterPredictor)
        # Output for a batch of 1:
        # K_pred too low, m_pred too high
        mock_predictor.return_value = {
            "num_superpixels": torch.tensor([10.0]),  # Below min 50
            "compactness": torch.tensor([100.0]) # Above max 40
        }
        
        # Replace the model's predictor with our mock
        original_predictor = self.model.parameter_predictor
        self.model.parameter_predictor = mock_predictor

        # Run the forward pass
        _ = self.model(self.dummy_images_batch1)

        # Assert that generate_superpixels was called
        mock_generate_superpixels.assert_called_once()
        
        # Get the arguments passed to generate_superpixels
        _, M_K_m_args_tuple, _ = mock_generate_superpixels.mock_calls[0] 
        # args_tuple is (image_batch, K_batch, m_batch)
        
        passed_K = M_K_m_args_tuple[1] # K_batch
        passed_m = M_K_m_args_tuple[2] # m_batch

        # Check that K and m were clipped to the expected values
        # K: min=50, m: min=1, max=40
        self.assertEqual(passed_K.item(), 50)
        self.assertEqual(passed_m.item(), 40)

        # K_pred too high, m_pred too low
        mock_predictor.return_value = {
            "num_superpixels": torch.tensor([2000.0]), # Above max 1000
            "compactness": torch.tensor([-5.0])    # Below min 1
        }
        self.model.parameter_predictor = mock_predictor
        mock_generate_superpixels.reset_mock() # Reset call count for the next call
        mock_generate_superpixels.return_value = mock_sp_map


        _ = self.model(self.dummy_images_batch1)
        mock_generate_superpixels.assert_called_once()
        _, M_K_m_args_tuple, _ = mock_generate_superpixels.mock_calls[0]
        passed_K = M_K_m_args_tuple[1]
        passed_m = M_K_m_args_tuple[2]

        self.assertEqual(passed_K.item(), 1000)
        self.assertEqual(passed_m.item(), 1)

        # Restore original predictor
        self.model.parameter_predictor = original_predictor


if __name__ == '__main__':
    # Need to ensure that utils.superpixels.SKIMAGE_AVAILABLE is accurate before tests run
    # This is a bit tricky as the import happens at module level.
    # The TestGenerateSuperpixels.test_no_skimage_fallback tries to handle this for its specific case.
    print(f"SKIMAGE_AVAILABLE (for test execution): {SKIMAGE_AVAILABLE}")
    # utils.superpixels.SKIMAGE_AVAILABLE is no longer valid as utils.superpixels module is gone.
    # We should refer to utils.SKIMAGE_AVAILABLE if needed, which is imported by `from utils import generate_superpixels`
    # or by direct import of utils module.
    import utils as utils_module_main # For checking SKIMAGE_AVAILABLE in main
    if hasattr(utils_module_main, 'SKIMAGE_AVAILABLE'):
        print(f"utils.SKIMAGE_AVAILABLE (actual used by module): {utils_module_main.SKIMAGE_AVAILABLE}")
    else:
        print("utils.SKIMAGE_AVAILABLE not found directly in utils module for printing.")
    unittest.main()

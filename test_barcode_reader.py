#!/usr/bin/env python3
"""
Test script for the enhanced barcode reader.
"""

import unittest
import tempfile
import shutil
from pathlib import Path
import cv2
import numpy as np
from unittest.mock import patch, MagicMock

from config import Config, PreprocessingConfig, OutputConfig, load_config, save_config
from barcode_processor import BarcodeProcessor
from logger import setup_logger


class TestConfig(unittest.TestCase):
    """Test configuration management."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config_path = Path(self.temp_dir) / "test_config.json"
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_default_config(self):
        """Test default configuration creation."""
        config = Config(
            preprocessing=PreprocessingConfig(),
            output=OutputConfig()
        )
        
        self.assertEqual(config.preprocessing.clahe_clip_limit, 3.0)
        self.assertEqual(config.output.annotation_color, (0, 255, 0))
        self.assertEqual(config.max_workers, 4)
    
    def test_save_load_config(self):
        """Test configuration save and load."""
        original_config = Config(
            preprocessing=PreprocessingConfig(clahe_clip_limit=2.5),
            output=OutputConfig(image_quality=90),
            max_workers=8
        )
        
        save_config(original_config, str(self.config_path))
        loaded_config = load_config(str(self.config_path))
        
        self.assertEqual(loaded_config.preprocessing.clahe_clip_limit, 2.5)
        self.assertEqual(loaded_config.output.image_quality, 90)
        self.assertEqual(loaded_config.max_workers, 8)


class TestBarcodeProcessor(unittest.TestCase):
    """Test barcode processing functionality."""
    
    def setUp(self):
        self.config = Config(
            preprocessing=PreprocessingConfig(),
            output=OutputConfig()
        )
        self.processor = BarcodeProcessor(self.config)
    
    def test_rotate_image(self):
        """Test image rotation."""
        # Create a simple test image
        image = np.zeros((100, 100), dtype=np.uint8)
        image[40:60, 40:60] = 255  # White square
        
        # Test 90-degree rotation
        rotated = self.processor.rotate_image(image, 90)
        self.assertEqual(rotated.shape, image.shape)
        
        # Test 0-degree rotation (should return original)
        no_rotation = self.processor.rotate_image(image, 0)
        np.testing.assert_array_equal(no_rotation, image)
    
    def test_preprocess_image(self):
        """Test image preprocessing."""
        # Create a test image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        
        preprocessed = self.processor.preprocess_image(image)
        
        # Should generate multiple versions
        self.assertGreater(len(preprocessed), 5)
        
        # All versions should be grayscale
        for version in preprocessed:
            self.assertEqual(len(version.shape), 2)
    
    def test_calculate_barcode_quality(self):
        """Test barcode quality calculation."""
        # Test high-quality barcode
        quality1 = self.processor.calculate_barcode_quality("1234567890123", "EAN13")
        
        # Test low-quality barcode
        quality2 = self.processor.calculate_barcode_quality("ab", "CODE39")
        
        # Test QR code (should have higher quality multiplier)
        quality3 = self.processor.calculate_barcode_quality("test data", "QRCODE")
        
        self.assertGreater(quality1, quality2)
        self.assertGreater(quality3, quality1)
        self.assertLessEqual(quality3, 1.0)
    
    @patch('barcode_processor.pyzbar.decode')
    def test_decode_barcodes_mock(self, mock_decode):
        """Test barcode decoding with mocked pyzbar."""
        # Mock barcode object
        mock_barcode = MagicMock()
        mock_barcode.data = b"123456789"
        mock_barcode.type = "CODE128"
        mock_barcode.rect.left = 10
        mock_barcode.rect.top = 20
        mock_barcode.rect.width = 100
        mock_barcode.rect.height = 30
        mock_barcode.polygon = [(10, 20), (110, 20), (110, 50), (10, 50)]
        
        mock_decode.return_value = [mock_barcode]
        
        # Create test image
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        
        # Decode barcodes
        barcodes = self.processor.decode_barcodes(image)
        
        self.assertEqual(len(barcodes), 1)
        self.assertEqual(barcodes[0]['data'], "123456789")
        self.assertEqual(barcodes[0]['type'], "CODE128")
        self.assertIn('quality', barcodes[0])
    
    def test_remove_duplicates(self):
        """Test duplicate removal."""
        barcodes = [
            {'data': 'test1', 'type': 'CODE128', 'quality': 0.8},
            {'data': 'test1', 'type': 'CODE128', 'quality': 0.9},  # Higher quality
            {'data': 'test2', 'type': 'QRCODE', 'quality': 0.7},
        ]
        
        unique = self.processor._remove_duplicates(barcodes)
        
        self.assertEqual(len(unique), 2)
        # Should keep the higher quality version of test1
        test1_barcode = next(b for b in unique if b['data'] == 'test1')
        self.assertEqual(test1_barcode['quality'], 0.9)


class TestLogger(unittest.TestCase):
    """Test logging functionality."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.log_file = Path(self.temp_dir) / "test.log"
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_setup_logger(self):
        """Test logger setup."""
        logger = setup_logger(
            name="test_logger",
            level="DEBUG",
            log_file=str(self.log_file),
            console_output=False
        )
        
        logger.info("Test message")
        
        # Check if log file was created and contains message
        self.assertTrue(self.log_file.exists())
        with open(self.log_file, 'r') as f:
            content = f.read()
            self.assertIn("Test message", content)


def create_test_image_with_text(text: str, size: tuple = (200, 100)) -> np.ndarray:
    """Create a test image with text (simulating a barcode)."""
    image = np.ones((*size, 3), dtype=np.uint8) * 255  # White background
    
    # Add some text to simulate barcode content
    cv2.putText(image, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    return image


class TestIntegration(unittest.TestCase):
    """Integration tests."""
    
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.config = Config(
            preprocessing=PreprocessingConfig(),
            output=OutputConfig()
        )
        self.processor = BarcodeProcessor(self.config)
    
    def tearDown(self):
        shutil.rmtree(self.temp_dir)
    
    def test_full_pipeline_no_barcodes(self):
        """Test full pipeline with image containing no barcodes."""
        # Create test image
        image = create_test_image_with_text("Not a barcode")
        
        # Process image
        barcodes = self.processor.decode_barcodes(image)
        
        # Should return empty list (no barcodes detected)
        self.assertEqual(len(barcodes), 0)


if __name__ == '__main__':
    # Run tests
    unittest.main(verbosity=2)
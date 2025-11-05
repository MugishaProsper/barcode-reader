"""Enhanced barcode processing with improved algorithms and performance."""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from pyzbar import pyzbar
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from pathlib import Path
import time

from config import Config, PreprocessingConfig


logger = logging.getLogger(__name__)


# Extended barcode type mapping
BARCODE_TYPES = {
    'CODE39': 'Code 39',
    'CODE128': 'Code 128',
    'QRCODE': 'QR Code',
    'EAN13': 'EAN-13',
    'EAN8': 'EAN-8',
    'UPCA': 'UPC-A',
    'UPCE': 'UPC-E',
    'DATAMATRIX': 'Data Matrix',
    'PDF417': 'PDF417',
    'ITF': 'Interleaved 2 of 5',
    'CODABAR': 'Codabar',
    'CODE93': 'Code 93',
    'AZTEC': 'Aztec Code'
}


class BarcodeProcessor:
    """Enhanced barcode processor with configurable preprocessing."""
    
    def __init__(self, config: Config):
        self.config = config
        self.preprocessing_config = config.preprocessing
        
    def rotate_image(self, image: np.ndarray, angle: int) -> np.ndarray:
        """Rotate image by specified angle."""
        if angle == 0:
            return image
            
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height),
                                flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated
    
    def preprocess_image(self, image: np.ndarray) -> List[np.ndarray]:
        """Apply comprehensive preprocessing techniques."""
        versions = []
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        
        # Base rotations if enabled
        rotation_angles = self.preprocessing_config.rotation_angles if self.preprocessing_config.enable_rotation_correction else [0]
        
        for angle in rotation_angles:
            rotated = self.rotate_image(gray, angle)
            
            # 1. Original
            versions.append(rotated)
            
            # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
            clahe = cv2.createCLAHE(
                clipLimit=self.preprocessing_config.clahe_clip_limit,
                tileGridSize=self.preprocessing_config.clahe_tile_size
            )
            versions.append(clahe.apply(rotated))
            
            # 3. Otsu's thresholding
            _, thresh = cv2.threshold(rotated, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            versions.append(thresh)
            
            # 4. Adaptive thresholding
            adaptive_thresh = cv2.adaptiveThreshold(
                rotated, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
            versions.append(adaptive_thresh)
            
            # 5. Inverted
            versions.append(cv2.bitwise_not(rotated))
            
            # 6. Morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.preprocessing_config.morph_kernel_size)
            morphed = cv2.morphologyEx(rotated, cv2.MORPH_CLOSE, kernel)
            versions.append(morphed)
            
            # 7. Sharpening
            blur = cv2.GaussianBlur(rotated, (0, 0), self.preprocessing_config.gaussian_blur_sigma)
            sharpened = cv2.addWeighted(
                rotated, self.preprocessing_config.sharpen_alpha,
                blur, self.preprocessing_config.sharpen_beta, 0
            )
            versions.append(sharpened)
            
            # 8. Edge enhancement
            edges = cv2.Canny(rotated, 50, 150)
            versions.append(edges)
            
            # 9. Bilateral filter (noise reduction while preserving edges)
            bilateral = cv2.bilateralFilter(rotated, 9, 75, 75)
            versions.append(bilateral)
        
        return versions
    
    def calculate_barcode_quality(self, barcode_data: str, barcode_type: str) -> float:
        """Calculate a quality score for the detected barcode."""
        quality = 1.0
        
        # Length-based quality (longer codes are generally more reliable)
        if len(barcode_data) < 3:
            quality *= 0.5
        elif len(barcode_data) > 50:
            quality *= 0.8
            
        # Type-based quality (some formats are more reliable)
        reliable_types = ['QRCODE', 'DATAMATRIX', 'PDF417']
        if barcode_type in reliable_types:
            quality *= 1.2
        
        # Character composition quality
        if barcode_data.isdigit():
            quality *= 1.1  # Numeric codes are often more reliable
        elif any(c in barcode_data for c in '!@#$%^&*()'):
            quality *= 0.9  # Special characters might indicate errors
            
        return min(quality, 1.0)
    
    def decode_barcodes(self, image: np.ndarray) -> List[Dict]:
        """Decode barcodes with enhanced preprocessing and quality assessment."""
        start_time = time.time()
        preprocessed_images = self.preprocess_image(image)
        all_barcodes = []
        
        logger.debug(f"Generated {len(preprocessed_images)} preprocessed versions")
        
        for i, processed_img in enumerate(preprocessed_images):
            try:
                barcodes = pyzbar.decode(processed_img, symbols=None)
                
                for barcode in barcodes:
                    try:
                        data = barcode.data.decode('utf-8', errors='replace')
                        barcode_type = barcode.type
                        
                        # Calculate quality score
                        quality = self.calculate_barcode_quality(data, barcode_type)
                        
                        # Get bounding box
                        x, y, w, h = barcode.rect.left, barcode.rect.top, barcode.rect.width, barcode.rect.height
                        
                        # Convert polygon to more precise coordinates
                        points = barcode.polygon
                        if len(points) > 4:
                            hull = cv2.convexHull(np.array([p for p in points], dtype=np.float32))
                            hull = [tuple(map(int, point[0])) for point in hull]
                        else:
                            hull = [(int(p.x), int(p.y)) for p in points]
                        
                        all_barcodes.append({
                            'data': data,
                            'type': barcode_type,
                            'type_name': BARCODE_TYPES.get(barcode_type, barcode_type),
                            'x': x, 'y': y, 'w': w, 'h': h,
                            'polygon': hull,
                            'quality': quality,
                            'preprocessing_method': i
                        })
                        
                    except Exception as e:
                        logger.warning(f"Error processing barcode: {e}")
                        continue
                        
            except Exception as e:
                logger.warning(f"Error in preprocessing method {i}: {e}")
                continue
        
        # Remove duplicates and keep highest quality
        unique_barcodes = self._remove_duplicates(all_barcodes)
        
        processing_time = time.time() - start_time
        logger.debug(f"Processed image in {processing_time:.2f}s, found {len(unique_barcodes)} unique barcodes")
        
        return unique_barcodes
    
    def _remove_duplicates(self, barcodes: List[Dict]) -> List[Dict]:
        """Remove duplicate barcodes, keeping the highest quality version."""
        if not barcodes:
            return []
        
        # Group by data and type
        groups = {}
        for barcode in barcodes:
            key = (barcode['data'], barcode['type'])
            if key not in groups:
                groups[key] = []
            groups[key].append(barcode)
        
        # Keep the highest quality barcode from each group
        unique = []
        for group in groups.values():
            best = max(group, key=lambda x: x['quality'])
            unique.append(best)
        
        return unique
    
    def draw_barcodes(self, image: np.ndarray, barcodes: List[Dict], output_path: str):
        """Draw enhanced annotations on detected barcodes."""
        img = image.copy()
        config = self.config.output
        
        for i, barcode in enumerate(barcodes):
            color = config.annotation_color
            thickness = config.annotation_thickness
            
            # Draw bounding rectangle
            x, y, w, h = barcode['x'], barcode['y'], barcode['w'], barcode['h']
            cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)
            
            # Draw polygon for more accurate representation
            if len(barcode['polygon']) >= 3:
                pts = np.array(barcode['polygon'], np.int32)
                cv2.polylines(img, [pts], isClosed=True, color=color, thickness=thickness)
            
            # Create label with quality info
            label_parts = [f"{barcode['type_name']}: {barcode['data'][:30]}"]
            if config.include_confidence:
                label_parts.append(f"Q:{barcode['quality']:.2f}")
            
            label = " | ".join(label_parts)
            
            # Calculate text size for background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = config.font_scale
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            
            # Draw background rectangle for text
            text_bg_color = (0, 0, 0)
            cv2.rectangle(img, (x, y - text_size[1] - 10), 
                         (x + text_size[0], y), text_bg_color, -1)
            
            # Draw text
            cv2.putText(img, label, (x, y - 5), font, font_scale, color, thickness)
        
        # Save with specified quality
        if output_path.lower().endswith(('.jpg', '.jpeg')):
            cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, config.image_quality])
        else:
            cv2.imwrite(output_path, img)
        
        logger.info(f"Saved annotated image: {output_path}")


def process_images_parallel(image_paths: List[str], processor: BarcodeProcessor, 
                          output_dir: str, draw: bool, max_workers: int = 4) -> List[Dict]:
    """Process multiple images in parallel."""
    all_records = []
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {
            executor.submit(_process_single_image, path, processor, output_dir, draw): path 
            for path in image_paths
        }
        
        for future in as_completed(future_to_path):
            path = future_to_path[future]
            try:
                records = future.result()
                all_records.extend(records)
            except Exception as e:
                logger.error(f"Error processing {path}: {e}")
                all_records.append({
                    'filename': Path(path).name,
                    'index': 0,
                    'type': 'ERROR',
                    'data': str(e),
                    'x': '', 'y': '', 'width': '', 'height': '',
                    'quality': 0.0
                })
    
    return all_records


def _process_single_image(img_path: str, processor: BarcodeProcessor, 
                         output_dir: str, draw: bool) -> List[Dict]:
    """Process a single image (helper function for parallel processing)."""
    try:
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError("Failed to load image")
        
        barcodes = processor.decode_barcodes(image)
        records = []
        
        logger.info(f"{Path(img_path).name}: Found {len(barcodes)} barcode(s)")
        
        for i, barcode in enumerate(barcodes):
            record = {
                'filename': Path(img_path).name,
                'index': i + 1,
                'type': barcode['type_name'],
                'data': barcode['data'],
                'x': barcode['x'],
                'y': barcode['y'],
                'width': barcode['w'],
                'height': barcode['h'],
                'quality': barcode['quality']
            }
            records.append(record)
            logger.info(f"  [{i+1}] {barcode['type_name']}: {barcode['data']} (Q: {barcode['quality']:.2f})")
        
        if draw and barcodes:
            out_path = Path(output_dir) / f"annotated_{Path(img_path).name}"
            processor.draw_barcodes(image, barcodes, str(out_path))
        
        if not barcodes:
            records.append({
                'filename': Path(img_path).name,
                'index': 0,
                'type': '', 'data': 'NO BARCODE DETECTED',
                'x': '', 'y': '', 'width': '', 'height': '',
                'quality': 0.0
            })
        
        return records
        
    except Exception as e:
        logger.error(f"Error processing {img_path}: {e}")
        return [{
            'filename': Path(img_path).name,
            'index': 0,
            'type': 'ERROR',
            'data': str(e),
            'x': '', 'y': '', 'width': '', 'height': '',
            'quality': 0.0
        }]
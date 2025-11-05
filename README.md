# Enhanced Barcode Reader

A powerful, configurable barcode and QR code reader for Windows with advanced image preprocessing and parallel processing capabilities.

## Features

- **Multi-format Support**: Code 39, Code 128, QR codes, EAN-13/8, UPC-A/E, Data Matrix, PDF417, ITF, Codabar, Code 93, Aztec
- **Advanced Preprocessing**: 9 different image enhancement techniques including CLAHE, adaptive thresholding, morphological operations, and rotation correction
- **Quality Assessment**: Automatic quality scoring for detected barcodes
- **Parallel Processing**: Multi-threaded processing for batch operations
- **Configurable**: JSON-based configuration system
- **Comprehensive Logging**: Detailed logging with multiple levels
- **Enhanced Annotations**: High-quality output images with quality indicators

## Installation

1. Clone or download this repository
2. Install Python 3.7+ 
3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Process a single image
python barcode_reader.py image.jpg --draw

# Process a folder with parallel processing
python barcode_reader.py images/ --parallel --workers 8 --draw

# Create default configuration file
python barcode_reader.py --create-config
```

## Usage

### Basic Usage

```bash
python barcode_reader.py INPUT [OPTIONS]
```

### Arguments

- `INPUT`: Image file or directory path

### Options

- `-o, --output DIR`: Output directory (default: barcode_results)
- `--csv FILE`: CSV output filename (default: barcodes.csv)
- `--draw`: Save annotated images with bounding boxes
- `--parallel`: Enable parallel processing for multiple images
- `--workers N`: Number of worker threads
- `--config FILE`: Configuration file path (default: config.json)
- `--create-config`: Create default configuration file
- `--log-level LEVEL`: Set logging level (DEBUG, INFO, WARNING, ERROR)
- `--log-file FILE`: Save logs to file
- `--quiet`: Suppress console output

### Examples

```bash
# Process single image with annotations
python barcode_reader.py photo.jpg --draw

# Batch process with custom output
python barcode_reader.py "C:\Scans" -o results --csv inventory.csv

# High-performance batch processing
python barcode_reader.py images/ --parallel --workers 8 --log-level INFO

# Debug mode with detailed logging
python barcode_reader.py problem_image.jpg --log-level DEBUG --log-file debug.log
```

## Configuration

The tool uses a JSON configuration file for advanced settings. Create one with:

```bash
python barcode_reader.py --create-config
```

### Configuration Options

```json
{
  "preprocessing": {
    "clahe_clip_limit": 3.0,
    "clahe_tile_size": [8, 8],
    "morph_kernel_size": [3, 3],
    "gaussian_blur_sigma": 3.0,
    "sharpen_alpha": 1.5,
    "sharpen_beta": -0.5,
    "enable_rotation_correction": true,
    "rotation_angles": [0, 90, 180, 270]
  },
  "output": {
    "csv_delimiter": ",",
    "image_quality": 95,
    "annotation_color": [0, 255, 0],
    "annotation_thickness": 2,
    "font_scale": 0.6,
    "include_confidence": true
  },
  "supported_extensions": [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif", ".webp"],
  "max_workers": 4,
  "log_level": "INFO"
}
```

## Output

### CSV Format

The tool generates a CSV file with the following columns:

- `filename`: Source image filename
- `index`: Barcode index within the image
- `type`: Barcode type (e.g., "QR Code", "Code 128")
- `data`: Decoded barcode content
- `x`, `y`: Top-left coordinates
- `width`, `height`: Bounding box dimensions
- `quality`: Quality score (0.0-1.0, if enabled)

### Annotated Images

When using `--draw`, the tool saves annotated versions of images with:
- Bounding rectangles around detected barcodes
- Precise polygon outlines for rotated codes
- Labels with barcode type and content
- Quality scores (if enabled)

## Advanced Features

### Image Preprocessing

The tool applies multiple preprocessing techniques to improve detection:

1. **Original grayscale**
2. **CLAHE** (Contrast Limited Adaptive Histogram Equalization)
3. **Otsu's thresholding**
4. **Adaptive thresholding**
5. **Inverted images**
6. **Morphological operations**
7. **Sharpening**
8. **Edge enhancement**
9. **Bilateral filtering**

### Rotation Correction

Automatically tries different rotation angles (0°, 90°, 180°, 270°) to detect rotated barcodes.

### Quality Assessment

Each detected barcode receives a quality score based on:
- Data length and composition
- Barcode type reliability
- Character validation

### Parallel Processing

For batch operations, the tool can process multiple images simultaneously using configurable worker threads.

## Troubleshooting

### Common Issues

1. **No barcodes detected**: Try adjusting preprocessing parameters in config.json
2. **Poor quality detection**: Enable rotation correction and increase preprocessing iterations
3. **Slow processing**: Use parallel processing with `--parallel --workers N`
4. **Memory issues**: Reduce the number of workers or process images in smaller batches

### Debug Mode

Use debug logging to troubleshoot detection issues:

```bash
python barcode_reader.py image.jpg --log-level DEBUG --log-file debug.log
```

## Supported Formats

### Image Formats
- PNG, JPG/JPEG, BMP, TIFF/TIF, WebP

### Barcode Types
- **1D**: Code 39, Code 128, EAN-13, EAN-8, UPC-A, UPC-E, ITF, Codabar, Code 93
- **2D**: QR Code, Data Matrix, PDF417, Aztec Code

## Requirements

- Python 3.7+
- OpenCV 4.5+
- pyzbar 0.1.8+
- Pillow 8.0+
- NumPy 1.20+

## License

This project is open source. Feel free to modify and distribute according to your needs.
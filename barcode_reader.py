#!/usr/bin/env python3
"""
Enhanced Windows Barcode Reader
Supports 1D & 2D barcodes with advanced preprocessing and parallel processing.
"""

import os
import csv
import argparse
import json
from pathlib import Path
from typing import List, Dict
import time

from config import Config, load_config, save_config
from logger import setup_logger
from barcode_processor import BarcodeProcessor, process_images_parallel

def find_images(input_path: Path, config: Config) -> List[Path]:
    """Find all supported image files in the given path."""
    images = []
    
    if input_path.is_dir():
        for ext in config.supported_extensions:
            images.extend(input_path.glob(f"*{ext}"))
            images.extend(input_path.glob(f"*{ext.upper()}"))
    else:
        if input_path.suffix.lower() in config.supported_extensions:
            images = [input_path]
    
    return sorted(images)


def save_results(records: List[Dict], csv_path: Path, config: Config):
    """Save results to CSV with proper encoding and formatting."""
    fieldnames = [
        'filename', 'index', 'type', 'data', 'x', 'y', 'width', 'height'
    ]
    
    if config.output.include_confidence:
        fieldnames.append('quality')
    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(
            f, 
            fieldnames=fieldnames,
            delimiter=config.output.csv_delimiter
        )
        writer.writeheader()
        writer.writerows(records)


def print_summary(records: List[Dict], processing_time: float):
    """Print processing summary."""
    total_files = len(set(r['filename'] for r in records))
    total_barcodes = len([r for r in records if r['data'] not in ['NO BARCODE DETECTED', 'ERROR']])
    error_files = len([r for r in records if r['type'] == 'ERROR'])
    
    print(f"\n{'='*50}")
    print(f"PROCESSING SUMMARY")
    print(f"{'='*50}")
    print(f"Files processed: {total_files}")
    print(f"Barcodes found: {total_barcodes}")
    print(f"Files with errors: {error_files}")
    print(f"Processing time: {processing_time:.2f}s")
    print(f"{'='*50}")


def create_default_config():
    """Create a default configuration file."""
    config = Config(
        preprocessing=PreprocessingConfig(),
        output=OutputConfig()
    )
    save_config(config)
    print("Created default config.json file")

def main():
    parser = argparse.ArgumentParser(
        description="Enhanced Windows Barcode Reader (1D & 2D)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python barcode_reader.py image.jpg --draw
  python barcode_reader.py "C:\\Scans\\" -o results --csv all_barcodes.csv
  python barcode_reader.py folder/ --parallel --workers 8 --log-level DEBUG
  python barcode_reader.py --create-config  # Create default config file
        """
    )
    
    # Main arguments
    parser.add_argument("input", nargs='?', help="Image file or folder")
    parser.add_argument("-o", "--output", default="barcode_results", help="Output folder")
    parser.add_argument("--csv", default="barcodes.csv", help="CSV output file")
    parser.add_argument("--draw", action="store_true", help="Save annotated images")
    
    # Performance options
    parser.add_argument("--parallel", action="store_true", help="Enable parallel processing")
    parser.add_argument("--workers", type=int, help="Number of worker threads (default: from config)")
    
    # Configuration options
    parser.add_argument("--config", default="config.json", help="Configuration file path")
    parser.add_argument("--create-config", action="store_true", help="Create default config file and exit")
    
    # Logging options
    parser.add_argument("--log-level", choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], 
                       help="Logging level (default: from config)")
    parser.add_argument("--log-file", help="Log file path")
    parser.add_argument("--quiet", action="store_true", help="Suppress console output")

    args = parser.parse_args()
    
    # Handle config creation
    if args.create_config:
        create_default_config()
        return
    
    if not args.input:
        parser.error("Input path is required (unless using --create-config)")
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print(f"Error loading config: {e}")
        print("Creating default configuration...")
        config = Config(
            preprocessing=PreprocessingConfig(),
            output=OutputConfig()
        )
    
    # Override config with command line arguments
    if args.workers:
        config.max_workers = args.workers
    if args.log_level:
        config.log_level = args.log_level
    
    # Setup logging
    logger = setup_logger(
        level=config.log_level,
        log_file=args.log_file,
        console_output=not args.quiet
    )
    
    logger.info("Starting Enhanced Barcode Reader")
    logger.info(f"Configuration: {args.config}")
    
    # Validate input path
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return
    
    # Setup output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find images
    images = find_images(input_path, config)
    if not images:
        logger.error("No supported images found!")
        logger.info(f"Supported extensions: {', '.join(config.supported_extensions)}")
        return
    
    logger.info(f"Found {len(images)} image(s) to process")
    
    # Initialize processor
    processor = BarcodeProcessor(config)
    
    # Process images
    start_time = time.time()
    
    if args.parallel and len(images) > 1:
        logger.info(f"Processing images in parallel with {config.max_workers} workers")
        all_records = process_images_parallel(
            [str(img) for img in images], 
            processor, 
            str(output_dir), 
            args.draw, 
            config.max_workers
        )
    else:
        logger.info("Processing images sequentially")
        all_records = []
        for img_path in images:
            from barcode_processor import _process_single_image
            records = _process_single_image(str(img_path), processor, str(output_dir), args.draw)
            all_records.extend(records)
    
    processing_time = time.time() - start_time
    
    # Save results
    csv_path = Path(args.csv)
    save_results(all_records, csv_path, config)
    
    # Print summary
    if not args.quiet:
        print_summary(all_records, processing_time)
        print(f"\nResults saved to:")
        print(f"  CSV: {csv_path.resolve()}")
        if args.draw:
            print(f"  Annotated images: {output_dir.resolve()}")
    
    logger.info("Processing completed successfully")


if __name__ == "__main__":
    main()
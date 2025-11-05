#!/usr/bin/env python3
"""
Benchmark script for the enhanced barcode reader.
"""

import time
import statistics
from pathlib import Path
import cv2
import numpy as np
from typing import List, Dict
import argparse

from config import Config, PreprocessingConfig, OutputConfig
from barcode_processor import BarcodeProcessor, process_images_parallel
from logger import setup_logger


def create_synthetic_barcode_image(size: tuple = (400, 300)) -> np.ndarray:
    """Create a synthetic image with barcode-like patterns."""
    image = np.ones((*size, 3), dtype=np.uint8) * 255  # White background
    
    # Add some vertical lines to simulate barcode
    for i in range(50, 350, 10):
        thickness = np.random.randint(1, 4)
        cv2.line(image, (i, 100), (i, 200), (0, 0, 0), thickness)
    
    # Add some noise
    noise = np.random.randint(0, 50, size, dtype=np.uint8)
    image = cv2.add(image, noise[:, :, np.newaxis])
    
    return image


def benchmark_preprocessing(processor: BarcodeProcessor, image: np.ndarray, iterations: int = 10) -> Dict:
    """Benchmark preprocessing performance."""
    times = []
    
    for _ in range(iterations):
        start_time = time.time()
        preprocessed = processor.preprocess_image(image)
        end_time = time.time()
        times.append(end_time - start_time)
    
    return {
        'mean_time': statistics.mean(times),
        'std_time': statistics.stdev(times) if len(times) > 1 else 0,
        'min_time': min(times),
        'max_time': max(times),
        'versions_generated': len(preprocessed)
    }


def benchmark_decoding(processor: BarcodeProcessor, image: np.ndarray, iterations: int = 10) -> Dict:
    """Benchmark barcode decoding performance."""
    times = []
    barcode_counts = []
    
    for _ in range(iterations):
        start_time = time.time()
        barcodes = processor.decode_barcodes(image)
        end_time = time.time()
        
        times.append(end_time - start_time)
        barcode_counts.append(len(barcodes))
    
    return {
        'mean_time': statistics.mean(times),
        'std_time': statistics.stdev(times) if len(times) > 1 else 0,
        'min_time': min(times),
        'max_time': max(times),
        'mean_barcodes': statistics.mean(barcode_counts),
        'total_iterations': iterations
    }


def benchmark_parallel_processing(images: List[np.ndarray], max_workers_list: List[int]) -> Dict:
    """Benchmark parallel processing with different worker counts."""
    results = {}
    temp_dir = Path("temp_benchmark")
    temp_dir.mkdir(exist_ok=True)
    
    # Save test images
    image_paths = []
    for i, image in enumerate(images):
        path = temp_dir / f"test_image_{i}.png"
        cv2.imwrite(str(path), image)
        image_paths.append(str(path))
    
    try:
        config = Config(
            preprocessing=PreprocessingConfig(),
            output=OutputConfig()
        )
        processor = BarcodeProcessor(config)
        
        for workers in max_workers_list:
            start_time = time.time()
            records = process_images_parallel(
                image_paths, processor, str(temp_dir), False, workers
            )
            end_time = time.time()
            
            results[workers] = {
                'time': end_time - start_time,
                'records_count': len(records),
                'images_processed': len(images)
            }
    
    finally:
        # Cleanup
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
    
    return results


def run_comprehensive_benchmark():
    """Run comprehensive benchmark suite."""
    logger = setup_logger(level="INFO")
    logger.info("Starting comprehensive benchmark")
    
    # Create test configurations
    configs = {
        'default': Config(
            preprocessing=PreprocessingConfig(),
            output=OutputConfig()
        ),
        'high_quality': Config(
            preprocessing=PreprocessingConfig(
                clahe_clip_limit=4.0,
                enable_rotation_correction=True,
                rotation_angles=[0, 45, 90, 135, 180, 225, 270, 315]
            ),
            output=OutputConfig()
        ),
        'fast': Config(
            preprocessing=PreprocessingConfig(
                enable_rotation_correction=False,
                rotation_angles=[0]
            ),
            output=OutputConfig()
        )
    }
    
    # Create test images
    test_images = [
        create_synthetic_barcode_image((200, 150)),
        create_synthetic_barcode_image((400, 300)),
        create_synthetic_barcode_image((800, 600)),
        create_synthetic_barcode_image((1200, 900))
    ]
    
    print("=" * 60)
    print("BARCODE READER BENCHMARK RESULTS")
    print("=" * 60)
    
    # Benchmark each configuration
    for config_name, config in configs.items():
        print(f"\nConfiguration: {config_name.upper()}")
        print("-" * 40)
        
        processor = BarcodeProcessor(config)
        
        for i, image in enumerate(test_images):
            size_name = f"{image.shape[1]}x{image.shape[0]}"
            
            # Benchmark preprocessing
            prep_results = benchmark_preprocessing(processor, image, 5)
            
            # Benchmark decoding
            decode_results = benchmark_decoding(processor, image, 5)
            
            print(f"\nImage {i+1} ({size_name}):")
            print(f"  Preprocessing: {prep_results['mean_time']:.3f}s ± {prep_results['std_time']:.3f}s")
            print(f"  Decoding: {decode_results['mean_time']:.3f}s ± {decode_results['std_time']:.3f}s")
            print(f"  Total: {prep_results['mean_time'] + decode_results['mean_time']:.3f}s")
            print(f"  Versions: {prep_results['versions_generated']}")
    
    # Benchmark parallel processing
    print(f"\nPARALLEL PROCESSING BENCHMARK")
    print("-" * 40)
    
    parallel_results = benchmark_parallel_processing(test_images[:4], [1, 2, 4, 8])
    
    for workers, result in parallel_results.items():
        throughput = result['images_processed'] / result['time']
        print(f"Workers {workers}: {result['time']:.2f}s ({throughput:.1f} images/sec)")
    
    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Benchmark the barcode reader")
    parser.add_argument("--comprehensive", action="store_true", 
                       help="Run comprehensive benchmark suite")
    parser.add_argument("--image", help="Benchmark specific image file")
    parser.add_argument("--iterations", type=int, default=10,
                       help="Number of iterations for timing")
    
    args = parser.parse_args()
    
    if args.comprehensive:
        run_comprehensive_benchmark()
    elif args.image:
        # Benchmark specific image
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image file not found: {image_path}")
            return
        
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Error: Could not load image: {image_path}")
            return
        
        config = Config(
            preprocessing=PreprocessingConfig(),
            output=OutputConfig()
        )
        processor = BarcodeProcessor(config)
        
        print(f"Benchmarking image: {image_path}")
        print(f"Image size: {image.shape[1]}x{image.shape[0]}")
        print(f"Iterations: {args.iterations}")
        print("-" * 40)
        
        prep_results = benchmark_preprocessing(processor, image, args.iterations)
        decode_results = benchmark_decoding(processor, image, args.iterations)
        
        print(f"Preprocessing: {prep_results['mean_time']:.3f}s ± {prep_results['std_time']:.3f}s")
        print(f"Decoding: {decode_results['mean_time']:.3f}s ± {decode_results['std_time']:.3f}s")
        print(f"Total: {prep_results['mean_time'] + decode_results['mean_time']:.3f}s")
        print(f"Barcodes found: {decode_results['mean_barcodes']:.1f}")
    else:
        print("Use --comprehensive for full benchmark or --image <path> for specific image")


if __name__ == "__main__":
    main()
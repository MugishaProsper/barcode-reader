"""Configuration management for barcode reader."""

import json
from pathlib import Path
from typing import Dict, List, Any
from dataclasses import dataclass, asdict


@dataclass
class PreprocessingConfig:
    """Image preprocessing configuration."""
    clahe_clip_limit: float = 3.0
    clahe_tile_size: tuple = (8, 8)
    morph_kernel_size: tuple = (3, 3)
    gaussian_blur_sigma: float = 3.0
    sharpen_alpha: float = 1.5
    sharpen_beta: float = -0.5
    enable_rotation_correction: bool = True
    rotation_angles: List[int] = None
    
    def __post_init__(self):
        if self.rotation_angles is None:
            self.rotation_angles = [0, 90, 180, 270]


@dataclass
class OutputConfig:
    """Output configuration."""
    csv_delimiter: str = ','
    image_quality: int = 95
    annotation_color: tuple = (0, 255, 0)
    annotation_thickness: int = 2
    font_scale: float = 0.6
    include_confidence: bool = True


@dataclass
class Config:
    """Main configuration class."""
    preprocessing: PreprocessingConfig
    output: OutputConfig
    supported_extensions: List[str] = None
    max_workers: int = 4
    enable_gpu: bool = False
    log_level: str = "INFO"
    
    def __post_init__(self):
        if self.supported_extensions is None:
            self.supported_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.webp']


def load_config(config_path: str = "config.json") -> Config:
    """Load configuration from JSON file."""
    config_file = Path(config_path)
    
    if config_file.exists():
        with open(config_file, 'r') as f:
            data = json.load(f)
        
        # Convert nested dicts to dataclasses
        preprocessing = PreprocessingConfig(**data.get('preprocessing', {}))
        output = OutputConfig(**data.get('output', {}))
        
        return Config(
            preprocessing=preprocessing,
            output=output,
            **{k: v for k, v in data.items() if k not in ['preprocessing', 'output']}
        )
    
    return Config(
        preprocessing=PreprocessingConfig(),
        output=OutputConfig()
    )


def save_config(config: Config, config_path: str = "config.json"):
    """Save configuration to JSON file."""
    with open(config_path, 'w') as f:
        json.dump(asdict(config), f, indent=2)
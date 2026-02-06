"""
Model downloader utility for SCRFD and ArcFace models.

Downloads models from InsightFace and converts to FP16 for Tensor Core optimization.
"""

import os
import requests
from pathlib import Path
from loguru import logger
from tqdm import tqdm


class ModelDownloader:
    """Download and prepare face detection and recognition models."""
    
    # Model URLs (InsightFace model zoo)
    MODELS = {
        "scrfd_2.5g_kps": {
            "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/scrfd_2.5g_bnkps.onnx",
            "filename": "scrfd_2.5g_kps.onnx",
            "fp16_filename": "scrfd_2.5g_kps_fp16.onnx"
        },
        "arcface_r100_v1": {
            "url": "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
            "filename": "w600k_r50.onnx",  # Extract from zip
            "fp16_filename": "arcface_r100_v1_fp16.onnx"
        }
    }
    
    def __init__(self, models_dir: str = "./models"):
        """
        Initialize model downloader.
        
        Args:
            models_dir: Directory to save models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"ModelDownloader initialized with directory: {models_dir}")
    
    def download_file(self, url: str, filename: str) -> Path:
        """
        Download file with progress bar.
        
        Args:
            url: Download URL
            filename: Output filename
        
        Returns:
            Path to downloaded file
        """
        filepath = self.models_dir / filename
        
        if filepath.exists():
            logger.info(f"File already exists: {filepath}")
            return filepath
        
        logger.info(f"Downloading {filename} from {url}")
        
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        
        with open(filepath, 'wb') as f, tqdm(
            desc=filename,
            total=total_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
        
        logger.info(f"Downloaded: {filepath}")
        return filepath
    
    def convert_to_fp16(self, model_path: Path, output_path: Path):
        """
        Convert ONNX model to FP16 precision.
        
        Args:
            model_path: Input model path
            output_path: Output model path
        """
        try:
            import onnx
            from onnxconverter_common import float16
            
            if output_path.exists():
                logger.info(f"FP16 model already exists: {output_path}")
                return
            
            logger.info(f"Converting {model_path.name} to FP16...")
            
            # Load model
            model = onnx.load(str(model_path))
            
            # Convert to FP16
            model_fp16 = float16.convert_float_to_float16(model)
            
            # Save
            onnx.save(model_fp16, str(output_path))
            
            logger.info(f"Saved FP16 model: {output_path}")
        
        except ImportError:
            logger.warning(
                "onnx and onnxconverter-common not installed. "
                "Skipping FP16 conversion. Install with: "
                "pip install onnx onnxconverter-common"
            )
            # Copy original model as fallback
            import shutil
            shutil.copy(model_path, output_path)
            logger.info(f"Copied original model to: {output_path}")
    
    def download_scrfd(self):
        """Download and prepare SCRFD model."""
        logger.info("Downloading SCRFD model...")
        
        model_info = self.MODELS["scrfd_2.5g_kps"]
        
        # Download
        model_path = self.download_file(model_info["url"], model_info["filename"])
        
        # Convert to FP16
        fp16_path = self.models_dir / model_info["fp16_filename"]
        self.convert_to_fp16(model_path, fp16_path)
        
        logger.info("SCRFD model ready!")
    
    def download_arcface(self):
        """Download and prepare ArcFace model."""
        logger.info("Downloading ArcFace model...")
        
        # Note: This is a simplified version
        # In practice, you may need to extract from zip and select the right model
        logger.warning(
            "ArcFace download not fully implemented. "
            "Please manually download from InsightFace model zoo: "
            "https://github.com/deepinsight/insightface/tree/master/model_zoo"
        )
        
        logger.info(
            "Recommended model: buffalo_l (w600k_r50.onnx)\n"
            "After downloading, place it in ./models/ and rename to arcface_r100_v1_fp16.onnx"
        )
    
    def download_all(self):
        """Download all models."""
        self.download_scrfd()
        self.download_arcface()


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download face detection and recognition models")
    parser.add_argument(
        "--models-dir",
        type=str,
        default="./models",
        help="Directory to save models"
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["scrfd", "arcface", "all"],
        default="all",
        help="Which model to download"
    )
    
    args = parser.parse_args()
    
    downloader = ModelDownloader(args.models_dir)
    
    if args.model == "scrfd":
        downloader.download_scrfd()
    elif args.model == "arcface":
        downloader.download_arcface()
    else:
        downloader.download_all()


if __name__ == "__main__":
    main()

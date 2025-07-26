#!/usr/bin/env python3
"""
Image Processing Suite
A comprehensive toolkit for image manipulation, optimization, and enhancement.
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
    import cv2
    import piexif
    from tqdm import tqdm
    from colorama import init, Fore, Style
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)

# Initialize colorama for colored output
init(autoreset=True)

class ImageProcessor:
    """Main class for processing and manipulating images."""

    def __init__(self):
        self.supported_formats = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"]
        self.output_dir = Path("processed_images")
        self.output_dir.mkdir(exist_ok=True)

    def resize_image(self, image_path: str, output_path: str, new_size: Tuple[int, int]):
        """Resize and save image to new dimensions."""
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            return

        try:
            with Image.open(image_path) as img:
                img_resized = img.resize(new_size, Image.ANTIALIAS)
                img_resized.save(output_path)
            print(f"✅ Image saved: {output_path}")
        except Exception as e:
            print(f"Error resizing image: {e}")

    def apply_watermark(self, image_path: str, watermark_text: str, output_path: str):
        """Apply text watermark to image."""
        try:
            with Image.open(image_path) as img:
                watermark = ImageDraw.Draw(img)
                font = ImageFont.load_default()
                textwidth, textheight = watermark.textsize(watermark_text, font)
                width, height = img.size
                # Position in the bottom right corner
                x = width - textwidth - 10
                y = height - textheight - 10
                watermark.text((x, y), watermark_text, font=font, fill=(255, 255, 255, 128))
                img.save(output_path)
            print(f"✅ Watermarked image saved: {output_path}")
        except Exception as e:
            print(f"Error applying watermark: {e}")

    def optimize_images(self, directory: str, quality: int = 85):
        """Optimize all images in a directory to reduce file size."""
        supported_files = [f for f in Path(directory).iterdir() if f.suffix.lower() in self.supported_formats]
        
        if not supported_files:
            print("❌ No supported image files found in the directory.")
            return

        for image_file in tqdm(supported_files, desc="Optimizing Images"):
            try:
                with Image.open(image_file) as img:
                    # Convert to RGB if necessary
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    # Save optimized version
                    img.save(image_file, optimize=True, quality=quality)
            except Exception as e:
                print(f"Error optimizing {image_file}: {e}")

    def convert_format(self, image_path: str, fmt: str, output_path: str):
        """Convert image to specified format (e.g., JPEG -> PNG)."""
        try:
            with Image.open(image_path) as img:
                img = img.convert("RGB")  # Ensure conversion does not fail due to transparency
                img.save(output_path, fmt.upper())
            print(f"✅ Converted image saved: {output_path}")
        except Exception as e:
            print(f"Error converting image format: {e}")

    def show_exif_data(self, image_path: str):
        """Display EXIF metadata from an image."""
        try:
            exif_data = piexif.load(image_path)
            for tag, value in exif_data["0th"].items():
                tag_name = piexif.TAGS["0th"].get(tag, {}).get("name", tag)
                print(f"{tag_name}: {value}")
        except Exception as e:
            print(f"Error displaying EXIF data: {e}")

    def batch_process(self, directory: str, operation: str, **kwargs):
        """Apply an operation to all images in a directory."""
        supported_files = [f for f in Path(directory).iterdir() if f.suffix.lower() in self.supported_formats]
        
        if not supported_files:
            print("❌ No supported image files found in the directory.")
            return

        for image_file in tqdm(supported_files, desc=f"Batch {operation.title()}"):
            image_output = str(image_file.with_name(f"processed_{image_file.name}"))
            try:
                if operation == "resize":
                    size = kwargs.get("size", (800, 600))
                    self.resize_image(str(image_file), image_output, size)
                elif operation == "watermark":
                    watermark = kwargs.get("text", "Sample Watermark")
                    self.apply_watermark(str(image_file), watermark, image_output)
            except Exception as e:
                print(f"Error during batch {operation} for {image_file}: {e}")


def main():
    parser = argparse.ArgumentParser(description='Image Processing Suite')
    parser.add_argument('--mode', choices=['resize', 'watermark', 'optimize', 'convert', 'exif', 'batch'], 
                       default='optimize', help='Operation mode')
    parser.add_argument('--image', help='Path to the image file')
    parser.add_argument('--directory', help='Directory path for batch processing')
    parser.add_argument('--output', help='Output file or directory path')
    parser.add_argument('--size', nargs=2, metavar=('width', 'height'), type=int, help='New size for resizing')
    parser.add_argument('--text', help='Text for watermarking')
    parser.add_argument('--format', help='New format for conversion')
    parser.add_argument('--quality', type=int, default=85, help='Quality for optimization (default: 85)')

    args = parser.parse_args()
    processor = ImageProcessor()

    if args.mode == 'resize' and args.image and args.size:
        processor.resize_image(args.image, args.output, (args.size[0], args.size[1]))
    elif args.mode == 'watermark' and args.image and args.text:
        processor.apply_watermark(args.image, args.text, args.output)
    elif args.mode == 'optimize' and args.directory:
        processor.optimize_images(args.directory, args.quality)
    elif args.mode == 'convert' and args.image and args.format:
        processor.convert_format(args.image, args.format, args.output)
    elif args.mode == 'exif' and args.image:
        processor.show_exif_data(args.image)
    elif args.mode == 'batch' and args.directory:
        processor.batch_process(args.directory, args.mode, **vars(args))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()


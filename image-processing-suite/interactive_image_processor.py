#!/usr/bin/env python3
"""
Interactive Image Processing Suite
A comprehensive toolkit with user-friendly interactive interface for image manipulation.
"""

import os
import sys
import json
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional, Dict

try:
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageEnhance
    import cv2
    import piexif
    from tqdm import tqdm
    from colorama import init, Fore, Style, Back
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install -r requirements.txt")
    sys.exit(1)

# Initialize colorama for colored output
init(autoreset=True)

class InteractiveImageProcessor:
    """Interactive image processing toolkit with comprehensive features."""
    
    def __init__(self):
        self.supported_formats = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp", ".gif"]
        self.output_dir = Path("processed_images")
        self.output_dir.mkdir(exist_ok=True)
        
        # Social media dimensions presets
        self.social_presets = {
            "instagram_square": (1080, 1080),
            "instagram_story": (1080, 1920),
            "facebook_cover": (820, 312),
            "twitter_header": (1500, 500),
            "youtube_thumbnail": (1280, 720),
            "linkedin_banner": (1584, 396)
        }
    
    def print_banner(self):
        """Display welcome banner."""
        print(f"\n{Fore.CYAN}{'='*60}")
        print(f"{Fore.CYAN}📸 INTERACTIVE IMAGE PROCESSING SUITE 📸")
        print(f"{Fore.CYAN}{'='*60}")
        print(f"{Fore.YELLOW}Your one-stop solution for image manipulation!")
        print(f"{Fore.GREEN}Output directory: {self.output_dir.absolute()}")
        print(f"{Fore.CYAN}{'='*60}\n")
    
    def get_image_info(self, image_path: str) -> Dict:
        """Get comprehensive image information."""
        try:
            with Image.open(image_path) as img:
                info = {
                    'filename': Path(image_path).name,
                    'format': img.format,
                    'mode': img.mode,
                    'size': img.size,
                    'width': img.width,
                    'height': img.height,
                    'file_size': os.path.getsize(image_path)
                }
                
                # Try to get EXIF data
                try:
                    exif_data = piexif.load(image_path)
                    info['has_exif'] = True
                    info['exif_tags'] = len(exif_data.get('0th', {}))
                except:
                    info['has_exif'] = False
                    info['exif_tags'] = 0
                
                return info
        except Exception as e:
            return {'error': str(e)}
    
    def display_image_info(self, image_path: str):
        """Display detailed image information."""
        info = self.get_image_info(image_path)
        
        if 'error' in info:
            print(f"{Fore.RED}❌ Error reading image: {info['error']}")
            return
        
        print(f"\n{Fore.CYAN}📋 IMAGE INFORMATION")
        print(f"{Fore.CYAN}{'-'*40}")
        print(f"{Fore.YELLOW}📁 Filename: {info['filename']}")
        print(f"{Fore.YELLOW}📏 Dimensions: {info['width']} x {info['height']} pixels")
        print(f"{Fore.YELLOW}🎨 Format: {info['format']} ({info['mode']})")
        print(f"{Fore.YELLOW}💾 File Size: {info['file_size']:,} bytes ({info['file_size']/1024/1024:.2f} MB)")
        print(f"{Fore.YELLOW}📊 EXIF Data: {'Yes' if info['has_exif'] else 'No'} ({info['exif_tags']} tags)")
    
    def resize_image_interactive(self):
        """Interactive image resizing with presets."""
        print(f"\n{Fore.CYAN}🔧 IMAGE RESIZING")
        print(f"{Fore.CYAN}{'-'*30}")
        
        image_path = input(f"{Fore.YELLOW}Enter image path: ").strip()
        if not os.path.exists(image_path):
            print(f"{Fore.RED}❌ File not found!")
            return
        
        self.display_image_info(image_path)
        
        print(f"\n{Fore.GREEN}📐 Resize Options:")
        print("1. Custom dimensions")
        print("2. Social media presets")
        print("3. Percentage scaling")
        print("4. Maintain aspect ratio")
        
        choice = input(f"\n{Fore.YELLOW}Select option (1-4): ").strip()
        
        try:
            with Image.open(image_path) as img:
                if choice == "1":
                    width = int(input(f"{Fore.YELLOW}Enter width: "))
                    height = int(input(f"{Fore.YELLOW}Enter height: "))
                    new_size = (width, height)
                
                elif choice == "2":
                    print(f"\n{Fore.GREEN}📱 Social Media Presets:")
                    for i, (name, size) in enumerate(self.social_presets.items(), 1):
                        print(f"{i}. {name.replace('_', ' ').title()}: {size[0]}x{size[1]}")
                    
                    preset_choice = int(input(f"\n{Fore.YELLOW}Select preset: ")) - 1
                    preset_name = list(self.social_presets.keys())[preset_choice]
                    new_size = self.social_presets[preset_name]
                
                elif choice == "3":
                    percentage = float(input(f"{Fore.YELLOW}Enter percentage (e.g., 50 for 50%): "))
                    scale = percentage / 100
                    new_size = (int(img.width * scale), int(img.height * scale))
                
                elif choice == "4":
                    max_width = int(input(f"{Fore.YELLOW}Enter maximum width: "))
                    ratio = max_width / img.width
                    new_size = (max_width, int(img.height * ratio))
                
                else:
                    print(f"{Fore.RED}❌ Invalid choice!")
                    return
                
                # Process image
                resized_img = img.resize(new_size, Image.LANCZOS)
                
                # Generate output filename
                original_name = Path(image_path).stem
                extension = Path(image_path).suffix
                output_path = self.output_dir / f"{original_name}_resized_{new_size[0]}x{new_size[1]}{extension}"
                
                resized_img.save(output_path, quality=95)
                
                print(f"{Fore.GREEN}✅ Resized image saved: {output_path}")
                print(f"{Fore.CYAN}📊 New dimensions: {new_size[0]} x {new_size[1]}")
                
        except Exception as e:
            print(f"{Fore.RED}❌ Error resizing image: {e}")
    
    def apply_watermark_interactive(self):
        """Interactive watermark application with advanced options."""
        print(f"\n{Fore.CYAN}🏷️  WATERMARK APPLICATION")
        print(f"{Fore.CYAN}{'-'*35}")
        
        image_path = input(f"{Fore.YELLOW}Enter image path: ").strip()
        if not os.path.exists(image_path):
            print(f"{Fore.RED}❌ File not found!")
            return
        
        self.display_image_info(image_path)
        
        watermark_text = input(f"{Fore.YELLOW}Enter watermark text: ").strip()
        
        print(f"\n{Fore.GREEN}📍 Position Options:")
        positions = {
            "1": "bottom-right",
            "2": "bottom-left", 
            "3": "top-right",
            "4": "top-left",
            "5": "center"
        }
        
        for key, value in positions.items():
            print(f"{key}. {value.replace('-', ' ').title()}")
        
        pos_choice = input(f"\n{Fore.YELLOW}Select position (1-5): ").strip()
        position = positions.get(pos_choice, "bottom-right")
        
        # Opacity selection
        opacity = int(input(f"{Fore.YELLOW}Enter opacity (0-255, default 128): ") or "128")
        
        try:
            with Image.open(image_path) as img:
                # Create watermark
                watermark_layer = Image.new('RGBA', img.size, (255, 255, 255, 0))
                draw = ImageDraw.Draw(watermark_layer)
                
                # Try to use a better font
                try:
                    font_size = max(img.width // 30, 20)  # Dynamic font size
                    font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
                except:
                    font = ImageFont.load_default()
                
                # Get text dimensions
                text_bbox = draw.textbbox((0, 0), watermark_text, font=font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
                
                # Calculate position
                margin = 20
                if position == "bottom-right":
                    x = img.width - text_width - margin
                    y = img.height - text_height - margin
                elif position == "bottom-left":
                    x = margin
                    y = img.height - text_height - margin
                elif position == "top-right":
                    x = img.width - text_width - margin
                    y = margin
                elif position == "top-left":
                    x = margin
                    y = margin
                else:  # center
                    x = (img.width - text_width) // 2
                    y = (img.height - text_height) // 2
                
                # Draw watermark
                draw.text((x, y), watermark_text, font=font, fill=(255, 255, 255, opacity))
                
                # Composite images
                if img.mode != 'RGBA':
                    img = img.convert('RGBA')
                
                watermarked = Image.alpha_composite(img, watermark_layer)
                
                # Convert back to RGB if needed
                if watermarked.mode == 'RGBA':
                    final_img = Image.new('RGB', watermarked.size, (255, 255, 255))
                    final_img.paste(watermarked, mask=watermarked.split()[-1])
                else:
                    final_img = watermarked
                
                # Save result
                original_name = Path(image_path).stem
                extension = Path(image_path).suffix
                output_path = self.output_dir / f"{original_name}_watermarked{extension}"
                
                final_img.save(output_path, quality=95)
                
                print(f"{Fore.GREEN}✅ Watermarked image saved: {output_path}")
                print(f"{Fore.CYAN}📍 Position: {position.replace('-', ' ').title()}")
                print(f"{Fore.CYAN}🎨 Opacity: {opacity}/255")
                
        except Exception as e:
            print(f"{Fore.RED}❌ Error applying watermark: {e}")
    
    def batch_optimize_interactive(self):
        """Interactive batch optimization with quality options."""
        print(f"\n{Fore.CYAN}⚡ BATCH OPTIMIZATION")
        print(f"{Fore.CYAN}{'-'*30}")
        
        directory = input(f"{Fore.YELLOW}Enter directory path: ").strip()
        if not os.path.exists(directory):
            print(f"{Fore.RED}❌ Directory not found!")
            return
        
        # Find all supported images
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(Path(directory).glob(f"*{ext}"))
            image_files.extend(Path(directory).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"{Fore.RED}❌ No supported images found!")
            return
        
        print(f"{Fore.GREEN}📁 Found {len(image_files)} images")
        
        # Quality options
        print(f"\n{Fore.GREEN}🎯 Quality Presets:")
        quality_presets = {
            "1": ("Web Optimized", 75),
            "2": ("High Quality", 90),
            "3": ("Maximum Quality", 95),
            "4": ("Custom", None)
        }
        
        for key, (name, quality) in quality_presets.items():
            if quality:
                print(f"{key}. {name} (Quality: {quality})")
            else:
                print(f"{key}. {name}")
        
        choice = input(f"\n{Fore.YELLOW}Select quality preset (1-4): ").strip()
        
        if choice == "4":
            quality = int(input(f"{Fore.YELLOW}Enter custom quality (1-100): "))
        else:
            quality = quality_presets.get(choice, ("High Quality", 90))[1]
        
        # Backup option
        backup = input(f"{Fore.YELLOW}Create backup of originals? (y/N): ").strip().lower() == 'y'
        
        if backup:
            backup_dir = Path(directory) / "backup_originals"
            backup_dir.mkdir(exist_ok=True)
        
        # Process images
        total_original_size = 0
        total_optimized_size = 0
        
        print(f"\n{Fore.CYAN}🔄 Processing images...")
        
        for image_file in tqdm(image_files, desc="Optimizing"):
            try:
                original_size = image_file.stat().st_size
                total_original_size += original_size
                
                # Backup if requested
                if backup:
                    shutil.copy2(image_file, backup_dir / image_file.name)
                
                # Optimize image
                with Image.open(image_file) as img:
                    # Convert to RGB if necessary
                    if img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    
                    # Save optimized version
                    img.save(image_file, optimize=True, quality=quality)
                
                optimized_size = image_file.stat().st_size
                total_optimized_size += optimized_size
                
            except Exception as e:
                print(f"\n{Fore.RED}❌ Error processing {image_file.name}: {e}")
        
        # Show results
        savings = total_original_size - total_optimized_size
        savings_percent = (savings / total_original_size) * 100 if total_original_size > 0 else 0
        
        print(f"\n{Fore.GREEN}🎉 OPTIMIZATION COMPLETE!")
        print(f"{Fore.CYAN}📊 Statistics:")
        print(f"   📁 Images processed: {len(image_files)}")
        print(f"   📉 Original size: {total_original_size/1024/1024:.2f} MB")
        print(f"   📈 Optimized size: {total_optimized_size/1024/1024:.2f} MB")
        print(f"   💾 Space saved: {savings/1024/1024:.2f} MB ({savings_percent:.1f}%)")
        
        if backup:
            print(f"   🔒 Backups saved in: {backup_dir}")
    
    def convert_format_interactive(self):
        """Interactive format conversion with batch support."""
        print(f"\n{Fore.CYAN}🔄 FORMAT CONVERSION")
        print(f"{Fore.CYAN}{'-'*30}")
        
        print("1. Single image conversion")
        print("2. Batch conversion")
        
        mode = input(f"\n{Fore.YELLOW}Select mode (1-2): ").strip()
        
        # Supported output formats
        output_formats = {
            "1": ("JPEG", "jpg"),
            "2": ("PNG", "png"),
            "3": ("WebP", "webp"),
            "4": ("BMP", "bmp"),
            "5": ("TIFF", "tiff")
        }
        
        print(f"\n{Fore.GREEN}📄 Output Formats:")
        for key, (name, ext) in output_formats.items():
            print(f"{key}. {name} (.{ext})")
        
        format_choice = input(f"\n{Fore.YELLOW}Select output format (1-5): ").strip()
        output_format_name, output_ext = output_formats.get(format_choice, ("JPEG", "jpg"))
        
        if mode == "1":
            # Single image conversion
            image_path = input(f"{Fore.YELLOW}Enter image path: ").strip()
            if not os.path.exists(image_path):
                print(f"{Fore.RED}❌ File not found!")
                return
            
            self._convert_single_image(image_path, output_format_name, output_ext)
            
        elif mode == "2":
            # Batch conversion
            directory = input(f"{Fore.YELLOW}Enter directory path: ").strip()
            if not os.path.exists(directory):
                print(f"{Fore.RED}❌ Directory not found!")
                return
            
            self._convert_batch_images(directory, output_format_name, output_ext)
    
    def _convert_single_image(self, image_path: str, output_format: str, output_ext: str):
        """Convert a single image to specified format."""
        try:
            with Image.open(image_path) as img:
                # Convert to RGB if necessary for JPEG
                if output_format == "JPEG" and img.mode in ("RGBA", "P"):
                    img = img.convert("RGB")
                
                # Generate output path
                original_name = Path(image_path).stem
                output_path = self.output_dir / f"{original_name}_converted.{output_ext}"
                
                # Save converted image
                img.save(output_path, format=output_format, quality=95 if output_format == "JPEG" else None)
                
                print(f"{Fore.GREEN}✅ Converted image saved: {output_path}")
                print(f"{Fore.CYAN}🔄 Format: {output_format}")
                
        except Exception as e:
            print(f"{Fore.RED}❌ Error converting image: {e}")
    
    def _convert_batch_images(self, directory: str, output_format: str, output_ext: str):
        """Convert all images in directory to specified format."""
        # Find all supported images
        image_files = []
        for ext in self.supported_formats:
            image_files.extend(Path(directory).glob(f"*{ext}"))
            image_files.extend(Path(directory).glob(f"*{ext.upper()}"))
        
        if not image_files:
            print(f"{Fore.RED}❌ No supported images found!")
            return
        
        print(f"{Fore.GREEN}📁 Found {len(image_files)} images")
        
        # Create output subdirectory
        output_subdir = self.output_dir / f"converted_to_{output_ext}"
        output_subdir.mkdir(exist_ok=True)
        
        success_count = 0
        
        for image_file in tqdm(image_files, desc="Converting"):
            try:
                with Image.open(image_file) as img:
                    # Convert to RGB if necessary for JPEG
                    if output_format == "JPEG" and img.mode in ("RGBA", "P"):
                        img = img.convert("RGB")
                    
                    # Generate output path
                    output_path = output_subdir / f"{image_file.stem}.{output_ext}"
                    
                    # Save converted image
                    img.save(output_path, format=output_format, quality=95 if output_format == "JPEG" else None)
                    success_count += 1
                    
            except Exception as e:
                print(f"\n{Fore.RED}❌ Error converting {image_file.name}: {e}")
        
        print(f"\n{Fore.GREEN}🎉 CONVERSION COMPLETE!")
        print(f"{Fore.CYAN}📊 Successfully converted: {success_count}/{len(image_files)} images")
        print(f"{Fore.CYAN}📁 Output directory: {output_subdir}")
    
    def show_exif_interactive(self):
        """Interactive EXIF data viewer with export option."""
        print(f"\n{Fore.CYAN}📊 EXIF DATA VIEWER")
        print(f"{Fore.CYAN}{'-'*30}")
        
        image_path = input(f"{Fore.YELLOW}Enter image path: ").strip()
        if not os.path.exists(image_path):
            print(f"{Fore.RED}❌ File not found!")
            return
        
        try:
            exif_data = piexif.load(image_path)
            
            print(f"\n{Fore.GREEN}📋 EXIF Data for: {Path(image_path).name}")
            print(f"{Fore.GREEN}{'-'*50}")
            
            all_exif = {}
            
            for ifd_name in ["0th", "Exif", "GPS", "1st"]:
                if ifd_name in exif_data and exif_data[ifd_name]:
                    print(f"\n{Fore.CYAN}📂 {ifd_name} IFD:")
                    
                    for tag, value in exif_data[ifd_name].items():
                        try:
                            tag_name = piexif.TAGS[ifd_name][tag]["name"]
                        except KeyError:
                            tag_name = str(tag)
                        
                        # Format value for display
                        if isinstance(value, bytes):
                            try:
                                value = value.decode('utf-8')
                            except:
                                value = str(value)
                        
                        print(f"   {Fore.YELLOW}{tag_name}: {value}")
                        all_exif[f"{ifd_name}_{tag_name}"] = str(value)
            
            # Export option
            export = input(f"\n{Fore.YELLOW}Export EXIF data to JSON? (y/N): ").strip().lower() == 'y'
            
            if export:
                exif_filename = self.output_dir / f"{Path(image_path).stem}_exif.json"
                with open(exif_filename, 'w') as f:
                    json.dump(all_exif, f, indent=2)
                print(f"{Fore.GREEN}✅ EXIF data exported: {exif_filename}")
                
        except Exception as e:
            print(f"{Fore.RED}❌ Error reading EXIF data: {e}")
    
    def main_menu(self):
        """Display main interactive menu."""
        while True:
            print(f"\n{Fore.CYAN}🎛️  MAIN MENU")
            print(f"{Fore.CYAN}{'-'*20}")
            print(f"{Fore.GREEN}1. 📏 Resize Images")
            print(f"{Fore.GREEN}2. 🏷️  Apply Watermark")
            print(f"{Fore.GREEN}3. ⚡ Batch Optimize")
            print(f"{Fore.GREEN}4. 🔄 Convert Format")
            print(f"{Fore.GREEN}5. 📊 View EXIF Data")
            print(f"{Fore.GREEN}6. 📁 Show Output Directory")
            print(f"{Fore.RED}7. ❌ Exit")
            
            choice = input(f"\n{Fore.YELLOW}Select option (1-7): ").strip()
            
            if choice == "1":
                self.resize_image_interactive()
            elif choice == "2":
                self.apply_watermark_interactive()
            elif choice == "3":
                self.batch_optimize_interactive()
            elif choice == "4":
                self.convert_format_interactive()
            elif choice == "5":
                self.show_exif_interactive()
            elif choice == "6":
                print(f"\n{Fore.CYAN}📁 Output Directory: {self.output_dir.absolute()}")
                if os.path.exists(self.output_dir):
                    files = list(self.output_dir.iterdir())
                    print(f"{Fore.CYAN}📊 Files in output directory: {len(files)}")
                    if files:
                        print(f"{Fore.YELLOW}Recent files:")
                        for file in sorted(files, key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
                            print(f"   • {file.name}")
            elif choice == "7":
                print(f"\n{Fore.GREEN}👋 Thank you for using the Image Processing Suite!")
                print(f"{Fore.CYAN}🎨 Keep creating amazing visuals!")
                break
            else:
                print(f"{Fore.RED}❌ Invalid choice. Please try again.")

def main():
    """Main function to run the interactive image processor."""
    processor = InteractiveImageProcessor()
    processor.print_banner()
    processor.main_menu()

if __name__ == "__main__":
    main()

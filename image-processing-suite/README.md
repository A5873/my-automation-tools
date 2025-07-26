# 📸 Image Processing Suite

A versatile toolkit for image manipulation and optimization in Python. Perfect for bulk resizing, watermarking, format conversion, and image metadata handling.

## ✨ Features

### 🖼️ Core Functionality
- **Resize Images**: Change dimensions of images with anti-aliasing.
- **Apply Watermark**: Embed text watermarks for branding or copyright protection.
- **Optimize Images**: Compress images to reduce file size while maintaining quality.
- **Convert Formats**: Transform images between popular formats (JPEG, PNG, WEBP, etc.).
- **EXIF Data Management**: Read and display metadata from image files.
- **Batch Processing**: Apply operations to entire directories in bulk.

### 📈 Advanced Features
- **Interactive Mode**: Choose operations through a user-friendly menu.
- **Command-Line Interface**: Automate operations with CLI commands.
- **Progress Tracking**: Visual feedback with progress bars for batch operations.

## 📦 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Optional: Configure Environment (if needed)

For improved performance on certain operations:

```bash
# Install system-level libraries (e.g., libjpeg, libpng)
```

## 🚀 Usage

### Interactive Mode (Recommended)

```bash
python image_processing_suite.py --mode interactive
```

This launches a menu allowing you to:
- Resize individual images or entire directories
- Apply watermarks to multiple images
- Optimize images for web use
- Convert between formats
- Examine EXIF metadata

### Command Line Mode

#### Resize an Image

```bash
python image_processing_suite.py --mode resize --image path/to/image.jpg --output path/to/resized_image.jpg --size 800 600
```

#### Add a Watermark

```bash
python image_processing_suite.py --mode watermark --image path/to/image.jpg --output path/to/watermarked_image.jpg --text "My Watermark"
```

#### Optimize an Entire Directory

```bash
python image_processing_suite.py --mode optimize --directory path/to/images --quality 85
```

#### Convert Image Format

```bash
python image_processing_suite.py --mode convert --image path/to/image.jpg --output path/to/image.png --format png
```

#### Display EXIF Metadata

```bash
python image_processing_suite.py --mode exif --image path/to/image.jpg
```

#### Batch Process a Directory

For resizing all images to 800x600:

```bash
python image_processing_suite.py --mode batch --directory path/to/images --size 800 600
```

## 📁 Output Structure

Processed images are saved alongside originals with enhanced naming:

```
processed_awesome_photo.jpg
```

## 🛡️ Best Practices

- Always back up original images before processing
- Test a small batch before running large operations
- Choose compression settings carefully to balance quality vs. file size

## 📄 Legal Note

This tool is for personal and educational use. Ensure you respect copyright laws, especially with watermarking.

## 🤝 Contributions

Contributions are welcome! Potential improvements include adding:
- More advanced watermark designs
- Integration with social media platforms for direct uploads
- Enhanced error handling and reporting
- Support for additional image formats

---

**Enhance Your Imagery! 🌟**


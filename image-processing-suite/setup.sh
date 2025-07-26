#!/bin/bash

# Image Processing Suite Setup Script

echo "📸 Setting up Image Processing Suite..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3 and try again."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "image_processor_env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv image_processor_env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source image_processor_env/bin/activate

# Install requirements
echo "📥 Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Create output directories
echo "📁 Creating output directories..."
mkdir -p processed_images
mkdir -p sample_images

# Make scripts executable
chmod +x image_processing_suite.py
chmod +x interactive_image_processor.py

echo "✅ Setup complete!"
echo ""
echo "🚀 To get started:"
echo "   source image_processor_env/bin/activate"
echo "   python interactive_image_processor.py    # Interactive mode"
echo "   python image_processing_suite.py --help  # CLI mode"
echo ""
echo "📚 Check the README.md for detailed usage instructions."

#!/bin/bash

# Web Scraper Toolkit Setup Script

echo "🕷️  Setting up Web Scraper Toolkit..."

# Check if Python 3 is installed
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 is required but not installed."
    echo "Please install Python 3 and try again."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "web_scraper_env" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv web_scraper_env
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source web_scraper_env/bin/activate

# Install requirements
echo "📥 Installing requirements..."
pip install --upgrade pip
pip install -r requirements.txt

# Create data directory
echo "📁 Creating data directory..."
mkdir -p scraped_data

# Make the main script executable
chmod +x web_scraper_toolkit.py

echo "✅ Setup complete!"
echo ""
echo "🚀 To get started:"
echo "   source web_scraper_env/bin/activate"
echo "   python web_scraper_toolkit.py"
echo ""
echo "📚 Or check the README.md for more usage examples."

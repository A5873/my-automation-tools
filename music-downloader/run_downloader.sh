#!/bin/bash

# Music Downloader Setup & Run Script
# This script activates the virtual environment and runs the downloader

echo "🎵 Starting Music Downloader..."

# Check if virtual environment exists
if [ ! -d "music_downloader_env" ]; then
    echo "❌ Virtual environment not found. Please run setup first:"
    echo "   python3 -m venv music_downloader_env"
    echo "   source music_downloader_env/bin/activate"
    echo "   pip install -r requirements.txt"
    exit 1
fi

# Activate virtual environment
source music_downloader_env/bin/activate

# Check if dependencies are installed
python -c "import requests, yt_dlp, mutagen" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Dependencies not installed. Installing now..."
    pip install -r requirements.txt
fi

# Run the enhanced music downloader
echo "🚀 Launching Enhanced Music Downloader..."
python enhanced_music_downloader.py "$@"

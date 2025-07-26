#!/bin/bash

# DevOps Automation Suite - Setup Script
# Enterprise-grade setup automation

set -e

echo "🚀 DevOps Automation Suite Setup"
echo "=================================="

# Check Python version
echo "📋 Checking Python version..."
python3 --version

# Create virtual environment
echo "🐍 Creating virtual environment..."
python3 -m venv venv

# Activate virtual environment
echo "⚡ Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo "📚 Installing Python dependencies..."
pip install -r requirements.txt

# Copy environment file
if [ ! -f .env ]; then
    echo "⚙️ Creating environment configuration..."
    cp .env.example .env
    echo "✏️ Please edit .env file with your configuration"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p logs
mkdir -p data
mkdir -p reports

# Check Docker
echo "🐳 Checking Docker..."
if command -v docker &> /dev/null; then
    echo "✅ Docker is installed"
    
    # Start services
    echo "🚀 Starting services with Docker Compose..."
    docker-compose up -d
    
    # Wait for services
    echo "⏳ Waiting for services to start..."
    sleep 10
    
else
    echo "⚠️ Docker not found. Please install Docker for full functionality."
fi

# Initialize database
echo "🗄️ Initializing database..."
python main.py setup

# Check system status
echo "📊 Checking system status..."
python main.py status

echo ""
echo "🎉 Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Edit .env file with your configuration"
echo "2. Run: python main.py status"
echo "3. Start web server: python main.py web"
echo "4. Access dashboard: http://localhost:8000"
echo ""

# Alex's Automation Tools 🚀

A comprehensive collection of Python automation tools for various data processing, content management, and productivity tasks.

## ✨ Complete Tools Suite

### 📄 Document Processor **[COMPLETE]**
A full-featured document manipulation and conversion toolkit:
- **Format Support**: PDF, DOCX, TXT, Markdown, HTML, Excel, CSV
- **Document Analysis**: Word count, readability scores, metadata extraction
- **Batch Processing**: Convert entire directories at once
- **Document Merging**: Combine multiple documents into one
- **Template Generation**: Jinja2-powered document creation
- **Report Generation**: Comprehensive document analysis reports

### 🕷️ Web Scraper Toolkit **[COMPLETE]**
A powerful, production-ready web scraping suite:
- **Price Monitoring**: Amazon & e-commerce price tracking with alerts
- **Job Aggregation**: Indeed & RemoteOK job listings collection
- **News Scraping**: Hacker News & Reddit content aggregation
- **Interactive & CLI Modes**: User-friendly interfaces
- **Respectful Scraping**: Built-in delays and user-agent rotation
- **Data Export**: JSON, CSV, and analysis-ready formats

### 🖼️ Image Processing Suite **[COMPLETE]**
Advanced image manipulation and processing toolkit:
- **Batch Processing**: Process multiple images simultaneously
- **Filter Library**: Blur, sharpen, brightness, contrast, and artistic effects
- **Format Conversion**: Convert between various image formats
- **Resize & Transform**: Smart resizing with aspect ratio preservation
- **Interactive Mode**: User-friendly processing interface
- **Watermarking**: Add text and image watermarks

### 🎵 Music Downloader **[READY - NEEDS SETUP]**
Comprehensive music downloading with enhanced features:
- YouTube music downloading with quality selection
- Metadata extraction and automatic tagging
- Batch downloading capabilities
- Progress tracking and error handling
- Multiple format support

### 📝 Lyrics Fetcher **[READY - NEEDS SETUP]**
Intelligent lyrics fetching and management:
- Multi-source lyrics searching
- Interactive mode for batch queries
- Album artwork integration
- Various music service support

## 📁 Project Structure

```
alex-automation-tools/
├── README.md                    # This file
├── LICENSE                      # MIT License
├── document-processor/          # ✅ Document conversion & analysis
│   ├── README.md
│   ├── document_processor.py
│   ├── setup.py
│   ├── requirements.txt
│   └── document_templates/
├── web-scraper-toolkit/         # ✅ Web scraping automation
│   ├── README.md
│   ├── web_scraper_toolkit.py
│   ├── examples.py
│   ├── requirements.txt
│   └── setup.sh
├── image-processing-suite/      # ✅ Image manipulation toolkit
│   ├── README.md
│   ├── image_processing_suite.py
│   ├── interactive_image_processor.py
│   ├── requirements.txt
│   └── setup.sh
├── music-downloader/            # ⚙️ Music downloading tools
│   ├── README.md
│   ├── enhanced_music_downloader.py
│   ├── music_discovery_downloader.py
│   ├── requirements.txt
│   └── test_download.py
└── lyrics-fetcher/              # ⚙️ Lyrics fetching tools
    ├── README.md
    ├── lyrics_fetcher.py
    ├── interactive_lyrics_fetcher.py
    └── requirements.txt
```

## 🛠️ Quick Setup

### For Complete Tools ✅

1. **Clone the repository:**
   ```bash
   git clone https://github.com/A5873/my-automation-tools
   cd my-automation-tools
   ```

2. **Choose your tool and set up:**

   **Document Processor:**
   ```bash
   cd document-processor
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   python document_processor.py --help
   ```

   **Web Scraper Toolkit:**
   ```bash
   cd web-scraper-toolkit
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   python web_scraper_toolkit.py  # Interactive mode
   ```

   **Image Processing Suite:**
   ```bash
   cd image-processing-suite
   chmod +x setup.sh
   ./setup.sh  # Automated setup
   python interactive_image_processor.py
   ```

### For Other Tools ⚙️

3. Navigate to the desired tool directory and follow the setup instructions in each tool's README

## 📋 Requirements

- Python 3.7+
- pip (Python package installer)
- Individual tool requirements are listed in their respective directories

## 🤝 Contributing

Feel free to contribute to any of these tools by:
1. Forking the repository
2. Creating a feature branch
3. Making your changes
4. Submitting a pull request

## 📄 License

This project is open source. See individual tool directories for specific licensing information.

## 👤 Author

**Alex** - [GitHub Profile](https://github.com/A5873) 

Feel free to connect and explore more projects!

---

Each tool in this collection is designed to be standalone and can be used independently. Check the individual README files for detailed usage instructions and examples.

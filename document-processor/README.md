# Document Processor Suite 📄

A comprehensive Python toolkit for document manipulation, conversion, and automation. Process and convert between multiple document formats with ease!

## Features

### 🔄 Document Conversion
- **PDF** ↔ TXT, DOCX
- **DOCX** ↔ PDF, TXT, HTML, Markdown
- **TXT** ↔ DOCX, PDF, HTML
- **Markdown** ↔ HTML, DOCX, PDF
- **HTML** ↔ TXT, Markdown, DOCX
- **Excel** ↔ CSV, TXT

### 📊 Document Analysis
- Word count, character count, paragraph analysis
- Readability scores (Flesch Reading Ease, Flesch-Kincaid Grade)
- Reading time estimation
- File metadata extraction

### 📋 Document Operations
- **Merge** multiple documents into one
- **Batch convert** entire directories
- **Generate reports** with comprehensive analysis
- **Template-based** document generation using Jinja2

### 📈 Supported Formats
**Input:** `.docx`, `.pdf`, `.txt`, `.md`, `.html`, `.xlsx`, `.csv`
**Output:** `.docx`, `.pdf`, `.txt`, `.md`, `.html`, `.xlsx`, `.csv`

## Installation

1. **Clone or download this tool**
2. **Create virtual environment:**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Convert a Document
```bash
# Convert PDF to Word
python document_processor.py --mode convert --input document.pdf --format docx

# Convert Word to Markdown
python document_processor.py --mode convert --input report.docx --format md
```

### Analyze a Document
```bash
python document_processor.py --mode analyze --input document.pdf
```

### Merge Documents
```bash
python document_processor.py --mode merge --files doc1.docx doc2.docx doc3.docx --output merged.docx --format docx
```

### Batch Convert Directory
```bash
python document_processor.py --mode batch --input /path/to/documents --format pdf
```

### Generate Analysis Report
```bash
python document_processor.py --mode report --files doc1.pdf doc2.docx doc3.txt --output analysis_report.html
```

## Usage Examples

### Python API
```python
from document_processor import DocumentProcessor

processor = DocumentProcessor()

# Convert document
result = processor.convert_document('input.pdf', 'docx', 'output.docx')

# Analyze document
analysis = processor.analyze_document('document.pdf')
print(f"Word count: {analysis['word_count']}")
print(f"Reading time: {analysis['readability']['reading_time_minutes']:.1f} minutes")

# Merge documents
merged = processor.merge_documents(['doc1.docx', 'doc2.docx'], 'merged.docx')

# Batch conversion
results = processor.batch_convert_documents('/path/to/docs', 'pdf')
```

### Command Line Interface
```bash
# Show help
python document_processor.py --help

# Convert single file
python document_processor.py --mode convert --input sample.docx --format pdf --output converted.pdf

# Analyze multiple documents and generate report
python document_processor.py --mode report --files *.pdf --output comprehensive_report.html
```

## Output Structure

The tool creates the following directories:
- `processed_documents/` - All converted and processed documents
- `document_templates/` - Template files for document generation

## Advanced Features

### Document Templates
Create Jinja2 templates for automated document generation:

```python
template_data = {
    'title': 'Monthly Report',
    'date': '2024-01-01',
    'content': 'Report content here...'
}

processor.generate_document_from_template(
    'template.txt', 
    template_data, 
    'generated_report.txt'
)
```

### Readability Analysis
Get detailed readability metrics:
- Flesch Reading Ease Score
- Flesch-Kincaid Grade Level
- Automated Readability Index
- Estimated reading time

### Batch Processing
Process entire directories with filtering:
```python
# Convert only PDF files in directory
results = processor.batch_convert_documents(
    '/path/to/docs', 
    'docx', 
    filter_extensions=['.pdf']
)
```

## Dependencies

Core libraries used:
- `python-docx` - Word document processing
- `PyPDF2` & `pdfplumber` - PDF processing
- `pandas` & `openpyxl` - Excel/CSV processing
- `markdown` & `beautifulsoup4` - HTML/Markdown processing
- `reportlab` & `fpdf2` - PDF generation
- `jinja2` - Template processing
- `textstat` - Readability analysis
- `tqdm` - Progress bars
- `colorama` - Colored output

## Error Handling

The tool includes comprehensive error handling:
- Missing file validation
- Format compatibility checking
- Graceful failure with detailed error messages
- Processing statistics tracking

## License

MIT License - see LICENSE file for details.

## Contributing

Feel free to submit issues, feature requests, or pull requests!

## Troubleshooting

### Common Issues

1. **"Missing required package" error**
   - Make sure virtual environment is activated
   - Run `pip install -r requirements.txt`

2. **PDF conversion issues**
   - Some PDFs may be encrypted or have complex layouts
   - Try using different PDF processing methods

3. **Memory issues with large files**
   - Process files in smaller batches
   - Consider using streaming for very large documents

### Platform-Specific Notes

- **macOS:** Some dependencies may require Xcode command line tools
- **Windows:** Ensure proper encoding handling for international characters
- **Linux:** May need additional system packages for certain features

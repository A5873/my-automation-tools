# 📝 Lyrics Fetcher

An intelligent Python tool for fetching song lyrics and album artwork using the Genius API. Features both command-line and interactive modes for flexible usage.

## ✨ Features

- **Multiple Search Options**: Search by song title, artist, or both
- **Interactive Mode**: User-friendly interface with multiple search results
- **Cover Art Download**: Optional album artwork downloading
- **Clean Output**: Well-formatted text files with metadata
- **Flexible Naming**: Auto-generated or custom filenames
- **API Integration**: Uses official Genius API for accurate results

## 📦 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Get Genius API Token

1. Visit [Genius API Clients](https://genius.com/api-clients)
2. Create a new API client
3. Copy your access token

### 3. Set Environment Variable

```bash
# Add to your shell profile (.bashrc, .zshrc, etc.)
export GENIUS_ACCESS_TOKEN="your_token_here"

# Or set temporarily
export GENIUS_ACCESS_TOKEN="your_token_here"
```

## 🚀 Usage

### Interactive Mode (Recommended)

```bash
python interactive_lyrics_fetcher.py
```

Features:
- Search with multiple results
- Select from found songs
- Option to download cover art
- Custom filename support

### Command Line Mode

```bash
# Basic usage
python lyrics_fetcher.py "Artist Song Title"

# With custom output file
python lyrics_fetcher.py "Tupac Dear Mama" -o dear-mama.txt

# With inline API token
python lyrics_fetcher.py "Song Title" -t your_token_here
```

## 🔧 Options

### Command Line Arguments

- `query`: Search query (required)
- `-o, --output`: Custom output filename
- `-t, --token`: Genius API token (if not set as environment variable)

### Interactive Features

- **Multiple Results**: Choose from up to 5 search results
- **Cover Art**: Optional album artwork download
- **Custom Names**: Set your own filenames
- **Metadata**: Includes artist, title, release date, and source URL

## 📁 Output Format

Lyrics are saved as text files with the following format:

```
Title: Song Title
Artist: Artist Name
Source: https://genius.com/...
--------------------------------------------------

[Verse 1]
Lyrics content here...

[Chorus]
More lyrics...
```

## 🔍 Requirements

Listed in `requirements.txt`:
- `requests>=2.25.1`
- `beautifulsoup4>=4.9.3`
- `lxml>=4.6.0`

## 🛠️ Troubleshooting

### Common Issues

1. **No API Token**: Make sure your `GENIUS_ACCESS_TOKEN` is set correctly
2. **No Results**: Try different search terms or check spelling
3. **Scraping Issues**: Some lyrics pages may have different formatting

### Error Messages

- `"No API token available"`: Set your Genius API token
- `"No songs found"`: Try broader search terms
- `"Could not find lyrics"`: Song page format may have changed

## 🎯 Tips for Best Results

1. **Search Format**: Use "Artist Song Title" format for best results
2. **Multiple Words**: Use exact song titles when possible
3. **Popular Songs**: More popular songs have better metadata
4. **Cover Art**: Enable cover art download for complete music library organization

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional metadata extraction
- Better lyrics parsing
- Support for other lyrics sources
- Batch processing features

## 📄 Legal Note

This tool is for personal use only. Please respect copyright laws and the terms of service of lyrics providers. The tool fetches publicly available information and should be used responsibly.

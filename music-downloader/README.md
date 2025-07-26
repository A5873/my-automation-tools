# 🎵 Music Downloader

A robust Python-based tool for discovering and downloading music from various legal sources, featuring:

- **Multiple Source Support**: Includes Internet Archive, YouTube (user-content), and more.
- **Advanced Features**: Metadata tagging, playlist generation, album artwork downloading.
- **Interactive and CLI modes**: User-friendly interactive mode and CLI support.

## 📦 Setup

1. **Virtual Environment & Installation**:

   ```bash
   python3 -m venv music_downloader_env
   source music_downloader_env/bin/activate
   pip install -r requirements.txt
   ```

2. **Running the Downloader**:
   
   Use the shell script for seamless setup and execution:

   ```bash
   ./run_downloader.sh
   ```

## 🔧 Usage

Run the downloader in interactive mode or use the CLI options:

- Interactive Mode:

  ```bash
  python enhanced_music_downloader.py
  ```

- CLI Options:

  ```bash
  python enhanced_music_downloader.py --artist "Artist Name" --discography
  ```

## 🗂️ Requirements

Listed in `requirements.txt`:

- `requests`
- `yt-dlp`
- `mutagen`
- `beautifulsoup4`

For detailed usage, please refer to the inline comments in the Python scripts.

## 🚀 Features

- **Legal Music Discovery**: Search, download, and build a personal music library.
- **Metadata Tagging**: Automatically tag music files.
- **Playlist Creation**: Generate M3U playlists.

## 🤝 Contribution

Contributions are welcome! Please open an issue or submit a pull request.


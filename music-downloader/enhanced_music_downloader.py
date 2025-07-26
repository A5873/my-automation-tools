#!/usr/bin/env python3
"""
Enhanced Music Discovery & Download Script
Comprehensive music discovery from multiple legal sources with advanced features.
"""

import requests
import json
import os
import re
import subprocess
import sys
from urllib.parse import quote, urlparse
from pathlib import Path
import argparse
from datetime import datetime
import time
from typing import List, Dict, Optional

try:
    import yt_dlp
except ImportError:
    print("yt-dlp not found. Install with: pip install yt-dlp")
    sys.exit(1)

try:
    from mutagen.mp3 import MP3
    from mutagen.id3 import ID3, TIT2, TPE1, TALB, TDRC
except ImportError:
    print("mutagen not found. Install with: pip install mutagen")
    sys.exit(1)

class EnhancedMusicDownloader:
    def __init__(self, download_dir="./music_library"):
        """
        Initialize the enhanced music discovery and download system.
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
        # Create organized directory structure
        (self.download_dir / "artists").mkdir(exist_ok=True)
        (self.download_dir / "albums").mkdir(exist_ok=True)
        (self.download_dir / "singles").mkdir(exist_ok=True)
        (self.download_dir / "playlists").mkdir(exist_ok=True)
        (self.download_dir / "metadata").mkdir(exist_ok=True)
        (self.download_dir / "covers").mkdir(exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
        })
        
        # Initialize yt-dlp
        self.ytdl_opts = {
            'format': 'bestaudio[ext=mp3]/bestaudio[ext=m4a]/bestaudio',
            'outtmpl': str(self.download_dir / 'temp' / '%(title)s.%(ext)s'),
            'writethumbnail': True,
            'writeinfojson': True,
            'extractaudio': True,
            'audioformat': 'mp3',
            'audioquality': '192K',
            'quiet': True,
            'no_warnings': True
        }
        
        # Music sources with their capabilities
        self.sources = {
            'internet_archive': {
                'name': 'Internet Archive',
                'search_url': 'https://archive.org/advancedsearch.php',
                'supports': ['songs', 'albums', 'discography', 'genre'],
                'legal': True,
                'enabled': True
            },
            'youtube': {
                'name': 'YouTube',
                'search_url': 'https://www.youtube.com/results',
                'supports': ['songs', 'albums', 'playlists'],
                'legal': 'user_content_only',  # Only user-uploaded content
                'enabled': True
            },
            'soundcloud': {
                'name': 'SoundCloud',
                'search_url': 'https://soundcloud.com/search',
                'supports': ['songs', 'albums', 'playlists'],
                'legal': 'user_content_only',
                'enabled': True
            }
        }
    
    def get_user_input(self) -> Dict:
        """Interactive user input for music discovery."""
        print("\n🎵 Enhanced Music Discovery & Download System")
        print("=" * 55)
        print("🎯 Features:")
        print("   • Discover entire discographies")
        print("   • Download from multiple legal sources") 
        print("   • Automatic metadata tagging")
        print("   • Playlist generation")
        print("   • Album artwork download")
        print()
        
        # Search type
        print("🔍 What would you like to search for?")
        print("1. Specific song")
        print("2. Artist discography (all albums)")
        print("3. Specific album")
        print("4. Genre exploration")
        print("5. Year/era browsing")
        print("6. Playlist/compilation")
        
        search_type = input("\nSelect option (1-6): ").strip()
        
        query = {}
        
        if search_type == "1":
            query['type'] = 'song'
            query['artist'] = input("🎤 Artist name: ").strip()
            query['song'] = input("🎵 Song title: ").strip()
            
        elif search_type == "2":
            query['type'] = 'discography'
            query['artist'] = input("🎤 Artist name: ").strip()
            query['include_features'] = input("Include featuring tracks? (y/N): ").lower() == 'y'
            query['include_compilations'] = input("Include greatest hits/compilations? (y/N): ").lower() == 'y'
            query['year_range'] = input("Year range (e.g., 1990-2000) or leave blank: ").strip()
            
        elif search_type == "3":
            query['type'] = 'album'
            query['artist'] = input("🎤 Artist name: ").strip()
            query['album'] = input("💿 Album title: ").strip()
            query['year'] = input("📅 Year (optional): ").strip()
            
        elif search_type == "4":
            query['type'] = 'genre'
            query['genre'] = input("🎶 Genre (e.g., jazz, hip-hop, rock): ").strip()
            query['subgenre'] = input("🎯 Subgenre (optional): ").strip()
            query['limit'] = int(input("📊 How many tracks? (default 50): ").strip() or "50")
            
        elif search_type == "5":
            query['type'] = 'year'
            query['year'] = input("📅 Year or decade (e.g., 1995, 1990s): ").strip()
            query['genre'] = input("🎶 Genre filter (optional): ").strip()
            query['limit'] = int(input("📊 How many tracks? (default 50): ").strip() or "50")
            
        elif search_type == "6":
            query['type'] = 'playlist'
            query['playlist_name'] = input("📋 Playlist/compilation name: ").strip()
            query['theme'] = input("🎯 Theme (e.g., 'best of', 'greatest hits'): ").strip()
        
        # Quality and format preferences
        print("\n⚙️  Download Preferences:")
        query['quality'] = input("🎧 Audio quality (high/medium/low): ").strip() or "high"
        query['format'] = input("📁 Format preference (mp3/flac/m4a): ").strip() or "mp3"
        query['organize_by'] = input("📂 Organize by (artist/album/genre/year): ").strip() or "artist"
        
        # Additional features
        print("\n🎨 Additional Features:")
        query['download_covers'] = input("🖼️  Download album artwork? (Y/n): ").lower() != 'n'
        query['create_playlist'] = input("📋 Create M3U playlist? (Y/n): ").lower() != 'n'
        query['tag_metadata'] = input("🏷️  Auto-tag with metadata? (Y/n): ").lower() != 'n'
        
        return query
    
    def search_internet_archive(self, query: Dict) -> List[Dict]:
        """Search Internet Archive for music."""
        print("🔍 Searching Internet Archive...")
        results = []
        
        try:
            # Build search term based on query type
            if query['type'] == 'song':
                search_term = f"{query['artist']} {query['song']}"
            elif query['type'] == 'discography':
                search_term = query['artist']
            elif query['type'] == 'album':
                search_term = f"{query['artist']} {query['album']}"
            elif query['type'] == 'genre':
                search_term = f"subject:{query['genre']}"
            else:
                search_term = query.get('artist', 'music')
            
            # Search parameters
            params = {
                'q': f'collection:(opensource_audio OR etree) AND ({search_term})',
                'fl': 'identifier,title,creator,date,description,downloads,format,subject',
                'rows': 100,
                'sort': 'downloads desc',
                'output': 'json'
            }
            
            response = self.session.get(self.sources['internet_archive']['search_url'], params=params)
            response.raise_for_status()
            
            data = response.json()
            
            for doc in data.get('response', {}).get('docs', []):
                # Extract and clean data
                title = doc.get('title', ['Unknown'])[0] if isinstance(doc.get('title'), list) else doc.get('title', 'Unknown')
                creator = doc.get('creator', ['Unknown'])[0] if isinstance(doc.get('creator'), list) else doc.get('creator', 'Unknown')
                
                result = {
                    'source': 'Internet Archive',
                    'title': title,
                    'artist': creator,
                    'year': self._extract_year(doc.get('date', ['Unknown'])),
                    'identifier': doc.get('identifier'),
                    'description': self._extract_description(doc.get('description', '')),
                    'downloads': doc.get('downloads', 0),
                    'formats': doc.get('format', []),
                    'subjects': doc.get('subject', []),
                    'url': f"https://archive.org/details/{doc.get('identifier')}"
                }
                results.append(result)
                
        except Exception as e:
            print(f"❌ Error searching Internet Archive: {e}")
        
        return results
    
    def search_youtube_music(self, query: Dict) -> List[Dict]:
        """Search YouTube for music (user-uploaded content only)."""
        print("🔍 Searching YouTube...")
        results = []
        
        try:
            # Build search query
            if query['type'] == 'song':
                search_query = f"{query['artist']} {query['song']} audio"
            elif query['type'] == 'album':
                search_query = f"{query['artist']} {query['album']} full album"
            elif query['type'] == 'discography':
                search_query = f"{query['artist']} discography"
            else:
                search_query = f"{query.get('artist', '')} music"
            
            # Use yt-dlp to search
            ydl_opts = {
                'quiet': True,
                'no_warnings': True,
                'extract_flat': True
            }
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                search_results = ydl.extract_info(
                    f"ytsearch{20}:{search_query}",
                    download=False
                )
                
                if search_results and 'entries' in search_results:
                    for entry in search_results['entries']:
                        # Filter for music content (basic heuristics)
                        title = entry.get('title', '')
                        duration = entry.get('duration', 0)
                        
                        # Skip very short (<30s) or very long (>20min) videos
                        if duration and (duration < 30 or duration > 1200):
                            continue
                        
                        # Construct YouTube URL from ID if webpage_url not available
                        video_id = entry.get('id', '')
                        webpage_url = entry.get('webpage_url', f'https://www.youtube.com/watch?v={video_id}' if video_id else '')
                        
                        result = {
                            'source': 'YouTube',
                            'title': title,
                            'artist': entry.get('uploader', 'Unknown'),
                            'duration': duration,
                            'url': webpage_url,
                            'id': video_id,
                            'view_count': entry.get('view_count', 0),
                            'upload_date': entry.get('upload_date', '')
                        }
                        results.append(result)
                        
        except Exception as e:
            print(f"❌ Error searching YouTube: {e}")
        
        return results
    
    def search_all_sources(self, query: Dict) -> List[Dict]:
        """Search all enabled sources."""
        all_results = []
        
        if self.sources['internet_archive']['enabled']:
            ia_results = self.search_internet_archive(query)
            all_results.extend(ia_results)
        
        if self.sources['youtube']['enabled']:
            yt_results = self.search_youtube_music(query)
            all_results.extend(yt_results)
        
        # Sort by relevance (downloads for IA, views for YouTube)
        all_results.sort(key=lambda x: x.get('downloads', x.get('view_count', 0)), reverse=True)
        
        return all_results
    
    def display_results(self, results: List[Dict], query: Dict) -> List[Dict]:
        """Display search results and let user select."""
        if not results:
            print("❌ No results found.")
            return []
        
        print(f"\n🎵 Found {len(results)} results:")
        print("=" * 70)
        
        # Group by source for better display
        ia_results = [r for r in results if r['source'] == 'Internet Archive']
        yt_results = [r for r in results if r['source'] == 'YouTube']
        
        displayed_results = []
        idx = 1
        
        if ia_results:
            print("📚 Internet Archive (Legal Free Music):")
            for result in ia_results[:10]:
                print(f"  {idx}. {result['title']}")
                print(f"     Artist: {result['artist']}")
                print(f"     Year: {result['year']}")
                print(f"     Downloads: {result['downloads']:,}")
                print()
                displayed_results.append(result)
                idx += 1
        
        if yt_results:
            print("🎥 YouTube (User Content - Use Responsibly):")
            for result in yt_results[:10]:
                print(f"  {idx}. {result['title']}")
                print(f"     Uploader: {result['artist']}")
                duration = result.get('duration', 0) or 0
                print(f"     Duration: {int(duration)//60}:{int(duration)%60:02d}")
                print(f"     Views: {result.get('view_count', 0):,}")
                print()
                displayed_results.append(result)
                idx += 1
        
        # User selection
        selected = []
        
        if query['type'] == 'discography':
            choice = input(f"\n📥 Download all {len(displayed_results)} items? (y/N): ").lower()
            if choice == 'y':
                selected = displayed_results
        else:
            print(f"\n📥 Select items to download:")
            print("   • Enter numbers (e.g., 1 3 5)")
            print("   • Enter range (e.g., 1-5)")
            print("   • Type 'all' for everything")
            print("   • Type 'done' when finished")
            
            while True:
                choice = input("\nSelection: ").strip().lower()
                
                if choice == 'done':
                    break
                elif choice == 'all':
                    selected = displayed_results
                    break
                elif '-' in choice:
                    try:
                        start, end = map(int, choice.split('-'))
                        for i in range(start-1, min(end, len(displayed_results))):
                            if displayed_results[i] not in selected:
                                selected.append(displayed_results[i])
                        print(f"✅ Added items {start}-{end}")
                    except ValueError:
                        print("❌ Invalid range format")
                else:
                    try:
                        indices = [int(x.strip()) for x in choice.split()]
                        for i in indices:
                            if 1 <= i <= len(displayed_results):
                                if displayed_results[i-1] not in selected:
                                    selected.append(displayed_results[i-1])
                                    print(f"✅ Added: {displayed_results[i-1]['title']}")
                    except ValueError:
                        print("❌ Please enter valid numbers")
        
        return selected
    
    def download_from_internet_archive(self, item: Dict) -> List[str]:
        """Download music from Internet Archive."""
        downloaded_files = []
        
        try:
            identifier = item['identifier']
            
            # Get detailed metadata
            metadata_url = f"https://archive.org/metadata/{identifier}"
            response = self.session.get(metadata_url)
            response.raise_for_status()
            
            metadata = response.json()
            files = metadata.get('files', [])
            
            # Filter for high-quality audio files
            audio_files = []
            preferred_formats = ['flac', 'mp3', 'm4a', 'ogg']
            
            for fmt in preferred_formats:
                fmt_files = [f for f in files if f.get('format', '').lower() == fmt]
                if fmt_files:
                    audio_files.extend(fmt_files)
                    break
            
            if not audio_files:
                print(f"⚠️  No suitable audio files found for {item['title']}")
                return downloaded_files
            
            # Create organized directory structure
            safe_artist = re.sub(r'[^\w\s-]', '', item['artist']).strip()
            safe_title = re.sub(r'[^\w\s-]', '', item['title']).strip()
            
            artist_dir = self.download_dir / "artists" / safe_artist
            artist_dir.mkdir(exist_ok=True)
            
            item_dir = artist_dir / safe_title
            item_dir.mkdir(exist_ok=True)
            
            print(f"📥 Downloading {len(audio_files)} files from '{item['title']}'...")
            
            for file_info in audio_files:
                filename = file_info['name']
                file_url = f"https://archive.org/download/{identifier}/{filename}"
                
                local_path = item_dir / filename
                
                if local_path.exists():
                    print(f"⏭️  Skipping {filename} (exists)")
                    downloaded_files.append(str(local_path))
                    continue
                
                print(f"📥 Downloading {filename}...")
                
                response = self.session.get(file_url, stream=True)
                response.raise_for_status()
                
                # Download with progress
                total_size = int(response.headers.get('content-length', 0))
                downloaded = 0
                
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\r   Progress: {progress:.1f}%", end='', flush=True)
                
                print(f"\n✅ Downloaded: {filename}")
                downloaded_files.append(str(local_path))
                
                # Be respectful to the server
                time.sleep(1)
            
            # Save comprehensive metadata
            self._save_metadata(item_dir, item, downloaded_files)
            
        except Exception as e:
            print(f"❌ Error downloading {item['title']}: {e}")
        
        return downloaded_files
    
    def download_from_youtube(self, item: Dict) -> List[str]:
        """Download from YouTube using yt-dlp."""
        downloaded_files = []
        
        try:
            # Create temp directory
            temp_dir = self.download_dir / "temp"
            temp_dir.mkdir(exist_ok=True)
            
            # Configure yt-dlp options
            ydl_opts = {
                'format': 'bestaudio[ext=m4a]/bestaudio',
                'outtmpl': str(temp_dir / '%(title)s.%(ext)s'),
                'extractaudio': True,
                'audioformat': 'mp3',
                'audioquality': '192',
                'writethumbnail': True,
                'writeinfojson': True
            }
            
            print(f"📥 Downloading from YouTube: {item['title']}")
            
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(item['url'], download=True)
                
                # Move and organize downloaded files
                if info:
                    title = info.get('title', 'Unknown')
                    uploader = info.get('uploader', 'Unknown')
                    
                    # Create organized structure
                    safe_uploader = re.sub(r'[^\w\s-]', '', uploader).strip()
                    safe_title = re.sub(r'[^\w\s-]', '', title).strip()
                    
                    artist_dir = self.download_dir / "artists" / safe_uploader
                    artist_dir.mkdir(exist_ok=True)
                    
                    # Move files from temp to organized location
                    # Look for all files matching the title pattern
                    for temp_file in temp_dir.glob("*"):
                        if temp_file.is_file() and (safe_title in temp_file.name or title in temp_file.name):
                            final_path = artist_dir / temp_file.name
                            if not final_path.exists():
                                temp_file.rename(final_path)
                                if temp_file.suffix in ['.mp3', '.m4a', '.flac']:
                                    downloaded_files.append(str(final_path))
                                    print(f"✅ Moved: {temp_file.name} -> {final_path}")
            
        except Exception as e:
            print(f"❌ Error downloading from YouTube: {e}")
        
        return downloaded_files
    
    def _extract_year(self, date_field) -> str:
        """Extract year from various date formats."""
        if isinstance(date_field, list):
            date_field = date_field[0] if date_field else 'Unknown'
        
        if isinstance(date_field, str):
            # Try to extract 4-digit year
            year_match = re.search(r'\b(19|20)\d{2}\b', date_field)
            if year_match:
                return year_match.group()
        
        return 'Unknown'
    
    def _extract_description(self, desc_field) -> str:
        """Clean and extract description."""
        if isinstance(desc_field, list):
            desc_field = desc_field[0] if desc_field else ''
        
        # Truncate long descriptions
        if len(desc_field) > 200:
            desc_field = desc_field[:200] + "..."
        
        return desc_field
    
    def _save_metadata(self, directory: Path, item: Dict, files: List[str]):
        """Save comprehensive metadata for downloaded items."""
        metadata_file = directory / "metadata.json"
        
        metadata = {
            'source': item['source'],
            'title': item['title'],
            'artist': item['artist'],
            'year': item.get('year', 'Unknown'),
            'description': item.get('description', ''),
            'url': item.get('url', ''),
            'downloaded_at': datetime.now().isoformat(),
            'files': [Path(f).name for f in files],
            'total_files': len(files)
        }
        
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    def create_playlist(self, downloaded_files: List[str], name: str):
        """Create M3U playlist from downloaded files."""
        if not downloaded_files:
            return
        
        playlist_dir = self.download_dir / "playlists"
        playlist_path = playlist_dir / f"{name}.m3u"
        
        with open(playlist_path, 'w', encoding='utf-8') as f:
            f.write("#EXTM3U\n")
            f.write(f"# Playlist: {name}\n")
            f.write(f"# Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"# Total tracks: {len(downloaded_files)}\n\n")
            
            for file_path in downloaded_files:
                f.write(f"{file_path}\n")
        
        print(f"🎵 Playlist created: {playlist_path}")
    
    def run_discovery_session(self):
        """Run the interactive music discovery session."""
        try:
            print(f"📁 Music library: {self.download_dir.absolute()}")
            
            while True:
                # Get user preferences
                query = self.get_user_input()
                
                # Search all sources
                print(f"\n🔍 Searching for {query['type']}...")
                results = self.search_all_sources(query)
                
                # Display results and get user selection
                selected_items = self.display_results(results, query)
                
                if not selected_items:
                    print("No items selected.")
                    continue
                
                # Download selected items
                all_downloaded = []
                
                for item in selected_items:
                    if item['source'] == 'Internet Archive':
                        files = self.download_from_internet_archive(item)
                        all_downloaded.extend(files)
                    elif item['source'] == 'YouTube':
                        files = self.download_from_youtube(item)
                        all_downloaded.extend(files)
                
                # Create playlist if requested
                if query.get('create_playlist') and all_downloaded:
                    playlist_name = f"{query.get('artist', 'music')}_{query['type']}_{datetime.now().strftime('%Y%m%d_%H%M')}"
                    self.create_playlist(all_downloaded, playlist_name)
                
                print(f"\n🎉 Downloaded {len(all_downloaded)} files!")
                print(f"📁 Music library: {self.download_dir.absolute()}")
                
                # Continue?
                another = input("\nDiscover more music? (Y/n): ").strip().lower()
                if another == 'n':
                    break
                    
        except KeyboardInterrupt:
            print("\n\n👋 Happy listening!")

def main():
    parser = argparse.ArgumentParser(description='Enhanced Music Discovery & Download System')
    parser.add_argument('-d', '--dir', default='./music_library',
                       help='Download directory')
    parser.add_argument('--artist', help='Artist to search for')
    parser.add_argument('--discography', action='store_true',
                       help='Download entire discography')
    
    args = parser.parse_args()
    
    downloader = EnhancedMusicDownloader(download_dir=args.dir)
    
    if args.artist and args.discography:
        # Quick discography download
        query = {
            'type': 'discography',
            'artist': args.artist,
            'include_features': False,
            'include_compilations': True,
            'quality': 'high',
            'format': 'mp3',
            'create_playlist': True
        }
        
        results = downloader.search_all_sources(query)
        if results:
            for item in results[:20]:  # Limit for safety
                if item['source'] == 'Internet Archive':
                    downloader.download_from_internet_archive(item)
    else:
        downloader.run_discovery_session()

if __name__ == "__main__":
    main()

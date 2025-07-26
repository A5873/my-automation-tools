#!/usr/bin/env python3
"""
Music Discovery & Download Script
Searches multiple legal sources for free music and builds your personal library.
"""

import requests
import json
import os
import re
from urllib.parse import quote, urlparse
from pathlib import Path
import argparse
from datetime import datetime
import time

class MusicDiscoveryDownloader:
    def __init__(self, download_dir="./music_library"):
        """
        Initialize the music discovery and download system.
        
        Args:
            download_dir (str): Directory to save downloaded music
        """
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.download_dir / "artists").mkdir(exist_ok=True)
        (self.download_dir / "albums").mkdir(exist_ok=True)
        (self.download_dir / "singles").mkdir(exist_ok=True)
        (self.download_dir / "metadata").mkdir(exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'MusicDiscovery/1.0 (Personal Use)'
        })
        
        # Sources to search (legal free music platforms)
        self.sources = {
            'internet_archive': {
                'name': 'Internet Archive',
                'base_url': 'https://archive.org/advancedsearch.php',
                'enabled': True
            },
            'freemusicarchive': {
                'name': 'Free Music Archive',
                'base_url': 'https://freemusicarchive.org/api/get',
                'enabled': True
            },
            'jamendo': {
                'name': 'Jamendo',
                'base_url': 'https://api.jamendo.com/v3.0',
                'enabled': True
            }
        }
    
    def get_user_input(self):
        """
        Interactive user input for music discovery.
        
        Returns:
            dict: User preferences and search criteria
        """
        print("\n🎵 Music Discovery & Download System")
        print("=" * 50)
        print("Discover and download free music from legal sources!")
        print()
        
        # Search type
        print("🔍 What would you like to search for?")
        print("1. Specific song")  
        print("2. Artist discography")
        print("3. Album")
        print("4. Genre exploration")
        print("5. Browse by year")
        
        search_type = input("\nSelect option (1-5): ").strip()
        
        search_query = {}
        
        if search_type == "1":
            search_query['type'] = 'song'
            search_query['artist'] = input("Enter artist name: ").strip()
            search_query['song'] = input("Enter song title: ").strip()
            
        elif search_type == "2":
            search_query['type'] = 'discography'
            search_query['artist'] = input("Enter artist name: ").strip()
            search_query['include_compilations'] = input("Include compilations? (y/N): ").strip().lower() == 'y'
            
        elif search_type == "3":
            search_query['type'] = 'album'
            search_query['artist'] = input("Enter artist name: ").strip()
            search_query['album'] = input("Enter album title: ").strip()
            
        elif search_type == "4":
            search_query['type'] = 'genre'
            search_query['genre'] = input("Enter genre (e.g., jazz, rock, electronic): ").strip()
            search_query['limit'] = int(input("How many tracks to find? (default 20): ").strip() or "20")
            
        elif search_type == "5":
            search_query['type'] = 'year'
            search_query['year'] = input("Enter year (e.g., 1990): ").strip()
            search_query['genre'] = input("Optional genre filter: ").strip() or None
            search_query['limit'] = int(input("How many tracks to find? (default 20): ").strip() or "20")
        
        # Download preferences
        print("\n📁 Download Options:")
        search_query['quality'] = input("Preferred quality (high/medium/any): ").strip() or "high"
        search_query['format'] = input("Preferred format (mp3/flac/any): ").strip() or "mp3"
        search_query['create_playlist'] = input("Create M3U playlist? (y/N): ").strip().lower() == 'y'
        
        return search_query
    
    def search_internet_archive(self, query):
        """
        Search Internet Archive for music.
        
        Args:
            query (dict): Search parameters
            
        Returns:
            list: Found tracks/albums
        """
        print("🔍 Searching Internet Archive...")
        
        results = []
        
        try:
            if query['type'] == 'song':
                search_term = f"{query['artist']} {query['song']}"
            elif query['type'] == 'discography':
                search_term = query['artist']
            elif query['type'] == 'album':
                search_term = f"{query['artist']} {query['album']}"
            elif query['type'] == 'genre':
                search_term = query['genre']
            else:
                search_term = query.get('artist', 'music')
            
            params = {
                'q': f'collection:opensource_audio AND ({search_term})',
                'fl': 'identifier,title,creator,date,description,downloads,format',
                'rows': 50,
                'page': 1,
                'output': 'json'
            }
            
            response = self.session.get(self.sources['internet_archive']['base_url'], params=params)
            response.raise_for_status()
            
            data = response.json()
            
            for doc in data.get('response', {}).get('docs', []):
                result = {
                    'source': 'Internet Archive',
                    'title': doc.get('title', ['Unknown'])[0] if isinstance(doc.get('title'), list) else doc.get('title', 'Unknown'),
                    'artist': doc.get('creator', ['Unknown'])[0] if isinstance(doc.get('creator'), list) else doc.get('creator', 'Unknown'),
                    'year': doc.get('date', ['Unknown'])[0] if isinstance(doc.get('date'), list) else doc.get('date', 'Unknown'),
                    'identifier': doc.get('identifier'),
                    'description': doc.get('description', [''])[0] if isinstance(doc.get('description'), list) else doc.get('description', ''),
                    'downloads': doc.get('downloads', 0),
                    'formats': doc.get('format', [])
                }
                results.append(result)
                
        except Exception as e:
            print(f"❌ Error searching Internet Archive: {e}")
        
        return results
    
    def search_jamendo(self, query):
        """
        Search Jamendo for music.
        
        Args:
            query (dict): Search parameters
            
        Returns:
            list: Found tracks
        """
        print("🔍 Searching Jamendo...")
        
        results = []
        
        try:
            if query['type'] == 'song':
                search_term = f"{query['artist']} {query['song']}"
                endpoint = '/tracks'
            elif query['type'] == 'discography' or query['type'] == 'album':
                search_term = query['artist']
                endpoint = '/albums'
            else:
                search_term = query.get('genre', 'music')
                endpoint = '/tracks'
            
            # Note: Jamendo requires API key for full access, this is a simplified version
            params = {
                'format': 'json',
                'limit': 50,
                'search': search_term
            }
            
            # For demonstration - in real use you'd need Jamendo API key
            # url = f"{self.sources['jamendo']['base_url']}{endpoint}"
            # response = self.session.get(url, params=params)
            
            # Placeholder results for now
            print("ℹ️  Jamendo search requires API key - skipping for now")
                
        except Exception as e:
            print(f"❌ Error searching Jamendo: {e}")
        
        return results
    
    def search_all_sources(self, query):
        """
        Search all enabled sources for music.
        
        Args:
            query (dict): Search parameters
            
        Returns:
            list: Combined results from all sources
        """
        all_results = []
        
        if self.sources['internet_archive']['enabled']:
            ia_results = self.search_internet_archive(query)
            all_results.extend(ia_results)
        
        if self.sources['jamendo']['enabled']:
            jamendo_results = self.search_jamendo(query)
            all_results.extend(jamendo_results)
        
        # Sort by downloads/popularity
        all_results.sort(key=lambda x: x.get('downloads', 0), reverse=True)
        
        return all_results
    
    def display_results(self, results, query):
        """
        Display search results to user.
        
        Args:
            results (list): Search results
            query (dict): Original search query
        """
        if not results:
            print("❌ No results found.")
            return []
        
        print(f"\n🎵 Found {len(results)} results:")
        print("=" * 60)
        
        for i, result in enumerate(results[:20], 1):  # Show top 20
            print(f"{i}. {result['title']}")
            print(f"   Artist: {result['artist']}")
            print(f"   Year: {result['year']}")
            print(f"   Source: {result['source']}")
            if result.get('downloads'):
                print(f"   Downloads: {result['downloads']:,}")
            print()
        
        # Let user select items to download
        selected = []
        
        if query['type'] == 'discography':
            choice = input(f"Download all {len(results)} items? (y/N): ").strip().lower()
            if choice == 'y':
                selected = results
        else:
            while True:
                choice = input(f"Select items to download (1-{min(len(results), 20)}, 'all', or 'done'): ").strip().lower()
                
                if choice == 'done':
                    break
                elif choice == 'all':
                    selected = results[:20]
                    break
                else:
                    try:
                        idx = int(choice) - 1
                        if 0 <= idx < min(len(results), 20):
                            if results[idx] not in selected:
                                selected.append(results[idx])
                                print(f"✅ Added: {results[idx]['title']}")
                            else:
                                print("Already selected!")
                        else:
                            print("Invalid selection!")
                    except ValueError:
                        print("Please enter a number, 'all', or 'done'")
        
        return selected
    
    def download_from_internet_archive(self, item):
        """
        Download music from Internet Archive.
        
        Args:
            item (dict): Item metadata
            
        Returns:
            list: Paths to downloaded files
        """
        downloaded_files = []
        
        try:
            identifier = item['identifier']
            
            # Get item details
            details_url = f"https://archive.org/metadata/{identifier}"
            response = self.session.get(details_url)
            response.raise_for_status()
            
            metadata = response.json()
            files = metadata.get('files', [])
            
            # Filter for audio files
            audio_files = []
            for file_info in files:
                name = file_info.get('name', '')
                format_type = file_info.get('format', '').lower()
                
                if format_type in ['mp3', 'flac', 'ogg', 'wav', 'm4a']:
                    audio_files.append(file_info)
            
            if not audio_files:
                print(f"⚠️  No audio files found for {item['title']}")
                return downloaded_files
            
            # Create artist directory
            safe_artist = re.sub(r'[^\w\s-]', '', item['artist']).strip()
            artist_dir = self.download_dir / "artists" / safe_artist
            artist_dir.mkdir(exist_ok=True)
            
            # Create album/item directory
            safe_title = re.sub(r'[^\w\s-]', '', item['title']).strip()
            item_dir = artist_dir / safe_title
            item_dir.mkdir(exist_ok=True)
            
            print(f"📥 Downloading {len(audio_files)} files from '{item['title']}'...")
            
            for file_info in audio_files:
                filename = file_info['name']
                file_url = f"https://archive.org/download/{identifier}/{filename}"
                
                local_path = item_dir / filename
                
                if local_path.exists():
                    print(f"⏭️  Skipping {filename} (already exists)")
                    downloaded_files.append(str(local_path))
                    continue
                
                print(f"📥 Downloading {filename}...")
                
                response = self.session.get(file_url, stream=True)
                response.raise_for_status()
                
                with open(local_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                downloaded_files.append(str(local_path))
                print(f"✅ Downloaded: {filename}")
                
                # Be respectful with download speed
                time.sleep(0.5)
            
            # Save metadata
            metadata_file = item_dir / "metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump({
                    'source': 'Internet Archive',
                    'identifier': identifier,
                    'title': item['title'],
                    'artist': item['artist'],
                    'year': item['year'],
                    'description': item['description'],
                    'downloads': item['downloads'],
                    'downloaded_at': datetime.now().isoformat(),
                    'files': [f.name for f in downloaded_files]
                }, f, indent=2)
            
        except Exception as e:
            print(f"❌ Error downloading {item['title']}: {e}")
        
        return downloaded_files
    
    def create_playlist(self, downloaded_files, name):
        """
        Create an M3U playlist from downloaded files.
        
        Args:
            downloaded_files (list): List of file paths
            name (str): Playlist name
        """
        if not downloaded_files:
            return
        
        playlist_path = self.download_dir / f"{name}.m3u"
        
        with open(playlist_path, 'w', encoding='utf-8') as f:
            f.write("#EXTM3U\n")
            for file_path in downloaded_files:
                f.write(f"{file_path}\n")
        
        print(f"🎵 Playlist created: {playlist_path}")
    
    def run_discovery_session(self):
        """
        Run the interactive music discovery session.
        """
        try:
            print(f"\n📁 Music library location: {self.download_dir.absolute()}")
            
            while True:
                # Get user search criteria
                query = self.get_user_input()
                
                # Search all sources
                print(f"\n🔍 Searching for {query['type']}...")
                results = self.search_all_sources(query)
                
                # Display and let user select
                selected_items = self.display_results(results, query)
                
                if not selected_items:
                    print("No items selected for download.")
                    continue
                
                # Download selected items
                all_downloaded = []
                
                for item in selected_items:
                    if item['source'] == 'Internet Archive':
                        files = self.download_from_internet_archive(item)
                        all_downloaded.extend(files)
                
                # Create playlist if requested
                if query.get('create_playlist') and all_downloaded:
                    playlist_name = f"{query.get('artist', 'music')}_{query['type']}_{datetime.now().strftime('%Y%m%d')}"
                    self.create_playlist(all_downloaded, playlist_name)
                
                print(f"\n🎉 Downloaded {len(all_downloaded)} files!")
                print(f"📁 Check your music library: {self.download_dir.absolute()}")
                
                # Continue or exit
                another = input("\nSearch for more music? (Y/n): ").strip().lower()
                if another == 'n':
                    break
                    
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description='Music Discovery & Download System')
    parser.add_argument('-d', '--dir', default='./music_library',
                       help='Download directory (default: ./music_library)')
    parser.add_argument('--artist', help='Artist to search for')
    parser.add_argument('--song', help='Song to search for')
    parser.add_argument('--album', help='Album to search for')
    parser.add_argument('--non-interactive', action='store_true',
                       help='Run in non-interactive mode')
    
    args = parser.parse_args()
    
    # Create downloader instance
    downloader = MusicDiscoveryDownloader(download_dir=args.dir)
    
    if args.non_interactive and (args.artist or args.song or args.album):
        # Non-interactive mode
        query = {
            'type': 'song' if args.song else 'discography',
            'artist': args.artist or '',
            'song': args.song or '',
            'album': args.album or '',
            'quality': 'high',
            'format': 'mp3',
            'create_playlist': True
        }
        
        results = downloader.search_all_sources(query)
        if results:
            # Auto-download first result in non-interactive mode
            if results[0]['source'] == 'Internet Archive':
                downloader.download_from_internet_archive(results[0])
    else:
        # Interactive mode
        downloader.run_discovery_session()

if __name__ == "__main__":
    main()

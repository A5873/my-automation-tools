#!/usr/bin/env python3
"""
Interactive Lyrics Fetcher Script
An interactive script that fetches song lyrics and cover art using the Genius API.
"""

import requests
import re
import os
from bs4 import BeautifulSoup
import argparse
from urllib.parse import urlparse
from pathlib import Path

class InteractiveLyricsFetcher:
    def __init__(self, access_token=None):
        """
        Initialize the lyrics fetcher with Genius API access token.
        
        Args:
            access_token (str): Genius API access token. If None, will try to get from environment.
        """
        self.access_token = access_token or os.getenv('GENIUS_ACCESS_TOKEN')
        if not self.access_token:
            print("⚠️  Warning: No Genius API access token provided.")
            print("Please set GENIUS_ACCESS_TOKEN environment variable or pass token directly.")
            print("Get your token from: https://genius.com/api-clients")
        
        self.base_url = "https://api.genius.com"
        self.headers = {
            'Authorization': f'Bearer {self.access_token}' if self.access_token else '',
            'User-Agent': 'InteractiveLyricsFetcher/1.0'
        }
    
    def get_user_input(self):
        """
        Interactively get song information from user.
        
        Returns:
            dict: User input containing song details
        """
        print("\n🎵 Interactive Lyrics Fetcher")
        print("=" * 40)
        
        # Get song details
        song_title = input("Enter song title: ").strip()
        artist_name = input("Enter artist name (optional): ").strip()
        
        # Build search query
        if artist_name:
            search_query = f"{artist_name} {song_title}"
        else:
            search_query = song_title
        
        # Get preferences
        print("\n📁 File Options:")
        custom_filename = input("Custom filename (press Enter for auto-generated): ").strip()
        
        print("\n🖼️  Cover Art Options:")
        download_cover = input("Download cover art? (y/N): ").strip().lower() in ['y', 'yes']
        
        return {
            'song_title': song_title,
            'artist_name': artist_name,
            'search_query': search_query,
            'custom_filename': custom_filename if custom_filename else None,
            'download_cover': download_cover
        }
    
    def search_songs(self, query, limit=5):
        """
        Search for songs on Genius with multiple results.
        
        Args:
            query (str): Search query
            limit (int): Maximum number of results
            
        Returns:
            list: List of song information dicts
        """
        if not self.access_token:
            print("❌ Error: No API token available for search.")
            return []
            
        search_url = f"{self.base_url}/search"
        params = {'q': query, 'per_page': limit}
        
        try:
            response = requests.get(search_url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            hits = data['response']['hits']
            
            songs = []
            for hit in hits:
                song = hit['result']
                songs.append({
                    'title': song['title'],
                    'artist': song['primary_artist']['name'],
                    'url': song['url'],
                    'id': song['id'],
                    'cover_url': song.get('song_art_image_url', ''),
                    'release_date': song.get('release_date_for_display', 'Unknown')
                })
            
            return songs
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error searching for song: {e}")
            return []
    
    def select_song(self, songs):
        """
        Let user select from multiple search results.
        
        Args:
            songs (list): List of song dictionaries
            
        Returns:
            dict: Selected song or None
        """
        if not songs:
            print("❌ No songs found.")
            return None
        
        if len(songs) == 1:
            song = songs[0]
            confirm = input(f"\nFound: {song['artist']} - {song['title']} ({song['release_date']})\nIs this correct? (Y/n): ").strip().lower()
            return song if confirm != 'n' else None
        
        print(f"\n🔍 Found {len(songs)} results:")
        print("-" * 60)
        
        for i, song in enumerate(songs, 1):
            print(f"{i}. {song['artist']} - {song['title']}")
            print(f"   Release: {song['release_date']}")
            print()
        
        while True:
            try:
                choice = input(f"Select a song (1-{len(songs)}) or 'q' to quit: ").strip()
                if choice.lower() == 'q':
                    return None
                
                choice_num = int(choice)
                if 1 <= choice_num <= len(songs):
                    return songs[choice_num - 1]
                else:
                    print(f"Please enter a number between 1 and {len(songs)}")
            except ValueError:
                print("Please enter a valid number or 'q' to quit")
    
    def get_lyrics_from_url(self, url):
        """
        Scrape lyrics from a Genius song URL.
        
        Args:
            url (str): Genius song URL
            
        Returns:
            str: Song lyrics or None if failed
        """
        try:
            # Add headers to mimic a regular browser
            headers = {
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Multiple approaches to find lyrics on Genius
            lyrics_text = None
            
            # Method 1: Look for data-lyrics-container
            lyrics_containers = soup.find_all('div', {'data-lyrics-container': 'true'})
            if lyrics_containers:
                lyrics_parts = []
                for container in lyrics_containers:
                    lyrics_parts.append(container.get_text(separator='\n'))
                lyrics_text = '\n'.join(lyrics_parts)
            
            # Method 2: Look for common lyrics class patterns
            if not lyrics_text:
                lyrics_divs = soup.find_all('div', class_=re.compile(r'Lyrics__Container|lyrics', re.I))
                if lyrics_divs:
                    lyrics_parts = []
                    for div in lyrics_divs:
                        lyrics_parts.append(div.get_text(separator='\n'))
                    lyrics_text = '\n'.join(lyrics_parts)
            
            # Method 3: Look for any div with 'lyrics' in the class name
            if not lyrics_text:
                all_divs = soup.find_all('div')
                for div in all_divs:
                    if div.get('class'):
                        class_names = ' '.join(div.get('class', []))
                        if 'lyrics' in class_names.lower():
                            text = div.get_text(separator='\n')
                            if len(text) > 100:  # Assume real lyrics are longer
                                lyrics_text = text
                                break
            
            if lyrics_text:
                # Clean up the lyrics text
                lyrics_text = re.sub(r'\n\s*\n', '\n\n', lyrics_text.strip())
                # Remove common non-lyric elements
                lines = lyrics_text.split('\n')
                cleaned_lines = []
                
                for line in lines:
                    line = line.strip()
                    # Skip empty lines, but keep intentional line breaks
                    if not line:
                        if cleaned_lines and cleaned_lines[-1] != '':
                            cleaned_lines.append('')
                        continue
                    
                    # Skip common Genius UI elements
                    skip_patterns = [
                        r'^\d+$',  # Just numbers
                        r'^See .* Live$',
                        r'^Get tickets',
                        r'^\[.*\]$',  # Brackets only (but keep lyrics in brackets)
                    ]
                    
                    should_skip = False
                    for pattern in skip_patterns:
                        if re.match(pattern, line, re.I):
                            should_skip = True
                            break
                    
                    if not should_skip:
                        cleaned_lines.append(line)
                
                return '\n'.join(cleaned_lines).strip()
            else:
                print("⚠️  Could not find lyrics on the page. The page structure might have changed.")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Error fetching lyrics: {e}")
            return None
    
    def download_cover_art(self, cover_url, filename_base):
        """
        Download cover art image.
        
        Args:
            cover_url (str): URL of the cover image
            filename_base (str): Base filename for the image
            
        Returns:
            str: Path to saved image or None if failed
        """
        if not cover_url:
            print("⚠️  No cover art URL available.")
            return None
        
        try:
            response = requests.get(cover_url)
            response.raise_for_status()
            
            # Determine file extension from URL or content type
            parsed_url = urlparse(cover_url)
            ext = os.path.splitext(parsed_url.path)[1]
            if not ext:
                content_type = response.headers.get('content-type', '')
                if 'jpeg' in content_type or 'jpg' in content_type:
                    ext = '.jpg'
                elif 'png' in content_type:
                    ext = '.png'
                else:
                    ext = '.jpg'  # Default
            
            cover_filename = f"{filename_base}_cover{ext}"
            
            with open(cover_filename, 'wb') as f:
                f.write(response.content)
            
            print(f"🖼️  Cover art saved to: {cover_filename}")
            return cover_filename
            
        except requests.exceptions.RequestException as e:
            print(f"❌ Error downloading cover art: {e}")
            return None
        except IOError as e:
            print(f"❌ Error saving cover art: {e}")
            return None
    
    def create_safe_filename(self, artist, title):
        """
        Create a safe filename from artist and title.
        
        Args:
            artist (str): Artist name
            title (str): Song title
            
        Returns:
            str: Safe filename
        """
        safe_title = re.sub(r'[^\w\s-]', '', title)
        safe_artist = re.sub(r'[^\w\s-]', '', artist)
        return f"{safe_artist} - {safe_title}".replace(' ', '_')
    
    def save_lyrics(self, song_info, lyrics, output_file):
        """
        Save lyrics to file with metadata.
        
        Args:
            song_info (dict): Song information
            lyrics (str): Song lyrics
            output_file (str): Output file path
            
        Returns:
            bool: Success status
        """
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Title: {song_info['title']}\n")
                f.write(f"Artist: {song_info['artist']}\n")
                f.write(f"Release Date: {song_info['release_date']}\n")
                f.write(f"Source: {song_info['url']}\n")
                f.write("=" * 60 + "\n\n")
                
                if lyrics:
                    f.write("LYRICS:\n")
                    f.write("-" * 20 + "\n\n")
                    f.write(lyrics)
                    f.write("\n\n" + "-" * 20)
                    f.write(f"\n\nLyrics sourced from: {song_info['url']}")
                    f.write("\nFor personal use only.")
                else:
                    f.write("Note: Could not extract lyrics from the source.\n")
                    f.write(f"Please visit the source URL above to view the full lyrics: {song_info['url']}")
            
            if lyrics:
                print(f"📝 Lyrics saved to: {output_file}")
            else:
                print(f"📝 Song info saved to: {output_file} (lyrics extraction failed)")
            return True
            
        except IOError as e:
            print(f"❌ Error saving file: {e}")
            return False
    
    def run_interactive_session(self):
        """
        Run the interactive lyrics fetching session.
        """
        try:
            while True:
                # Get user input
                user_input = self.get_user_input()
                
                if not user_input['song_title']:
                    print("❌ Song title is required!")
                    continue
                
                print(f"\n🔍 Searching for: {user_input['search_query']}")
                
                # Search for songs
                songs = self.search_songs(user_input['search_query'])
                
                # Let user select song
                selected_song = self.select_song(songs)
                if not selected_song:
                    print("❌ No song selected.")
                    another = input("\nSearch for another song? (Y/n): ").strip().lower()
                    if another == 'n':
                        break
                    continue
                
                print(f"\n✅ Selected: {selected_song['artist']} - {selected_song['title']}")
                
                # Generate filename
                if user_input['custom_filename']:
                    filename_base = user_input['custom_filename']
                else:
                    filename_base = self.create_safe_filename(
                        selected_song['artist'], 
                        selected_song['title']
                    )
                
                lyrics_file = f"{filename_base}.txt"
                
                # Fetch lyrics (but don't save full content due to copyright)
                print("📥 Fetching song information...")
                lyrics = self.get_lyrics_from_url(selected_song['url'])
                
                # Save song info
                success = self.save_lyrics(selected_song, lyrics, lyrics_file)
                
                # Download cover art if requested
                if user_input['download_cover'] and selected_song['cover_url']:
                    print("📥 Downloading cover art...")
                    self.download_cover_art(selected_song['cover_url'], filename_base)
                
                if success:
                    print(f"\n🎉 Complete! Files saved with base name: {filename_base}")
                
                # Ask if user wants to continue
                another = input("\nSearch for another song? (Y/n): ").strip().lower()
                if another == 'n':
                    break
                    
        except KeyboardInterrupt:
            print("\n\n👋 Goodbye!")
        except Exception as e:
            print(f"\n❌ An error occurred: {e}")

def main():
    parser = argparse.ArgumentParser(description='Interactive lyrics and cover art fetcher')
    parser.add_argument('-t', '--token', help='Genius API access token')
    parser.add_argument('--non-interactive', action='store_true', 
                       help='Run in non-interactive mode with provided query')
    parser.add_argument('query', nargs='?', help='Song search query for non-interactive mode')
    
    args = parser.parse_args()
    
    # Create fetcher instance
    fetcher = InteractiveLyricsFetcher(access_token=args.token)
    
    if args.non_interactive and args.query:
        # Non-interactive mode (for backward compatibility)
        songs = fetcher.search_songs(args.query, limit=1)
        if songs:
            selected_song = songs[0]
            filename_base = fetcher.create_safe_filename(
                selected_song['artist'], 
                selected_song['title']
            )
            lyrics = fetcher.get_lyrics_from_url(selected_song['url'])
            fetcher.save_lyrics(selected_song, lyrics, f"{filename_base}.txt")
            fetcher.download_cover_art(selected_song['cover_url'], filename_base)
    else:
        # Interactive mode
        fetcher.run_interactive_session()

if __name__ == "__main__":
    main()

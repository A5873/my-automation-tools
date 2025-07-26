#!/usr/bin/env python3
"""
Lyrics Fetcher Script
Fetches song lyrics using the Genius API and saves them to a file.
"""

import requests
import re
import os
from bs4 import BeautifulSoup
import argparse

class LyricsFetcher:
    def __init__(self, access_token=None):
        """
        Initialize the lyrics fetcher with Genius API access token.
        
        Args:
            access_token (str): Genius API access token. If None, will try to get from environment.
        """
        self.access_token = access_token or os.getenv('GENIUS_ACCESS_TOKEN')
        if not self.access_token:
            print("Warning: No Genius API access token provided.")
            print("Please set GENIUS_ACCESS_TOKEN environment variable or pass token directly.")
            print("Get your token from: https://genius.com/api-clients")
        
        self.base_url = "https://api.genius.com"
        self.headers = {
            'Authorization': f'Bearer {self.access_token}' if self.access_token else '',
            'User-Agent': 'LyricsFetcher/1.0'
        }
    
    def search_song(self, query):
        """
        Search for a song on Genius.
        
        Args:
            query (str): Search query (e.g., "artist song title")
            
        Returns:
            dict: Song information or None if not found
        """
        if not self.access_token:
            print("Error: No API token available for search.")
            return None
            
        search_url = f"{self.base_url}/search"
        params = {'q': query}
        
        try:
            response = requests.get(search_url, headers=self.headers, params=params)
            response.raise_for_status()
            
            data = response.json()
            hits = data['response']['hits']
            
            if hits:
                # Return the first hit
                song = hits[0]['result']
                return {
                    'title': song['title'],
                    'artist': song['primary_artist']['name'],
                    'url': song['url'],
                    'id': song['id']
                }
            else:
                print(f"No songs found for query: {query}")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error searching for song: {e}")
            return None
    
    def get_lyrics_from_url(self, url):
        """
        Scrape lyrics from a Genius song URL.
        
        Args:
            url (str): Genius song URL
            
        Returns:
            str: Song lyrics or None if failed
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Find lyrics container (Genius uses various div classes)
            lyrics_div = soup.find('div', {'data-lyrics-container': 'true'})
            if not lyrics_div:
                # Try alternative selectors
                lyrics_div = soup.find('div', class_=re.compile(r'lyrics|Lyrics'))
            
            if lyrics_div:
                # Extract text and clean it up
                lyrics = lyrics_div.get_text(separator='\n')
                # Remove extra whitespace and clean up
                lyrics = re.sub(r'\n\s*\n', '\n\n', lyrics.strip())
                return lyrics
            else:
                print("Could not find lyrics on the page.")
                return None
                
        except requests.exceptions.RequestException as e:
            print(f"Error fetching lyrics: {e}")
            return None
    
    def fetch_and_save_lyrics(self, query, output_file=None):
        """
        Search for a song, fetch its lyrics, and save to file.
        
        Args:
            query (str): Search query
            output_file (str): Output file path. If None, auto-generates filename.
            
        Returns:
            str: Path to saved file or None if failed
        """
        print(f"Searching for: {query}")
        
        # Search for the song
        song_info = self.search_song(query)
        if not song_info:
            return None
        
        print(f"Found: {song_info['artist']} - {song_info['title']}")
        
        # Fetch lyrics
        lyrics = self.get_lyrics_from_url(song_info['url'])
        if not lyrics:
            return None
        
        # Generate filename if not provided
        if not output_file:
            safe_title = re.sub(r'[^\w\s-]', '', song_info['title'])
            safe_artist = re.sub(r'[^\w\s-]', '', song_info['artist'])
            output_file = f"{safe_artist} - {safe_title}.txt".replace(' ', '_')
        
        # Save to file
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                f.write(f"Title: {song_info['title']}\n")
                f.write(f"Artist: {song_info['artist']}\n")
                f.write(f"Source: {song_info['url']}\n")
                f.write("-" * 50 + "\n\n")
                f.write(lyrics)
            
            print(f"Lyrics saved to: {output_file}")
            return output_file
            
        except IOError as e:
            print(f"Error saving file: {e}")
            return None

def main():
    parser = argparse.ArgumentParser(description='Fetch song lyrics and save to file')
    parser.add_argument('query', help='Song search query (e.g., "artist song title")')
    parser.add_argument('-o', '--output', help='Output file path')
    parser.add_argument('-t', '--token', help='Genius API access token')
    
    args = parser.parse_args()
    
    # Create fetcher instance
    fetcher = LyricsFetcher(access_token=args.token)
    
    # Fetch and save lyrics
    result = fetcher.fetch_and_save_lyrics(args.query, args.output)
    
    if result:
        print(f"\n✅ Success! Lyrics saved to: {result}")
    else:
        print("\n❌ Failed to fetch lyrics.")

if __name__ == "__main__":
    main()

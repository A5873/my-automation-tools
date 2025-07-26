#!/usr/bin/env python3
"""
Simple test script to demonstrate the music downloader working
"""
import sys
sys.path.append('.')

from enhanced_music_downloader import EnhancedMusicDownloader

def test_download():
    # Create downloader instance
    downloader = EnhancedMusicDownloader(download_dir="./test_music")
    
    # Test search for Beatles songs
    query = {
        'type': 'song',
        'artist': 'The Beatles',
        'song': 'Yesterday',
        'quality': 'high',
        'format': 'mp3',
        'organize_by': 'artist',
        'download_covers': True,
        'create_playlist': True,
        'tag_metadata': True
    }
    
    print("🔍 Testing search for The Beatles - Yesterday...")
    results = downloader.search_all_sources(query)
    
    print(f"\n✅ Found {len(results)} results!")
    
    # Show first 5 results
    print("\n📋 Top 5 results:")
    for i, result in enumerate(results[:5], 1):
        source = result['source']
        title = result['title']
        artist = result['artist']
        
        if source == 'Internet Archive':
            downloads = result.get('downloads', 0)
            print(f"  {i}. [{source}] {title} by {artist} ({downloads:,} downloads)")
        else:  # YouTube
            views = result.get('view_count', 0)
            duration = result.get('duration', 0)
            mins = int(duration) // 60
            secs = int(duration) % 60
            print(f"  {i}. [{source}] {title} by {artist} ({views:,} views, {mins}:{secs:02d})")
    
    # Ask user if they want to download the first YouTube result
    if results:
        print(f"\n🎵 Would you like to download the first result?")
        print(f"   Title: {results[0]['title']}")
        print(f"   Source: {results[0]['source']}")
        
        choice = input("Download? (y/N): ").strip().lower()
        if choice == 'y':
            print("\n📥 Starting download...")
            if results[0]['source'] == 'Internet Archive':
                files = downloader.download_from_internet_archive(results[0])
            elif results[0]['source'] == 'YouTube':
                files = downloader.download_from_youtube(results[0])
            
            if files:
                print(f"✅ Successfully downloaded {len(files)} files!")
                for file_path in files:
                    print(f"   📁 {file_path}")
            else:
                print("❌ No files were downloaded")
    
    print(f"\n📂 Check your music in: ./test_music/")

if __name__ == "__main__":
    test_download()

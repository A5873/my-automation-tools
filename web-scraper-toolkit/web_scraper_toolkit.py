#!/usr/bin/env python3
"""
Web Scraper Toolkit
A comprehensive suite of web scrapers for various data collection tasks.
"""

import requests
import json
import csv
import os
import re
import time
import random
from datetime import datetime, timedelta
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import List, Dict, Optional, Any
import argparse
import schedule
import threading

try:
    from bs4 import BeautifulSoup
    from fake_useragent import UserAgent
    import pandas as pd
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
except ImportError as e:
    print(f"Missing required package: {e}")
    print("Install with: pip install -r requirements.txt")
    exit(1)

class BaseScraper:
    """Base class for all scrapers with common functionality."""
    
    def __init__(self, delay_range=(1, 3), use_fake_agent=True):
        self.delay_range = delay_range
        self.session = requests.Session()
        self.ua = UserAgent() if use_fake_agent else None
        
        # Set up session headers
        if self.ua:
            self.session.headers.update({
                'User-Agent': self.ua.random
            })
        else:
            self.session.headers.update({
                'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36'
            })
        
        # Create output directory
        self.output_dir = Path("scraped_data")
        self.output_dir.mkdir(exist_ok=True)
    
    def random_delay(self):
        """Add random delay between requests to be respectful."""
        time.sleep(random.uniform(*self.delay_range))
    
    def save_to_json(self, data: List[Dict], filename: str):
        """Save data to JSON file."""
        filepath = self.output_dir / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"✅ Data saved to: {filepath}")
        return filepath
    
    def save_to_csv(self, data: List[Dict], filename: str):
        """Save data to CSV file."""
        if not data:
            print("No data to save")
            return None
        
        filepath = self.output_dir / f"{filename}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv"
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False, encoding='utf-8')
        print(f"✅ Data saved to: {filepath}")
        return filepath

class PriceMonitor(BaseScraper):
    """Monitor prices from various e-commerce sites."""
    
    def __init__(self):
        super().__init__(delay_range=(2, 5))
        self.trackers = []
    
    def add_product(self, url: str, name: str, target_price: float = None):
        """Add a product to track."""
        self.trackers.append({
            'url': url,
            'name': name,
            'target_price': target_price,
            'created_at': datetime.now().isoformat()
        })
        print(f"📝 Added tracker for: {name}")
    
    def scrape_amazon_price(self, url: str) -> Dict:
        """Scrape Amazon product price."""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Multiple selectors for price (Amazon changes these frequently)
            price_selectors = [
                'span.a-price-whole',
                'span#priceblock_dealprice',
                'span#priceblock_ourprice',
                'span.a-price.a-text-price.a-size-medium.apexPriceToPay',
                'span.a-price-range'
            ]
            
            price_text = None
            for selector in price_selectors:
                price_element = soup.select_one(selector)
                if price_element:
                    price_text = price_element.text.strip()
                    break
            
            # Extract title
            title_element = soup.select_one('#productTitle')
            title = title_element.text.strip() if title_element else "Unknown Product"
            
            # Clean price
            if price_text:
                price_clean = re.sub(r'[^\d.,]', '', price_text)
                try:
                    price = float(price_clean.replace(',', ''))
                except ValueError:
                    price = None
            else:
                price = None
            
            return {
                'title': title,
                'price': price,
                'currency': 'USD',
                'url': url,
                'scraped_at': datetime.now().isoformat(),
                'available': price is not None
            }
            
        except Exception as e:
            print(f"❌ Error scraping Amazon: {e}")
            return {
                'title': 'Error',
                'price': None,
                'currency': 'USD',
                'url': url,
                'scraped_at': datetime.now().isoformat(),
                'available': False,
                'error': str(e)
            }
    
    def scrape_generic_price(self, url: str) -> Dict:
        """Generic price scraper for other sites."""
        try:
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Common price selectors
            price_selectors = [
                '[class*="price"]',
                '[id*="price"]',
                '.price',
                '.cost',
                '.amount'
            ]
            
            price_text = None
            for selector in price_selectors:
                elements = soup.select(selector)
                for element in elements:
                    text = element.text.strip()
                    if re.search(r'[\$€£¥]|\d+\.\d{2}', text):
                        price_text = text
                        break
                if price_text:
                    break
            
            # Extract title
            title_element = soup.select_one('title, h1, [class*="title"], [class*="name"]')
            title = title_element.text.strip() if title_element else urlparse(url).netloc
            
            # Clean price
            price = None
            if price_text:
                price_match = re.search(r'(\d+\.?\d*)', price_text.replace(',', ''))
                if price_match:
                    try:
                        price = float(price_match.group(1))
                    except ValueError:
                        pass
            
            return {
                'title': title,
                'price': price,
                'currency': 'USD',
                'url': url,
                'scraped_at': datetime.now().isoformat(),
                'available': price is not None
            }
            
        except Exception as e:
            print(f"❌ Error scraping {url}: {e}")
            return {
                'title': 'Error',
                'price': None,
                'currency': 'USD',
                'url': url,
                'scraped_at': datetime.now().isoformat(),
                'available': False,
                'error': str(e)
            }
    
    def check_all_prices(self):
        """Check prices for all tracked products."""
        results = []
        
        print(f"🔍 Checking {len(self.trackers)} products...")
        
        for tracker in self.trackers:
            print(f"📊 Checking: {tracker['name']}")
            
            # Determine scraper based on URL
            if 'amazon.com' in tracker['url']:
                result = self.scrape_amazon_price(tracker['url'])
            else:
                result = self.scrape_generic_price(tracker['url'])
            
            result.update({
                'tracked_name': tracker['name'],
                'target_price': tracker.get('target_price')
            })
            
            # Check if target price is met
            if result['price'] and tracker.get('target_price'):
                if result['price'] <= tracker['target_price']:
                    result['price_alert'] = True
                    print(f"🎉 PRICE ALERT! {tracker['name']} is now ${result['price']}")
            
            results.append(result)
            self.random_delay()
        
        return results

class JobScraper(BaseScraper):
    """Scrape job listings from various job boards."""
    
    def scrape_indeed_jobs(self, query: str, location: str = "", limit: int = 50) -> List[Dict]:
        """Scrape job listings from Indeed."""
        jobs = []
        
        try:
            base_url = "https://www.indeed.com/jobs"
            params = {
                'q': query,
                'l': location,
                'limit': min(limit, 50)
            }
            
            response = self.session.get(base_url, params=params)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find job cards
            job_cards = soup.find_all('div', class_='job_seen_beacon')
            
            for card in job_cards[:limit]:
                try:
                    # Extract job details
                    title_element = card.find('h2', class_='jobTitle')
                    title = title_element.text.strip() if title_element else "Unknown"
                    
                    company_element = card.find('span', class_='companyName')
                    company = company_element.text.strip() if company_element else "Unknown"
                    
                    location_element = card.find('div', class_='companyLocation')
                    job_location = location_element.text.strip() if location_element else "Unknown"
                    
                    salary_element = card.find('span', class_='salaryText')
                    salary = salary_element.text.strip() if salary_element else "Not specified"
                    
                    summary_element = card.find('div', class_='summary')
                    summary = summary_element.text.strip() if summary_element else "No description"
                    
                    # Get job URL
                    link_element = title_element.find('a') if title_element else None
                    job_url = urljoin("https://www.indeed.com", link_element['href']) if link_element else ""
                    
                    jobs.append({
                        'title': title,
                        'company': company,
                        'location': job_location,
                        'salary': salary,
                        'summary': summary[:200] + "..." if len(summary) > 200 else summary,
                        'url': job_url,
                        'source': 'Indeed',
                        'scraped_at': datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    print(f"⚠️ Error parsing job card: {e}")
                    continue
            
            print(f"✅ Found {len(jobs)} jobs on Indeed")
            
        except Exception as e:
            print(f"❌ Error scraping Indeed: {e}")
        
        return jobs
    
    def scrape_remote_jobs(self, query: str, limit: int = 30) -> List[Dict]:
        """Scrape remote job listings."""
        jobs = []
        
        try:
            # Using a generic remote job board approach
            base_url = "https://remoteok.io/api"
            
            response = self.session.get(base_url)
            response.raise_for_status()
            
            data = response.json()
            
            # Filter jobs by query
            query_lower = query.lower()
            matching_jobs = [
                job for job in data[1:limit+1]  # Skip first item (metadata)
                if isinstance(job, dict) and (
                    query_lower in job.get('position', '').lower() or
                    query_lower in job.get('description', '').lower() or
                    query_lower in ' '.join(job.get('tags', [])).lower()
                )
            ]
            
            for job in matching_jobs:
                try:
                    jobs.append({
                        'title': job.get('position', 'Unknown'),
                        'company': job.get('company', 'Unknown'),
                        'location': 'Remote',
                        'salary': f"${job.get('salary_min', 0):,} - ${job.get('salary_max', 0):,}" if job.get('salary_min') else "Not specified",
                        'summary': job.get('description', 'No description')[:200] + "...",
                        'url': job.get('url', ''),
                        'tags': ', '.join(job.get('tags', [])),
                        'source': 'RemoteOK',
                        'scraped_at': datetime.now().isoformat()
                    })
                except Exception as e:
                    print(f"⚠️ Error parsing remote job: {e}")
                    continue
            
            print(f"✅ Found {len(jobs)} remote jobs")
            
        except Exception as e:
            print(f"❌ Error scraping remote jobs: {e}")
        
        return jobs

class NewsScraper(BaseScraper):
    """Scrape news articles from various sources."""
    
    def scrape_hacker_news(self, limit: int = 30) -> List[Dict]:
        """Scrape top stories from Hacker News."""
        articles = []
        
        try:
            # Get top stories
            response = self.session.get("https://hacker-news.firebaseio.com/v0/topstories.json")
            response.raise_for_status()
            
            story_ids = response.json()[:limit]
            
            print(f"📰 Fetching {len(story_ids)} Hacker News stories...")
            
            for story_id in story_ids:
                try:
                    story_response = self.session.get(f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json")
                    story_response.raise_for_status()
                    
                    story = story_response.json()
                    
                    if story and story.get('type') == 'story':
                        articles.append({
                            'title': story.get('title', 'No title'),
                            'author': story.get('by', 'Unknown'),
                            'url': story.get('url', f"https://news.ycombinator.com/item?id={story_id}"),
                            'score': story.get('score', 0),
                            'comments': story.get('descendants', 0),
                            'time': datetime.fromtimestamp(story.get('time', 0)).isoformat() if story.get('time') else '',
                            'source': 'Hacker News',
                            'scraped_at': datetime.now().isoformat()
                        })
                    
                    self.random_delay()
                    
                except Exception as e:
                    print(f"⚠️ Error fetching story {story_id}: {e}")
                    continue
            
            print(f"✅ Scraped {len(articles)} Hacker News articles")
            
        except Exception as e:
            print(f"❌ Error scraping Hacker News: {e}")
        
        return articles
    
    def scrape_reddit_posts(self, subreddit: str, limit: int = 25) -> List[Dict]:
        """Scrape posts from a Reddit subreddit."""
        posts = []
        
        try:
            url = f"https://www.reddit.com/r/{subreddit}/hot.json"
            headers = {'User-Agent': 'WebScraperToolkit/1.0'}
            
            response = self.session.get(url, headers=headers)
            response.raise_for_status()
            
            data = response.json()
            
            for post_data in data['data']['children'][:limit]:
                post = post_data['data']
                
                posts.append({
                    'title': post.get('title', 'No title'),
                    'author': post.get('author', 'Unknown'),
                    'url': f"https://reddit.com{post.get('permalink', '')}",
                    'external_url': post.get('url', ''),
                    'score': post.get('score', 0),
                    'comments': post.get('num_comments', 0),
                    'subreddit': post.get('subreddit', subreddit),
                    'created': datetime.fromtimestamp(post.get('created_utc', 0)).isoformat() if post.get('created_utc') else '',
                    'selftext': post.get('selftext', '')[:200] + "..." if len(post.get('selftext', '')) > 200 else post.get('selftext', ''),
                    'source': 'Reddit',
                    'scraped_at': datetime.now().isoformat()
                })
            
            print(f"✅ Scraped {len(posts)} posts from r/{subreddit}")
            
        except Exception as e:
            print(f"❌ Error scraping Reddit: {e}")
        
        return posts

class WebScraperToolkit:
    """Main toolkit class that orchestrates all scrapers."""
    
    def __init__(self):
        self.price_monitor = PriceMonitor()
        self.job_scraper = JobScraper()
        self.news_scraper = NewsScraper()
    
    def interactive_mode(self):
        """Run the toolkit in interactive mode."""
        print("\n🕷️  Web Scraper Toolkit")
        print("=" * 50)
        
        while True:
            print("\n🔧 Available Tools:")
            print("1. 💰 Price Monitor")
            print("2. 💼 Job Scraper")
            print("3. 📰 News Scraper")
            print("4. ⚙️  Scheduler")
            print("5. 🔍 View Results")
            print("6. ❌ Exit")
            
            choice = input("\nSelect a tool (1-6): ").strip()
            
            if choice == "1":
                self._price_monitor_menu()
            elif choice == "2":
                self._job_scraper_menu()
            elif choice == "3":
                self._news_scraper_menu()
            elif choice == "4":
                self._scheduler_menu()
            elif choice == "5":
                self._view_results()
            elif choice == "6":
                print("👋 Goodbye!")
                break
            else:
                print("Invalid choice. Please try again.")
    
    def _price_monitor_menu(self):
        """Price monitor interactive menu."""
        print("\n💰 Price Monitor")
        print("1. Add product to track")
        print("2. Check all prices")
        print("3. Back to main menu")
        
        choice = input("Select option: ").strip()
        
        if choice == "1":
            url = input("Product URL: ").strip()
            name = input("Product name: ").strip()
            target_str = input("Target price (optional): ").strip()
            target_price = float(target_str) if target_str else None
            
            self.price_monitor.add_product(url, name, target_price)
            
        elif choice == "2":
            if not self.price_monitor.trackers:
                print("No products being tracked. Add some first!")
                return
            
            results = self.price_monitor.check_all_prices()
            self.price_monitor.save_to_json(results, "price_check")
            self.price_monitor.save_to_csv(results, "price_check")
    
    def _job_scraper_menu(self):
        """Job scraper interactive menu."""
        print("\n💼 Job Scraper")
        print("1. Search Indeed jobs")
        print("2. Search remote jobs")
        print("3. Back to main menu")
        
        choice = input("Select option: ").strip()
        
        if choice == "1":
            query = input("Job query (e.g., 'python developer'): ").strip()
            location = input("Location (optional): ").strip()
            limit = int(input("Number of jobs (default 25): ").strip() or "25")
            
            jobs = self.job_scraper.scrape_indeed_jobs(query, location, limit)
            if jobs:
                self.job_scraper.save_to_json(jobs, f"indeed_jobs_{query.replace(' ', '_')}")
                self.job_scraper.save_to_csv(jobs, f"indeed_jobs_{query.replace(' ', '_')}")
            
        elif choice == "2":
            query = input("Job query: ").strip()
            limit = int(input("Number of jobs (default 20): ").strip() or "20")
            
            jobs = self.job_scraper.scrape_remote_jobs(query, limit)
            if jobs:
                self.job_scraper.save_to_json(jobs, f"remote_jobs_{query.replace(' ', '_')}")
                self.job_scraper.save_to_csv(jobs, f"remote_jobs_{query.replace(' ', '_')}")
    
    def _news_scraper_menu(self):
        """News scraper interactive menu."""
        print("\n📰 News Scraper")
        print("1. Hacker News top stories")
        print("2. Reddit subreddit posts")
        print("3. Back to main menu")
        
        choice = input("Select option: ").strip()
        
        if choice == "1":
            limit = int(input("Number of stories (default 30): ").strip() or "30")
            articles = self.news_scraper.scrape_hacker_news(limit)
            if articles:
                self.news_scraper.save_to_json(articles, "hacker_news")
                self.news_scraper.save_to_csv(articles, "hacker_news")
        
        elif choice == "2":
            subreddit = input("Subreddit name (without r/): ").strip()
            limit = int(input("Number of posts (default 25): ").strip() or "25")
            
            posts = self.news_scraper.scrape_reddit_posts(subreddit, limit)
            if posts:
                self.news_scraper.save_to_json(posts, f"reddit_{subreddit}")
                self.news_scraper.save_to_csv(posts, f"reddit_{subreddit}")
    
    def _scheduler_menu(self):
        """Scheduler menu for automated scraping."""
        print("\n⚙️  Scheduler")
        print("Coming soon! This will allow you to schedule scraping tasks.")
    
    def _view_results(self):
        """View previously scraped results."""
        output_dir = Path("scraped_data")
        if not output_dir.exists():
            print("No results directory found.")
            return
        
        files = list(output_dir.glob("*.json"))
        if not files:
            print("No result files found.")
            return
        
        print("\n📊 Recent Results:")
        for i, file in enumerate(files[-10:], 1):  # Show last 10 files
            print(f"{i}. {file.name}")
        
        choice = input("\nEnter file number to view (or press Enter to go back): ").strip()
        
        if choice.isdigit():
            try:
                file_idx = int(choice) - 1
                if 0 <= file_idx < len(files[-10:]):
                    with open(files[-10:][file_idx], 'r') as f:
                        data = json.load(f)
                    print(f"\n📄 Contents of {files[-10:][file_idx].name}:")
                    print(json.dumps(data[:3], indent=2))  # Show first 3 items
                    if len(data) > 3:
                        print(f"... and {len(data) - 3} more items")
            except (ValueError, IndexError):
                print("Invalid selection.")

def main():
    parser = argparse.ArgumentParser(description='Web Scraper Toolkit')
    parser.add_argument('--mode', choices=['interactive', 'price', 'jobs', 'news'], 
                       default='interactive', help='Scraper mode')
    parser.add_argument('--query', help='Search query')
    parser.add_argument('--limit', type=int, default=25, help='Number of results')
    
    args = parser.parse_args()
    
    toolkit = WebScraperToolkit()
    
    if args.mode == 'interactive':
        toolkit.interactive_mode()
    else:
        print(f"Running in {args.mode} mode...")
        # Add CLI mode implementations here

if __name__ == "__main__":
    main()

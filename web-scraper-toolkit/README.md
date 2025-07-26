# 🕷️ Web Scraper Toolkit

A comprehensive Python-based web scraping suite that automates data collection from various online sources. Perfect for price monitoring, job hunting, news aggregation, and general web data extraction.

## ✨ Features

### 💰 **Price Monitor**
- Track product prices from Amazon and other e-commerce sites
- Set target price alerts
- Automatic price history tracking
- Support for multiple currencies
- Generic price detection for any website

### 💼 **Job Scraper**
- Search Indeed for job listings
- Find remote jobs from RemoteOK
- Filter by location, salary, and keywords
- Export results to CSV/JSON
- Track job market trends

### 📰 **News Scraper**
- Hacker News top stories aggregation
- Reddit subreddit post collection
- Trending topics detection
- Social media sentiment analysis
- RSS feed integration (coming soon)

### 🛠️ **Core Features**
- **Respectful Scraping**: Built-in delays and rotation
- **Multiple Output Formats**: JSON, CSV, and more
- **Interactive & CLI Modes**: User-friendly interface or automation
- **Fake User Agents**: Avoid detection with rotating headers
- **Error Handling**: Robust error recovery and logging
- **Modular Design**: Easy to extend with new scrapers

## 📦 Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Optional: ChromeDriver for Selenium

For JavaScript-heavy sites (future features):
```bash
# macOS with Homebrew
brew install chromedriver

# Or download from: https://chromedriver.chromium.org/
```

## 🚀 Usage

### Interactive Mode (Recommended)

```bash
python web_scraper_toolkit.py
```

This launches an interactive menu where you can:
- Set up price monitoring
- Search for jobs
- Scrape news articles
- View previous results

### Command Line Mode

```bash
# Price monitoring
python web_scraper_toolkit.py --mode price --query "product_name"

# Job searching
python web_scraper_toolkit.py --mode jobs --query "python developer" --limit 30

# News scraping
python web_scraper_toolkit.py --mode news --limit 50
```

## 📊 Example Usage

### Price Monitoring Example

```python
from web_scraper_toolkit import WebScraperToolkit

# Create toolkit instance
toolkit = WebScraperToolkit()

# Add products to monitor
toolkit.price_monitor.add_product(
    url="https://amazon.com/product-url",
    name="MacBook Pro",
    target_price=1500.00
)

# Check all prices
results = toolkit.price_monitor.check_all_prices()
```

Output:
```json
[
  {
    "title": "MacBook Pro 13-inch",
    "price": 1299.99,
    "currency": "USD",
    "tracked_name": "MacBook Pro",
    "target_price": 1500.00,
    "price_alert": true,
    "scraped_at": "2025-01-26T15:30:00"
  }
]
```

### Job Scraping Example

```python
# Search for remote Python jobs
jobs = toolkit.job_scraper.scrape_remote_jobs("python", limit=25)

# Search Indeed for local jobs
local_jobs = toolkit.job_scraper.scrape_indeed_jobs(
    query="data scientist",
    location="San Francisco, CA",
    limit=50
)
```

### News Aggregation Example

```python
# Get Hacker News top stories
hn_stories = toolkit.news_scraper.scrape_hacker_news(limit=30)

# Get Reddit posts from specific subreddit
reddit_posts = toolkit.news_scraper.scrape_reddit_posts("python", limit=25)
```

## 📁 Output Structure

All scraped data is saved in the `scraped_data/` directory:

```
scraped_data/
├── price_check_20250126_1530.json
├── price_check_20250126_1530.csv
├── indeed_jobs_python_developer_20250126_1535.json
├── remote_jobs_python_20250126_1540.json
├── hacker_news_20250126_1545.json
└── reddit_python_20250126_1550.json
```

## 🔧 Configuration

### Delay Settings
Adjust scraping delays to be more respectful:

```python
# Slower scraping (2-5 seconds between requests)
scraper = BaseScraper(delay_range=(2, 5))

# Faster scraping (0.5-1 seconds) - use cautiously
scraper = BaseScraper(delay_range=(0.5, 1))
```

### User Agent Rotation
```python
# Enable fake user agent rotation (default)
scraper = BaseScraper(use_fake_agent=True)

# Use static user agent
scraper = BaseScraper(use_fake_agent=False)
```

## 📋 Supported Sites

### Price Monitoring
- ✅ Amazon (comprehensive)
- ✅ Generic e-commerce (heuristic-based)
- 🔄 eBay (coming soon)
- 🔄 Best Buy (coming soon)
- 🔄 Walmart (coming soon)

### Job Boards
- ✅ Indeed
- ✅ RemoteOK
- 🔄 LinkedIn (coming soon)
- 🔄 Stack Overflow Jobs (coming soon)
- 🔄 AngelList (coming soon)

### News Sources
- ✅ Hacker News
- ✅ Reddit
- 🔄 RSS Feeds (coming soon)
- 🔄 Twitter trending (coming soon)

## ⚡ Advanced Features

### Scheduled Scraping
```python
import schedule
import time

# Schedule price checks every hour
schedule.every().hour.do(toolkit.price_monitor.check_all_prices)

# Schedule job searches daily
schedule.every().day.at("09:00").do(
    lambda: toolkit.job_scraper.scrape_indeed_jobs("python developer")
)

while True:
    schedule.run_pending()
    time.sleep(60)
```

### Data Analysis
```python
import pandas as pd

# Load and analyze scraped data
df = pd.read_csv('scraped_data/price_check_latest.csv')

# Price trend analysis
price_trends = df.groupby('tracked_name')['price'].describe()
print(price_trends)
```

## 🛡️ Best Practices

### Ethical Scraping
- **Respect robots.txt**: Check site policies before scraping
- **Use delays**: Don't overwhelm servers with rapid requests
- **Cache results**: Avoid re-scraping the same data
- **User agents**: Rotate to appear more natural
- **Rate limiting**: Stay within reasonable request limits

### Error Handling
The toolkit includes comprehensive error handling:
- Network timeouts and retries
- HTML parsing failures
- Rate limiting detection
- Graceful degradation when selectors change

### Data Quality
- **Validation**: All scraped data is validated before saving
- **Deduplication**: Automatic removal of duplicate entries
- **Timestamps**: All data includes scraping timestamps
- **Source attribution**: Clear indication of data sources

## 🚫 Limitations & Disclaimers

- **Legal Compliance**: Ensure you comply with website terms of service
- **Rate Limits**: Some sites may block rapid requests
- **Structure Changes**: Websites may change, breaking scrapers
- **Personal Use**: This tool is intended for personal/educational use
- **No Warranty**: Use at your own risk

## 🔄 Future Enhancements

- **Real Estate**: Zillow, Redfin property monitoring
- **Stock Prices**: Financial data aggregation
- **Social Media**: Twitter, Instagram content scraping
- **Academic**: Research paper and citation tracking
- **E-commerce**: Comprehensive shopping comparison
- **API Integration**: REST API for external access

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- New scraper modules
- Better error handling
- Performance optimizations
- Additional output formats
- Scheduling improvements

## 📄 Legal Note

This tool is for educational and personal use only. Always respect website terms of service, robots.txt files, and applicable laws. The developers are not responsible for misuse of this tool.

---

**Happy Scraping! 🕷️**

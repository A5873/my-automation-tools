#!/usr/bin/env python3
"""
Web Scraper Toolkit Examples
Demonstrates various use cases and features of the toolkit.
"""

from web_scraper_toolkit import WebScraperToolkit, PriceMonitor, JobScraper, NewsScraper
import json
import time

def price_monitoring_example():
    """Example: Monitor product prices."""
    print("💰 Price Monitoring Example")
    print("=" * 40)
    
    monitor = PriceMonitor()
    
    # Add some products to track
    monitor.add_product(
        url="https://www.amazon.com/dp/B08N5WRWNW",  # Example: Echo Dot
        name="Amazon Echo Dot",
        target_price=30.00
    )
    
    # Check prices
    results = monitor.check_all_prices()
    
    # Display results
    for result in results:
        print(f"📊 {result['tracked_name']}: ${result['price']}")
        if result.get('price_alert'):
            print("🎉 PRICE ALERT! Target price reached!")
    
    # Save results
    monitor.save_to_json(results, "example_price_check")
    print("✅ Results saved to scraped_data/")

def job_search_example():
    """Example: Search for jobs."""
    print("\n💼 Job Search Example")
    print("=" * 40)
    
    scraper = JobScraper()
    
    # Search for Python developer jobs
    print("🔍 Searching for Python developer jobs...")
    jobs = scraper.scrape_indeed_jobs("python developer", "remote", limit=10)
    
    # Display results
    for job in jobs[:3]:  # Show first 3
        print(f"🏢 {job['company']}: {job['title']}")
        print(f"📍 {job['location']}")
        print(f"💰 {job['salary']}")
        print()
    
    # Search remote jobs
    print("🌐 Searching remote jobs...")
    remote_jobs = scraper.scrape_remote_jobs("python", limit=5)
    
    for job in remote_jobs[:2]:  # Show first 2
        print(f"🏢 {job['company']}: {job['title']}")
        print(f"🏷️  Tags: {job.get('tags', 'N/A')}")
        print()
    
    # Save results
    if jobs:
        scraper.save_to_json(jobs, "example_indeed_jobs")
    if remote_jobs:
        scraper.save_to_json(remote_jobs, "example_remote_jobs")

def news_aggregation_example():
    """Example: Aggregate news from multiple sources."""
    print("\n📰 News Aggregation Example")
    print("=" * 40)
    
    scraper = NewsScraper()
    
    # Get Hacker News stories
    print("📖 Fetching Hacker News stories...")
    hn_stories = scraper.scrape_hacker_news(limit=5)
    
    print("🔥 Top Hacker News Stories:")
    for story in hn_stories[:3]:
        print(f"📰 {story['title']}")
        print(f"👤 By: {story['author']} | 🔼 Score: {story['score']}")
        print()
    
    # Get Reddit posts
    print("🤖 Fetching Reddit posts from r/python...")
    reddit_posts = scraper.scrape_reddit_posts("python", limit=5)
    
    print("🔥 Top r/python Posts:")
    for post in reddit_posts[:3]:
        print(f"📰 {post['title']}")
        print(f"👤 By: u/{post['author']} | 🔼 Score: {post['score']}")
        print()
    
    # Save results
    if hn_stories:
        scraper.save_to_json(hn_stories, "example_hacker_news")
    if reddit_posts:
        scraper.save_to_json(reddit_posts, "example_reddit_python")

def comprehensive_example():
    """Example: Use the full toolkit."""
    print("\n🚀 Comprehensive Toolkit Example")
    print("=" * 40)
    
    toolkit = WebScraperToolkit()
    
    # Quick data collection
    print("📊 Collecting data from multiple sources...")
    
    # Get tech news
    hn_stories = toolkit.news_scraper.scrape_hacker_news(limit=5)
    
    # Get job market data
    jobs = toolkit.job_scraper.scrape_remote_jobs("developer", limit=5)
    
    # Summary report
    print("\n📈 Data Collection Summary:")
    print(f"📰 News articles collected: {len(hn_stories)}")
    print(f"💼 Job listings found: {len(jobs)}")
    
    # Analyze data
    if jobs:
        companies = [job['company'] for job in jobs]
        unique_companies = len(set(companies))
        print(f"🏢 Unique companies: {unique_companies}")
    
    if hn_stories:
        avg_score = sum(story['score'] for story in hn_stories) / len(hn_stories)
        print(f"📊 Average HN story score: {avg_score:.1f}")

def data_analysis_example():
    """Example: Analyze scraped data."""
    print("\n📊 Data Analysis Example")
    print("=" * 40)
    
    try:
        import pandas as pd
        
        # Create sample data for analysis
        scraper = JobScraper()
        jobs = scraper.scrape_remote_jobs("python", limit=10)
        
        if jobs:
            # Convert to DataFrame
            df = pd.DataFrame(jobs)
            
            print("📈 Job Market Analysis:")
            print(f"Total jobs analyzed: {len(df)}")
            
            # Company analysis
            company_counts = df['company'].value_counts().head(5)
            print("\n🏢 Top hiring companies:")
            for company, count in company_counts.items():
                print(f"   {company}: {count} positions")
            
            # Save analysis
            df.to_csv('scraped_data/job_analysis.csv', index=False)
            print("\n✅ Analysis saved to scraped_data/job_analysis.csv")
        
    except ImportError:
        print("📊 Install pandas for data analysis: pip install pandas")

def scheduled_scraping_example():
    """Example: Scheduled scraping setup."""
    print("\n⏰ Scheduled Scraping Example")
    print("=" * 40)
    
    try:
        import schedule
        
        def daily_job_check():
            """Daily job market check."""
            scraper = JobScraper()
            jobs = scraper.scrape_remote_jobs("python", limit=20)
            if jobs:
                scraper.save_to_json(jobs, "daily_python_jobs")
                print(f"✅ Daily check: Found {len(jobs)} Python jobs")
        
        def weekly_price_check():
            """Weekly price monitoring."""
            monitor = PriceMonitor()
            # You would add your products here
            print("💰 Weekly price check scheduled")
        
        # Schedule tasks
        schedule.every().day.at("09:00").do(daily_job_check)
        schedule.every().monday.at("08:00").do(weekly_price_check)
        
        print("📅 Scheduled tasks:")
        print("   • Daily job check at 9:00 AM")
        print("   • Weekly price check on Mondays at 8:00 AM")
        print("\n⚠️  To run scheduler, use:")
        print("   python -c \"import schedule; import time; [schedule.run_pending() or time.sleep(60) for _ in iter(int, 1)]\"")
        
    except ImportError:
        print("⏰ Install schedule for automation: pip install schedule")

def main():
    """Run all examples."""
    print("🕷️  Web Scraper Toolkit Examples")
    print("=" * 50)
    print("This script demonstrates various features of the toolkit.")
    print("Each example shows different scraping capabilities.\n")
    
    examples = [
        ("News Aggregation", news_aggregation_example),
        ("Job Search", job_search_example),
        ("Price Monitoring", price_monitoring_example),
        ("Comprehensive Usage", comprehensive_example),
        ("Data Analysis", data_analysis_example),
        ("Scheduled Scraping", scheduled_scraping_example)
    ]
    
    for name, func in examples:
        try:
            func()
            time.sleep(2)  # Brief pause between examples
        except KeyboardInterrupt:
            print("\n\n👋 Examples interrupted by user")
            break
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            continue
    
    print("\n🎉 Examples completed!")
    print("📁 Check the 'scraped_data' directory for results.")
    print("📚 See README.md for more detailed usage instructions.")

if __name__ == "__main__":
    main()

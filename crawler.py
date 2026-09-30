import asyncio
import json
from crawl4ai import AsyncWebCrawler

# Base URLs
MODULE_URL = "https://zerodha.com/varsity/module/introduction-to-stock-markets/"
BASE_URL = "https://zerodha.com"

async def get_chapter_links(crawler):
    """Extracts all chapter links from the main module page."""
    print(f"Fetching module page: {MODULE_URL}")
    result = await crawler.arun(url=MODULE_URL)
    
    chapter_links = []
    # crawl4ai extracts internal links automatically
    if result.links and "internal" in result.links:
        for link_obj in result.links["internal"]:
            href = link_obj.get('href', '')
            if "/varsity/chapter/" in href:
                full_url = href if href.startswith("http") else BASE_URL + href
                if full_url not in chapter_links:
                    chapter_links.append(full_url)
                    
    print(f"Found {len(chapter_links)} chapters.")
    return chapter_links

async def scrape_chapters():
    documents = []
    
    async with AsyncWebCrawler(verbose=True) as crawler:
        chapter_links = await get_chapter_links(crawler)
        
        for i, link in enumerate(chapter_links, 1):
            print(f"Scraping chapter {i}/{len(chapter_links)}: {link}")
            
            # Extract clean markdown
            result = await crawler.arun(url=link, bypass_cache=True)
            
            if result.success:
                # FIX 1: Safely handle the title (It's now inside metadata in v0.9+)
                title = f"Chapter {i}"
                if result.metadata and isinstance(result.metadata, dict):
                    title = result.metadata.get("title", title)
                
                # FIX 2: Handle Markdown format safely
                # In v0.9+, result.markdown can be an object or a plain string
                if hasattr(result.markdown, "raw_markdown"):
                    content = result.markdown.raw_markdown
                else:
                    content = result.markdown or ""
                
                documents.append({
                    "url": link,
                    "title": title,
                    "content": content
                })
            else:
                print(f"Failed to scrape: {link}")
                
            await asyncio.sleep(1) # Polite delay
            
    # Save the extracted data
    with open('varsity_docs.json', 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=4)
        
    print(f"\n✅ Scraping complete! Saved {len(documents)} documents to varsity_docs.json")

if __name__ == "__main__":
    asyncio.run(scrape_chapters())
"""Inspect OddsPortal /matches/ page structure to understand rendering model."""

from algobet.scraper import OddsPortalScraper

URL = "https://www.oddsportal.com/matches/"

with OddsPortalScraper(headless=True) as scraper:
    scraper.navigate_to_upcoming(URL)

    info = scraper._page.evaluate("""
        () => {
            const gameRows = document.querySelectorAll('div[data-testid="game-row"]').length;
            const sportTabs = Array.from(document.querySelectorAll('a[href*="/matches/"]'))
                .map(a => ({ text: a.innerText.trim(), href: a.getAttribute('href') }))
                .filter(a => a.text && a.href);
            const dateTabs = Array.from(document.querySelectorAll('a[href*="/#"]'))
                .map(a => ({ text: a.innerText.trim(), href: a.getAttribute('href') }))
                .filter(a => a.text);
            const totalCountEl = document.querySelector('[data-testid="total-count"], .total-count');
            const bodyHeight = document.body.scrollHeight;
            const viewportHeight = window.innerHeight;
            return { gameRows, sportTabs, dateTabs, bodyHeight, viewportHeight,
                     totalCount: totalCountEl ? totalCountEl.innerText : null };
        }
    """)

    print(f"Game rows in DOM:   {info['gameRows']}")
    print(
        f"Body scroll height: {info['bodyHeight']}px  (viewport: {info['viewportHeight']}px)"
    )
    print(f"Total count elem:   {info['totalCount']}")
    print(f"\nSport/section tabs ({len(info['sportTabs'])}):")
    for t in info["sportTabs"][:15]:
        print(f"  {t['text']:20s}  {t['href']}")
    print(f"\nDate tabs ({len(info['dateTabs'])}):")
    for t in info["dateTabs"][:10]:
        print(f"  {t['text']:20s}  {t['href']}")

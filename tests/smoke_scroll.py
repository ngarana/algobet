"""Smoke test: verify infinite scroll loads more matches than initial page render."""

import logging
import sys

from algobet.scraper import OddsPortalScraper

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

URL = "https://www.oddsportal.com/matches/"

with OddsPortalScraper(headless=True) as scraper:
    scraper.navigate_to_upcoming(URL)

    rows_before = scraper._page.evaluate(
        "document.querySelectorAll('div[data-testid=\"game-row\"]').length"
    )
    print(f"\nRows after page load (before scroll): {rows_before}")

    scraper._scroll_for_lazy_content()

    rows_after = scraper._page.evaluate(
        "document.querySelectorAll('div[data-testid=\"game-row\"]').length"
    )
    print(f"Rows after full scroll:               {rows_after}")

    # Scrape without any filtering to see raw totals
    matches_all = scraper.scrape_upcoming_matches(only_future_matches=False)
    matches_with_odds = [m for m in matches_all if m.get("odds_home")]

    print(f"\nTotal parsed (no filter):  {len(matches_all)}")
    print(f"Matches with odds:         {len(matches_with_odds)}")

    if rows_after > rows_before:
        print(
            f"\n✅ Infinite scroll loaded {rows_after - rows_before} additional rows."
        )
    else:
        print(f"\n⚠️  All {rows_after} rows were loaded on initial page render.")

    if matches_all:
        s = matches_all[0]
        print(
            f"\nSample: {s['home_team']} vs {s['away_team']} | {s['tournament_name']} ({s['country']})"
        )
    else:
        print("No matches parsed.")
        sys.exit(1)

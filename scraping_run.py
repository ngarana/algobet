from playwright.sync_api import sync_playwright

captured = []
with sync_playwright() as p:
    browser = p.chromium.launch(headless=True)
    page = browser.new_page()

    def on_request(req):
        if "/api/v1/scraping/by-date" in req.url and req.method == "POST":
            captured.append(
                {"url": req.url, "method": req.method, "body": req.post_data}
            )

    page.on("request", on_request)

    page.goto(
        "http://127.0.0.1:3001/scraping", wait_until="networkidle", timeout=120000
    )
    page.get_by_role("button", name="BY DATE").click()
    page.get_by_label("Date").fill("2026-04-21")
    page.get_by_role("button", name="Start Fetch").click()
    page.wait_for_timeout(5000)

    print("REQUESTS:", captured)
    body_text = page.locator("body").inner_text()
    for needle in [
        "2026-04-21",
        "WebSocket connection established",
        "JOB MONITOR",
    ]:
        print(f"HAS::{needle}::{needle in body_text}")
    print("BODY_SNIPPET_START")
    print(body_text[:5000])
    print("BODY_SNIPPET_END")

    browser.close()

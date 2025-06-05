from playwright.async_api import async_playwright
from urllib.parse import urlparse


async def fetch_rendered_html_playwright(url: str, timeout: int = 15000) -> str:
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-web-security",
                    "--disable-features=IsolateOrigins,site-per-process",
                ],
            )

            domain = urlparse(url).netloc.replace("www.", "")

            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
                java_script_enabled=True,
                viewport={"width": 1280, "height": 720},
            )

            await context.add_cookies(
                [
                    {"name": "RTA", "value": "1", "domain": f".{domain}", "path": "/"},
                    {
                        "name": "age_verified",
                        "value": "1",
                        "domain": f".{domain}",
                        "path": "/",
                    },
                ]
            )

            page = await context.new_page()
            await page.goto(url, timeout=timeout)
            await page.wait_for_timeout(3000)

            html = await page.content()
            await browser.close()
            return html
    except Exception as e:
        print(f"[Playwright Error] {url}: {e}")
        return ""

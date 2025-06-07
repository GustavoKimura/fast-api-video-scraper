# === 📦 IMPORTS ===
import os, random, hashlib, asyncio, re, time, psutil, torch, open_clip, tldextract, trafilatura
from urllib.parse import urlparse, urlunparse, urljoin
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
from bs4 import BeautifulSoup, Tag
from readability import Document
from langdetect import DetectorFactory, detect
from deep_translator import GoogleTranslator
from aiohttp import ClientTimeout, ClientSession
from keybert import KeyBERT
from playwright.async_api import async_playwright
from boilerpy3 import extractors
from numpy import dot
from numpy.linalg import norm
from collections import defaultdict, Counter
from typing import List, Tuple
from itertools import combinations
from difflib import SequenceMatcher


# === 🔒 DOMAIN CONCURRENCY CONTROL ===
def get_domain_concurrency():
    cores = os.cpu_count() or 4
    ram_gb = psutil.virtual_memory().total // 1_073_741_824

    base = max(2, min((cores // 2), 10))

    if ram_gb >= 16:
        base += 2
    elif ram_gb <= 4:
        base = max(2, base - 1)

    return base


domain_counters = defaultdict(lambda: asyncio.Semaphore(get_domain_concurrency()))


# === ⚙️ CONFIGURATION ===
def dynamic_parallel_task_limit():
    cores = os.cpu_count() or 4
    ram_gb = psutil.virtual_memory().total // 1_073_741_824

    multiplier = 12 if ram_gb >= 16 else 8
    return min(cores * multiplier, 512)


MAX_PARALLEL_TASKS = dynamic_parallel_task_limit()
LINKS_TO_SCRAP = 10
SUMMARIES = 5

DetectorFactory.seed = 0
timeout_obj = ClientTimeout(total=5)
content_extractor = extractors.LargestContentExtractor()
SEARXNG_BASE_URL = "http://searxng:8080/search"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (X11; Linux x86_64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
]

BLOCKED_DOMAINS = {
    "facebook.com",
    "twitter.com",
    "instagram.com",
    "youtube.com",
    "linkedin.com",
    "pinterest.com",
    "tiktok.com",
    "quora.com",
    "fandom.com",
    "wikia.org",
    "wikihow.com",
    "stackoverflow.com",
    "github.com",
    "reuters.com",
    "bloomberg.com",
    "marketwatch.com",
}

LANGUAGES_BLACKLIST = {"da", "so", "tl", "nl", "sv", "af", "el"}


# === 🧠 MODELS ===
class OpenCLIPEmbedder:
    def __init__(
        self, model_name="ViT-B-32", pretrained="laion2b_s34b_b79k", device=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)

    def encode(self, text: str):
        tokens = self.tokenizer(text).to(self.device)
        with torch.no_grad():
            features = self.model.encode_text(tokens).float()
        return features.cpu().numpy()[0]


model_embed = OpenCLIPEmbedder(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k")
kw_model = KeyBERT("sentence-transformers/all-MiniLM-L6-v2")


def expand_query_semantically(query: str, top_n: int = 5) -> List[str]:
    raw_keywords = kw_model.extract_keywords(
        query,
        keyphrase_ngram_range=(1, 3),
        use_mmr=True,
        diversity=0.8,
        top_n=top_n * 2,
    )
    query_vec = model_embed.encode(query)
    filtered = []

    for kw, _ in raw_keywords:
        kw = str(kw).strip()
        kw_vec = model_embed.encode(kw)
        sim = cosine_sim(query_vec, kw_vec)

        if not any(
            SequenceMatcher(None, kw.lower(), existing_kw.lower()).ratio() >= 0.85
            for existing_kw, _ in filtered
        ):
            filtered.append((kw, sim))

    top_keywords = sorted(filtered, key=lambda x: x[1], reverse=True)[:top_n]
    return [query] + [kw for kw, _ in top_keywords]


# === 🧩 SEMANTIC RANKING ===
def cosine_sim(a, b):
    return float(dot(a, b) / (norm(a) * norm(b) + 1e-8))


def rank_by_similarity(results, query, min_duration=30, max_duration=1800):
    query_embed = model_embed.encode(query)
    query_tags = {kw for kw, _ in extract_tags(query)}
    tag_boosts = boost_by_tag_cooccurrence(results)
    seen_domains = set()

    final = []
    for r in results:
        if r.get("duration"):
            dur_secs = duration_to_seconds(r["duration"])
            if not (min_duration <= dur_secs <= max_duration):
                continue

        score = 0.0
        if r.get("tags"):
            tag_text = " ".join(r["tags"])
            tag_embed = model_embed.encode(tag_text)
            sim = cosine_sim(query_embed, tag_embed)
            result_tags = {
                re.sub(r"[^a-z0-9]", "", tag.lower()) for tag in r.get("tags", [])
            }
            query_tags_norm = {
                re.sub(r"[^a-z0-9]", "", tag.lower()) for tag in query_tags
            }
            overlap = len(query_tags_norm & result_tags)
            boost = sum(tag_boosts.get(tag, 0) for tag in r["tags"])
            score = sim + 0.05 * overlap + boost

        domain = get_main_domain(r["url"])
        if domain in seen_domains:
            score *= 0.85
        else:
            seen_domains.add(domain)

        r["score"] = score
        final.append(r)

    return sorted(final, key=lambda x: x["score"], reverse=True)


def rank_deep_links(links, query_embed):
    scored = []
    for link in links:
        snippet = re.sub(r"[-_/]", " ", link.split("/")[-1])
        embed = model_embed.encode(snippet)
        sim = cosine_sim(query_embed, embed)
        scored.append((sim, link))
    scored.sort(reverse=True)
    return [link for _, link in scored[:5]]


# === 🔧 UTILITIES ===
def get_user_agent():
    return random.choice(USER_AGENTS)


def normalize_url(url):
    return urlunparse(urlparse(url)._replace(query="", fragment=""))


def get_main_domain(url):
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain


def is_valid_link(url):
    domain = get_main_domain(url).lower()
    if domain in BLOCKED_DOMAINS:
        return False
    if not url.startswith(("http://", "https://")):
        return False
    if any(c in url for c in ['"', "'", "\\", " "]):
        return False
    if url.split("?")[0].split("#")[0].split(".")[-1] in [
        "pdf",
        "doc",
        "xls",
        "zip",
        "rar",
        "ppt",
    ]:
        return False
    return True


def detect_language(text):
    try:
        lang = detect(text)
        return lang if lang not in LANGUAGES_BLACKLIST else "en"
    except:
        return "en"


def translate_text(text, target="en"):
    try:
        return GoogleTranslator(source="auto", target=target).translate(text)
    except:
        return text


def duration_to_seconds(duration_str: str):
    if not duration_str or not isinstance(duration_str, str):
        return 0

    parts = duration_str.strip().split(":")
    try:
        parts = [int(p) for p in parts]
        if len(parts) == 3:
            hours, minutes, seconds = parts
        elif len(parts) == 2:
            hours = 0
            minutes, seconds = parts
        elif len(parts) == 1:
            hours = 0
            minutes = 0
            seconds = parts[0]
        else:
            return 0
        return hours * 3600 + minutes * 60 + seconds
    except ValueError:
        return 0


def extract_tags(text: str, top_n: int = 10) -> List[Tuple[str, float]]:
    clean = re.sub(r"[^a-zA-Z0-9\s]", "", text).lower()

    raw_results = kw_model.extract_keywords(
        clean, keyphrase_ngram_range=(1, 3), use_mmr=True, diversity=0.7, top_n=top_n
    )  # type: ignore

    if raw_results and isinstance(raw_results[0], list):
        flat: List[Tuple[str, float]] = []
        for sublist in raw_results:
            flat.extend(
                item for item in sublist if isinstance(item, tuple) and len(item) == 2
            )
        return flat

    if isinstance(raw_results, list) and all(
        isinstance(t, tuple) and len(t) == 2 for t in raw_results
    ):
        return raw_results  # type: ignore

    return []


def boost_by_tag_cooccurrence(results):
    co_pairs = Counter()
    for r in results:
        tags = list(set(r.get("tags", [])))
        for a, b in combinations(sorted(tags), 2):
            co_pairs[(a, b)] += 1

    tag_boosts = {}
    for (a, b), count in co_pairs.items():
        if count >= 2:
            tag_boosts.setdefault(a, 0)
            tag_boosts.setdefault(b, 0)
            tag_boosts[a] += 0.05
            tag_boosts[b] += 0.05
    return tag_boosts


# === 🌐 RENDER + EXTRACT ===
async def fetch_rendered_html_playwright(url, timeout=30000):
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-web-security",
                ],
            )
            context = await browser.new_context(
                user_agent=get_user_agent(),
                viewport={"width": 1280, "height": 720},
                java_script_enabled=True,
                bypass_csp=True,
                locale="en-US",
            )

            page = await context.new_page()

            try:
                from playwright_stealth import stealth_async

                await stealth_async(page)
            except:
                pass

            await page.goto(url, timeout=timeout)
            await page.wait_for_load_state("networkidle")
            await page.mouse.wheel(0, 5000)
            await page.wait_for_timeout(2000)
            await page.mouse.click(640, 360)
            await page.wait_for_timeout(5000)
            html = await page.content()
            await browser.close()

            if any(
                s in html.lower()
                for s in [
                    "captcha",
                    "cf-chl-bypass",
                    "turnstile",
                    "checking your browser",
                ]
            ):
                return ""
            return html
    except Exception as e:
        print(f"[Playwright Error] {url}: {e}")
        return ""


def preprocess_html(html):
    try:
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(
            [
                "script",
                "style",
                "noscript",
                "header",
                "footer",
                "nav",
                "form",
                "input",
                "button",
                "aside",
                "meta",
                "link",
            ]
        ):
            tag.decompose()
        return str(soup)
    except:
        return html


def extract_content(html):
    try:
        return (
            trafilatura.extract(html, include_comments=False, include_tables=False)
            or ""
        )
    except:
        return ""


def safe_strip(v):
    return v.strip() if isinstance(v, str) else ""


async def extract_metadata(html):
    soup = BeautifulSoup(html, "lxml")

    title = (
        soup.title.string if soup.title and isinstance(soup.title.string, str) else ""
    )

    desc_tag = soup.find("meta", {"name": "description"}) or soup.find(
        "meta", {"property": "og:description"}
    )
    desc = (
        desc_tag.get("content")
        if isinstance(desc_tag, Tag) and desc_tag.has_attr("content")
        else ""
    )

    author_tag = soup.find("meta", {"name": "author"})
    author = (
        author_tag.get("content")
        if isinstance(author_tag, Tag) and author_tag.has_attr("content")
        else ""
    )

    return {
        "title": title.strip() if isinstance(title, str) else "",
        "description": desc.strip() if isinstance(desc, str) else "",
        "author": author.strip() if isinstance(author, str) else "",
    }


# === 🔍 SEARCH PIPELINE ===
def auto_generate_tags_from_text(text, top_k=5):
    return [
        kw
        for kw, _ in kw_model.extract_keywords(
            text, keyphrase_ngram_range=(1, 3), use_mmr=True, diversity=0.7, top_n=top_k
        )
    ]


def extract_deep_links(html: str, base_url: str) -> list[str]:
    soup = BeautifulSoup(html, "lxml")
    urls = set()

    for a in soup.find_all("a", href=True):
        if not isinstance(a, Tag):
            continue
        href = a["href"]
        full_url = urljoin(base_url, str(href))

        if not is_valid_link(full_url):
            continue

        if any(
            k in full_url.lower()
            for k in ["/watch", "/view", "/video", "viewkey=", ".mp4", ".m3u8"]
        ):
            urls.add(full_url)

    return list(urls)[:10]


def extract_video_sources(html, base_url):
    soup = BeautifulSoup(html, "lxml")
    sources = set()

    for tag in soup.find_all(["video", "source"]):
        if not isinstance(tag, Tag):
            continue
        src = tag.get("src") or tag.get("data-src")
        if src:
            full_url = urljoin(base_url, str(src))
            if re.search(r"\.(mp4|webm|m3u8|mov)", full_url):
                sources.add(full_url)

    for iframe in soup.find_all("iframe", src=True):
        if not isinstance(iframe, Tag):
            continue
        src = iframe["src"]
        if (
            "player" in src
            or "embed" in src
            or any(ext in src for ext in ["mp4", "m3u8"])
        ):
            sources.add(urljoin(base_url, str(src)))

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            if not isinstance(script, Tag):
                continue

            if not isinstance(script.string, str):
                continue

            import json

            data = json.loads(script.string)

            if isinstance(data, dict) and data.get("@type") == "VideoObject":
                url = data.get("contentUrl") or data.get("embedUrl")
                if url and re.search(r"\.(mp4|webm|m3u8|mov)", url):
                    sources.add(url)
        except:
            continue

    return list(sources)


async def process_url_async(url, query_embed):
    if not is_valid_link(url):
        return None
    html = await fetch_rendered_html_playwright(url)
    deep_links = rank_deep_links(extract_deep_links(html, url), query_embed)

    for deep_url in deep_links[:5]:
        deep_html = await fetch_rendered_html_playwright(deep_url)
        if not deep_html:
            continue

        if (
            re.search(r"\.(mp4|webm|m3u8|mov)", deep_html, re.IGNORECASE)
            or "<video" in deep_html
        ):
            html = deep_html
            url = deep_url
            break

    text = content_extractor.get_content(html)
    if len(text) < 200:
        try:
            text = content_extractor.get_content(
                BeautifulSoup(Document(html).summary(), "lxml").get_text("\n")
            )
        except:
            text = ""
    if len(text) < 200:
        try:
            text = content_extractor.get_content(
                BeautifulSoup(html, "lxml").get_text("\n")
            )
        except:
            return None

    lang = detect_language(text)
    if lang != "en":
        text = translate_text(text)

    text_hash = hashlib.md5(text.encode()).hexdigest()
    meta = await extract_metadata(html)
    tags = auto_generate_tags_from_text(f"{text.strip()} {meta['title']}", top_k=10)
    result = {
        "url": url,
        "summary": text.strip(),
        "video_links": extract_video_sources(html, url),
        "hash": text_hash,
        "title": meta["title"],
        "description": meta["description"],
        "author": meta["author"],
        "language": "en",
        "tags": tags,
    }

    return result


async def search_engine_async(query, link_count):
    payload = {"q": query, "format": "json", "language": "en"}
    async with ClientSession(timeout=timeout_obj) as session:
        async with session.post(
            SEARXNG_BASE_URL, data=payload, headers={"User-Agent": get_user_agent()}
        ) as resp:
            if resp.status != 200:
                return []
            data = await resp.json()
            return [
                r.get("url")
                for r in data.get("results", [])
                if is_valid_link(r.get("url"))
            ][:link_count]


async def advanced_search_async(query):
    seen_hashes = set()
    expanded_queries = expand_query_semantically(query)
    query_embed = model_embed.encode(query)
    all_links, results, processed = [], [], set()
    collected, sem = set(), asyncio.Semaphore(dynamic_parallel_task_limit())

    async def worker(url):
        domain = get_main_domain(url)
        async with sem, domain_counters[domain]:
            try:
                result = await asyncio.wait_for(
                    process_url_async(url, query_embed), timeout=12
                )
                if result and result.get("hash") not in seen_hashes:
                    seen_hashes.add(result["hash"])
                    return result
                return None
            except asyncio.TimeoutError:
                return None

    i, tasks = 0, []
    max_time = 25
    start_time = time.monotonic()
    while len(results) < SUMMARIES:
        if time.monotonic() - start_time > max_time:
            break

        if i >= len(all_links):
            for q in expanded_queries:
                links = await search_engine_async(
                    q, LINKS_TO_SCRAP // len(expanded_queries)
                )
                new_links = [u for u in links if u not in collected]
                all_links += new_links
                collected.update(new_links)

        while i < len(all_links) and len(tasks) < MAX_PARALLEL_TASKS:
            url = all_links[i]
            if url not in processed:
                processed.add(url)
                tasks.append(asyncio.create_task(worker(url)))
            i += 1

        if not tasks:
            await asyncio.sleep(0.1)
            continue

        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        for d in done:
            tasks.remove(d)
            if r := d.result():
                results.append(r)

    return results


# === 🌐 FASTAPI ROUTES ===
@asynccontextmanager
async def lifespan(_: FastAPI):
    async def auto_flush_loop():
        while True:
            await asyncio.sleep(10)

    flush_task = asyncio.create_task(auto_flush_loop())
    yield
    flush_task.cancel()
    try:
        await flush_task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index():
    return open("index.html", encoding="utf-8").read()


@app.get("/search")
async def search(query: str = "", power_scraping: bool = False):
    global LINKS_TO_SCRAP, SUMMARIES

    if power_scraping:
        cores = os.cpu_count() or 4
        LINKS_TO_SCRAP = min(cores * 30, 1000)
        SUMMARIES = min(cores * 10, 500)
    else:
        LINKS_TO_SCRAP = 10
        SUMMARIES = 5

    results = await advanced_search_async(query)

    if not any(r.get("video_links") for r in results):
        results = results[:5]
    else:
        results = [r for r in results if r.get("video_links")]

    results = rank_by_similarity(results, query)

    return JSONResponse(
        content=[
            {
                "title": r["title"] or r["url"],
                "summary": r["summary"],
                "video_links": r.get("video_links", []),
                "tags": r.get("tags", []),
            }
            for r in results
        ]
    )

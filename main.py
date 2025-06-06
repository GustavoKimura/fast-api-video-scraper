# === 📦 IMPORTS ===
import os, random, hashlib, asyncio, re, time, psutil
from urllib.parse import urlparse, urlunparse, urljoin
import torch, open_clip, aiohttp, tldextract, trafilatura
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
from bs4 import BeautifulSoup, Tag
from readability import Document
from langdetect import DetectorFactory, detect
from deep_translator import GoogleTranslator
from aiohttp import ClientTimeout
from keybert import KeyBERT
from playwright.async_api import async_playwright
from boilerpy3 import extractors
from numpy import dot
from numpy.linalg import norm
from collections import defaultdict
from typing import List, Tuple

# === 🔒 DOMAIN CONCURRENCY CONTROL ===
domain_counters = defaultdict(lambda: asyncio.Semaphore(4))


# === ⚙️ CONFIGURATION ===
def max_throughput_config():
    cores = os.cpu_count() or 4
    ram_gb = psutil.virtual_memory().total // 1_073_741_824
    task_multiplier = 6 if ram_gb > 16 else 3

    return min(cores * task_multiplier, 512)


MAX_PARALLEL_TASKS = max_throughput_config()
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

BLOCKED_KEYWORDS = [
    "quora",
    "board",
    "discussion",
    "signup",
    "login",
    "register",
    "comment",
    "thread",
    "showthread",
    "archive",
]

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


# === 🧩 SEMANTIC RANKING ===
def cosine_sim(a, b):
    return float(dot(a, b) / (norm(a) * norm(b) + 1e-8))


def rank_by_similarity(results, query_embed, min_duration=60, max_duration=900):
    query_tags = set(extract_tags(query_embed))
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
            overlap = len(query_tags.intersection(set(r["tags"])))
            score = sim + 0.05 * overlap
        r["score"] = score
        final.append(r)

    return sorted(final, key=lambda x: x["score"], reverse=True)


def rank_deep_links(links, query_embed):
    scored = []
    for link in links:
        title_snippet = re.sub(r"[-_/]", " ", link.split("/")[-1])
        embed = model_embed.encode(title_snippet)
        sim = cosine_sim(query_embed, embed)
        scored.append((sim, link))
    return [link for _, link in sorted(scored, reverse=True)[:5]]


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
    if any(kw in url.lower() for kw in BLOCKED_KEYWORDS):
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


# === 🌐 RENDER + EXTRACT ===
async def fetch_rendered_html_playwright(url, timeout=15000):
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
            )
            context = await browser.new_context(
                user_agent=get_user_agent(),
                viewport={"width": 1280, "height": 720},
                java_script_enabled=True,
                bypass_csp=True,
                locale="en-US",
            )
            domain = urlparse(url).netloc.replace("www.", "")
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
            if any(
                full_url.endswith(ext) for ext in [".mp4", ".webm", ".m3u8", ".mov"]
            ):
                sources.add(full_url)

    for iframe in soup.find_all("iframe", src=True):
        if not isinstance(iframe, Tag):
            continue
        src = iframe["src"]
        if "player" in src or any(ext in src for ext in ["mp4", "m3u8", "embed"]):
            sources.add(urljoin(base_url, str(src)))

    return list(sources)


async def process_url_async(url, query_embed):
    if not is_valid_link(url):
        return None
    html = await fetch_rendered_html_playwright(url)
    if (
        not html
        or "Just a moment..." in html
        or "checking your browser" in html.lower()
    ):
        return None

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


async def search_engine_async(query):
    payload = {"q": query, "format": "json", "language": "en"}
    async with aiohttp.ClientSession(timeout=timeout_obj) as session:
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
            ][:LINKS_TO_SCRAP]


async def advanced_search_async(query):
    query_embed = model_embed.encode(query)
    all_links, results, processed = [], [], set()
    collected, sem = set(), asyncio.Semaphore(MAX_PARALLEL_TASKS)

    async def worker(url):
        domain = get_main_domain(url)
        async with sem, domain_counters[domain]:
            try:
                return await asyncio.wait_for(
                    process_url_async(url, query_embed), timeout=12
                )
            except asyncio.TimeoutError:
                return None

    i, tasks = 0, []
    max_time = 25
    start_time = time.monotonic()
    while len(results) < LINKS_TO_SCRAP:
        if time.monotonic() - start_time > max_time:
            break

        if i >= len(all_links):
            links = await search_engine_async(query)
            new_links = [u for u in links if u not in collected]
            if not new_links:
                break
            all_links += new_links
            collected.update(new_links)

        while i < len(all_links) and len(tasks) < MAX_PARALLEL_TASKS:
            url = all_links[i]
            if url not in processed:
                processed.add(url)
                tasks.append(asyncio.create_task(worker(url)))
            i += 1

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


from fastapi.responses import StreamingResponse
import json


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

    query_embed = model_embed.encode(query)
    all_links, processed = [], set()
    collected = []
    sem = asyncio.Semaphore(MAX_PARALLEL_TASKS)

    async def worker(url):
        domain = get_main_domain(url)
        async with sem, domain_counters[domain]:
            try:
                return await asyncio.wait_for(
                    process_url_async(url, query_embed), timeout=12
                )
            except:
                return None

    async def streamer():
        nonlocal collected, all_links
        i = 0
        start_time = time.monotonic()

        while len(collected) < LINKS_TO_SCRAP and (time.monotonic() - start_time) < 25:
            if i >= len(all_links):
                links = await search_engine_async(query)
                all_links += [u for u in links if u not in processed]

            batch = []
            while i < len(all_links) and len(batch) < MAX_PARALLEL_TASKS:
                url = all_links[i]
                if url not in processed:
                    processed.add(url)
                    batch.append(asyncio.create_task(worker(url)))
                i += 1

            if batch:
                done, _ = await asyncio.wait(batch, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    try:
                        result = task.result()
                        if result and result.get("video_links"):
                            collected.append(result)
                            progress = int(100 * len(collected) / LINKS_TO_SCRAP)
                            yield f"data: {json.dumps({'progress': progress, 'result': result})}\n\n"
                    except:
                        continue

        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(streamer(), media_type="text/event-stream")

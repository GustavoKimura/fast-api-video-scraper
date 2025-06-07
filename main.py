# === 📦 IMPORTS ===
import os, random, asyncio, re, time, psutil, torch, open_clip, tldextract, trafilatura
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
from playwright_stealth import stealth_async


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
VIDEOS_TO_SEARCH = 10
VIDEO_RESULTS_LIMIT = 5
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
    "reddit.com",
    "google.com",
    "support.google.com",
    "zhihu.com",
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


def expand_query_semantically(query: str, top_n: int = 5):
    raw_keywords = extract_keywords(query, diversity=0.8, top_n=top_n * 2)
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
        if not r.get("duration"):
            continue
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
                re.sub(r"[^a-z0-9]", "", tag.lower()) for tag, _ in query_tags
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


def extract_tags(text: str, top_n: int = 10):
    clean = re.sub(r"[^a-zA-Z0-9\s]", "", text).lower()
    raw_results = extract_keywords(clean, diversity=0.7, top_n=top_n)

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
        tags = list(set(tag.lower() for tag in r.get("tags", [])))
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


def is_probable_video_url(url: str):
    video_exts = (".mp4", ".webm", ".m3u8", ".mov")
    parsed = urlparse(url)
    path = parsed.path.lower()
    return any(path.endswith(ext) for ext in video_exts)


def extract_keywords(text: str, top_n: int = 10, diversity=0.7):
    return kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        use_mmr=True,
        diversity=diversity,
        top_n=top_n,
    )


# === 🌐 RENDER + EXTRACT ===
async def fetch_rendered_html_playwright(url, timeout=90000):
    browser = None
    try:
        os.makedirs("screenshots", exist_ok=True)
        async with async_playwright() as p:
            try:
                browser = await p.chromium.launch(
                    headless=True,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--disable-web-security",
                    ],
                )
            except Exception as e:
                print(f"[PLAYWRIGHT ERROR] Launch failed: {e}")
                return ""

            context = await browser.new_context(
                user_agent=get_user_agent(),
                viewport={"width": 1280, "height": 720},
                java_script_enabled=True,
                bypass_csp=True,
                locale="en-US",
            )

            page = await context.new_page()

            try:
                await stealth_async(page)
            except Exception as e:
                print(f"[PLAYWRIGHT WARNING] Failed to apply stealth: {e}")

            try:
                await page.goto(url, timeout=timeout)
                await page.wait_for_load_state("networkidle")
                await auto_bypass_consent_dialog(page)
                html = await page.content()
                return html
            except Exception as e:
                print(f"[PLAYWRIGHT ERROR] Failed to load {url}: {e}")
                return ""
    except Exception as e:
        print(f"[PLAYWRIGHT FATAL] {url}: {e}")
        return ""
    finally:
        if browser:
            try:
                await browser.close()
            except:
                print(f"[PLAYWRIGHT WARNING] Failed to close browser for {url}")


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


async def auto_bypass_consent_dialog(page):
    try:
        await page.wait_for_timeout(1000)
        selectors = [
            "text=Enter",
            "text=I Agree",
            "text=Continue",
            "text=Yes",
            "text=Proceed",
            "button:has-text('Enter')",
            "button:has-text('Continue')",
            "button:has-text('Accept')",
            "button:has-text('OK')",
            "button:has-text('Got it')",
            "button:has-text('Agree')",
        ]
        for selector in selectors:
            element = await page.query_selector(selector)
            if element:
                try:
                    if await element.is_visible():
                        await element.click(force=True)
                        await page.wait_for_timeout(1000)
                        break
                except Exception as e:
                    if "not visible" not in str(e):
                        print(f"[CONSENT ERROR] {e}")
    except Exception as e:
        print(f"[CONSENT ERROR] {e}")


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


def auto_generate_tags_from_text(text, top_k=5):
    return [kw for kw, _ in extract_keywords(text, diversity=0.7, top_n=top_k)]


def extract_video_candidate_links(html: str, base_url: str):
    soup = BeautifulSoup(html, "lxml")
    urls = set()
    for a in soup.find_all("a", href=True):
        if not isinstance(a, Tag):
            continue
        href = str(a["href"])
        full_url = urljoin(base_url, href).split("?")[0].split("#")[0]
        if not is_valid_link(full_url):
            continue
        if is_probable_video_url(full_url) or href.lower().endswith(
            (".mp4", ".webm", ".m3u8")
        ):
            urls.add(full_url)
    print(f"[DEBUG] Deep candidate links from homepage: {len(urls)}")
    return list(urls)[:20]


def extract_video_sources(html, base_url):
    soup = BeautifulSoup(html, "lxml")
    sources = set()

    for tag in soup.find_all(["video", "source"]):
        if not isinstance(tag, Tag):
            continue
        for src_attr in [
            "src",
            "data-src",
            "data-hd-src",
            "data-video-url",
            "data-mp4",
            "data-webm",
        ]:
            src = tag.get(src_attr)
            if src:
                full_url = urljoin(base_url, str(src))
                if re.search(r"\.(mp4|webm|m3u8|mov)(\?.*)?$", full_url, re.IGNORECASE):
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

    for script in soup.find_all("script"):
        if not isinstance(script, Tag) or not script.string:
            continue
        matches = re.findall(
            r'(https?://[^\s\'"]+\.(?:mp4|webm|m3u8|mov))', script.string
        )
        sources.update(urljoin(base_url, m) for m in matches)

        jw_match = re.search(
            r'jwplayer\(["\'].*?["\']\)\.setup\(\{.*?file\s*:\s*["\'](https?[^"\']+\.(?:mp4|webm|m3u8|mov))["\']',
            script.string,
            re.DOTALL,
        )
        if jw_match:
            sources.add(jw_match.group(1))

        file_matches = re.findall(
            r'["\']file["\']\s*:\s*["\'](https?://[^\s\'"]+\.(?:mp4|webm|m3u8|mov))["\']',
            script.string,
        )
        sources.update(file_matches)

        list_matches = re.findall(
            r'["\']sources["\']\s*:\s*\[(.*?)\]', script.string, re.DOTALL
        )
        for group in list_matches:
            nested_files = re.findall(
                r'["\']file["\']\s*:\s*["\'](https?://[^\s\'"]+\.(?:mp4|webm|m3u8|mov))["\']',
                group,
            )
            sources.update(nested_files)

    if sources:
        print(f"[EXTRACTED] Found {len(sources)} video URLs")

    return list(dict.fromkeys(sorted(sources)))


async def extract_video_metadata(url, query_embed):
    if not is_valid_link(url):
        return None

    html = await fetch_rendered_html_playwright(url)
    if not html.strip():
        print(f"[ERROR] No HTML content fetched from {url}")
        return None

    video_links = extract_video_sources(html, url)

    if not video_links:
        candidate_links = extract_video_candidate_links(html, url)
        ranked_links = rank_deep_links(candidate_links, query_embed)
        for deep_url in ranked_links[:5]:
            deep_html = await fetch_rendered_html_playwright(deep_url)
            if (
                "404" in deep_html
                or "Page not found" in deep_html
                or re.search(r"\b(error|fail|unavailable)\b", deep_html, re.I)
            ):
                continue
            deep_sources = extract_video_sources(deep_html, deep_url)
            if deep_sources:
                html = deep_html
                url = deep_url
                video_links = deep_sources
                break

    soup = BeautifulSoup(html, "lxml")
    deep_links = rank_deep_links(extract_video_candidate_links(html, url), query_embed)

    for deep_url in deep_links[:5]:
        deep_html = await fetch_rendered_html_playwright(deep_url)
        if (
            not deep_html
            or "<html" in deep_html
            and "</html>" in deep_html
            and len(deep_html) < 1000
        ):
            continue
        if (
            re.search(r"\.(mp4|webm|m3u8|mov)", deep_html, re.IGNORECASE)
            or "<video" in deep_html
        ):
            html = deep_html
            url = deep_url
            break

    try:
        text = content_extractor.get_content(html)
    except Exception as e:
        print(f"[ERROR] boilerpy3 failed: {e}")
        text = ""

    if len(text) < 200:
        try:
            doc = Document(html)
            text = doc.summary()
            text = BeautifulSoup(text, "lxml").get_text("\n")
        except:
            text = ""

    if len(text) < 200:
        text = extract_content(html)

    if len(text) < 100:
        print(f"[ERROR] Unable to extract usable text from {url}")
        return None

    lang = detect_language(text)
    if lang != "en":
        text = translate_text(text)

    meta = await extract_metadata(html)
    tags = auto_generate_tags_from_text(f"{text.strip()} {meta['title']}", top_k=10)
    return {
        "url": url,
        "video_links": video_links,
        "title": meta["title"] or urlparse(url).netloc,
        "description": meta["description"],
        "tags": tags,
        "score": 0.0,
    }


# === 🔍 SEARCH PIPELINE ===
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


async def search_videos_async(query):
    expanded_queries = list(dict.fromkeys(expand_query_semantically(query)))
    print(f"[DEBUG] Expanded queries: {expanded_queries}")

    query_embed = model_embed.encode(query)
    all_links, results, processed = [], [], set()
    collected = set()
    sem = asyncio.Semaphore(dynamic_parallel_task_limit())

    async def worker(url):
        domain = get_main_domain(url)
        print(f"[WORKER] Starting worker for: {url} (domain: {domain})")
        async with sem, domain_counters[domain]:
            try:
                result = await asyncio.wait_for(
                    extract_video_metadata(url, query_embed), timeout=60
                )
                if not result or not result.get("video_links"):
                    print(f"[WORKER] No usable result for: {url}")
                    return None
                processed.add(url)
                print(f"[WORKER] Completed: {url}")
                return result
            except asyncio.TimeoutError:
                print(f"[WORKER TIMEOUT] {url}")
                return None
            except Exception as e:
                print(f"[WORKER ERROR] {url}: {e}")
                return None

    i, tasks = 0, []
    max_time = 90
    start_time = time.monotonic()

    while len(results) < VIDEO_RESULTS_LIMIT:
        elapsed = time.monotonic() - start_time
        if elapsed > max_time:
            print("[LOOP] Max time reached, exiting loop.")
            break

        if i >= len(all_links):
            print("[LOOP] Fetching new links for expanded queries...")
            for q in expanded_queries:
                try:
                    links = await search_engine_async(
                        q, VIDEOS_TO_SEARCH // len(expanded_queries)
                    )
                    new_links = [u for u in links if u not in collected]
                    all_links += new_links
                    collected.update(new_links)
                except Exception as e:
                    print(f"[LOOP ERROR] Failed to fetch links for query '{q}': {e}")

        while i < len(all_links) and len(tasks) < MAX_PARALLEL_TASKS:
            url = all_links[i]
            if url not in processed:
                tasks.append(asyncio.create_task(worker(url)))
            i += 1

        if not tasks:
            await asyncio.sleep(0.1)
            continue

        done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        for d in done:
            tasks.remove(d)
            try:
                result = d.result()
                if result:
                    print(f"[RESULT] Received result from task: {result.get('url')}")
                    results.append(result)
            except Exception as e:
                print(f"[RESULT ERROR] Failed task: {e}")

    if tasks:
        done, _ = await asyncio.wait(tasks, timeout=15)
        for d in done:
            try:
                results.append(d.result())
            except Exception as e:
                print(f"[FINAL RESULT ERROR] {e}")

    return results


# === 🚀 FASTAPI ROUTES ===
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
    print(
        "----------------------------------------------------------------------------------------------------"
    )
    global VIDEOS_TO_SEARCH, VIDEO_RESULTS_LIMIT

    if power_scraping:
        cores = os.cpu_count() or 4
        VIDEOS_TO_SEARCH = min(cores * 30, 1000)
        VIDEO_RESULTS_LIMIT = min(cores * 10, 500)
    else:
        VIDEOS_TO_SEARCH = 5
        VIDEO_RESULTS_LIMIT = 1

    results = await search_videos_async(query)
    print(f"[RESULTS] Total results fetched: {len(results)}")

    video_results = [r for r in results if r.get("video_links")]
    if video_results:
        print(f"[RESULTS] {len(video_results)} result(s) contain video_links")
        results = video_results
    else:
        print(f"[RESULTS] No video_links found, falling back to top 5 videos by score")
        results = sorted(
            results,
            key=lambda x: (
                x.get("score", 0),
                -len(x["description"]),
                get_main_domain(x["url"]),
            ),
        )

    results = rank_by_similarity(results, query)
    print(f"[RANKING] Results ranked by semantic similarity to query")

    return JSONResponse(
        content=[
            {
                "url": r["url"],
                "title": r["title"],
                "description": r["description"],
                "video_links": r["video_links"],
                "tags": r["tags"],
                "video_score": r["score"],
            }
            for r in results
        ]
    )

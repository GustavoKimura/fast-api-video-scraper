# === 📦 IMPORTS ===
import os, random, asyncio, re, time, psutil, torch, open_clip, tldextract, trafilatura, json, base64, concurrent.futures, subprocess, logging
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
from playwright.async_api import async_playwright, Browser, BrowserContext
from boilerpy3 import extractors
from numpy import dot, ndarray, zeros
from numpy.linalg import norm
from collections import defaultdict, Counter, OrderedDict
from typing import List, Tuple, Callable
from itertools import combinations
from difflib import SequenceMatcher
from playwright_stealth import stealth_async
from threading import Lock

# === ℹ️ LOGGING ===
start_time = time.monotonic()


class ElapsedFormatter(logging.Formatter):
    def format(self, record):
        elapsed = time.monotonic() - start_time
        record.elapsed_time = f"{elapsed:.2f}s"
        return super().format(record)


formatter_str = "%(elapsed_time)s [%(levelname)s] %(message)s"

logging.basicConfig(level=logging.DEBUG, format=formatter_str)

for handler in logging.getLogger().handlers:
    handler.setFormatter(ElapsedFormatter(formatter_str))

for lib in ["trafilatura", "boilerpy3", "readability", "urllib3"]:
    logging.getLogger(lib).setLevel(logging.WARNING)

# === 🧾 EXECUTOR ===
executor = concurrent.futures.ThreadPoolExecutor(max_workers=(os.cpu_count() or 4) * 2)


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


def get_ffprobe_concurrency():
    cores = os.cpu_count() or 4
    ram_gb = psutil.virtual_memory().total // 1_073_741_824

    base = min(cores // 2, 8)
    if ram_gb >= 16:
        base += 2
    elif ram_gb <= 4:
        base = max(1, base - 1)
    return max(2, base)


async def run_ffprobe(cmd):
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        executor, lambda: subprocess.check_output(cmd, timeout=10).decode().strip()
    )


async def run_ffprobe_json(cmd):
    loop = asyncio.get_running_loop()

    def run():
        result = subprocess.check_output(cmd, timeout=10).decode().strip()
        return json.loads(result)

    return await loop.run_in_executor(executor, run)


domain_counters = defaultdict(lambda: asyncio.Semaphore(get_domain_concurrency()))
ffprobe_sem = asyncio.Semaphore(get_ffprobe_concurrency())


# === ⚙️ CONFIGURATION ===
def dynamic_parallel_task_limit():
    cores = os.cpu_count() or 4
    ram_gb = psutil.virtual_memory().total // 1_073_741_824
    multiplier = 16 if ram_gb >= 16 else 12
    return min(cores * multiplier, 512)


MAX_PARALLEL_TASKS = dynamic_parallel_task_limit()
DetectorFactory.seed = 0
timeout_obj = ClientTimeout(total=5)
content_extractor = extractors.LargestContentExtractor()
SEARXNG_BASE_URL = "http://searxng:8080/search"
client_session: ClientSession | None = None
_session_lock = asyncio.Lock()
_playwright_obj = None
_playwright_browser: Browser | None = None
_playwright_context: BrowserContext | None = None
_browser_lock = asyncio.Lock()


async def init_browser():
    global _playwright_browser, _playwright_context, _playwright_obj

    if _playwright_obj is None:
        for attempt in range(2):
            try:
                _playwright_obj = await async_playwright().start()
                break
            except Exception as e:
                logging.error(f"PLAYWRIGHT INIT ERROR (attempt {attempt + 1}): {e}")
                await asyncio.sleep(1)

    if _playwright_browser and not _playwright_browser.is_connected():
        try:
            await _playwright_browser.close()
        except:
            pass
        _playwright_browser = None
        _playwright_context = None

    async with _browser_lock:
        if _playwright_browser is None or _playwright_context is None:
            if _playwright_obj is None:
                raise RuntimeError("Playwright failed to start after retries.")

            _playwright_browser = await _playwright_obj.chromium.launch(
                headless=True,
                args=[
                    "--disable-blink-features=AutomationControlled",
                    "--no-sandbox",
                    "--disable-dev-shm-usage",
                    "--disable-web-security",
                ],
            )
            _playwright_context = await _playwright_browser.new_context(
                user_agent=get_user_agent(),
                viewport={"width": 1280, "height": 720},
                java_script_enabled=True,
                bypass_csp=True,
                locale="en-US",
            )

    return _playwright_context


async def get_client_session():
    global client_session
    async with _session_lock:
        if client_session is None or client_session.closed:
            client_session = ClientSession(timeout=timeout_obj)
    return client_session


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

BAD_TAGS = [
    "cookie",
    "policy",
    "access",
    "block",
    "label",
    "html",
    "optimize",
    "minor",
    "child",
    "feedback",
    "error",
    "null",
    "undefined",
    "function",
]

NOT_VIDEO_EXTENSIONS = [
    ".js",
    ".css",
    ".jpg",
    ".png",
    ".woff",
    ".ico",
    ".svg",
    ".gif",
    ".webp",
    ".pdf",
    ".doc",
    ".xls",
    ".zip",
    ".rar",
    ".ppt",
    ".txt",
    ".json",
    ".xml",
    ".csv",
    ".mp3",
    ".wav",
    ".ogg",
]

LANGUAGES_BLACKLIST = {"da", "so", "tl", "nl", "sv", "af", "el"}


# === 🧠 MODELS ===
class OpenCLIPEmbedder:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "laion2b_s34b_b79k",
        device: str | None = None,
        max_cache_size: int = 5000,
    ):
        self.device: str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, _, _ = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(self.device)
        self.tokenizer: Callable[[str], torch.Tensor] = open_clip.get_tokenizer(
            model_name
        )

        self._cache: OrderedDict[str, ndarray] = OrderedDict()
        self._cache_lock = Lock()
        self._max_cache_size = max_cache_size

    def encode_cached(self, text: str) -> ndarray:
        with self._cache_lock:
            if text in self._cache:
                self._cache.move_to_end(text)
                return self._cache[text]

        try:
            tokens = self.tokenizer(text).to(self.device)
            with torch.no_grad():
                features = self.model.encode_text(tokens).float()
            result = features.cpu().numpy()[0]
        except Exception as e:
            logging.error(f"EMBED ERROR - Failed to encode '{text}': {e}")
            result = zeros((512,))

        with self._cache_lock:
            self._cache[text] = result
            self._cache.move_to_end(text)
            if len(self._cache) > self._max_cache_size:
                self._cache.popitem(last=False)

        return result


model_embed = OpenCLIPEmbedder(model_name="ViT-B-32", pretrained="laion2b_s34b_b79k")
kw_model = KeyBERT("sentence-transformers/all-MiniLM-L6-v2")


def expand_query_semantically(query: str, top_n: int = 5):
    raw_keywords = extract_keywords(query, diversity=0.8)
    query_vec = model_embed.encode_cached(query)
    filtered = []

    for kw, _ in raw_keywords:
        kw = str(kw).strip()
        kw_vec = model_embed.encode_cached(kw)
        sim = cosine_sim(query_vec, kw_vec)

        if not any(
            SequenceMatcher(None, kw.lower(), existing_kw.lower()).ratio() >= 0.85
            for existing_kw, _ in filtered
        ):
            filtered.append((kw, sim))

    top_keywords = [
        kw
        for kw, _ in sorted(filtered, key=lambda x: x[1], reverse=True)
        if len(kw.split()) >= 2 and kw.lower() not in query.lower()
    ][:top_n]
    return [query] + top_keywords


# === 🧩 SEMANTIC RANKING ===
def cosine_sim(a, b):
    norm_a = norm(a)
    norm_b = norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot(a, b) / (norm_a * norm_b))


def rank_by_similarity(results, query, min_duration=30, max_duration=3600):
    query_embed = model_embed.encode_cached(query)

    raw_tags = extract_tags(query)
    query_tags = {kw for kw, _ in raw_tags if isinstance(kw, str)}

    tag_boosts = boost_by_tag_cooccurrence(results)
    seen_domains = set()
    final = []

    for r in results:
        try:
            dur_secs = float(r.get("duration", 0))
        except (ValueError, TypeError):
            continue

        if dur_secs < 0:
            continue
        else:
            if not (min_duration <= dur_secs <= max_duration):
                continue
            score_penalty = 0.0

        score = 0.0

        if r.get("is_stream", False):
            score_penalty += 0.4

        if r.get("tags"):
            tag_text = " ".join(r["tags"])
            tag_embed = model_embed.encode_cached(tag_text)
            sim = cosine_sim(query_embed, tag_embed)
            result_tags = normalize_tags(r.get("tags", []))
            if not result_tags:
                result_tags = normalize_tags(r.get("title", "").split())
            query_tags_norm = normalize_tags(list(query_tags))
            overlap = len(query_tags_norm & result_tags)
            boost = sum(tag_boosts.get(normalize_tag(tag), 0) for tag in r["tags"])
            score = sim + 0.05 * overlap + boost

        domain = get_main_domain(r["url"])
        if domain in seen_domains:
            score *= 0.85
        else:
            seen_domains.add(domain)

        if score == 0.0 and r.get("title"):
            title_embed = model_embed.encode_cached(r["title"])
            score = cosine_sim(query_embed, title_embed) * 0.6

        r["score"] = score - score_penalty

        final.append(r)

    if final:
        scores = [r["score"] for r in final]
        min_score = min(scores)
        max_score = max(scores)
        if max_score > min_score:
            for r in final:
                r["score"] = (r["score"] - min_score) / (max_score - min_score)

    for r in final:
        logging.debug(
            f"SCORE DEBUG - {r['score']:.4f} | {r.get('url', '')[:50]} | Tags: {r.get('tags', [])}"
        )

    if not final or all(r.get("score", 0.0) == 0.0 for r in final):
        logging.warning(
            "RANKING - No videos scored above 0. Using fallback sort by duration."
        )
        final.sort(key=lambda x: float(x.get("duration", 0)), reverse=True)

    return sorted(final, key=lambda x: x["score"], reverse=True)


def rank_deep_links(links, query_embed):
    scored = []
    for link in links:
        snippet = re.sub(r"[-_/]", " ", urlparse(link).path.split("/")[-1])
        embed = model_embed.encode_cached(snippet)
        sim = cosine_sim(query_embed, embed)
        scored.append((sim, link))
    scored.sort(reverse=True)
    return [link for _, link in scored[:5]]


# === 🔧 UTILITIES ===
def get_user_agent():
    return random.choice(USER_AGENTS)


def normalize_url(url):
    return urlunparse(urlparse(url)._replace(fragment=""))


def get_main_domain(url):
    ext = tldextract.extract(url)
    if ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    elif ext.domain:
        return ext.domain
    else:
        return urlparse(url).netloc


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
        logging.warning(f"Invalid duration format: {duration_str}")
        return 0


async def get_video_duration(url: str, html: str = ""):
    if any(ext in url.lower() for ext in NOT_VIDEO_EXTENSIONS):
        return 0.0

    async with ffprobe_sem:
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                url,
            ]
            duration_str = await run_ffprobe(cmd)
            if duration_str:
                duration = float(duration_str)
                if duration > 0:
                    return duration
        except Exception as e:
            logging.error(f"FFPROBE ERROR - {url}: {e}")

    try:
        if html:
            soup = BeautifulSoup(html, "lxml")
            meta = soup.find("meta", attrs={"property": "og:video:duration"})
            if isinstance(meta, Tag) and meta and meta.get("content"):
                return float(str(meta["content"]))

        match = re.search(r'"duration"\s*:\s*"PT(\d+)M(\d+)S"', html)
        if match:
            minutes = int(match.group(1))
            seconds = int(match.group(2))
            return minutes * 60 + seconds

        match = re.search(r"(\d+)[-_]?min", url.lower())
        if match:
            return int(match.group(1)) * 60
    except Exception as e:
        logging.error(f"FALLBACK DURATION ERROR - {url}: {e}")

    return 0.0


async def get_video_resolution_score(url: str):
    if any(ext in url.lower() for ext in NOT_VIDEO_EXTENSIONS):
        return 0

    async with ffprobe_sem:
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,bit_rate",
                "-of",
                "json",
                url,
            ]
            data = await run_ffprobe_json(cmd)
            if "streams" in data and data["streams"]:
                stream = data["streams"][0]
                width = stream.get("width", 0)
                height = stream.get("height", 0)
                bitrate = stream.get("bit_rate", 0)

                resolution_score = width * height
                bitrate_score = (
                    int(bitrate) if isinstance(bitrate, str) else bitrate or 0
                )
                return resolution_score + (bitrate_score // 1000)
        except Exception as e:
            logging.error(f"FFPROBE-RES ERROR - {url}: {e}")
    return 0


def extract_tags(text: str):
    clean = re.sub(r"[^a-zA-Z0-9\s]", "", text).lower()
    raw_results = extract_keywords(clean, diversity=0.7)

    if (
        raw_results
        and isinstance(raw_results, list)
        and all(isinstance(sub, list) for sub in raw_results)
    ):
        flat: List[Tuple[str, float]] = []
        for sublist in raw_results:
            flat.extend(
                item for item in sublist if isinstance(item, tuple) and len(item) == 2
            )
        return flat

    flat: List[Tuple[str, float]] = []
    for item in raw_results:
        if isinstance(item, tuple) and len(item) == 2:
            flat.append(item)
        elif isinstance(item, list):
            flat.extend(i for i in item if isinstance(i, tuple) and len(i) == 2)
    return flat


def boost_by_tag_cooccurrence(results):
    co_pairs = Counter()
    tag_boosts = {}
    seen_tag_sets = {}

    def canonical_tag(tag):
        words = re.findall(r"\w+", tag.lower())
        return tuple(sorted(set(words)))

    for r in results:
        tags_raw = r.get("tags", [])
        if not tags_raw:
            fallback_tags = auto_generate_tags_from_text(
                r.get("title", "") + " " + r.get("url", "")
            )
            if fallback_tags:
                tags_raw = fallback_tags

        canonical_tags = []
        for tag in tags_raw:
            if not isinstance(tag, str) or not tag.strip():
                continue
            norm = normalize_tag(str(tag))
            can = canonical_tag(norm)
            seen_tag_sets[norm] = can
            canonical_tags.append(can)

        unique_canonicals = list(set(canonical_tags))
        for a, b in combinations(sorted(unique_canonicals), 2):
            co_pairs[(a, b)] += 1

    for (a, b), count in co_pairs.items():
        if count >= 2:
            for tag in [a, b]:
                name = " ".join(tag)
                tag_boosts.setdefault(name, 0)
                tag_boosts[name] += 0.05

    return tag_boosts


def is_probable_video_url(url: str):
    video_exts = (".mp4", ".webm", ".m3u8", ".mov")
    non_video_exts = (".js", ".css", ".json", ".txt")

    path = urlparse(url).path.lower()
    if any(path.endswith(ext) for ext in non_video_exts):
        return False
    return (
        any(path.endswith(ext) for ext in video_exts)
        and not get_main_domain(url).lower() in BLOCKED_DOMAINS
    )


def extract_keywords(text: str, diversity=0.7):
    return kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        use_mmr=True,
        diversity=diversity,
    )


def is_sensible_keyword(kw):
    if not kw or len(kw) > 40:
        return False
    if len(kw.split()) == 1 and len(kw) < 5:
        return False
    if re.search(r"\d{6,}", kw):
        return False
    if re.fullmatch(r"[a-zA-Z0-9]{10,}", kw):
        return False
    if re.fullmatch(r"[a-z0-9]{8,}", kw):
        return False
    if sum(c.isdigit() for c in kw) > len(kw) * 0.4:
        return False
    if re.search(r"[{}[\];<>$]", kw):
        return False
    if re.search(r"\b(ns|prod|widget|meta|error|stack)\b", kw.lower()):
        return False
    if len(re.findall(r"[a-zA-Z]", kw)) < len(kw) * 0.5:
        return False
    if len(re.findall(r"[aeiouy]+", kw.lower())) < 2:
        return False

    return normalize_tag(kw) not in BAD_TAGS


def clean_tag_text(tag):
    tag = re.sub(r"(.)\1{2,}", r"\1", tag)
    tag = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", tag)
    tag = re.sub(r"([a-z])(\d+)", r"\1 \2", tag)
    return tag.lower().strip()


def normalize_tag(tag: str):
    tag = clean_tag_text(re.sub(r"[^a-z0-9]", "", tag.lower()))
    tag = re.sub(r"\b(consent|cookie|agree|age|18(?:\+)?(?:only)?)\b", "", tag)
    return tag.strip()


def normalize_tags(tags: list[str]):
    return {normalize_tag(tag) for tag in tags}


async def deduplicate_videos(videos: list[dict]) -> list[dict]:
    seen = {}
    scores = {}

    async def get_score(video):
        return await get_video_resolution_score(video["url"])

    for video in videos:
        base = re.sub(r"\.(mp4|webm|m3u8|mov)$", "", video["url"].lower())
        key = (
            normalize_tag(video["title"]),
            frozenset(normalize_tags(video.get("tags", []))),
            round(float(video.get("duration", 0)), 1),
            base,
        )

        if key not in seen:
            seen[key] = video
            scores[key] = await get_score(video)
        else:
            new_score = await get_score(video)
            if new_score > scores[key]:
                seen[key] = video
                scores[key] = new_score

    return list(seen.values())


# === 🌐 RENDER + EXTRACT ===
async def fetch_rendered_html_playwright(
    url, timeout=150000, retries=2, power_scraping=False
):
    if power_scraping:
        timeout = 900000

    context = await init_browser()
    video_requests = []

    async def intercept_video_requests(route):
        try:
            req_url = route.request.url
            if any(
                ext in req_url.lower()
                for ext in [
                    ".mp4",
                    ".webm",
                    ".m3u8",
                    ".ts",
                    ".mov",
                    ".mpd",
                    "/get_file/",
                    "/download/",
                    "/hls/",
                    "/flv/",
                ]
            ):
                if req_url not in video_requests:
                    logging.debug(f"INTERCEPT - Video URL: {req_url}")
                    video_requests.append(req_url)
            await route.continue_()
        except Exception as e:
            if "closed" not in str(e):
                logging.warning(f"INTERCEPT ERROR - {e}")

    await context.route("**/*", intercept_video_requests)

    async def _internal_fetch_with_playwright(url, timeout):
        html = ""
        page = None

        async def safe_evaluate(page, script, arg=None):
            try:
                if page.is_closed():
                    return None
                return (
                    await page.evaluate(script, arg)
                    if arg
                    else await page.evaluate(script)
                )
            except Exception as e:
                logging.error(f"SAFE EVAL ERROR - {e}")
                return None

        try:
            try:
                page = await context.new_page()
            except Exception as e:
                logging.error(f"Failed to create new page: {e}")
                return "", []

            async def handle_stream_response(response):
                try:
                    url = response.url.lower()
                    if any(
                        ext in url
                        for ext in [
                            ".m3u8",
                            ".ts",
                            ".mpd",
                            ".mp4",
                            ".webm",
                            "stream",
                            "cdn",
                        ]
                    ):
                        if url not in video_requests:
                            video_requests.append(url)
                except Exception as e:
                    logging.error(f"STREAM RESPONSE ERROR - {e}")

            page.on(
                "response",
                lambda response: asyncio.create_task(handle_stream_response(response)),
            )
            page.set_default_navigation_timeout(timeout)

            try:
                await stealth_async(page)
            except Exception as e:
                logging.info(f"STEALTH ERROR - {e}")

            try:
                try:
                    await page.goto(url, timeout=timeout)
                except Exception as e:
                    logging.warning(f"PAGE GOTO ERROR - {url}: {e}")
                    return "", []

                await safe_evaluate(
                    page,
                    """window.alert = () => {}; window.confirm = () => true;""",
                )

                try:
                    await auto_bypass_consent_dialog(page)
                except Exception as e:
                    logging.warning(f"Bypass failed but continuing: {e}")

                await page.wait_for_load_state("networkidle", timeout=timeout)
                await page.wait_for_timeout(2000)

                try:
                    for selector in [
                        "video",
                        ".thumb",
                        ".player",
                        ".video-thumb",
                        ".video-thumb--type-video",
                    ]:
                        try:
                            await page.click(selector, timeout=1500)
                            await page.wait_for_timeout(1000)
                            logging.debug(f"Clicked element: {selector}")
                        except Exception as e:
                            logging.debug(f"Failed to click {selector}: {e}")

                    await page.wait_for_timeout(2000)
                    logging.debug("Clicked video/player element to trigger JS playback")
                except Exception as e:
                    logging.debug(
                        f"No clickable player element found or failed to click: {e}"
                    )

                try:
                    await page.mouse.wheel(0, 1000)
                    await page.wait_for_timeout(1000)
                    logging.debug("Scrolled page to trigger lazy-load elements")
                except Exception as e:
                    logging.debug(f"Scroll failed: {e}")

                try:
                    await page.wait_for_selector(
                        "video, source, iframe[src*='cdn'], script", timeout=5000
                    )
                except Exception as e:
                    logging.debug(f"No video/player selectors found during wait: {e}")

                if not page or page.is_closed():
                    logging.warning("Page is not available or already closed.")
                    return "", []

                html = await page.content()

                video_urls_dom = await safe_evaluate(
                    page,
                    """() => {
                        const urls = new Set();
                        document.querySelectorAll('video, source').forEach(el => {
                            const src = el.src || el.getAttribute('data-src') || el.getAttribute('data-hd-src');
                            if (src && (src.includes('.mp4') || src.includes('.webm') || src.includes('.m3u8'))) {
                                urls.add(src);
                            }
                        });
                        return Array.from(urls);
                    }""",
                )
                if isinstance(video_urls_dom, list):
                    video_requests.extend(video_urls_dom)
            except Exception as e:
                logging.info(f"NAVIGATION ERROR - {url}: {e}")
                return "", []

            try:
                await page.locator("video source").first.wait_for(timeout=8000)
            except Exception:
                logging.info("No <video><source> found, proceeding anyway.")

            await safe_evaluate(
                page,
                """() => {
                    const fallbackClick = document.querySelector('video, .video-player, [data-video], .play-btn, .player');
                    if (fallbackClick) {
                        try { fallbackClick.click(); } catch (e) {}
                    }
                }""",
            )

            await safe_evaluate(
                page,
                """() => {
                    document.querySelectorAll('button, .play, .btn-play, .vjs-big-play-button')
                    .forEach(el => { try { el.click(); } catch (e) {} });
                }""",
            )
            await page.wait_for_timeout(1500)
            await safe_evaluate(
                page,
                """() => {
                    const links = document.querySelectorAll('a[href*="/video"], a.thumbnail, a.video-thumb');
                    for (const link of links) {
                        try { link.click(); } catch (_) {}
                    }
                }""",
            )
            await page.wait_for_timeout(1200)

            last_height = await safe_evaluate(page, "document.body.scrollHeight")
            for _ in range(5):
                await safe_evaluate(
                    page, "window.scrollBy(0, document.body.scrollHeight)"
                )
                await page.wait_for_timeout(800)
                new_height = await safe_evaluate(page, "document.body.scrollHeight")
                if new_height == last_height:
                    break
                last_height = new_height

            await safe_evaluate(
                page,
                """() => {
                    const selectors = [
                        '[class*="overlay"]',
                        '[class*="modal"]',
                        '[id*="age"]',
                        '[id*="consent"]',
                        '[class*="consent"]',
                        '.dialog-desktop-container',
                        'div[role="dialog"]',
                        '.vjs-modal-dialog'
                    ];
                    const blockers = document.querySelectorAll(selectors.join(','));
                    blockers.forEach(el => {
                        el.style.display = 'none';
                        el.style.pointerEvents = 'none';
                        el.style.visibility = 'hidden';
                        try { el.remove(); } catch (_) {}
                    });
                }""",
            )

            try:
                content = extract_content(html)
            except Exception as e:
                logging.warning(f"CONTENT EXTRACT FAIL - {e}")
                content = ""
            del html

        finally:
            try:
                if page:
                    try:
                        await page.close()
                    except Exception as e:
                        logging.warning(f"Failed to close page: {e}")
                if context:
                    try:
                        await context.clear_cookies()
                    except Exception as e:
                        logging.warning(f"Failed to clear cookies: {e}")
            except Exception as e:
                logging.warning(f"General cleanup error: {e}")

        if video_requests:
            logging.debug(
                f"STREAM RESPONSE - {len(set(video_requests))} stream(s) from {url}"
            )

        if (
            page is not None
            and not page.is_closed()
            and (not content or not content.strip())
        ):
            try:
                os.makedirs("screenshots", exist_ok=True)
                await page.screenshot(
                    path=f"screenshots/timeout_debug_{int(time.time())}.png",
                    full_page=True,
                )
                logging.warning(f"Saved debug screenshot for {url}")
            except Exception as e:
                logging.warning(f"Screenshot capture failed: {e}")

        return content, [v for v in set(video_requests) if not v.startswith("blob:")]

    try:
        for attempt in range(retries + 1):
            try:
                html, video_urls = await _internal_fetch_with_playwright(
                    url, timeout + attempt * 15000
                )
                if html and html.strip():
                    return html, video_urls
            except Exception as e:
                logging.warning(f"RETRY {attempt} - Error fetching {url}: {e}")
            finally:
                await asyncio.sleep(2 * (attempt + 1))
    finally:
        await context.unroute("**/*")

    return "", []


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
        await page.wait_for_timeout(150)

        selectors = [
            '[id*="consent"]',
            '[class*="consent"]',
            '[class*="dialog"]',
            '[class*="popup"]',
            '[id*="age"]',
            '[class*="overlay"]',
            '[role="dialog"]',
            '[class*="parental"]',
            '[class*="gate"]',
            '[class*="block"]',
            '[class*="modal"]',
        ]
        for sel in selectors:
            try:
                await page.evaluate(
                    f"""
                    () => {{
                        document.querySelectorAll('{sel}').forEach(el => {{
                            el.style.setProperty('display', 'none', 'important');
                            el.style.setProperty('visibility', 'hidden', 'important');
                            el.style.setProperty('pointer-events', 'none', 'important');
                            el.style.setProperty('z-index', '-9999', 'important');
                            try {{ el.remove(); }} catch (_) {{}}
                        }});
                    }}
                """
                )
            except Exception as e:
                logging.debug(f"HIDE FAIL: {sel} -> {e}")

        click_texts = [
            "Enter",
            "Continue",
            "OK",
            "Accept",
            "I Agree",
            "Yes",
            "Sim",
            "Entrar",
            "I am 18",
            "I am 18+",
            "I'm 18",
            "Tenho 18 anos",
            "Proceed",
            "Start",
        ]

        for text in click_texts:
            try:
                element = await page.query_selector(f"text={text}")
                if element and await element.is_visible():
                    await element.click(force=True)
                    await page.wait_for_timeout(200)
            except Exception as e:
                logging.debug(f"CLICK FAIL: {text} -> {e}")

        await page.evaluate(
            """
            () => {
                const btns = Array.from(document.querySelectorAll('button, a')).filter(el => {
                    const text = el.innerText?.toLowerCase?.() || "";
                    return ['enter', 'accept', 'yes', 'i agree', 'continue'].some(t => text.includes(t));
                });
                for (const btn of btns) {
                    try { btn.click(); } catch (_) {}
                }
            }
        """
        )

        await page.keyboard.press("Enter")
        await page.wait_for_timeout(300)

        frames = page.frames
        for frame in frames:
            try:
                await frame.evaluate(
                    """
                    () => {
                        document.querySelectorAll('button, .btn, .accept, .agree').forEach(el => {
                            if (el.innerText?.toLowerCase().includes('agree')) {
                                try { el.click(); } catch (_) {}
                            }
                        });
                    }
                """
                )
            except Exception:
                continue

    except Exception as e:
        logging.error(f"[CONSENT ERROR] {type(e).__name__}: {e}")


async def extract_metadata(html, url):
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

    if not title:
        try:
            doc = Document(html)
            title = doc.short_title()
        except:
            title = urlparse(url).netloc

    if not title.strip():
        title = os.path.basename(urlparse(url).path).replace("-", " ").replace("_", " ")
        if not title:
            title = urlparse(url).netloc

    return {
        "title": title.strip() if isinstance(title, str) else "",
        "description": desc.strip() if isinstance(desc, str) else "",
        "author": author.strip() if isinstance(author, str) else "",
    }


def auto_generate_tags_from_text(text, top_k=5):
    raw = extract_keywords(text, diversity=0.7)
    flat: List[Tuple[str, float]] = []

    for item in raw:
        if isinstance(item, tuple) and len(item) == 2:
            flat.append(item)
        elif isinstance(item, list):
            flat.extend(i for i in item if isinstance(i, tuple) and len(i) == 2)

    tag_set = []
    for kw, _ in flat:
        if not kw:
            continue
        norm_kw = normalize_tag(kw)
        if (
            is_sensible_keyword(kw)
            and norm_kw not in BAD_TAGS
            and not re.search(r"(consent|cookie|agree|explicit|age|18)", norm_kw)
        ):
            tag_set.append(norm_kw)

    return tag_set[:top_k]


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

        if (
            is_probable_video_url(full_url)
            or href.lower().endswith((".mp4", ".webm", ".m3u8"))
            or "/video" in href.lower()
            or re.search(r"/\d{4,}", href)
        ):
            urls.add(full_url)

    logging.debug(f"Deep candidate links from homepage: {len(urls)}")
    return list(urls)[:20]


async def extract_video_sources(html, base_url, power_scraping):
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
            "data-preview",
            "data-srcset",
            "data-file",
        ]:
            src = tag.get(src_attr)
            if src:
                full_url = urljoin(base_url, str(src))
                if re.search(r"\.(mp4|webm|m3u8|mov)(\?.*)?$", full_url, re.IGNORECASE):
                    sources.add(full_url)

    for iframe in soup.find_all("iframe", src=True):
        if not isinstance(iframe, Tag):
            continue
        src = iframe.get("src", "")
        if src and ("player" in src or "embed" in src):
            iframe_url = urljoin(base_url, str(src))
            try:
                iframe_html, _ = await fetch_rendered_html_playwright(
                    iframe_url, power_scraping
                )
                nested_sources = await extract_video_sources(
                    iframe_html, iframe_url, power_scraping
                )
                sources.update(nested_sources)
            except Exception as e:
                logging.error(f"IFRAME RECURSION ERROR - {iframe_url}: {e}")

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

    if not sources:
        matches = re.findall(r'(https?://[^\s\'"]+\.(mp4|webm|m3u8|mov))', html)
        if matches:
            for match in matches:
                sources.add(match[0])

    meta_tags = [
        ("property", "og:video"),
        ("property", "og:video:url"),
        ("property", "og:video:secure_url"),
        ("name", "twitter:player"),
        ("name", "twitter:player:stream"),
    ]

    for attr_name, attr_value in meta_tags:
        tag = soup.find("meta", attrs={attr_name: attr_value})
        if isinstance(tag, Tag):
            content = tag.get("content")
            if isinstance(content, str):
                sources.add(urljoin(base_url, content))

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            if not isinstance(script, Tag) or not script.string:
                continue
            data = json.loads(script.string)
            if isinstance(data, dict):
                data = [data]
            for item in data:
                if isinstance(item, dict) and item.get("@type") == "VideoObject":
                    video_url = item.get("contentUrl") or item.get("embedUrl")
                    if video_url and is_probable_video_url(video_url):
                        sources.add(urljoin(base_url, video_url))
        except (json.JSONDecodeError, TypeError):
            continue

    if sources:
        logging.info(f"EXTRACTED - Found {len(sources)} video URLs")

    logging.debug(f"Found {len(sources)} video links on {base_url}")

    sorted_sources = sorted(
        sources,
        key=lambda x: (
            ".mp4" in x,
            ".webm" in x,
            ".m3u8" in x,
            ".ts" in x,
            "cdn" in x,
            "stream" in x,
        ),
        reverse=True,
    )

    return list(dict.fromkeys(sorted_sources))


async def extract_video_metadata(url, query_embed, power_scraping):
    if not is_valid_link(url):
        return None

    html, intercepted_links = await fetch_rendered_html_playwright(url)
    video_links = intercepted_links
    html = preprocess_html(html)
    if not html or not html.strip():
        logging.error(f"No HTML content fetched from {url}")
        return None

    video_links = await extract_video_sources(html, url, power_scraping)

    if not video_links and intercepted_links:
        logging.warning(f"Fallback: using intercepted links directly for {url}")
        video_links = intercepted_links

    video_links = list(dict.fromkeys(video_links + intercepted_links))
    candidate_links = []

    if not video_links:
        candidate_links = extract_video_candidate_links(html, url)
        if not candidate_links:
            logging.info(f"BAILOUT - No video sources or candidate links for {url}")
            return None

    if not video_links:
        ranked_deep_links = rank_deep_links(candidate_links, query_embed)[:5]

        async def try_fetch_deep_link(deep_url):
            try:
                deep_html, _ = await fetch_rendered_html_playwright(
                    deep_url, power_scraping=power_scraping
                )
                if not deep_html or re.search(
                    r"(404|Page not found|\b(error|fail|unavailable)\b)",
                    deep_html,
                    re.I,
                ):
                    return None, None, None
                deep_sources = await extract_video_sources(
                    deep_html, deep_url, power_scraping
                )
                if deep_sources:
                    return deep_url, deep_html, deep_sources
            except Exception as e:
                logging.error(f"DEEP LINK ERROR - {deep_url}: {e}")
            return None, None, None

        results = await asyncio.gather(
            *[try_fetch_deep_link(url) for url in ranked_deep_links]
        )

        for deep_url, deep_html, deep_sources in results:
            if deep_url and deep_html and deep_sources:
                html = deep_html
                url = deep_url
                video_links = deep_sources
                break

    soup = BeautifulSoup(html, "lxml")
    for script in soup.find_all("script"):
        if not isinstance(script, Tag):
            continue

        script_text = getattr(script, "string", None)
        if not script_text:
            continue

        matches = re.findall(r'atob\(["\']([^"\']+)["\']\)', script_text)
        for b64 in matches:
            try:
                decoded = base64.b64decode(b64.strip()).decode("utf-8")
                if decoded.startswith("http") and is_probable_video_url(decoded):
                    if decoded not in video_links:
                        logging.debug(f"BASE64 VIDEO - Decoded and added: {decoded}")
                        video_links.append(decoded)
            except Exception as e:
                logging.warning(f"BASE64 DECODE FAIL - {b64[:30]}...: {e}")

    try:
        text = content_extractor.get_content(html)
    except Exception as e:
        logging.error(f"BOILERPY3 FAILED - {e}")
        text = ""

    if not text:
        soup = BeautifulSoup(html, "lxml")
        text = soup.get_text(separator="\n").strip()

    if len(text) < 200:
        try:
            doc = Document(html)
            text = doc.summary()
            text = BeautifulSoup(text, "lxml").get_text("\n")
        except:
            text = ""

    if len(text) < 200:
        text = extract_content(html)

    if len(text) < 100 and video_links:
        if video_links:
            logging.warning(
                f"Short extracted text (<100 chars) but videos were found. Proceeding anyway."
            )
        else:
            logging.error(f"Unable to extract usable text and no videos: {url}")
            return None

    lang = detect_language(text)
    if lang != "en":
        text = translate_text(text)

    meta = await extract_metadata(html, url)
    tags = auto_generate_tags_from_text(f"{text.strip()} {meta['title']}", top_k=10)

    videos = []
    seen_video_urls = set()
    fallback_log_tracker = defaultdict(list)

    async def get_metadata_for(video_url):
        if video_url in seen_video_urls:
            return None
        seen_video_urls.add(video_url)

        try:
            duration = await asyncio.wait_for(
                get_video_duration(video_url, html),
                timeout=60 if power_scraping else 30,
            )
        except Exception as e:
            logging.warning(f"DURATION FAIL - {video_url}: {e}")
            duration = 0.0

        is_stream = "m3u8" in video_url

        if duration == 0.0:
            duration = 60.0 if is_stream else 30.0
            parsed = urlparse(video_url)
            base = f"{parsed.scheme}://{parsed.netloc}"
            fallback_log_tracker[base].append(video_url)

        if video_url.startswith("blob:"):
            logging.debug(f"Skipping blob URL: {video_url}")
            return None

        if not tags:
            logging.warning(f"Accepting video with no tags: {video_url}")

        title = (
            meta["title"]
            or os.path.basename(urlparse(video_url).path.rstrip("/"))
            or urlparse(video_url).netloc
            or "Untitled"
        )

        return {
            "url": video_url,
            "title": title,
            "tags": tags,
            "duration": f"{duration:.2f}",
            "score": 0.0,
            "is_stream": is_stream,
        }

    tasks = [get_metadata_for(url) for url in video_links]
    results = await asyncio.gather(*tasks, return_exceptions=False)
    videos = [v for v in results if v is not None]

    for base, urls in fallback_log_tracker.items():
        logging.warning(f"Fallback duration for {base} ({len(urls)} video(s)): 30.0s")

    if not video_links:
        if intercepted_links:
            video_links = intercepted_links
            logging.warning(f"Fallback: using intercepted links directly for {url}")
        else:
            logging.info(f"BAILOUT - No video sources or candidate links for {url}")
            return None

    videos = await deduplicate_videos(videos)

    return {
        "url": url,
        "title": meta["title"] or urlparse(url).netloc,
        "description": meta["description"],
        "videos": videos,
    }


# === 🔍 SEARCH PIPELINE ===
async def search_engine_async(query, link_count):
    payload = {"q": query, "format": "json", "language": "en"}
    retries = 3
    last_data = {}

    for attempt in range(retries):
        try:
            session = await get_client_session()
            async with session.post(
                SEARXNG_BASE_URL,
                data=payload,
                headers={"User-Agent": get_user_agent()},
            ) as resp:
                if resp.status != 200:
                    raise Exception(f"HTTP {resp.status}")
                data = await resp.json()
                last_data = data
                results = []
                for r in data.get("results", []):
                    u = r.get("url")
                    if not is_valid_link(u):
                        continue
                    if urlparse(u).path.strip("/") == "":
                        continue
                    results.append(u)
                if results:
                    return results[:link_count]
        except Exception as e:
            logging.info(f"SEARCH ERROR - Attempt {attempt+1}: {e}")
            await asyncio.sleep(2 * (attempt + 1))

    logging.info("FALLBACK - Using partial data from last attempt.")

    fallback = []
    for r in last_data.get("results", []) if last_data else []:
        u = r.get("url")
        if not is_valid_link(u):
            continue
        if urlparse(u).path.strip("/") == "":
            continue
        fallback.append(u)

    return fallback[:link_count]


async def search_videos_async(
    query="4k videos", videos_to_return: int = 1, power_scraping=False
):
    video_count = 0
    max_videos = videos_to_return

    expanded_queries = [q for q in expand_query_semantically(query) if q.strip()][:5]
    if not expanded_queries or all(not q.strip() for q in expanded_queries):
        expanded_queries = [query]
    logging.debug(f"Expanded queries: {expanded_queries}")

    query_embed = model_embed.encode_cached(query)
    all_links, results, processed, seen_video_urls = [], [], set(), set()
    collected = set()
    sem = asyncio.Semaphore(min(MAX_PARALLEL_TASKS, 8))
    domain_failures = defaultdict(int)

    async def worker(url):
        domain = get_main_domain(url)
        logging.info(f"WORKER - Starting: {url}")

        if domain_failures[domain] > 3:
            logging.info(f"SKIP - Too many failures for {domain}")
            return url, None

        async with sem, domain_counters[domain]:
            try:
                try:
                    result = await asyncio.wait_for(
                        extract_video_metadata(url, query_embed, power_scraping),
                        timeout=60 if not power_scraping else 120,
                    )
                except asyncio.TimeoutError:
                    logging.warning(
                        f"RETRY - Timeout, retrying with reduced timeout: {url}"
                    )
                    result = await asyncio.wait_for(
                        extract_video_metadata(url, query_embed, power_scraping),
                        timeout=45 if power_scraping else 30,
                    )

                if result and result.get("videos"):
                    return url, result
            except asyncio.TimeoutError:
                logging.warning(f"TIMEOUT - {url}")
            except Exception as e:
                logging.info(f"WORKER ERROR - {url}: {e.__class__.__name__} - {e}")
            finally:
                domain_failures[domain] += 1
                return url, None

    i, tasks = 0, []
    max_time = 360
    start_time = time.monotonic()

    while video_count < max_videos:
        elapsed = time.monotonic() - start_time
        if elapsed > max_time:
            logging.info(f"TIMEOUT - Collected {video_count}/{max_videos}")
            break

        if i >= len(all_links):
            logging.info("LOOP - Fetching more links...")
            needed = max_videos - video_count
            cores = os.cpu_count() or 4
            ram_gb = psutil.virtual_memory().total // 1_073_741_824
            multiplier = max(2, min(1 + (cores // 4) + (ram_gb // 8), 6))
            estimated_links = max(20, multiplier * needed)
            per_query = max(2, estimated_links // len(expanded_queries))

            for q in expanded_queries:
                try:
                    links = await search_engine_async(q, per_query)
                    await asyncio.sleep(0.5)
                    new_links = [
                        u for u in links if u not in collected and u not in processed
                    ]
                    all_links.extend(new_links)
                    all_links = list(dict.fromkeys(all_links))
                    collected.update(new_links)
                except Exception as e:
                    logging.error(f"LINK ERROR - {q}: {e}")

        while i < len(all_links) and len(tasks) < MAX_PARALLEL_TASKS:
            url = all_links[i]
            if url not in processed:
                tasks.append(asyncio.create_task(worker(url)))
            i += 1

        if not tasks:
            await asyncio.sleep(0.2)
            continue

        results_raw = await asyncio.gather(*tasks, return_exceptions=True)
        tasks.clear()

        for result in results_raw:
            if isinstance(result, BaseException):
                logging.error(f"TASK ERROR - {type(result).__name__}: {result}")
                continue

            if not isinstance(result, (tuple, list)) or len(result) != 2:
                logging.error(f"TASK FORMAT ERROR - Unexpected result: {result}")
                continue

            url, data = result
            if url:
                processed.add(url)
            if not data:
                continue

            new_videos = []
            for video in data.get("videos", []):
                norm_url = normalize_url(video["url"])
                if norm_url not in seen_video_urls:
                    seen_video_urls.add(norm_url)
                    new_videos.append(video)

            if new_videos:
                remaining = max_videos - video_count
                accepted = new_videos[:remaining]
                data["videos"] = accepted
                results.append(data)
                video_count += len(accepted)
                logging.info(
                    f"RESULT - {len(accepted)} accepted from {url} — total: {video_count}/{max_videos}"
                )

                if video_count >= max_videos:
                    logging.info("SUCCESS - Target reached.")
                    break

        if video_count >= max_videos:
            break

    all_videos = []
    for result in results:
        for video in result["videos"]:
            video["parent_url"] = result["url"]
            video["source_title"] = result["title"]
            all_videos.append(video)

    ranked = rank_by_similarity(all_videos, query)
    if not any(v.get("score", 0.0) > 0.0 for v in ranked):
        ranked.sort(key=lambda v: float(v.get("duration", 0)), reverse=True)
    ranked = ranked[:max_videos]

    if len(ranked) < max_videos:
        logging.info(
            f"FILLING - Only {len(ranked)}/{max_videos} ranked. Padding with fallback..."
        )
        seen_urls = {v["url"] for v in ranked}
        fallback = [v for v in all_videos if v["url"] not in seen_urls]
        ranked.extend(fallback[: max_videos - len(ranked)])

    if len(ranked) < max_videos:
        leftovers = [
            v for v in all_videos if v["url"] not in {v["url"] for v in ranked}
        ]
        ranked.extend(leftovers[: max_videos - len(ranked)])

    ranked = ranked[:max_videos]

    grouped = {}
    for video in ranked:
        url = video["parent_url"]
        if url not in grouped:
            grouped[url] = {
                "url": url,
                "title": video.get("source_title", ""),
                "description": "",
                "videos": [],
            }
        grouped[url]["videos"].append(video)

    results = list(grouped.values())

    if video_count < max_videos:
        logging.warning(
            f"Only {video_count}/{max_videos} collected. Consider increasing timeout or search depth."
        )

    if not ranked:
        logging.warning(
            "No videos ranked. Collected but possibly filtered or invalid format."
        )

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
    await asyncio.gather(flush_task, return_exceptions=True)
    try:
        await flush_task
    except asyncio.CancelledError:
        pass

    if client_session:
        await client_session.close()

    if _playwright_context:
        await _playwright_context.close()

    if _playwright_browser:
        await _playwright_browser.close()

    if _playwright_obj:
        await _playwright_obj.stop()


app = FastAPI(lifespan=lifespan)


@app.get("/", response_class=HTMLResponse)
def index():
    return open("index.html", encoding="utf-8").read()


@app.get("/search")
async def search(query: str = "", power_scraping: bool = False):
    global start_time
    start_time = time.monotonic()

    logging.info("=== STARTING SEARCH ===")
    MINIMUM_VIDEOS_TO_RETURN = 5
    videos_to_return = (
        min((os.cpu_count() or 4) * 10, 500)
        if power_scraping
        else MINIMUM_VIDEOS_TO_RETURN
    )

    try:
        results = await search_videos_async(query, videos_to_return, power_scraping)
    except asyncio.TimeoutError:
        logging.info(f"TIMEOUT - Search query timed out: {query}")
        results = []

    logging.info(f"RESULTS - Total results fetched: {len(results)}")
    video_results = [r for r in results if r.get("videos")]

    if video_results:
        logging.info(f"RESULTS - {len(video_results)} result(s) contain videos")
        results = video_results
    else:
        logging.info(f"RESULTS - No videos found, falling back to raw results")

    all_videos = []
    for result in results:
        for video in result.get("videos", []):
            video["parent_url"] = result["url"]
            video["source_title"] = result["title"]
            all_videos.append(video)

    try:
        deduped = await asyncio.wait_for(deduplicate_videos(all_videos), timeout=30)
    except asyncio.TimeoutError:
        logging.info("DEDUPLICATION TIMEOUT - Skipping deduplication step")
        deduped = all_videos

    ranked_videos = rank_by_similarity(deduped, query)[:videos_to_return]

    ranked_results = {}
    for video in ranked_videos:
        parent_url = video["parent_url"]
        if parent_url not in ranked_results:
            ranked_results[parent_url] = {
                "url": parent_url,
                "title": video.get("source_title", ""),
                "description": "",
                "videos": [],
            }
        ranked_results[parent_url]["videos"].append(video)

    results = list(ranked_results.values())

    os.environ["DISABLE_POST_RESPONSE_LOGS"] = "1"

    logging.debug("=== FINAL RESULTS JSON ===")
    logging.debug(json.dumps(results, indent=2))

    return JSONResponse(
        content=[
            {
                "url": r["url"],
                "title": r["title"],
                "description": r["description"],
                "videos": [
                    {
                        **{k: v for k, v in video.items() if k not in {"score"}},
                        "duration": (
                            video["duration"]
                            if float(video.get("duration", 0)) > 0
                            else "unknown"
                        ),
                        "is_stream": video.get("is_stream", False),
                    }
                    for video in r["videos"]
                ],
            }
            for r in results
        ]
    )

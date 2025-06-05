# === 📦 IMPORTS ===
import os, json, time, uuid, random, hashlib, asyncio
from urllib.parse import urlparse, urlunparse

import torch, open_clip, aiohttp, tldextract, trafilatura
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from contextlib import asynccontextmanager
from bs4 import BeautifulSoup, Tag
from readability import Document
from langdetect import DetectorFactory, detect
from deep_translator import GoogleTranslator
from aiohttp import ClientTimeout
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from keybert import KeyBERT
from playwright.async_api import async_playwright

# === ⚙️ CONFIGURATION ===
DetectorFactory.seed = 0
timeout_obj = ClientTimeout(total=5)
MAX_PARALLEL_TASKS = os.cpu_count() or 4
CACHE_EXPIRATION = 10
SEARXNG_BASE_URL = "http://localhost:8888/search"

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
    "Mozilla/5.0 (X11; Linux x86_64)",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
]

BLOCKED_DOMAINS = {...}
BLOCKED_KEYWORDS = [...]
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


# === 🧠 Qdrant Indexer ===
class QdrantIndexer:
    def __init__(self, client, collection="videos", batch_size=50):
        self.client = client
        self.collection = collection
        self.batch_size = batch_size
        self.lock = asyncio.Lock()
        self.buffer: list[PointStruct] = []

    async def add_point(self, point: PointStruct):
        async with self.lock:
            try:
                self.client.delete(
                    collection_name=self.collection,
                    points_selector={"points": [point.id]},
                )
            except Exception as e:
                print(f"[Indexer] Failed to delete old point {point.id}: {e}")

            self.buffer.append(point)
            if len(self.buffer) >= self.batch_size:
                await self.flush()

    async def flush(self):
        async with self.lock:
            if self.buffer:
                print(f"[Indexer] Flushing {len(self.buffer)} items to Qdrant...")
                self.client.upsert(
                    collection_name=self.collection, points=self.buffer.copy()
                )
                self.buffer.clear()


# === 🔍 QDRANT SETUP ===
qdrant = QdrantClient(host="localhost", port=6333)
qdrant.recreate_collection(
    "videos", vectors_config=VectorParams(size=512, distance=Distance.COSINE)
)
indexer = QdrantIndexer(qdrant)


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


# === 📥 CACHE SYSTEM ===
def build_cache_path(url, folder, ext):
    def ensure_str(value) -> str:
        if isinstance(value, memoryview):
            value = bytes(value)
        if isinstance(value, (bytes, bytearray)):
            return value.decode("utf-8", errors="ignore")
        return str(value)

    normalized = normalize_url(ensure_str(url))

    if not isinstance(normalized, str):
        raise TypeError("Normalized URL must be a string")

    fname = hashlib.md5(normalized.encode("utf-8")).hexdigest()
    return f"{folder}/{fname}.{ext}"


def read_cache(path):
    if (
        os.path.exists(path)
        and (time.time() - os.path.getmtime(path)) < CACHE_EXPIRATION
    ):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f) if path.endswith(".json") else f.read()
    return None


def save_cache(path, content):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        if path.endswith(".json"):
            json.dump(content, f, ensure_ascii=False, indent=2)
        else:
            f.write(content)


def cache_html(url):
    return build_cache_path(url, "cache/html", "html")


def cache_summary(url):
    return build_cache_path(url, "cache/summary", "json")


def clean_expired_cache(folder="cache", expiration=CACHE_EXPIRATION):
    if not os.path.exists(folder):
        return
    now = time.time()
    for root, _, files in os.walk(folder):
        for f in files:
            path = os.path.join(root, f)
            if (now - os.path.getmtime(path)) > expiration:
                try:
                    os.remove(path)
                except:
                    continue


# === 🌐 RENDER + EXTRACT ===
async def fetch_rendered_html_playwright(url, timeout=15000):
    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
            )
            context = await browser.new_context(
                user_agent=get_user_agent(), viewport={"width": 1280, "height": 720}
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


def filter_text(text):
    blacklist = ["advertisement", "cookies", "policy", "login", "register"]
    return "\n".join(
        [
            l.strip()
            for l in text.splitlines()
            if len(l.strip()) >= 15 and not any(b in l.lower() for b in blacklist)
        ]
    )


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


def extract_relevant_links_qdrant(query_embed, top_k=5):
    return [
        r.payload["source_url"]
        for r in qdrant.search(
            collection_name="videos", query_vector=query_embed.tolist(), limit=top_k
        )
        if r.payload and "source_url" in r.payload
    ]


def extract_video_links_qdrant(query_embed, top_k=5):
    return [
        r.payload["video_url"]
        for r in qdrant.search(
            collection_name="videos", query_vector=query_embed.tolist(), limit=top_k
        )
        if r.payload and "video_url" in r.payload
    ]


async def process_url_async(url, query_embed):
    if not is_valid_link(url):
        return None
    html = await fetch_rendered_html_playwright(url)
    if not html:
        return None

    text = filter_text(extract_content(html))
    if len(text) < 200:
        try:
            text = filter_text(
                BeautifulSoup(Document(html).summary(), "lxml").get_text("\n")
            )
        except:
            text = ""
    if len(text) < 200:
        try:
            text = filter_text(BeautifulSoup(html, "lxml").get_text("\n"))
        except:
            return None

    lang = detect_language(text)
    if lang != "en":
        text = translate_text(text)

    summary_cache = read_cache(cache_summary(url))
    text_hash = hashlib.md5(text.encode()).hexdigest()
    if isinstance(summary_cache, dict) and summary_cache.get("hash") == text_hash:
        return summary_cache

    meta = await extract_metadata(html)
    result = {
        "url": url,
        "summary": text.strip(),
        "summary_links": extract_relevant_links_qdrant(query_embed),
        "video_links": extract_video_links_qdrant(query_embed),
        "hash": text_hash,
        "title": meta["title"],
        "description": meta["description"],
        "author": meta["author"],
        "language": "en",
    }
    save_cache(cache_summary(url), result)

    point = PointStruct(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, normalize_url(url))),
        vector=model_embed.encode(
            f"{meta['title']} {meta['description']} {text}"
        ).tolist(),
        payload={
            "title": meta["title"],
            "description": meta["description"],
            "tags": auto_generate_tags_from_text(text),
            "video_url": (result["video_links"][0] if result["video_links"] else ""),
            "source_url": url,
        },
    )
    await indexer.add_point(point)
    return result


async def search_engine_async(query, link_count):
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
            ][:link_count]


async def advanced_search_async(query, links_to_scrap, max_sites):
    query_embed = model_embed.encode(query)
    all_links, results, processed = [], [], set()
    collected, sem = set(), asyncio.Semaphore(MAX_PARALLEL_TASKS)

    async def worker(url):
        async with sem:
            try:
                return await process_url_async(url, query_embed)
            except:
                return None

    i, tasks = 0, []
    while len(results) < max_sites:
        if i >= len(all_links):
            links = await search_engine_async(query, links_to_scrap)
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

    await indexer.flush()

    return results


# === 🌐 FASTAPI ROUTES ===
@asynccontextmanager
async def lifespan(_: FastAPI):
    async def auto_flush_loop():
        while True:
            await asyncio.sleep(10)
            await indexer.flush()

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
async def search(
    query: str = "batata inglesa", links_to_scrap: int = 10, summaries: int = 5
):
    clean_expired_cache()
    results = await advanced_search_async(query, links_to_scrap, summaries)
    return JSONResponse(
        content=[
            {
                "title": r["title"] or r["url"],
                "summary": r["summary"],
                "links": r.get("summary_links", []),
                "videos": r.get("video_links", []),
            }
            for r in results
        ]
    )


@app.get("/qdrant_search")
async def qdrant_search(query: str, top_k: int = 10):
    vec = model_embed.encode(query).tolist()
    results = sorted(
        qdrant.search("videos", query_vector=vec, limit=top_k),
        key=lambda r: r.score,
        reverse=True,
    )
    return JSONResponse(
        content=[
            {
                "title": (p := r.payload or {}).get("title"),
                "description": p.get("description"),
                "video_url": p.get("video_url"),
                "source_url": p.get("source_url"),
                "score": r.score,
                "tags": p.get("tags", []),
            }
            for r in results
        ]
    )

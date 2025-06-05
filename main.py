import asyncio
import hashlib
import json
import os
import random
import time
from urllib.parse import urljoin, urlparse, urlunparse
import uuid

import aiohttp
import numpy as np
import tldextract
import trafilatura
from aiohttp import ClientTimeout
from bs4 import BeautifulSoup, Tag
from deep_translator import GoogleTranslator
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from langdetect import DetectorFactory, detect
from readability import Document
from playwright_scraper import fetch_rendered_html_playwright
from embedder import OpenCLIPEmbedder
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from keybert import KeyBERT


# App setup
app = FastAPI()
qdrant = QdrantClient(host="localhost", port=6333)
qdrant.recreate_collection(
    collection_name="videos",
    vectors_config=VectorParams(size=512, distance=Distance.COSINE),
)
kw_model = KeyBERT("sentence-transformers/all-MiniLM-L6-v2")


# Initialization
DetectorFactory.seed = 0

# Constants
MAX_PARALLEL_TASKS = os.cpu_count() or 4
CACHE_EXPIRATION = 10
SEARXNG_BASE_URL = "http://localhost:8888/search"
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

# Clients & models
timeout_obj = ClientTimeout(total=5)
model_embed = OpenCLIPEmbedder()


def detect_language(text):
    try:
        lang = detect(text)
        return lang if lang not in LANGUAGES_BLACKLIST else "en"
    except Exception:
        return "en"


def translate_text(text, target="en"):
    try:
        return GoogleTranslator(source="auto", target=target).translate(text)
    except Exception:
        return text


def get_user_agent():
    return random.choice(USER_AGENTS)


def normalize_url(url):
    parts = urlparse(url)
    return urlunparse((parts.scheme, parts.netloc, parts.path, "", "", ""))


def get_main_domain(url):
    ext = tldextract.extract(url)
    return f"{ext.domain}.{ext.suffix}" if ext.suffix else ext.domain


def is_valid_link(url):
    domain = get_main_domain(url).lower()
    if domain in BLOCKED_DOMAINS:
        return False
    if not url.lower().startswith(("http://", "https://")):
        return False
    if any(c in url for c in ['"', "'", "\\", " "]):
        return False
    if any(kw in url.lower() for kw in BLOCKED_KEYWORDS):
        return False
    if url.lower().split("?")[0].split("#")[0].split(".")[-1] in [
        "pdf",
        "doc",
        "xls",
        "zip",
        "rar",
        "ppt",
    ]:
        return False
    return True


def build_cache_path(url, folder, extension):
    filename = hashlib.md5(normalize_url(url).encode()).hexdigest()
    return f"{folder}/{filename}.{extension}"


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


def read_cache_html(url):
    return read_cache(build_cache_path(url, "cache/html", "html"))


def save_cache_html(url, html):
    save_cache(build_cache_path(url, "cache/html", "html"), html)


def read_cache_summary(url):
    return read_cache(build_cache_path(url, "cache/summary", "json"))


def save_cache_summary(url, summary):
    save_cache(build_cache_path(url, "cache/summary", "json"), summary)


def clean_expired_cache(folder="cache", expiration=CACHE_EXPIRATION):
    if not os.path.exists(folder):
        return
    now = time.time()
    for root, _, files in os.walk(folder):
        for file in files:
            path = os.path.join(root, file)
            if os.path.isfile(path) and (now - os.path.getmtime(path)) > expiration:
                try:
                    os.remove(path)
                except Exception:
                    continue


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
                "ads",
                "advertisement",
                "form",
                "input",
                "button",
                "aside",
                "comment",
                "meta",
                "link",
            ]
        ):
            tag.decompose()
        return str(soup)
    except Exception:
        return html


def extract_content(html):
    try:
        return (
            trafilatura.extract(html, include_comments=False, include_tables=False)
            or ""
        )
    except Exception:
        return ""


def safe_strip(value):
    return value.strip() if isinstance(value, str) else ""


def auto_generate_tags_from_text(text, top_k=5):
    keywords = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 3),
        stop_words="english",
        use_mmr=True,
        diversity=0.7,
        top_n=top_k,
    )
    return [kw for kw, _ in keywords]


async def extract_metadata(html):
    soup = BeautifulSoup(html, "lxml")
    meta = {
        "title": safe_strip(soup.title.string) if soup.title else "",
        "description": "",
        "author": "",
    }
    desc = soup.find("meta", attrs={"name": "description"}) or soup.find(
        "meta", attrs={"property": "og:description"}
    )
    author = soup.find("meta", attrs={"name": "author"})
    if isinstance(desc, Tag):
        meta["description"] = safe_strip(desc.get("content"))
    if isinstance(author, Tag):
        meta["author"] = safe_strip(author.get("content"))
    return meta


def filter_text(text):
    lines = text.splitlines()
    blacklist = [
        "advertisement",
        "ads",
        "cookies",
        "terms",
        "privacy",
        "policy",
        "login",
        "sign up",
        "click here",
        "subscribe",
        "register",
        "comment section",
    ]
    return "\n".join(
        line.strip()
        for line in lines
        if len(line.strip()) >= 15 and not any(b in line.lower() for b in blacklist)
    )


async def search_engine_async(query, link_count):
    payload = {"q": query, "format": "json", "language": "en"}
    ranked_links = []

    try:
        async with aiohttp.ClientSession(timeout=timeout_obj) as session:
            async with session.post(
                url=SEARXNG_BASE_URL,
                data=payload,
                headers={"User-Agent": get_user_agent()},
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                for result in data.get("results", []):
                    link = result.get("url")
                    if link and is_valid_link(link):
                        ranked_links.append((0.0, link))
        return [link for _, link in ranked_links[:link_count]]
    except Exception as e:
        return []


def extract_relevant_links_from_html_qdrant(query_embed, top_k=5):
    results = qdrant.search(
        collection_name="videos", query_vector=query_embed.tolist(), limit=top_k
    )
    return [
        r.payload["source_url"]
        for r in results
        if r.payload and "source_url" in r.payload
    ]


def extract_video_links_qdrant(query_embed, top_k=5):
    results = qdrant.search(
        collection_name="videos", query_vector=query_embed.tolist(), limit=top_k
    )
    return [
        r.payload["video_url"]
        for r in results
        if r.payload and "video_url" in r.payload
    ]


async def process_url_async(url, session, query_embed):
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
        except Exception:
            text = ""

    if len(text) < 200:
        try:
            soup = BeautifulSoup(html, "lxml")
            for tag in soup(
                ["script", "style", "header", "footer", "nav", "aside", "form"]
            ):
                tag.decompose()
            text = filter_text(soup.get_text("\n"))
        except Exception:
            return None

    video_links = extract_video_links_qdrant(query_embed)
    if len(text.strip()) < 100 and not video_links:
        return None

    lang = detect_language(text)
    if lang != "en":
        text = translate_text(text, "en")

    summary_cache = read_cache_summary(url)
    text_hash = hashlib.md5(text.encode()).hexdigest()
    if isinstance(summary_cache, dict) and summary_cache.get("hash") == text_hash:
        return summary_cache

    meta = await extract_metadata(html)
    result = {
        "url": url,
        "summary": text.strip(),
        "summary_links": extract_relevant_links_from_html_qdrant(query_embed),
        "video_links": video_links,
        "hash": text_hash,
        "title": meta.get("title", ""),
        "description": meta.get("description", ""),
        "author": meta.get("author", ""),
        "language": "en",
    }

    save_cache_summary(url, result)

    vector = model_embed.encode(
        f"{result['title']} {result['description']} {result['summary']}"
    )

    qdrant.upsert(
        collection_name="videos",
        points=[
            PointStruct(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, result["url"])),
                vector=vector.tolist(),
                payload={
                    "title": result["title"],
                    "description": result["description"],
                    "tags": auto_generate_tags_from_text(text),
                    "video_url": (
                        result["video_links"][0] if result["video_links"] else ""
                    ),
                    "source_url": result["url"],
                },
            )
        ],
    )

    return result


async def advanced_search_async(query, links_to_scrap, max_sites):
    query_embed = model_embed.encode(query)
    collected = set()
    all_links = []
    results = []
    processed = set()
    sem = asyncio.Semaphore(MAX_PARALLEL_TASKS)
    max_links = links_to_scrap

    async with aiohttp.ClientSession(timeout=timeout_obj) as session:

        async def worker(url):
            async with sem:
                try:
                    return await process_url_async(url, session, query_embed)
                except Exception:
                    return None

        def launch(i, tasks):
            while i < len(all_links) and len(tasks) < MAX_PARALLEL_TASKS:
                url = all_links[i]
                if url not in processed:
                    processed.add(url)
                    tasks.append(asyncio.create_task(worker(url)))
                i += 1
            return i

        i, tasks = 0, []
        while len(results) < max_sites:
            if i >= len(all_links):
                max_links += links_to_scrap
                links = await search_engine_async(query, max_links)
                new_links = [
                    u for u in links if is_valid_link(u) and u not in collected
                ]
                if not new_links:
                    break
                collected.update(new_links)
                all_links.extend(new_links)

            i = launch(i, tasks)
            if not tasks:
                break

            done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                tasks.remove(d)
                r = d.result()
                if r:
                    results.append(r)

    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return results


@app.get("/", response_class=HTMLResponse)
def index():
    return open("./index.html", encoding="utf-8").read()


@app.get("/search")
async def search(
    query: str = "O que é batata inglesa?", links_to_scrap: int = 10, summaries: int = 5
):
    clean_expired_cache()
    results = await advanced_search_async(query, links_to_scrap, summaries)
    return JSONResponse(
        content=[
            {
                "title": item["title"] or item["url"],
                "summary": item["summary"],
                "links": item.get("summary_links", []),
                "videos": item.get("video_links", []),
            }
            for item in results
        ]
    )


@app.get("/qdrant_search")
async def qdrant_search(query: str, top_k: int = 10):
    query_vector = model_embed.encode(query).tolist()

    results = qdrant.search(
        collection_name="videos", query_vector=query_vector, limit=top_k
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

import asyncio
import hashlib
import json
import os
import random
import time
from urllib.parse import urlencode, urljoin, urlparse, urlunparse

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
from sentence_transformers import SentenceTransformer

# App setup
app = FastAPI()

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
model_embed = SentenceTransformer("intfloat/e5-base-v2")


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
                "iframe",
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


async def download_html_async(url, session, lang="en"):
    cached = read_cache_html(url)
    if cached:
        return cached

    headers = {
        "User-Agent": get_user_agent(),
        "Accept-Language": "pt-BR,pt;q=0.9" if lang == "pt" else "en-US,en;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    }
    cookies = {"age_verified": "1", "RTA": "1"}

    for attempt in range(1, 3):
        try:
            async with session.get(
                url, headers=headers, cookies=cookies, timeout=timeout_obj
            ) as resp:
                if resp.status == 200:
                    html = await resp.text()
                    html = preprocess_html(html)
                    save_cache_html(url, html)
                    return html
        except Exception:
            pass
        await asyncio.sleep(2**attempt + random.uniform(0, 0.5))
    return ""


def cosine_similarity(v1, v2):
    return float(np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)))


async def search_engine_async(query, link_count):
    query_embed = model_embed.encode([query])[0]
    query_string = urlencode({"q": query, "format": "json", "language": "en"})
    url = f"{SEARXNG_BASE_URL}?{query_string}"
    ranked_links = []

    try:
        async with aiohttp.ClientSession(timeout=timeout_obj) as session:
            async with session.get(
                url, headers={"User-Agent": get_user_agent()}
            ) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json()
                for result in data.get("results", []):
                    link = result.get("url")
                    if link and is_valid_link(link):
                        title = result.get("title", "")
                        snippet = result.get("content", "")
                        score = cosine_similarity(
                            query_embed,
                            model_embed.encode([f"passage: {title} {snippet}"])[0],
                        )
                        ranked_links.append((score, link))
        ranked_links.sort(reverse=True)
        return [link for _, link in ranked_links[:link_count]]
    except Exception as e:
        return []


def extract_relevant_links_from_html(html, base_url, query_embed):
    soup = BeautifulSoup(html, "lxml")
    links_with_scores = []
    seen = set()

    for a in soup.find_all("a", href=True):
        if isinstance(a, Tag):
            href = a.get("href")
            text = a.get_text(strip=True)
            url = urljoin(base_url, str(href))
            if is_valid_link(url) and url not in seen and text:
                seen.add(url)
                sim = cosine_similarity(
                    query_embed, model_embed.encode([f"passage: {text}"])[0]
                )
                links_with_scores.append((sim, url))

    if links_with_scores:
        threshold = np.percentile([s for s, _ in links_with_scores], 75)
        return [
            url for sim, url in links_with_scores if sim >= max(float(threshold), 0.35)
        ][:5]
    return []


def extract_video_links_from_html(html, base_url, query_embed):
    soup = BeautifulSoup(html, "lxml")
    seen = set()
    links_with_scores = []

    for tag in soup.find_all(["a", "video", "source"], href=True) + soup.find_all(
        "a", src=True
    ):
        if not isinstance(tag, Tag):
            continue
        href = tag.get("href") or tag.get("src")
        text = tag.get("title") or tag.get("alt") or tag.get_text(strip=True)

        if href:
            url = urljoin(base_url, str(href))
            if not is_valid_link(url) or url in seen:
                continue
            seen.add(url)

            is_video = any(
                x in url.lower() for x in ["/video", "/watch", "/view", ".mp4", ".m3u8"]
            )
            score = 0.0

            if text:
                score = cosine_similarity(
                    query_embed, model_embed.encode([f"passage: {text}"])[0]
                )
                if is_video:
                    score += 0.3
            elif is_video:
                score = 0.4

            if score > 0.3:
                links_with_scores.append((score, url))
    links_with_scores.sort(reverse=True)
    return [url for _, url in links_with_scores[:5]]


async def process_url_async(url, session, query_embed):
    if not is_valid_link(url):
        return None

    html = await download_html_async(url, session)
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

    if len(text.strip()) < 300:
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
        "summary_links": extract_relevant_links_from_html(html, url, query_embed),
        "video_links": extract_video_links_from_html(html, url, query_embed),
        "hash": text_hash,
        "title": meta.get("title", ""),
        "description": meta.get("description", ""),
        "author": meta.get("author", ""),
        "language": "en",
    }

    save_cache_summary(url, result)
    return result


async def advanced_search_async(query, links_to_scrap, max_sites):
    query_embed = model_embed.encode([f"query: {query}"])[0]
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
                    summary_embed = model_embed.encode([f"passage: {r['summary']}"])[0]
                    title_embed = (
                        model_embed.encode([f"passage: {r['title']}"])[0]
                        if r["title"]
                        else query_embed
                    )
                    r["similarity"] = 0.7 * cosine_similarity(
                        query_embed, summary_embed
                    ) + 0.3 * cosine_similarity(query_embed, title_embed)
                    results.append(r)

    results = sorted(results, key=lambda x: x.get("similarity", 0), reverse=True)[
        :max_sites
    ]
    for r in results:
        r["similarity"] = float(r["similarity"])

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
                "links": item.get("relevant_links", []),
                "videos": item.get("video_links", []),
            }
            for item in results
        ]
    )

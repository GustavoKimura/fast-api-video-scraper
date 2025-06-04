import asyncio
import hashlib
import json
import logging
import os
import random
import time
from urllib.parse import urlencode, urljoin, urlparse, urlunparse

import aiohttp
import nltk
import numpy as np
import tldextract
import trafilatura
from aiohttp import ClientTimeout
from bs4 import BeautifulSoup, Tag
from deep_translator import GoogleTranslator
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fake_useragent import UserAgent
from langdetect import DetectorFactory, detect
from readability import Document
from sentence_transformers import SentenceTransformer
from transformers import GPT2TokenizerFast

app = FastAPI()
logging.basicConfig(level=logging.INFO)
log = logging.getLogger()

DetectorFactory.seed = 0


MAX_PARALLEL_TASKS = os.cpu_count() or 4
VERBOSE = True
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
    "reddit.com",
    "quora.com",
    "4chan.org",
    "tumblr.com",
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
    "reddit",
    "quora",
    "forum",
    "community",
    "board",
    "discussion",
    "chat",
    "signup",
    "login",
    "register",
    "comment",
    "thread",
    "showthread",
    "archive",
]

LANGUAGES_BLACKLIST = {"da", "so", "tl", "nl", "sv", "af", "el"}

timeout_obj = ClientTimeout(total=5)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
model_embed = SentenceTransformer("all-MiniLM-L6-v2")
ua = UserAgent()


def ensure_punkt():
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)


ensure_punkt()


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
    try:
        return ua.random
    except Exception:
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
    if any(char in url for char in ['"', "'", "\\", " "]):
        return False
    if any(bad in url.lower() for bad in BLOCKED_KEYWORDS):
        return False
    ext = url.lower().split("?")[0].split("#")[0].split(".")[-1]
    if ext in ["pdf", "doc", "xls", "zip", "rar", "ppt"]:
        return False
    if url.count("/") <= 2:
        return False
    return True


def cache_path_html(url):
    return f"cache/html/{hashlib.md5(normalize_url(url).encode()).hexdigest()}.html"


def cache_path_summary(url):
    return f"cache/summary/{hashlib.md5(normalize_url(url).encode()).hexdigest()}.json"


def read_cache(path):
    if os.path.exists(path):
        age = time.time() - os.path.getmtime(path)
        if age < CACHE_EXPIRATION:
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
    return read_cache(cache_path_html(url))


def save_cache_html(url, html):
    save_cache(cache_path_html(url), html)


def read_cache_summary(url):
    return read_cache(cache_path_summary(url))


def save_cache_summary(url, summary):
    save_cache(cache_path_summary(url), summary)


def clean_expired_cache(folder="cache", exp_secs=CACHE_EXPIRATION):
    if not os.path.exists(folder):
        return
    now = time.time()
    for root, _, files in os.walk(folder):
        for f in files:
            p = os.path.join(root, f)
            if os.path.isfile(p) and now - os.path.getmtime(p) > exp_secs:
                try:
                    os.remove(p)
                except Exception:
                    pass


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
    if isinstance(desc, Tag):
        meta["description"] = safe_strip(desc.get("content"))
    author = soup.find("meta", attrs={"name": "author"})
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
    filtered = []
    for l in lines:
        ls = l.strip()
        if len(ls) >= 15 and not any(x in ls.lower() for x in blacklist):
            filtered.append(ls)
    return "\n".join(filtered)


async def download_html_async(url, session, lang="en"):
    cached = read_cache_html(url)
    if cached:
        return cached
    headers = {
        "User-Agent": get_user_agent(),
        "Accept-Language": "en-US,en;q=0.9" if lang != "pt" else "pt-BR,pt;q=0.9",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Connection": "keep-alive",
    }
    cookies = {
        "age_verified": "1",
        "RTA": "1",
    }
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


def extract_relevant_links_from_html(html, base_url):
    soup = BeautifulSoup(html, "lxml")
    links, seen = [], set()
    for a in soup.find_all("a", href=True):
        if not isinstance(a, Tag):
            continue
        href = a.get("href")
        if not isinstance(href, str):
            continue
        url = urljoin(base_url, href)
        if is_valid_link(url) and url not in seen:
            seen.add(url)
            links.append(url)
    return links[:15]


async def process_url_async(url, session):
    if not is_valid_link(url):
        return None
    html = await download_html_async(url, session)
    if not html:
        return None
    text = filter_text(extract_content(html))
    relevant_links = extract_relevant_links_from_html(html, url)
    if len(text) < 200:
        try:
            soup = BeautifulSoup(Document(html).summary(), "lxml")
            text = filter_text(soup.get_text("\n"))
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
        lang = "en"
    summary_cache = read_cache_summary(url)
    text_hash = hashlib.md5(text.encode()).hexdigest()
    if isinstance(summary_cache, dict) and summary_cache.get("hash") == text_hash:
        return summary_cache
    meta = await extract_metadata(html)
    result = {
        "url": url,
        "summary": text.strip(),
        "relevant_links": relevant_links,
        "hash": text_hash,
        "title": meta.get("title", ""),
        "description": meta.get("description", ""),
        "author": meta.get("author", ""),
        "language": lang,
    }
    save_cache_summary(url, result)
    return result


async def search_engine_async(query, links_to_scrap):
    query_string = urlencode({"q": query, "format": "json", "language": "en"})
    url = f"{SEARXNG_BASE_URL}?{query_string}"
    collected_links = set()
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
                        norm = normalize_url(link)
                        if norm not in collected_links:
                            collected_links.add(norm)
                    if len(collected_links) >= links_to_scrap:
                        break
    except Exception as e:
        log.warning(f"[SEARXNG ERROR] {e}")
    return list(collected_links)[:links_to_scrap]


def cosine_similarity(v1, v2):
    return float(np.dot(v1 / np.linalg.norm(v1), v2 / np.linalg.norm(v2)))


async def advanced_search_async(query, links_to_scrap, max_sites):
    collected, all_links, results, processed = set(), [], [], set()
    sem = asyncio.Semaphore(MAX_PARALLEL_TASKS)
    max_links = links_to_scrap
    async with aiohttp.ClientSession(timeout=timeout_obj) as session:

        async def worker(url):
            async with sem:
                try:
                    return await process_url_async(url, session)
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
    query_embed = model_embed.encode([query])[0]
    for r in results:
        title = r["title"] or ""
        sim_title = model_embed.encode([title])[0] if title else query_embed
        r["similarity"] = 0.7 * cosine_similarity(
            query_embed, model_embed.encode([r["summary"]])[0]
        ) + 0.3 * cosine_similarity(query_embed, sim_title)
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
    return """
<html>
  <head>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
      body {
        margin: 0;
        padding: 16px;
        font-family: sans-serif;
        background: #000;
      }
      form {
        display: flex;
        flex-direction: column;
        gap: 12px;
        max-width: 400px;
        width: 100%;
        margin: 0 auto 16px auto;
      }
      input {
        font-size: 16px;
        padding: 8px;
        width: 100%;
        box-sizing: border-box;
      }
      label {
        font-size: 14px;
        display: flex;
        align-items: center;
        gap: 6px;
        color: white;
      }
      button {
        font-size: 16px;
        padding: 8px;
      }
      #o {
        background: #1a1a1a;
        color: #e0e0e0;
        padding: 12px;
        border-radius: 4px;
        margin-top: 16px;
        text-align: justify;
        word-break: break-word;
        overflow-wrap: break-word;
        max-width: 100%;
        box-sizing: border-box;
      }
      #o a {
        color: #5ab4f0;
        text-decoration: none;
        word-break: break-word;
      }
      #o a:hover {
        color: #82cfff;
        text-decoration: underline;
      }
    </style>
  </head>
  <body>
    <form id="f">
      <label for="q">Busca:</label>
      <input id="q" type="text" placeholder="Qualquer besteira aqui..." required>
      <label for="l">Quantidade de links a processar (Ex. 10):</label>
      <input id="l" type="number" placeholder="Ex. 10" value="10" min="1" required>
      <label for="s">Quantidade de resumos desejados (Ex. 5):</label>
      <input id="s" type="number" placeholder="Ex. 5" value="5" min="1" required>
      <button>Buscar</button>
    </form>
    <div id="o">Os resultados da busca aparecerão aqui.</div>
    <script>
      const f = document.getElementById('f');
      const q = document.getElementById('q');
      const l = document.getElementById('l');
      const s = document.getElementById('s');
      const o = document.getElementById('o');
      f.onsubmit = async e => {
        e.preventDefault();
        o.textContent = 'Carregando...';
        let query = encodeURIComponent(q.value);
        let linksToScrap = parseInt(l.value);
        let summaries = parseInt(s.value);
        let result = await fetch(`/search?query=${query}&links_to_scrap=${linksToScrap}&summaries=${summaries}`);
        let data = await result.json();
        o.innerHTML = data.map((item, idx) => `
          <div style="margin-bottom:20px; border-bottom:1px solid #eee; padding-bottom:12px;">
            <strong>${idx + 1}. ${item.title}</strong><br>
            <div style="margin:8px 0;">${item.summary.split('\\n').map(p => `<p style="margin: 0 0 10px 0;">${p}</p>`).join('')}</div>
            <div>Links:<br>
              ${item.links.map(link => `<a href="${link}" target="_blank">${link}</a>`).join('<br>')}
            </div>
          </div>
        `).join('');
      }
    </script>
  </body>
</html>
    """


@app.get("/search")
async def search(
    query: str = "Search something...", links_to_scrap: int = 10, summaries: int = 5
):
    clean_expired_cache()
    results = await advanced_search_async(query, links_to_scrap, summaries)
    minimal_results = [
        {
            "title": item["title"] or item["url"],
            "summary": item["summary"],
            "links": item.get("relevant_links", []) or [item["url"]],
        }
        for item in results
    ]
    return JSONResponse(content=minimal_results)

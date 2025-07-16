# 🎬 Fast API Video Scraper

A high-performance API for searching, extracting, and semantically ranking videos from the web. Built with dynamic scraping and advanced natural language processing, this project enables smart video content discovery and integration in modern applications.

---

## 🚀 Features

- **Automated video search** across multiple web sources with rich metadata extraction.
- **Semantic expansion** of queries using state-of-the-art multimodal embedding models.
- **Intelligent ranking** of video results according to semantic similarity to queries.
- **Metadata extraction:** duration, resolution, tags, language, and more.
- **Automatic translation** for multilingual search results.
- **RESTful API** designed for effortless integration.
- **Deduplication and domain filtering** to avoid common video platforms and duplicates.

---

## 🧬 Technologies

- [FastAPI](https://fastapi.tiangolo.com/) for asynchronous API backend.
- [Playwright](https://playwright.dev/python/) for robust, anti-detection dynamic web scraping.
- **OpenCLIP** models for multimodal semantic embeddings.
- **BoilerPy3**, **Readability**, **Trafilatura** for clean text extraction.
- [ffmpeg/ffprobe](https://ffmpeg.org/) for video analysis and metadata extraction.
- [aiohttp](https://docs.aiohttp.org/) for asynchronous HTTP requests.

---

## ⚡ Installation

First, clone the repository and set up your Python environment:

```
git clone https://github.com/yourusername/video-semantic-search.git
cd video-semantic-search
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Install Playwright and its browser dependencies:

```
playwright install
```

Make sure `ffmpeg` and `ffprobe` are available in your system path.

---

## 🐳 Docker & Automated Startup

All services can be launched and health-checked automatically using `launch.py`, found at the root of this repository.

### 1. Requirements

- **Docker** and **Docker Compose** must be installed.
- Python >= 3.9

### 2. Bootstrapping

To start all project dependencies and open the web interface, simply run:

```
python launch.py
```

- This script checks for Docker and Compose, launches the required containers, waits for the FastAPI backend to become available, and then opens the browser at http://localhost:8000.
- If any service fails to start, the script will output a clear error and terminate.

---

## 📝 Usage

1. Ensure all dependencies (system and Python) are installed and Docker is running.
2. Boot the stack using `python launch.py`, or manually start services as needed.
3. Access the API via:

    ```
    GET /search?query=surfing+4k&power_scraping=false
    ```

4. Example API response:

    ```
    [
      {
        "url": "https://example.com/video123",
        "title": "Extreme Surfing in 4K",
        "description": "A breathtaking 4K compilation of top surf scenes.",
        "videos": [
          {
            "url": "https://cdn.example.com/videos/surf4k.mp4",
            "title": "Extreme Surfing in 4K",
            "tags": ["surf", "4k", "extreme sports"],
            "duration": "125.0",
            "is_stream": false
          }
        ]
      }
    ]
    ```

---

## ⚙️ Configuration

- Main FastAPI server is started by default on `localhost:8000`.
- The project’s semantic models and tools are self-contained; no cloud API keys required.
- Default video search excludes sources like YouTube, Twitter, and Facebook for compliance and quality.

---

## 💡 Notes

- Results may vary according to page structure, layout, and language.
- The more CPU and RAM available, the higher the parallelism and performance.
- Minimum recommended: 8 CPU cores and 16 GB RAM for large scraping jobs.
- The included `launch.py` script is customized for this project to offer seamless service orchestration and a friendly CLI boot experience.

---

## 🏷️ License

MIT © Kimura Dev

---

> This project was engineered following best practices, delivering efficient natural language-based video search and multimodal analysis. Contributions are welcome!
```

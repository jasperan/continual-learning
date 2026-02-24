import json
import re
from pathlib import Path
from bs4 import BeautifulSoup
import requests


ORACLE_DOC_SOURCES = {
    "oci_overview": "https://docs.oracle.com/en-us/iaas/Content/GSG/Concepts/baremetalintro.htm",
    "oracle_26ai": "https://docs.oracle.com/en/database/oracle/oracle-database/26/",
    "free_container": "https://docs.oracle.com/en-us/iaas/Content/FreeTier/freetier_topic-Always_Free_Resources.htm",
    "ai_vector_search": "https://docs.oracle.com/en/database/oracle/oracle-database/23/vecse/",
}


def parse_html_to_text(html: str) -> str:
    """Extract clean text from HTML content."""
    soup = BeautifulSoup(html, "html.parser")
    for element in soup(["script", "style", "nav", "header", "footer"]):
        element.decompose()
    text = soup.get_text(separator=" ", strip=True)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def chunk_text(text: str, max_tokens: int = 512) -> list[str]:
    """Split text into chunks, respecting sentence boundaries.
    Uses approximate word count as token proxy (1 token ~ 0.75 words).
    """
    max_words = int(max_tokens * 0.75)
    sentences = re.split(r"(?<=[.!?])\s+", text)
    chunks = []
    current_chunk = []
    current_words = 0

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        word_count = len(sentence.split())

        # If a single sentence exceeds max_words, force-split it by word count
        if word_count > max_words:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_words = 0
            words = sentence.split()
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i : i + max_words]))
            continue

        if current_words + word_count > max_words and current_chunk:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_words = word_count
        else:
            current_chunk.append(sentence)
            current_words += word_count

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return [c for c in chunks if c.strip()]


def fetch_and_process(url: str, max_tokens: int = 512) -> list[str]:
    """Fetch a URL, parse HTML, and return text chunks."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    text = parse_html_to_text(response.text)
    return chunk_text(text, max_tokens=max_tokens)


def save_chunks_as_jsonl(chunks: list[str], output_path: str, source: str) -> None:
    """Save text chunks as JSONL file."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for i, chunk in enumerate(chunks):
            record = {"text": chunk, "source": source, "chunk_id": i}
            f.write(json.dumps(record) + "\n")


def load_chunks_from_jsonl(path: str) -> list[dict]:
    """Load text chunks from a JSONL file."""
    items = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

from __future__ import annotations

import re
from pathlib import Path

from bs4 import BeautifulSoup
from pypdf import PdfReader


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_text(path: Path, filename: str, content_type: str) -> tuple[str, dict]:
    suffix = path.suffix.lower()
    metadata = {"filename": filename, "content_type": content_type, "source_type": suffix.lstrip(".")}

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for index, page in enumerate(reader.pages, start=1):
            page_text = page.extract_text() or ""
            if page_text.strip():
                pages.append(f"\n\n[Page {index}]\n{page_text}")
        metadata["page_count"] = len(reader.pages)
        return clean_text("".join(pages)), metadata

    raw = path.read_text(encoding="utf-8", errors="ignore")
    if suffix in {".html", ".htm"} or "html" in content_type:
        soup = BeautifulSoup(raw, "html.parser")
        for element in soup(["script", "style", "noscript"]):
            element.decompose()
        raw = soup.get_text("\n")

    return clean_text(raw), metadata

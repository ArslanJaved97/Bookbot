import re
from pathlib import Path
import pymupdf as fitz  # PyMuPDF


def clean_text(text):
    text = re.sub(r"-\n(?=[a-z])", "", text)               # de-hyphenate
    text = re.sub(r"(?<![.!?:;])\n(?=[a-z])", " ", text)   # join wrapped lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pages(pdf_path):
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = clean_text(page.get_text("text"))
        if text:
            pages.append((i, text))
    doc.close()
    return pages


def chunk_pages(pages, target_words=220, overlap_words=40):
    chunks = []
    for page_num, text in pages:
        paragraphs = re.split(r"\n\s*\n", text)
        buf, buf_words = [], 0
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            w = len(para.split())
            if buf_words + w <= target_words:
                buf.append(para)
                buf_words += w
                continue
            if buf:
                chunks.append({"page": page_num, "text": "\n\n".join(buf)})
                tail = " ".join(buf[-1].split()[-overlap_words:])
                buf = [tail] if tail else []
                buf_words = len(tail.split())
            buf.append(para)
            buf_words += w
        if buf:
            chunks.append({"page": page_num, "text": "\n\n".join(buf)})
    return chunks
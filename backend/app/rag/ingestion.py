"""
Knowledge base ingestion pipeline.

Loads documents from knowledge_base/raw/, splits them into chunks,
and builds/updates the vector store index.

Run directly:
    python -m app.rag.ingestion
"""

from __future__ import annotations

import glob
import logging
import os

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app.config import get_settings
from app.rag.embeddings import get_embeddings
from app.rag.vector_store import FAISSVectorStore

logger = logging.getLogger(__name__)


def load_markdown_files(directory: str) -> list[Document]:
    """Load all .md files from a directory."""
    docs = []
    md_files = glob.glob(os.path.join(directory, "*.md"))

    for filepath in md_files:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                content = f.read()

            if not content.strip():
                continue

            docs.append(
                Document(
                    page_content=content,
                    metadata={
                        "source": os.path.basename(filepath),
                        "source_path": filepath,
                        "type": "markdown",
                    },
                )
            )
            logger.info("Loaded: %s (%d chars)", filepath, len(content))
        except Exception as e:
            logger.warning("Failed to load %s: %s", filepath, e)

    return docs


def load_pdf_files(directory: str) -> list[Document]:
    """
    Load all .pdf files from a directory.

    Uses PyMuPDF (fitz) for ultra-fast C-speed text extraction (sub-second for 300+ pages),
    with pdfplumber / pypdf as fallback options.
    """
    docs = []
    pdf_files = glob.glob(os.path.join(directory, "*.pdf"))

    for filepath in pdf_files:
        filename = os.path.basename(filepath)
        try:
            docs.extend(_load_pdf_fitz(filepath, filename))
        except Exception as fitz_err:
            logger.info("PyMuPDF fitz failed (%s), trying pdfplumber fallback", fitz_err)
            try:
                docs.extend(_load_pdf_pdfplumber(filepath, filename))
            except Exception as e:
                logger.warning("Failed to load PDF %s: %s", filepath, e)

    return docs


def _load_pdf_fitz(filepath: str, filename: str) -> list[Document]:
    """Extract text at maximum speed using PyMuPDF / fitz C-bindings."""
    import pymupdf

    docs = []
    doc = pymupdf.open(filepath)
    total_pages = len(doc)
    logger.info("Loading PDF (PyMuPDF fast mode): %s (%d pages)", filename, total_pages)

    for page_num in range(total_pages):
        page = doc[page_num]
        text = page.get_text("text")
        if text and text.strip():
            docs.append(
                Document(
                    page_content=text.strip(),
                    metadata={
                        "source": filename,
                        "source_path": filepath,
                        "page": page_num + 1,
                        "type": "pdf",
                    },
                )
            )

        if (page_num + 1) % 100 == 0 or page_num + 1 == total_pages:
            logger.info("  → %s: %d/%d pages extracted", filename, page_num + 1, total_pages)

    doc.close()
    logger.info("Loaded PDF (PyMuPDF): %s → %d text pages extracted", filename, len(docs))
    return docs


def _load_pdf_pdfplumber(filepath: str, filename: str) -> list[Document]:
    """Extract text + tables from PDF using pdfplumber."""
    import pdfplumber

    docs = []
    with pdfplumber.open(filepath) as pdf:
        total_pages = len(pdf.pages)
        logger.info("Loading PDF (pdfplumber): %s (%d pages)", filename, total_pages)

        for page_num, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                docs.append(
                    Document(
                        page_content=text.strip(),
                        metadata={
                            "source": filename,
                            "source_path": filepath,
                            "page": page_num + 1,
                            "type": "pdf",
                        },
                    )
                )

            if (page_num + 1) % 50 == 0 or page_num + 1 == total_pages:
                logger.info("  → %s: %d/%d pages processed", filename, page_num + 1, total_pages)

    logger.info("Loaded PDF: %s → %d text pages extracted", filename, len(docs))
    return docs


def split_documents(
    docs: list[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> list[Document]:
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ", "\n### ", "\n\n", "\n", ". ", " "],
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    logger.info("Split %d documents into %d chunks", len(docs), len(chunks))
    return chunks


async def ingest_knowledge_base(
    raw_dir: str | None = None,
    index_path: str | None = None,
    batch_size: int = 100,
) -> int:
    """
    Low-memory ingestion pipeline: load → split → batch embed → persist.

    Processes embeddings in small batches (default 100 chunks) to keep RAM usage
    under 200MB even for thousands of pages.
    """
    import gc

    settings = get_settings()
    raw_dir = raw_dir or settings.knowledge_base_raw_dir
    index_path = index_path or settings.faiss_index_path

    logger.info("Starting knowledge base ingestion from: %s", raw_dir)

    if not os.path.exists(raw_dir):
        logger.warning("Knowledge base directory not found: %s", raw_dir)
        return 0

    # Load all document types
    docs = load_markdown_files(raw_dir) + load_pdf_files(raw_dir)

    if not docs:
        logger.warning("No documents found in %s", raw_dir)
        return 0

    logger.info("Loaded %d documents total", len(docs))

    # Split into chunks
    chunks = split_documents(
        docs,
        chunk_size=settings.rag_chunk_size,
        chunk_overlap=settings.rag_chunk_overlap,
    )

    # Initialize embeddings and vector store
    embeddings = get_embeddings(settings.embedding_model)
    store = FAISSVectorStore(index_path, embeddings)

    total_chunks = len(chunks)
    logger.info("Ingesting %d chunks in batches of %d...", total_chunks, batch_size)

    # Add documents in small batches to preserve memory
    for i in range(0, total_chunks, batch_size):
        batch = chunks[i : i + batch_size]
        await store.add_documents(batch)
        gc.collect()  # Release memory
        logger.info("  → Embedded & indexed %d/%d chunks", min(i + batch_size, total_chunks), total_chunks)

    # Persist to disk
    await store.persist()

    logger.info(
        "Ingestion complete: %d documents → %d chunks → FAISS index at %s",
        len(docs),
        total_chunks,
        index_path,
    )
    return total_chunks


# Allow running as a script: python -m app.rag.ingestion
if __name__ == "__main__":
    import asyncio

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )
    count = asyncio.run(ingest_knowledge_base())
    print(f"\n[OK] Ingested {count} chunks into FAISS index.")

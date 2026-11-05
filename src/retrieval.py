"""
RAG retrieval: loads the notes in rag_docs/, stores them in Chroma, and
returns the most relevant chunks for a question.

Each note in rag_docs/ is a .md file that looks like this:

    # Title of the note
    Source: https://link-to-where-you-learned-it

    First paragraph in your own words...

    Second paragraph...

retrieve(query, k=3) -> list of {"text", "title", "source"}
"""
import re
from pathlib import Path

import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "rag_docs"
COLLECTION = "xray_notes"      # Chroma needs at least 3 characters here
MAX_CHARS = 600                # about one or two paragraphs per chunk

_collection = None


def _parse_note(path):
    """Read one note. Returns (title, source, body)."""
    text = path.read_text(encoding="utf-8")
    title_match = re.search(r"^#\s+(.+)$", text, flags=re.M)
    source_match = re.search(r"^Source:\s*(.+)$", text, flags=re.M | re.I)
    title = title_match.group(1).strip() if title_match else path.stem
    source = source_match.group(1).strip() if source_match else ""
    # Body = everything except the title line and the Source line
    body = re.sub(r"^#\s+.+$", "", text, count=1, flags=re.M)
    body = re.sub(r"^Source:.*$", "", body, flags=re.M | re.I)
    return title, source, body.strip()


def _chunk(body):
    """Split on blank lines, then join paragraphs until a chunk is ~MAX_CHARS."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", body) if p.strip()]
    chunks, current = [], ""
    for p in paragraphs:
        if current and len(current) + len(p) > MAX_CHARS:
            chunks.append(current)
            current = p
        else:
            current = f"{current}\n\n{p}" if current else p
    if current:
        chunks.append(current)
    return chunks


def build_index(embedding_function=None):
    """Read all notes and load them into an in-memory Chroma collection.

    With embedding_function=None, Chroma uses its default model
    (all-MiniLM-L6-v2, ~80 MB, downloaded automatically the first time).
    """
    global _collection
    notes = sorted(p for p in DOCS_DIR.glob("*.md")
                   if not p.name.startswith("_"))   # skip _TEMPLATE.md etc.
    if not notes:
        raise FileNotFoundError(f"No .md notes found in {DOCS_DIR}")

    ids, docs, metas = [], [], []
    for path in notes:
        title, source, body = _parse_note(path)
        for i, chunk in enumerate(_chunk(body)):
            ids.append(f"{path.stem}-{i}")
            docs.append(chunk)
            metas.append({"title": title, "source": source, "file": path.name})

    client = chromadb.EphemeralClient()   # in memory: rebuilt at every start (fast for a few notes)
    kwargs = {"embedding_function": embedding_function} if embedding_function else {}
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    _collection = client.create_collection(COLLECTION, **kwargs)
    _collection.add(ids=ids, documents=docs, metadatas=metas)
    return len(notes), len(ids)


def retrieve(query, k=3):
    if _collection is None:
        build_index()
    k = min(k, _collection.count())
    res = _collection.query(query_texts=[query], n_results=k)
    return [
        {"text": doc, "title": meta["title"], "source": meta["source"]}
        for doc, meta in zip(res["documents"][0], res["metadatas"][0])
    ]


if __name__ == "__main__":
    n_notes, n_chunks = build_index()
    print(f"Indexed {n_notes} notes as {n_chunks} chunks.\n")
    for q in ["pneumonia signs on a chest X-ray in children",
              "what does a normal chest X-ray look like",
              "why can the heatmap focus on letters"]:
        print("QUERY:", q)
        for r in retrieve(q):
            print(f"  - [{r['title']}] {r['text'][:90]!r}...")
        print()

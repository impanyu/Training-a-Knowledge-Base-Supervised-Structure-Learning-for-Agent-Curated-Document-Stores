from ca.retrieval import ChromaBackend, KeywordBackend

DOCS = [
    {"title": "Paris", "text": "Paris is the capital and largest city of France."},
    {"title": "Tokyo", "text": "Tokyo is the capital of Japan."},
    {"title": "Berlin", "text": "Berlin is the capital of Germany."},
]


def test_keyword_backend_ranks_relevant_doc_first():
    r = KeywordBackend(DOCS)
    hits = r.search("capital of France", k=2)
    assert hits[0]["title"] == "Paris"
    assert len(hits) == 2
    assert len(r.search("capital", k=10)) == 3


def test_chroma_backend_semantic_search(tmp_path):
    # first run downloads the default embedding model (~80MB); allow network
    r = ChromaBackend(DOCS, persist_dir=str(tmp_path / "idx"))
    hits = r.search("what city is the French capital?", k=1)
    assert hits[0]["title"] == "Paris"
    # reload from disk without docs
    r2 = ChromaBackend.load(str(tmp_path / "idx"))
    assert r2.search("Japanese capital", k=1)[0]["title"] == "Tokyo"

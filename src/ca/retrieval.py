"""Pluggable retrieval. Default: Chroma vector DB (built-in all-MiniLM
embedding). KeywordBackend is a dependency-free stub for unit tests."""
import chromadb


class KeywordBackend:
    def __init__(self, docs: list[dict]):
        self.docs = docs

    def search(self, query: str, k: int = 5) -> list[dict]:
        q = set(query.lower().split())
        ranked = sorted(self.docs,
                        key=lambda d: -len(q & set(d["text"].lower().split())))
        return ranked[:k]


class ChromaBackend:
    COLLECTION = "corpus"

    def __init__(self, docs: list[dict] | None = None, persist_dir: str | None = None):
        self._client = (chromadb.PersistentClient(path=persist_dir)
                        if persist_dir else chromadb.EphemeralClient())
        self.col = self._client.get_or_create_collection(self.COLLECTION)
        if docs and self.col.count() == 0:
            for i in range(0, len(docs), 1000):  # stay under chroma batch limits
                batch = docs[i:i + 1000]
                self.col.add(
                    ids=[str(i + j) for j in range(len(batch))],
                    documents=[d["text"] for d in batch],
                    metadatas=[{"title": d["title"]} for d in batch],
                )

    def search(self, query: str, k: int = 5) -> list[dict]:
        n = min(k, self.col.count())
        if n == 0:
            return []
        res = self.col.query(query_texts=[query], n_results=n)
        return [{"title": m["title"], "text": doc}
                for doc, m in zip(res["documents"][0], res["metadatas"][0])]

    @classmethod
    def load(cls, persist_dir: str) -> "ChromaBackend":
        return cls(docs=None, persist_dir=persist_dir)

import chromadb
from chromadb.utils import embedding_functions
from pathlib import Path
from typing import List, Dict
from rich.console import Console

console = Console()


class VectorStore:
    def __init__(self, persist_path: Path, embedding_model: str = "all-MiniLM-L6-v2"):
        self.client = chromadb.PersistentClient(path=str(persist_path))
        self.ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=embedding_model
        )
        self.collection = self.client.get_or_create_collection(
            name="darwin_knowledge",
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )

    def add_papers(self, papers: List[Dict]) -> None:
        if not papers:
            return

        documents, metadatas, ids = [], [], []
        for p in papers:
            documents.append(f"Title: {p['title']}\n\nAbstract: {p['abstract']}")
            metadatas.append({
                "title": p["title"],
                "published": p["published"],
                "url": p.get("url", ""),
                "authors": ", ".join(p.get("authors", [])),
            })
            ids.append(p["id"])

        # Batch add, skip existing
        batch_size = 50
        added = 0
        for i in range(0, len(documents), batch_size):
            b_docs = documents[i:i + batch_size]
            b_meta = metadatas[i:i + batch_size]
            b_ids = ids[i:i + batch_size]

            try:
                existing = set(self.collection.get(ids=b_ids)["ids"])
            except Exception:
                existing = set()

            new_docs = [d for d, id_ in zip(b_docs, b_ids) if id_ not in existing]
            new_meta = [m for m, id_ in zip(b_meta, b_ids) if id_ not in existing]
            new_ids = [id_ for id_ in b_ids if id_ not in existing]

            if new_docs:
                self.collection.add(documents=new_docs, metadatas=new_meta, ids=new_ids)
                added += len(new_docs)

        console.print(f"[green]✓ Added {added} new papers. Total: {self.collection.count()}[/green]")

    def search(self, query: str, n_results: int = 5) -> List[Dict]:
        count = self.collection.count()
        if count == 0:
            return []
        n = min(n_results, count)
        results = self.collection.query(query_texts=[query], n_results=n)
        output = []
        for i, doc in enumerate(results["documents"][0]):
            output.append({
                "content": doc,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            })
        return output

    def count(self) -> int:
        return self.collection.count()

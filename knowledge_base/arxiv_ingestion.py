import arxiv
import json
import time
from pathlib import Path
from typing import List, Dict
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


class ArXivIngestion:
    GPU_QUERIES = [
        "GPU architecture memory bandwidth optimization",
        "parallel processor design semiconductor novel",
        "compute memory hierarchy GPU performance",
        "VLSI chip low power high throughput architecture",
        "neuromorphic computing processor design",
        "near memory processing in-memory compute",
        "photonic chip computing optical interconnect",
        "GPU memory bandwidth bottleneck solution",
        "shader processor unified architecture design",
        "tensor processing unit accelerator architecture",
    ]

    def __init__(self, save_path: Path):
        self.save_path = save_path
        self.save_path.mkdir(parents=True, exist_ok=True)

    def fetch_papers(self, query: str, max_results: int = 20) -> List[Dict]:
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        papers = []
        for result in search.results():
            papers.append({
                "id": result.entry_id.split("/")[-1],
                "title": result.title.strip(),
                "abstract": result.summary.strip(),
                "authors": [str(a) for a in result.authors[:5]],
                "published": str(result.published.date()),
                "categories": result.categories,
                "url": result.entry_id,
            })
        return papers

    def ingest_all(self, max_per_query: int = 20) -> List[Dict]:
        all_papers = []
        seen_ids = set()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("Fetching papers...", total=len(self.GPU_QUERIES))

            for query in self.GPU_QUERIES:
                progress.update(task, description=f"[cyan]Fetching: {query[:50]}")
                try:
                    papers = self.fetch_papers(query, max_per_query)
                    for p in papers:
                        if p["id"] not in seen_ids:
                            seen_ids.add(p["id"])
                            all_papers.append(p)
                    time.sleep(3)  # ArXiv rate limit
                except Exception as e:
                    console.print(f"[red]Error: {e}")
                progress.advance(task)

        output = self.save_path / "papers.json"
        with open(output, "w") as f:
            json.dump(all_papers, f, indent=2)

        console.print(f"[green]✓ Ingested {len(all_papers)} unique papers[/green]")
        return all_papers

    def load_cached(self) -> List[Dict]:
        cache = self.save_path / "papers.json"
        if cache.exists():
            with open(cache) as f:
                data = json.load(f)
            console.print(f"[green]✓ Loaded {len(data)} cached papers[/green]")
            return data
        return []

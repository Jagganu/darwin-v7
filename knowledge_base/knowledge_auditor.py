import json
import requests
from typing import List, Dict
from rich.console import Console
from .vector_store import VectorStore

console = Console()

CONTRADICTION_PROMPT = """You are a scientific auditor for GPU/CPU architecture research.

Paper A:
{paper_a}

Paper B:
{paper_b}

Do these papers make conflicting technical claims about the same topic?

Respond ONLY in JSON:
{{
  "conflict_detected": true or false,
  "conflict_type": "benchmark_contradiction | claim_contradiction | assumption_conflict | none",
  "description": "one sentence describing the conflict",
  "research_target": "what question this conflict opens up for investigation",
  "severity": "high | medium | low"
}}"""

FRONTIER_TOPICS = [
    "photonic computing GPU optical",
    "analog AI compute crossbar",
    "asynchronous clockless processor",
    "quantum classical hybrid GPU",
    "neuromorphic GPU spike-based",
    "DNA computing architecture",
    "processing in memory compute",
    "stochastic computing architecture",
]


class KnowledgeAuditor:
    def __init__(self, vector_store: VectorStore, ollama_url: str, model: str):
        self.vs = vector_store
        self.ollama_url = ollama_url
        self.model = model

    def _call_llm(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.ollama_url}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False, "format": "json"},
            timeout=90,
        )
        return resp.json()["response"]

    def find_contradictions(self, max_pairs: int = 20) -> List[Dict]:
        conflicts = []
        checked = 0

        audit_topics = [
            "HBM memory bandwidth scaling performance",
            "GPU power consumption efficiency",
            "cache hierarchy latency performance",
            "parallel thread execution overhead",
            "memory compute bottleneck GPU",
        ]

        for topic in audit_topics:
            related = self.vs.search(topic, n_results=4)
            for i in range(len(related)):
                for j in range(i + 1, len(related)):
                    if checked >= max_pairs:
                        return conflicts

                    a = related[i]["content"][:600]
                    b = related[j]["content"][:600]
                    prompt = CONTRADICTION_PROMPT.format(paper_a=a, paper_b=b)

                    try:
                        result = json.loads(self._call_llm(prompt))
                        if result.get("conflict_detected"):
                            conflict = {
                                "paper_a": related[i]["metadata"].get("title", ""),
                                "paper_b": related[j]["metadata"].get("title", ""),
                                "conflict_type": result["conflict_type"],
                                "description": result["description"],
                                "research_target": result["research_target"],
                                "severity": result["severity"],
                            }
                            conflicts.append(conflict)
                            console.print(f"[yellow]⚡ Conflict: {result['description'][:80]}[/yellow]")
                    except Exception:
                        pass

                    checked += 1

        return conflicts

    def find_knowledge_gaps(self) -> List[str]:
        gaps = []
        for topic in FRONTIER_TOPICS:
            results = self.vs.search(topic, n_results=3)
            if not results:
                gaps.append(f"No coverage: {topic}")
                continue
            avg_dist = sum(r["distance"] for r in results) / len(results)
            if avg_dist > 0.65:
                gaps.append(f"Sparse: {topic} (distance={avg_dist:.2f})")
        return gaps

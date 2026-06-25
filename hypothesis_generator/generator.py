import json
import requests
from typing import List, Dict, Optional
from rich.console import Console

console = Console()

HYPOTHESIS_PROMPT = """You are DARWIN, a GPU/CPU architecture research AI.

Knowledge Context (recent papers):
{context}

Known Knowledge Gaps:
{gaps}

Detected Conflicts in Literature:
{conflicts}

Past Failure Lessons:
{lessons}

Generate ONE novel GPU/CPU architecture hypothesis. Rules:
1. Break exactly ONE existing assumption about GPU/CPU design
2. Must be physically plausible
3. Must make at least one quantified, testable prediction
4. Must NOT already exist in standard literature

Respond ONLY in JSON:
{{
  "hypothesis": "clear one-paragraph statement of the theory",
  "broken_assumption": "which accepted assumption this breaks and why",
  "mechanism": "how it would work physically at the hardware level",
  "prediction": {{
    "claim": "specific quantified prediction (e.g. 40% bandwidth increase)",
    "test_condition": "exact conditions to test this",
    "falsification": "what result would definitively disprove this"
  }},
  "inspired_by": "what in the literature inspired this",
  "estimated_impact": "low | medium | high | revolutionary",
  "known_unknowns": ["thing we don't know yet 1", "thing 2"]
}}"""


class HypothesisGenerator:
    def __init__(self, ollama_url: str, model: str):
        self.ollama_url = ollama_url
        self.model = model

    def _call_llm(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.9, "top_p": 0.95},
            },
            timeout=180,
        )
        return resp.json()["response"]

    def generate(
        self,
        context_papers: List[Dict],
        gaps: List[str],
        conflicts: List[Dict],
        lessons: List[Dict],
    ) -> Optional[Dict]:
        context = "\n---\n".join([p["content"][:300] for p in context_papers[:4]])
        gaps_text = "\n".join(f"- {g}" for g in gaps[:5]) or "None identified yet"
        conflicts_text = "\n".join(f"- {c['description']}" for c in conflicts[:3]) or "None yet"
        lessons_text = "\n".join(f"- {l['reasons'][0]}" for l in lessons[:3] if l.get("reasons")) or "None yet"

        prompt = HYPOTHESIS_PROMPT.format(
            context=context,
            gaps=gaps_text,
            conflicts=conflicts_text,
            lessons=lessons_text,
        )

        try:
            raw = self._call_llm(prompt)
            result = json.loads(raw)
            hyp_hash = abs(hash(result.get("hypothesis", ""))) % 100000
            result["id"] = f"H{hyp_hash:05d}"
            return result
        except Exception as e:
            console.print(f"[red]Generator error: {e}[/red]")
            return None

    def generate_batch(
        self,
        n: int,
        context_papers: List[Dict],
        gaps: List[str],
        conflicts: List[Dict],
        lessons: List[Dict],
    ) -> List[Dict]:
        results = []
        for i in range(n):
            console.print(f"[cyan]  Generating hypothesis {i + 1}/{n}...[/cyan]")
            h = self.generate(context_papers, gaps, conflicts, lessons)
            if h:
                results.append(h)
        return results

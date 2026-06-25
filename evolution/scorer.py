import json
import requests
from typing import Dict, List
from rich.console import Console

console = Console()

SCORER_PROMPT = """You are the Discovery Scorer in DARWIN, a GPU/CPU architecture research AI.

Score this theory on 5 dimensions for GPU/CPU architecture research value.

Theory:
  Hypothesis       : {hypothesis}
  Broken Assumption: {broken_assumption}
  Mechanism        : {mechanism}
  Primary Prediction: {prediction}
  Simulation Result: {sim_result}
  Debate avg score : {debate_score:.2f}

Score each dimension 0-100 where 100 = best possible:

- novelty          : How new is this? (0 = already exists, 100 = never been done)
- performance_gain : How much does it improve compute/memory performance?
- manufacturability: How feasible is it to actually fabricate?
- cost_reduction   : Does it reduce manufacturing or operating cost?
- scientific_insight: How much does it teach us about GPU/CPU fundamentals?

Weights: novelty=25%, performance=25%, manufacturability=20%, cost=15%, insight=15%

Respond ONLY in JSON:
{{
  "novelty": 0-100,
  "performance_gain": 0-100,
  "manufacturability": 0-100,
  "cost_reduction": 0-100,
  "scientific_insight": 0-100,
  "final_score": 0-100,
  "verdict": "BREAKTHROUGH | HIGH_VALUE | MODERATE | LOW_VALUE | INCREMENTAL",
  "strongest_dimension": "which dimension scores highest",
  "weakest_dimension": "which dimension scores lowest",
  "evolution_direction": "how the evolution engine should mutate this theory"
}}"""

WEIGHTS = {
    "novelty": 0.25,
    "performance_gain": 0.25,
    "manufacturability": 0.20,
    "cost_reduction": 0.15,
    "scientific_insight": 0.15,
}


class DiscoveryScorer:
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
                "options": {"temperature": 0.3},
            },
            timeout=120,
        )
        return resp.json()["response"]

    def score(self, hypothesis: Dict) -> Dict:
        pred    = hypothesis.get("predictions", {}).get("primary_prediction", {})
        sim     = hypothesis.get("simulation", {})
        debate  = hypothesis.get("debate", {})

        prompt = SCORER_PROMPT.format(
            hypothesis=hypothesis.get("hypothesis", ""),
            broken_assumption=hypothesis.get("broken_assumption", ""),
            mechanism=hypothesis.get("mechanism", ""),
            prediction=pred.get("claim", ""),
            sim_result=sim.get("estimated_performance_delta", "not simulated"),
            debate_score=debate.get("avg_score", 0.5),
        )

        try:
            result = json.loads(self._call_llm(prompt))
        except Exception as e:
            console.print(f"[red]Scorer error: {e}[/red]")
            result = {
                "novelty": 50, "performance_gain": 50, "manufacturability": 50,
                "cost_reduction": 50, "scientific_insight": 50,
                "final_score": 50, "verdict": "MODERATE",
                "strongest_dimension": "unknown", "weakest_dimension": "unknown",
                "evolution_direction": "general improvement",
            }

        # Recalculate final score with correct weights
        result["final_score"] = int(sum(
            result.get(dim, 50) * w for dim, w in WEIGHTS.items()
        ))

        result["theory_id"] = hypothesis.get("id")
        return result

    def score_batch(self, hypotheses: List[Dict]) -> List[Dict]:
        console.print(f"\n[bold]═══ Discovery Scorer ═══[/bold]")
        scored = []

        for h in hypotheses:
            s = self.score(h)
            h["discovery_score"] = s
            verdict = s.get("verdict", "")
            color = {
                "BREAKTHROUGH": "bold green",
                "HIGH_VALUE":   "green",
                "MODERATE":     "yellow",
                "LOW_VALUE":    "dim",
                "INCREMENTAL":  "dim red",
            }.get(verdict, "white")

            console.print(
                f"  {h.get('id','?')} "
                f"[{color}]{verdict}[/{color}] "
                f"score=[yellow]{s['final_score']}[/yellow]  "
                f"N={s.get('novelty',0)} P={s.get('performance_gain',0)} "
                f"M={s.get('manufacturability',0)}"
            )
            scored.append(h)

        scored.sort(key=lambda x: x.get("discovery_score", {}).get("final_score", 0), reverse=True)
        return scored

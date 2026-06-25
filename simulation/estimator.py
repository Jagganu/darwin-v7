import json
import requests
from typing import Dict, Optional
from rich.console import Console

console = Console()

ESTIMATE_PROMPT = """You are a GPU/CPU architecture performance estimator.

A theory has passed Red Team, Multi-Agent Debate, and Prediction Generation.
Estimate its likely simulation results WITHOUT running an actual simulator.
Use your knowledge of GPU/CPU architecture principles.

Theory:
  Hypothesis : {hypothesis}
  Mechanism  : {mechanism}
  Prediction : {prediction_claim}
  Baseline   : {baseline}

Estimate realistic simulation outcomes based on physics and known architecture trade-offs.
Be conservative — most novel ideas underperform initial predictions.

Respond ONLY in JSON:
{{
  "estimated_performance_delta": "e.g. +18% throughput",
  "estimated_power_delta": "e.g. +5% TDP",
  "estimated_area_delta": "e.g. +12% die area",
  "estimated_latency_delta": "e.g. -22% memory latency",
  "estimated_cost_delta": "e.g. +8% manufacturing cost",
  "confidence_in_estimate": 0.0 to 1.0,
  "key_bottleneck": "what limits this design most",
  "quick_score": 0 to 100,
  "worth_full_simulation": true or false,
  "reason": "one sentence explaining quick_score and recommendation"
}}"""


class FastEstimator:
    """
    LLM-based fast estimator. Runs in seconds.
    Filters top-N theories before sending to expensive gem5 simulation.
    """

    def __init__(self, ollama_url: str, model: str):
        self.ollama_url = ollama_url
        self.model = model

    def estimate(self, hypothesis: Dict) -> Optional[Dict]:
        pred = hypothesis.get("predictions", {}).get("primary_prediction", {})
        prompt = ESTIMATE_PROMPT.format(
            hypothesis=hypothesis.get("hypothesis", ""),
            mechanism=hypothesis.get("mechanism", ""),
            prediction_claim=pred.get("claim", ""),
            baseline=pred.get("baseline", "current GPU design"),
        )
        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.2},
                },
                timeout=120,
            )
            result = json.loads(resp.json()["response"])
            result["theory_id"] = hypothesis.get("id")
            return result
        except Exception as e:
            console.print(f"[red]Estimator error: {e}[/red]")
            return None

    def rank_and_filter(self, hypotheses: list, top_n: int = 5) -> tuple:
        """
        Resource Manager logic:
        1000 theories → fast estimate → top 50 → medium sim → top 5 → deep sim
        This method handles the first filter.
        """
        console.print(f"[cyan]Fast-estimating {len(hypotheses)} theories...[/cyan]")
        scored = []
        skipped = []

        for h in hypotheses:
            est = self.estimate(h)
            if est is None:
                skipped.append(h)
                continue

            score = est.get("quick_score", 0)
            worth = est.get("worth_full_simulation", False)
            h["fast_estimate"] = est

            console.print(
                f"  [dim]{h.get('id','?')}[/dim] "
                f"score=[yellow]{score}[/yellow] "
                f"{'[green]→ SIM[/green]' if worth else '[red]→ SKIP[/red]'} "
                f"— {est.get('reason', '')[:50]}"
            )

            if worth:
                scored.append((score, h))
            else:
                skipped.append(h)

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [h for _, h in scored[:top_n]]
        rest = [h for _, h in scored[top_n:]] + skipped

        console.print(
            f"[bold]Resource Manager:[/bold] "
            f"[green]{len(top)} → full simulation[/green] / "
            f"[dim]{len(rest)} skipped[/dim]"
        )
        return top, rest

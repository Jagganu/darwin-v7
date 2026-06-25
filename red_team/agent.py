import json
import requests
from typing import List, Dict, Tuple
from rich.console import Console

console = Console()

RED_TEAM_PROMPT = """You are the Red Team Agent in DARWIN. Your ONLY job is to DESTROY this hypothesis.
Be ruthless. Be specific. Be technical.

Hypothesis:
  Statement: {hypothesis}
  Broken Assumption: {broken_assumption}
  Mechanism: {mechanism}
  Prediction: {prediction}

Attack from every angle:
1. Does it violate thermodynamics, information theory, or semiconductor physics?
2. Does prior art already exist (similar patents or published designs)?
3. Are there hidden manufacturing constraints that make it impossible?
4. Does the prediction logically follow from the mechanism?
5. Are there scaling problems that appear only at production scale?

Respond ONLY in JSON:
{{
  "survives": true or false,
  "fatal_flaws": ["fatal flaw 1", "fatal flaw 2"],
  "survivable_weaknesses": ["weakness that can be fixed 1"],
  "prior_art": "description of any similar existing work, or null",
  "verdict": "REJECTED | WEAKENED | SURVIVES",
  "verdict_reason": "one sentence explaining verdict"
}}"""


class RedTeamAgent:
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
                "options": {"temperature": 0.2},  # low temp = critical thinking
            },
            timeout=120,
        )
        return resp.json()["response"]

    def attack(self, hypothesis: Dict) -> Dict:
        pred = hypothesis.get("prediction", {})
        prompt = RED_TEAM_PROMPT.format(
            hypothesis=hypothesis.get("hypothesis", ""),
            broken_assumption=hypothesis.get("broken_assumption", ""),
            mechanism=hypothesis.get("mechanism", ""),
            prediction=pred.get("claim", ""),
        )
        try:
            result = json.loads(self._call_llm(prompt))
            result["hypothesis_id"] = hypothesis.get("id")
            return result
        except Exception as e:
            console.print(f"[red]Red team LLM error: {e}[/red]")
            return {
                "survives": True,
                "verdict": "ERROR",
                "fatal_flaws": [],
                "survivable_weaknesses": [],
                "prior_art": None,
                "verdict_reason": "LLM error — passed by default",
            }

    def filter_batch(self, hypotheses: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        survivors, rejected = [], []

        for h in hypotheses:
            console.print(f"\n[red bold]🔴 Attacking:[/red bold] {h.get('hypothesis', '')[:65]}...")
            result = self.attack(h)

            if result["verdict"] == "REJECTED":
                console.print(f"  [red]✗ REJECTED — {result.get('verdict_reason', '')}[/red]")
                rejected.append({"hypothesis": h, "attack": result})
            else:
                console.print(f"  [green]✓ {result['verdict']} — {result.get('verdict_reason', '')}[/green]")
                h["red_team"] = result
                survivors.append(h)

        console.print(
            f"\n[bold]Red Team Results:[/bold] "
            f"[green]{len(survivors)} survived[/green] / "
            f"[red]{len(rejected)} rejected[/red]"
        )
        return survivors, rejected

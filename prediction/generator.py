import json
import sqlite3
import requests
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console

console = Console()

PREDICTION_PROMPT = """You are the Prediction Generator in DARWIN, a GPU/CPU research AI.

A theory has survived Red Team filtering and multi-agent debate.
Now you must force it to make PRECISE, TESTABLE, FALSIFIABLE predictions.

Theory:
  Statement : {hypothesis}
  Mechanism : {mechanism}
  Debate findings: {debate_findings}

Rules for predictions:
1. Every prediction must have a SPECIFIC NUMBER (%, speedup, watts, ns latency, etc)
2. Every prediction must have exact TEST CONDITIONS
3. Every prediction must have a clear FALSIFICATION CRITERION
4. No vague claims like "better performance" or "improved efficiency"
5. Predictions must be directly caused by the mechanism, not side effects

Respond ONLY in JSON:
{{
  "primary_prediction": {{
    "claim": "exact quantified claim e.g. 42% reduction in memory access latency",
    "metric": "the exact metric being measured",
    "expected_value": "numeric value with unit",
    "baseline": "what we are comparing against",
    "test_condition": "exact configuration to test e.g. 256-core GPU, 7nm node, 300W TDP",
    "falsification": "if result is below X the theory is disproved",
    "confidence": 0.0 to 1.0
  }},
  "secondary_predictions": [
    {{
      "claim": "second quantified prediction",
      "metric": "metric name",
      "expected_value": "value with unit",
      "test_condition": "conditions",
      "falsification": "falsification criterion"
    }}
  ],
  "negative_predictions": [
    "what this theory predicts will NOT happen (equally important)"
  ],
  "test_protocol": {{
    "simulator": "gem5 | GPGPU-Sim | McPAT | custom",
    "estimated_runtime": "minutes or hours",
    "required_inputs": ["input 1", "input 2"],
    "success_criteria": "how we know the test passed"
  }},
  "validity": "VALID | VAGUE | UNTESTABLE",
  "validity_reason": "why predictions are or are not testable"
}}"""

REWRITE_PROMPT = """This prediction is too vague to be scientifically useful:

Original prediction: {vague_prediction}
Theory: {hypothesis}

Rewrite it as a SPECIFIC, QUANTIFIED, FALSIFIABLE prediction.
Include: exact metric, numeric value, test conditions, falsification criterion.

Respond ONLY in JSON with the same structure as a primary_prediction:
{{
  "claim": "...",
  "metric": "...",
  "expected_value": "...",
  "baseline": "...",
  "test_condition": "...",
  "falsification": "...",
  "confidence": 0.0 to 1.0
}}"""


class PredictionGenerator:
    def __init__(self, ollama_url: str, model: str, db_path: Path):
        self.ollama_url = ollama_url
        self.model      = model
        self.db_path    = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    theory_id TEXT,
                    primary_prediction TEXT,
                    secondary_predictions TEXT,
                    negative_predictions TEXT,
                    test_protocol TEXT,
                    validity TEXT,
                    validity_reason TEXT,
                    simulation_result TEXT,
                    confirmed INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _call_llm(self, prompt: str, temp: float = 0.4) -> str:
        resp = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": temp},
            },
            timeout=180,
        )
        return resp.json()["response"]

    def _is_vague(self, prediction: Dict) -> bool:
        claim = prediction.get("claim", "").lower()
        vague_words = ["better", "improved", "faster", "more efficient", "higher", "lower", "significant"]
        has_number = any(c.isdigit() for c in claim)
        has_vague  = any(w in claim for w in vague_words) and not has_number
        return has_vague or not has_number

    def _strengthen_prediction(self, vague: Dict, hypothesis: str) -> Dict:
        console.print("  [yellow]⚠ Prediction vague — rewriting...[/yellow]")
        prompt = REWRITE_PROMPT.format(
            vague_prediction=vague.get("claim", ""),
            hypothesis=hypothesis,
        )
        try:
            return json.loads(self._call_llm(prompt))
        except Exception:
            return vague  # return original if rewrite fails

    def generate(self, hypothesis: Dict) -> Optional[Dict]:
        debate = hypothesis.get("debate", {})
        findings = debate.get("all_findings", [])[:4]
        findings_text = "\n".join(f"- {f}" for f in findings) or "No debate findings"

        prompt = PREDICTION_PROMPT.format(
            hypothesis=hypothesis.get("hypothesis", ""),
            mechanism=hypothesis.get("mechanism", ""),
            debate_findings=findings_text,
        )

        try:
            result = json.loads(self._call_llm(prompt))
        except Exception as e:
            console.print(f"[red]Prediction generator error: {e}[/red]")
            return None

        # Check validity
        if result.get("validity") == "UNTESTABLE":
            console.print(
                f"  [red]✗ UNTESTABLE — {result.get('validity_reason', '')}[/red]"
            )
            return None  # Theory dies here — no prediction = no theory

        # Strengthen vague primary prediction
        primary = result.get("primary_prediction", {})
        if self._is_vague(primary):
            result["primary_prediction"] = self._strengthen_prediction(
                primary, hypothesis.get("hypothesis", "")
            )

        # Save to DB
        tid = hypothesis.get("id", "unknown")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO predictions "
                "(theory_id, primary_prediction, secondary_predictions, "
                "negative_predictions, test_protocol, validity, validity_reason) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    tid,
                    json.dumps(result.get("primary_prediction", {})),
                    json.dumps(result.get("secondary_predictions", [])),
                    json.dumps(result.get("negative_predictions", [])),
                    json.dumps(result.get("test_protocol", {})),
                    result.get("validity", "VALID"),
                    result.get("validity_reason", ""),
                ),
            )
            conn.commit()

        # Display
        p = result["primary_prediction"]
        console.print(f"  [green]✓ Prediction:[/green] {p.get('claim', '')}")
        console.print(f"  [dim]  Falsified if: {p.get('falsification', '')}[/dim]")
        console.print(f"  [dim]  Simulator: {result.get('test_protocol', {}).get('simulator', 'unknown')}[/dim]")

        result["theory_id"] = tid
        return result

    def filter_batch(self, hypotheses: List[Dict]):
        survivors, no_prediction = [], []

        for h in hypotheses:
            console.print(
                f"\n[bold]🔬 Prediction: [cyan]{h.get('hypothesis', '')[:55]}...[/cyan][/bold]"
            )
            pred = self.generate(h)
            if pred:
                h["predictions"] = pred
                survivors.append(h)
            else:
                console.print("  [red]✗ No valid prediction — theory rejected[/red]")
                no_prediction.append(h)

        console.print(
            f"\n[bold]Prediction Filter:[/bold] "
            f"[green]{len(survivors)} valid[/green] / "
            f"[red]{len(no_prediction)} untestable (rejected)[/red]"
        )
        return survivors, no_prediction

    def get_predictions(self, theory_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT primary_prediction, secondary_predictions, test_protocol, validity "
                "FROM predictions WHERE theory_id = ? ORDER BY created_at DESC LIMIT 1",
                (theory_id,),
            ).fetchone()
        if row:
            return {
                "primary": json.loads(row[0]),
                "secondary": json.loads(row[1]),
                "test_protocol": json.loads(row[2]),
                "validity": row[3],
            }
        return None

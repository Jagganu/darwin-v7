import json
import sqlite3
import requests
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console

console = Console()

SURPRISE_ANALYSIS_PROMPT = """You are the Surprise Engine in DARWIN, a GPU/CPU research AI.

A simulation returned an UNEXPECTED result. Your job is to extract maximum
scientific value from this anomaly.

Theory:
  Hypothesis : {hypothesis}
  Prediction : {expected}

Actual simulation result:
  {actual}

Deviation: {deviation:.1f}% from expected

Analyze this surprise:
1. Why might the result deviate this much?
2. What does this anomaly reveal that we didn't know?
3. What new research questions does this open?
4. Could this be MORE interesting than the original theory?

Respond ONLY in JSON:
{{
  "surprise_type": "positive_surprise | negative_surprise | unexpected_tradeoff | new_phenomenon",
  "explanation": "why the result likely deviated",
  "new_insight": "what this anomaly reveals about GPU/CPU architecture",
  "new_questions": [
    "research question 1 opened by this anomaly",
    "research question 2"
  ],
  "spawn_new_branch": true or false,
  "new_hypothesis": "if spawn_new_branch: a new hypothesis inspired by the anomaly",
  "priority": "high | medium | low",
  "more_interesting_than_original": true or false
}}"""

SURPRISE_THRESHOLD = 0.20   # >20% deviation from prediction = surprise


class SurpriseEngine:
    """
    Monitors simulation results for anomalies.
    When result deviates significantly from prediction:
      → Flags as surprise
      → Generates new research questions
      → Optionally spawns new hypothesis branch
    """

    def __init__(self, ollama_url: str, model: str, db_path: Path):
        self.ollama_url = ollama_url
        self.model      = model
        self.db_path    = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS surprises (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    theory_id TEXT,
                    surprise_type TEXT,
                    explanation TEXT,
                    new_insight TEXT,
                    new_questions TEXT,
                    spawned_hypothesis TEXT,
                    priority TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _call_llm(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.7},
            },
            timeout=120,
        )
        return resp.json()["response"]

    def _calculate_deviation(self, sim: Dict, pred: Dict) -> float:
        """Returns deviation as a fraction (0.0 = perfect, 1.0 = completely wrong)."""
        hit = sim.get("prediction_hit", None)
        if hit is not None:
            return 1.0 - hit

        # Fallback: compare quick_score vs prediction confidence
        sim_score  = sim.get("quick_score", 50) / 100
        pred_conf  = pred.get("confidence", 0.5)
        return abs(sim_score - pred_conf)

    def check(self, hypothesis: Dict) -> Optional[Dict]:
        """Check a simulated hypothesis for surprises."""
        sim  = hypothesis.get("simulation", {})
        pred = hypothesis.get("predictions", {}).get("primary_prediction", {})
        tid  = hypothesis.get("id", "unknown")

        deviation = self._calculate_deviation(sim, pred)

        if deviation < SURPRISE_THRESHOLD:
            return None   # Result as expected — no surprise

        console.print(
            f"\n[bold yellow]⚡ SURPRISE DETECTED[/bold yellow] "
            f"theory={tid}  deviation={deviation:.1%}"
        )

        prompt = SURPRISE_ANALYSIS_PROMPT.format(
            hypothesis=hypothesis.get("hypothesis", ""),
            expected=pred.get("claim", ""),
            actual=json.dumps(sim, indent=2)[:600],
            deviation=deviation * 100,
        )

        try:
            result = json.loads(self._call_llm(prompt))
        except Exception as e:
            console.print(f"[red]Surprise engine LLM error: {e}[/red]")
            return None

        result["theory_id"]  = tid
        result["deviation"]  = deviation

        # Save to DB
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO surprises "
                "(theory_id, surprise_type, explanation, new_insight, "
                "new_questions, spawned_hypothesis, priority) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    tid,
                    result.get("surprise_type", "unknown"),
                    result.get("explanation", ""),
                    result.get("new_insight", ""),
                    json.dumps(result.get("new_questions", [])),
                    result.get("new_hypothesis", ""),
                    result.get("priority", "medium"),
                ),
            )
            conn.commit()

        stype = result.get("surprise_type", "unknown")
        color = "green" if "positive" in stype else "yellow"
        console.print(f"  [{color}]Type: {stype}[/{color}]")
        console.print(f"  Insight: {result.get('new_insight', '')[:80]}")

        if result.get("spawn_new_branch") and result.get("new_hypothesis"):
            console.print(
                f"  [bold green]→ New branch spawned:[/bold green] "
                f"{result['new_hypothesis'][:70]}..."
            )

        return result

    def scan_batch(self, hypotheses: List[Dict]) -> tuple:
        """
        Scan all simulated hypotheses for surprises.
        Returns (surprises, new_hypothesis_seeds)
        """
        console.print(f"\n[bold]═══ Surprise Engine ═══[/bold]")
        surprises, seeds = [], []

        for h in hypotheses:
            s = self.check(h)
            if s:
                surprises.append(s)
                if s.get("spawn_new_branch") and s.get("new_hypothesis"):
                    # Package as a seed for the next generation
                    seeds.append({
                        "hypothesis": s["new_hypothesis"],
                        "inspired_by_surprise": h.get("id"),
                        "new_questions": s.get("new_questions", []),
                    })

        if not surprises:
            console.print("  [dim]No surprises detected in this batch.[/dim]")
        else:
            console.print(
                f"  [yellow]{len(surprises)} surprise(s) found → "
                f"{len(seeds)} new branch(es) queued[/yellow]"
            )

        return surprises, seeds

    def get_all_surprises(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT theory_id, surprise_type, new_insight, priority, created_at "
                "FROM surprises ORDER BY created_at DESC"
            ).fetchall()
        return [
            {"theory_id": r[0], "type": r[1], "insight": r[2],
             "priority": r[3], "created_at": r[4]}
            for r in rows
        ]

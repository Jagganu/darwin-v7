import json
import sqlite3
import requests
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from rich.console import Console

console = Console()

MERGER_PROMPT = """You are the Theory Merger in DARWIN, a GPU/CPU architecture research AI.

Two theories individually scored MODERATE or below. But sometimes two
mediocre theories combine into one strong one.

Theory A (score={score_a}, weakest={weak_a}):
  Hypothesis : {hyp_a}
  Mechanism  : {mech_a}
  Prediction : {pred_a}
  Strength   : {strength_a}

Theory B (score={score_b}, weakest={weak_b}):
  Hypothesis : {hyp_b}
  Mechanism  : {mech_b}
  Prediction : {pred_b}
  Strength   : {strength_b}

Attempt to merge:
- Take Theory A's strongest element
- Take Theory B's strongest element
- Combine them into ONE coherent theory
- The merged theory must be BETTER than either parent

If these two theories CANNOT be meaningfully merged, say so.

Respond ONLY in JSON:
{{
  "mergeable": true or false,
  "reason_if_not": "why they cannot be merged",
  "merged_hypothesis": "the combined theory statement",
  "mechanism": "how the combined mechanism works",
  "from_a": "what was taken from Theory A",
  "from_b": "what was taken from Theory B",
  "improvement": "why the merged theory is stronger than either parent",
  "prediction": {{
    "claim": "specific quantified prediction",
    "test_condition": "exact conditions",
    "falsification": "what would disprove this"
  }},
  "estimated_score_improvement": "e.g. from avg 45 to merged 68",
  "known_unknowns": ["unknown 1", "unknown 2"]
}}"""


class TheoryMerger:
    """
    Separate from the Evolution Engine.
    Evolution breeds TOP theories.
    Theory Merger specifically targets MEDIOCRE theories (score 30-60)
    to see if combining them produces something valuable.

    Runs after Discovery Scorer, before Evolution Engine.
    """

    MEDIOCRE_THRESHOLD_LOW  = 30
    MEDIOCRE_THRESHOLD_HIGH = 62

    def __init__(self, ollama_url: str, model: str, db_path: Path):
        self.ollama_url = ollama_url
        self.model      = model
        self.db_path    = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS merges (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    theory_a_id TEXT,
                    theory_b_id TEXT,
                    merged_theory TEXT,
                    success INTEGER,
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
                "options": {"temperature": 0.75},
            },
            timeout=180,
        )
        return resp.json()["response"]

    def _get_strength(self, h: Dict) -> str:
        score = h.get("discovery_score", {})
        dims  = {
            "novelty":          score.get("novelty", 0),
            "performance_gain": score.get("performance_gain", 0),
            "manufacturability":score.get("manufacturability", 0),
            "cost_reduction":   score.get("cost_reduction", 0),
            "scientific_insight":score.get("scientific_insight", 0),
        }
        best = max(dims, key=dims.get)
        return f"{best} ({dims[best]})"

    def merge(self, a: Dict, b: Dict) -> Optional[Dict]:
        score_a = a.get("discovery_score", {}).get("final_score", 50)
        score_b = b.get("discovery_score", {}).get("final_score", 50)
        pred_a  = a.get("predictions", {}).get("primary_prediction", {})
        pred_b  = b.get("predictions", {}).get("primary_prediction", {})

        prompt = MERGER_PROMPT.format(
            hyp_a=a.get("hypothesis", ""),      mech_a=a.get("mechanism", ""),
            pred_a=pred_a.get("claim", ""),     score_a=score_a,
            weak_a=a.get("discovery_score", {}).get("weakest_dimension", ""),
            strength_a=self._get_strength(a),
            hyp_b=b.get("hypothesis", ""),      mech_b=b.get("mechanism", ""),
            pred_b=pred_b.get("claim", ""),     score_b=score_b,
            weak_b=b.get("discovery_score", {}).get("weakest_dimension", ""),
            strength_b=self._get_strength(b),
        )

        try:
            result = json.loads(self._call_llm(prompt))
        except Exception as e:
            console.print(f"[red]Merger error: {e}[/red]")
            return None

        if not result.get("mergeable"):
            console.print(
                f"  [dim]Cannot merge {a.get('id','')} + {b.get('id','')}: "
                f"{result.get('reason_if_not','')[:60]}[/dim]"
            )
            self._log(a.get("id",""), b.get("id",""), None, success=False)
            return None

        merged_id = f"M_{a.get('id','?')}_{b.get('id','?')}"
        merged = {
            "id":               merged_id,
            "hypothesis":       result.get("merged_hypothesis", ""),
            "mechanism":        result.get("mechanism", ""),
            "broken_assumption":f"Merged: {a.get('broken_assumption','')} + {b.get('broken_assumption','')}",
            "prediction":       result.get("prediction", {}),
            "known_unknowns":   result.get("known_unknowns", []),
            "parents":          [a.get("id",""), b.get("id","")],
            "operation":        "merge",
            "merge_details":    result,
        }

        self._log(a.get("id",""), b.get("id",""), json.dumps(merged), success=True)

        console.print(
            f"  [green]✓ Merged {a.get('id','')} + {b.get('id','')} → {merged_id}[/green]"
        )
        console.print(f"    {result.get('improvement','')[:70]}")
        return merged

    def _log(self, a_id: str, b_id: str, theory_json: Optional[str], success: bool):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO merges (theory_a_id, theory_b_id, merged_theory, success) VALUES (?, ?, ?, ?)",
                (a_id, b_id, theory_json or "", 1 if success else 0),
            )
            conn.commit()

    def merge_mediocre(self, all_theories: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Finds mediocre theories and tries to merge them.
        Returns (merged_theories, remaining_mediocre_theories)
        """
        console.print(f"\n[bold]═══ Theory Merger ═══[/bold]")

        mediocre = [
            h for h in all_theories
            if self.MEDIOCRE_THRESHOLD_LOW
               <= h.get("discovery_score", {}).get("final_score", 0)
               <= self.MEDIOCRE_THRESHOLD_HIGH
        ]

        if len(mediocre) < 2:
            console.print(f"  [dim]Not enough mediocre theories to merge (need ≥2, have {len(mediocre)})[/dim]")
            return [], mediocre

        console.print(f"  Found {len(mediocre)} mediocre theories — attempting merges...")

        merged_results = []
        used = set()

        for i in range(len(mediocre)):
            for j in range(i + 1, len(mediocre)):
                if i in used or j in used:
                    continue

                a, b = mediocre[i], mediocre[j]
                console.print(f"  Attempting: {a.get('id','')} + {b.get('id','')}...")
                result = self.merge(a, b)

                if result:
                    merged_results.append(result)
                    used.add(i)
                    used.add(j)

        unmerged = [mediocre[i] for i in range(len(mediocre)) if i not in used]
        console.print(f"  [green]{len(merged_results)} successful merges[/green]")
        return merged_results, unmerged

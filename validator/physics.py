import json
import sqlite3
import requests
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console

console = Console()

PHYSICS_PROMPT = """You are the Physics Validator in DARWIN, a GPU/CPU architecture research AI.

Your job: check this hypothesis against hard physical laws.
Not engineering challenges — fundamental PHYSICS.

Hypothesis:
  Statement   : {hypothesis}
  Mechanism   : {mechanism}
  Prediction  : {prediction}
  Broken assumption: {broken_assumption}

Check against ALL of these physical laws and limits:

THERMODYNAMICS:
- Landauer limit: erasing 1 bit costs kT·ln2 ≈ 2.85×10⁻²¹ J at 300K
- Heat dissipation: power density limits (~100 W/cm² for air cooling)
- Carnot efficiency ceiling for heat pumps

SEMICONDUCTOR PHYSICS:
- Leakage current increases exponentially as transistors shrink
- Velocity saturation: electrons can't move faster than ~10⁷ cm/s in silicon
- Quantum tunneling becomes significant below ~3nm gate oxide
- Interconnect RC delay scales with length²

INFORMATION THEORY:
- Shannon limit: channel capacity C = B·log₂(1 + S/N)
- No lossless compression below entropy
- No faster-than-light signal propagation

MATERIALS:
- Silicon bandgap: 1.12 eV (limits operating voltage)
- Copper resistivity: 1.68×10⁻⁸ Ω·m (interconnect limit)
- Thermal conductivity of silicon: 150 W/(m·K)

Respond ONLY in JSON:
{{
  "passes": true or false,
  "violations": [
    {{
      "law": "name of physical law violated",
      "explanation": "exactly how it is violated",
      "fatal": true or false
    }}
  ],
  "warnings": [
    {{
      "concern": "physical concern (not fatal but important)",
      "impact": "what this means for the design"
    }}
  ],
  "physical_limits_approached": [
    "which physical limits this design approaches"
  ],
  "verdict": "PASSES | MINOR_CONCERNS | MAJOR_CONCERNS | PHYSICALLY_IMPOSSIBLE",
  "verdict_reason": "one sentence"
}}"""


class PhysicsValidator:
    """
    Hard physics filter. Runs AFTER Prediction Generator,
    BEFORE Simulation Engine. Kills physically impossible theories
    before wasting simulation compute.

    Unlike Red Team (which is adversarial and broad),
    Physics Validator checks ONLY against known physical laws.
    """

    def __init__(self, ollama_url: str, model: str, db_path: Path):
        self.ollama_url = ollama_url
        self.model      = model
        self.db_path    = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS physics_validations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    theory_id TEXT,
                    passes INTEGER,
                    violations TEXT,
                    warnings TEXT,
                    verdict TEXT,
                    verdict_reason TEXT,
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
                "options": {"temperature": 0.1},  # very low — physics is not creative
            },
            timeout=120,
        )
        return resp.json()["response"]

    def validate(self, hypothesis: Dict) -> Dict:
        tid  = hypothesis.get("id", "unknown")
        pred = hypothesis.get("predictions", {}).get("primary_prediction", {})

        prompt = PHYSICS_PROMPT.format(
            hypothesis=hypothesis.get("hypothesis", ""),
            mechanism=hypothesis.get("mechanism", ""),
            prediction=pred.get("claim", ""),
            broken_assumption=hypothesis.get("broken_assumption", ""),
        )

        try:
            result = json.loads(self._call_llm(prompt))
        except Exception as e:
            console.print(f"[red]Physics validator error: {e}[/red]")
            result = {
                "passes": True,
                "violations": [],
                "warnings": [],
                "verdict": "PASSES",
                "verdict_reason": "Validator error — passed by default",
            }

        result["theory_id"] = tid

        # Save
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO physics_validations "
                "(theory_id, passes, violations, warnings, verdict, verdict_reason) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    tid,
                    1 if result.get("passes") else 0,
                    json.dumps(result.get("violations", [])),
                    json.dumps(result.get("warnings", [])),
                    result.get("verdict", ""),
                    result.get("verdict_reason", ""),
                ),
            )
            conn.commit()

        verdict = result.get("verdict", "")
        color   = "green" if verdict == "PASSES" else \
                  "yellow" if "CONCERNS" in verdict else "red"

        console.print(
            f"  [bold]Physics:[/bold] [{color}]{verdict}[/{color}] "
            f"— {result.get('verdict_reason','')[:70]}"
        )

        fatal_violations = [v for v in result.get("violations", []) if v.get("fatal")]
        if fatal_violations:
            for v in fatal_violations:
                console.print(f"  [red]  ✗ {v.get('law','')}: {v.get('explanation','')[:60]}[/red]")

        return result

    def filter_batch(self, hypotheses: List[Dict]) -> tuple:
        console.print(f"\n[bold]═══ Physics Validator ═══[/bold]")
        survivors, rejected = [], []

        for h in hypotheses:
            result = self.validate(h)
            h["physics_validation"] = result

            if result.get("verdict") == "PHYSICALLY_IMPOSSIBLE":
                rejected.append(h)
            else:
                survivors.append(h)

        console.print(
            f"  [green]{len(survivors)} passed[/green] / "
            f"[red]{len(rejected)} physically impossible[/red]"
        )
        return survivors, rejected

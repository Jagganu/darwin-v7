import json
import sqlite3
import requests
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console

console = Console()

REALITY_CHECK_PROMPT = """You are the Reality Checker in DARWIN, a GPU/CPU architecture research AI.

A simulation returned results. Now check if these results are REALISTIC
based on your knowledge of real GPU/CPU hardware, published benchmarks,
and manufacturing constraints.

Theory:
  Hypothesis : {hypothesis}
  Prediction : {prediction}

Simulation result:
  {sim_result}

Cross-reference against real hardware knowledge:

REAL GPU BENCHMARKS (as of 2024-2025):
- NVIDIA H100: 3.35 TB/s HBM3 bandwidth, 80GB HBM3, 700W TDP
- NVIDIA RTX 4090: 1 TB/s GDDR6X bandwidth, 450W TDP, 82.6 TFLOPS
- AMD RX 7900 XTX: 960 GB/s bandwidth, 355W TDP
- Intel Arc A770: 560 GB/s bandwidth, 225W TDP

REAL MEMORY SPECS:
- HBM3: 819 GB/s per stack, ~$30-50 per GB
- GDDR6X: 21 Gbps per pin, ~$5-8 per GB
- LPDDR5X: 85 GB/s total bandwidth

REAL MANUFACTURING COSTS:
- TSMC 3nm: ~$20,000 per wafer
- TSMC 5nm: ~$16,000 per wafer
- Die yield drops ~50% per node for complex designs

Check:
1. Is the claimed performance gain physically achievable given real hardware?
2. Does the cost prediction match real manufacturing economics?
3. Have similar approaches been tried and failed in real products?
4. Is the simulation result suspiciously optimistic vs real hardware?

Respond ONLY in JSON:
{{
  "reality_aligned": true or false,
  "optimism_factor": "how much the sim over/under estimates e.g. 2.3x optimistic",
  "real_world_issues": [
    "issue that simulation missed based on real hardware knowledge"
  ],
  "comparable_real_designs": [
    "real GPU/CPU design that tried something similar and what happened"
  ],
  "adjusted_confidence": 0.0 to 1.0,
  "confidence_change": "increased | decreased | unchanged",
  "confidence_reason": "why confidence changed",
  "verdict": "REALISTIC | OPTIMISTIC | PESSIMISTIC | UNREALISTIC",
  "recommendation": "proceed | revise_prediction | redesign | abandon"
}}"""


class RealityChecker:
    """
    Runs AFTER Simulation Engine.
    Cross-references simulation results against:
    - Real published GPU/CPU benchmarks
    - Known manufacturing constraints
    - Historical attempts at similar designs

    A theory that only works in simulation is dangerous.
    This module grounds the system in physical reality.

    Feedback: updates Confidence Tracker with reality-adjusted score.
    """

    def __init__(self, ollama_url: str, model: str, db_path: Path):
        self.ollama_url = ollama_url
        self.model      = model
        self.db_path    = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS reality_checks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    theory_id TEXT,
                    reality_aligned INTEGER,
                    optimism_factor TEXT,
                    real_world_issues TEXT,
                    adjusted_confidence REAL,
                    verdict TEXT,
                    recommendation TEXT,
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
                "options": {"temperature": 0.2},
            },
            timeout=120,
        )
        return resp.json()["response"]

    def check(self, hypothesis: Dict) -> Optional[Dict]:
        tid  = hypothesis.get("id", "unknown")
        pred = hypothesis.get("predictions", {}).get("primary_prediction", {})
        sim  = hypothesis.get("simulation", {})

        if not sim:
            console.print(f"  [dim]{tid}: No simulation data — skipping reality check[/dim]")
            return None

        prompt = REALITY_CHECK_PROMPT.format(
            hypothesis=hypothesis.get("hypothesis", ""),
            prediction=pred.get("claim", ""),
            sim_result=json.dumps({
                "performance_delta": sim.get("estimated_performance_delta", ""),
                "power_delta":       sim.get("estimated_power_delta", ""),
                "cost_delta":        sim.get("estimated_cost_delta", ""),
                "quick_score":       sim.get("quick_score", 0),
            }, indent=2),
        )

        try:
            result = json.loads(self._call_llm(prompt))
        except Exception as e:
            console.print(f"[red]Reality checker error: {e}[/red]")
            return None

        result["theory_id"] = tid

        # Save
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO reality_checks "
                "(theory_id, reality_aligned, optimism_factor, real_world_issues, "
                "adjusted_confidence, verdict, recommendation) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    tid,
                    1 if result.get("reality_aligned") else 0,
                    result.get("optimism_factor", ""),
                    json.dumps(result.get("real_world_issues", [])),
                    result.get("adjusted_confidence", 0.5),
                    result.get("verdict", ""),
                    result.get("recommendation", ""),
                ),
            )
            conn.commit()

        verdict = result.get("verdict", "")
        color   = "green" if verdict == "REALISTIC" else \
                  "yellow" if verdict in ("OPTIMISTIC", "PESSIMISTIC") else "red"

        console.print(
            f"  {tid} [{color}]{verdict}[/{color}] "
            f"conf={result.get('adjusted_confidence', 0):.2f} "
            f"→ {result.get('recommendation', '')}"
        )

        issues = result.get("real_world_issues", [])
        if issues:
            console.print(f"  [dim]Issues: {issues[0][:70]}[/dim]")

        return result

    def check_batch(self, hypotheses: List[Dict], confidence_tracker) -> tuple:
        console.print(f"\n[bold]═══ Reality Checker ═══[/bold]")
        checked, abandoned = [], []

        for h in hypotheses:
            result = self.check(h)
            if not result:
                checked.append(h)
                continue

            h["reality_check"] = result
            recommendation = result.get("recommendation", "proceed")

            if recommendation == "abandon":
                console.print(f"  [red]✗ Abandoned: {h.get('id','')}[/red]")
                abandoned.append(h)
            else:
                # Update confidence tracker with reality-adjusted score
                adj_conf = result.get("adjusted_confidence", 0.5)
                tid = h.get("id", "")
                if tid:
                    confidence_tracker.update(
                        tid,
                        evidence_for=["Reality check passed"] if result.get("reality_aligned") else [],
                        evidence_against=result.get("real_world_issues", [])[:2],
                    )
                checked.append(h)

        return checked, abandoned

    def get_check(self, theory_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT reality_aligned, verdict, adjusted_confidence, recommendation "
                "FROM reality_checks WHERE theory_id = ? ORDER BY created_at DESC LIMIT 1",
                (theory_id,),
            ).fetchone()
        if row:
            return {"aligned": row[0], "verdict": row[1], "confidence": row[2], "recommendation": row[3]}
        return None

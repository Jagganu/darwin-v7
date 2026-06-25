import json
import sqlite3
import requests
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console

console = Console()

PLANNER_PROMPT = """You are the Research Planner in DARWIN, a GPU/CPU architecture research AI.

You act as the Principal Investigator. After each discovery cycle,
decide where to focus the next cycle for MAXIMUM discovery value.

Current state:
  Surviving theories     : {n_theories}
  Failure patterns       : {failures}
  Open research questions: {questions}
  Knowledge gaps         : {gaps}
  Discovered laws        : {laws}
  Surprises this cycle   : {surprises}
  Meta-learner focus     : {meta_focus}
  Cycle number           : {cycle}

Your tasks:
1. Identify the MOST VALUABLE unknown right now
2. Decide what the next cycle should focus on
3. Generate targeted search terms for the Knowledge Auditor
4. Decide if more hypotheses or deeper simulation is needed
5. Set priorities for the next 10 cycles

Calculate expected value for each research direction:
  EV = (potential_impact × probability_of_success) / estimated_cost

Respond ONLY in JSON:
{{
  "most_valuable_unknown": "the single most important thing we don't know",
  "next_cycle_focus": "what the next discovery cycle should target",
  "auditor_search_targets": [
    "specific search query for Knowledge Auditor contradiction hunt 1",
    "specific search query 2",
    "specific search query 3"
  ],
  "recommended_n_hypotheses": 5,
  "recommended_strategy": "breadth | depth | exploit_surprise | reinforce_law",
  "reasoning": "why this strategy now",
  "10_cycle_plan": [
    "cycle 1: ...",
    "cycle 2: ...",
    "cycle 3: ..."
  ],
  "expected_value_scores": {{
    "memory_architecture": 0.0 to 1.0,
    "compute_units": 0.0 to 1.0,
    "interconnects": 0.0 to 1.0,
    "materials": 0.0 to 1.0,
    "topology": 0.0 to 1.0
  }}
}}"""


class ResearchPlanner:
    """
    Runs at END of each discovery cycle.
    Acts as the Principal Investigator.

    Feedback loops:
      Research Planner → Knowledge Auditor  (targeted contradiction hunt)
      Research Planner → Question Generator (what to ask next)
      Research Planner → Hypothesis Generator (via meta focus)
    """

    def __init__(self, ollama_url: str, model: str, db_path: Path):
        self.ollama_url = ollama_url
        self.model      = model
        self.db_path    = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS research_plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle INTEGER,
                    most_valuable_unknown TEXT,
                    next_cycle_focus TEXT,
                    auditor_search_targets TEXT,
                    recommended_strategy TEXT,
                    reasoning TEXT,
                    plan_10_cycle TEXT,
                    ev_scores TEXT,
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
                "options": {"temperature": 0.4},
            },
            timeout=120,
        )
        return resp.json()["response"]

    def _get_state(self) -> Dict:
        with sqlite3.connect(self.db_path) as conn:
            n_theories = conn.execute("SELECT COUNT(*) FROM theories").fetchone()[0]

            failures = conn.execute(
                "SELECT failure_stage, COUNT(*) FROM failures GROUP BY failure_stage"
            ).fetchall()

            questions = conn.execute(
                "SELECT question FROM research_questions WHERE answered = 0 LIMIT 5"
            ).fetchall()

            laws = conn.execute(
                "SELECT law_name, confidence FROM discovered_laws ORDER BY confidence DESC LIMIT 3"
            ).fetchall()

            surprises = conn.execute(
                "SELECT surprise_type, new_insight FROM surprises ORDER BY created_at DESC LIMIT 3"
            ).fetchall()

            meta = conn.execute(
                "SELECT recommended_focus FROM meta_insights ORDER BY created_at DESC LIMIT 1"
            ).fetchone()

        return {
            "n_theories": n_theories,
            "failures": {r[0]: r[1] for r in failures},
            "questions": [r[0] for r in questions],
            "laws": [(r[0], r[1]) for r in laws],
            "surprises": [(r[0], r[1]) for r in surprises],
            "meta_focus": meta[0] if meta else "none yet",
        }

    def plan(self, cycle: int, gaps: List[str]) -> Optional[Dict]:
        console.print(f"\n[bold]═══ Research Planner — Cycle {cycle} ═══[/bold]")

        state = self._get_state()

        failures_text  = json.dumps(state["failures"])
        questions_text = "\n".join(f"- {q}" for q in state["questions"]) or "None"
        gaps_text      = "\n".join(f"- {g}" for g in gaps[:5]) or "None"
        laws_text      = "\n".join(f"- {l[0]} (conf={l[1]:.2f})" for l in state["laws"]) or "None yet"
        surprises_text = "\n".join(f"- [{s[0]}] {s[1][:60]}" for s in state["surprises"]) or "None"

        prompt = PLANNER_PROMPT.format(
            n_theories=state["n_theories"],
            failures=failures_text,
            questions=questions_text,
            gaps=gaps_text,
            laws=laws_text,
            surprises=surprises_text,
            meta_focus=state["meta_focus"],
            cycle=cycle,
        )

        try:
            result = json.loads(self._call_llm(prompt))
        except Exception as e:
            console.print(f"[red]Research planner error: {e}[/red]")
            return None

        result["cycle"] = cycle

        # Save
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO research_plans "
                "(cycle, most_valuable_unknown, next_cycle_focus, auditor_search_targets, "
                "recommended_strategy, reasoning, plan_10_cycle, ev_scores) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    cycle,
                    result.get("most_valuable_unknown", ""),
                    result.get("next_cycle_focus", ""),
                    json.dumps(result.get("auditor_search_targets", [])),
                    result.get("recommended_strategy", ""),
                    result.get("reasoning", ""),
                    json.dumps(result.get("10_cycle_plan", [])),
                    json.dumps(result.get("expected_value_scores", {})),
                ),
            )
            conn.commit()

        # Display
        console.print(f"  [bold]Most valuable unknown:[/bold] {result.get('most_valuable_unknown','')[:80]}")
        console.print(f"  [bold]Next focus:[/bold] {result.get('next_cycle_focus','')[:80]}")
        console.print(f"  [bold]Strategy:[/bold] {result.get('recommended_strategy','')} — {result.get('reasoning','')[:60]}")

        targets = result.get("auditor_search_targets", [])
        if targets:
            console.print(f"  [cyan]Auditor targets:[/cyan]")
            for t in targets[:3]:
                console.print(f"    → {t}")

        return result

    def get_latest(self) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT most_valuable_unknown, next_cycle_focus, auditor_search_targets, "
                "recommended_strategy FROM research_plans ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        if row:
            return {
                "unknown":   row[0],
                "focus":     row[1],
                "targets":   json.loads(row[2]),
                "strategy":  row[3],
            }
        return None

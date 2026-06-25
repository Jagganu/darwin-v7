import json
import sqlite3
import requests
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console

console = Console()

META_LEARN_PROMPT = """You are the Meta-Learner in DARWIN, a GPU/CPU architecture research AI.

Analyze the research history below and extract patterns about what kinds
of hypotheses SUCCEED vs FAIL.

Research History:
{history}

Success Statistics:
{stats}

Your task: identify patterns that will help future generations generate
better hypotheses from the start.

Respond ONLY in JSON:
{{
  "success_patterns": [
    {{
      "pattern": "description of what tends to succeed",
      "success_rate": "e.g. 23%",
      "example": "brief example"
    }}
  ],
  "failure_patterns": [
    {{
      "pattern": "description of what tends to fail",
      "failure_rate": "e.g. 78%",
      "reason": "why this tends to fail"
    }}
  ],
  "generator_bias": {{
    "prefer": ["type of hypothesis to generate more of"],
    "avoid": ["type of hypothesis to avoid generating"],
    "assumption_targets": ["which assumptions are most productive to break"]
  }},
  "research_insight": "one paragraph describing what DARWIN has learned about GPU/CPU architecture research so far",
  "recommended_focus": "where to focus the next 100 cycles"
}}"""


class MetaLearner:
    """
    Runs every N cycles. Analyzes full research history.
    Produces bias instructions for Hypothesis Generator.
    Feeds back into the Question Generator and Assumption Miner.
    """

    CYCLE_THRESHOLD = 20   # Run meta-learning every N theories stored

    def __init__(self, ollama_url: str, model: str, db_path: Path):
        self.ollama_url = ollama_url
        self.model      = model
        self.db_path    = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS meta_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cycle INTEGER,
                    success_patterns TEXT,
                    failure_patterns TEXT,
                    generator_bias TEXT,
                    research_insight TEXT,
                    recommended_focus TEXT,
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
            timeout=180,
        )
        return resp.json()["response"]

    def _get_history(self) -> tuple:
        with sqlite3.connect(self.db_path) as conn:
            # Get surviving theories
            theories = conn.execute(
                "SELECT id, hypothesis, confidence, status FROM theories ORDER BY created_at DESC LIMIT 50"
            ).fetchall()

            # Get failures
            failures = conn.execute(
                "SELECT theory_text, failure_stage, failure_reasons FROM failures ORDER BY created_at DESC LIMIT 50"
            ).fetchall()

            # Get debate results
            debate_stats = conn.execute(
                "SELECT agent_name, verdict, COUNT(*) as count FROM debate_results GROUP BY agent_name, verdict"
            ).fetchall()

        return theories, failures, debate_stats

    def _compute_stats(self, theories, failures, debate_stats) -> Dict:
        total_generated = len(theories) + len(failures)
        survival_rate   = len(theories) / max(total_generated, 1)

        stage_failures = {}
        for f in failures:
            stage = f[1]
            stage_failures[stage] = stage_failures.get(stage, 0) + 1

        agent_rejects = {}
        for row in debate_stats:
            if row[1] == "REJECT":
                agent_rejects[row[0]] = row[2]

        return {
            "total_generated":  total_generated,
            "total_survived":   len(theories),
            "survival_rate":    f"{survival_rate:.1%}",
            "stage_failures":   stage_failures,
            "most_rejecting_agent": max(agent_rejects, key=agent_rejects.get) if agent_rejects else "none",
        }

    def should_run(self, cycle: int) -> bool:
        return cycle > 0 and cycle % self.CYCLE_THRESHOLD == 0

    def learn(self, cycle: int) -> Optional[Dict]:
        console.print(f"\n[bold]═══ Meta-Learner — Cycle {cycle} ═══[/bold]")

        theories, failures, debate_stats = self._get_history()

        if len(theories) + len(failures) < 5:
            console.print("  [dim]Not enough history yet. Need at least 5 theories.[/dim]")
            return None

        stats = self._compute_stats(theories, failures, debate_stats)

        # Format history for LLM
        survived_text = "\n".join(
            f"SURVIVED (conf={t[2]:.2f}): {t[1][:80]}" for t in theories[:15]
        )
        failed_text = "\n".join(
            f"FAILED [{f[1]}]: {f[0][:80]}" for f in failures[:15]
        )
        history = f"SURVIVING THEORIES:\n{survived_text}\n\nFAILED THEORIES:\n{failed_text}"

        prompt = META_LEARN_PROMPT.format(
            history=history,
            stats=json.dumps(stats, indent=2),
        )

        try:
            result = json.loads(self._call_llm(prompt))
        except Exception as e:
            console.print(f"[red]Meta-learner LLM error: {e}[/red]")
            return None

        result["cycle"] = cycle
        result["stats"] = stats

        # Save
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO meta_insights "
                "(cycle, success_patterns, failure_patterns, generator_bias, research_insight, recommended_focus) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    cycle,
                    json.dumps(result.get("success_patterns", [])),
                    json.dumps(result.get("failure_patterns", [])),
                    json.dumps(result.get("generator_bias", {})),
                    result.get("research_insight", ""),
                    result.get("recommended_focus", ""),
                ),
            )
            conn.commit()

        # Display key insight
        console.print(f"\n  [bold green]Research Insight:[/bold green]")
        console.print(f"  {result.get('research_insight', '')[:200]}")
        console.print(f"\n  [bold]Focus:[/bold] {result.get('recommended_focus', '')[:100]}")
        console.print(f"  Survival rate so far: {stats['survival_rate']}")

        return result

    def get_latest_bias(self) -> Optional[Dict]:
        """Returns the latest generator bias — injected into Hypothesis Generator."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT generator_bias, recommended_focus FROM meta_insights ORDER BY created_at DESC LIMIT 1"
            ).fetchone()
        if row:
            return {
                "generator_bias": json.loads(row[0]),
                "focus": row[1],
            }
        return None

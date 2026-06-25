import json
import sqlite3
import requests
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console

console = Console()

QUESTION_PROMPT = """You are the Question Generator in DARWIN, a GPU/CPU architecture research AI.

Good science starts with good questions. Before generating any hypothesis,
force yourself to ask the RIGHT questions first.

Current knowledge state:
  Knowledge gaps  : {gaps}
  Contradictions  : {conflicts}
  Recent surprises: {surprises}
  Meta-focus      : {focus}

Generate 5 targeted scientific questions that will guide hypothesis generation.

For each question ask:
- What information is MISSING that would change our understanding?
- Which assumption is WEAKEST in the current literature?
- What experiment would be MOST informative right now?
- What would CHANGE MY MIND about current GPU/CPU design?
- Where is the BIGGEST unexplored opportunity?

Respond ONLY in JSON:
{{
  "questions": [
    {{
      "question": "specific scientific question",
      "type": "missing_info | weak_assumption | key_experiment | mind_changer | opportunity",
      "why_important": "why answering this matters",
      "hypothesis_direction": "what kind of hypothesis this question points toward",
      "priority": "high | medium | low"
    }}
  ],
  "most_critical": "which single question is most important right now",
  "research_direction": "one paragraph on where the most fertile ground is"
}}"""


class QuestionGenerator:
    """
    Runs BEFORE the Hypothesis Generator.
    Forces the system to identify what it doesn't know
    before generating new theories.

    Feedback loop: Surprise Engine → Question Generator
    (anomalies automatically become new questions)
    """

    def __init__(self, ollama_url: str, model: str, db_path: Path):
        self.ollama_url = ollama_url
        self.model      = model
        self.db_path    = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS research_questions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    question TEXT,
                    type TEXT,
                    why_important TEXT,
                    hypothesis_direction TEXT,
                    priority TEXT,
                    answered INTEGER DEFAULT 0,
                    source TEXT DEFAULT 'question_generator',
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

    def _get_recent_surprises(self) -> List[str]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT new_insight FROM surprises ORDER BY created_at DESC LIMIT 5"
            ).fetchall()
        return [r[0] for r in rows if r[0]]

    def generate(
        self,
        gaps: List[str],
        conflicts: List[Dict],
        focus: str = "",
        surprise_seeds: List[Dict] = None,
    ) -> List[Dict]:
        surprises = self._get_recent_surprises()

        # Add surprise-spawned questions directly
        if surprise_seeds:
            for seed in surprise_seeds:
                for q in seed.get("new_questions", []):
                    self._save_question({
                        "question": q,
                        "type": "surprise_branch",
                        "why_important": "Spawned by simulation anomaly",
                        "hypothesis_direction": seed.get("hypothesis", ""),
                        "priority": "high",
                    }, source="surprise_engine")

        conflicts_text = "\n".join(f"- {c.get('description','')}" for c in conflicts[:3]) or "None yet"
        gaps_text      = "\n".join(f"- {g}" for g in gaps[:5]) or "None identified"
        surprise_text  = "\n".join(f"- {s}" for s in surprises[:3]) or "None yet"

        prompt = QUESTION_PROMPT.format(
            gaps=gaps_text,
            conflicts=conflicts_text,
            surprises=surprise_text,
            focus=focus or "general GPU/CPU architecture improvement",
        )

        try:
            result = json.loads(self._call_llm(prompt))
        except Exception as e:
            console.print(f"[red]Question generator error: {e}[/red]")
            return []

        questions = result.get("questions", [])

        console.print(f"\n[bold]═══ Question Generator ═══[/bold]")
        console.print(f"  Critical: [cyan]{result.get('most_critical', '')[:80]}[/cyan]")
        console.print(f"  Direction: {result.get('research_direction', '')[:100]}")

        for q in questions:
            priority = q.get("priority", "medium")
            color = "red" if priority == "high" else "yellow" if priority == "medium" else "dim"
            console.print(f"  [{color}]●[/{color}] {q.get('question','')[:80]}")
            self._save_question(q)

        return questions

    def _save_question(self, q: Dict, source: str = "question_generator"):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO research_questions "
                "(question, type, why_important, hypothesis_direction, priority, source) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    q.get("question", ""),
                    q.get("type", ""),
                    q.get("why_important", ""),
                    q.get("hypothesis_direction", ""),
                    q.get("priority", "medium"),
                    source,
                ),
            )
            conn.commit()

    def get_open_questions(self, n: int = 10) -> List[Dict]:
        """Returns unanswered high-priority questions — fed into Hypothesis Generator."""
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT question, type, hypothesis_direction, priority FROM research_questions "
                "WHERE answered = 0 ORDER BY priority DESC, created_at DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [
            {"question": r[0], "type": r[1], "direction": r[2], "priority": r[3]}
            for r in rows
        ]

    def inject_into_gaps(self, existing_gaps: List[str]) -> List[str]:
        """Converts open questions into gap strings for the Hypothesis Generator."""
        questions = self.get_open_questions(5)
        question_gaps = [f"Open question: {q['question']}" for q in questions]
        return existing_gaps + question_gaps

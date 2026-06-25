import json
import sqlite3
import requests
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console

console = Console()

MINE_PROMPT = """You are the Assumption Miner in DARWIN, a GPU/CPU architecture research AI.

Every major GPU/CPU breakthrough came from breaking an existing assumption.
Your job: systematically identify ALL current assumptions and select the
most productive one to break this cycle.

Current GPU/CPU architecture assumes:
{known_assumptions}

Research questions this cycle:
{questions}

Knowledge gaps:
{gaps}

Already broken this session (don't repeat):
{already_broken}

Tasks:
1. List 10 assumptions currently accepted in GPU/CPU design
2. Rank them by "productivity if broken" — how big would the discovery be?
3. Select the ONE best assumption to break this cycle
4. Explain what a theory breaking it would look like

Respond ONLY in JSON:
{{
  "all_assumptions": [
    {{
      "assumption": "statement of what is assumed",
      "domain": "memory | compute | power | materials | topology | software",
      "strength": "how strongly held is this assumption 0.0-1.0",
      "break_potential": "how big would the discovery be if broken 0.0-1.0",
      "evidence_for": "why people believe this",
      "evidence_against": "any cracks in this assumption"
    }}
  ],
  "selected_assumption": "the ONE assumption to break this cycle",
  "selection_reason": "why this is the most productive to break now",
  "breaking_direction": "what direction a theory breaking this would take",
  "example_broken_assumption": "an example of a past GPU breakthrough that broke an assumption"
}}"""

# Default assumptions seeded into the system
DEFAULT_GPU_ASSUMPTIONS = [
    "Memory must be physically separate from compute units",
    "Computing must be digital (binary)",
    "Clock-driven synchronous execution is required",
    "Silicon is the best substrate for compute",
    "GPU threads must execute the same instruction (SIMD)",
    "Cache hierarchies must be managed by hardware",
    "Floating point is the best format for AI compute",
    "Power delivery must be through the PCB",
    "Cooling must be external to the chip",
    "Interconnects must use electrical signals",
    "Memory bandwidth scales with memory capacity",
    "More transistors always equals more performance",
    "Instruction sets must be backward compatible",
    "GPUs and CPUs must be separate chips",
    "Error correction must be done in software or separate hardware",
]


class AssumptionMiner:
    """
    Runs after Question Generator, before Hypothesis Generator.
    Selects one assumption to break per cycle.
    Tracks which assumptions have been broken to avoid repetition.
    """

    def __init__(self, ollama_url: str, model: str, db_path: Path):
        self.ollama_url = ollama_url
        self.model      = model
        self.db_path    = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS assumptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    assumption TEXT UNIQUE,
                    domain TEXT,
                    strength REAL,
                    break_potential REAL,
                    broken INTEGER DEFAULT 0,
                    broken_by_theory TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            # Seed default assumptions
            for a in DEFAULT_GPU_ASSUMPTIONS:
                conn.execute(
                    "INSERT OR IGNORE INTO assumptions (assumption, domain, strength, break_potential) "
                    "VALUES (?, 'general', 0.8, 0.5)",
                    (a,),
                )
            conn.commit()

    def _call_llm(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.6},
            },
            timeout=120,
        )
        return resp.json()["response"]

    def _get_already_broken(self) -> List[str]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT assumption FROM assumptions WHERE broken = 1"
            ).fetchall()
        return [r[0] for r in rows]

    def mine(self, questions: List[Dict], gaps: List[str]) -> Optional[Dict]:
        console.print("\n[bold]═══ Assumption Miner ═══[/bold]")

        already_broken  = self._get_already_broken()
        questions_text  = "\n".join(f"- {q.get('question','')}" for q in questions[:5]) or "None"
        gaps_text       = "\n".join(f"- {g}" for g in gaps[:5]) or "None"
        broken_text     = "\n".join(f"- {a}" for a in already_broken[:10]) or "None yet"
        known_text      = "\n".join(f"- {a}" for a in DEFAULT_GPU_ASSUMPTIONS)

        prompt = MINE_PROMPT.format(
            known_assumptions=known_text,
            questions=questions_text,
            gaps=gaps_text,
            already_broken=broken_text,
        )

        try:
            result = json.loads(self._call_llm(prompt))
        except Exception as e:
            console.print(f"[red]Assumption miner error: {e}[/red]")
            return None

        # Save new assumptions to DB
        for a in result.get("all_assumptions", []):
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO assumptions (assumption, domain, strength, break_potential) "
                    "VALUES (?, ?, ?, ?)",
                    (
                        a.get("assumption", ""),
                        a.get("domain", "general"),
                        a.get("strength", 0.5),
                        a.get("break_potential", 0.5),
                    ),
                )
                conn.commit()

        selected = result.get("selected_assumption", "")
        console.print(f"  [bold green]Selected assumption to break:[/bold green]")
        console.print(f"  → {selected}")
        console.print(f"  Reason: {result.get('selection_reason', '')[:80]}")
        console.print(f"  Direction: {result.get('breaking_direction', '')[:80]}")

        return result

    def mark_broken(self, assumption: str, theory_id: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE assumptions SET broken = 1, broken_by_theory = ? WHERE assumption = ?",
                (theory_id, assumption),
            )
            conn.commit()

    def get_all(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT assumption, domain, break_potential, broken FROM assumptions ORDER BY break_potential DESC"
            ).fetchall()
        return [{"assumption": r[0], "domain": r[1], "potential": r[2], "broken": r[3]} for r in rows]

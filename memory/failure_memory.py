import sqlite3
import json
from pathlib import Path
from typing import Dict, List
from rich.console import Console

console = Console()


class FailureMemory:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS failures (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    theory_id TEXT,
                    theory_text TEXT,
                    failure_reasons TEXT,
                    failure_stage TEXT,
                    lessons TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def record(self, theory: Dict, reasons: List[str], stage: str, lessons: List[str] = None):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO failures (theory_id, theory_text, failure_reasons, failure_stage, lessons) "
                "VALUES (?, ?, ?, ?, ?)",
                (
                    theory.get("id", "unknown"),
                    theory.get("hypothesis", "")[:300],
                    json.dumps(reasons),
                    stage,
                    json.dumps(lessons or []),
                ),
            )
            conn.commit()
        first = reasons[0][:60] if reasons else "unknown"
        console.print(f"[dim]💾 Failure logged [{stage}]: {first}...[/dim]")

    def get_lessons(self, n: int = 10) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT theory_text, failure_reasons, failure_stage, lessons "
                "FROM failures ORDER BY created_at DESC LIMIT ?",
                (n,),
            ).fetchall()
        return [
            {
                "theory": r[0],
                "reasons": json.loads(r[1]),
                "stage": r[2],
                "lessons": json.loads(r[3]),
            }
            for r in rows
        ]

    def get_patterns(self) -> Dict[str, int]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT failure_stage, COUNT(*) FROM failures GROUP BY failure_stage ORDER BY COUNT(*) DESC"
            ).fetchall()
        return {r[0]: r[1] for r in rows}

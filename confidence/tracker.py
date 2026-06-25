import sqlite3
import json
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console
from rich.table import Table

console = Console()


class ConfidenceTracker:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS confidence (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    theory_id TEXT,
                    confidence REAL,
                    evidence_for TEXT,
                    evidence_against TEXT,
                    known_unknowns TEXT,
                    untested_regions TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def create(self, theory_id: str, initial: float = 0.5, known_unknowns: List[str] = None) -> Dict:
        record = {
            "theory_id": theory_id,
            "confidence": initial,
            "evidence_for": [],
            "evidence_against": [],
            "known_unknowns": known_unknowns or [],
            "untested_regions": [],
        }
        self._save(record)
        return record

    def update(
        self,
        theory_id: str,
        evidence_for: List[str] = None,
        evidence_against: List[str] = None,
        known_unknowns: List[str] = None,
        untested_regions: List[str] = None,
    ) -> Dict:
        record = self.get(theory_id) or self.create(theory_id)

        if evidence_for:
            record["evidence_for"].extend(evidence_for)
            # Bayesian-style update: diminishing returns
            boost = 0.1 * len(evidence_for) * (1 - record["confidence"])
            record["confidence"] = min(0.98, record["confidence"] + boost)

        if evidence_against:
            record["evidence_against"].extend(evidence_against)
            penalty = 0.15 * len(evidence_against) * record["confidence"]
            record["confidence"] = max(0.02, record["confidence"] - penalty)

        if known_unknowns:
            record["known_unknowns"].extend(known_unknowns)

        if untested_regions:
            record["untested_regions"].extend(untested_regions)

        self._save(record)
        return record

    def _save(self, record: Dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO confidence (theory_id, confidence, evidence_for, evidence_against, known_unknowns, untested_regions) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    record["theory_id"],
                    record["confidence"],
                    json.dumps(record["evidence_for"]),
                    json.dumps(record["evidence_against"]),
                    json.dumps(record["known_unknowns"]),
                    json.dumps(record["untested_regions"]),
                ),
            )
            conn.commit()

    def get(self, theory_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT theory_id, confidence, evidence_for, evidence_against, known_unknowns, untested_regions "
                "FROM confidence WHERE theory_id = ? ORDER BY updated_at DESC LIMIT 1",
                (theory_id,),
            ).fetchone()
        if row:
            return {
                "theory_id": row[0],
                "confidence": row[1],
                "evidence_for": json.loads(row[2]),
                "evidence_against": json.loads(row[3]),
                "known_unknowns": json.loads(row[4]),
                "untested_regions": json.loads(row[5]),
            }
        return None

    def display(self, theory_id: str):
        r = self.get(theory_id)
        if not r:
            console.print("[red]No record found[/red]")
            return

        bar_len = int(r["confidence"] * 20)
        bar = "█" * bar_len + "░" * (20 - bar_len)

        console.print(f"\n[bold cyan]Confidence: {theory_id}[/bold cyan]")
        console.print(f"  Score : [{bar}] {r['confidence']:.2f}")
        console.print(f"  For   : {len(r['evidence_for'])} items")
        console.print(f"  Against: {len(r['evidence_against'])} items")
        if r["known_unknowns"]:
            console.print("  [yellow]Known unknowns:[/yellow]")
            for u in r["known_unknowns"][:3]:
                console.print(f"    - {u}")
        if r["untested_regions"]:
            console.print("  [dim]Untested regions:[/dim]")
            for u in r["untested_regions"][:3]:
                console.print(f"    - {u}")

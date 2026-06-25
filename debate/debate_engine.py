import json
import sqlite3
from pathlib import Path
from typing import List, Dict, Tuple
from rich.console import Console
from rich.table import Table

from .agents import DebateAgent

console = Console()

AGENT_NAMES = ["physics", "manufacturing", "cost", "performance", "reliability", "environmental"]

# Verdict thresholds
REJECT_THRESHOLD  = 2   # REJECT votes needed to kill theory
CONCERN_THRESHOLD = 4   # CONCERN votes needed to weaken theory
FATAL_KILLS       = 1   # Any single fatal flag kills theory immediately


class DebateEngine:
    def __init__(self, ollama_url: str, model: str, db_path: Path):
        self.agents  = [DebateAgent(name, ollama_url, model) for name in AGENT_NAMES]
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS debate_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    theory_id TEXT,
                    agent_name TEXT,
                    verdict TEXT,
                    score REAL,
                    findings TEXT,
                    fatal INTEGER,
                    recommendation TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _save_agent_result(self, theory_id: str, result: Dict):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO debate_results "
                "(theory_id, agent_name, verdict, score, findings, fatal, recommendation) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    theory_id,
                    result["domain"],
                    result["verdict"],
                    result.get("score", 0.5),
                    json.dumps(result.get("findings", [])),
                    1 if result.get("fatal") else 0,
                    result.get("recommendation", ""),
                ),
            )
            conn.commit()

    def _aggregate(self, agent_results: List[Dict]) -> Dict:
        rejects  = sum(1 for r in agent_results if r["verdict"] == "REJECT")
        concerns = sum(1 for r in agent_results if r["verdict"] == "CONCERN")
        fatals   = sum(1 for r in agent_results if r.get("fatal"))
        avg_score = sum(r.get("score", 0.5) for r in agent_results) / len(agent_results)

        # Determine overall verdict
        if fatals >= FATAL_KILLS:
            overall = "REJECTED"
            reason  = f"{fatals} agent(s) flagged fatal flaw"
        elif rejects >= REJECT_THRESHOLD:
            overall = "REJECTED"
            reason  = f"{rejects} agents voted REJECT"
        elif concerns >= CONCERN_THRESHOLD:
            overall = "WEAKENED"
            reason  = f"{concerns} agents raised concerns"
        else:
            overall = "APPROVED"
            reason  = f"Passed {len(agent_results) - rejects - concerns}/{len(agent_results)} agents"

        all_findings = []
        for r in agent_results:
            for f in r.get("findings", []):
                all_findings.append(f"[{r['domain']}] {f}")

        return {
            "verdict": overall,
            "reason": reason,
            "avg_score": avg_score,
            "rejects": rejects,
            "concerns": concerns,
            "fatals": fatals,
            "all_findings": all_findings,
            "agent_results": agent_results,
        }

    def debate(self, hypothesis: Dict) -> Dict:
        tid = hypothesis.get("id", "unknown")
        console.print(f"\n[bold]⚖️  Debate: [cyan]{hypothesis.get('hypothesis', '')[:60]}...[/cyan][/bold]")
        console.print("─" * 70)

        agent_results = []
        for agent in self.agents:
            result = agent.evaluate(hypothesis)
            self._save_agent_result(tid, result)
            agent_results.append(result)

        summary = self._aggregate(agent_results)

        color = "green" if summary["verdict"] == "APPROVED" else \
                "yellow" if summary["verdict"] == "WEAKENED" else "red"

        console.print(f"\n  [bold {color}]→ {summary['verdict']}[/bold {color}] — {summary['reason']}")
        console.print(f"  Average score: {summary['avg_score']:.2f}")

        return summary

    def filter_batch(self, hypotheses: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        survivors, rejected = [], []

        for h in hypotheses:
            result = self.debate(h)
            h["debate"] = result

            if result["verdict"] == "REJECTED":
                rejected.append(h)
            else:
                # Attach weakness list for confidence tracker
                h["debate_weaknesses"] = [
                    f for f in result["all_findings"]
                    if "[physics]" in f or "[manufacturing]" in f
                ]
                survivors.append(h)

        console.print(
            f"\n[bold]Debate Results:[/bold] "
            f"[green]{len(survivors)} approved/weakened[/green] / "
            f"[red]{len(rejected)} rejected[/red]"
        )
        return survivors, rejected

    def get_debate_history(self, theory_id: str) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT agent_name, verdict, score, findings, recommendation "
                "FROM debate_results WHERE theory_id = ?",
                (theory_id,),
            ).fetchall()
        return [
            {
                "agent": r[0], "verdict": r[1], "score": r[2],
                "findings": json.loads(r[3]), "recommendation": r[4],
            }
            for r in rows
        ]

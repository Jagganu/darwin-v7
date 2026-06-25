import sqlite3
import json
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console

console = Console()


class TheoryGraph:
    def __init__(self, db_path: Path):
        self.db_path = db_path
        self.graph = nx.DiGraph()
        self._init_db()
        self._load_graph()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS theories (
                    id TEXT PRIMARY KEY,
                    hypothesis TEXT,
                    broken_assumption TEXT,
                    mechanism TEXT,
                    prediction TEXT,
                    confidence REAL DEFAULT 0.5,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    full_data TEXT
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS theory_relations (
                    source_id TEXT,
                    target_id TEXT,
                    relation_type TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (source_id, target_id, relation_type)
                )
            """)
            conn.commit()

    def _load_graph(self):
        with sqlite3.connect(self.db_path) as conn:
            for row in conn.execute("SELECT id, hypothesis, confidence, status FROM theories"):
                self.graph.add_node(row[0], hypothesis=row[1], confidence=row[2], status=row[3])
            for row in conn.execute("SELECT source_id, target_id, relation_type FROM theory_relations"):
                self.graph.add_edge(row[0], row[1], relation=row[2])

    def add_theory(self, theory: Dict) -> str:
        tid = theory.get("id", f"T{len(self.graph.nodes):05d}")
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO theories "
                "(id, hypothesis, broken_assumption, mechanism, prediction, full_data) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    tid,
                    theory.get("hypothesis", ""),
                    theory.get("broken_assumption", ""),
                    theory.get("mechanism", ""),
                    json.dumps(theory.get("prediction", {})),
                    json.dumps(theory),
                ),
            )
            conn.commit()
        self.graph.add_node(tid, hypothesis=theory.get("hypothesis", ""), confidence=0.5, status="active")
        return tid

    def add_relation(self, source: str, target: str, relation: str):
        """relation: inspired | contradicts | merged_into | disproved"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO theory_relations (source_id, target_id, relation_type) VALUES (?, ?, ?)",
                (source, target, relation),
            )
            conn.commit()
        self.graph.add_edge(source, target, relation=relation)

    def update_confidence(self, tid: str, confidence: float):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("UPDATE theories SET confidence = ? WHERE id = ?", (confidence, tid))
            conn.commit()
        if tid in self.graph.nodes:
            self.graph.nodes[tid]["confidence"] = confidence

    def get_lineage(self, tid: str) -> Dict:
        if tid not in self.graph:
            return {}
        return {
            "ancestors": list(nx.ancestors(self.graph, tid)),
            "descendants": list(nx.descendants(self.graph, tid)),
            "inspired_by": [s for s, t, d in self.graph.in_edges(tid, data=True) if d.get("relation") == "inspired"],
            "disproved": [t for s, t, d in self.graph.out_edges(tid, data=True) if d.get("relation") == "disproved"],
        }

    def get_all(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, hypothesis, confidence, status FROM theories ORDER BY confidence DESC"
            ).fetchall()
        return [{"id": r[0], "hypothesis": r[1], "confidence": r[2], "status": r[3]} for r in rows]

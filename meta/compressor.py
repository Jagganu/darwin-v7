import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional
from rich.console import Console

# FIX: removed `import requests` — no longer calling Ollama directly

console = Console()

COMPRESS_PROMPT = """You are the Theory Compressor in DARWIN, a GPU/CPU architecture research AI.

You have accumulated many specific theories that all survived rigorous testing.
Your task: find the UNDERLYING PRINCIPLE that explains all or most of them.

This is how scientific laws are born — Newton unified falling apples and
planetary orbits under one principle. Find the equivalent here.

Surviving theories:
{theories}

Their common patterns:
- Broken assumptions: {assumptions}
- Mechanisms: {mechanisms}

Find:
1. A single unifying principle that explains WHY these all work
2. Predict what OTHER theories this principle would generate
3. Name it as a candidate "law" of GPU/CPU architecture

Respond ONLY in JSON:
{{
  "principle": "one clear statement of the unifying principle",
  "law_name": "proposed name for this law e.g. 'Jagan's Locality Principle'",
  "explains": ["theory_id 1", "theory_id 2"],
  "does_not_explain": ["theory_id that doesn't fit"],
  "new_predictions": [
    "what new theory this principle predicts",
    "another prediction"
  ],
  "analogy": "what existing law in physics/CS this resembles",
  "confidence": 0.0 to 1.0,
  "falsification": "what would disprove this principle"
}}"""


class TheoryCompressor:
    """
    After accumulating many scored theories:
    → Groups them by broken_assumption and mechanism type
    → Finds unifying principles
    → Names candidate architecture laws
    → Feeds back as research targets

    This is where DARWIN moves from "many good designs" to "new laws".
    """

    MIN_THEORIES = 8   # Minimum theories before compression makes sense

    # FIX: constructor was (self, ollama_url: str, model: str, db_path: Path) — crashed
    # at startup because main.py passes (self.llm, DB_PATH): DB_PATH landed in the
    # `model` slot and `db_path` was never provided.
    def __init__(self, llm, db_path: Path):
        self.llm     = llm
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS discovered_laws (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    law_name TEXT,
                    principle TEXT,
                    explains TEXT,
                    new_predictions TEXT,
                    analogy TEXT,
                    confidence REAL,
                    falsification TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _load_theories(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT id, hypothesis, broken_assumption, mechanism, confidence "
                "FROM theories WHERE confidence > 0.5 ORDER BY confidence DESC LIMIT 30"
            ).fetchall()
        return [
            {"id": r[0], "hypothesis": r[1], "broken_assumption": r[2],
             "mechanism": r[3], "confidence": r[4]}
            for r in rows
        ]

    def _group_by_theme(self, theories: List[Dict]) -> List[List[Dict]]:
        """Simple grouping by shared keywords in broken_assumption."""
        from collections import defaultdict
        groups = defaultdict(list)

        theme_keywords = {
            "memory":    ["memory", "bandwidth", "cache", "hbm", "dram"],
            "compute":   ["compute", "core", "shader", "pipeline", "execution"],
            "materials": ["silicon", "material", "substrate", "photonic", "analog"],
            "topology":  ["interconnect", "topology", "mesh", "noc", "bus"],
            "power":     ["power", "thermal", "heat", "energy", "voltage"],
        }

        for t in theories:
            text = (t.get("broken_assumption", "") + " " + t.get("mechanism", "")).lower()
            assigned = False
            for theme, keywords in theme_keywords.items():
                if any(k in text for k in keywords):
                    groups[theme].append(t)
                    assigned = True
                    break
            if not assigned:
                groups["other"].append(t)

        return [g for g in groups.values() if len(g) >= 2]

    def compress(self) -> List[Dict]:
        theories = self._load_theories()

        if len(theories) < self.MIN_THEORIES:
            console.print(
                f"  [dim]Theory Compressor: need {self.MIN_THEORIES} theories "
                f"(have {len(theories)}). Skipping.[/dim]"
            )
            return []

        console.print(f"\n[bold]═══ Theory Compressor ═══[/bold]")
        console.print(f"  Compressing {len(theories)} theories...")

        groups = self._group_by_theme(theories)
        laws   = []

        for group in groups:
            if len(group) < 2:
                continue

            theories_text = "\n".join(
                f"[{t['id']} conf={t['confidence']:.2f}] {t['hypothesis'][:120]}"
                for t in group
            )
            assumptions = list({t.get("broken_assumption", "")[:60] for t in group})
            mechanisms  = list({t.get("mechanism", "")[:60] for t in group})

            prompt = COMPRESS_PROMPT.format(
                theories=theories_text,
                assumptions="\n".join(f"- {a}" for a in assumptions[:5]),
                mechanisms="\n".join(f"- {m}" for m in mechanisms[:5]),
            )

            try:
                # FIX: was json.loads(self._call_llm(prompt)) hitting Ollama directly.
                # Temperature 0.5 preserved.
                result = self.llm.call_json(prompt, temperature=0.5)
            except Exception as e:
                console.print(f"  [red]Compressor error: {e}[/red]")
                continue

            law_name = result.get("law_name", "Unknown Law")
            conf     = result.get("confidence", 0.0)

            # Save
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "INSERT INTO discovered_laws "
                    "(law_name, principle, explains, new_predictions, analogy, confidence, falsification) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        law_name,
                        result.get("principle", ""),
                        json.dumps(result.get("explains", [])),
                        json.dumps(result.get("new_predictions", [])),
                        result.get("analogy", ""),
                        conf,
                        result.get("falsification", ""),
                    ),
                )
                conn.commit()

            console.print(f"\n  [bold yellow]⭐ Candidate Law Discovered:[/bold yellow]")
            console.print(f"  [bold]{law_name}[/bold]  (confidence={conf:.2f})")
            console.print(f"  Principle: {result.get('principle', '')[:120]}")
            console.print(f"  Analogy: {result.get('analogy', '')[:80]}")
            if result.get("new_predictions"):
                console.print(f"  Predicts: {result['new_predictions'][0][:80]}")

            result["group_size"] = len(group)
            laws.append(result)

        return laws

    def get_all_laws(self) -> List[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT law_name, principle, confidence, created_at FROM discovered_laws ORDER BY confidence DESC"
            ).fetchall()
        return [
            {"name": r[0], "principle": r[1], "confidence": r[2], "created_at": r[3]}
            for r in rows
        ]

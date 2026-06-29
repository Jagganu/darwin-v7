import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Tuple
from rich.console import Console

# FIX: removed `import requests` — no longer calling Ollama directly

console = Console()

CROSSBREED_PROMPT = """You are the Evolution Engine in DARWIN, a GPU/CPU architecture research AI.

Two theories have survived and been scored. Crossbreed them into a stronger child theory.

Theory A (score={score_a}):
  Hypothesis : {hyp_a}
  Mechanism  : {mech_a}
  Broken assumption: {assump_a}
  Weakest dimension: {weak_a}

Theory B (score={score_b}):
  Hypothesis : {hyp_b}
  Mechanism  : {mech_b}
  Broken assumption: {assump_b}
  Weakest dimension: {weak_b}

Create a child theory that:
1. Combines the strongest elements of both
2. Fixes the weakest dimension of each parent
3. Breaks a NEW assumption not broken by either parent
4. Is more specific and testable than both parents

Respond ONLY in JSON:
{{
  "hypothesis": "the child theory",
  "broken_assumption": "new assumption this breaks",
  "mechanism": "combined mechanism",
  "inherited_from_a": "what it took from Theory A",
  "inherited_from_b": "what it took from Theory B",
  "improvement_over_parents": "what makes this better",
  "prediction": {{
    "claim": "specific quantified prediction",
    "test_condition": "exact test conditions",
    "falsification": "what would disprove this"
  }},
  "known_unknowns": ["unknown 1", "unknown 2"]
}}"""

MUTATE_PROMPT = """You are the Evolution Engine in DARWIN, a GPU/CPU architecture research AI.

Mutate this theory by breaking ONE different assumption than the original.

Original Theory (score={score}):
  Hypothesis: {hypothesis}
  Broken assumption: {broken_assumption}
  Weakest dimension: {weakest}
  Evolution direction: {direction}

Create a MUTATED version that:
1. Keeps the core insight of the original
2. Breaks a DIFFERENT assumption to fix its weakest dimension
3. Remains physically feasible

Respond ONLY in JSON:
{{
  "hypothesis": "mutated hypothesis",
  "broken_assumption": "the NEW assumption this breaks",
  "mechanism": "how the mutation changes the mechanism",
  "mutation_type": "assumption_swap | mechanism_twist | scale_change | material_change",
  "prediction": {{
    "claim": "quantified prediction",
    "test_condition": "conditions",
    "falsification": "falsification criterion"
  }},
  "known_unknowns": ["unknown 1"]
}}"""


class EvolutionEngine:
    # FIX: constructor was (self, ollama_url: str, model: str, db_path: Path) — crashed
    # at startup because main.py passes (self.llm, DB_PATH): DB_PATH ended up in the
    # `model` slot and `db_path` was never provided.
    def __init__(self, llm, db_path: Path):
        self.llm     = llm
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS evolution_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    generation INTEGER,
                    parent_ids TEXT,
                    child_id TEXT,
                    operation TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _log(self, generation: int, parent_ids: List[str], child_id: str, op: str):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO evolution_log (generation, parent_ids, child_id, operation) VALUES (?, ?, ?, ?)",
                (generation, json.dumps(parent_ids), child_id, op),
            )
            conn.commit()

    # ── Diversity Manager ──────────────────────────────────────────────────
    def _similarity(self, a: str, b: str) -> float:
        """Simple word-overlap similarity. Replace with embeddings for production."""
        if not a or not b:
            return 0.0
        words_a = set(a.lower().split())
        words_b = set(b.lower().split())
        if not words_a or not words_b:
            return 0.0
        return len(words_a & words_b) / len(words_a | words_b)

    def deduplicate(self, theories: List[Dict], threshold: float = 0.70) -> List[Dict]:
        """Remove near-identical theories — keeps highest scored."""
        kept = []
        for candidate in theories:
            too_similar = False
            c_text = candidate.get("hypothesis", "")
            c_score = candidate.get("discovery_score", {}).get("final_score", 0)

            for i, existing in enumerate(kept):
                e_text = existing.get("hypothesis", "")
                sim = self._similarity(c_text, e_text)
                if sim > threshold:
                    e_score = existing.get("discovery_score", {}).get("final_score", 0)
                    if c_score > e_score:
                        kept[i] = candidate   # replace with better one
                    too_similar = True
                    break

            if not too_similar:
                kept.append(candidate)

        removed = len(theories) - len(kept)
        if removed:
            console.print(f"  [dim]Diversity manager: removed {removed} near-duplicates[/dim]")
        return kept

    # ── Crossbreeding ──────────────────────────────────────────────────────
    def crossbreed(self, a: Dict, b: Dict, generation: int) -> Dict:
        score_a = a.get("discovery_score", {}).get("final_score", 50)
        score_b = b.get("discovery_score", {}).get("final_score", 50)

        prompt = CROSSBREED_PROMPT.format(
            hyp_a=a.get("hypothesis", ""),    mech_a=a.get("mechanism", ""),
            assump_a=a.get("broken_assumption", ""),
            weak_a=a.get("discovery_score", {}).get("weakest_dimension", ""),
            score_a=score_a,
            hyp_b=b.get("hypothesis", ""),    mech_b=b.get("mechanism", ""),
            assump_b=b.get("broken_assumption", ""),
            weak_b=b.get("discovery_score", {}).get("weakest_dimension", ""),
            score_b=score_b,
        )

        try:
            # FIX: was json.loads(self._call_llm(prompt)) hitting Ollama directly.
            # High temperature 0.8 preserved — crossbreeding needs creativity.
            child = self.llm.call_json(prompt, temperature=0.8)
        except Exception as e:
            console.print(f"[red]Crossbreed error: {e}[/red]")
            return {}

        child_id = f"G{generation}_C{abs(hash(child.get('hypothesis',''))  % 10000):04d}"
        child["id"]           = child_id
        child["generation"]   = generation
        child["parents"]      = [a.get("id","?"), b.get("id","?")]
        child["operation"]    = "crossbreed"

        self._log(generation, [a.get("id","?"), b.get("id","?")], child_id, "crossbreed")
        return child

    # ── Mutation ───────────────────────────────────────────────────────────
    def mutate(self, theory: Dict, generation: int) -> Dict:
        score = theory.get("discovery_score", {}).get("final_score", 50)

        prompt = MUTATE_PROMPT.format(
            hypothesis=theory.get("hypothesis", ""),
            broken_assumption=theory.get("broken_assumption", ""),
            weakest=theory.get("discovery_score", {}).get("weakest_dimension", ""),
            direction=theory.get("discovery_score", {}).get("evolution_direction", ""),
            score=score,
        )

        try:
            # FIX: was json.loads(self._call_llm(prompt)) hitting Ollama directly.
            mutant = self.llm.call_json(prompt, temperature=0.8)
        except Exception as e:
            console.print(f"[red]Mutation error: {e}[/red]")
            return {}

        mutant_id = f"G{generation}_M{abs(hash(mutant.get('hypothesis',''))  % 10000):04d}"
        mutant["id"]         = mutant_id
        mutant["generation"] = generation
        mutant["parents"]    = [theory.get("id","?")]
        mutant["operation"]  = "mutation"

        self._log(generation, [theory.get("id","?")], mutant_id, "mutation")
        return mutant

    # ── Full Evolution Step ────────────────────────────────────────────────
    def evolve(self, population: List[Dict], generation: int,
               top_k: int = 4, n_children: int = 3) -> List[Dict]:
        console.print(f"\n[bold]═══ Evolution Engine — Generation {generation} ═══[/bold]")

        if len(population) < 2:
            console.print("  [yellow]Population too small to evolve (need ≥2)[/yellow]")
            return []

        # Sort by score
        sorted_pop = sorted(
            population,
            key=lambda h: h.get("discovery_score", {}).get("final_score", 0),
            reverse=True,
        )
        survivors = sorted_pop[:top_k]
        console.print(f"  Top {len(survivors)} survivors selected for breeding")

        next_gen = []

        # Crossbreed pairs from top survivors
        pairs = [(survivors[i], survivors[j])
                 for i in range(len(survivors))
                 for j in range(i+1, len(survivors))]

        for i, (a, b) in enumerate(pairs[:n_children]):
            console.print(
                f"  [cyan]Crossbreeding[/cyan] {a.get('id','?')} × {b.get('id','?')}..."
            )
            child = self.crossbreed(a, b, generation)
            if child:
                next_gen.append(child)

        # Mutate top survivor
        console.print(f"  [magenta]Mutating[/magenta] {survivors[0].get('id','?')}...")
        mutant = self.mutate(survivors[0], generation)
        if mutant:
            next_gen.append(mutant)

        # Deduplicate
        next_gen = self.deduplicate(next_gen + survivors)

        console.print(f"  [green]Next generation: {len(next_gen)} theories[/green]")
        return next_gen

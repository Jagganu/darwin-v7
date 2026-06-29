from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config import (
    OLLAMA_BASE_URL, OLLAMA_MODEL,
    CHROMA_DIR, PAPERS_DIR, DB_PATH, DATA_DIR,
    ARXIV_MAX_RESULTS,
    PRIMARY_PROVIDER, PRIMARY_MODEL,
    FALLBACK_PROVIDER, FALLBACK_MODEL,
    ANTHROPIC_API_KEY, NVIDIA_API_KEY,
    OPENAI_COMPAT_URL, OPENAI_COMPAT_KEY,
)
from llm_client import LLMClient

# Phase 1
from knowledge_base.arxiv_ingestion import ArXivIngestion
from knowledge_base.vector_store import VectorStore
from knowledge_base.knowledge_auditor import KnowledgeAuditor

# Pre-hypothesis
from question.generator import QuestionGenerator
from assumption.miner import AssumptionMiner
from hypothesis_generator.generator import HypothesisGenerator

# Filtering
from red_team.agent import RedTeamAgent
from debate.debate_engine import DebateEngine
from prediction.generator import PredictionGenerator
from validator.physics import PhysicsValidator

# Simulation & Reality
from simulation.engine import SimulationEngine
from reality.checker import RealityChecker
from surprise.engine import SurpriseEngine

# Scoring & Evolution
from evolution.scorer import DiscoveryScorer
from merger.engine import TheoryMerger
from evolution.engine import EvolutionEngine

# Learning
from meta.learner import MetaLearner
from meta.compressor import TheoryCompressor
from planner.research import ResearchPlanner

# Memory & Confidence
from memory.theory_graph import TheoryGraph
from memory.failure_memory import FailureMemory
from confidence.tracker import ConfidenceTracker

console = Console()


class DARWIN:
    def __init__(self):
        console.print(Panel.fit(
            "[bold cyan]DARWIN v7[/bold cyan]\n"
            "[dim]Discovery Architecture for Research,\n"
            "Worldwide Innovation and Novel theory generation[/dim]\n\n"
            "[dim]GPU/CPU Architecture Research Engine[/dim]",
            border_style="cyan",
        ))
        console.print("\n[cyan]Initializing 21 modules...[/cyan]")

        # ── Unified LLM Client ─────────────────────────────────────────────
        self.llm = LLMClient(
            primary_provider   = PRIMARY_PROVIDER,
            primary_model      = PRIMARY_MODEL,
            fallback_provider  = FALLBACK_PROVIDER,
            fallback_model     = FALLBACK_MODEL,
            ollama_url         = OLLAMA_BASE_URL,
            anthropic_api_key  = ANTHROPIC_API_KEY,
            nvidia_api_key     = NVIDIA_API_KEY,
            openai_compatible_url = OPENAI_COMPAT_URL,
            openai_compatible_key = OPENAI_COMPAT_KEY,
        )
        self.llm.test_connection()

        # ── Phase 1 ────────────────────────────────────────────────────────
        self.ingestion   = ArXivIngestion(PAPERS_DIR)
        self.vs          = VectorStore(CHROMA_DIR)
        self.auditor     = KnowledgeAuditor(self.vs, self.llm)

        # ── Pre-hypothesis ─────────────────────────────────────────────────
        self.questions   = QuestionGenerator(self.llm, DB_PATH)
        self.assumptions = AssumptionMiner(self.llm, DB_PATH)
        self.generator   = HypothesisGenerator(self.llm)

        # ── Filtering ──────────────────────────────────────────────────────
        self.red_team    = RedTeamAgent(self.llm)
        self.debate      = DebateEngine(self.llm, DB_PATH)
        self.prediction  = PredictionGenerator(self.llm, DB_PATH)
        self.physics     = PhysicsValidator(self.llm, DB_PATH)

        # ── Simulation & Reality ───────────────────────────────────────────
        self.simulation  = SimulationEngine(self.llm, DATA_DIR, DB_PATH)
        self.reality     = RealityChecker(self.llm, DB_PATH)
        self.surprise    = SurpriseEngine(self.llm, DB_PATH)

        # ── Scoring & Evolution ────────────────────────────────────────────
        self.scorer      = DiscoveryScorer(self.llm)
        self.merger      = TheoryMerger(self.llm, DB_PATH)
        self.evolution   = EvolutionEngine(self.llm, DB_PATH)

        # ── Learning ───────────────────────────────────────────────────────
        self.meta        = MetaLearner(self.llm, DB_PATH)
        self.compressor  = TheoryCompressor(self.llm, DB_PATH)
        self.planner     = ResearchPlanner(self.llm, DB_PATH)

        # ── Memory & Confidence ────────────────────────────────────────────
        self.theory_graph = TheoryGraph(DB_PATH)
        self.failure_mem  = FailureMemory(DB_PATH)
        self.confidence   = ConfidenceTracker(DB_PATH)

        console.print("[green]✓ All 21 modules ready — DARWIN v7 complete[/green]\n")

    # ── Phase 1: Knowledge Base ────────────────────────────────────────────
    def build_knowledge_base(self, force_refresh: bool = False):
        console.print("[bold]═══ Phase 1: Knowledge Base ═══[/bold]")
        papers = self.ingestion.load_cached()
        if not papers or force_refresh:
            papers = self.ingestion.ingest_all(max_per_query=ARXIV_MAX_RESULTS)
        self.vs.add_papers(papers)
        return papers

    # ── Phase 1b: Knowledge Audit ─────────────────────────────────────────
    def audit(self, search_targets: list = None):
        console.print("\n[bold]═══ Phase 1b: Knowledge Audit ═══[/bold]")
        conflicts = self.auditor.find_contradictions(max_pairs=15)
        gaps      = self.auditor.find_knowledge_gaps()

        # Feedback Loop 3: Research Planner → Knowledge Auditor
        if search_targets:
            console.print(f"  [cyan]Planner-directed targets: {len(search_targets)}[/cyan]")
            for target in search_targets[:3]:
                extra = self.vs.search(target, n_results=3)
                for r in extra:
                    gaps.append(f"Planner target: {r['metadata'].get('title','')[:60]}")

        console.print(f"[yellow]Conflicts: {len(conflicts)}  |  Gaps: {len(gaps)}[/yellow]")
        return conflicts, gaps

    # ── Full Discovery Pipeline ────────────────────────────────────────────
    def discover(self, n: int = 5, conflicts=None, gaps=None,
                 generation: int = 1, surprise_seeds: list = None):
        console.print(f"\n[bold]═══ Discovery Cycle — Generation {generation} ═══[/bold]")

        meta_bias     = self.meta.get_latest_bias() or {}
        focus         = meta_bias.get("focus", "")
        lessons       = self.failure_mem.get_lessons(10)
        context       = self.vs.search("GPU architecture innovation novel low cost", n_results=5)

        # Question Generator
        questions     = self.questions.generate(
            gaps or [], conflicts or [], focus=focus, surprise_seeds=surprise_seeds or []
        )
        enriched_gaps = self.questions.inject_into_gaps(gaps or [])

        # Assumption Miner
        assumption_result = self.assumptions.mine(questions, enriched_gaps)
        if assumption_result:
            selected = assumption_result.get("selected_assumption", "")
            enriched_gaps = [f"Break assumption: {selected}"] + enriched_gaps

        # Hypothesis Generator (Meta-Learner → Generator via enriched_gaps)
        console.print("\n[cyan]Generating hypotheses...[/cyan]")
        hypotheses = self.generator.generate_batch(n, context, enriched_gaps, conflicts or [], lessons)
        if not hypotheses:
            console.print("[red]No hypotheses generated. Check your LLM connection.[/red]")
            return None

        # Red Team
        console.print("\n[bold]══ Red Team ══[/bold]")
        after_rt, rt_rejected = self.red_team.filter_batch(hypotheses)
        for r in rt_rejected:
            # FIX: was self.failure_mem.record(r["hypothesis"], ...) which passed a
            # string into record(theory: Dict, ...). record() then called
            # theory.get("id") on a string → AttributeError. Every other rejection
            # stage passes the full hypothesis dict. Now consistent.
            self.failure_mem.record(r, r.get("attack", {}).get("fatal_flaws", ["unknown"]), "red_team")
        if not after_rt:
            return None

        # Multi-Agent Debate
        console.print("\n[bold]══ Multi-Agent Debate ══[/bold]")
        after_debate, debate_rejected = self.debate.filter_batch(after_rt)
        for h in debate_rejected:
            self.failure_mem.record(h, h.get("debate", {}).get("all_findings", ["rejected"])[:3], "debate")
        if not after_debate:
            return None

        # Prediction Generator
        console.print("\n[bold]══ Prediction Generator ══[/bold]")
        after_pred, no_pred = self.prediction.filter_batch(after_debate)
        for h in no_pred:
            self.failure_mem.record(h, ["No valid prediction"], "prediction")
        if not after_pred:
            return None

        # Physics Validator
        after_physics, phys_rejected = self.physics.filter_batch(after_pred)
        for h in phys_rejected:
            violations = [v.get("explanation","") for v in h.get("physics_validation",{}).get("violations",[])]
            self.failure_mem.record(h, violations or ["Physics violation"], "physics_validator")
        if not after_physics:
            return None

        # Simulation
        simulated, sim_skipped = self.simulation.simulate_batch(after_physics, top_n=5)

        # Reality Checker
        reality_passed, reality_abandoned = self.reality.check_batch(simulated, self.confidence)
        for h in reality_abandoned:
            self.failure_mem.record(h, ["Failed reality check"], "reality_checker")

        # Surprise Engine (Feedback Loop 1 source)
        all_simulated = reality_passed + sim_skipped
        surprises, new_seeds = self.surprise.scan_batch(all_simulated)

        # Discovery Scorer
        scored = self.scorer.score_batch(all_simulated)

        # Theory Merger
        merged, _ = self.merger.merge_mediocre(scored)
        if merged:
            scored.extend(merged)

        # Store theories
        console.print(f"\n[green]Storing {len(scored)} theories...[/green]")
        for h in scored:
            tid = self.theory_graph.add_theory(h)
            debate_score = h.get("debate", {}).get("avg_score", 0.5)
            pred_conf    = h.get("predictions", {}).get("primary_prediction", {}).get("confidence", 0.5)
            disc_score   = h.get("discovery_score", {}).get("final_score", 50) / 100
            reality_conf = h.get("reality_check", {}).get("adjusted_confidence", 0.5)
            initial_conf = (debate_score + pred_conf + disc_score + reality_conf) / 4

            self.confidence.create(tid, initial=initial_conf, known_unknowns=h.get("known_unknowns", []))
            broken = h.get("broken_assumption", "")
            if broken:
                self.assumptions.mark_broken(broken, tid)
            for pid in h.get("parents", []):
                self.theory_graph.add_relation(pid, tid,
                    "merged_into" if h.get("operation") == "merge" else "inspired")

        # Evolution
        next_gen = self.evolution.evolve(scored, generation=generation)

        # Meta-Learning
        total_stored = len(self.theory_graph.get_all())
        if self.meta.should_run(total_stored):
            self.meta.learn(cycle=total_stored)

        # Theory Compression
        laws = self.compressor.compress()

        return scored, next_gen, laws, new_seeds

    # ── Results ────────────────────────────────────────────────────────────
    def show_results(self):
        console.print("\n[bold]═══ DARWIN v7 Results ═══[/bold]")

        theories = self.theory_graph.get_all()
        if theories:
            t = Table(title="Top Surviving Theories", show_lines=True)
            t.add_column("ID",         style="cyan",   width=12)
            t.add_column("Hypothesis", style="white",  width=38)
            t.add_column("Prediction", style="green",  width=26)
            t.add_column("Conf",       style="yellow", width=6)
            for th in theories[:8]:
                pred = self.prediction.get_predictions(th["id"])
                pred_text = pred["primary"].get("claim","")[:24]+"..." if pred and pred.get("primary") else ""
                t.add_row(th["id"], th["hypothesis"][:36]+"...", pred_text, f"{th['confidence']:.2f}")
            console.print(t)

        laws = self.compressor.get_all_laws()
        if laws:
            console.print("\n[bold yellow]⭐ Discovered Architecture Laws:[/bold yellow]")
            for law in laws:
                console.print(f"  [{law['confidence']:.2f}] [bold]{law['name']}[/bold]: {law['principle'][:80]}")

        broken = [a for a in self.assumptions.get_all() if a["broken"]]
        if broken:
            console.print(f"\n[bold]Assumptions broken ({len(broken)}):[/bold]")
            for a in broken[:5]:
                console.print(f"  [green]✓[/green] {a['assumption'][:70]}")

        patterns = self.failure_mem.get_patterns()
        if patterns:
            console.print("\n[bold]Failure Breakdown:[/bold]")
            for stage, count in patterns.items():
                console.print(f"  {stage}: {count}")

    # ── Full Run ───────────────────────────────────────────────────────────
    def run(self, n_hypotheses: int = 5, generations: int = 1, refresh_kb: bool = False):
        self.build_knowledge_base(force_refresh=refresh_kb)
        conflicts, gaps = self.audit()

        all_survivors   = []
        surprise_seeds  = []
        planner_targets = []
        cycle = 1

        for gen in range(1, generations + 1):
            if gen > 1:
                # Feedback Loop 3: Research Planner → Knowledge Auditor
                conflicts, gaps = self.audit(search_targets=planner_targets)

            result = self.discover(
                n=n_hypotheses, conflicts=conflicts, gaps=gaps,
                generation=gen, surprise_seeds=surprise_seeds,
            )
            if result:
                scored, next_gen, laws, new_seeds = result
                all_survivors.extend(scored)
                surprise_seeds = new_seeds

                # Research Planner directs next cycle
                plan = self.planner.plan(cycle=cycle, gaps=gaps)
                if plan:
                    planner_targets = plan.get("auditor_search_targets", [])
                cycle += 1

        self.show_results()
        console.print("\n[bold green]✓ DARWIN v7 complete[/bold green]")
        return all_survivors


if __name__ == "__main__":
    import sys
    n       = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    gens    = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    refresh = "--refresh" in sys.argv
    DARWIN().run(n_hypotheses=n, generations=gens, refresh_kb=refresh)

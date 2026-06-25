import json
import sqlite3
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from rich.console import Console

from .estimator import FastEstimator
from .config_generator import SimConfigGenerator

console = Console()


class SimulationEngine:
    """
    Coordinates:
      1. Fast LLM estimator  (always runs — no tools required)
      2. Config generation   (gem5 / GPGPU-Sim configs written to disk)
      3. Actual simulation   (optional — requires gem5/GPGPU-Sim installed)
      4. Result parsing      (extracts metrics from sim output)

    On machines without gem5/GPGPU-Sim, fast estimator results are used.
    When simulators are available, set GEM5_PATH/GPGPUSIM_PATH in config.
    """

    def __init__(self, ollama_url: str, model: str, data_dir: Path, db_path: Path,
                 gem5_path: str = "", gpgpusim_path: str = ""):
        self.estimator    = FastEstimator(ollama_url, model)
        self.config_gen   = SimConfigGenerator(ollama_url, model, data_dir)
        self.db_path      = db_path
        self.gem5_path    = gem5_path        # e.g. "/usr/local/gem5/build/X86/gem5.opt"
        self.gpgpusim_path = gpgpusim_path  # e.g. "/usr/local/gpgpu-sim/bin/gpgpusim"
        self._init_db()

    def _init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS simulation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    theory_id TEXT,
                    sim_type TEXT,
                    result_json TEXT,
                    prediction_hit REAL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def _save_result(self, theory_id: str, sim_type: str, result: Dict, hit: float):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO simulation_results (theory_id, sim_type, result_json, prediction_hit) "
                "VALUES (?, ?, ?, ?)",
                (theory_id, sim_type, json.dumps(result), hit),
            )
            conn.commit()

    def _run_gem5(self, config_path: Path) -> Optional[Dict]:
        """Run gem5 if installed. Returns parsed stats or None."""
        if not self.gem5_path or not Path(self.gem5_path).exists():
            return None
        try:
            result = subprocess.run(
                [self.gem5_path, str(config_path)],
                capture_output=True, text=True, timeout=3600
            )
            return self._parse_gem5_output(result.stdout)
        except Exception as e:
            console.print(f"  [red]gem5 error: {e}[/red]")
            return None

    def _parse_gem5_output(self, output: str) -> Dict:
        """Extract key metrics from gem5 stats.txt output."""
        metrics = {}
        for line in output.split("\n"):
            if "sim_seconds" in line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        metrics["sim_seconds"] = float(parts[1])
                    except ValueError:
                        pass
            if "system.cpu.ipc" in line:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        metrics["ipc"] = float(parts[1])
                    except ValueError:
                        pass
        return metrics

    def _calculate_prediction_hit(self, estimate: Dict, prediction: Dict) -> float:
        """
        How well did the simulation match the prediction?
        Returns 0.0 (miss) to 1.0 (perfect hit).
        """
        expected = prediction.get("expected_value", "")
        estimated_delta = estimate.get("estimated_performance_delta", "")

        # Extract numbers for comparison
        import re
        exp_nums = re.findall(r"[-+]?\d+\.?\d*", expected)
        est_nums = re.findall(r"[-+]?\d+\.?\d*", estimated_delta)

        if exp_nums and est_nums:
            try:
                exp_val = float(exp_nums[0])
                est_val = float(est_nums[0])
                if exp_val == 0:
                    return 0.5
                ratio = min(abs(est_val), abs(exp_val)) / max(abs(est_val), abs(exp_val))
                return ratio
            except (ValueError, ZeroDivisionError):
                pass

        return estimate.get("confidence_in_estimate", 0.5)

    def simulate(self, hypothesis: Dict) -> Dict:
        tid = hypothesis.get("id", "unknown")
        pred = hypothesis.get("predictions", {}).get("primary_prediction", {})

        console.print(f"\n  [bold cyan]Simulating: {tid}[/bold cyan]")

        # Step 1: Always run fast estimator
        estimate = self.estimator.estimate(hypothesis)
        if not estimate:
            return {"theory_id": tid, "status": "error", "sim_type": "none"}

        sim_result = estimate.copy()
        sim_type   = "fast_estimate"

        # Step 2: Generate sim configs (regardless of whether simulators are installed)
        configs = self.config_gen.generate_all(hypothesis)

        # Step 3: Try actual gem5 simulation if available
        if self.gem5_path and configs.get("gem5"):
            gem5_result = self._run_gem5(Path(configs["gem5"]))
            if gem5_result:
                sim_result.update(gem5_result)
                sim_type = "gem5"
                console.print("  [green]✓ gem5 simulation complete[/green]")
        else:
            console.print("  [dim]gem5 not installed — using fast estimate[/dim]")

        # Step 4: Calculate prediction hit rate
        hit = self._calculate_prediction_hit(sim_result, pred)
        sim_result["prediction_hit"] = hit
        sim_result["theory_id"]      = tid
        sim_result["sim_type"]       = sim_type
        sim_result["configs"]        = configs

        # Step 5: Save
        self._save_result(tid, sim_type, sim_result, hit)

        score = sim_result.get("quick_score", 0)
        console.print(
            f"  score=[yellow]{score}[/yellow]  "
            f"prediction_hit=[cyan]{hit:.2f}[/cyan]  "
            f"[{sim_type}]"
        )
        return sim_result

    def simulate_batch(self, hypotheses: List[Dict], top_n: int = 5) -> Tuple[List[Dict], List[Dict]]:
        console.print(f"\n[bold]═══ Simulation Engine ═══[/bold]")

        # Resource Manager: fast filter first
        to_simulate, skipped = self.estimator.rank_and_filter(hypotheses, top_n=top_n)

        results = []
        for h in to_simulate:
            sim = self.simulate(h)
            h["simulation"] = sim
            results.append(h)

        # Mark skipped with their fast estimate
        for h in skipped:
            h["simulation"] = h.get("fast_estimate", {"status": "skipped"})

        return results, skipped

    def get_results(self, theory_id: str) -> Optional[Dict]:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT sim_type, result_json, prediction_hit FROM simulation_results "
                "WHERE theory_id = ? ORDER BY created_at DESC LIMIT 1",
                (theory_id,),
            ).fetchone()
        if row:
            return {
                "sim_type": row[0],
                "result": json.loads(row[1]),
                "prediction_hit": row[2],
            }
        return None

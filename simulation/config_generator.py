import json
import requests
from pathlib import Path
from typing import Dict, Optional
from rich.console import Console

console = Console()

GEM5_PROMPT = """You are a gem5 GPU/CPU architecture simulator expert.

Given this novel architecture theory, generate a gem5 Python configuration
that would test its key prediction. Output ONLY valid Python gem5 config code.

Theory:
  Hypothesis : {hypothesis}
  Mechanism  : {mechanism}
  Prediction : {prediction}
  Test condition: {test_condition}

Generate a minimal but realistic gem5 config that:
1. Models the key architectural change described
2. Will produce data to confirm or deny the prediction
3. Uses gem5's SE (syscall emulation) mode for speed
4. Runs a representative GPU/compute workload

Output ONLY Python code, no explanation:"""

GPGPU_PROMPT = """You are a GPGPU-Sim configuration expert.

Given this novel GPU architecture theory, generate a gpgpusim.config
that would test its prediction.

Theory:
  Hypothesis : {hypothesis}
  Mechanism  : {mechanism}
  Prediction : {prediction}

Generate a realistic gpgpusim.config. Output ONLY the config file content:"""


class SimConfigGenerator:
    """
    Generates simulation config files for gem5 and GPGPU-Sim.
    Actual simulation requires those tools installed separately.
    See README for installation instructions.
    """

    def __init__(self, ollama_url: str, model: str, output_dir: Path):
        self.ollama_url = ollama_url
        self.model = model
        self.output_dir = output_dir / "sim_configs"
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _call_llm(self, prompt: str) -> str:
        resp = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2},
            },
            timeout=180,
        )
        return resp.json()["response"]

    def generate_gem5_config(self, hypothesis: Dict) -> Optional[Path]:
        tid = hypothesis.get("id", "unknown")
        pred = hypothesis.get("predictions", {}).get("primary_prediction", {})

        prompt = GEM5_PROMPT.format(
            hypothesis=hypothesis.get("hypothesis", ""),
            mechanism=hypothesis.get("mechanism", ""),
            prediction=pred.get("claim", ""),
            test_condition=pred.get("test_condition", ""),
        )

        try:
            code = self._call_llm(prompt)
            # Strip markdown fences if present
            code = code.replace("```python", "").replace("```", "").strip()

            path = self.output_dir / f"{tid}_gem5.py"
            path.write_text(code)
            console.print(f"  [green]✓ gem5 config → {path.name}[/green]")
            return path
        except Exception as e:
            console.print(f"  [red]gem5 config error: {e}[/red]")
            return None

    def generate_gpgpu_config(self, hypothesis: Dict) -> Optional[Path]:
        tid = hypothesis.get("id", "unknown")
        pred = hypothesis.get("predictions", {}).get("primary_prediction", {})

        prompt = GPGPU_PROMPT.format(
            hypothesis=hypothesis.get("hypothesis", ""),
            mechanism=hypothesis.get("mechanism", ""),
            prediction=pred.get("claim", ""),
        )

        try:
            cfg = self._call_llm(prompt)
            cfg = cfg.replace("```", "").strip()

            path = self.output_dir / f"{tid}_gpgpusim.config"
            path.write_text(cfg)
            console.print(f"  [green]✓ GPGPU-Sim config → {path.name}[/green]")
            return path
        except Exception as e:
            console.print(f"  [red]GPGPU config error: {e}[/red]")
            return None

    def generate_all(self, hypothesis: Dict) -> Dict:
        tid = hypothesis.get("id", "unknown")
        console.print(f"  [cyan]Generating sim configs for {tid}...[/cyan]")
        return {
            "gem5": str(self.generate_gem5_config(hypothesis) or ""),
            "gpgpu": str(self.generate_gpgpu_config(hypothesis) or ""),
        }

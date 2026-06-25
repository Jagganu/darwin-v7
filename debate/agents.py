import json
import requests
from typing import Dict
from rich.console import Console

console = Console()

# ── Agent Prompts ──────────────────────────────────────────────────────────────

AGENT_PROMPTS = {
    "physics": """You are the Physics Agent in a GPU/CPU architecture research debate.
Your domain: thermodynamics, quantum mechanics, electromagnetism, information theory.

Hypothesis under review:
  Statement    : {hypothesis}
  Mechanism    : {mechanism}
  Prediction   : {prediction}

Evaluate ONLY from a physics perspective:
- Does it violate thermodynamics (energy, entropy, heat dissipation)?
- Does it violate Maxwell's equations or signal integrity laws?
- Does it approach or exceed fundamental physical limits (Landauer limit, etc)?
- Is the claimed mechanism physically realizable?

Respond ONLY in JSON:
{{
  "verdict": "PASS | CONCERN | REJECT",
  "domain": "physics",
  "score": 0.0 to 1.0,
  "findings": ["finding 1", "finding 2"],
  "fatal": true or false,
  "recommendation": "one sentence"
}}""",

    "manufacturing": """You are the Manufacturing Agent in a GPU/CPU architecture research debate.
Your domain: semiconductor fabrication, lithography, yield, TSMC/Intel process nodes, packaging.

Hypothesis under review:
  Statement    : {hypothesis}
  Mechanism    : {mechanism}
  Prediction   : {prediction}

Evaluate ONLY from a manufacturing perspective:
- Can it be fabricated at current process nodes (3nm, 5nm, 7nm)?
- What are the yield implications?
- Does it require materials not yet available at scale?
- What are the packaging challenges?
- Is it compatible with existing EDA toolchains?

Respond ONLY in JSON:
{{
  "verdict": "PASS | CONCERN | REJECT",
  "domain": "manufacturing",
  "score": 0.0 to 1.0,
  "findings": ["finding 1", "finding 2"],
  "fatal": true or false,
  "recommendation": "one sentence"
}}""",

    "cost": """You are the Cost Agent in a GPU/CPU architecture research debate.
Your domain: semiconductor economics, CAPEX/OPEX, wafer costs, HBM pricing, market viability.

Hypothesis under review:
  Statement    : {hypothesis}
  Mechanism    : {mechanism}
  Prediction   : {prediction}

Evaluate ONLY from a cost perspective:
- What is the estimated die cost vs current designs?
- Does it require expensive new materials or processes?
- What is the R&D cost to productize this?
- Is the cost/performance ratio better than existing solutions?
- Would this be manufacturable at a price point the market would accept?

Respond ONLY in JSON:
{{
  "verdict": "PASS | CONCERN | REJECT",
  "domain": "cost",
  "score": 0.0 to 1.0,
  "findings": ["finding 1", "finding 2"],
  "fatal": true or false,
  "recommendation": "one sentence"
}}""",

    "performance": """You are the Performance Agent in a GPU/CPU architecture research debate.
Your domain: compute throughput, memory bandwidth, latency, IPC, FLOPS/watt, Amdahl's law.

Hypothesis under review:
  Statement    : {hypothesis}
  Mechanism    : {mechanism}
  Prediction   : {prediction}

Evaluate ONLY from a performance perspective:
- Does the predicted performance gain logically follow from the mechanism?
- What bottlenecks does it introduce or remove?
- How does it perform across different workloads (AI, graphics, HPC)?
- Does Amdahl's law limit the claimed speedup?
- What is the realistic performance envelope?

Respond ONLY in JSON:
{{
  "verdict": "PASS | CONCERN | REJECT",
  "domain": "performance",
  "score": 0.0 to 1.0,
  "findings": ["finding 1", "finding 2"],
  "fatal": true or false,
  "recommendation": "one sentence"
}}""",

    "reliability": """You are the Reliability Agent in a GPU/CPU architecture research debate.
Your domain: electromigration, aging, MTBF, thermal cycling, ECC, fault tolerance.

Hypothesis under review:
  Statement    : {hypothesis}
  Mechanism    : {mechanism}
  Prediction   : {prediction}

Evaluate ONLY from a reliability perspective:
- Does the design introduce new failure modes?
- What are the long-term aging effects (electromigration, oxide wear)?
- How does it behave under sustained thermal load?
- Does it require new error correction schemes?
- What is the expected MTBF vs current designs?

Respond ONLY in JSON:
{{
  "verdict": "PASS | CONCERN | REJECT",
  "domain": "reliability",
  "score": 0.0 to 1.0,
  "findings": ["finding 1", "finding 2"],
  "fatal": true or false,
  "recommendation": "one sentence"
}}""",

    "environmental": """You are the Environmental Agent in a GPU/CPU architecture research debate.
Your domain: power consumption, TDP, carbon footprint, rare earth materials, e-waste, cooling.

Hypothesis under review:
  Statement    : {hypothesis}
  Mechanism    : {mechanism}
  Prediction   : {prediction}

Evaluate ONLY from an environmental perspective:
- What is the estimated TDP vs current designs?
- Does it use rare earth materials or conflict minerals?
- What are the cooling requirements?
- What is the manufacturing carbon footprint?
- Is it better or worse than current designs for data center PUE?

Respond ONLY in JSON:
{{
  "verdict": "PASS | CONCERN | REJECT",
  "domain": "environmental",
  "score": 0.0 to 1.0,
  "findings": ["finding 1", "finding 2"],
  "fatal": true or false,
  "recommendation": "one sentence"
}}""",
}

AGENT_COLORS = {
    "physics":       "blue",
    "manufacturing": "magenta",
    "cost":          "yellow",
    "performance":   "cyan",
    "reliability":   "red",
    "environmental": "green",
}


class DebateAgent:
    def __init__(self, name: str, ollama_url: str, model: str):
        self.name = name
        self.ollama_url = ollama_url
        self.model = model
        self.prompt_template = AGENT_PROMPTS[name]
        self.color = AGENT_COLORS[name]

    def evaluate(self, hypothesis: Dict) -> Dict:
        pred = hypothesis.get("prediction", {})
        prompt = self.prompt_template.format(
            hypothesis=hypothesis.get("hypothesis", ""),
            mechanism=hypothesis.get("mechanism", ""),
            prediction=pred.get("claim", ""),
        )

        try:
            resp = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "format": "json",
                    "options": {"temperature": 0.3},
                },
                timeout=120,
            )
            result = json.loads(resp.json()["response"])
        except Exception as e:
            console.print(f"[red]Agent {self.name} error: {e}[/red]")
            result = {
                "verdict": "CONCERN",
                "domain": self.name,
                "score": 0.5,
                "findings": [f"Agent error: {str(e)[:60]}"],
                "fatal": False,
                "recommendation": "Could not evaluate — treat as uncertain",
            }

        result["domain"] = self.name  # ensure domain is set
        verdict = result.get("verdict", "CONCERN")
        color = "red" if verdict == "REJECT" else "yellow" if verdict == "CONCERN" else "green"
        console.print(
            f"  [{self.color}]{self.name.upper():14}[/{self.color}] "
            f"[{color}]{verdict}[/{color}] "
            f"score={result.get('score', 0):.2f} — "
            f"{result.get('recommendation', '')[:55]}"
        )
        return result

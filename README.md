# DARWIN v7
**Discovery Architecture for Research, Worldwide Innovation and Novel theory generation**

A self-improving GPU/CPU architecture research engine.

---

## Architecture

```
Knowledge Base → Knowledge Auditor → Knowledge Gap Detector
→ Question Generator → Assumption Miner → Hypothesis Generator
→ Red-Team Agent → Multi-Agent Debate → Prediction Generator
→ Uncertainty Engine → Physics Validator → Resource Manager
→ Simulation Engine → Reality Checker → Surprise Engine
→ Confidence Tracker → Discovery Scorer → Theory Merger
→ Diversity Manager → Evolution Engine → Theory Graph
→ Meta-Learner → Theory Compression → Research Planner
→ Failure Memory → Novel Theory Output
```

**Phase 1 (this repo):** Modules 1–5
- Knowledge Base, Hypothesis Generator, Red-Team Agent, Theory Graph, Confidence Tracker

---

## Setup

### Requirements
- Python 3.10+
- [Ollama](https://ollama.ai) running locally
- At least 8GB RAM

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Pull a model in Ollama
```bash
ollama pull qwen2.5:14b
# or any model you have:
# ollama pull llama3.1:8b
```

### 3. Set your model (optional)
```bash
export OLLAMA_MODEL=qwen2.5:14b
```

### 4. Run DARWIN
```bash
python main.py          # generates 5 hypotheses
python main.py 10       # generates 10 hypotheses
python main.py 5 --refresh  # re-fetch papers from ArXiv
```

---

## What Happens on First Run

1. Downloads ~200 GPU architecture papers from ArXiv (~2 min)
2. Embeds them into ChromaDB vector store (~5 min, CPU)
3. Audits for contradictions and knowledge gaps
4. Generates N hypotheses via Ollama
5. Red Team Agent attacks each one
6. Survivors stored in Theory Graph + SQLite
7. Results displayed

---

## Data Files

```
darwin/data/
├── papers/papers.json     ← cached ArXiv papers
├── chroma/                ← vector embeddings
└── darwin.db              ← theories, failures, confidence
```

---

## Configuration

Edit `config.py`:

```python
OLLAMA_MODEL = "qwen2.5:14b"   # change to your model
ARXIV_MAX_RESULTS = 20          # papers per query
```

---

## Roadmap

- [x] Phase 1: Knowledge Base + Hypothesis Generator + Red Team + Memory
- [ ] Phase 2: Multi-Agent Debate (6 specialized agents)
- [ ] Phase 3: Prediction Generator + Physics Validator
- [ ] Phase 4: Simulation Engine (gem5, GPGPU-Sim)
- [ ] Phase 5: Evolution Engine + Theory Compression
- [ ] Phase 6: Meta-Learner + Surprise Engine + Reality Checker

---

*Designed by Jagan. Built to discover what doesn't exist yet.*

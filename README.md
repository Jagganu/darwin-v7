# DARWIN v7
**Discovery Architecture for Research, Worldwide Innovation and Novel theory generation**

A self-improving GPU/CPU architecture research engine.

---

## Architecture

```
Knowledge Base (IEEE papers, patents, benchmarks, hardware data)
      ↓
Knowledge Auditor → finds contradictions, anomalies
      ↓                    ↑
Knowledge Gap Detector     │ Research Planner feeds back here
      ↓                    │ (targeted contradiction hunt)
Question Generator ←───────┘
      ↓                    ↑
Assumption Miner           │ Surprise Engine feeds back here
      ↓                    │ (new research branch launched)
Hypothesis Generator ←─────┘
      ↓                    ↑
Red-Team Agent             │ Meta-Learner feeds back here
      ↓                    │ (biases toward what works)
Multi-Agent Debate (6 agents)
      ↓
Prediction Generator → no prediction = theory dies
      ↓
Uncertainty Engine → maps known unknowns
      ↓
Physics Validator
      ↓
Resource Manager → 1000 → 50 → 5 → deep sim
      ↓
Simulation Engine (gem5, GPGPU-Sim, McPAT, HotSpot)
      ↓
Reality Checker → cross-reference real hardware data
      ↓
Surprise Engine ───────────────────────→ Question Generator
      ↓
Confidence Tracker → updated with real evidence
      ↓
Discovery Scorer (novelty/performance/cost/manufacturability)
      ↓
Theory Merger + Diversity Manager
      ↓
Evolution Engine
      ↓
Theory Graph → stores lineage and causality
      ↓
Meta-Learner ──────────────────────────→ Hypothesis Generator
      ↓
Theory Compression → many theories → one principle → new law
      ↓
Research Planner ──────────────────────→ Knowledge Auditor
      ↓
Failure Memory DB
      ↓
Novel Theory Output
```

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

Edit `config.py` or use environment variables:

```bash
# Ollama (default)
export OLLAMA_MODEL=qwen2.5:14b

# Claude API
export ANTHROPIC_API_KEY=sk-ant-...

# NVIDIA NIM
export NVIDIA_API_KEY=...
export DARWIN_PRIMARY=nvidia
export DARWIN_MODEL=meta/llama-3.1-70b-instruct

# Fallback provider
export DARWIN_FALLBACK=anthropic
export DARWIN_FB_MODEL=claude-sonnet-4-6
```

---

## Build Status

| Phase | Planned | Built | Status |
|---|---|---|---|
| 1 | Knowledge Base + Hypothesis Generator + Red Team + Memory | ArXiv ingestion, ChromaDB, Knowledge Auditor, Hypothesis Generator, Red Team, Theory Graph, Failure Memory, Confidence Tracker | ✅ |
| 2 | Multi-Agent Debate (6 agents) | Physics, Manufacturing, Cost, Performance, Reliability, Environmental agents + Debate Engine | ✅ |
| 3 | Prediction Generator + Physics Validator | Both built + falsification enforcement + vague prediction rewriter | ✅ |
| 4 | Simulation Engine (gem5, GPGPU-Sim) | Fast LLM estimator + gem5/GPGPU-Sim config generator + Resource Manager | ⚠️ Partial |
| 5 | Evolution Engine + Theory Compression | Evolution Engine + Theory Compressor + Discovery Scorer + Theory Merger + Diversity Manager | ✅ |
| 6 | Meta-Learner + Surprise Engine + Reality Checker | All 3 built | ✅ |

**Bonus modules not in original plan:**
- Question Generator
- Assumption Miner
- Research Planner
- LLM Client (Ollama + Claude API + NVIDIA NIM)

> **Phase 4 note:** gem5 and GPGPU-Sim configs are generated and written to disk — but actually running them requires those tools installed on your machine. The fast LLM estimator runs instead. Real simulation needs a proper Linux server with gem5 compiled.

**5.5 out of 6 fully complete.** Phase 4 needs gem5 installed to be 100%.

---

*Designed by Jagan. Built to discover what doesn't exist yet.*

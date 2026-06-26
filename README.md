# Project ICDS — Intelligent Candidate Discovery System

This repository contains the code and artifacts for Project ICDS, built for the Redrob Hackathon v4. 
The system ranks a 100,000-candidate pool against a founding Senior AI Engineer job description using a hybrid pipeline of precomputed semantic embeddings and multi-layered behavioral and fit heuristics.

---

## Setup & Installation

Ensure you have Python 3.10+ installed. Install the dependencies listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

---

## Pipeline Execution

The pipeline operates in two phases: offline pre-computation and local sandboxed ranking.

### Phase 1: Pre-Computation (Offline)

To precompute the honeypots list, query embedding, and candidate pool embeddings:

1. **Honeypot Scanning**: Identify profiles with impossible dates, duration discrepancies, or fraudulent skills.
   ```bash
   python scripts/scan_honeypots.py
   ```
2. **Query Encoding**: Generate embedding for the target JD query using a local ONNX session of `all-MiniLM-L6-v2`.
   ```bash
   python scripts/precompute_query.py
   ```
3. **Candidate Encoding**: Encode all 100,000 candidates' concise representations (`title | headline | top_skills`).
   ```bash
   python scripts/precompute_embeddings.py
   ```

*(Precomputed numpy embeddings are stored under `data/candidate_embeddings.npy` and `data/query_embedding.npy`)*

---

### Phase 2: Ranking & Submission Generation (Sandboxed)

The final ranking step runs completely offline, uses CPU only, and completes in under 30 seconds.

Run the following command at the repository root to produce the validated submission CSV:

```bash
python rank.py --candidates ./data/candidates.jsonl --out ./submissions/submission.csv
```

---

## Verification

To verify that the output meets all format and scoring constraints, run:

```bash
python validate_submission.py submissions/submission.csv
```

---

## Methodology Summary

Our approach relies on a multi-stage filtering and behavioral calibration pipeline:

- **Layer 0 (L0 Hard Filters)**: Drops honeypots (detected dynamically via date/experience inconsistencies), candidates with purely CV/Speech/Robotics expertise without NLP/IR exposure, and pure research candidates lacking any production/ship/infrastructure terms in their career history.
- **Layer 1 (L1 Fit Scoring)**: Computes cosine similarity of candidate representation against the JD query using local ONNX `all-MiniLM-L6-v2`. Penalizes "job-hoppers/title-chasers" (average company tenure < 18 months) and consulting-only careers (without product company exposure). Penalizes recent LLM-only experience lacking pre-LLM ML foundation and non-coding senior roles.
- **Layer 2 (L2 Behavioral Calibration)**: Boosts candidates with notice periods $\le 30$ days. Penalizes low recruiter response rates ($< 30\%$) and platform inactivity ($> 6$ months).
- **Layer 3 (L3 Tie-Breaking & Reasoning)**: Resolves equal scores (rounded to 4 decimal places) by sorting candidate ID ascending. Generates fact-grounded, non-hallucinated, tone-consistent, and stylistically varied justifications for the top 100 picks.

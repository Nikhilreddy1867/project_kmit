# AI Governance Platform — Adult / Census Income

An end-to-end, **locally runnable AI governance platform** built around a single real
machine-learning model. It does not just train a classifier; it audits one — for
performance, fairness and explainability — then records a formal governance
decision, exposes all of it through an API, a dashboard and a deterministic agent
layer, and finally seals the evidence with cryptographic checksums in an audit
registry.

The model under audit predicts whether a person's annual income exceeds **\$50,000**
from the **UCI Adult / Census Income** dataset (`id=2`), a 1994 US Census extract.
That model exists **to be governed, not deployed** — and the platform's conclusion,
reached from its own evidence, is that it must never be used for a real decision.

Verified end-to-end on **Windows 11 Pro · Python 3.14.6 (64-bit)** · Intel i7 · CPU only.
**113 tests passing.**

---

## 0. Project description

### 0.1 What problem this addresses

Most ML projects stop at a metrics table. That table is where the hard governance
questions *begin*:

- Accuracy is **0.8782** — but a model that predicts "everyone earns ≤\$50K" scores
  **0.7607** on this data. How much of that number is real?
- The model is accurate overall — but it selects women at **31.5%** the rate it
  selects men. Is that the model, or the 1994 labour market it learned from?
- `sex` ranks **10th of 14** features by importance. Does that mean the model is fair?
  (It does not — and this platform demonstrates precisely why.)
- Who decided this model may be used, on what evidence, and would anyone notice if
  that evidence changed after the fact?

This project answers those questions with artefacts rather than assertions, and wires
them into software a reviewer can actually interrogate.

### 0.2 The eight phases

| # | Phase | Deliverable |
|---|---|---|
| **1** | Baseline modelling | Leakage-free pipelines; LogReg / Random Forest / XGBoost compared on one stratified split |
| **2** | Fairness audit | Group metrics by `sex` and `race`, disparate impact, four-fifths screening, Wilson intervals |
| **3** | Explainability audit | Permutation importance on original features + exact TreeSHAP local explanations |
| **4** | Governance assessment | Model card, 12-entry risk register, signed-off decision record |
| **5** | Read-only API | FastAPI service over the audit evidence, with Swagger |
| **6** | Dashboard | Streamlit UI consuming only the API |
| **7** | Agent layer | Four deterministic, rule-based governance agents + orchestrator |
| **8** | Audit registry | SQLite registry with SHA-256 evidence integrity verification |

Each phase treats the previous phases' outputs as **read-only evidence**. Nothing
downstream ever rewrites what it audits — a property enforced by tests, not habit.

### 0.3 Architecture at a glance

```text
   src/            →  data/ models/ predictions/ results/     ← IMMUTABLE EVIDENCE
   (Phases 1–4)                    │
                                   │ read-only
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
      app/services/         app/agents/          app/registry/
      (Phase 5 API)         (Phase 7 agents)     (Phase 8 registry)
              │                    │                    │
              └──────────── HTTP (JSON) ────────────────┘
                                   ▼
                          dashboard/ (Phase 6)
                          7 pages, reads no files
```

Data flows **one way only**. The API, agents and dashboard can read the evidence;
none of them can write to it. The single component that writes anything at all is the
Phase 8 registry, and it writes to exactly one gitignored SQLite file under
`runtime/`.

### 0.4 Technology

| Layer | Stack |
|---|---|
| ML | scikit-learn 1.9 · XGBoost 3.4 · pandas 3.0 · numpy 2.5 |
| Explainability | scikit-learn permutation importance · XGBoost native TreeSHAP (`pred_contribs`) |
| API | FastAPI 0.141 · Pydantic 2.13 · uvicorn |
| Dashboard | Streamlit 1.61 · Plotly 6.9 |
| Registry | SQLite (stdlib `sqlite3`) · `hashlib` SHA-256 |
| Tests | pytest 9.1 · `fastapi.testclient` · `streamlit.testing.v1.AppTest` |

No cloud services, no API keys, no containers — everything runs on one laptop.

### 0.5 Headline findings

**Performance.** XGBoost leads on every metric: ROC-AUC **0.9299**, F1 **0.7239**,
accuracy **0.8782** on 9,769 held-out rows. It also produces **778 false negatives
against 412 false positives** — the dominant error is failing to identify real high
earners, and that error is not evenly distributed.

**Fairness.** Every model selects women at roughly one third the male rate
(disparate impact **0.299 / 0.307 / 0.315**). Three of four non-reference race groups
fall below the 0.80 screening threshold. Yet precision is nearly equal across sex
(0.792 vs 0.783), and the disadvantaged groups receive *fewer* false positives — a
live demonstration that fairness metrics are **mutually incompatible**, not merely
inconvenient.

**Explainability.** `sex` ranks 10/14 and `race` 11/14 in importance, while
`marital-status` ranks 1st and `relationship` — whose categories are literally
`Husband` and `Wife` — outranks `sex` itself. **Low protected-attribute importance is
not evidence of fairness.** An audit that had run explainability alone would have
concluded the opposite of the truth.

**Governance decision.** ✅ *Conditionally approved for research and education only* ·
⛔ *Blocked from real-world deployment.* Five of twelve risks are Critical. The
strongest ground is not the disparity but obsolescence: a 1994 \$50K threshold is
roughly \$105–110K today, so the model is miscalibrated against any present-day
population by construction.

### 0.6 What this project deliberately does not claim

- **No finding of discrimination.** The audits measure output differences. A legal
  determination would require a deployment context, a jurisdiction and a standard —
  none of which exist here.
- **No causal claim.** Permutation importance and SHAP describe how a fitted function
  responds to its inputs on one dataset. Neither is a causal estimand.
- **No separation of model bias from label bias.** Group base rates differ in the
  1994 labels themselves (30.4% vs 11.2% by sex). No method used here can tell
  "the model is unfair" apart from "the labour market it recorded was unequal".
- **The four-fifths rule is a screening trigger**, not a verdict or a statistical test.

These limits are carried in the schemas, the API responses, the agent findings and
the dashboard — so a consumer cannot read the numbers without also receiving the
caveats.

### 0.7 Engineering properties worth noting

- **Leakage prevention is structural.** The split precedes all fitting; every learned
  transformation lives inside a `Pipeline`. `pipeline.fit(X_train, y_train)` is the
  only `fit` call in the codebase.
- **Deterministic throughout.** `random_state=42` everywhere; the agent layer has no
  timestamps or randomness, so the same evidence yields byte-identical output.
- **No recomputation downstream.** The API serves audit numbers verbatim — including
  reading CSVs with `float_precision="round_trip"`, because pandas' default parser
  lands one ULP off and would have made the API cite numbers the audit never wrote.
- **Absence is reported, never imputed.** Random Forest has no explainability audit,
  so that endpoint returns 404 and the agent returns `status: unavailable`. No
  estimated stand-in is ever produced.
- **Evidence integrity is verifiable.** 34 artefacts are checksummed; the registry
  re-verifies them on demand and reports verified / missing / changed.

### 0.8 Quick start

```powershell
cd C:\Users\nreddy\Downloads\project_KMIT
py -3.14 -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Phases 1–3: build the evidence (~1 minute total)
python src\train.py
python src\fairness_audit.py
python src\explainability_audit.py

# Phase 8: seal it in the registry
python -m app.registry.cli register

# Phases 5-7: API   (window 1)
uvicorn app.main:app --reload --port 8000
# Phase 6: dashboard (window 2)
streamlit run dashboard\streamlit_app.py

# ...or both halves in ONE process (also how it deploys - see 3e)
streamlit run streamlit_app.py
```

| Service | URL |
|---|---|
| Dashboard | **http://localhost:8501** |
| Swagger API docs | **http://127.0.0.1:8000/docs** |

Phase 4 needs no command — the model card, risk register and decision record are
committed documents under `results/governance/`.

Full test suite: `pytest -q` → **113 passed** (30 API · 28 agents · 22 registry ·
22 dashboard · 11 deployment). The dashboard tests need the API running and skip
cleanly otherwise; the deployment tests need no server at all.

### 0.9 Document map

| Section | Contents |
|---|---|
| §1–§3 | Project structure, Windows setup, running the ML pipeline |
| §3a | Phase 5 — API endpoints, error semantics |
| §3b | Phase 6 — dashboard, two-window setup, failure states |
| §3c | Phase 7 — agents, architecture diagram, hard constraints |
| §3d | Phase 8 — registry, idempotency, integrity semantics |
| §3e | **Deploying to Streamlit Community Cloud** (single-process entry point) |
| §4–§5 | Results table and full methodology |
| §6–§7 | Assumptions, deliberate non-choices, troubleshooting |

Deeper governance reading lives in the artefacts themselves:
[`model_card.md`](results/governance/model_card.md),
[`governance_summary.md`](results/governance/governance_summary.md) (the decision
record), [`fairness_report.md`](results/fairness/fairness_report.md) and
[`explainability_report.md`](results/explainability/explainability_report.md).

---

## 1. Project structure

```text
project_KMIT/
├── data/raw/                            # IMMUTABLE EVIDENCE — never rewritten
│   ├── adult_raw.csv                    # untouched snapshot from UCI (48,842 × 15)
│   └── adult_metadata.txt               # UCI metadata + variable dictionary
├── models/                              # Phase 1 — full fitted Pipelines
│   ├── xgboost_pipeline.joblib          # selected baseline (preprocessing + model)
│   ├── logistic_regression_pipeline.joblib
│   └── random_forest_pipeline.joblib    # ~100 MB, gitignored; regenerate via train.py
├── predictions/                         # Phase 1 — test predictions + sex/race
│   └── <model>_test_predictions.csv
├── results/                             # all audit outputs
│   ├── model_metrics.csv                # Phase 1 comparison table
│   ├── model_comparison.png
│   ├── confusion_matrix_<model>.png
│   ├── classification_report_<model>.txt
│   ├── fairness/                        # Phase 2
│   │   ├── fairness_metrics_by_group.csv
│   │   ├── fairness_summary.csv
│   │   ├── fairness_report.md
│   │   └── chart_*.png
│   ├── explainability/                  # Phase 3
│   │   ├── global_feature_importance.csv|png
│   │   ├── local_explanations.csv
│   │   ├── logistic_regression_coefficients.csv
│   │   └── explainability_report.md
│   └── governance/                      # Phase 4
│       ├── model_card.md
│       ├── governance_risk_register.csv
│       └── governance_summary.md        # ← the decision record
├── src/                                 # Phases 1–3 — the audit pipeline
│   ├── data_loader.py                   # fetch + raw snapshot + cleaning
│   ├── preprocessing.py                 # ColumnTransformer + the canonical split
│   ├── train.py                         # entry point: train all 3 models
│   ├── evaluate.py                      # metrics, prediction export, plots
│   ├── fairness_audit.py                # Phase 2
│   └── explainability_audit.py          # Phase 3
├── app/                                 # Phases 5, 7, 8 — backend
│   ├── main.py                          # FastAPI app: all routes
│   ├── services/                        # artifact_reader.py (only file I/O)
│   │   └── governance_service.py        # payload assembly
│   ├── schemas/models.py                # Pydantic response schemas
│   ├── agents/                          # Phase 7 — 4 deterministic agents
│   │   ├── performance_agent.py  fairness_agent.py
│   │   ├── explainability_agent.py  risk_agent.py
│   │   ├── orchestrator.py  schemas.py
│   └── registry/                        # Phase 8 — audit registry
│       ├── integrity.py                 # SHA-256 discovery + verification
│       ├── db.py                        # SQLite (the only writer)
│       ├── service.py  schemas.py  cli.py
│   └── embedded.py                      # runs the API in-process (deployment)
├── dashboard/                           # Phase 6 — Streamlit UI (7 pages)
│   ├── streamlit_app.py
│   └── api_client.py                    # HTTP only; reads no files
├── tests/                               # 102 tests
│   ├── test_api.py  test_agents.py
│   ├── test_registry.py  test_dashboard.py
│   └── test_deployment.py               # single-process deploy path
├── runtime/                             # gitignored local state
│   └── governance_registry.db           # rebuild: python -m app.registry.cli register
├── streamlit_app.py                     # Cloud entry point: API in-process + dashboard
├── .python-version                      # pins 3.13 for Streamlit Cloud
├── requirements.txt
└── README.md
```

---

## 2. Setup — Windows (exact commands)

Open **PowerShell** and `cd` into the project folder.

```powershell
cd C:\Users\nreddy\Downloads\project_KMIT

# 1. Create a virtual environment
py -3.14 -m venv .venv

# 2. Activate it
.\.venv\Scripts\Activate.ps1

# 3. Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks activation with *"running scripts is disabled on this system"*, allow
signed local scripts for your user once:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

**Using `cmd.exe` instead of PowerShell?** Only the activation line changes:

```bat
.venv\Scripts\activate.bat
```

**Prefer not to activate at all?** Call the venv interpreter directly:

```powershell
.\.venv\Scripts\python.exe src\train.py
```

---

## 3. Run

```powershell
# (venv active)
python src\train.py
```

That single command does everything: downloads the dataset on first run, cleans it, splits it,
trains all three models, evaluates them on the held-out test set, and writes every artefact
listed in section 1. Runtime on an i7 CPU: **~30 seconds** after the download.

### Other entry points

```powershell
# Inspect the cleaned data only (shape, class balance, nulls, dtypes)
python src\data_loader.py

# Re-score the saved .joblib models without retraining.
# The split is deterministic, so the metrics reproduce exactly.
python src\evaluate.py

# Force a fresh download, ignoring data\raw\adult_raw.csv
python src\train.py --force-download
```

---

## 3a. Governance API (Phase 5) — run locally on Windows

A read-only FastAPI backend that serves the completed audit evidence over HTTP.
It **never trains, re-scores or writes** — it reads the CSV and Markdown files in
`results/` and serves them verbatim.

### Prerequisites

The audits must have been run at least once, so the artefacts exist:

```powershell
python src\train.py
python src\fairness_audit.py
python src\explainability_audit.py
```

`GET /health` reports `status: "degraded"` and lists exactly which artefacts are
missing if any step was skipped.

### Install and start

```powershell
cd C:\Users\nreddy\Downloads\project_KMIT
.\.venv\Scripts\Activate.ps1

# API dependencies (already in requirements.txt)
pip install -r requirements.txt

# Start the server with hot reload
uvicorn app.main:app --reload --port 8000
```

Without activating the venv:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Stop it with `Ctrl+C`. If port 8000 is busy, pass `--port 8001`.

### URLs

| URL | Purpose |
|---|---|
| http://127.0.0.1:8000/docs | **Swagger UI** — interactive docs, try any endpoint |
| http://127.0.0.1:8000/redoc | ReDoc reference |
| http://127.0.0.1:8000/openapi.json | OpenAPI 3.1 schema |
| http://127.0.0.1:8000/ | redirects to `/docs` |

### Endpoints (all `GET`, all read-only)

| Endpoint | Returns |
|---|---|
| `/health` | Liveness + presence/size/mtime of all 11 audit artefacts |
| `/api/models` | Evaluated models with Phase 1 metrics and audit-coverage flags |
| `/api/models/{model_name}/performance` | Held-out metrics, confusion matrix, error analysis, caveats |
| `/api/models/{model_name}/fairness` | Phase 2 per-group metrics + disparity measures. Optional `?attribute=sex\|race` |
| `/api/models/{model_name}/explainability` | Phase 3 importance, proxy assessment, TreeSHAP local cases. Optional `?top_n=N` |
| `/api/governance/decision` | Research-only approval + deployment block, grounds, conditions. Optional `?include_markdown=true` |
| `/api/governance/risks` | 12-entry risk register. Optional `?overall_risk=`, `?category=`, `?status=` |
| `/api/governance/model-card` | Model card Markdown + section list. Optional `?sections_only=true` |

Valid `model_name` values: `xgboost`, `random_forest`, `logistic_regression`.
Explainability exists for `xgboost` and `logistic_regression` only — Random Forest
returns **404** with the available alternatives rather than fabricated numbers.

### Quick check

```powershell
curl http://127.0.0.1:8000/health
curl http://127.0.0.1:8000/api/governance/decision
```

### Error semantics

| Status | Meaning |
|---|---|
| 404 | Unknown model or attribute name; the response lists valid alternatives |
| 405 | A mutating verb was used — the API is read-only by construction |
| 500 | An artefact exists but is malformed; the response says which |
| 503 | A required artefact is missing; the response names the script that regenerates it |

### Tests

```powershell
pytest -q
```

30 tests covering the health endpoint, every audit-data endpoint, error handling,
CORS, and two contract guarantees: served values are **exactly equal** to the
audit CSVs (no recomputation or rounding), and serving the API leaves every
artefact file unmodified.

### Notes

- **CORS** allows any `http://localhost:<port>` or `http://127.0.0.1:<port>`
  origin, for a future local dashboard. Credentials are disabled and the wildcard
  origin is deliberately not used.
- Artefacts are cached in memory and invalidated on file **mtime**, so re-running
  an audit is picked up without restarting the server.
- Floats are parsed with `float_precision="round_trip"`. Pandas' default CSV
  parser can land one ULP off the written value, which would make the API report
  a number that differs from the audit it cites.

---

## 3b. Governance dashboard (Phase 6) — run locally on Windows

A local Streamlit dashboard over the Phase 5 API. It reads **no files**: every
figure it shows arrives over HTTP from `http://127.0.0.1:8000/api/...`, and it
recalculates nothing.

### You need TWO PowerShell windows

The API and the dashboard are separate processes. Start the API first — the
dashboard is useless without it.

**Window 1 — the API:**

```powershell
cd C:\Users\nreddy\Downloads\project_KMIT
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000
```

Leave it running. Wait for `Application startup complete.`

**Window 2 — the dashboard:**

```powershell
cd C:\Users\nreddy\Downloads\project_KMIT
.\.venv\Scripts\Activate.ps1
streamlit run dashboard\streamlit_app.py
```

Streamlit prints its URL and opens your browser automatically.

### Open

**http://localhost:8501**

Stop either service with `Ctrl+C` in its own window.

### Without activating the venv

```powershell
# Window 1
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
# Window 2
.\.venv\Scripts\python.exe -m streamlit run dashboard\streamlit_app.py
```

### If a port is already in use

```powershell
# API on a different port
uvicorn app.main:app --reload --port 8001
# Dashboard on a different port
streamlit run dashboard\streamlit_app.py --server.port 8502
```

If you move the API, update **API base URL** in the dashboard sidebar to match
(e.g. `http://127.0.0.1:8001`). The default is `http://127.0.0.1:8000`.

### Pages

| Page | Contents |
|---|---|
| **Overview** | Platform purpose, dataset context, API health, primary model (XGBoost) headline metrics, and the prominent governance decision |
| **Model Performance** | Model selector; accuracy / precision / recall / F1 / ROC-AUC; confusion-matrix counts; error-pattern and threshold caveats from the API |
| **Fairness Audit** | Model and sex/race selectors; group table; selection-rate / TPR / FPR chart; disparate-impact ratios with four-fifths screening context; small-group and non-causation caveats |
| **Explainability** | XGBoost global importance chart and ranking table; TreeSHAP local cases; proxy-feature and association-not-causation warnings; explicit unavailable message for unaudited models |
| **Governance Decision & Risks** | Decision grounds and research-use conditions; filterable risk register; severity counts; model-card sections in expanders |

A **Data provenance and limitations** section appears at the bottom of every
page, populated from the model card via the API.

### Sidebar

- **Page** — navigation
- **API base URL** — defaults to `http://127.0.0.1:8000`
- **↻ Refresh data** — clears the cache; use after re-running an audit
- **Connection badge** — connected / degraded / not reachable

### Graceful failure states

| State | What you see |
|---|---|
| API not started | Red banner with the exact `uvicorn` command to run |
| API degraded | Amber warning naming the missing artefacts; available pages still work |
| 404 (e.g. `random_forest` explainability) | Explicit "not available" message listing the models that *do* have data — never fabricated numbers |
| Malformed / non-JSON response | Clear "unexpected response shape" message naming the missing fields |
| Timeout | Message advising retry or restart |

### Verify the dashboard

With the API running in window 1:

```powershell
pytest tests\test_dashboard.py -q
```

15 tests render every page headlessly via Streamlit's `AppTest` and assert the
decision wording, the four-fifths screening framing, the proxy warnings, the
unavailable-model message, and graceful degradation when the API is down. They
skip automatically if the API is not running.

Run everything at once (API tests + dashboard tests):

```powershell
pytest -q
```

### Notes

- The four-fifths (0.80) line is labelled a **screening indicator, not a legal
  conclusion**, in the interface as well as in the artefacts.
- The confusion matrix shows **raw counts only** — deriving percentages in the
  dashboard would mean recomputing a metric the audit owns.
- Charts reuse the same validated colour palette as the static Phase 2/3 figures,
  so a colour means the same thing everywhere.

---

## 3c. Governance agents (Phase 7)

Four **deterministic, rule-based** governance agents that read the existing audit
evidence and emit structured findings. They are **not autonomous decision-makers**
and not language models: they quote evidence verbatim, classify it against fixed
documented thresholds, and cannot train, write, recalculate a metric, or alter the
governance decision.

### Architecture

```text
                    ┌──────────────────────────────────────────────┐
                    │  IMMUTABLE EVIDENCE  (read-only, never written by API/agents)
                    │                                              │
   Phase 1 ─────────┤  data/raw/adult_raw.csv      (1994 UCI snapshot)
   src/train.py     │  models/*.joblib             (fitted pipelines)
                    │  predictions/*.csv           (test predictions + sex/race)
   Phase 2 ─────────┤  results/model_metrics.csv
   src/fairness_    │  results/fairness/*.csv|md
     audit.py       │  results/explainability/*.csv|md
   Phase 3 ─────────┤  results/governance/model_card.md
   src/explain...   │  results/governance/governance_risk_register.csv
   Phase 4 (docs) ──┤  results/governance/governance_summary.md
                    └────────────────────┬─────────────────────────┘
                                         │ read-only, mtime-cached
                                         │ float_precision="round_trip"
                    ┌────────────────────▼─────────────────────────┐
                    │  PHASE 5 — FastAPI service  (app/)           │
                    │                                              │
                    │  services/artifact_reader.py   ← only file I/O
                    │  services/governance_service.py ← shapes payloads
                    │  schemas/models.py              ← Pydantic responses
                    │                                              │
                    │  GET /health                                 │
                    │  GET /api/models                             │
                    │  GET /api/models/{m}/performance             │
                    │  GET /api/models/{m}/fairness                │
                    │  GET /api/models/{m}/explainability          │
                    │  GET /api/governance/decision                │
                    │  GET /api/governance/risks                   │
                    │  GET /api/governance/model-card              │
                    └──────────┬───────────────────────┬───────────┘
                               │ in-process            │ HTTP (JSON)
                               │ (same response models)│
             ┌─────────────────▼──────────────┐        │
             │  PHASE 7 — agents (app/agents/)│        │
             │  deterministic · rule-based    │        │
             │                                │        │
             │   performance_agent  ─┐        │        │
             │   fairness_agent      ├─►      │        │
             │   explainability_agent│ orchestrator    │
             │   risk_agent         ─┘        │        │
             │                                │        │
             │  GET /api/agents               │        │
             │  GET /api/agents/{agent}       │        │
             │  GET /api/agents/review        │        │
             │                                │        │
             │  Decision is COPIED, never     │        │
             │  derived: research-only /      │        │
             │  deployment blocked            │        │
             └─────────────────┬──────────────┘        │
                               │ HTTP (JSON)           │
                    ┌──────────▼───────────────────────▼───────────┐
                    │  PHASE 6 — Streamlit dashboard (dashboard/)  │
                    │  reads NO files · recalculates nothing       │
                    │                                              │
                    │  Overview · Model Performance · Fairness     │
                    │  Explainability · Governance & Risks         │
                    │  Agent Review                                │
                    └──────────────────────────────────────────────┘

Data flows one way only: evidence → API → agents/dashboard.
Nothing to the right of the evidence box ever writes to the left of it.
```

### The four agents

| Agent | Reads | Reports |
|---|---|---|
| `performance` | `/api/models/{m}/performance` | Headline metrics, error counts and asymmetry, threshold and single-split limitations |
| `fairness` | `/api/models/{m}/fairness` | Disparities by sex/race, four-fifths screening context, small-group uncertainty, metric conflicts |
| `explainability` | `/api/models/{m}/explainability` | Top features, proxy-feature concerns, association-vs-causation and dilution limits |
| `risk` | `/api/governance/decision`, `/api/governance/risks` | Blocking and Critical risks, research-use conditions, the deployment recommendation |

Every finding carries: **agent name · severity · finding · evidence source (API
endpoint) · limitations · recommended action**, plus the raw quoted evidence.

### Hard constraints

- **Never train, never write.** Read-only access to the artefacts only.
- **Never recalculate.** Values are quoted verbatim. Where a judgement needs a
  threshold comparison, the agent uses the **audit's own boolean**
  (`fails_four_fifths_rule`, `small_group_flag`) rather than re-deriving it.
- **Never override the decision.** The Risk agent copies the committed decision;
  the orchestrator's `overall_recommendation` **is** the committed headline,
  restated. There is no code path by which agents can approve deployment.
- **Never claim a legal violation or causation.** Enforced by tests that reject
  forbidden phrasings in the fairness agent's output.
- **Never invent missing evidence.** A model with no explainability audit yields
  `status: unavailable` and an explicit "no evidence exists" finding.
- **Deterministic.** Fixed thresholds, no randomness, no timestamps — the same
  artefacts always produce byte-identical output.

### Run (same two windows as §3b)

```powershell
# Window 1 — API (now also serves the agent endpoints)
cd C:\Users\nreddy\Downloads\project_KMIT
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --reload --port 8000

# Window 2 — dashboard
cd C:\Users\nreddy\Downloads\project_KMIT
.\.venv\Scripts\Activate.ps1
streamlit run dashboard\streamlit_app.py
```

Then open **http://localhost:8501** and choose **Agent Review** in the sidebar.

### Try the endpoints

```powershell
curl http://127.0.0.1:8000/api/agents
curl "http://127.0.0.1:8000/api/agents/fairness?model_name=xgboost"
curl "http://127.0.0.1:8000/api/agents/review?model_name=xgboost"
```

Swagger documents all three under the **agents** tag:
http://127.0.0.1:8000/docs

### Test

```powershell
pytest tests\test_agents.py -q      # 28 tests, no server needed
pytest -q                           # everything (API needed for dashboard tests)
```

The agent tests assert that **every quoted value equals the source endpoint
exactly** (`==`, not approximate), that the decision is preserved field for field,
that output is deterministic across runs, and that running the agents modifies no
artefact.

## 3d. Governance audit registry (Phase 8)

A local **SQLite-backed audit registry** that records the current audit as one
governance review run, with SHA-256 checksums of every referenced artefact, so the
evidence a conclusion rests on can be verified later.

The registry **records evidence; it does not make decisions.** The governance
decision stored on a run is a copy of the committed decision record.

### Where it lives

```text
runtime/governance_registry.db      # gitignored local state
```

Rebuild it at any time from the committed evidence — nothing is lost by deleting it.

### Create or refresh the registry (Windows)

```powershell
cd C:\Users\nreddy\Downloads\project_KMIT
.\.venv\Scripts\Activate.ps1

# Create or refresh the current audit-run record. Idempotent.
python -m app.registry.cli register

# List registered runs
python -m app.registry.cli list

# Verify the active run's checksums (exit code 0 = verified, 2 = integrity broken)
python -m app.registry.cli integrity
```

Without activating the venv, prefix with `.\.venv\Scripts\python.exe -m ...`.

To use a different database location:

```powershell
python -m app.registry.cli register --db C:\temp\my_registry.db
# or
$env:GOVERNANCE_REGISTRY_DB = "C:\temp\my_registry.db"
```

### Why `register` is idempotent

A run's id is **content-addressed**: it is derived from an *evidence digest*, which
is a SHA-256 over the sorted `path:sha256` pairs of every registered artefact, plus
the dataset and subject-model identity.

- **Evidence unchanged** → same run id → the existing row is **refreshed in place**.
  `created_at` is preserved; `refreshed_at` and `refresh_count` advance. No duplicate.
- **Any artefact changed** → different digest → **new run** is created and the
  previously active run is marked `superseded`. History is never rewritten.

The model version is itself content-based (`sha256:` of the fitted pipeline), so it
changes if and only if the serialised model changes — no version numbers to forget
to bump.

### What a run stores

| Field | Source |
|---|---|
| `run_id` | content-addressed from the evidence digest |
| `created_at` / `refreshed_at` / `refresh_count` | registry |
| `dataset_name` / `dataset_version` / `dataset_context` | UCI Adult id 2, 1994 census context |
| `model_name` / `model_version` / `model_run_identifier` | `sha256:` of the pipeline + split/threshold params |
| `artifacts[]` | path, group, **SHA-256**, size, mtime for each of ~34 artefacts |
| `evidence_digest` | SHA-256 over the whole artefact set |
| `performance_summary` | quoted verbatim from the Phase 1 audit |
| `governance_decision` / `blocking_risk_ids` | copied from the committed decision record |
| `audit_coverage` | performance, fairness, explainability, governance, agents |
| `status` | `active`, `superseded` or `archived` |

### Endpoints

| Endpoint | Returns |
|---|---|
| `/api/registry/runs` | All runs, newest first. Optional `?status=active` |
| `/api/registry/runs/{run_id}` | Full run detail + artefact manifest |
| `/api/registry/runs/{run_id}/integrity` | **Recomputes** every checksum: verified / missing / changed + overall status |
| `/api/registry/runs/{run_id}/timeline` | Registry events + evidence events. Optional `?limit=N` |

Creating or refreshing a run is a deliberate **CLI action**, not an HTTP call, so
the API surface stays read-only. Before the registry exists these endpoints return
**503** with the exact command to run — never an empty list, which would read as
"this audit was never registered".

### Integrity semantics

| Status | Meaning |
|---|---|
| `verified` | Every artefact still hashes to its registered value |
| `incomplete` | Some registered artefacts are gone; none altered |
| `modified` | Some artefacts were altered |
| `modified_and_incomplete` | Both |

`integrity_ok` is true only for `verified`. Two deliberate design points:

- **mtime is excluded from the digest**, so touching a file without changing its
  bytes does not break integrity.
- **`changed` is not an accusation.** Re-running an audit script legitimately
  rewrites its outputs. It means this run's conclusions no longer describe the files
  on disk, so a new run should be registered. Checksums detect modification, not
  authorship — this is a local integrity check, not a signed provenance chain.

### Dashboard

Open **http://localhost:8501** → **Model Registry** for the run list, selected-run
metadata, recorded decision, evidence coverage, live integrity status and timeline.

### Test

```powershell
pytest tests\test_registry.py -q     # 22 tests, no server needed
```

Includes a **deliberate integrity mismatch**: artefacts are copied into a temp
directory, one is modified and one deleted, and verification is run against those
copies — real files, real SHA-256 recomputation, and the repository's evidence is
never touched. A final test asserts the real artefacts are byte-identical after the
whole module has run.

### Architecture addition

```text
   ┌──────────────────────────────────────────────┐
   │  IMMUTABLE EVIDENCE (read-only)              │
   │  data/ · models/ · predictions/ · results/   │
   └───────────────┬──────────────────────────────┘
                   │ SHA-256 (read-only)
                   ▼
   ┌──────────────────────────────────────────────┐
   │  PHASE 8 — registry (app/registry/)          │
   │  integrity.py  discovery + SHA-256           │
   │  db.py         SQLite (the ONLY writer)      │
   │  service.py    register / read / verify      │
   │  cli.py        python -m app.registry.cli    │
   │                                              │
   │  writes ONLY to runtime/*.db                 │
   └───────────────┬──────────────────────────────┘
                   │ read-only HTTP
                   ▼
        /api/registry/runs[...]  →  dashboard "Model Registry"
```

---

### Note on `/api/agents/review` route order

`/api/agents/review` is declared **before** `/api/agents/{agent_name}` in
`app/main.py`. Reversed, FastAPI would match `review` as an agent name and the
review endpoint would become unreachable. A test pins this.

> Note: `evaluate.py` rewrites `results\model_metrics.csv` without the `fit_seconds`
> column (it never trains, so it has no timings). Run `train.py` if you want that column.

---

## 3e. Deploying to Streamlit Community Cloud

The platform normally runs as **two processes** (uvicorn + Streamlit). Streamlit
Community Cloud runs **one process per app**, so a separately-launched API cannot
exist there — deployed naively, every dashboard page would show *"API unavailable"*.

The repository root therefore contains a deployment entry point,
[`streamlit_app.py`](streamlit_app.py), which starts the API **in-process** on a
background thread and then hands over to the unmodified dashboard.

### What Cloud needs (all committed)

| File | Purpose |
|---|---|
| `streamlit_app.py` | Entry point at the repo root — Cloud's default main file |
| `requirements.txt` | Dependencies |
| `.python-version` | Pins **3.13** (Cloud tops out below the 3.14 used locally) |

### Deploy

1. Go to **https://share.streamlit.io** and sign in with GitHub.
2. **New app** → **Deploy from existing repo**.
3. Repository `Nikhilreddy1867/project_kmit` · Branch `main` · Main file path
   **`streamlit_app.py`** (the root one, not `dashboard/streamlit_app.py`).
4. Under *Advanced settings*, confirm Python **3.13**.
5. Deploy. First boot takes a few minutes while the dependencies install.

### What the entry point does

```text
Cloud starts ONE process
        │
        ├─ app.embedded.ensure_api_running()   → uvicorn on 127.0.0.1:8000,
        │                                        daemon thread, loopback only
        ├─ app.embedded.ensure_registry()      → creates runtime/ registry
        │                                        (gitignored, so absent on a
        │                                         fresh clone). Idempotent.
        └─ dashboard/streamlit_app.py :: main() → unchanged, still HTTP-only
```

**The data contract is unchanged.** The dashboard still fetches everything over HTTP
through `api_client`, still reads no files and still imports nothing from `app/`.
What changed is *who starts the server*, not how data reaches the UI — all the
deployment glue lives in the root entry point so those files are byte-identical
between local and deployed runs.

### Local development is unaffected

`ensure_api_running()` first probes port 8000 and **defers to an already-running
server**, so the two-terminal workflow in §3b keeps working exactly as documented.
You can also use the entry point locally to get both halves from one command:

```powershell
streamlit run streamlit_app.py        # API + dashboard in one process
```

### Deliberate design points

- **Loopback only.** The embedded API binds `127.0.0.1`, so it is *not* publicly
  reachable even though the Streamlit app is. This matters: `predictions/*.csv`
  carry `sex` and `race` for all 9,769 test records, and a publicly-bound API would
  serve them to anyone. A test asserts `0.0.0.0` never appears in `app/embedded.py`.
- **Cold start.** The API becomes ready on the first page load (Streamlit executes
  the script per session), which adds roughly 10–15 s to the very first render.
  `ensure_api_running()` blocks until `/health` answers, so no page ever renders
  against a half-started server.
- **Registry auto-bootstrap.** `runtime/` is gitignored, so a fresh deployment has
  no registry. The entry point registers one at boot; failure is logged and
  non-fatal, degrading one page rather than breaking the app.
- **Module caching.** The dashboard is imported once and cached in `sys.modules`.
  Re-executing it on every rerun would rebuild its `@st.cache_data` functions and
  make the cache miss on every click.

### Known limitations of the deployed app

- **`requirements.txt` installs the full ML stack** (`scikit-learn`, `xgboost`,
  `matplotlib`, `ucimlrepo`) even though neither the API nor the dashboard imports
  any of them — only `src/` does. Cloud reads the root `requirements.txt` and does
  not support a separate deployment file, and trimming it would break the documented
  local setup in §2. The cost is a slower first build, not incorrect behaviour.
- **Ephemeral filesystem.** The registry database lives in the container and is
  rebuilt on each cold start. Integrity checks still work; the run history does not
  persist across restarts.
- **Free-tier resources.** The audit scripts in `src/` are *not* meant to run on
  Cloud — the evidence is committed and read-only. Only the API and dashboard run.

### Verify the deployment path locally

```powershell
pytest tests\test_deployment.py -q
```

11 tests, and notably they need **no externally running server** — which is the
whole point, since Cloud has none. They cover the embedded API start, idempotency,
loopback binding, registry bootstrap and failure tolerance, the Python pin, and an
end-to-end render of all seven pages against the in-process API.

---

## 4. Results (held-out test set, 9,769 rows, positive class = `>50K`)

| model | accuracy | precision | recall | f1 | roc_auc | fit (s) |
|---|---|---|---|---|---|---|
| **xgboost** | **0.8782** | **0.7911** | **0.6672** | **0.7239** | **0.9299** | 15.3 |
| random_forest | 0.8684 | 0.7834 | 0.6219 | 0.6934 | 0.9186 | 4.5 |
| logistic_regression | 0.8507 | 0.7314 | 0.5941 | 0.6557 | 0.9042 | 1.0 |

XGBoost wins on every metric. Ranking and absolute values match the published literature for
Adult, which is a good sanity check that no leakage is inflating the scores.

**Read accuracy with care.** The target is imbalanced — 76.1% of records are `<=50K`, so a model
that predicts "everyone earns `<=50K`" scores **0.761 accuracy** while being useless. F1 and
ROC-AUC are the metrics to judge on. Note that recall on the high-earner class is only 0.59–0.67:
all three models miss roughly a third of actual high earners. That error is not evenly
distributed across demographic groups, which is exactly what Phase 2 must quantify.

---

## 5. Methodology

### Dataset
- Source: `ucimlrepo.fetch_ucirepo(id=2)` — 48,842 rows, 14 features, target `income`.
- `data/raw/adult_raw.csv` is written **verbatim** (no cleaning) as an immutable audit trail.
  Later runs read the snapshot, so results are reproducible and work offline.
- The UCI endpoint intermittently resets connections mid-download, so the fetch retries
  up to 5 times with linear backoff. Transport-level retry only — the data is unchanged.

### Cleaning
- `?`, blanks and similar tokens → real nulls. Found: `workclass` 2,799, `occupation` 2,809,
  `native-country` 857 (6,465 missing cells total).
- Target encoded `>50K` → **1**, `<=50K` → **0**.
- The raw file mixes `<=50K` with `<=50K.` (the trailing period comes from the original
  `adult.test` split). Both spellings are collapsed, otherwise the target would have 4 levels
  instead of 2 — a silent, results-wrecking bug.
- `sex` and `race` are **kept as model features** and echoed into every prediction CSV.

### Split
- 80/20 via `train_test_split(..., test_size=0.2, stratify=y, random_state=42)`
  → 39,073 train / 9,769 test.
- `stratify=y` holds the positive rate at 0.2393 in **both** folds. Adult is imbalanced, so an
  unstratified split can shift the class ratio between folds and make recall and ROC-AUC noisy
  and non-comparable across models.
- `random_state=42` makes the split deterministic, so all three models see identical rows and
  `evaluate.py` can rebuild the exact same test set later.

### Data-leakage prevention
This is the part that matters most for governance, and it is enforced structurally:

1. **Split happens first.** `make_split` is called on cleaned-but-untransformed data. At that
   moment no fitted transformer exists anywhere in the process.
2. **All learned transformations live inside the Pipeline.** Imputation medians, the one-hot
   vocabulary and scaler mean/std are parameters *learned from data*. They sit in a
   `ColumnTransformer` that is the first step of each model's `Pipeline`, so
   `pipeline.fit(X_train, y_train)` — the only `fit` call in the codebase — fits them on the
   **training fold only**. Computing a median or scaler over the full dataset before splitting
   is the classic leak: test-set information reaches training and inflates the score.
3. **The test set is touched only via `predict` / `predict_proba`**, which *apply* the
   already-fitted transformers without refitting.
4. **Each model gets its own preprocessor instance**, so nothing is shared or reused across fits.
5. **The whole Pipeline is serialised**, not just the estimator. Inference cannot accidentally
   apply different preprocessing — no train/serve skew.

### Preprocessing detail
| branch | steps | applies to |
|---|---|---|
| numeric (6 cols) | `SimpleImputer(strategy="median")` → `StandardScaler` | Logistic Regression |
| numeric (6 cols) | `SimpleImputer(strategy="median")` | Random Forest, XGBoost |
| categorical (8 cols) | `SimpleImputer(strategy="most_frequent")` → `OneHotEncoder(handle_unknown="ignore")` | all models |

- **Median, not mean**, for numeric imputation: `capital-gain` and `fnlwgt` are heavily
  right-skewed, so the mean gets dragged by outliers.
- **Scaling only for Logistic Regression.** It is a gradient/penalty-based linear model, so
  unscaled `fnlwgt` (~1e5) would swamp `age` (~1e1) and the L2 penalty would be applied
  unevenly. Trees split on ordered thresholds, so monotonic rescaling is a mathematical no-op
  for Random Forest and XGBoost.
- **`handle_unknown="ignore"`** encodes an unseen category as an all-zero block instead of
  raising — required for robustness against categories that appear only in test or in future
  production data.

### Models
All use `random_state=42`. Fixed, sensible defaults — **no hyperparameter tuning** — so the
baseline is fast and reproducible on a laptop.

- `LogisticRegression(max_iter=2000, solver="lbfgs")` — interpretable governance baseline.
- `RandomForestClassifier(n_estimators=300, min_samples_leaf=2, n_jobs=-1)`
- `XGBClassifier(n_estimators=400, learning_rate=0.1, max_depth=6, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0, tree_method="hist", n_jobs=-1)`

### Prediction output schema
One row per test record, per model:

| column | meaning |
|---|---|
| `actual_income` | ground truth, 0 / 1 |
| `predicted_income` | predicted class at the default 0.5 threshold |
| `predicted_probability` | `P(income > 50K)` from `predict_proba` |
| `sex` | original value, for fairness analysis |
| `race` | original value, for fairness analysis |
| `actual_income_label` | convenience: `<=50K` / `>50K` |
| `predicted_income_label` | convenience: `<=50K` / `>50K` |

`sex` and `race` are copied from the **raw, pre-transform** test features, so they are
human-readable values rather than one-hot columns — group-by ready.

---

## 6. Assumptions and deliberate non-choices

Recorded explicitly, since each is a decision Phase 2 may want to revisit:

- **Rows with a missing target are dropped** (none in practice). You cannot supervise on a null
  label, and imputing a *label* would fabricate ground truth.
- **`fnlwgt` and `education-num` are kept.** `fnlwgt` is a census sampling weight, not a personal
  attribute, and is often dropped; `education-num` is the ordinal twin of `education`, so the two
  are redundant. Both were left in rather than removed silently.
- **No class re-weighting, no resampling, default 0.5 threshold.** All three models are compared
  under identical, untouched class priors so the table is apples-to-apples. Threshold tuning and
  `class_weight="balanced"` are Phase-2 fairness levers.
- **No hyperparameter tuning and no cross-validation.** This is a baseline. Any future tuning
  must run CV *inside the training fold* to stay leakage-free.
- **Duplicate rows are not removed.** Adult contains some exact duplicates; dropping them is a
  modelling decision, so it was left to Phase 2.
- **`sex` and `race` are used as predictors.** Dropping them would not remove bias — it is
  recoverable from correlated proxies like `occupation` and `relationship` — while it *would*
  remove the ability to measure bias. Fairness-aware training is Phase 2.

---

## 7. Environment notes / troubleshooting

- Installed on this machine: `pandas 3.0.5`, `numpy 2.5.2`, `scikit-learn 1.9.0`,
  `xgboost 3.4.0`, `matplotlib 3.11.1`, `joblib 1.5.3`, `ucimlrepo 0.0.7`.
- `requirements.txt` uses **lower bounds only**, so pip picks wheels that exist for your Python
  version. All of the above ship `cp314` wheels — no compiler needed.
- The code is written to work on both pandas 2.x and 3.x. Text-column detection is done by
  *exclusion* of numeric/bool/datetime dtypes rather than `dtype == object`, because pandas 3
  stores strings as a dedicated `str` dtype and an `== object` test would silently skip every
  text column.
- `random_forest_pipeline.joblib` is ~100 MB (300 unpruned trees). Expected, not an error.
  Raise `min_samples_leaf` or lower `n_estimators` to shrink it.
- matplotlib uses the headless `Agg` backend, so plots are written straight to PNG and no window
  ever opens.
- **`ConnectionResetError` / `WinError 10054` on first run**: transient UCI server issue. The
  loader retries automatically; if all 5 attempts fail, check your proxy/VPN and re-run.
- **`ModuleNotFoundError: No module named 'xgboost'`**: the venv is not active. Re-run
  `.\.venv\Scripts\Activate.ps1`, or call `.\.venv\Scripts\python.exe` directly.

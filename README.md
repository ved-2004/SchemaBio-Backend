# SchemaBio

**AI-driven scientific workflow platform for antibiotic resistance drug discovery and translational execution.**

SchemaBio turns fragmented data—resistance assays, compound screens, variants, and target rationales—into a single source of truth, then hands off to experiment design and execution planning. Not a chatbot: a **structured scientific operating system**.

---

## Why SchemaBio

Antibiotic resistance programs live in spreadsheets, VCFs, and PDFs. Decisions get made in silos. SchemaBio:

- **Ingests** whatever you have (CSV, VCF, PDF, notes) and parses it **deterministically**—no hallucination, no black box.
- **Normalizes** everything into evidence-linked entities and signals: organisms, targets, compounds, variants, assay types, resistance fold-shifts, compound hits.
- **Estimates** where your program sits in the pipeline (hit discovery → resistance characterization → validation → preclinical → manufacturing).
- **Flags** what’s missing (controls, replicates, ADMET, target engagement) so the next layers know what to recommend.
- **Hands off** clean contracts to Experiment Design (next experiments, controls, prioritization) and Execution (partners, funding, GMP readiness)—so you can build or plug in AI there without re-parsing the world.

**The ingestion layer is the source of truth.** Everything downstream consumes its JSON.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  LAYER 1 — Ingestion (implemented)                                │
│  CSV · VCF · PDF · Text  →  ProgramState + ExperimentDesignInput │
│  + ExecutionPlanningInput (deterministic parsing, zero LLM)       │
└────────────────────────────┬────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│  LAYER 2 — Experiment Design (to build)                           │
│  Consumes experiment_design_input → ranked experiments, controls, │
│  hypothesis prioritization, literature-backed reasoning           │
└────────────────────────────┬────────────────────────────────────┘
                              │
┌─────────────────────────────▼───────────────────────────────────┐
│  LAYER 3 — Execution / Translational Planning (to build)         │
│  Consumes execution_planning_input → CRO/CDMO recommendations,    │
│  funding paths, evidence package gaps, GMP readiness               │
└──────────────────────────────────────────────────────────────────┘
```

---

## What’s in the box

- **Ingestion API** — `POST /api/upload-and-parse` (real files), `GET /api/demo-ingestion` (mocked antibiotic resistance demo). Returns a single JSON: `program_state`, `experiment_design_input`, `execution_planning_input`.
- **Parsers** — Resistance assay CSV, compound screen CSV, VCF, PDF/notes. Schema detection, entity and signal extraction, missing-data flags, evidence index.
- **Stage estimation** — Deterministic rules: hit_discovery, resistance_mechanism_characterization, experimental_validation_planning, preclinical_package_gap_analysis, manufacturing_feasibility_review. Confidence + reasoning_basis.
- **Frontend** — React 18, Vite, TypeScript, shadcn/Tailwind. Ingestion page (upload + demo), Program Dashboard driven by ingestion response, Experiments / Execution / Literature / Reports shells ready for the next two layers.
- **Handoff docs** — For Claude or any builder: ingestion summary, experiment-design input schema and suggested outputs, execution-planning input schema and suggested outputs, paste-ready “next steps” context. See `/docs` and `/docs/examples`.

---

The app is deployed via Vercel.

---

## API (ingestion layer)

| Method | Endpoint | Description |
|--------|----------|-------------|
| **GET** | `/api/demo-ingestion` | Full mocked antibiotic resistance `IngestionResponse` |
| **POST** | `/api/upload-and-parse` | Upload files → run ingestion → return `IngestionResponse` |
| **POST** | `/api/program-state` | Optional: return `ProgramState` from test input or demo |
| **GET** | `/api/health` | Health check |

Request/response contracts and example JSONs: see `docs/ingestion-layer-summary.md` and `docs/examples/`.

---

## Repo structure

```
SchemaBio/
├── README.md
├── CURSOR_GUIDE.md
├── backend/
│   ├── main.py                 # FastAPI app + ingestion router
│   ├── models/
│   │   ├── ingestion.py        # Pydantic: IngestionResponse, ProgramState, etc.
│   │   └── drug_program.py    # Legacy DrugProgram (AIDEN)
│   ├── routers/
│   │   └── ingestion.py        # /api/upload-and-parse, /api/demo-ingestion
│   ├── services/
│   │   ├── ingestion_service.py   # Orchestration
│   │   ├── stage_estimator.py     # Deterministic stage rules
│   │   └── parser_adapter.py      # Parsers → entities/signals
│   ├── parsers/                # VCF, assay, compound, PDF, universal
│   ├── data/
│   │   └── demo_ingestion.py   # Mock IngestionResponse
│   └── agents/                 # Legacy AIDEN pipeline (optional)
├── frontend/                   # React + Vite + TypeScript + shadcn
│   └── src/
│       ├── pages/              # Ingestion, Dashboard, Experiments, Execution, …
│       ├── contexts/IngestionContext.tsx
│       ├── types/ingestion.ts
│       └── lib/ingestionApi.ts
├── docs/
│   ├── ingestion-layer-summary.md
│   ├── experiment-design-layer-handoff.md
│   ├── execution-planning-layer-handoff.md
│   ├── claude-paste-ready-next-steps.md
│   └── examples/              # example_ingestion_response.json, etc.
├── data/demo/                  # Sample CSVs, VCF for testing
└── tests/
```

---

## Tech stack

- **Backend:** FastAPI, Pydantic, pandas, PyMuPDF, cyvcf2/pysam (VCF). No LLM in ingestion.
- **Frontend:** React 18, Vite, TypeScript, shadcn/ui, Tailwind, TanStack Query.
- **Proxy:** Vite dev server proxies `/api` → `http://localhost:8000`.

---

## One-liner

> SchemaBio ingests resistance assays, compound screens, VCFs, and PDFs into a single evidence-linked JSON, estimates program stage, and hands off clean inputs to Experiment Design and Execution layers—so the next AI you plug in reasons over structure, not raw files.

---

## License

MIT.

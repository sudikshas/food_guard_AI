# SafeCart — Real-Time Food Safety Intelligence

> **Hybrid AI system** combining a deterministic 3-pass NLP safety engine with LLM-powered explanation via AWS Bedrock — deployed end-to-end on AWS for real-time food recall and allergen detection at the point of purchase.

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=flat-square&logo=fastapi)](https://fastapi.tiangolo.com)
[![React](https://img.shields.io/badge/React-18-61dafb?style=flat-square&logo=react)](https://react.dev)
[![AWS](https://img.shields.io/badge/AWS-EC2%20·%20RDS%20·%20Bedrock%20·%20Textract-FF9900?style=flat-square&logo=amazonaws)](https://aws.amazon.com)
[![Live App](https://img.shields.io/badge/Live-App%20deployed-brightgreen?style=flat-square)](https://54.210.208.14)


---

## The problem

~1,500 FDA food recalls are issued every year. Most consumers never know a product they already bought was recalled — because the data is public but completely inaccessible: buried in legal enforcement notices, written in technical language, and not searchable by the barcode in your hand.

For the 32 million Americans managing food allergies, the problem is compounded. Ingredient labels use hundreds of scientific synonyms, and there is no mandatory FDA declaration for many dietary restrictions.

SafeCart makes public food safety data **personalized, explainable, and available at the exact moment a shopping decision is made.**

---

## Architecture

```
User scans barcode / uploads receipt
              │
              ▼
     ┌──────────────────┐
     │  FastAPI backend │  ← AWS EC2, Gunicorn
     └────────┬─────────┘
              │
    ┌─────────┼──────────────────┐
    ▼         ▼                  ▼
 FDA API   Open Food         AWS Textract
(recalls)   Facts API        (receipt OCR)
    │         │                  │
    └─────────┴──────────────────┘
                     │
        ┌────────────▼────────────┐
        │  Ingredient Risk Engine  │  ← Deterministic, 3-pass NLP
        │  (Python, pure rules)    │    200+ allergen synonyms
        └────────────┬────────────┘    Compound exclusion logic
                     │
              ┌──────▼──────┐
              │ LLM Layer   │  ← AWS Bedrock (Claude Haiku)
              │ (additive)  │    Explanation only — never verdict
              └──────┬──────┘
                     │
              ┌──────▼──────────────────┐
              │  Verdict + Notification  │  → DONT_BUY / CAUTION / OK
              └─────────────────────────┘
```

**Key architectural decision:** LLMs are used for explanation and data extraction — never for safety verdicts. Every DONT_BUY decision is deterministic and interpretable. If Bedrock is unavailable, the safety system continues without interruption; only the plain-language explanation degrades.

---

## Core components

### 1. Ingredient risk engine (`backend/ingredient_risk_engine.py`)

A three-pass deterministic NLP pipeline that maps a product's ingredient text against a user's allergen and diet profile.

**Pass 1 — Exact token match** → `DEFINITE` confidence  
**Pass 2 — Substring match with compound exclusion** → `PROBABLE` confidence  
**Pass 3 — Advisory phrase extraction** (`"may contain"`, `"shared facility"`) → `PROBABLE` + `is_advisory=True`

Both DEFINITE and PROBABLE trigger the `ALLERGEN` hard stop (DONT_BUY). Advisory language is promoted to the same gate as confirmed ingredients — because for someone with anaphylaxis risk, "may contain peanuts" is a product they cannot safely eat.

**The compound exclusion problem:** Substring matching creates false positives where plant-derived compounds share tokens with allergen synonyms (`"cocoa butter"` → milk, `"oat milk"` → milk, `"eggplant"` → eggs). The fix required **separate exclusion dictionaries for allergen checks vs diet checks** — because the same compound requires different handling in each context. `"peanut butter"` is excluded from the Milk allergen check (it's not dairy) but must never be excluded from the Peanut allergen check.

**Two-layer verdict system:**
- Layer 1 (hard stops): Recall → Allergen match → Strict diet violation → `DONT_BUY`
- Layer 2 (soft signals): Flagged additives, Keto/Paleo conflicts, missing ingredient data → `CAUTION`

Design objective: **recall over precision throughout.** A missed allergen warning is life-threatening; a false positive is inconvenient.

### 2. Receipt-to-recall matching (`backend/fuzzy_recall_matcher.py`)

Grocery receipts are noisy ("ORG ALM BTR 16OZ"), retailer-specific, and bear little resemblance to FDA product descriptions. Four similarity approaches were benchmarked:

| Approach | Notes |
|---|---|
| TF-IDF character n-grams | Strong on token overlap, weak on abbreviations |
| RapidFuzz partial ratio | Good on substrings, poor on reordering |
| RapidFuzz token-set ratio | Handles reordering, sensitive to noise |
| Cross-encoder (ms-marco-MiniLM-L-6-v2) | Best semantic, slowest |

No single measure dominated. **Ensemble waterfall:** z-score each measure's similarity → top scorer must clear threshold → word-by-word confirmation must also pass → alert user. Final accuracy: **86%** on a held-out set of real receipt/recall pairs.

### 3. LLM integration (`backend/LLM_services.py`)

LLMs handle two tasks where rule-based approaches cannot:

**Recall explainer:** FDA enforcement text → plain-language consumer card (headline / what happened / who's at risk / what to do / recall class / distribution region). Generated once at ingestion, stored as JSONB. System prompt engineered for strict JSON-only output with schema validation and raw-FDA-text fallback.

**Ingredient disambiguator (opt-in):** Resolves genuinely ambiguous tokens — `"natural flavors"`, `"modified food starch"`, `"vitamin D3"` — where allergen status depends on manufacturing context the label doesn't disclose. Results cached 30 days keyed on `SHA-256(token + full_ingredient_list)`, not token alone, because the same token yields different inferences in different product contexts.

**LLM in the data pipeline:** Claude Haiku also extracts structured UPCs and US state distribution codes from messy FDA free-text notices — a pattern where LLM-as-ETL substantially outperforms regex over inconsistent formats.

---

## Evaluation

### Ingredient risk engine

Evaluated on a 40-case hand-labeled test set structured around known failure mode categories: compound allergen derivatives, plant-based false positives, advisory phrase variants, and custom allergen strings.

| Version | Change | Precision | Recall |
|---|---|---|---|
| V1 | Exact token match only | High | Low — missed derivatives |
| V2 | + Substring matching | ~70% | **100%** — but plant-based FPs introduced |
| V3 | + Compound exclusion dicts | **96%** | **100%** |

Ablation study confirmed all three passes contribute; removing any single pass meaningfully degrades at least one metric.

### Receipt matching

86% accuracy on held-out real-world receipt/recall pairs using the ensemble waterfall approach.

---

## Results & impact

- **96% precision / 100% recall** on allergen detection (V3, hand-labeled evaluation)
- **86% accuracy** on receipt-to-recall matching (ensemble, held-out set)
- **Single barcode scan** replaces manual label reading + FDA database search
- **Post-purchase protection** via receipt scanning — users alerted on items recalled weeks after purchase
- **Personalized** to individual allergen lists and 8 diet profiles (Vegan, Vegetarian, Gluten-Free, Dairy-Free, Keto, Paleo, Halal, Kosher)
- **State-specific** recall filtering against FDA distribution patterns
- **Production deployed** on AWS at time of capstone presentation

---

## Honest limitations

| Limitation | Impact | Mitigation path |
|---|---|---|
| Open Food Facts has ~50% ingredient coverage for US products | Engine returns LOW_CONFIDENCE for half of scans | Label image OCR to supplement missing text |
| FDA rarely includes UPCs — most matching is fuzzy | Inherent matching uncertainty | Richer product name normalization; feedback loop |
| 40-case eval set is small and failure-driven | Real-world precision expected lower on novel derivatives | User feedback loop; expanded adversarial test cases |
| No behavioral monitoring in production | Verdict distribution drift undetected until user-reported | Daily distribution job with alerting |
| LLM disambiguation is opt-in | Ambiguous tokens unresolved for most users | Default-on with cost controls |

---

## Tech stack

| Layer | Technology |
|---|---|
| Backend | FastAPI, Gunicorn (uvicorn workers), Python 3.11 |
| Frontend | React, TypeScript, Vite, Tailwind CSS, ZXing barcode |
| Database | PostgreSQL on AWS RDS |
| LLM | Claude Haiku via AWS Bedrock |
| OCR | AWS Textract (receipt scanning)|
| Infrastructure | AWS EC2 (Ubuntu), nginx, APScheduler |
| NLP | RapidFuzz, scikit-learn TF-IDF, sentence-transformers cross-encoder |
| Product data | Open Food Facts API (with RDS cache) |
| Recall data | openFDA enforcement API (refreshed every 6 hrs) |

---

## Repository structure

```
Capstone-Recall-Alert/
├── backend/
│   ├── ingredient_risk_engine.py   # Core 3-pass NLP safety engine
│   ├── LLM_services.py             # Bedrock integration: explainer + disambiguator
│   ├── fuzzy_recall_matcher.py     # Receipt-to-recall ensemble matching
│   ├── recall_update.py            # FDA recall ingestion pipeline + APScheduler
│   ├── receipt_scan.py             # Textract OCR + cleaning + matching
│   ├── risk_routes.py              # Primary scan endpoint + notification builder
│   ├── barcode_routes.py           # Product lookup + recall cross-reference
│   ├── user_routes.py              # Auth, profiles, cart
│   ├── user_alerts.py              # Alert generation + email notifications
│   └── database.py                 # RDS connection helpers
├── frontend/                       # React + TypeScript mobile-first UI
└── misc/                           # DB schema, migration scripts, team docs
```

---

## Local setup

```bash
# Backend
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add DB credentials
uvicorn app:app --reload
# API at http://localhost:8000/docs

# Frontend
cd frontend
npm install && npm run dev
# UI at http://localhost:5173
```

Requires: AWS credentials configured with IAM access to Bedrock, Textract, and RDS.

---

## Key design principles

**1. Safety verdicts are never delegated to LLMs.**  
A language model that hallucinates a "safe" verdict on a product containing a user's allergen is a medical liability. Every DONT_BUY comes from the deterministic engine — interpretable, auditable, consistent under load.

**2. The asymmetry of outcomes drives the objective function.**  
A missed allergen warning is life-threatening; a false positive caution is inconvenient. The system is tuned for recall over precision throughout, and this is an explicit design choice — not a side effect.

**3. Unknown is not the same as safe.**  
Missing ingredient data surfaces as a LOW_CONFIDENCE warning (CAUTION), not a silent safe verdict. A product without ingredient text is not safe by default.

**4. LLM failure modes must be tolerable.**  
Every Bedrock call wraps in try/catch with graceful fallback. Recall explanation degrades to raw FDA text; ingredient disambiguation degrades to deterministic-only; safety verdict is unaffected. The system never fails silently.

---

*UC Berkeley MIDS Capstone · School of Information · 2025–2026*

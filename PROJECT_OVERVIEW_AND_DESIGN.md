# MC4 Project Overview & Design Guide

## 1. How the Full Project Fits Together

### High-level architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  FRONTEND (Next.js 16, React 18, Tailwind)  →  http://localhost:3000        │
│  app/page.tsx → Layout.tsx → 8 screens + Chatbot (floating)                  │
│  API calls: /api/* rewritten to backend                                      │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │ rewrites /api/* → localhost:8000
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  BACKEND (FastAPI, Python)  →  http://localhost:8000                        │
│  fastapi_server.py: CSV cache, REST endpoints, proxies chatbot              │
│  Data: backend/datasets/*.csv (from data_generator.py)                       │
└───────────────────────────────────┬───────────────────────────────────────┘
                                    │ import run_chatbot_query
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  Text2SQL_V2 (Chatbot engine)  →  used by backend, not a separate server    │
│  chatbot_api.py: run_chatbot_query(question) → SQL → SQLite (chatbot.db)     │
│  Same CSVs in backend/datasets/ + schema_metadata.json                      │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Data flow

1. **Data generation**  
   `backend/data_generator.py` produces all CSVs in `backend/datasets/` (time dimension, SKU/recipe/mill masters, forecasts, mill load, raw material, etc.).

2. **Backend**  
   `fastapi_server.py` loads those CSVs into `data_cache` on startup. Endpoints aggregate by `horizon` (week/month/year) and optional `period`. No database for the main app; everything is in-memory from CSVs.

3. **Chatbot**  
   `backend/setup_chatbot_db.py` (or chatbot_api’s first run) builds `Text2SQL_V2/chatbot.db` from the same CSVs. The backend imports `run_chatbot_query` from `Text2SQL_V2/chatbot_api.py`, which uses that SQLite DB + LLM (Google/OpenAI from `.env`) to turn natural language → SQL → result → summary (and optional viz/email for inserts).

4. **Frontend**  
   Single SPA: `app/page.tsx` holds `activeScreen`, `horizon`, `scenario` and renders one of 8 screens. `Layout.tsx` provides sidebar nav, top bar (horizon + scenario selectors), main area, footer, and the floating Chatbot. Each screen fetches from `/api/...` (rewritten to the FastAPI backend).

### File-by-file roles

| Path | Role |
|------|------|
| **Frontend** | |
| `app/layout.tsx` | Root layout, Inter font, globals.css |
| `app/page.tsx` | Screen state (executive, demand, recipe, …), passes horizon to screens |
| `app/globals.css` | Tailwind + CSS variables + scrollbar + Recharts overrides |
| `components/Layout.tsx` | Sidebar (nav, logo), header (horizon, scenario, status), main, footer, Chatbot |
| `components/Chatbot.tsx` | Floating button → drawer, messages, example questions, POST to backend `/api/chatbot/query` (currently hardcoded 127.0.0.1:8000; should use relative `/api` for consistency) |
| `components/screens/ExecutiveOverview.tsx` | KPIs (demand, recipe time, capacity, risk), recipe time vs capacity chart, AI insight panel. Fetches `/api/kpis/executive`, `/api/planning/recipe`, `/api/capacity/mill` |
| `components/screens/DemandForecast.tsx` | SKU forecast table + line chart by flour type. Fetches `/api/forecast/sku` |
| `components/screens/RecipePlanning.tsx` | Eligibility matrix, recipe time chart, recipe mix (Superior). Fetches `/api/planning/recipe`, `/api/planning/recipe-eligibility`. Chart uses **scheduled_hours** from API (not required_hours) |
| `components/screens/MillCapacity.tsx` | Capacity ledger table + utilization bar chart. Fetches `/api/capacity/mill` |
| `components/screens/RawMaterials.tsx` | Wheat price line chart + current prices table. Fetches `/api/raw-material` |
| `components/screens/Scenarios.tsx` | Scenario builder form + static comparison card (no API yet) |
| `components/screens/Alerts.tsx` | Alert list from `/api/alerts` (capacity overloads) |
| `components/screens/Reports.tsx` | Report cards (Email / Download) – UI only, no backend wiring yet |
| `tailwind.config.js` | primary + mc4 (blue, dark, light) color extensions |
| `next.config.js` | Rewrites `/api/*` → `http://localhost:8000/api/*` |
| **Backend** | |
| `fastapi_server.py` | All REST endpoints, CORS, CSV cache, chatbot proxy |
| `data_generator.py` | Generates all datasets |
| `forecast_models.py` | Prophet models (saved under backend/models/) |
| `setup_chatbot_db.py` | Builds Text2SQL DB from backend/datasets for chatbot |
| **Text2SQL_V2** | |
| `chatbot_api.py` | run_chatbot_query(question), schema from backend/datasets, SQLite in Text2SQL_V2 |
| `config.py` | LLM_PROVIDER, API keys from env |
| `agents/text2sql_agent.py` | NL → SQL |
| `agents/summarizer_agent.py` | Result → summary + optional viz |
| `schema_metadata.json` | Table/column descriptions for Text2SQL |

### Connection summary

- **Frontend ↔ Backend**: Next.js rewrites so that `fetch('/api/...')` hits FastAPI. Chatbot should use `/api/chatbot/query` for consistency.
- **Backend ↔ Text2SQL**: Backend imports and calls `run_chatbot_query`; no HTTP between them. Text2SQL reads CSVs from `backend/datasets/` and uses `backend/.env` (via config) for LLM keys.
- **Shared data**: Same CSVs drive both the REST API (in-memory) and the chatbot SQLite DB.

---

## 2. Senior Frontend / UI-UX Design Recommendations

### 2.1 Typography & hierarchy

- **Current**: Single font (Inter) everywhere; headings are mostly `text-2xl font-bold` and `text-lg font-semibold`.
- **Improve**:
  - Use a clear hierarchy: one font for UI (e.g. Inter or a slightly more characterful sans), and optionally a distinct display font for the app title / hero (e.g. “MC4 AI Command Center”).
  - Define tokens: `--font-sans`, `--font-display`, and scale (e.g. `--text-hero`, `--text-h1`… `--text-caption`).
  - Page titles: larger, with slightly more letter-spacing or weight so each screen is instantly recognizable.
  - Subsection titles: consistent size (e.g. `text-lg` or `text-base`) and color (e.g. `text-gray-700` or `--color-heading`).

### 2.2 Color & branding

- **Current**: `mc4-blue` (#0ea5e9), `mc4-dark`, `mc4-light` in Tailwind; generic gray backgrounds.
- **Improve**:
  - Expand the palette in `tailwind.config.js`: e.g. semantic colors (success, warning, error, info) and a neutral scale (50–950) for backgrounds and borders.
  - Use CSS variables in `globals.css` for surfaces (e.g. `--surface-card`, `--surface-page`) so dark mode or theme changes are one place.
  - Reserve one accent (e.g. mc4-blue) for primary actions and key metrics; use muted colors for secondary info so the dashboard doesn’t feel “all blue.”

### 2.3 Layout & spacing

- **Current**: `space-y-6` on pages; cards with `p-6`; some content feels tight.
- **Improve**:
  - Use a consistent spacing scale (e.g. 4, 6, 8, 12, 16, 24) and apply it to section gaps, card padding, and grid gaps.
  - Give the main content area a max-width on large screens (e.g. 1400–1600px) and center it for better scan-ability.
  - Sidebar: consider a slightly wider collapsed state (e.g. icons + short labels) so navigation stays understandable when collapsed.

### 2.4 Cards & surfaces

- **Current**: White cards, `shadow`, `border border-gray-200`.
- **Improve**:
  - Standardize card style: one token for background, border, radius, and shadow (e.g. `card` utility class).
  - Slightly softer shadows (e.g. `shadow-sm` / `shadow` with a neutral tint) and consistent radius (e.g. `rounded-xl`).
  - For KPI tiles, add a very subtle left border or icon background in the brand color to create a clear “metric” look and improve scan-ability.

### 2.5 Loading & empty states

- **Current**: “Loading...” text only; some screens show “No data” with minimal context.
- **Improve**:
  - Skeleton loaders for KPIs, tables, and charts (match the shape of content) so the layout doesn’t jump.
  - Empty states: illustration or icon + short message + optional action (e.g. “No alerts – you’re all set” with a checkmark; “No data for this period” with a tip to change horizon).
  - Disable or dim horizon/scenario controls during loading where it makes sense.

### 2.6 Charts & data viz

- **Current**: Recharts with CartesianGrid, Tooltip, Legend; some charts have long X labels (e.g. recipe names) and debug blocks in production.
- **Improve**:
  - Remove debug `<details>` and `console.log` from production (e.g. ExecutiveOverview “Debug: View chart data”).
  - Use a small, consistent palette for lines/bars (e.g. from Tailwind or a chart palette) and ensure sufficient contrast.
  - Truncate or rotate X-axis labels; consider “show more” or tooltip-only names for long recipe names.
  - ExecutiveOverview: ensure “Recipe Time Demand vs Capacity” reference line and legend clearly explain what “Avg Capacity” means (e.g. “Total capacity / # recipes” or a clearer metric).

### 2.7 Chatbot UX

- **Current**: Floating button, fixed size drawer, example questions, SQL/data in details.
- **Improve**:
  - Use relative URL for API: `POST /api/chatbot/query` (via Next rewrite) so it works in all environments.
  - Consider a slightly larger default drawer on desktop and a max-height so the input stays above the fold.
  - Differentiate user vs assistant bubbles (e.g. shape, alignment, color) and add a small “copy” on SQL/data blocks.
  - Optional: show a tiny “powered by Text2SQL” or model hint in the footer of the chat.

### 2.8 Accessibility & responsiveness

- **Current**: Semantic HTML is partial; focus states use `focus:ring-mc4-blue`.
- **Improve**:
  - Ensure all interactive elements have visible focus (keyboard) and that nav can be used with keyboard only.
  - Add `aria-label` or `aria-labelledby` where needed (e.g. horizon selector, scenario selector).
  - Tables: use `<th scope="col">` and consider a sticky header on scroll.
  - On small screens: collapse sidebar to icon-only or drawer; stack KPI tiles in a single column; allow charts to scroll horizontally if needed.

### 2.9 Footer & explainability

- **Current**: Footer with a single line of “How was this calculated?” steps.
- **Improve**:
  - Keep it concise; optionally make it collapsible or a tooltip so it doesn’t consume vertical space on every screen.
  - If you add more explainability later, link to a dedicated “Methodology” or “Help” page.

### 2.10 Scenarios & Reports

- **Current**: Scenarios form doesn’t call an API; Reports are UI-only.
- **Improve**:
  - When backend supports scenario run: wire “Run Scenario” to API and show results in the comparison panel.
  - For Reports: add “Email” / “Download” handlers (even if they trigger a “Coming soon” or a simple export) so the UI is ready for backend integration.

---

## 3. Quick wins already applied (or to apply)

- **RecipePlanning**: Chart should use `scheduled_hours` (API returns this), not `required_hours`.
- **Chatbot**: Use `axios.post('/api/chatbot/query', { question })` so the same rewrite and env work as the rest of the app.
- **Design system**: Introduce `globals.css` tokens (e.g. `--surface-card`, `--radius-card`, `--shadow-card`) and a `.card` class; use in all screens.
- **Loading**: Replace “Loading...” with a small skeleton or spinner component reused across screens.
- **Cleanup**: Remove debug `<details>` and unnecessary `console.log` from ExecutiveOverview (and any other screen).

---

## 4. Summary

- **Architecture**: Frontend (Next.js) talks to Backend (FastAPI) via `/api/*` rewrite; Backend uses Text2SQL in-process and the same CSV data for both REST and chatbot.
- **Design**: Introduce a clear typographic and color system, consistent cards and spacing, better loading/empty states, and small UX improvements (Chatbot API URL, chart field, remove debug). That will make the app feel more cohesive and production-ready while keeping your existing structure and branding.

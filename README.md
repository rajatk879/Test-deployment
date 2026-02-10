# MC4 Forecasting & Planning System

**MC4 AI Command Center** - Recipe-Time-Driven Planning System for Flour Milling Operations

## 🎯 Overview

This system addresses the core business reality of MC4 (Fourth Milling Company): **planning happens in recipe run-time per mill**, not at SKU level. The system translates SKU demand → Bulk Flour → Recipe → Mill Time, enabling data-driven production planning.

## 🏗️ Architecture

```
MC4/
├── backend/              # Python FastAPI backend
│   ├── data_generator.py    # Synthetic data generation
│   ├── forecast_models.py   # ML forecasting models
│   ├── fastapi_server.py    # API server
│   └── setup_chatbot_db.py  # Text2SQL setup
├── frontend/            # Next.js React frontend
│   ├── app/                 # Next.js app directory
│   └── components/          # UI components
└── Text2SQL_V2/         # Text-to-SQL chatbot module
```

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Node.js 18+
- npm or yarn

### Backend Setup

1. **Navigate to backend directory:**
```bash
cd backend
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Generate synthetic data:**
```bash
python data_generator.py
```
This creates all datasets in `datasets/` directory with:
- Time dimension (2020-2027) with Arabian calendar events
- SKU master (FOOM & Miller brands)
- Recipe master and allocation rules
- Mill master (Dammam, Medina, Al-Kharj)
- Raw material prices by country
- SKU forecasts with seasonality
- Derived planning tables

4. **Train forecast models:**
```bash
python forecast_models.py
```

5. **Setup Text2SQL database:**
```bash
python setup_chatbot_db.py
```

6. **Start FastAPI server:**
```bash
python fastapi_server.py
# Or: uvicorn fastapi_server:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### Frontend Setup

1. **Navigate to frontend directory:**
```bash
cd frontend
```

2. **Install dependencies:**
```bash
npm install
```

3. **Start development server:**
```bash
npm run dev
```

The UI will be available at `http://localhost:3000`

## 📊 Key Features

### 1. Executive Overview
- Real-time KPIs: Demand, Recipe Time, Capacity, Risk
- Visual recipe time vs capacity charts
- AI-powered insights

### 2. Demand & Forecast
- SKU-level forecasts aggregated by flour type
- Weekly/Monthly/Yearly views
- Trend visualization

### 3. Recipe Planning
- Recipe eligibility matrix (which recipes can produce which flour)
- Recipe time requirements
- Recipe mix selector with live recalculation

### 4. Mill Capacity
- Mill utilization dashboard
- Capacity ledger (available vs planned)
- Overload alerts

### 5. Raw Materials
- Wheat prices by country (Saudi Arabia, Egypt, Pakistan, Turkey, etc.)
- Price trends and availability

### 6. Scenarios & What-If
- Build custom scenarios
- Compare base vs scenario outcomes
- Impact analysis

### 7. Text-to-SQL Chatbot
- Natural language queries
- Recipe-time-centric questions
- Capacity and planning insights

### 8. Alerts & Reports
- Capacity overload alerts
- Email automation
- Report generation

## 🔑 Business Logic

### Core Planning Flow

```
SKU Forecast (Commercial)
    ↓
Bulk Flour Requirement
    ↓
Recipe Requirement (with allocation rules)
    ↓
Mill Time Requirement (hours/days)
    ↓
Capacity Check & Schedule
```

### Key Constraints

- **One recipe per mill at a time** (can split days)
- **Same flour can be produced by multiple recipes** (e.g., Superior: 80/70 or 80 Straight)
- **Capacity is time-based**, not just tonnage
- **Planning unit: Recipe run-time per mill per period**

### Seasonality

- **Ramadan**: 2.5x demand spike
- **Hajj**: 1.8x demand spike
- **Weekend effect**: Lower demand (Fri-Sat)
- **Annual seasonality**: Sinusoidal pattern

## 📡 API Endpoints

### Executive KPIs
```
GET /api/kpis/executive?horizon=month&period=2026-03
```

### Forecasting
```
GET /api/forecast/sku?horizon=month&period=2026-03&sku_id=SKU001
```

### Planning
```
GET /api/planning/recipe?horizon=month&period=2026-03
GET /api/planning/recipe-eligibility
```

### Capacity
```
GET /api/capacity/mill?horizon=month&period=2026-03&mill_id=M1
```

### Raw Materials
```
GET /api/raw-material?horizon=month&period=2026-03&country=Saudi Arabia
```

### Chatbot
```
POST /api/chatbot/query
{
  "question": "How many hours will we run recipe 80/70 next week?"
}
```

### Alerts
```
GET /api/alerts?horizon=week&period=2026-W10
```

## 🗄️ Data Model

### Core Tables

- `time_dimension` - Calendar with Arabian events
- `sku_master` - SKU catalog (FOOM, Miller brands)
- `recipe_master` - Recipe definitions with rates
- `recipe_allocation` - Recipe-to-flour mapping
- `mill_master` - Mill definitions (Dammam, Medina, Al-Kharj)
- `sku_forecast` - SKU demand forecasts
- `flour_demand` - Aggregated flour demand
- `recipe_time` - Recipe time requirements
- `mill_capacity` - Mill capacity
- `mill_load` - Mill load and utilization
- `raw_material` - Raw material prices by country

### Aggregated Views

- `recipe_time_weekly` - Weekly aggregation
- `recipe_time_monthly` - Monthly aggregation
- `recipe_time_yearly` - Yearly aggregation

## 🧠 Text-to-SQL Chatbot

The chatbot understands business questions and converts them to SQL:

**Example Queries:**
- "How many hours will we run recipe 80/70 next week?"
- "Which mills are overloaded next month?"
- "What is the forecast for Superior Flour 10kg next month?"
- "Show me recipe time utilization by mill"
- "Why is Mill A overloaded in Week 12?"

## 🎨 UI Design

- **Professional MC4 branding** with logo placeholder
- **Responsive design** for desktop and tablet
- **Real-time updates** via API
- **Interactive charts** (Recharts)
- **Context-aware chatbot** (right-side drawer)

## 📝 Environment Variables

Create `.env` file in `backend/`:

```env
LLM_PROVIDER=google  # or openai
GOOGLE_API_KEY=your_key_here
OPENAI_API_KEY=your_key_here  # if using OpenAI
```

## 🔧 Development

### Backend Development

```bash
cd backend
uvicorn fastapi_server:app --reload --port 8000
```

### Frontend Development

```bash
cd frontend
npm run dev
```

### Regenerate Data

```bash
cd backend
python data_generator.py
```

## 📈 Model Training

Forecast models use Prophet with:
- Yearly seasonality
- Weekly seasonality
- Arabian holidays (Ramadan, Hajj)
- Multiplicative seasonality mode

Models are saved in `backend/models/` (one per SKU).

## 🚨 Alerts & Notifications

The system generates alerts for:
- Mill capacity overload
- Recipe imbalance
- Raw material price spikes

Email automation can be configured via SendGrid (see `Text2SQL_V2/mailer.py`).

## 📚 Documentation

- `backend/README.md` - Backend API documentation
- `Text2SQL_V2/BUSINESS_QUERIES.md` - Business query examples
- `Text2SQL_V2/PLOT_QUERIES.md` - Visualization queries

## 🤝 Contributing

This is a demo system for MC4. For production deployment:
1. Replace synthetic data with real data sources
2. Configure proper authentication
3. Set up production database
4. Configure email service
5. Add monitoring and logging

## 📄 License

Internal use for MC4 (Fourth Milling Company)

---

**Built for MC4 - Recipe-Time-Driven Planning** 🏭

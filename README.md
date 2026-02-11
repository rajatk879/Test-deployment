# MC4 Backend - Forecasting & Planning API

## Setup

1. **Install dependencies:**
```bash
pip install -r requirements.txt
```

2. **Generate synthetic data:**
```bash
python data_generator.py
```

3. **Train forecast models:**
```bash
python forecast_models.py
```

4. **Setup Text2SQL database:**
```bash
python setup_chatbot_db.py
```

5. **Start FastAPI server:**
```bash
python fastapi_server.py
# Or with uvicorn:
uvicorn fastapi_server:app --reload --port 8000
```

## API Endpoints

### Health & Status
- `GET /` - API info
- `GET /health` - Health check

### Executive KPIs
- `GET /api/kpis/executive?horizon=month&period=2026-03` - Executive overview KPIs

### Forecasting
- `GET /api/forecast/sku?horizon=month&period=2026-03&sku_id=SKU001` - SKU forecasts

### Planning
- `GET /api/planning/recipe?horizon=month&period=2026-03` - Recipe time requirements
- `GET /api/planning/recipe-eligibility` - Recipe eligibility matrix

### Capacity
- `GET /api/capacity/mill?horizon=month&period=2026-03&mill_id=M1` - Mill capacity and load

### Raw Materials
- `GET /api/raw-material?horizon=month&period=2026-03&country=Saudi Arabia` - Raw material prices

### Chatbot
- `POST /api/chatbot/query` - Text-to-SQL chatbot
  ```json
  {
    "question": "How many hours will we run recipe 80/70 next week?"
  }
  ```

### Alerts
- `GET /api/alerts?horizon=week&period=2026-W10` - Capacity and planning alerts

## Data Structure

All data is stored in `datasets/` directory:
- `time_dimension.csv` - Time dimension with Arabian calendar events
- `sku_master.csv` - SKU catalog
- `recipe_master.csv` - Recipe definitions
- `recipe_allocation.csv` - Recipe-to-flour mapping
- `mill_master.csv` - Mill definitions
- `sku_forecast.csv` - SKU demand forecasts
- `flour_demand.csv` - Aggregated flour demand
- `recipe_time.csv` - Recipe time requirements
- `mill_capacity.csv` - Mill capacity
- `mill_load.csv` - Mill load and utilization
- `raw_material.csv` - Raw material prices by country

## Models

Forecast models are stored in `models/` directory (one per SKU).

# API Test Payloads for fastapi_server.py

This document contains example payloads and requests for testing the FastAPI server endpoints.

## POST Endpoints

### 1. Chatbot Query
**Endpoint:** `POST /api/chatbot/query`

**Payload:**
```json
{
  "question": "What is the total forecast demand for next month?"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8009/api/chatbot/query" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the total forecast demand for next month?"}'
```

---

### 2. Send Report via Email
**Endpoint:** `POST /api/reports/{report_id}/email`

**Available report_id values:**
- `monthly-plan`
- `capacity-outlook`
- `demand-forecast`
- `raw-material`

**Payload:**
```json
{
  "from_date": "2024-01-01",
  "to_date": "2024-03-31",
  "horizon": "month",
  "period": "2024-01"
}
```

**Minimal Payload (optional parameters):**
```json
{
  "horizon": "month"
}
```

**cURL Example:**
```bash
curl -X POST "http://localhost:8009/api/reports/monthly-plan/email" \
  -H "Content-Type: application/json" \
  -d '{
    "from_date": "2024-01-01",
    "to_date": "2024-03-31",
    "horizon": "month",
    "period": "2024-01"
  }'
```

---

## GET Endpoints (with Query Parameters)

### 3. Executive KPIs
**Endpoint:** `GET /api/kpis/executive`

**Query Parameters:**
- `from_date` (optional): "2024-01-01"
- `to_date` (optional): "2024-03-31"
- `horizon` (optional): "week" | "month" | "year"
- `period` (optional): "2024-01"
- `scenario` (optional): "base" | "ramadan" | "hajj" | "eid_fitr" | "eid_adha" | "summer" | "winter"

**cURL Example:**
```bash
curl "http://localhost:8009/api/kpis/executive?from_date=2024-01-01&to_date=2024-03-31&scenario=ramadan"
```

---

### 4. SKU Forecast
**Endpoint:** `GET /api/forecast/sku`

**Query Parameters:**
- `from_date` (optional): "2024-01-01"
- `to_date` (optional): "2024-03-31"
- `horizon` (optional): "week" | "month" | "year"
- `period` (optional): "2024-01"
- `sku_id` (optional): "SKU001"
- `scenario` (optional): "base" | "ramadan" | etc.

**cURL Example:**
```bash
curl "http://localhost:8009/api/forecast/sku?from_date=2024-01-01&to_date=2024-03-31&horizon=month&scenario=base"
```

---

### 5. Mill Capacity
**Endpoint:** `GET /api/capacity/mill`

**Query Parameters:**
- `from_date` (optional): "2024-01-01"
- `to_date` (optional): "2024-03-31"
- `horizon` (optional): "week" | "month" | "year"
- `period` (optional): "2024-01"
- `mill_id` (optional): "MILL001"
- `scenario` (optional): "base" | "ramadan" | etc.

**cURL Example:**
```bash
curl "http://localhost:8009/api/capacity/mill?from_date=2024-01-01&to_date=2024-03-31&mill_id=MILL001"
```

---

### 6. Recipe Planning
**Endpoint:** `GET /api/planning/recipe`

**Query Parameters:**
- `from_date` (optional): "2024-01-01"
- `to_date` (optional): "2024-03-31"
- `horizon` (optional): "week" | "month" | "year"
- `period` (optional): "2024-01"
- `scenario` (optional): "base" | "ramadan" | etc.

**cURL Example:**
```bash
curl "http://localhost:8009/api/planning/recipe?horizon=month&scenario=base"
```

---

### 7. Raw Material Prices
**Endpoint:** `GET /api/raw-material`

**Query Parameters:**
- `from_date` (optional): "2024-01-01"
- `to_date` (optional): "2024-03-31"
- `horizon` (optional): "week" | "month" | "year"
- `period` (optional): "2024-01"
- `country` (optional): "Saudi Arabia"
- `scenario` (optional): "base" | "ramadan" | etc.

**cURL Example:**
```bash
curl "http://localhost:8009/api/raw-material?from_date=2024-01-01&to_date=2024-03-31&country=Saudi%20Arabia"
```

---

### 8. Alerts
**Endpoint:** `GET /api/alerts`

**Query Parameters:**
- `from_date` (optional): "2024-01-01"
- `to_date` (optional): "2024-03-31"
- `horizon` (optional): "week" | "month" | "year"
- `period` (optional): "2024-01"

**cURL Example:**
```bash
curl "http://localhost:8009/api/alerts?from_date=2024-01-01&to_date=2024-03-31"
```

---

### 9. Health Check
**Endpoint:** `GET /health`

**cURL Example:**
```bash
curl "http://localhost:8009/health"
```

---

### 10. Date Range Meta
**Endpoint:** `GET /api/meta/date-range`

**cURL Example:**
```bash
curl "http://localhost:8009/api/meta/date-range"
```

---

## Other KPI Endpoints

### 11. Demand-Recipe KPIs
**Endpoint:** `GET /api/kpis/demand-recipe`

**cURL Example:**
```bash
curl "http://localhost:8009/api/kpis/demand-recipe?from_date=2024-01-01&to_date=2024-03-31&scenario=base"
```

### 12. Recipe Planning KPIs
**Endpoint:** `GET /api/kpis/recipe-planning`

**cURL Example:**
```bash
curl "http://localhost:8009/api/kpis/recipe-planning?scenario=ramadan"
```

### 13. Mill Operations KPIs
**Endpoint:** `GET /api/kpis/mill-operations`

**cURL Example:**
```bash
curl "http://localhost:8009/api/kpis/mill-operations?from_date=2024-01-01&to_date=2024-03-31"
```

### 14. Raw Materials KPIs
**Endpoint:** `GET /api/kpis/raw-materials`

**cURL Example:**
```bash
curl "http://localhost:8009/api/kpis/raw-materials?scenario=base"
```

### 15. Sustainability KPIs
**Endpoint:** `GET /api/kpis/sustainability`

**cURL Example:**
```bash
curl "http://localhost:8009/api/kpis/sustainability?from_date=2024-01-01&to_date=2024-03-31"
```

---

## Using Python requests

```python
import requests

# POST example - Chatbot
response = requests.post(
    "http://localhost:8009/api/chatbot/query",
    json={"question": "What is the forecast for next month?"}
)
print(response.json())

# POST example - Email Report
response = requests.post(
    "http://localhost:8009/api/reports/monthly-plan/email",
    json={
        "from_date": "2024-01-01",
        "to_date": "2024-03-31",
        "horizon": "month"
    }
)
print(response.json())

# GET example - Executive KPIs
response = requests.get(
    "http://localhost:8009/api/kpis/executive",
    params={
        "from_date": "2024-01-01",
        "to_date": "2024-03-31",
        "scenario": "ramadan"
    }
)
print(response.json())
```

---

## Notes

- Default server port: `8009`
- All dates should be in `YYYY-MM-DD` format
- Scenarios: `base`, `ramadan`, `hajj`, `eid_fitr`, `eid_adha`, `summer`, `winter`
- Horizons: `week`, `month`, `year`
- Most endpoints return JSON with a `data` key containing an array of records
- KPI endpoints return JSON objects with specific KPI metrics

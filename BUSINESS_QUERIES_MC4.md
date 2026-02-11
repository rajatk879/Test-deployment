# MC4 Domain Business Queries for Text2SQL Chatbot

These queries are designed for the MC4 flour milling and production planning domain. They cover mill operations, recipe planning, SKU forecasts, flour demand, and raw material (wheat) supply.

---

## 1. SKU Forecast Analysis
**Query:** "What is the total forecasted tons for SKU001 in January 2020?"

**Business Context:** Demand planners need to understand forecasted demand for specific SKUs to plan production and inventory.

**Expected SQL Logic:**
- Filter `sku_forecast` by `sku_id = 'SKU001'` and `date BETWEEN '2020-01-01' AND '2020-01-31'`
- Sum `forecast_tons`

---

## 2. Mill Utilization Analysis
**Query:** "Show me the utilization percentage for each mill in January 2020"

**Business Context:** Operations managers need to track mill utilization to optimize production capacity and identify bottlenecks.

**Expected SQL Logic:**
- Filter `mill_load` by date range for January 2020
- Group by `mill_id`
- Show `mill_id` and average or latest `utilization_pct`
- Join with `mill_master` to get mill names

---

## 3. Flour Type Demand Comparison
**Query:** "Compare total bulk flour demand by flour type for January 2020"

**Business Context:** Production planners need to understand demand distribution across different flour types to allocate production capacity.

**Expected SQL Logic:**
- Filter `bulk_flour_demand` by date range for January 2020
- Group by `flour_type`
- Sum `required_bulk_tons`

---

## 4. Recipe Production Schedule
**Query:** "What recipes were scheduled at M1 mill in January 2020 and how many tons were produced?"

**Business Context:** Production managers need to track which recipes are being run and their production output.

**Expected SQL Logic:**
- Filter `mill_recipe_schedule` by `mill_id = 'M1'` and date range for January 2020
- Group by `recipe_id`
- Sum `tons_produced`
- Join with `recipe_master` to get recipe names

---

## 5. Mill Capacity vs Load
**Query:** "Show me the scheduled hours vs available hours for each mill in January 2020"

**Business Context:** Capacity planners need to compare scheduled production hours against available capacity to identify overload situations.

**Expected SQL Logic:**
- Filter `mill_load` by date range for January 2020
- Group by `mill_id`
- Sum `scheduled_hours` and `available_hours`
- Calculate utilization: `SUM(scheduled_hours) / SUM(available_hours) * 100`

---

## 6. Recipe Demand by Flour Type
**Query:** "What is the recipe demand for Superior flour in January 2020?"

**Business Context:** Recipe planners need to understand how much of each recipe is required to meet flour type demand.

**Expected SQL Logic:**
- Filter `recipe_demand` by `flour_type LIKE '%Superior%'` (case-insensitive) and date range for January 2020
- Group by `recipe_id`
- Sum `recipe_required_tons`
- Join with `recipe_master` for recipe names

---

## 7. Raw Material Price Analysis
**Query:** "What is the average wheat price per ton by country in January 2020?"

**Business Context:** Procurement teams need to compare wheat prices across countries to optimize sourcing decisions.

**Expected SQL Logic:**
- Filter `raw_material` by date range for January 2020
- Group by `country`
- Calculate AVG(`wheat_price_sar_per_ton`)

---

## 8. Mill Overload Analysis
**Query:** "Which mills had overload hours in January 2020?"

**Business Context:** Operations managers need to identify mills operating beyond capacity to take corrective action.

**Expected SQL Logic:**
- Filter `mill_load` by date range for January 2020
- Where `overload_hours > 0`
- Group by `mill_id`
- Sum `overload_hours`
- Join with `mill_master` for mill names

---

## 9. Recipe Production Rates
**Query:** "What are the production rates (tons per hour) for each recipe at M1 mill?"

**Business Context:** Production planners need to know production rates to estimate production time and capacity requirements.

**Expected SQL Logic:**
- Filter `mill_recipe_rates` by `mill_id = 'M1'`
- Join with `recipe_master` to get recipe names
- Show `recipe_name`, `tons_per_hour`

---

## 10. SKU Forecast by Flour Type
**Query:** "What is the total forecasted tons for Superior flour SKUs in January 2020?"

**Business Context:** Demand planners need aggregated forecasts by flour type to plan production and inventory.

**Expected SQL Logic:**
- Join `sku_forecast` with `sku_master` on `sku_id`
- Filter by `flour_type LIKE '%Superior%'` (case-insensitive) and date range for January 2020
- Sum `forecast_tons`

---

## Visualization Queries

### 11. Forecast Trend Over Time
**Query:** "Show me a line chart of forecasted tons for SKU001 over time in January 2020"

**Expected SQL Logic:**
- Filter `sku_forecast` by `sku_id = 'SKU001'` and date range
- Order by `date`
- Show `date` and `forecast_tons`

**Chart Type:** Line chart

---

### 12. Mill Utilization Comparison
**Query:** "Plot a bar chart comparing utilization percentage across all mills in January 2020"

**Expected SQL Logic:**
- Filter `mill_load` by date range for January 2020
- Group by `mill_id`
- Calculate average `utilization_pct`
- Join with `mill_master` for mill names

**Chart Type:** Bar chart

---

### 13. Flour Type Demand Breakdown
**Query:** "Create a pie chart showing bulk flour demand by flour type in January 2020"

**Expected SQL Logic:**
- Filter `bulk_flour_demand` by date range for January 2020
- Group by `flour_type`
- Sum `required_bulk_tons`

**Chart Type:** Pie chart

---

### 14. Recipe Production Comparison
**Query:** "Visualize total tons produced by recipe at M1 mill in January 2020 as a bar chart"

**Expected SQL Logic:**
- Filter `mill_recipe_schedule` by `mill_id = 'M1'` and date range
- Group by `recipe_id`
- Sum `tons_produced`
- Join with `recipe_master` for recipe names

**Chart Type:** Bar chart

---

### 15. Weekly Mill Load Trend
**Query:** "Show me a line chart of scheduled hours by week for M1 mill"

**Expected SQL Logic:**
- Filter `mill_load_weekly` by `mill_id = 'M1'`
- Order by `week`
- Show `week` and `scheduled_hours`

**Chart Type:** Line chart

---

## Notes for Testing

1. **Date Formats:** Use 'YYYY-MM-DD' format (e.g., '2020-01-01')
2. **Mill IDs:** Use exact format 'M1', 'M2', 'M3' (case sensitive)
3. **Recipe IDs:** Use exact format 'R1', 'R2', 'R3', etc. (case sensitive)
4. **SKU IDs:** Use exact format 'SKU001', 'SKU002', etc. (case sensitive)
5. **Flour Types:** Use 'Superior', 'Bakery', 'Patent', 'Brown', 'Superior Brown' (case-insensitive match)
6. **Period Formats:** Use 'YYYY-MM' for months, 'YYYY-W##' for weeks, YYYY for years

## Query Complexity Levels

- **Simple (1-5):** Single table queries with basic filters and aggregations
- **Medium (6-10):** Multi-table joins or complex filtering
- **Advanced (11+):** Complex joins, calculations, or visualization queries

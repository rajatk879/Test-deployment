# 10 Business-Relevant Queries for the Chatbot

These queries are designed to be simple, natural language questions that business users would ask. They cover various aspects of the business including sales, inventory, forecasting, and raw materials.

---

## 1. Sales Performance Analysis
**Query:** "What were the total sales units for Store_01 in December 2025?"

**Business Context:** Store managers need to track monthly sales performance to understand store effectiveness and plan for the next period.

**Expected SQL Logic:** 
- Filter `sku_daily_sales` by `store_id = 'Store_01'` and `date BETWEEN '2025-12-01' AND '2025-12-31'`
- Sum `actual_sales_units`

---

## 2. Top Performing SKUs
**Query:** "Show me the top 5 best-selling SKUs by total sales units across all stores"

**Business Context:** Product managers need to identify bestsellers to optimize inventory and marketing efforts.

**Expected SQL Logic:**
- Group `sku_daily_sales` by `sku_id`
- Sum `actual_sales_units`
- Order by total descending
- Limit to top 5

---

## 3. Inventory Status Check
**Query:** "What is the current closing inventory for Leather_FG raw material?"

**Business Context:** Supply chain managers need to check current inventory levels to plan procurement.

**Expected SQL Logic:**
- Filter `raw_material_inventory` by `raw_material = 'Leather_FG'` and latest date
- Select `closing_inventory`

---

## 4. Forecast Analysis
**Query:** "What is the 7-day sales forecast for WL-SKU-001 at Store_01?"

**Business Context:** Sales and operations teams need short-term forecasts to plan inventory and staffing.

**Expected SQL Logic:**
- Filter `sku_daily_forecast_7day` by `sku_id = 'WL-SKU-001'`, `store_id = 'Store_01'`
- Compute forecast_date using date + forecast_horizon
- Sum forecast_units for the 7-day period

---

## 5. Channel Performance Comparison
**Query:** "Compare total sales units between E-Commerce and Offline Retail channels for the last month"

**Business Context:** Marketing teams need to understand channel performance to allocate budget and resources.

**Expected SQL Logic:**
- Filter `sku_daily_sales` by date range (last month)
- Group by `sales_channel`
- Sum `actual_sales_units`

---

## 6. Low Inventory Alert
**Query:** "Which raw materials have closing inventory below safety stock level?"

**Business Context:** Procurement teams need alerts for materials at risk of stockout to trigger reordering.

**Expected SQL Logic:**
- Filter `raw_material_inventory` for latest date
- Where `closing_inventory < safety_stock`
- Show `raw_material`, `closing_inventory`, `safety_stock`

---

## 7. Product Category Sales
**Query:** "What are the total sales units by product category?"

**Business Context:** Category managers need to understand which product categories drive revenue.

**Expected SQL Logic:**
- Join `sku_daily_sales` with `sku_master` on `sku_id`
- Group by `category`
- Sum `actual_sales_units`

---

## 8. Raw Material Demand Forecast
**Query:** "What is the 30-day forecasted demand for EVA_Foam raw material?"

**Business Context:** Supply chain planners need material demand forecasts to plan procurement and production.

**Expected SQL Logic:**
- Filter `raw_material_demand` by `raw_material = 'EVA_Foam'` and `forecast_horizon = '30day'`
- Sum `material_demand_units`

---

## 9. Store Comparison
**Query:** "Compare total sales units for Store_01, Store_02, and Store_03 in November 2025"

**Business Context:** Regional managers need to compare store performance to identify best practices and areas for improvement.

**Expected SQL Logic:**
- Filter `sku_daily_sales` by `store_id IN ('Store_01', 'Store_02', 'Store_03')` and date range for November 2025
- Group by `store_id`
- Sum `actual_sales_units`

---

## 10. Promotion Impact Analysis
**Query:** "What was the difference in sales units for promoted vs non-promoted SKUs in December 2025?"

**Business Context:** Marketing teams need to measure promotion effectiveness to optimize promotional strategies.

**Expected SQL Logic:**
- Filter `sku_daily_sales` by date range (December 2025)
- Group by `promotion_flag`
- Sum `actual_sales_units`
- Compare results

---

## Additional Complex Queries (Advanced)

### 11. Material Requirements Planning
**Query:** "For product WL-PROD-101, what raw materials are needed and how much of each?"

**Business Context:** Production planners need BOM information to calculate material requirements.

**Expected SQL Logic:**
- Filter `product_bom` by `product_id = 'WL-PROD-101'`
- Show `raw_material`, `material_type`, `consumption_per_unit`

---

### 12. Forecast vs Actual Comparison
**Query:** "Compare 7-day forecasted sales with actual sales for WL-SKU-001 at Store_01"

**Business Context:** Demand planners need to measure forecast accuracy to improve forecasting models.

**Expected SQL Logic:**
- Join `sku_daily_forecast_7day` with `sku_daily_sales` on sku_id, store_id, and computed forecast_date
- Compare forecast_units vs actual_sales_units

---

### 13. Inventory Turnover
**Query:** "What is the consumption rate for Leather_FG in the last 30 days?"

**Business Context:** Inventory managers need to understand consumption patterns to optimize stock levels.

**Expected SQL Logic:**
- Filter `raw_material_inventory` by `raw_material = 'Leather_FG'` and date range (last 30 days)
- Sum `consumed_quantity`

---

### 14. Multi-Store Forecast Summary
**Query:** "What is the total 30-day forecast for all stores combined?"

**Business Context:** Executives need aggregate forecasts for strategic planning and resource allocation.

**Expected SQL Logic:**
- Filter `sku_daily_forecast_30day` 
- Sum `forecast_units` across all stores and SKUs

---


**Expected SQL Logic:**
- Filter `raw_material_inventory` for latest date
- Where `closing_inventory < (safety_stock * 1.5)`
- Show `raw_material`, `closing_inventory`, `safety_stock`

---

## Notes for Testing

1. **Date Formats:** Use 'YYYY-MM-DD' format (e.g., '2025-12-31')
2. **Store IDs:** Use exact format 'Store_XX' (e.g., 'Store_01')
3. **SKU IDs:** Use exact format 'WL-SKU-XXX' (e.g., 'WL-SKU-001')
4. **Product IDs:** Use exact format 'WL-PROD-XXX' (e.g., 'WL-PROD-101')
5. **Raw Materials:** Use exact names like 'Leather_FG', 'EVA_Foam', 'Rubber_Sole'
6. **Sales Channels:** Use exact values 'E-Commerce' or 'Offline Retail'

## Query Complexity Levels

- **Simple (1-5):** Single table queries with basic filters and aggregations
- **Medium (6-9):** Multi-table joins or complex filtering
- **Advanced (10+):** Complex joins, calculations, or multi-step logic

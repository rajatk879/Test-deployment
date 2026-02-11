# 5 Simple Plot Queries for the Chatbot

These queries are designed to generate visualizations (charts/plots) when asked to the chatbot. Each query includes visualization keywords that trigger chart generation.

---

## 1. Sales Trend Over Time
**Query:** "Show me a line chart of daily sales units for Store_01 in December 2025"

**Business Context:** Store managers need to see sales trends over time to identify patterns, peak days, and performance trends.

**Expected SQL Logic:**
- Filter `sku_daily_sales` by `store_id = 'Store_01'` and `date BETWEEN '2025-12-01' AND '2025-12-31'`
- Group by `date`
- Sum `actual_sales_units`
- Order by `date`

**Chart Type:** Line chart (auto-detected from "line chart" keyword)

**Why it's useful:** Visualizes daily sales patterns, helps identify weekends, promotions, or unusual spikes.

---

## 2. Top SKUs Comparison
**Query:** "Plot a bar chart comparing total sales units for the top 5 SKUs"

**Business Context:** Product managers need to visually compare bestsellers to understand product performance hierarchy.

**Expected SQL Logic:**
- Group `sku_daily_sales` by `sku_id`
- Sum `actual_sales_units`
- Order by total descending
- Limit to top 5

**Chart Type:** Bar chart (auto-detected from "bar chart" keyword)

**Why it's useful:** Easy visual comparison of top performers, helps prioritize inventory and marketing efforts.

---

## 3. Channel Performance Comparison
**Query:** "Visualize total sales units by sales channel as a bar chart"

**Business Context:** Marketing teams need to compare E-Commerce vs Offline Retail performance to allocate resources.

**Expected SQL Logic:**
- Group `sku_daily_sales` by `sales_channel`
- Sum `actual_sales_units`

**Chart Type:** Bar chart (auto-detected from "visualize" and "bar chart" keywords)

**Why it's useful:** Quick visual comparison of channel performance, helps with budget allocation decisions.

---

## 4. Raw Material Inventory Levels
**Query:** "Show me a bar chart of closing inventory for all raw materials"

**Business Context:** Supply chain managers need to see inventory levels across all materials at a glance to identify imbalances.

**Expected SQL Logic:**
- Filter `raw_material_inventory` for latest date
- Select `raw_material` and `closing_inventory`
- Order by `closing_inventory` descending

**Chart Type:** Bar chart (auto-detected from "bar chart" keyword)

**Why it's useful:** Visual overview of inventory health, quickly identifies overstocked or understocked materials.

---

## 5. Forecast Distribution
**Query:** "Plot a histogram showing the distribution of forecast units across all SKUs"

**Business Context:** Demand planners need to understand the distribution of forecast values to identify outliers and forecast patterns.

**Expected SQL Logic:**
- Select `forecast_units` from `sku_daily_forecast_7day` (or `sku_daily_forecast_30day`)
- Return all forecast_units values

**Chart Type:** Histogram (auto-detected from "histogram" keyword)

**Why it's useful:** Shows forecast distribution, helps identify if forecasts are concentrated in certain ranges or have outliers.

---

## Additional Plot Queries (Advanced)

### 6. Sales by Category
**Query:** "Create a pie chart showing sales units by product category"

**Expected SQL Logic:**
- Join `sku_daily_sales` with `sku_master` on `sku_id`
- Group by `category`
- Sum `actual_sales_units`

**Chart Type:** Pie chart (auto-detected from "pie chart" keyword)

---

### 7. Store Comparison
**Query:** "Draw a bar chart comparing total sales for Store_01, Store_02, and Store_03"

**Expected SQL Logic:**
- Filter `sku_daily_sales` by `store_id IN ('Store_01', 'Store_02', 'Store_03')`
- Group by `store_id`
- Sum `actual_sales_units`

**Chart Type:** Bar chart (auto-detected from "bar chart" keyword)

---

### 8. Material Type Distribution
**Query:** "Visualize raw material demand by material type as a bar chart"

**Expected SQL Logic:**
- Join `raw_material_demand` with `product_bom` on `raw_material` to get `material_type`
- Group by `material_type`
- Sum `material_demand_units`

**Chart Type:** Bar chart (auto-detected from "visualize" and "bar chart" keywords)

---

## Tips for Using Plot Queries

1. **Include visualization keywords:** Use words like "chart", "plot", "graph", "visualize", "bar chart", "line chart", "histogram", "pie chart"

2. **Be specific about what to plot:** Mention the metric (sales units, inventory, forecast units) and grouping (by date, by SKU, by store, etc.)

3. **Specify date ranges when needed:** For time-series charts, include date filters

4. **Use aggregation keywords:** Words like "total", "sum", "average" help the bot understand aggregation needs

5. **Example patterns:**
   - "Show me a [chart type] of [metric] by [dimension]"
   - "Plot [metric] for [filter condition]"
   - "Visualize [metric] grouped by [dimension]"

---

## Chart Type Detection

The chatbot automatically detects chart type based on keywords in your query:

- **Line chart:** "line", "trend", "time series"
- **Bar chart:** "bar", "compare", "comparison"
- **Scatter plot:** "scatter", "relationship", "correlation"
- **Histogram:** "hist", "distribution"
- **Pie chart:** "pie"

If no specific type is mentioned, the bot will auto-select based on the data structure.

---

*End of Plot Queries Guide*

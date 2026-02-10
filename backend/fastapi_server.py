"""
MC4 FastAPI Server v3 — Star Schema Backend
============================================
Loads the 5-layer dimensional model produced by data_generator v3 and
exposes KPI + data endpoints that the frontend consumes.

Every page in the UI has a dedicated /api/kpis/<page> endpoint that
returns pre-computed KPIs.  Data endpoints (/api/forecast/sku, etc.)
join dim/map/fact tables to produce the flat shapes the frontend expects.
"""

import os
import sys

# Load .env from backend directory first (so GOOGLE_API_KEY etc. are set before Text2SQL imports)
_backend_dir = os.path.dirname(os.path.abspath(__file__))
_env_path = os.path.join(_backend_dir, ".env")
if os.path.exists(_env_path):
    from dotenv import load_dotenv
    load_dotenv(_env_path)

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
import pandas as pd
import numpy as np
from datetime import datetime
import os
import sys
import io
import base64

try:
    from sendgrid import SendGridAPIClient
    from sendgrid.helpers.mail import Mail, Attachment, FileContent, FileName, FileType, Disposition
    SENDGRID_AVAILABLE = True
except ImportError:
    SENDGRID_AVAILABLE = False

from dotenv import load_dotenv
load_dotenv()

# Add parent directory for Text2SQL import
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import Text2SQL chatbot (optional: if API key missing or import fails, server still starts; chatbot returns 503)
run_chatbot_query = None
CHATBOT_AVAILABLE = False
try:
    text2sql_path = os.path.join(parent_dir, "Text2SQL_V2")
    if text2sql_path not in sys.path:
        sys.path.insert(0, text2sql_path)
    from chatbot_api import run_chatbot_query as _run_chatbot_query
    run_chatbot_query = _run_chatbot_query
    CHATBOT_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Text2SQL chatbot not available: {e}")
    print("   Set GOOGLE_API_KEY (or OPENAI_API_KEY) in backend/.env to enable the chatbot. Other API endpoints will work.")
try:
    text2sql_path = os.path.join(parent_dir, "Text2SQL_V2")
    if text2sql_path not in sys.path:
        sys.path.insert(0, text2sql_path)
    from chatbot_api import run_chatbot_query as _run_chatbot_query
    run_chatbot_query = _run_chatbot_query
    CHATBOT_AVAILABLE = True
except Exception as e:
    print(f"⚠️ Text2SQL chatbot not available: {e}")
    print("   Set GOOGLE_API_KEY (or OPENAI_API_KEY) in backend/.env to enable the chatbot. Other API endpoints will work.")

# ═══════════════════════════════════════════════════════════════════════════════
app = FastAPI(title="MC4 Forecasting API", version="3.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DATA_DIR = "datasets"
data_cache = {}

# ═══════════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _load(name):
    path = f"{DATA_DIR}/{name}.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()


def load_data():
    global data_cache
    try:
        data_cache = {
            # Layer 1
            "dim_mill": _load("dim_mill"),
            "dim_recipe": _load("dim_recipe"),
            "dim_flour_type": _load("dim_flour_type"),
            "dim_sku": _load("dim_sku"),
            "dim_wheat_type": _load("dim_wheat_type"),
            "dim_country": _load("dim_country"),
            # Layer 2
            "map_flour_recipe": _load("map_flour_recipe"),
            "map_recipe_mill": _load("map_recipe_mill"),
            "map_sku_flour": _load("map_sku_flour"),
            "map_recipe_wheat": _load("map_recipe_wheat"),
            "map_wheat_country": _load("map_wheat_country"),
            # Layer 3
            "fact_sku_forecast": _load("fact_sku_forecast"),
            "fact_bulk_flour": _load("fact_bulk_flour_requirement"),
            "fact_recipe_demand": _load("fact_recipe_demand"),
            "fact_mill_capacity": _load("fact_mill_capacity"),
            "fact_schedule": _load("fact_mill_schedule_daily"),
            "fact_plan": _load("fact_mill_recipe_plan"),
            "fact_wheat_req": _load("fact_wheat_requirement"),
            "fact_waste": _load("fact_waste_metrics"),
            "raw_material_prices": _load("raw_material_prices"),
            # Layer 4
            "fact_kpi": _load("fact_kpi_snapshot"),
            # Time
            "time_dimension": _load("time_dimension"),
        }
        # Parse dates once
        for key in ["fact_sku_forecast", "fact_bulk_flour", "fact_recipe_demand",
                     "fact_mill_capacity", "fact_schedule", "raw_material_prices"]:
            df = data_cache.get(key, pd.DataFrame())
            if not df.empty and "date" in df.columns:
                data_cache[key]["date"] = pd.to_datetime(df["date"])
        print("✅ Data loaded (MC4 v3 star-schema)")
    except Exception as e:
        print(f"⚠️ Error loading data: {e}")
        data_cache = {}


@app.on_event("startup")
async def startup_event():
    load_data()


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

SCENARIO_MULTIPLIERS = {
    "base": 1.0,
    "ramadan": 1.35,
    "hajj": 1.25,
    "eid_fitr": 1.20,
    "eid_adha": 1.22,
    "summer": 1.15,
    "winter": 0.95,
}


def _mult(scenario: str) -> float:
    return SCENARIO_MULTIPLIERS.get(scenario or "base", 1.0)


def _date_range(from_date, to_date):
    if from_date and to_date:
        return pd.to_datetime(from_date), pd.to_datetime(to_date)
    # Fallback: last 3 months
    now = pd.Timestamp.now()
    return now - pd.DateOffset(months=3), now


def _filter_dates(df, from_dt, to_dt, date_col="date"):
    if df.empty or date_col not in df.columns:
        return df
    return df[(df[date_col] >= from_dt) & (df[date_col] <= to_dt)].copy()


def _add_period(df, horizon="month", date_col="date"):
    """Add a 'period' column based on the requested horizon."""
    if df.empty or date_col not in df.columns:
        return df
    if horizon == "week":
        df["period"] = (
            df[date_col].dt.isocalendar().year.astype(str) + "-W"
            + df[date_col].dt.isocalendar().week.astype(str).str.zfill(2)
        )
    elif horizon == "month":
        df["period"] = df[date_col].dt.to_period("M").astype(str)
    else:
        df["period"] = df[date_col].dt.year.astype(str)
    return df


class ChatbotQuery(BaseModel):
    question: str


class EmailRequest(BaseModel):
    report_id: str


# ═══════════════════════════════════════════════════════════════════════════════
# HEALTH / META
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/")
async def root():
    return {"message": "MC4 Forecasting API", "status": "running", "version": "3.0.0"}


@app.get("/health")
async def health():
    return {"status": "healthy", "data_loaded": len(data_cache) > 0}


@app.get("/api/meta/date-range")
async def get_date_range():
    try:
        date_ranges = []
        for key in ["fact_sku_forecast", "raw_material_prices", "fact_schedule"]:
            df = data_cache.get(key, pd.DataFrame())
            if not df.empty and "date" in df.columns:
                s = pd.to_datetime(df["date"])
                date_ranges.append((s.min(), s.max()))
        if not date_ranges:
            return {"min_date": None, "max_date": None}
        return {
            "min_date": min(s for s, _ in date_ranges).date().isoformat(),
            "max_date": max(e for _, e in date_ranges).date().isoformat(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# KPI ENDPOINTS (one per UI page)
# ═══════════════════════════════════════════════════════════════════════════════

# ---------- Executive Summary KPIs ----------------------------------------

@app.get("/api/kpis/executive")
async def get_executive_kpis(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    horizon: Optional[str] = Query(None, pattern="^(week|month|year)$"),
    period: Optional[str] = None,
    scenario: Optional[str] = Query("base"),
):
    """
    Returns 6 executive KPIs:
      demand, recipe_time, capacity, risk, waste, vision2030
    """
    try:
        scenario = scenario or "base"
        m = _mult(scenario)
        from_dt, to_dt = _date_range(from_date, to_date)

        # ── Demand ─────────────────────────────────────────────────────
        fcast = _filter_dates(data_cache.get("fact_sku_forecast", pd.DataFrame()), from_dt, to_dt)
        current_demand = float(fcast["forecast_tons"].sum()) * m if not fcast.empty else 0

        period_days = (to_dt - from_dt).days
        prev_fcast = _filter_dates(
            data_cache.get("fact_sku_forecast", pd.DataFrame()),
            from_dt - pd.Timedelta(days=period_days),
            from_dt - pd.Timedelta(days=1),
        )
        prev_demand = float(prev_fcast["forecast_tons"].sum()) if not prev_fcast.empty else 0
        demand_growth = ((current_demand - prev_demand) / prev_demand * 100) if prev_demand > 0 else 0

        # ── Recipe Time ───────────────────────────────────────────────
        sched = _filter_dates(data_cache.get("fact_schedule", pd.DataFrame()), from_dt, to_dt)
        cap = _filter_dates(data_cache.get("fact_mill_capacity", pd.DataFrame()), from_dt, to_dt)

        sched_hours = float(sched["planned_hours"].sum()) * m if not sched.empty else 0
        avail_hours = float(cap["available_hours"].sum()) if not cap.empty else 0
        utilization = (sched_hours / avail_hours * 100) if avail_hours > 0 else 0

        # ── Capacity ──────────────────────────────────────────────────
        overload_mills = 0
        if not sched.empty and not cap.empty:
            daily_load = sched.groupby(["date", "mill_id"])["planned_hours"].sum().reset_index()
            daily_load["planned_hours"] *= m
            daily_load = daily_load.merge(cap, on=["date", "mill_id"], how="left")
            overloaded = daily_load[daily_load["planned_hours"] > daily_load["available_hours"]]
            overload_mills = int(overloaded["mill_id"].nunique())

        # ── Risk (wheat price) ────────────────────────────────────────
        rp = _filter_dates(data_cache.get("raw_material_prices", pd.DataFrame()), from_dt, to_dt)
        avg_price = float(rp["wheat_price_sar_per_ton"].mean()) if not rp.empty else 0

        prev_rp = _filter_dates(
            data_cache.get("raw_material_prices", pd.DataFrame()),
            from_dt - pd.Timedelta(days=period_days),
            from_dt - pd.Timedelta(days=1),
        )
        prev_price = float(prev_rp["wheat_price_sar_per_ton"].mean()) if not prev_rp.empty else avg_price
        price_change = ((avg_price - prev_price) / prev_price * 100) if prev_price > 0 else 0

        # ── Waste ─────────────────────────────────────────────────────
        waste_df = data_cache.get("fact_waste", pd.DataFrame())
        if not waste_df.empty and "period" in waste_df.columns:
            # filter monthly waste by overlapping periods
            start_p = from_dt.to_period("M").strftime("%Y-%m")
            end_p = to_dt.to_period("M").strftime("%Y-%m")
            wm = waste_df[(waste_df["period"] >= start_p) & (waste_df["period"] <= end_p)]
            waste_rate = float(wm["waste_pct"].mean()) if not wm.empty else 4.2
        else:
            wm = pd.DataFrame()
            waste_rate = 4.2
        prev_start_p = (from_dt - pd.DateOffset(months=1)).to_period("M").strftime("%Y-%m")
        prev_end_p = (to_dt - pd.DateOffset(months=1)).to_period("M").strftime("%Y-%m")
        if not waste_df.empty:
            pwm = waste_df[(waste_df["period"] >= prev_start_p) & (waste_df["period"] <= prev_end_p)]
            prev_waste = float(pwm["waste_pct"].mean()) if not pwm.empty else waste_rate
        else:
            prev_waste = waste_rate
        waste_delta = round(waste_rate - prev_waste, 2)

        # ── Vision 2030 ──────────────────────────────────────────────
        dim_mill = data_cache.get("dim_mill", pd.DataFrame())
        energy_eff = float(dim_mill["energy_efficiency_index"].mean()) if not dim_mill.empty else 75
        water_eff = float(dim_mill["water_efficiency_index"].mean()) if not dim_mill.empty else 72
        waste_target = 3.9
        waste_score = max(0, min(100, (1 - (waste_rate / waste_target - 1)) * 100))
        v2030 = round(waste_score * 0.30 + energy_eff * 0.30 + water_eff * 0.25 + 85 * 0.15, 1)

        kpi = data_cache.get("fact_kpi", pd.DataFrame())
        if not kpi.empty:
            v2030_rows = kpi[kpi["kpi_name"] == "vision_2030_index"]
            if not v2030_rows.empty:
                prev_v = v2030_rows.sort_values("period").iloc[-2]["value"] if len(v2030_rows) > 1 else v2030
            else:
                prev_v = v2030
        else:
            prev_v = v2030
        v2030_delta = round(v2030 - float(prev_v), 1)

        return {
            "demand": {"total_tons": round(current_demand, 2), "growth_pct": round(demand_growth, 2)},
            "recipe_time": {"total_hours": round(sched_hours, 2), "utilization_pct": round(utilization, 2)},
            "capacity": {"utilization_pct": round(utilization, 2), "overload_mills": overload_mills},
            "risk": {"avg_wheat_price": round(avg_price, 2), "price_change_pct": round(price_change, 2)},
            "waste": {"waste_rate_pct": round(waste_rate, 2), "delta_pct": waste_delta},
            "vision2030": {"score": v2030, "delta": v2030_delta},
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Demand → Recipe KPIs ------------------------------------------

@app.get("/api/kpis/demand-recipe")
async def get_demand_recipe_kpis(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    scenario: Optional[str] = Query("base"),
):
    """5 KPIs for the Demand → Recipe Translation page."""
    try:
        m = _mult(scenario or "base")
        from_dt, to_dt = _date_range(from_date, to_date)

        fcast = _filter_dates(data_cache.get("fact_sku_forecast", pd.DataFrame()), from_dt, to_dt)
        bflour = _filter_dates(data_cache.get("fact_bulk_flour", pd.DataFrame()), from_dt, to_dt)
        rdem = _filter_dates(data_cache.get("fact_recipe_demand", pd.DataFrame()), from_dt, to_dt)

        total_units = int(fcast["forecast_units"].sum() * m) if not fcast.empty else 0
        bulk_tons = float(bflour["required_tons"].sum() * m) if not bflour.empty else 0
        recipe_hours = float(rdem["required_hours"].sum() * m) if not rdem.empty else 0
        conf = float(fcast["confidence_pct"].mean() * 100) if not fcast.empty else 0
        season = float(fcast["seasonality_index"].mean()) if not fcast.empty else 1.0

        return {
            "total_sku_forecast_units": total_units,
            "bulk_flour_required_tons": round(bulk_tons, 2),
            "total_recipe_hours": round(recipe_hours, 2),
            "forecast_confidence_pct": round(conf, 1),
            "seasonality_index": round(season, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Recipe Planning KPIs ------------------------------------------

@app.get("/api/kpis/recipe-planning")
async def get_recipe_planning_kpis(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    scenario: Optional[str] = Query("base"),
):
    """6 KPIs for the Recipe & Mill Planning page."""
    try:
        m = _mult(scenario or "base")
        from_dt, to_dt = _date_range(from_date, to_date)

        sched = _filter_dates(data_cache.get("fact_schedule", pd.DataFrame()), from_dt, to_dt)
        cap = _filter_dates(data_cache.get("fact_mill_capacity", pd.DataFrame()), from_dt, to_dt)

        planned = float(sched["planned_hours"].sum()) * m if not sched.empty else 0
        available = float(cap["available_hours"].sum()) if not cap.empty else 0
        slack = available - planned

        changeovers = int(sched["changeover_hours"].gt(0).sum()) if not sched.empty else 0

        # Wheat cost index
        wheat_req = data_cache.get("fact_wheat_req", pd.DataFrame())
        if not wheat_req.empty:
            start_p = from_dt.to_period("M").strftime("%Y-%m")
            end_p = to_dt.to_period("M").strftime("%Y-%m")
            wr = wheat_req[(wheat_req["period"] >= start_p) & (wheat_req["period"] <= end_p)]
            if not wr.empty and wr["required_tons"].sum() > 0:
                wheat_ci = float((wr["avg_cost"] * wr["required_tons"]).sum() / wr["required_tons"].sum())
            else:
                wheat_ci = 0
        else:
            wheat_ci = 0

        # Waste impact
        waste_df = data_cache.get("fact_waste", pd.DataFrame())
        if not waste_df.empty:
            start_p = from_dt.to_period("M").strftime("%Y-%m")
            end_p = to_dt.to_period("M").strftime("%Y-%m")
            wm = waste_df[(waste_df["period"] >= start_p) & (waste_df["period"] <= end_p)]
            waste_pct = float(wm["waste_pct"].mean()) if not wm.empty else 0
        else:
            waste_pct = 0

        return {
            "planned_recipe_hours": round(planned, 2),
            "available_mill_hours": round(available, 2),
            "slack_shortfall_hours": round(slack, 2),
            "avg_changeovers": changeovers,
            "wheat_cost_index": round(wheat_ci, 2),
            "waste_impact_pct": round(waste_pct, 2),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Mill Operations KPIs -----------------------------------------

@app.get("/api/kpis/mill-operations")
async def get_mill_operations_kpis(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    scenario: Optional[str] = Query("base"),
):
    """5 KPIs for the Mill Runtime & Sequencing page."""
    try:
        m = _mult(scenario or "base")
        from_dt, to_dt = _date_range(from_date, to_date)

        sched = _filter_dates(data_cache.get("fact_schedule", pd.DataFrame()), from_dt, to_dt)
        cap = _filter_dates(data_cache.get("fact_mill_capacity", pd.DataFrame()), from_dt, to_dt)

        sched_hours = float(sched["planned_hours"].sum()) * m if not sched.empty else 0
        avail_hours = float(cap["available_hours"].sum()) if not cap.empty else 0
        util = (sched_hours / avail_hours * 100) if avail_hours > 0 else 0

        # Overload
        overload = 0.0
        if not sched.empty and not cap.empty:
            dl = sched.groupby(["date", "mill_id"])["planned_hours"].sum().reset_index()
            dl["planned_hours"] *= m
            dl = dl.merge(cap, on=["date", "mill_id"], how="left")
            overload = float(np.maximum(0, dl["planned_hours"] - dl["available_hours"]).sum())

        # Recipe switches
        switches = 0
        if not sched.empty:
            switches = int(
                sched.sort_values(["mill_id", "date"])
                .groupby("mill_id")
                .apply(lambda g: (g["recipe_id"] != g["recipe_id"].shift()).sum())
                .sum()
            )

        # Avg run length
        avg_run = 0
        if switches > 0 and not sched.empty:
            total_days = sched["date"].nunique()
            avg_run = round(total_days / max(1, switches), 1)

        # Downtime risk
        risk = min(100, max(0, util * 0.5 + (overload / max(1, avail_hours)) * 5000))

        return {
            "mill_utilization_pct": round(util, 2),
            "overload_hours": round(overload, 2),
            "recipe_switch_count": switches,
            "avg_run_length_days": avg_run,
            "downtime_risk_score": round(risk, 1),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Raw Materials KPIs --------------------------------------------

@app.get("/api/kpis/raw-materials")
async def get_raw_materials_kpis(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    scenario: Optional[str] = Query("base"),
):
    """5 KPIs for the Raw Materials & Wheat Origins page."""
    try:
        m = _mult(scenario or "base")
        from_dt, to_dt = _date_range(from_date, to_date)

        start_p = from_dt.to_period("M").strftime("%Y-%m")
        end_p = to_dt.to_period("M").strftime("%Y-%m")

        wheat_req = data_cache.get("fact_wheat_req", pd.DataFrame())
        if not wheat_req.empty:
            wr = wheat_req[(wheat_req["period"] >= start_p) & (wheat_req["period"] <= end_p)]
        else:
            wr = pd.DataFrame()

        total_wheat = float(wr["required_tons"].sum()) * m if not wr.empty else 0

        rp = _filter_dates(data_cache.get("raw_material_prices", pd.DataFrame()), from_dt, to_dt)
        avg_cost = float(rp["wheat_price_sar_per_ton"].mean()) if not rp.empty else 0

        # Import dependency
        map_wc = data_cache.get("map_wheat_country", pd.DataFrame())
        dim_country = data_cache.get("dim_country", pd.DataFrame())
        sa_pct = 0
        if not wr.empty and not map_wc.empty:
            sa_wt = map_wc[map_wc["country_id"] == "SA"]["wheat_type_id"].unique()
            sa_tons = float(wr[wr["wheat_type_id"].isin(sa_wt)]["required_tons"].sum())
            sa_pct = (sa_tons / total_wheat * 100) if total_wheat > 0 else 0
        import_dep = round(100 - sa_pct, 1)

        # High-risk share
        hr_pct = 0
        if not wr.empty and not map_wc.empty and not dim_country.empty:
            hr_ids = dim_country[dim_country["geopolitical_risk"] > 0.3]["country_id"].values
            hr_wt = map_wc[map_wc["country_id"].isin(hr_ids)]["wheat_type_id"].unique()
            hr_tons = float(wr[wr["wheat_type_id"].isin(hr_wt)]["required_tons"].sum())
            hr_pct = round((hr_tons / total_wheat * 100) if total_wheat > 0 else 0, 1)

        yield_var = round(np.random.uniform(0.05, 0.15), 3)

        return {
            "total_wheat_requirement_tons": round(total_wheat, 2),
            "avg_wheat_cost_sar": round(avg_cost, 2),
            "import_dependency_pct": import_dep,
            "high_risk_supply_pct": hr_pct,
            "yield_variability_index": yield_var,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Sustainability KPIs -------------------------------------------

@app.get("/api/kpis/sustainability")
async def get_sustainability_kpis(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    scenario: Optional[str] = Query("base"),
):
    """5 KPIs for the Waste & Vision 2030 page."""
    try:
        from_dt, to_dt = _date_range(from_date, to_date)
        start_p = from_dt.to_period("M").strftime("%Y-%m")
        end_p = to_dt.to_period("M").strftime("%Y-%m")

        waste_df = data_cache.get("fact_waste", pd.DataFrame())
        if not waste_df.empty:
            wm = waste_df[(waste_df["period"] >= start_p) & (waste_df["period"] <= end_p)]
        else:
            wm = pd.DataFrame()

        waste_rate = float(wm["waste_pct"].mean()) if not wm.empty else 4.2
        energy = float(wm["energy_per_ton"].mean()) if not wm.empty else 55
        water = float(wm["water_per_ton"].mean()) if not wm.empty else 0.7

        target = 3.9
        waste_gap = round(waste_rate - target, 2)

        dim_mill = data_cache.get("dim_mill", pd.DataFrame())
        e_eff = float(dim_mill["energy_efficiency_index"].mean()) if not dim_mill.empty else 75
        w_eff = float(dim_mill["water_efficiency_index"].mean()) if not dim_mill.empty else 72
        waste_score = max(0, min(100, (1 - (waste_rate / target - 1)) * 100))
        v2030 = round(waste_score * 0.30 + e_eff * 0.30 + w_eff * 0.25 + 85 * 0.15, 1)

        return {
            "waste_rate_pct": round(waste_rate, 2),
            "waste_target_pct": target,
            "waste_gap": waste_gap,
            "energy_per_ton": round(energy, 2),
            "water_per_ton": round(water, 3),
            "vision_2030_score": v2030,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# WASTE / SUSTAINABILITY DATA ENDPOINT
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/waste-metrics")
async def get_waste_metrics(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
):
    """Return detailed waste metrics for heatmap / trend / bar charts."""
    try:
        from_dt, to_dt = _date_range(from_date, to_date)
        start_p = from_dt.to_period("M").strftime("%Y-%m")
        end_p = to_dt.to_period("M").strftime("%Y-%m")

        waste_df = data_cache.get("fact_waste", pd.DataFrame())
        if waste_df.empty:
            return {"data": []}

        wm = waste_df[(waste_df["period"] >= start_p) & (waste_df["period"] <= end_p)]

        # Enrich with names
        dim_recipe = data_cache.get("dim_recipe", pd.DataFrame())
        dim_mill = data_cache.get("dim_mill", pd.DataFrame())

        if not dim_recipe.empty:
            wm = wm.merge(dim_recipe[["recipe_id", "recipe_name"]], on="recipe_id", how="left")
        if not dim_mill.empty:
            wm = wm.merge(dim_mill[["mill_id", "mill_name"]], on="mill_id", how="left")

        return {"data": wm.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# DATA ENDPOINTS (backward-compatible with existing frontend)
# ═══════════════════════════════════════════════════════════════════════════════

# ---------- SKU Forecast --------------------------------------------------

@app.get("/api/forecast/sku")
async def get_sku_forecast(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    horizon: Optional[str] = Query(None, pattern="^(week|month|year)$"),
    period: Optional[str] = None,
    sku_id: Optional[str] = None,
    scenario: Optional[str] = Query("base"),
):
    try:
        scenario = scenario or "base"
        m = _mult(scenario)
        fcast = data_cache.get("fact_sku_forecast", pd.DataFrame()).copy()
        if fcast.empty:
            return {"data": []}

        if sku_id:
            fcast = fcast[fcast["sku_id"] == sku_id]

        from_dt, to_dt = _date_range(from_date, to_date)
        fcast = _filter_dates(fcast, from_dt, to_dt)
        fcast = _add_period(fcast, horizon or "month")

        # Enrich with dim_sku + dim_flour_type
        dim_sku = data_cache.get("dim_sku", pd.DataFrame())
        dim_flour = data_cache.get("dim_flour_type", pd.DataFrame())

        if not dim_sku.empty:
            fcast = fcast.merge(dim_sku[["sku_id", "sku_name", "flour_type_id"]], on="sku_id", how="left")
        if not dim_flour.empty and "flour_type_id" in fcast.columns:
            fcast = fcast.merge(dim_flour[["flour_type_id", "flour_name"]], on="flour_type_id", how="left")
            fcast.rename(columns={"flour_name": "flour_type"}, inplace=True)

        agg = fcast.groupby(["period", "sku_id", "sku_name", "flour_type"]).agg(
            forecast_tons=("forecast_tons", "sum"),
        ).reset_index()

        if m != 1.0:
            agg["forecast_tons"] = (agg["forecast_tons"] * m).round(2)

        if period:
            agg = agg[agg["period"] == period]

        return {"data": agg.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Bulk Flour Demand ---------------------------------------------

@app.get("/api/demand/bulk-flour")
async def get_bulk_flour_demand(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    horizon: str = Query("month", pattern="^(week|month|year)$"),
    period: Optional[str] = None,
    flour_type: Optional[str] = None,
):
    try:
        bf = data_cache.get("fact_bulk_flour", pd.DataFrame()).copy()
        if bf.empty:
            return {"data": []}

        from_dt, to_dt = _date_range(from_date, to_date)
        bf = _filter_dates(bf, from_dt, to_dt)

        # Enrich with flour name
        dim_flour = data_cache.get("dim_flour_type", pd.DataFrame())
        if not dim_flour.empty:
            bf = bf.merge(dim_flour[["flour_type_id", "flour_name"]], on="flour_type_id", how="left")
            bf.rename(columns={"flour_name": "flour_type"}, inplace=True)

        if flour_type:
            bf = bf[bf["flour_type"].str.contains(flour_type, case=False, na=False)]

        bf = _add_period(bf, horizon)
        agg = bf.groupby(["period", "flour_type"]).agg(
            required_bulk_tons=("required_tons", "sum"),
        ).reset_index()

        if period:
            agg = agg[agg["period"] == period]

        return {"data": agg.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Recipe Planning -----------------------------------------------

@app.get("/api/planning/recipe")
async def get_recipe_planning(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    horizon: Optional[str] = Query(None, pattern="^(week|month|year)$"),
    period: Optional[str] = None,
    scenario: Optional[str] = Query("base"),
):
    try:
        scenario = scenario or "base"
        m = _mult(scenario)

        sched = data_cache.get("fact_schedule", pd.DataFrame()).copy()
        if sched.empty:
            return {"data": []}

        from_dt, to_dt = _date_range(from_date, to_date)
        sched = _filter_dates(sched, from_dt, to_dt)
        sched = _add_period(sched, horizon or "month")

        dim_recipe = data_cache.get("dim_recipe", pd.DataFrame())

        agg = sched.groupby(["period", "recipe_id"]).agg(
            scheduled_hours=("planned_hours", "sum"),
            tons_produced=("tons_produced", "sum"),
            changeover_hours=("changeover_hours", "sum"),
        ).reset_index()

        if not dim_recipe.empty:
            agg = agg.merge(
                dim_recipe[["recipe_id", "recipe_name", "milling_rate_tph", "avg_waste_pct"]],
                on="recipe_id", how="left",
            )
            agg.rename(columns={"milling_rate_tph": "cost_index"}, inplace=True)

        if m != 1.0:
            agg["scheduled_hours"] = (agg["scheduled_hours"] * m).round(2)
            agg["tons_produced"] = (agg["tons_produced"] * m).round(2)

        if period:
            agg = agg[agg["period"] == period]

        return {"data": agg.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Recipe Eligibility Matrix -------------------------------------

@app.get("/api/planning/recipe-eligibility")
async def get_recipe_eligibility():
    try:
        mfr = data_cache.get("map_flour_recipe", pd.DataFrame()).copy()
        dim_recipe = data_cache.get("dim_recipe", pd.DataFrame())
        dim_flour = data_cache.get("dim_flour_type", pd.DataFrame())

        if mfr.empty:
            return {"data": []}

        if not dim_recipe.empty:
            mfr = mfr.merge(
                dim_recipe[["recipe_id", "recipe_name", "milling_rate_tph"]],
                on="recipe_id", how="left",
            )
            mfr.rename(columns={"milling_rate_tph": "base_tons_per_hour"}, inplace=True)

        if not dim_flour.empty:
            mfr = mfr.merge(dim_flour[["flour_type_id", "flour_name"]], on="flour_type_id", how="left")
            mfr.rename(columns={"flour_name": "flour_type"}, inplace=True)

        return {"data": mfr.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Mill Capacity -------------------------------------------------

@app.get("/api/capacity/mill")
async def get_mill_capacity(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    horizon: Optional[str] = Query(None, pattern="^(week|month|year)$"),
    period: Optional[str] = None,
    mill_id: Optional[str] = None,
    scenario: Optional[str] = Query("base"),
):
    try:
        scenario = scenario or "base"
        m = _mult(scenario)

        sched = data_cache.get("fact_schedule", pd.DataFrame()).copy()
        cap = data_cache.get("fact_mill_capacity", pd.DataFrame()).copy()

        if cap.empty:
            return {"data": []}

        from_dt, to_dt = _date_range(from_date, to_date)
        sched = _filter_dates(sched, from_dt, to_dt)
        cap = _filter_dates(cap, from_dt, to_dt)

        if mill_id:
            sched = sched[sched["mill_id"] == mill_id]
            cap = cap[cap["mill_id"] == mill_id]

        # Aggregate schedule to daily mill level
        if not sched.empty:
            daily_sched = sched.groupby(["date", "mill_id"]).agg(
                scheduled_hours=("planned_hours", "sum"),
            ).reset_index()
        else:
            daily_sched = pd.DataFrame(columns=["date", "mill_id", "scheduled_hours"])

        merged = cap.merge(daily_sched, on=["date", "mill_id"], how="left")
        merged["scheduled_hours"] = merged["scheduled_hours"].fillna(0) * m

        merged = _add_period(merged, horizon or "month")

        agg = merged.groupby(["period", "mill_id"]).agg(
            scheduled_hours=("scheduled_hours", "sum"),
            available_hours=("available_hours", "sum"),
        ).reset_index()

        agg["overload_hours"] = np.maximum(0, agg["scheduled_hours"] - agg["available_hours"]).round(2)
        agg["utilization_pct"] = np.where(
            agg["available_hours"] > 0,
            (agg["scheduled_hours"] / agg["available_hours"] * 100).round(2),
            0,
        )

        dim_mill = data_cache.get("dim_mill", pd.DataFrame())
        if not dim_mill.empty:
            agg = agg.merge(dim_mill, on="mill_id", how="left")

        if period:
            agg = agg[agg["period"] == period]

        return {"data": agg.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Mill Schedule (Gantt) -----------------------------------------

@app.get("/api/capacity/mill-schedule")
async def get_mill_schedule(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    horizon: str = Query("week", pattern="^(week|month|year)$"),
    period: Optional[str] = None,
    mill_id: Optional[str] = None,
):
    try:
        sched = data_cache.get("fact_schedule", pd.DataFrame()).copy()
        if sched.empty:
            return {"data": []}

        from_dt, to_dt = _date_range(from_date, to_date)
        sched = _filter_dates(sched, from_dt, to_dt)
        sched = _add_period(sched, horizon)

        if mill_id:
            sched = sched[sched["mill_id"] == mill_id]
        if period:
            sched = sched[sched["period"] == period]

        dim_recipe = data_cache.get("dim_recipe", pd.DataFrame())
        if not dim_recipe.empty:
            sched = sched.merge(dim_recipe[["recipe_id", "recipe_name"]], on="recipe_id", how="left")

        # Rename for backward compatibility
        sched.rename(columns={"planned_hours": "duration_hours"}, inplace=True)

        return {"data": sched.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Raw Material --------------------------------------------------

@app.get("/api/raw-material")
async def get_raw_material(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    horizon: Optional[str] = Query(None, pattern="^(week|month|year)$"),
    period: Optional[str] = None,
    country: Optional[str] = None,
    scenario: Optional[str] = Query("base"),
):
    try:
        rp = data_cache.get("raw_material_prices", pd.DataFrame()).copy()
        if rp.empty:
            return {"data": []}

        from_dt, to_dt = _date_range(from_date, to_date)
        rp = _filter_dates(rp, from_dt, to_dt)

        if country:
            rp = rp[rp["country"].str.contains(country, case=False, na=False)]

        rp = _add_period(rp, horizon or "month")

        agg = rp.groupby(["period", "country"]).agg(
            wheat_price_sar_per_ton=("wheat_price_sar_per_ton", "mean"),
            availability_tons=("availability_tons", "mean"),
        ).reset_index()

        if period:
            agg = agg[agg["period"] == period]

        return {"data": agg.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# ALERTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/alerts")
async def get_alerts(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    horizon: Optional[str] = Query(None, pattern="^(week|month|year)$"),
    period: Optional[str] = None,
):
    try:
        from_dt, to_dt = _date_range(from_date, to_date)

        sched = _filter_dates(data_cache.get("fact_schedule", pd.DataFrame()).copy(), from_dt, to_dt)
        cap = _filter_dates(data_cache.get("fact_mill_capacity", pd.DataFrame()).copy(), from_dt, to_dt)
        dim_mill = data_cache.get("dim_mill", pd.DataFrame())

        alerts = []
        if sched.empty or cap.empty:
            return {"alerts": alerts}

        daily_load = sched.groupby(["date", "mill_id"])["planned_hours"].sum().reset_index()
        daily_load = daily_load.merge(cap, on=["date", "mill_id"], how="left")
        daily_load["overload"] = np.maximum(0, daily_load["planned_hours"] - daily_load["available_hours"])

        overloads = daily_load[daily_load["overload"] > 0]
        if overloads.empty:
            return {"alerts": alerts}

        overloads = _add_period(overloads, horizon or "month")
        agg = overloads.groupby(["period", "mill_id"]).agg(overload_hours=("overload", "sum")).reset_index()

        if not dim_mill.empty:
            agg = agg.merge(dim_mill[["mill_id", "mill_name"]], on="mill_id", how="left")

        for _, row in agg.iterrows():
            mill_name = row.get("mill_name", row["mill_id"])
            alerts.append({
                "type": "capacity_overload",
                "severity": "high",
                "title": f"Mill {mill_name} Overloaded",
                "message": f"Mill {mill_name} is overloaded by {row['overload_hours']:.1f} hours in {row['period']}",
                "mill_id": row["mill_id"],
                "period": row["period"],
            })

        return {"alerts": alerts}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# REPORTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.get("/api/reports/{report_id}")
async def get_report_data(
    report_id: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    horizon: Optional[str] = Query(None, pattern="^(week|month|year)$"),
    period: Optional[str] = None,
):
    try:
        from_dt, to_dt = _date_range(from_date, to_date)

        if report_id == "monthly-plan":
            sched = _filter_dates(data_cache.get("fact_schedule", pd.DataFrame()).copy(), from_dt, to_dt)
            if sched.empty:
                return {"data": []}
            sched = _add_period(sched, horizon or "month")

            dim_recipe = data_cache.get("dim_recipe", pd.DataFrame())
            dim_mill = data_cache.get("dim_mill", pd.DataFrame())

            if not dim_recipe.empty:
                sched = sched.merge(dim_recipe[["recipe_id", "recipe_name"]], on="recipe_id", how="left")
            if not dim_mill.empty:
                sched = sched.merge(dim_mill[["mill_id", "mill_name"]], on="mill_id", how="left")

            agg = sched.groupby(["period", "mill_id", "mill_name", "recipe_id", "recipe_name"]).agg(
                duration_hours=("planned_hours", "sum"),
                tons_produced=("tons_produced", "sum"),
            ).reset_index()

            if period:
                agg = agg[agg["period"] == period]
            return {"data": agg.to_dict(orient="records")}

        elif report_id == "capacity-outlook":
            cap_data = await get_mill_capacity(from_date=from_date, to_date=to_date, horizon=horizon, period=period)
            return cap_data

        elif report_id == "demand-forecast":
            fc_data = await get_sku_forecast(from_date=from_date, to_date=to_date, horizon=horizon, period=period)
            return fc_data

        elif report_id == "raw-material":
            rm_data = await get_raw_material(from_date=from_date, to_date=to_date, horizon=horizon, period=period)
            return rm_data

        else:
            raise HTTPException(status_code=404, detail=f"Report {report_id} not found")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/reports/{report_id}/download")
async def download_report_csv(
    report_id: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    horizon: str = Query("month", pattern="^(week|month|year)$"),
    period: Optional[str] = None,
):
    try:
        report_data = await get_report_data(report_id, from_date, to_date, horizon, period)
        data = report_data.get("data", [])
        if not data:
            raise HTTPException(status_code=404, detail="No data available for this report")

        df = pd.DataFrame(data)
        stream = io.StringIO()
        df.to_csv(stream, index=False)
        csv_content = stream.getvalue()
        stream.close()

        filename = f"{report_id}_{horizon}_{period or 'all'}.csv"
        return StreamingResponse(
            io.BytesIO(csv_content.encode("utf-8")),
            media_type="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/reports/{report_id}/email")
async def send_report_email(
    report_id: str,
    horizon: str = Query("month", pattern="^(week|month|year)$"),
    period: Optional[str] = None,
):
    if not SENDGRID_AVAILABLE:
        raise HTTPException(status_code=503, detail="SendGrid not installed")

    try:
        sendgrid_api_key = os.getenv("SENDGRID_API_KEY")
        from_email = os.getenv("MAIL_USERNAME")
        alert_emails_str = os.getenv("ALERT_EMAIL") or os.getenv("ALERT_EMAILS", "")
        alert_emails = [e.strip() for e in alert_emails_str.split(",") if e.strip()]

        if not sendgrid_api_key:
            raise HTTPException(status_code=500, detail="SENDGRID_API_KEY not configured")
        if not from_email:
            raise HTTPException(status_code=500, detail="MAIL_USERNAME not configured")
        if not alert_emails:
            raise HTTPException(status_code=500, detail="ALERT_EMAIL not configured")

        report_data = await get_report_data(report_id, horizon=horizon, period=period)
        data = report_data.get("data", [])
        if not data:
            raise HTTPException(status_code=404, detail="No data available")

        df = pd.DataFrame(data)
        csv_buf = io.StringIO()
        df.to_csv(csv_buf, index=False)
        csv_content = csv_buf.getvalue()
        csv_buf.close()

        report_names = {
            "monthly-plan": "Monthly Recipe Plan",
            "capacity-outlook": "Capacity Outlook",
            "demand-forecast": "Demand Forecast",
            "raw-material": "Raw Material Report",
        }
        report_name = report_names.get(report_id, report_id)

        subject = f"MC4 {report_name} Report — {horizon.capitalize()} {period or 'All'}"
        body_html = f"""<html><body>
        <h2>MC4 {report_name} Report</h2>
        <p>Attached: {report_name}</p>
        <ul><li>Horizon: {horizon}</li><li>Period: {period or 'All'}</li><li>Records: {len(data)}</li></ul>
        </body></html>"""

        filename = f"{report_id}_{horizon}_{period or 'all'}.csv"
        encoded_csv = base64.b64encode(csv_content.encode("utf-8")).decode()

        attachment = Attachment(
            FileContent(encoded_csv), FileName(filename),
            FileType("text/csv"), Disposition("attachment"),
        )

        message = Mail(from_email=from_email, to_emails=alert_emails,
                       subject=subject, html_content=body_html)
        message.add_attachment(attachment)

        sg = SendGridAPIClient(sendgrid_api_key)
        response = sg.send(message)

        if response.status_code in [200, 201, 202]:
            return {"success": True, "message": f"Email sent to {', '.join(alert_emails)}"}
        raise HTTPException(status_code=500, detail=f"SendGrid: {response.status_code}")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
# CHATBOT
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/chatbot/query")
async def chatbot_query(query: ChatbotQuery):
    if not CHATBOT_AVAILABLE:
        raise HTTPException(status_code=503, detail="Chatbot service not available")
    try:
        result = run_chatbot_query(query.question)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8008)

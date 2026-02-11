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

from fastapi import FastAPI, HTTPException, Query, Request
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

# Import forecast models
try:
    from forecast_models import MC4ForecastModel, train_and_save_models, SimpleModel, StatsModelWrapper
    # Import classes to ensure they're available for pickle loading
    import forecast_models
    FORECAST_MODELS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Forecast models not available: {e}")
    FORECAST_MODELS_AVAILABLE = False
    MC4ForecastModel = None
    train_and_save_models = None
    forecast_models = None

# Import forecast service
try:
    from forecast_service import (
        generate_future_forecast,
        generate_forecasts_and_propagate_datasets,
        generate_time_dimension_for_dates
    )
    FORECAST_SERVICE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Forecast service not available: {e}")
    FORECAST_SERVICE_AVAILABLE = False
    generate_future_forecast = None
    generate_forecasts_and_propagate_datasets = None
    generate_time_dimension_for_dates = None

# Import data generator module for deriving datasets
try:
    import data_generator as dg
    DATA_GENERATOR_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ Data generator module not available: {e}")
    DATA_GENERATOR_AVAILABLE = False
    dg = None

# Import Text2SQL chatbot (optional: if API key missing or import fails, server still starts; chatbot returns 503)
run_chatbot_query = None
CHATBOT_AVAILABLE = False
try:
    # Import from local backend modules (Text2SQL_V2 is now integrated into backend)
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
MODEL_DIR = "models"
data_cache = {}
forecaster = None  # Global forecaster instance

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
        
        # Deduplicate fact_sku_forecast to prevent inflated aggregation
        fcast = data_cache.get("fact_sku_forecast", pd.DataFrame())
        if not fcast.empty and "sku_id" in fcast.columns:
            before = len(fcast)
            data_cache["fact_sku_forecast"] = fcast.drop_duplicates(
                subset=["date", "sku_id"], keep="last"
            ).reset_index(drop=True)
            after = len(data_cache["fact_sku_forecast"])
            if before != after:
                print(f"   ⚠️ Removed {before - after} duplicate forecast rows")
        
        print("✅ Data loaded (MC4 v3 star-schema)")
    except Exception as e:
        print(f"⚠️ Error loading data: {e}")
        data_cache = {}


def initialize_forecaster():
    """Initialize and train forecast models if needed"""
    global forecaster
    
    if not FORECAST_MODELS_AVAILABLE:
        print("⚠️ Forecast models not available, skipping model initialization")
        return
    
    try:
        # Check if models directory exists and has models
        model_dir_path = os.path.join(_backend_dir, MODEL_DIR)
        os.makedirs(model_dir_path, exist_ok=True)
        
        # Check if we have forecast data to train on
        fcast_df = data_cache.get("fact_sku_forecast", pd.DataFrame())
        if fcast_df.empty:
            print("⚠️ No forecast data available for training")
            return
        
        # Check if models exist
        sku_list = fcast_df["sku_id"].unique()
        existing_models = [f for f in os.listdir(model_dir_path) if f.endswith(".pkl")]
        
        # Train models if they don't exist or if we need to retrain
        if len(existing_models) < len(sku_list):
            print(f"🔄 Training forecast models for {len(sku_list)} SKUs...")
            time_dim_path = os.path.join(_backend_dir, DATA_DIR, "time_dimension.csv")
            fcast_path = os.path.join(_backend_dir, DATA_DIR, "fact_sku_forecast.csv")
            
            # Save current forecast data to CSV if needed (for training)
            if not os.path.exists(fcast_path):
                fcast_df.to_csv(fcast_path, index=False)
            
            # Train models
            forecaster = train_and_save_models(
                data_path=fcast_path,
                time_dim_path=time_dim_path,
                model_dir=model_dir_path
            )
            print("✅ Forecast models trained successfully")
        else:
            # Load existing models
            print(f"📦 Loading existing forecast models from {model_dir_path}...")
            forecaster = MC4ForecastModel(model_dir=model_dir_path)
            
            # Try to load models, retrain if pickle fails due to module path issues
            loaded_count = 0
            failed_skus = []
            
            for sku_id in sku_list:
                try:
                    model_path = os.path.join(model_dir_path, f"model_{sku_id}.pkl")
                    if os.path.exists(model_path):
                        # Try loading with proper module context
                        import pickle
                        # Map __main__ classes to forecast_models classes for pickle compatibility
                        class CustomUnpickler(pickle.Unpickler):
                            def find_class(self, module, name):
                                if module == '__main__' and name in ['SimpleModel', 'StatsModelWrapper']:
                                    if forecast_models:
                                        return getattr(forecast_models, name)
                                return super().find_class(module, name)
                        
                        with open(model_path, 'rb') as f:
                            try:
                                # Try normal loading first
                                model = pickle.load(f)
                                forecaster.models[sku_id] = model
                                loaded_count += 1
                            except (AttributeError, ModuleNotFoundError, KeyError):
                                # If that fails, use custom unpickler
                                f.seek(0)
                                unpickler = CustomUnpickler(f)
                                model = unpickler.load()
                                forecaster.models[sku_id] = model
                                loaded_count += 1
                except Exception as e:
                    print(f"   ⚠️ Failed to load model for {sku_id}: {str(e)}")
                    failed_skus.append(sku_id)
            
            # If many models failed to load, retrain them
            if len(failed_skus) > len(sku_list) / 2:
                print(f"🔄 Retraining models due to loading issues...")
                fcast_path = os.path.join(_backend_dir, DATA_DIR, "fact_sku_forecast.csv")
                time_dim_path = os.path.join(_backend_dir, DATA_DIR, "time_dimension.csv")
                if os.path.exists(fcast_path):
                    forecaster = train_and_save_models(
                        data_path=fcast_path,
                        time_dim_path=time_dim_path,
                        model_dir=model_dir_path
                    )
                    loaded_count = len(forecaster.models)
                    print(f"✅ Retrained {loaded_count} models successfully")
            
            # Load time dimension for holidays
            time_dim_path = os.path.join(_backend_dir, DATA_DIR, "time_dimension.csv")
            if os.path.exists(time_dim_path):
                time_dim = pd.read_csv(time_dim_path)
                time_dim["date"] = pd.to_datetime(time_dim["date"])
                
                ramadan_days = time_dim[time_dim["is_ramadan"] == True][["date"]].copy()
                ramadan_days["holiday"] = "ramadan"
                hajj_days = time_dim[time_dim["is_hajj"] == True][["date"]].copy()
                hajj_days["holiday"] = "hajj"
                
                holidays_df = pd.concat([ramadan_days, hajj_days], ignore_index=True)
                holidays_df.rename(columns={"date": "ds"}, inplace=True)
                forecaster.holidays_df = holidays_df
            
            print(f"✅ Loaded {loaded_count}/{len(sku_list)} forecast models")
            
    except Exception as e:
        print(f"⚠️ Error initializing forecaster: {e}")
        import traceback
        traceback.print_exc()
        forecaster = None


@app.on_event("startup")
async def startup_event():
    load_data()
    initialize_forecaster()
    
    # Automatically generate forecasts for next year if needed
    try:
        if FORECAST_SERVICE_AVAILABLE:
            fcast = data_cache.get("fact_sku_forecast", pd.DataFrame())
            if not fcast.empty and "date" in fcast.columns:
                max_date = pd.to_datetime(fcast["date"]).max()
                target_end_date = pd.Timestamp("2027-02-10")  # 1 year from 2026-02-10
                
                if max_date < target_end_date:
                    print(f"🔄 Auto-generating forecasts from {max_date.date()} to {target_end_date.date()}")
                    result = generate_forecasts_and_propagate_datasets(
                        start_date=max_date + pd.Timedelta(days=1),
                        end_date=target_end_date,
                        forecaster=forecaster,
                        data_cache=data_cache,
                        backend_dir=_backend_dir,
                        data_dir=DATA_DIR,
                        update_derived_datasets_func=_update_derived_datasets
                    )
                    if result["success"]:
                        # Reload data to include new forecasts
                        load_data()
                        print(f"✅ Auto-generated {result['records']} forecast records")
    except Exception as e:
        print(f"⚠️ Error in auto-forecast generation: {e}")


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
# FORECAST GENERATION & DATASET UPDATE
# ═══════════════════════════════════════════════════════════════════════════════

@app.post("/api/forecast/generate")
async def generate_and_update_forecasts(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    days_ahead: Optional[int] = 365,
):
    """
    Generate forecasts for future dates and update the dataset.
    If start_date is not provided, starts from the day after the last available forecast.
    If end_date is not provided, generates forecasts for 'days_ahead' days.
    """
    try:
        fcast = data_cache.get("fact_sku_forecast", pd.DataFrame())
        dim_sku = data_cache.get("dim_sku", pd.DataFrame())
        
        if dim_sku.empty:
            raise HTTPException(status_code=400, detail="No SKU data available")
        
        # Determine date range
        if start_date:
            from_dt = pd.to_datetime(start_date)
        else:
            if fcast.empty or "date" not in fcast.columns:
                from_dt = pd.Timestamp("2026-02-11")  # Default start
            else:
                max_date = pd.to_datetime(fcast["date"]).max()
                from_dt = max_date + pd.Timedelta(days=1)
        
        if end_date:
            to_dt = pd.to_datetime(end_date)
        else:
            to_dt = from_dt + pd.Timedelta(days=days_ahead)
        
        print(f"🔄 Generating forecasts from {from_dt.date()} to {to_dt.date()}")
        
        # Generate forecasts and propagate all datasets
        if not FORECAST_SERVICE_AVAILABLE:
            raise HTTPException(status_code=500, detail="Forecast service not available")
        
        result = generate_forecasts_and_propagate_datasets(
            start_date=from_dt,
            end_date=to_dt,
            forecaster=forecaster,
            data_cache=data_cache,
            backend_dir=_backend_dir,
            data_dir=DATA_DIR,
            update_derived_datasets_func=_update_derived_datasets
        )
        
        if not result["success"]:
            return result
        
        # Reload data cache
        load_data()
        
        return {
            "success": True,
            "message": result["message"],
            "start_date": from_dt.isoformat(),
            "end_date": to_dt.isoformat(),
            "records": result["records"],
            "sku_count": result["sku_count"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


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
            return {"min_date": "2026-01-01", "max_date": None}
        actual_min = min(s for s, _ in date_ranges).date()
        default_min = pd.Timestamp("2026-01-01").date()
        # Use the later of actual_min or default_min to ensure default is 2026-01-01
        min_date = max(actual_min, default_min).isoformat()
        return {
            "min_date": min_date,
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
        # Use helper function to get forecast data (includes generated forecasts for future dates)
        fcast = _get_forecast_data(from_dt, to_dt)
        current_demand = float(fcast["forecast_tons"].sum()) * m if not fcast.empty else 0

        period_days = (to_dt - from_dt).days
        prev_from = from_dt - pd.Timedelta(days=period_days)
        prev_to = from_dt - pd.Timedelta(days=1)
        prev_fcast = _get_forecast_data(prev_from, prev_to)
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

        # Use helper function to get forecast data (includes generated forecasts for future dates)
        fcast = _get_forecast_data(from_dt, to_dt)
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
    """8 KPIs for the Recipe & Mill Planning page (incl. cost_impact & risk_score)."""
    try:
        m = _mult(scenario or "base")
        from_dt, to_dt = _date_range(from_date, to_date)

        sched = _filter_dates(data_cache.get("fact_schedule", pd.DataFrame()), from_dt, to_dt)
        cap = _filter_dates(data_cache.get("fact_mill_capacity", pd.DataFrame()), from_dt, to_dt)

        planned = float(sched["planned_hours"].sum()) * m if not sched.empty else 0
        available = float(cap["available_hours"].sum()) if not cap.empty else 0
        slack = available - planned
        overload = max(0.0, planned - available)

        changeovers = int(sched["changeover_hours"].gt(0).sum()) if not sched.empty else 0

        # Wheat cost index (SAR / ton weighted average)
        wheat_req = data_cache.get("fact_wheat_req", pd.DataFrame())
        start_p = from_dt.to_period("M").strftime("%Y-%m")
        end_p = to_dt.to_period("M").strftime("%Y-%m")
        if not wheat_req.empty:
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
            wm = waste_df[(waste_df["period"] >= start_p) & (waste_df["period"] <= end_p)]
            waste_pct = float(wm["waste_pct"].mean()) if not wm.empty else 0
        else:
            waste_pct = 0

        # ── Cost impact % ───────────────────────────────────────────────
        # cost_per_hour per recipe = (milling_rate_tph / (avg_yield/100)) × wheat_cost_per_ton
        # Cost impact = how much the ACTUAL recipe mix deviates from the
        # average-cost baseline (equal-cost assumption).
        dim_recipe = data_cache.get("dim_recipe", pd.DataFrame())
        cost_impact = 0.0
        if not sched.empty and not dim_recipe.empty and wheat_ci > 0:
            rh = sched.groupby("recipe_id")["planned_hours"].sum().reset_index()
            rh = rh.merge(
                dim_recipe[["recipe_id", "milling_rate_tph", "avg_yield"]],
                on="recipe_id", how="left",
            )
            rh["avg_yield"] = rh["avg_yield"].fillna(78)
            rh["milling_rate_tph"] = rh["milling_rate_tph"].fillna(38)
            # wheat consumed per hour of milling
            rh["wheat_per_hour"] = rh["milling_rate_tph"] / (rh["avg_yield"] / 100.0)
            rh["cost_per_hour"] = rh["wheat_per_hour"] * wheat_ci
            rh["planned_hours"] = rh["planned_hours"] * m

            actual_cost = float((rh["planned_hours"] * rh["cost_per_hour"]).sum())
            avg_cph = float(rh["cost_per_hour"].mean())
            baseline_cost = float(rh["planned_hours"].sum()) * avg_cph
            if baseline_cost > 0:
                cost_impact = round((actual_cost - baseline_cost) / baseline_cost * 100, 2)

        # ── Risk score (0-100) ──────────────────────────────────────────
        util_pct = (planned / available * 100) if available > 0 else 0
        risk_score = min(100.0, max(0.0,
            util_pct * 0.30                                     # capacity pressure
            + abs(cost_impact) * 2.0                            # cost deviation
            + waste_pct * 5.0                                   # waste contribution
            + (overload / max(1, available)) * 200              # overload penalty
        ))

        return {
            "planned_recipe_hours": round(planned, 2),
            "available_mill_hours": round(available, 2),
            "slack_shortfall_hours": round(slack, 2),
            "avg_changeovers": changeovers,
            "wheat_cost_index": round(wheat_ci, 2),
            "waste_impact_pct": round(waste_pct, 2),
            "cost_impact_pct": round(cost_impact, 2),
            "risk_score": round(risk_score, 1),
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
# FORECAST GENERATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════
# Note: Forecasting logic has been moved to forecast_service.py
# The functions below are kept for backward compatibility but delegate to the service

def _generate_future_forecast(start_date, end_date, dim_sku):
    """
    Generate forecast data for future dates using ML models trained on historical data.
    Falls back to simple trend-based forecast if models are not available.
    
    This function now delegates to forecast_service.generate_future_forecast()
    """
    if FORECAST_SERVICE_AVAILABLE:
        return generate_future_forecast(
            start_date=start_date,
            end_date=end_date,
            dim_sku=dim_sku,
            forecaster=forecaster,
            data_cache=data_cache
        )
    else:
        # Fallback if service not available
        print("⚠️ Forecast service not available, returning empty DataFrame")
        return pd.DataFrame()


def _generate_time_dimension_extended(start_date, end_date):
    """
    Generate time dimension for extended date range (same logic as data_generator.py).
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    df = pd.DataFrame({"date": dates})
    df["week"] = (
        df["date"].dt.isocalendar().year.astype(str)
        + "-W"
        + df["date"].dt.isocalendar().week.astype(str).str.zfill(2)
    )
    df["month"] = df["date"].dt.to_period("M").astype(str)
    df["year"] = df["date"].dt.year
    df["quarter"] = df["date"].dt.quarter
    df["day_of_week"] = df["date"].dt.dayofweek
    
    # Saudi Arabia weekend
    df["is_weekend"] = np.where(
        df["date"] < pd.Timestamp("2022-01-01"),
        df["day_of_week"].isin([4, 5]),   # Fri, Sat
        df["day_of_week"].isin([5, 6]),   # Sat, Sun
    )
    
    # Ramadan dates
    ramadan_starts = {
        2020: (4, 24), 2021: (4, 13), 2022: (4, 2), 2023: (3, 23),
        2024: (3, 11), 2025: (3, 1), 2026: (2, 18), 2027: (2, 7),
        2028: (1, 27), 2029: (1, 15),
    }
    
    def _ramadan(date):
        if date.year in ramadan_starts:
            m, d = ramadan_starts[date.year]
            s = pd.Timestamp(date.year, m, d)
            return s <= date <= s + pd.Timedelta(days=29)
        return False
    
    hajj_months = {
        2020: 7, 2021: 7, 2022: 7, 2023: 6,
        2024: 6, 2025: 6, 2026: 5, 2027: 5,
        2028: 5, 2029: 4,
    }
    
    def _hajj(date):
        return date.year in hajj_months and date.month == hajj_months[date.year]
    
    df["is_ramadan"] = df["date"].apply(_ramadan)
    df["is_hajj"] = df["date"].apply(_hajj)
    df["is_eid"] = df["is_ramadan"] | df["is_hajj"]
    return df


def _update_time_dimension(start_date, end_date):
    """
    Update time_dimension.csv with future dates.
    """
    try:
        time_dim_path = os.path.join(_backend_dir, DATA_DIR, "time_dimension.csv")
        
        # Load existing
        if os.path.exists(time_dim_path):
            existing_df = pd.read_csv(time_dim_path)
            existing_df["date"] = pd.to_datetime(existing_df["date"])
            max_existing_date = existing_df["date"].max()
        else:
            existing_df = pd.DataFrame()
            max_existing_date = None
        
        # Generate new time dimension if needed
        if max_existing_date is None or end_date > max_existing_date:
            gen_start = max_existing_date + pd.Timedelta(days=1) if max_existing_date else start_date
            new_time_dim = _generate_time_dimension_extended(gen_start, end_date)
            
            if not new_time_dim.empty:
                if not existing_df.empty:
                    combined_df = pd.concat([existing_df, new_time_dim], ignore_index=True)
                else:
                    combined_df = new_time_dim.copy()
                
                combined_df = combined_df.sort_values("date").reset_index(drop=True)
                combined_df.to_csv(time_dim_path, index=False)
                data_cache["time_dimension"] = combined_df.copy()
                print(f"✅ Updated time_dimension.csv with dates up to {end_date.date()}")
    except Exception as e:
        print(f"⚠️ Error updating time_dimension: {e}")


def _update_derived_datasets(start_date, end_date):
    """
    Generate and update all datasets derived from forecasts for the given date range.
    This includes:
    - fact_bulk_flour_requirement
    - fact_recipe_demand
    - fact_mill_schedule_daily
    - fact_mill_recipe_plan
    - fact_wheat_requirement
    - fact_waste_metrics
    - fact_kpi_snapshot
    - raw_material_prices
    - fact_mill_capacity
    - recipe_mix
    """
    if not DATA_GENERATOR_AVAILABLE:
        print("⚠️ Data generator functions not available, skipping derived dataset updates")
        return
    
    try:
        print(f"🔄 Generating derived datasets for {start_date.date()} to {end_date.date()}...")
        
        # Get required dimension tables
        dim_sku = data_cache.get("dim_sku", pd.DataFrame())
        dim_recipe = data_cache.get("dim_recipe", pd.DataFrame())
        dim_mill = data_cache.get("dim_mill", pd.DataFrame())
        dim_wheat_type = data_cache.get("dim_wheat_type", pd.DataFrame())
        dim_country = data_cache.get("dim_country", pd.DataFrame())
        map_flour_recipe = data_cache.get("map_flour_recipe", pd.DataFrame())
        map_recipe_mill = data_cache.get("map_recipe_mill", pd.DataFrame())
        map_recipe_wheat = data_cache.get("map_recipe_wheat", pd.DataFrame())
        map_wheat_country = data_cache.get("map_wheat_country", pd.DataFrame())
        
        # Get forecast data for the date range
        fcast = data_cache.get("fact_sku_forecast", pd.DataFrame())
        if fcast.empty:
            print("⚠️ No forecast data available")
            return
        
        fcast["date"] = pd.to_datetime(fcast["date"])
        fcast_range = fcast[(fcast["date"] >= start_date) & (fcast["date"] <= end_date)].copy()
        
        if fcast_range.empty:
            print("⚠️ No forecast data in specified range")
            return
        
        # Get time dimension for the range
        time_dim = data_cache.get("time_dimension", pd.DataFrame())
        if not time_dim.empty:
            time_dim["date"] = pd.to_datetime(time_dim["date"])
            time_dim_range = time_dim[(time_dim["date"] >= start_date) & (time_dim["date"] <= end_date)].copy()
        else:
            time_dim_range = _generate_time_dimension_extended(start_date, end_date)
        
        # Compute intermediate results once to avoid recomputation
        bulk_flour = None
        recipe_mix = None
        recipe_demand = None
        mill_capacity = None
        schedule = None
        
        # 1. Bulk flour requirement
        if not dim_sku.empty:
            bulk_flour = dg.derive_fact_bulk_flour_requirement(fcast_range, dim_sku)
            _append_to_dataset("fact_bulk_flour_requirement", bulk_flour, start_date)
        
        # 2. Recipe mix
        if not map_flour_recipe.empty:
            recipe_mix = dg.compute_dynamic_recipe_mix(time_dim_range, map_flour_recipe)
            _append_to_dataset("recipe_mix", recipe_mix, start_date)
        
        # 3. Recipe demand
        if bulk_flour is not None and recipe_mix is not None and not dim_recipe.empty:
            recipe_demand = dg.derive_fact_recipe_demand(bulk_flour, recipe_mix, dim_recipe)
            _append_to_dataset("fact_recipe_demand", recipe_demand, start_date)
        
        # 4. Mill capacity
        if not dim_mill.empty:
            mill_capacity = dg.generate_fact_mill_capacity(time_dim_range, dim_mill)
            _append_to_dataset("fact_mill_capacity", mill_capacity, start_date)
        
        # 5. Mill schedule (requires recipe_demand and mill_capacity)
        if recipe_demand is not None and mill_capacity is not None and not map_recipe_mill.empty and not dim_recipe.empty:
            schedule = dg.generate_fact_mill_schedule_daily(recipe_demand, mill_capacity, map_recipe_mill, dim_recipe)
            _append_to_dataset("fact_mill_schedule_daily", schedule, start_date)
        
        # 6. Mill recipe plan (requires schedule)
        if schedule is not None and not schedule.empty and mill_capacity is not None and not mill_capacity.empty:
            plan = dg.derive_fact_mill_recipe_plan(schedule, mill_capacity)
            if not plan.empty:
                _append_to_dataset("fact_mill_recipe_plan", plan, start_date)
        
        # 7. Wheat requirement
        if recipe_demand is not None and not recipe_demand.empty and len(map_recipe_wheat) > 0 and len(dim_wheat_type) > 0:
            wheat_req = dg.derive_fact_wheat_requirement(recipe_demand, map_recipe_wheat, dim_wheat_type)
            if not wheat_req.empty:
                _append_to_dataset("fact_wheat_requirement", wheat_req, start_date)
        
        # 8. Waste metrics (requires schedule)
        if schedule is not None and not schedule.empty and len(dim_recipe) > 0 and len(dim_mill) > 0:
            waste = dg.generate_fact_waste_metrics(schedule, dim_recipe, dim_mill)
            if not waste.empty:
                _append_to_dataset("fact_waste_metrics", waste, start_date)
        
        # 9. Raw material prices
        raw_prices = dg.generate_raw_material_prices(time_dim_range)
        _append_to_dataset("raw_material_prices", raw_prices, start_date)
        
        # 10. KPI snapshot (requires all above) - reload full datasets after updates
        # Reload all datasets to get the complete data including new additions
        load_data()
        
        fcast_full = data_cache.get("fact_sku_forecast", pd.DataFrame())
        bulk_flour_full = data_cache.get("fact_bulk_flour", pd.DataFrame())
        recipe_demand_full = data_cache.get("fact_recipe_demand", pd.DataFrame())
        schedule_full = data_cache.get("fact_schedule", pd.DataFrame())
        mill_capacity_full = data_cache.get("fact_mill_capacity", pd.DataFrame())
        wheat_req_full = data_cache.get("fact_wheat_req", pd.DataFrame())
        waste_full = data_cache.get("fact_waste", pd.DataFrame())
        
        if (len(fcast_full) > 0 and len(bulk_flour_full) > 0 and len(recipe_demand_full) > 0 and 
            len(schedule_full) > 0 and len(mill_capacity_full) > 0 and len(wheat_req_full) > 0 and 
            len(waste_full) > 0 and len(dim_mill) > 0 and len(dim_country) > 0 and 
            len(map_wheat_country) > 0):
            # Generate KPI snapshot for all periods (function handles period filtering internally)
            kpi = dg.generate_fact_kpi_snapshot(
                fcast_full, bulk_flour_full, recipe_demand_full,
                schedule_full, mill_capacity_full, wheat_req_full,
                waste_full, dim_mill, dim_country, map_wheat_country,
                raw_prices
            )
            # Filter to new periods only
            new_periods = pd.date_range(start_date, end_date, freq="M").to_period("M").astype(str)
            if not kpi.empty and "period" in kpi.columns:
                kpi_new = kpi[kpi["period"].isin(new_periods)]
                if not kpi_new.empty:
                    _append_to_dataset("fact_kpi_snapshot", kpi_new, start_date)
        
        print(f"✅ Updated all derived datasets")
        
    except Exception as e:
        print(f"⚠️ Error updating derived datasets: {e}")
        import traceback
        traceback.print_exc()


def _append_to_dataset(dataset_name, new_data, start_date):
    """
    Append new data to a dataset CSV file, removing any overlapping dates.
    """
    if new_data.empty:
        return
    
    try:
        dataset_path = os.path.join(_backend_dir, DATA_DIR, f"{dataset_name}.csv")
        
        # Load existing
        if os.path.exists(dataset_path):
            existing_df = pd.read_csv(dataset_path)
            if "date" in existing_df.columns:
                existing_df["date"] = pd.to_datetime(existing_df["date"])
                # Remove overlapping dates - use proper comparison
                start_date_dt = pd.to_datetime(start_date)
                mask = existing_df["date"] < start_date_dt
                existing_df = existing_df[mask].copy()
            elif "period" in existing_df.columns:
                # For period-based datasets, remove overlapping periods
                if "date" in new_data.columns:
                    new_data_dates = pd.to_datetime(new_data["date"])
                else:
                    new_data_dates = pd.to_datetime([start_date] * len(new_data))
                
                if "period" in new_data.columns:
                    new_periods = new_data["period"].unique().tolist()
                    if len(new_periods) > 0:
                        # Use proper boolean indexing
                        mask = ~existing_df["period"].isin(new_periods)
                        existing_df = existing_df[mask].copy()
        else:
            existing_df = pd.DataFrame()
        
        # Combine
        if len(existing_df) > 0:
            combined_df = pd.concat([existing_df, new_data], ignore_index=True)
        else:
            combined_df = new_data.copy()
        
        # Sort and save
        if "date" in combined_df.columns:
            combined_df = combined_df.sort_values("date").reset_index(drop=True)
        elif "period" in combined_df.columns:
            combined_df = combined_df.sort_values("period").reset_index(drop=True)
        
        combined_df.to_csv(dataset_path, index=False)
        
        # Update cache if this dataset is cached
        if dataset_name in data_cache:
            data_cache[dataset_name] = combined_df.copy()
            
    except Exception as e:
        print(f"⚠️ Error appending to {dataset_name}: {e}")
        import traceback
        traceback.print_exc()


def _update_forecast_dataset(new_forecast_df):
    """
    Update the fact_sku_forecast dataset with new forecast data.
    Also updates sku_forecast.csv (training dataset) to match.
    Appends new forecasts to the CSV files and updates the data cache.
    """
    if new_forecast_df.empty:
        return
    
    try:
        fcast_path = os.path.join(_backend_dir, DATA_DIR, "fact_sku_forecast.csv")
        sku_fcast_path = os.path.join(_backend_dir, DATA_DIR, "sku_forecast.csv")
        
        # Load existing data
        if os.path.exists(fcast_path):
            existing_df = pd.read_csv(fcast_path)
            existing_df["date"] = pd.to_datetime(existing_df["date"])
        else:
            existing_df = pd.DataFrame()
        
        # Get max date from existing data
        max_existing_date = existing_df["date"].max() if not existing_df.empty else None
        
        # Filter new forecasts to only include dates after max_existing_date
        new_forecast_df["date"] = pd.to_datetime(new_forecast_df["date"])
        if max_existing_date:
            new_forecast_df = new_forecast_df[new_forecast_df["date"] > max_existing_date]
        
        if new_forecast_df.empty:
            return
        
        # Combine and deduplicate
        if not existing_df.empty:
            # Remove any overlapping dates from existing data
            existing_df = existing_df[existing_df["date"] < new_forecast_df["date"].min()]
            combined_df = pd.concat([existing_df, new_forecast_df], ignore_index=True)
        else:
            combined_df = new_forecast_df.copy()
        
        # Sort by date and SKU
        combined_df = combined_df.sort_values(["date", "sku_id"]).reset_index(drop=True)
        
        # Deduplicate: keep last entry per (date, sku_id) to avoid inflated aggregation
        if not combined_df.empty and "date" in combined_df.columns and "sku_id" in combined_df.columns:
            before_dedup = len(combined_df)
            combined_df = combined_df.drop_duplicates(subset=["date", "sku_id"], keep="last").reset_index(drop=True)
            if len(combined_df) < before_dedup:
                print(f"   ⚠️ Removed {before_dedup - len(combined_df)} duplicate forecast rows")
        
        # Save to fact_sku_forecast.csv
        combined_df.to_csv(fcast_path, index=False)
        
        # Also save to sku_forecast.csv (training dataset) - same data
        combined_df.to_csv(sku_fcast_path, index=False)
        
        # Update cache
        data_cache["fact_sku_forecast"] = combined_df.copy()
        
        print(f"✅ Updated forecast datasets (fact_sku_forecast.csv & sku_forecast.csv) with {len(new_forecast_df)} new records")
        
        # Update time dimension for the new date range
        if not new_forecast_df.empty:
            min_date = new_forecast_df["date"].min()
            max_date = new_forecast_df["date"].max()
            _update_time_dimension(min_date, max_date)
            
            # Update all derived datasets
            _update_derived_datasets(min_date, max_date)
        
    except Exception as e:
        print(f"⚠️ Error updating forecast dataset: {e}")
        import traceback
        traceback.print_exc()


def _get_forecast_data(from_dt, to_dt, sku_id=None, update_dataset=True):
    """
    Get forecast data for the specified date range, generating forecasts for future dates if needed.
    Returns a DataFrame with forecast data covering the entire requested range.
    If update_dataset is True, new forecasts will be saved to the dataset.
    """
    fcast = data_cache.get("fact_sku_forecast", pd.DataFrame()).copy()
    dim_sku = data_cache.get("dim_sku", pd.DataFrame())
    
    if dim_sku.empty:
        return pd.DataFrame()
    
    # Check if we need to generate forecasts for future dates
    max_available_date = None
    if not fcast.empty and "date" in fcast.columns:
        max_available_date = pd.to_datetime(fcast["date"]).max()
    
    # Filter existing data
    if not fcast.empty:
        filter_end = min(to_dt, max_available_date) if max_available_date else to_dt
        fcast = _filter_dates(fcast, from_dt, filter_end)
        if sku_id:
            fcast = fcast[fcast["sku_id"] == sku_id]
    
    # Generate forecasts for future dates if needed
    future_forecast = pd.DataFrame()
    if max_available_date is None or to_dt > max_available_date:
        forecast_start = max_available_date + pd.Timedelta(days=1) if max_available_date else from_dt
        forecast_end = to_dt
        
        if forecast_start <= forecast_end:
            print(f"📈 Generating forecasts from {forecast_start.date()} to {forecast_end.date()}")
            if FORECAST_SERVICE_AVAILABLE:
                future_forecast = generate_future_forecast(
                    start_date=forecast_start,
                    end_date=forecast_end,
                    dim_sku=dim_sku,
                    forecaster=forecaster,
                    data_cache=data_cache
                )
            else:
                future_forecast = pd.DataFrame()
            
            if sku_id and not future_forecast.empty:
                future_forecast = future_forecast[future_forecast["sku_id"] == sku_id]
            
            # Update dataset with new forecasts
            if update_dataset and not future_forecast.empty:
                _update_forecast_dataset(future_forecast.copy())
    
    # Combine historical and future forecasts
    if not future_forecast.empty:
        if fcast.empty:
            fcast = future_forecast.copy()
        else:
            # Ensure both have the same columns
            common_cols = set(fcast.columns) & set(future_forecast.columns)
            fcast = pd.concat([
                fcast[list(common_cols)],
                future_forecast[list(common_cols)]
            ], ignore_index=True)
    
    # Deduplicate: keep last entry per (date, sku_id) to avoid inflated aggregation
    if not fcast.empty and "date" in fcast.columns and "sku_id" in fcast.columns:
        fcast = fcast.drop_duplicates(subset=["date", "sku_id"], keep="last").reset_index(drop=True)
    
    return fcast


# ═══════════════════════════════════════════════════════════════════════════════
# DATA ENDPOINTS (backward-compatible with existing frontend)
# ═══════════════════════════════════════════════════════════════════════════════

# ---------- SKU Forecast --------------------------------------------------

@app.get("/api/forecast/sku")
async def get_sku_forecast(
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    horizon: Optional[str] = Query(None, pattern="^(day|week|month|year)$"),
    period: Optional[str] = None,
    sku_id: Optional[str] = None,
    scenario: Optional[str] = Query("base"),
):
    try:
        scenario = scenario or "base"
        m = _mult(scenario)
        from_dt, to_dt = _date_range(from_date, to_date)
        
        # Get forecast data (includes generated forecasts for future dates)
        fcast = _get_forecast_data(from_dt, to_dt, sku_id)
        
        if fcast.empty:
            return {"data": []}

        # Enrich with dim_sku + dim_flour_type
        dim_sku = data_cache.get("dim_sku", pd.DataFrame())
        dim_flour = data_cache.get("dim_flour_type", pd.DataFrame())

        if not dim_sku.empty:
            fcast = fcast.merge(dim_sku[["sku_id", "sku_name", "flour_type_id"]], on="sku_id", how="left")
        if not dim_flour.empty and "flour_type_id" in fcast.columns:
            fcast = fcast.merge(dim_flour[["flour_type_id", "flour_name"]], on="flour_type_id", how="left")
            fcast.rename(columns={"flour_name": "flour_type"}, inplace=True)

        # If horizon is "day", return daily data without aggregation
        if horizon == "day":
            fcast["date"] = fcast["date"].dt.strftime("%Y-%m-%d")
            if m != 1.0:
                fcast["forecast_tons"] = (fcast["forecast_tons"] * m).round(2)
            return {"data": fcast[["date", "sku_id", "sku_name", "flour_type", "forecast_tons"]].to_dict(orient="records")}
        
        # Add period grouping for other horizons
        fcast = _add_period(fcast, horizon or "month")

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
        dim_mill = data_cache.get("dim_mill", pd.DataFrame())

        # Include mill_id in groupby so the Executive chart can stack bars by mill
        group_cols = ["period", "recipe_id"]
        if "mill_id" in sched.columns:
            group_cols = ["period", "recipe_id", "mill_id"]

        agg = sched.groupby(group_cols).agg(
            scheduled_hours=("planned_hours", "sum"),
            tons_produced=("tons_produced", "sum"),
            changeover_hours=("changeover_hours", "sum"),
        ).reset_index()

        if not dim_recipe.empty:
            agg = agg.merge(
                dim_recipe[["recipe_id", "recipe_name", "milling_rate_tph", "avg_yield", "avg_waste_pct"]],
                on="recipe_id", how="left",
            )
            agg.rename(columns={"milling_rate_tph": "cost_index"}, inplace=True)

        # Compute real cost_per_hour using wheat price data
        wheat_req = data_cache.get("fact_wheat_req", pd.DataFrame())
        wheat_ci = 0.0
        if not wheat_req.empty:
            from_dt2, to_dt2 = _date_range(from_date, to_date)
            sp = from_dt2.to_period("M").strftime("%Y-%m")
            ep = to_dt2.to_period("M").strftime("%Y-%m")
            wr = wheat_req[(wheat_req["period"] >= sp) & (wheat_req["period"] <= ep)]
            if not wr.empty and wr["required_tons"].sum() > 0:
                wheat_ci = float((wr["avg_cost"] * wr["required_tons"]).sum() / wr["required_tons"].sum())

        if "avg_yield" in agg.columns and "cost_index" in agg.columns and wheat_ci > 0:
            yield_frac = agg["avg_yield"].fillna(78) / 100.0
            wheat_per_hr = agg["cost_index"].fillna(38) / yield_frac
            agg["cost_per_hour"] = (wheat_per_hr * wheat_ci).round(2)
        else:
            agg["cost_per_hour"] = 0.0

        # Join mill_name for display
        if "mill_id" in agg.columns and not dim_mill.empty and "mill_name" in dim_mill.columns:
            agg = agg.merge(dim_mill[["mill_id", "mill_name"]], on="mill_id", how="left")

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

            # For reports, we aggregate to reduce rows, but ensure we get ALL aggregated groups
            agg = sched.groupby(["period", "mill_id", "mill_name", "recipe_id", "recipe_name"]).agg(
                duration_hours=("planned_hours", "sum"),
                tons_produced=("tons_produced", "sum"),
            ).reset_index()

            if period:
                agg = agg[agg["period"] == period]
            
            # Ensure all rows are included - no truncation
            # Convert to list explicitly to avoid any pandas display limits
            records = []
            for idx, row in agg.iterrows():
                records.append(row.to_dict())
            
            print(f"📊 Report {report_id}: Returning {len(records)} aggregated records from {len(sched)} raw schedule rows")
            return {"data": records}

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
        
        # Report metadata
        report_names = {
            "monthly-plan": "Monthly Recipe Plan",
            "capacity-outlook": "Capacity Outlook",
            "demand-forecast": "Demand Forecast",
            "raw-material": "Raw Material Report",
        }
        report_name = report_names.get(report_id, report_id.replace("-", " ").title())
        
        # Format the DataFrame for structured CSV output
        df_formatted = df.copy()
        
        # Format dates
        date_cols = [col for col in df_formatted.columns if "date" in col.lower() or col == "period"]
        for col in date_cols:
            if col in df_formatted.columns:
                try:
                    df_formatted[col] = pd.to_datetime(df_formatted[col], errors="coerce").dt.strftime("%Y-%m-%d")
                except:
                    pass  # Keep as-is if not a date
        
        # Format numeric columns with appropriate precision (keep as numeric for CSV, formatting happens on export)
        # We'll format during CSV writing, but for now ensure proper data types
        numeric_cols = df_formatted.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            # Round to appropriate precision but keep as numeric
            if "pct" in col.lower() or "rate" in col.lower() or "utilization" in col.lower():
                df_formatted[col] = df_formatted[col].round(2)
            elif "price" in col.lower() or "cost" in col.lower() or "sar" in col.lower():
                df_formatted[col] = df_formatted[col].round(2)
            elif "tons" in col.lower() or "tons_produced" in col.lower() or "required_tons" in col.lower():
                df_formatted[col] = df_formatted[col].round(2)
            elif "hours" in col.lower() or "duration" in col.lower():
                df_formatted[col] = df_formatted[col].round(2)
            else:
                df_formatted[col] = df_formatted[col].round(0)
        
        # Reorder columns for better readability (IDs first, then names, then metrics)
        id_cols = [col for col in df_formatted.columns if col.endswith("_id")]
        name_cols = [col for col in df_formatted.columns if col.endswith("_name") or col.endswith("_type")]
        period_cols = [col for col in df_formatted.columns if "period" in col.lower() or "date" in col.lower()]
        metric_cols = [col for col in df_formatted.columns if col not in id_cols + name_cols + period_cols]
        
        # Preferred column order
        preferred_order = period_cols + id_cols + name_cols + metric_cols
        existing_cols = [col for col in preferred_order if col in df_formatted.columns]
        remaining_cols = [col for col in df_formatted.columns if col not in existing_cols]
        df_formatted = df_formatted[existing_cols + remaining_cols]
        
        # Human-readable column names
        column_mapping = {
            "period": "Period",
            "date": "Date",
            "mill_id": "Mill ID",
            "mill_name": "Mill Name",
            "recipe_id": "Recipe ID",
            "recipe_name": "Recipe Name",
            "sku_id": "SKU ID",
            "sku_name": "SKU Name",
            "flour_type": "Flour Type",
            "flour_type_id": "Flour Type ID",
            "country": "Country",
            "wheat_type": "Wheat Type",
            "wheat_type_id": "Wheat Type ID",
            "duration_hours": "Duration (Hours)",
            "scheduled_hours": "Scheduled Hours",
            "planned_hours": "Planned Hours",
            "actual_hours": "Actual Hours",
            "tons_produced": "Tons Produced",
            "utilization_pct": "Utilization (%)",
            "capacity_hours": "Capacity (Hours)",
            "available_hours": "Available Hours",
            "overload_hours": "Overload Hours",
            "forecast_tons": "Forecast (Tons)",
            "forecast_units": "Forecast (Units)",
            "price_sar_per_ton": "Price (SAR/Ton)",
            "price": "Price",
            "required_tons": "Required Tons",
            "changeover_hours": "Changeover Hours",
            "milling_rate_tph": "Milling Rate (Tons/Hour)",
            "avg_waste_pct": "Avg Waste (%)",
            "cost_index": "Cost Index",
        }
        
        df_formatted.rename(columns=column_mapping, inplace=True)
        
        # Generate CSV with metadata header
        stream = io.StringIO()
        
        # Write metadata header
        stream.write(f"# {report_name}\n")
        stream.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if from_date:
            stream.write(f"# Date Range: {from_date} to {to_date or 'N/A'}\n")
        if period:
            stream.write(f"# Period: {period}\n")
        stream.write(f"# Horizon: {horizon}\n")
        stream.write(f"# Total Records: {len(df_formatted)}\n")
        stream.write("#\n")
        
        # Write CSV data with proper formatting
        # Use to_csv with formatting options for better structure
        df_formatted.to_csv(
            stream,
            index=False,
            float_format="%.2f",  # Format floats to 2 decimal places
            na_rep="",  # Empty string for NaN values
            encoding="utf-8"
        )
        csv_content = stream.getvalue()
        stream.close()

        filename = f"{report_id}_{horizon}_{period or 'all'}_{datetime.now().strftime('%Y%m%d')}.csv"
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
    request: Request,
):
    """
    Send report via email. Accepts parameters from request body:
    - from_date, to_date, horizon, period, scenario
    """
    if not SENDGRID_AVAILABLE:
        raise HTTPException(status_code=503, detail="SendGrid not installed")

    try:
        # Parse JSON body
        request_data = await request.json()
        
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

        # Extract parameters from request body
        from_date = request_data.get("from_date")
        to_date = request_data.get("to_date")
        horizon = request_data.get("horizon", "month")
        period = request_data.get("period")

        # Get full report data with all date filters - use same function as download
        # This ensures we get the exact same data that would be downloaded
        report_data = await get_report_data(
            report_id,
            from_date=from_date,
            to_date=to_date,
            horizon=horizon,
            period=period
        )
        data = report_data.get("data", [])
        if not data:
            raise HTTPException(status_code=404, detail="No data available")

        # Log the number of records retrieved for debugging
        print(f"📧 Email report {report_id}: Retrieved {len(data)} records from get_report_data")
        
        # Ensure pandas doesn't truncate - set display options (though this shouldn't affect CSV writing)
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        
        # Convert to DataFrame - ensure all rows are included
        df = pd.DataFrame(data)
        
        # Verify we have all rows - no truncation
        if len(df) != len(data):
            print(f"⚠️ Warning: DataFrame has {len(df)} rows but data list has {len(data)} rows")
            # Force include all rows by recreating DataFrame
            df = pd.DataFrame(data)
        else:
            print(f"✅ DataFrame created with {len(df)} rows - all data included")
        
        # Double-check: ensure DataFrame is not empty and has expected number of rows
        if len(df) == 0:
            raise HTTPException(status_code=404, detail="No data available after processing")
        
        print(f"📊 Processing DataFrame with {len(df)} rows for CSV generation")
        
        # Use the same structured formatting as download function
        report_names = {
            "monthly-plan": "Monthly Recipe Plan",
            "capacity-outlook": "Capacity Outlook",
            "demand-forecast": "Demand Forecast",
            "raw-material": "Raw Material Report",
        }
        report_name = report_names.get(report_id, report_id.replace("-", " ").title())
        
        # Format the DataFrame for structured CSV output (same as download)
        df_formatted = df.copy()
        
        # Format dates
        date_cols = [col for col in df_formatted.columns if "date" in col.lower() or col == "period"]
        for col in date_cols:
            if col in df_formatted.columns:
                try:
                    df_formatted[col] = pd.to_datetime(df_formatted[col], errors="coerce").dt.strftime("%Y-%m-%d")
                except:
                    pass
        
        # Format numeric columns
        numeric_cols = df_formatted.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if "pct" in col.lower() or "rate" in col.lower() or "utilization" in col.lower():
                df_formatted[col] = df_formatted[col].round(2)
            elif "price" in col.lower() or "cost" in col.lower() or "sar" in col.lower():
                df_formatted[col] = df_formatted[col].round(2)
            elif "tons" in col.lower() or "tons_produced" in col.lower() or "required_tons" in col.lower():
                df_formatted[col] = df_formatted[col].round(2)
            elif "hours" in col.lower() or "duration" in col.lower():
                df_formatted[col] = df_formatted[col].round(2)
            else:
                df_formatted[col] = df_formatted[col].round(0)
        
        # Reorder columns
        id_cols = [col for col in df_formatted.columns if col.endswith("_id")]
        name_cols = [col for col in df_formatted.columns if col.endswith("_name") or col.endswith("_type")]
        period_cols = [col for col in df_formatted.columns if "period" in col.lower() or "date" in col.lower()]
        metric_cols = [col for col in df_formatted.columns if col not in id_cols + name_cols + period_cols]
        preferred_order = period_cols + id_cols + name_cols + metric_cols
        existing_cols = [col for col in preferred_order if col in df_formatted.columns]
        remaining_cols = [col for col in df_formatted.columns if col not in existing_cols]
        df_formatted = df_formatted[existing_cols + remaining_cols]
        
        # Human-readable column names
        column_mapping = {
            "period": "Period",
            "date": "Date",
            "mill_id": "Mill ID",
            "mill_name": "Mill Name",
            "recipe_id": "Recipe ID",
            "recipe_name": "Recipe Name",
            "sku_id": "SKU ID",
            "sku_name": "SKU Name",
            "flour_type": "Flour Type",
            "flour_type_id": "Flour Type ID",
            "country": "Country",
            "wheat_type": "Wheat Type",
            "wheat_type_id": "Wheat Type ID",
            "duration_hours": "Duration (Hours)",
            "scheduled_hours": "Scheduled Hours",
            "planned_hours": "Planned Hours",
            "actual_hours": "Actual Hours",
            "tons_produced": "Tons Produced",
            "utilization_pct": "Utilization (%)",
            "capacity_hours": "Capacity (Hours)",
            "available_hours": "Available Hours",
            "overload_hours": "Overload Hours",
            "forecast_tons": "Forecast (Tons)",
            "forecast_units": "Forecast (Units)",
            "price_sar_per_ton": "Price (SAR/Ton)",
            "price": "Price",
            "required_tons": "Required Tons",
            "changeover_hours": "Changeover Hours",
            "milling_rate_tph": "Milling Rate (Tons/Hour)",
            "avg_waste_pct": "Avg Waste (%)",
            "cost_index": "Cost Index",
        }
        df_formatted.rename(columns=column_mapping, inplace=True)
        
        # Generate CSV with metadata header (same as download)
        csv_buf = io.StringIO()
        
        # Write metadata header
        csv_buf.write(f"# {report_name}\n")
        csv_buf.write(f"# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        if from_date:
            csv_buf.write(f"# Date Range: {from_date} to {to_date or 'N/A'}\n")
        if period:
            csv_buf.write(f"# Period: {period}\n")
        csv_buf.write(f"# Horizon: {horizon}\n")
        csv_buf.write(f"# Total Records: {len(df_formatted)}\n")
        csv_buf.write("#\n")
        
        # Write CSV data with proper formatting
        # Ensure no row limits - write ALL rows explicitly
        # Use to_csv with no truncation - pandas should write all rows by default
        df_formatted.to_csv(
            csv_buf,
            index=False,
            float_format="%.2f",
            na_rep="",
            encoding="utf-8"
        )
        csv_content = csv_buf.getvalue()
        csv_buf.close()
        
        # Verify CSV contains all rows
        csv_lines = csv_content.count('\n')
        # CSV should have: 1 header line + number of data rows + metadata lines
        expected_lines = 1 + len(df_formatted) + 7  # 7 metadata lines
        print(f"📧 Email CSV generated: {len(df_formatted)} DataFrame rows")
        print(f"📧 CSV file: {csv_lines} total lines (expected ~{expected_lines} with metadata)")
        
        # Double-check: count data rows in CSV (excluding header and metadata)
        csv_data_lines = csv_content.split('\n')
        data_row_count = len([line for line in csv_data_lines if line and not line.startswith('#')]) - 1  # -1 for header
        print(f"📧 CSV data rows (excluding header/metadata): {data_row_count}")
        
        if data_row_count != len(df_formatted):
            print(f"⚠️ WARNING: CSV has {data_row_count} data rows but DataFrame has {len(df_formatted)} rows!")

        subject = f"MC4 {report_name} Report — {horizon.capitalize()} {period or 'All'}"
        if from_date:
            subject += f" ({from_date} to {to_date or 'N/A'})"
        
        body_html = f"""<html><body>
        <h2>MC4 {report_name} Report</h2>
        <p>Attached: {report_name}</p>
        <ul>
        <li>Horizon: {horizon}</li>
        <li>Period: {period or 'All'}</li>
        {f'<li>Date Range: {from_date} to {to_date or "N/A"}</li>' if from_date else ''}
        <li>Total Records: {len(data)}</li>
        </ul>
        </body></html>"""

        filename = f"{report_id}_{horizon}_{period or 'all'}_{datetime.now().strftime('%Y%m%d')}.csv"
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
            return {"success": True, "message": f"Email sent to {', '.join(alert_emails)} with {len(data)} records"}
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


class ChatbotEmailRequest(BaseModel):
    question: str
    answer: str
    sql_query: Optional[str] = None
    data: Optional[list] = None
    chart: Optional[str] = None  # base64 encoded image


@app.post("/api/chatbot/email")
async def send_chatbot_insight_email(request: ChatbotEmailRequest):
    """
    Send chatbot insight via email with structured format.
    Includes question, answer, SQL query, data table, and chart image.
    """
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
        
        # Build structured HTML email
        html_parts = []
        
        # Header
        timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html_parts.append(f"""
        <html>
        <head>
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
                .container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px 8px 0 0; }}
                .content {{ background: #f9f9f9; padding: 20px; border: 1px solid #ddd; }}
                .section {{ background: white; padding: 15px; margin: 15px 0; border-radius: 5px; border-left: 4px solid #667eea; }}
                .section h3 {{ margin-top: 0; color: #667eea; }}
                .question {{ background: #e8f4f8; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .answer {{ background: #f0f9f4; padding: 15px; border-radius: 5px; margin: 10px 0; }}
                .sql-code {{ background: #2d2d2d; color: #f8f8f2; padding: 15px; border-radius: 5px; overflow-x: auto; font-family: "Courier New", monospace; font-size: 12px; }}
                table {{ width: 100%; border-collapse: collapse; margin: 15px 0; background: white; }}
                table th {{ background: #667eea; color: white; padding: 12px; text-align: left; font-weight: bold; }}
                table td {{ padding: 10px; border-bottom: 1px solid #ddd; }}
                table tr:hover {{ background: #f5f5f5; }}
                .chart-container {{ text-align: center; margin: 20px 0; }}
                .chart-container img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 5px; }}
                .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; border-top: 1px solid #ddd; margin-top: 20px; }}
            </style>
        </head>
        <body>
        <div class="container">
        <div class="header">
            <h1>📊 MC4 AI Assistant Insight</h1>
            <p>Generated on {timestamp_str}</p>
        </div>
        <div class="content">
        """)
        
        # Question Section - escape HTML in question
        import html as html_escape
        question_escaped = html_escape.escape(request.question)
        html_parts.append(f"""
        <div class="section">
            <h3>❓ Question</h3>
            <div class="question">
                <strong>{question_escaped}</strong>
            </div>
        </div>
        """)
        
        # Answer Section - escape HTML in answer and preserve line breaks
        answer_escaped = html_escape.escape(request.answer)
        answer_formatted = answer_escaped.replace(chr(10), '<br>')
        html_parts.append(f"""
        <div class="section">
            <h3>💡 Answer</h3>
            <div class="answer">
                {answer_formatted}
            </div>
        </div>
        """)
        
        # SQL Query Section - escape HTML in SQL
        if request.sql_query:
            sql_escaped = html_escape.escape(request.sql_query)
            html_parts.append(f"""
            <div class="section">
                <h3>🔍 SQL Query</h3>
                <div class="sql-code">
                    <pre>{sql_escaped}</pre>
                </div>
            </div>
            """)
        
        # Data Table Section
        if request.data and len(request.data) > 0:
            # Convert data to HTML table
            df = pd.DataFrame(request.data)
            table_html = df.to_html(index=False, classes='', escape=False, table_id='data-table')
            # Add styling to table
            table_html = table_html.replace('<table', '<table style="width: 100%; border-collapse: collapse; margin: 15px 0; background: white;"')
            table_html = table_html.replace('<th>', '<th style="background: #667eea; color: white; padding: 12px; text-align: left; font-weight: bold; border: 1px solid #ddd;">')
            table_html = table_html.replace('<td>', '<td style="padding: 10px; border: 1px solid #ddd;">')
            
            html_parts.append(f"""
            <div class="section">
                <h3>📋 Data Results ({len(request.data)} rows)</h3>
                {table_html}
            </div>
            """)
        
        # Chart Section
        if request.chart:
            html_parts.append(f"""
            <div class="section">
                <h3>📈 Visualization</h3>
                <div class="chart-container">
                    <img src="data:image/png;base64,{request.chart}" alt="Generated Chart" />
                </div>
            </div>
            """)
        
        # Footer
        html_parts.append("""
        </div>
        <div class="footer">
            <p>This email was generated by MC4 AI Assistant (Text2SQL)</p>
            <p>MC4 Planning System</p>
        </div>
        </div>
        </body>
        </html>
        """)
        
        html_content = "".join(html_parts)
        
        # Create email subject
        subject = f"MC4 AI Insight: {request.question[:50]}{'...' if len(request.question) > 50 else ''}"
        
        # Create and send email
        message = Mail(
            from_email=from_email,
            to_emails=alert_emails,
            subject=subject,
            html_content=html_content
        )
        
        # If chart exists, attach it as well
        if request.chart:
            chart_data = base64.b64decode(request.chart)
            attachment = Attachment(
                FileContent(base64.b64encode(chart_data).decode()),
                FileName(f"chart_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"),
                FileType("image/png"),
                Disposition("attachment")
            )
            message.add_attachment(attachment)
        
        sg = SendGridAPIClient(sendgrid_api_key)
        response = sg.send(message)
        
        if response.status_code in [200, 201, 202]:
            return {
                "success": True,
                "message": f"Email sent successfully to {', '.join(alert_emails)}"
            }
        raise HTTPException(status_code=500, detail=f"SendGrid: {response.status_code}")
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8011)

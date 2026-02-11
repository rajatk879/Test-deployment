"""
MC4 Forecast Service
====================
Handles all forecasting logic and dataset propagation from forecasted data.
This service generates forecasts and ensures all derived datasets are updated.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import forecast models
try:
    from forecast_models import MC4ForecastModel
    FORECAST_MODELS_AVAILABLE = True
except ImportError:
    FORECAST_MODELS_AVAILABLE = False
    MC4ForecastModel = None


def generate_time_dimension_for_dates(dates):
    """
    Generate time dimension attributes for given dates (same logic as data_generator.py)
    """
    df = pd.DataFrame({"date": dates})
    df["day_of_week"] = df["date"].dt.dayofweek
    
    # Saudi Arabia weekend: From Jan 2022: Saturday-Sunday (dayofweek 5, 6)
    df["is_weekend"] = np.where(
        df["date"] < pd.Timestamp("2022-01-01"),
        df["day_of_week"].isin([4, 5]),   # Fri, Sat
        df["day_of_week"].isin([5, 6]),   # Sat, Sun
    )
    
    # Ramadan dates (approximate Gregorian starts)
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
    return df


def generate_future_forecast(start_date, end_date, dim_sku, forecaster=None, data_cache=None):
    """
    Generate forecast data for future dates using ML models trained on historical data.
    Falls back to simple trend-based forecast if models are not available.
    
    Args:
        start_date: Start date for forecast (pd.Timestamp or datetime)
        end_date: End date for forecast (pd.Timestamp or datetime)
        dim_sku: DataFrame with SKU dimension data
        forecaster: MC4ForecastModel instance (optional)
        data_cache: Dictionary containing cached datasets (optional)
    
    Returns:
        DataFrame with forecast data including columns:
        - sku_id
        - date
        - forecast_tons
        - forecast_units
        - confidence_pct
        - seasonality_index
        - scenario_id
    """
    # Generate dates for the future period
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    if len(dates) == 0:
        return pd.DataFrame()
    
    # Calculate number of periods
    periods = len(dates)
    
    # Get SKU list
    sku_list = dim_sku["sku_id"].unique() if not dim_sku.empty else []
    
    all_forecasts = []
    
    # Get historical forecast data if available
    fcast_historical = pd.DataFrame()
    if data_cache is not None:
        fcast_historical = data_cache.get("fact_sku_forecast", pd.DataFrame())
    
    # Use ML models if available
    if forecaster is not None and FORECAST_MODELS_AVAILABLE:
        print(f"🤖 Generating ML-based forecasts for {periods} days from {start_date.date()} to {end_date.date()}")
        
        # Get the last date from historical data to know where models start
        if not fcast_historical.empty and "date" in fcast_historical.columns:
            model_last_date = pd.to_datetime(fcast_historical["date"]).max()
        else:
            model_last_date = start_date - pd.Timedelta(days=1)
        
        # Calculate how many periods we need from model's last date to end_date
        # Models generate from their last training date, so we need enough periods to cover end_date
        periods_from_model = (end_date - model_last_date).days
        if periods_from_model <= 0:
            periods_from_model = periods
        
        for sku_id in sku_list:
            try:
                # Generate forecast for this SKU
                # The model generates from its last training date, start_date filters the results
                forecast_df = forecaster.forecast(
                    sku_id=sku_id,
                    periods=periods_from_model,
                    start_date=start_date.strftime("%Y-%m-%d") if isinstance(start_date, pd.Timestamp) else str(start_date)
                )
                
                # Ensure we have the right date range
                if not forecast_df.empty:
                    forecast_df = forecast_df[
                        (forecast_df["date"] >= start_date) & 
                        (forecast_df["date"] <= end_date)
                    ].copy()
                
                if not forecast_df.empty:
                    forecast_df["sku_id"] = sku_id
                    all_forecasts.append(forecast_df)
                    
            except Exception as e:
                print(f"⚠️ Forecast failed for {sku_id}: {str(e)}")
                # Fall through to simple model for this SKU
                continue
        
        if all_forecasts:
            result = pd.concat(all_forecasts, ignore_index=True)
            
            # ── Normalize ML forecast to match historical per-SKU scale ──
            # ML models may produce values at aggregate scale rather than per-SKU.
            # Scale each SKU's forecast so the average matches historical per-SKU daily avg.
            if not fcast_historical.empty and "date" in fcast_historical.columns and "forecast_tons" in fcast_historical.columns:
                hist_sku_avg = fcast_historical.groupby("sku_id")["forecast_tons"].mean()
                # Get the last historical date to calculate growth factor
                hist_max_date = pd.to_datetime(fcast_historical["date"]).max()
                days_since_hist = (start_date - hist_max_date).days
                
                for sid in result["sku_id"].unique():
                    ml_avg = result.loc[result["sku_id"] == sid, "forecast_tons"].mean()
                    hist_avg = hist_sku_avg.get(sid, None)
                    if hist_avg and ml_avg and ml_avg > 0:
                        # If ML values are more than 2x off from historical, rescale
                        ratio = hist_avg / ml_avg
                        if ratio < 0.5 or ratio > 2.0:
                            # Apply a small growth factor (2.5% annual) to allow natural growth
                            # Only apply growth if forecasting into the future
                            if days_since_hist > 0:
                                growth = 1.0 + 0.025 * (days_since_hist / 365.0)
                            else:
                                growth = 1.0
                            result.loc[result["sku_id"] == sid, "forecast_tons"] = (
                                result.loc[result["sku_id"] == sid, "forecast_tons"] * ratio * growth
                            ).clip(lower=0).round(2)
                print(f"✅ Normalized ML forecasts to match historical per-SKU scale")
            
            # Enrich with additional fields
            if not dim_sku.empty:
                result = result.merge(dim_sku[["sku_id", "pack_size_kg"]], on="sku_id", how="left")
                result["forecast_units"] = (result["forecast_tons"] * 1000 / result["pack_size_kg"]).round(0).astype(int)
            else:
                result["forecast_units"] = (result["forecast_tons"] * 1000).round(0).astype(int)
            
            # Add confidence and seasonality index
            result["confidence_pct"] = 0.85  # ML models provide better confidence
            result["seasonality_index"] = 1.0  # Will be calculated based on actual patterns
            result["scenario_id"] = "base"
            
            # Calculate seasonality index based on time dimension
            time_dim = generate_time_dimension_for_dates(result["date"])
            result = result.merge(time_dim[["date", "is_ramadan", "is_hajj"]], on="date", how="left")
            result["seasonality_index"] = np.maximum(
                np.where(result["is_ramadan"], 1.35, 1.0),
                np.where(result["is_hajj"], 1.25, 1.0)
            ).round(2)
            result = result.drop(columns=["is_ramadan", "is_hajj"], errors="ignore")
            
            return result[["sku_id", "date", "forecast_tons", "forecast_units",
                           "confidence_pct", "seasonality_index", "scenario_id"]]
    
    # Fallback to simple trend-based forecast if ML models not available
    print(f"⚠️ ML models not available, using simple trend-based forecast")
    
    # Generate time dimension attributes
    time_dim = generate_time_dimension_for_dates(dates)
    
    # Cross-join with SKUs
    t = time_dim[["date", "is_weekend", "is_ramadan", "is_hajj"]].copy()
    t["_k"] = 1
    s = dim_sku[["sku_id", "pack_size_kg"]].copy()
    s["_k"] = 1
    df = t.merge(s, on="_k").drop("_k", axis=1)
    
    # Get historical averages for each SKU as baseline
    if not fcast_historical.empty:
        sku_avg = fcast_historical.groupby("sku_id")["forecast_tons"].mean().to_dict()
    else:
        # Fallback base demand
        sku_avg = {
            "SKU001": 280, "SKU002": 120, "SKU003": 30, "SKU004": 210,
            "SKU005": 90, "SKU006": 120, "SKU007": 80, "SKU008": 150,
            "SKU009": 80, "SKU010": 140, "SKU011": 240, "SKU012": 120,
            "SKU013": 200, "SKU014": 90,
        }
    
    df["base"] = df["sku_id"].map(sku_avg).fillna(100)
    
    # Simple trend projection (2.5% annual growth)
    doy = df["date"].dt.dayofyear
    df["seasonal"] = 0.08 * np.sin(2 * np.pi * doy / 365)
    df["event_mult"] = np.maximum(
        np.where(df["is_ramadan"], 1.35, 1.0),
        np.where(df["is_hajj"], 1.25, 1.0),
    )
    df["weekend_mult"] = np.where(df["is_weekend"], 0.92, 1.0)
    df["trend"] = 1.0 + 0.025 * (df["date"] - pd.Timestamp("2020-01-01")).dt.days / 365.0
    
    df["forecast_tons"] = (
        df["base"] * (1 + df["seasonal"]) * df["event_mult"] * df["weekend_mult"] * df["trend"]
    ).clip(lower=0).round(2)
    
    df["forecast_units"] = (df["forecast_tons"] * 1000 / df["pack_size_kg"]).round(0).astype(int)
    df["confidence_pct"] = 0.75  # Lower confidence for simple model
    df["seasonality_index"] = ((1 + df["seasonal"]) * df["event_mult"]).round(2)
    df["scenario_id"] = "base"
    
    return df[["sku_id", "date", "forecast_tons", "forecast_units",
               "confidence_pct", "seasonality_index", "scenario_id"]]


def generate_forecasts_and_propagate_datasets(
    start_date,
    end_date,
    forecaster=None,
    data_cache=None,
    backend_dir=None,
    data_dir="datasets",
    update_derived_datasets_func=None
):
    """
    Generate forecasts for the given date range and propagate all derived datasets.
    
    This is the main entry point for forecasting. It:
    1. Generates forecasts for the specified date range
    2. Updates the forecast dataset (fact_sku_forecast.csv)
    3. Updates all derived datasets that depend on forecasts
    
    Args:
        start_date: Start date for forecast (pd.Timestamp or datetime)
        end_date: End date for forecast (pd.Timestamp or datetime)
        forecaster: MC4ForecastModel instance (optional)
        data_cache: Dictionary containing cached datasets (required)
        backend_dir: Path to backend directory (required for file operations)
        data_dir: Name of datasets directory (default: "datasets")
        update_derived_datasets_func: Function to update derived datasets (optional)
    
    Returns:
        dict with keys:
        - success: bool
        - message: str
        - records: int (number of forecast records generated)
        - sku_count: int (number of SKUs forecasted)
    """
    if data_cache is None:
        return {
            "success": False,
            "message": "data_cache is required",
            "records": 0,
            "sku_count": 0
        }
    
    try:
        # Get dimension data
        dim_sku = data_cache.get("dim_sku", pd.DataFrame())
        if dim_sku.empty:
            return {
                "success": False,
                "message": "No SKU dimension data available",
                "records": 0,
                "sku_count": 0
            }
        
        # Generate forecasts
        new_forecasts = generate_future_forecast(
            start_date=start_date,
            end_date=end_date,
            dim_sku=dim_sku,
            forecaster=forecaster,
            data_cache=data_cache
        )
        
        if new_forecasts.empty:
            return {
                "success": False,
                "message": "No forecasts generated",
                "records": 0,
                "sku_count": 0
            }
        
        # Update forecast dataset
        _update_forecast_dataset_internal(
            new_forecasts=new_forecasts,
            data_cache=data_cache,
            backend_dir=backend_dir,
            data_dir=data_dir
        )
        
        # Update all derived datasets
        if update_derived_datasets_func is not None:
            min_date = new_forecasts["date"].min()
            max_date = new_forecasts["date"].max()
            update_derived_datasets_func(min_date, max_date)
        
        return {
            "success": True,
            "message": f"Generated {len(new_forecasts)} forecast records",
            "records": len(new_forecasts),
            "sku_count": new_forecasts["sku_id"].nunique()
        }
        
    except Exception as e:
        print(f"⚠️ Error in generate_forecasts_and_propagate_datasets: {e}")
        import traceback
        traceback.print_exc()
        return {
            "success": False,
            "message": f"Error: {str(e)}",
            "records": 0,
            "sku_count": 0
        }


def _update_forecast_dataset_internal(new_forecasts, data_cache, backend_dir=None, data_dir="datasets"):
    """
    Internal function to update the forecast dataset.
    This updates fact_sku_forecast.csv and sku_forecast.csv.
    """
    if new_forecasts.empty:
        return
    
    if backend_dir is None:
        print("⚠️ backend_dir not provided, cannot update forecast dataset files")
        # Still update cache if possible
        if data_cache is not None:
            existing = data_cache.get("fact_sku_forecast", pd.DataFrame())
            if not existing.empty:
                combined = pd.concat([existing, new_forecasts], ignore_index=True)
            else:
                combined = new_forecasts.copy()
            data_cache["fact_sku_forecast"] = combined
        return
    
    try:
        fcast_path = os.path.join(backend_dir, data_dir, "fact_sku_forecast.csv")
        sku_fcast_path = os.path.join(backend_dir, data_dir, "sku_forecast.csv")
        
        # Load existing data
        if os.path.exists(fcast_path):
            existing_df = pd.read_csv(fcast_path)
            existing_df["date"] = pd.to_datetime(existing_df["date"])
        else:
            existing_df = pd.DataFrame()
        
        # Get max date from existing data
        max_existing_date = existing_df["date"].max() if not existing_df.empty else None
        
        # Filter new forecasts to only include dates after max_existing_date
        new_forecasts = new_forecasts.copy()
        new_forecasts["date"] = pd.to_datetime(new_forecasts["date"])
        if max_existing_date:
            new_forecasts = new_forecasts[new_forecasts["date"] > max_existing_date]
        
        if new_forecasts.empty:
            return
        
        # Combine and deduplicate
        if not existing_df.empty:
            # Remove any overlapping dates from existing data
            existing_df = existing_df[existing_df["date"] < new_forecasts["date"].min()]
            combined_df = pd.concat([existing_df, new_forecasts], ignore_index=True)
        else:
            combined_df = new_forecasts.copy()
        
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
        if data_cache is not None:
            data_cache["fact_sku_forecast"] = combined_df.copy()
        
        print(f"✅ Updated forecast datasets (fact_sku_forecast.csv & sku_forecast.csv) with {len(new_forecasts)} new records")
        
    except Exception as e:
        print(f"⚠️ Error updating forecast dataset: {e}")
        import traceback
        traceback.print_exc()

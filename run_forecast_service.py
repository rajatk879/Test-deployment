#!/usr/bin/env python3
"""
Standalone script to run the forecast service
==============================================
This script allows you to generate forecasts directly from the command line.

Usage:
    python run_forecast_service.py [--start-date YYYY-MM-DD] [--end-date YYYY-MM-DD] [--days-ahead N]

Examples:
    # Generate forecasts for next 365 days from last available date
    python run_forecast_service.py

    # Generate forecasts for specific date range
    python run_forecast_service.py --start-date 2026-02-11 --end-date 2027-02-10

    # Generate forecasts for next 30 days
    python run_forecast_service.py --days-ahead 30
"""

import os
import sys
import argparse
import pandas as pd
from datetime import datetime, timedelta

# Add current directory to path
_backend_dir = os.path.dirname(os.path.abspath(__file__))
if _backend_dir not in sys.path:
    sys.path.insert(0, _backend_dir)

# Import forecast service and models
try:
    from forecast_service import (
        generate_forecasts_and_propagate_datasets,
        generate_future_forecast
    )
    from forecast_models import MC4ForecastModel, train_and_save_models
    FORECAST_AVAILABLE = True
except ImportError as e:
    print(f"❌ Error importing forecast modules: {e}")
    print("   Make sure you're running this from the backend directory")
    sys.exit(1)

# Import data generator for dataset updates
try:
    import data_generator as dg
    DATA_GENERATOR_AVAILABLE = True
except ImportError:
    DATA_GENERATOR_AVAILABLE = False
    print("⚠️ Data generator not available - derived datasets won't be updated")

DATA_DIR = "datasets"
MODEL_DIR = "models"


def load_data_cache():
    """Load all datasets into a cache dictionary"""
    cache = {}
    datasets_dir = os.path.join(_backend_dir, DATA_DIR)
    
    if not os.path.exists(datasets_dir):
        print(f"❌ Datasets directory not found: {datasets_dir}")
        return None
    
    # Load dimension tables
    dim_files = {
        "dim_sku": "dim_sku.csv",
        "time_dimension": "time_dimension.csv",
    }
    
    # Load fact tables
    fact_files = {
        "fact_sku_forecast": "fact_sku_forecast.csv",
    }
    
    all_files = {**dim_files, **fact_files}
    
    for key, filename in all_files.items():
        filepath = os.path.join(datasets_dir, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                # Convert date columns
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                cache[key] = df
                print(f"✅ Loaded {key}: {len(df)} records")
            except Exception as e:
                print(f"⚠️ Error loading {filename}: {e}")
        else:
            print(f"⚠️ File not found: {filename}")
    
    return cache


def initialize_forecaster(data_cache):
    """Initialize and load forecast models"""
    model_dir_path = os.path.join(_backend_dir, MODEL_DIR)
    os.makedirs(model_dir_path, exist_ok=True)
    
    # Check if models exist
    fcast_df = data_cache.get("fact_sku_forecast", pd.DataFrame())
    if fcast_df.empty:
        print("⚠️ No forecast data available for training")
        return None
    
    sku_list = fcast_df["sku_id"].unique()
    existing_models = [f for f in os.listdir(model_dir_path) if f.endswith(".pkl")]
    
    # Train models if they don't exist
    if len(existing_models) < len(sku_list):
        print(f"🔄 Training forecast models for {len(sku_list)} SKUs...")
        fcast_path = os.path.join(_backend_dir, DATA_DIR, "fact_sku_forecast.csv")
        time_dim_path = os.path.join(_backend_dir, DATA_DIR, "time_dimension.csv")
        
        if not os.path.exists(fcast_path):
            fcast_df.to_csv(fcast_path, index=False)
        
        try:
            forecaster = train_and_save_models(
                data_path=fcast_path,
                time_dim_path=time_dim_path,
                model_dir=model_dir_path
            )
            print("✅ Forecast models trained successfully")
            return forecaster
        except Exception as e:
            print(f"⚠️ Error training models: {e}")
            return None
    else:
        # Load existing models
        print(f"📦 Loading existing forecast models from {model_dir_path}...")
        forecaster = MC4ForecastModel(model_dir=model_dir_path)
        
        # Load models
        loaded_count = 0
        for sku_id in sku_list:
            model_path = os.path.join(model_dir_path, f"model_{sku_id}.pkl")
            if os.path.exists(model_path):
                try:
                    import pickle
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                        forecaster.models[sku_id] = model
                        loaded_count += 1
                except Exception as e:
                    print(f"   ⚠️ Failed to load model for {sku_id}: {str(e)}")
        
        print(f"✅ Loaded {loaded_count}/{len(sku_list)} forecast models")
        return forecaster


def update_derived_datasets(min_date, max_date):
    """Update derived datasets using data_generator"""
    if not DATA_GENERATOR_AVAILABLE:
        print("⚠️ Data generator not available - skipping derived dataset updates")
        return
    
    try:
        print(f"🔄 Updating derived datasets for date range {min_date.date()} to {max_date.date()}...")
        dg.update_derived_datasets(min_date, max_date)
        print("✅ Derived datasets updated")
    except Exception as e:
        print(f"⚠️ Error updating derived datasets: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate forecasts using the forecast service",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--start-date",
        type=str,
        help="Start date for forecast (YYYY-MM-DD). Default: day after last available forecast"
    )
    parser.add_argument(
        "--end-date",
        type=str,
        help="End date for forecast (YYYY-MM-DD). Default: start_date + days_ahead"
    )
    parser.add_argument(
        "--days-ahead",
        type=int,
        default=365,
        help="Number of days to forecast ahead (default: 365). Ignored if --end-date is provided"
    )
    parser.add_argument(
        "--skip-derived",
        action="store_true",
        help="Skip updating derived datasets"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("MC4 Forecast Service - Standalone Runner")
    print("=" * 60)
    
    # Load data cache
    print("\n📊 Loading datasets...")
    data_cache = load_data_cache()
    if data_cache is None:
        print("❌ Failed to load data cache")
        sys.exit(1)
    
    # Initialize forecaster
    print("\n🤖 Initializing forecast models...")
    forecaster = initialize_forecaster(data_cache)
    if forecaster is None:
        print("⚠️ Forecaster not available - will use fallback simple forecast")
    
    # Determine date range
    fcast = data_cache.get("fact_sku_forecast", pd.DataFrame())
    
    if args.start_date:
        from_dt = pd.to_datetime(args.start_date)
    else:
        if fcast.empty or "date" not in fcast.columns:
            from_dt = pd.Timestamp("2026-02-11")  # Default start
        else:
            max_date = pd.to_datetime(fcast["date"]).max()
            from_dt = max_date + pd.Timedelta(days=1)
    
    if args.end_date:
        to_dt = pd.to_datetime(args.end_date)
    else:
        to_dt = from_dt + pd.Timedelta(days=args.days_ahead)
    
    print(f"\n📈 Generating forecasts from {from_dt.date()} to {to_dt.date()}")
    print(f"   ({(to_dt - from_dt).days} days)")
    
    # Generate forecasts
    print("\n🔄 Running forecast service...")
    result = generate_forecasts_and_propagate_datasets(
        start_date=from_dt,
        end_date=to_dt,
        forecaster=forecaster,
        data_cache=data_cache,
        backend_dir=_backend_dir,
        data_dir=DATA_DIR,
        update_derived_datasets_func=None if args.skip_derived else update_derived_datasets
    )
    
    if result["success"]:
        print("\n" + "=" * 60)
        print("✅ Forecast generation completed successfully!")
        print("=" * 60)
        print(f"   Records generated: {result['records']}")
        print(f"   SKUs forecasted: {result['sku_count']}")
        print(f"   Date range: {from_dt.date()} to {to_dt.date()}")
        print(f"\n   Forecasts saved to: {os.path.join(_backend_dir, DATA_DIR, 'fact_sku_forecast.csv')}")
    else:
        print("\n" + "=" * 60)
        print("❌ Forecast generation failed!")
        print("=" * 60)
        print(f"   Error: {result['message']}")
        sys.exit(1)


if __name__ == "__main__":
    main()

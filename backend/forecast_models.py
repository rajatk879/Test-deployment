"""
MC4 Forecast Models
Time-series forecasting for SKU demand with seasonality handling
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# Try to import Prophet, fallback to statsmodels if not available
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except (ImportError, AttributeError):
    PROPHET_AVAILABLE = False
    print("⚠️ Prophet not available, using statsmodels as fallback")

# Import statsmodels as fallback
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("⚠️ statsmodels not available")

# Model wrapper classes (must be top-level for pickling)
class StatsModelWrapper:
    """Wrapper for statsmodels ExponentialSmoothing to match Prophet interface"""
    def __init__(self, model_params, last_date, last_value):
        # Save only serializable parameters, not the full model
        self.model_params = model_params  # dict of fitted parameters
        self.last_date = pd.to_datetime(last_date)
        self.last_value = float(last_value)
        self.model_type = 'statsmodels'
    
    def make_future_dataframe(self, periods, freq='D'):
        future_dates = pd.date_range(start=self.last_date, periods=periods + 1, freq=freq)[1:]
        return pd.DataFrame({'ds': future_dates})
    
    def predict(self, future):
        # Simple forecast based on saved parameters
        n_periods = len(future)
        # Use exponential smoothing formula: y_t = alpha * y_{t-1} + (1-alpha) * level
        alpha = self.model_params.get('smoothing_level', 0.3)
        level = self.model_params.get('initial_level', self.last_value)
        trend = self.model_params.get('smoothing_trend', 0.0)
        
        forecast_values = []
        current_level = level
        for i in range(n_periods):
            current_level = alpha * self.last_value + (1 - alpha) * current_level + trend * (i + 1)
            forecast_values.append(max(0, current_level))  # Ensure non-negative
        
        future['yhat'] = forecast_values
        future['yhat_lower'] = np.array(forecast_values) * 0.9
        future['yhat_upper'] = np.array(forecast_values) * 1.1
        return future

class SimpleModel:
    """Simple trend-based model - fully pickleable and reliable"""
    def __init__(self, data):
        # Store only primitive values for easy pickling
        self.mean = float(data['y'].mean())
        # Calculate trend from recent data (last 30 days or all if less)
        recent_data = data['y'].tail(min(30, len(data)))
        if len(recent_data) > 1:
            self.trend = float((recent_data.iloc[-1] - recent_data.iloc[0]) / len(recent_data))
        else:
            self.trend = 0.0
        
        # Use rolling std for better uncertainty estimate
        if len(data) > 7:
            rolling_std = data['y'].tail(30).std() if len(data) >= 30 else data['y'].std()
            self.std = float(rolling_std) if not np.isnan(rolling_std) else float(data['y'].iloc[0] * 0.1)
        else:
            self.std = float(data['y'].iloc[0] * 0.1) if len(data) > 0 else 1.0
        
        self.last_value = float(data['y'].iloc[-1])
        self.last_date = pd.to_datetime(data['ds'].iloc[-1])
        self.model_type = 'simple'
    
    def make_future_dataframe(self, periods, freq='D'):
        future_dates = pd.date_range(start=self.last_date, periods=periods + 1, freq=freq)[1:]
        return pd.DataFrame({'ds': future_dates})
    
    def predict(self, future):
        n_periods = len(future)
        base_value = self.last_value
        # Linear trend projection
        forecast_values = base_value + self.trend * np.arange(1, n_periods + 1)
        # Ensure non-negative
        forecast_values = np.maximum(forecast_values, 0)
        future['yhat'] = forecast_values
        # Confidence intervals
        future['yhat_lower'] = np.maximum(forecast_values - 1.96 * self.std, 0)  # 95% CI
        future['yhat_upper'] = forecast_values + 1.96 * self.std
        return future

class MC4ForecastModel:
    """Forecast model for MC4 SKU demand"""
    
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(model_dir, exist_ok=True)
        self.models = {}
    
    # def prepare_data(self, df, sku_id=None):
    #     """Prepare data for Prophet"""
    #     if sku_id:
    #         df = df[df["sku_id"] == sku_id].copy()
        
    #     # Aggregate by date
    #     df_agg = df.groupby("date")["forecast_tons"].sum().reset_index()
    #     df_agg["date"] = pd.to_datetime(df_agg["date"])
    #     df_agg = df_agg.sort_values("date")
    #     df_agg.columns = ["ds", "y"]
        
    #     return df_agg
    def prepare_data(self, df, sku_id=None):
        if sku_id:
            df = df[df["sku_id"] == sku_id].copy()

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date")

        df_prep = df[["date", "forecast_tons"]].copy()
        df_prep.columns = ["ds", "y"]

        return df_prep

    
    def train_model(self, df, sku_id, include_holidays=True):
        """Train forecasting model for a specific SKU"""
        df_prep = self.prepare_data(df, sku_id)
        
        if len(df_prep) < 30:
            raise ValueError(f"Insufficient data for SKU {sku_id}")
        
        # Try Prophet first, fallback to statsmodels
        if PROPHET_AVAILABLE:
            try:
                return self._train_prophet(df_prep, include_holidays)
            except Exception as e:
                print(f"   ⚠️ Prophet failed for {sku_id}, using fallback: {str(e)}")
        
        # Fallback to statsmodels ExponentialSmoothing (if available)
        # Note: For reliability, we'll use SimpleModel instead since statsmodels has pickling issues
        # if STATSMODELS_AVAILABLE:
        #     try:
        #         return self._train_statsmodels(df_prep)
        #     except Exception as e:
        #         print(f"   ⚠️ statsmodels failed, using simple model: {str(e)}")
        
        # Use simple model (reliable and fully pickleable)
        return self._train_simple(df_prep)
    
    def _train_prophet(self, df_prep, include_holidays=True):
        """Train Prophet model"""

        model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            daily_seasonality=False,
            seasonality_mode='multiplicative',
            changepoint_prior_scale=0.05
        )

        # Use real Ramadan / Hajj days from time_dimension.csv (loaded in train_and_save_models)
        if include_holidays and hasattr(self, "holidays_df") and self.holidays_df is not None:
            model.holidays = self.holidays_df

        # Fit model
        model.fit(df_prep)

        return model

    
    def _train_statsmodels(self, df_prep):
        """Train statsmodels ExponentialSmoothing model"""
        # Set date as index
        df_indexed = df_prep.set_index('ds')
        
        # Use ExponentialSmoothing with trend and seasonality
        try:
            # Try with seasonality first
            fitted_model = ExponentialSmoothing(
                df_indexed['y'],
                trend='add',
                seasonal='add',
                seasonal_periods=min(365, len(df_indexed) // 2)  # Annual seasonality, but adjust if not enough data
            ).fit(optimized=True, remove_bias=False)
        except Exception as e:
            try:
                # Fallback to simpler model with just trend
                fitted_model = ExponentialSmoothing(
                    df_indexed['y'],
                    trend='add'
                ).fit(optimized=True, remove_bias=False)
            except Exception as e2:
                # If statsmodels also fails, use simple model
                print(f"   ⚠️ statsmodels failed, using simple model: {str(e2)}")
                return self._train_simple(df_prep)
        
        # Extract only serializable parameters from the fitted model
        model_params = {
            'smoothing_level': float(fitted_model.params.get('smoothing_level', 0.3)),
            'smoothing_trend': float(fitted_model.params.get('smoothing_trend', 0.0)),
            'initial_level': float(fitted_model.params.get('initial_level', df_indexed['y'].iloc[0])),
        }
        
        last_date = df_indexed.index[-1]
        last_value = float(df_indexed['y'].iloc[-1])
        
        return StatsModelWrapper(model_params, last_date, last_value)
    
    def _train_simple(self, df_prep):
        """Simple trend model as last resort"""
        return SimpleModel(df_prep)
    
    def train_all_skus(self, df, sku_list=None):
        """Train models for all SKUs"""
        if sku_list is None:
            sku_list = df["sku_id"].unique()
        
        print(f"🔄 Training models for {len(sku_list)} SKUs...")
        for i, sku_id in enumerate(sku_list, 1):
            try:
                print(f"   [{i}/{len(sku_list)}] Training {sku_id}...")
                model = self.train_model(df, sku_id)
                self.models[sku_id] = model
                
                # Save model
                model_path = os.path.join(self.model_dir, f"model_{sku_id}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(model, f)
                    
            except Exception as e:
                print(f"   ⚠️ Failed to train {sku_id}: {str(e)}")
        
        print(f"✅ Trained {len(self.models)} models successfully")
    
    def forecast(self, sku_id, periods=30, start_date=None):
        """Generate forecast for a SKU"""
        if sku_id not in self.models:
            # Try to load from disk
            model_path = os.path.join(self.model_dir, f"model_{sku_id}.pkl")
            if os.path.exists(model_path):
                try:
                    with open(model_path, 'rb') as f:
                        self.models[sku_id] = pickle.load(f)
                except Exception as e:
                    raise ValueError(f"Failed to load model for SKU {sku_id}: {str(e)}")
            else:
                raise ValueError(f"No model found for SKU {sku_id}")
        
        model = self.models[sku_id]
        
        # Create future dataframe
        if start_date:
            future = model.make_future_dataframe(periods=periods, freq='D')
            future = future[future['ds'] >= pd.to_datetime(start_date)]
        else:
            future = model.make_future_dataframe(periods=periods, freq='D')
        
        # Predict
        forecast = model.predict(future)
        
        # Handle both Prophet and custom model outputs
        if 'ds' in forecast.columns:
            result = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            result.rename(columns={
                'ds': 'date',
                'yhat': 'forecast_tons',
                'yhat_lower': 'forecast_lower',
                'yhat_upper': 'forecast_upper'
            }, inplace=True)
            return result
        else:
            # Ensure we have the required columns
            if 'yhat' not in forecast.columns:
                raise ValueError("Model prediction missing 'yhat' column")
            
            # Create date column if missing
            if 'date' not in forecast.columns and 'ds' not in forecast.columns:
                if hasattr(forecast, 'index') and isinstance(forecast.index, pd.DatetimeIndex):
                    forecast['date'] = forecast.index
                else:
                    forecast['date'] = pd.date_range(start=datetime.now(), periods=len(forecast), freq='D')
            
            # Use 'ds' if available, otherwise 'date'
            date_col = 'ds' if 'ds' in forecast.columns else 'date'
            
            result = forecast[[date_col, 'yhat', 'yhat_lower', 'yhat_upper']].copy()
            result.rename(columns={
                date_col: 'date',
                'yhat': 'forecast_tons',
                'yhat_lower': 'forecast_lower',
                'yhat_upper': 'forecast_upper'
            }, inplace=True)
            return result
    
    def forecast_all(self, periods=30, start_date=None):
        """Generate forecasts for all trained SKUs"""
        all_forecasts = []
        for sku_id in self.models.keys():
            try:
                forecast = self.forecast(sku_id, periods, start_date)
                forecast['sku_id'] = sku_id
                all_forecasts.append(forecast)
            except Exception as e:
                print(f"⚠️ Forecast failed for {sku_id}: {str(e)}")
        
        if all_forecasts:
            return pd.concat(all_forecasts, ignore_index=True)
        return pd.DataFrame()

def train_and_save_models(
    data_path="datasets/sku_forecast.csv",
    time_dim_path="datasets/time_dimension.csv",
    model_dir="models"
):
    """Train all models from historical data with proper Arabian holiday handling"""

    print("📊 Loading historical data...")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"])

    # Filter to historical data only (before today for training)
    today = datetime.now()
    df_train = df[df["date"] < today].copy()

    if len(df_train) == 0:
        raise ValueError("No historical data found for training")

    print(f"📈 Training on {len(df_train)} records from {df_train['date'].min()} to {df_train['date'].max()}")

    # -------------------------------------------------
    # Load time dimension for real Ramadan / Hajj flags
    # -------------------------------------------------
    holidays_df = None
    if os.path.exists(time_dim_path):
        print("📅 Loading time dimension for holiday signals...")
        time_dim = pd.read_csv(time_dim_path)
        time_dim["date"] = pd.to_datetime(time_dim["date"])

        ramadan_days = time_dim[time_dim["is_ramadan"] == True][["date"]].copy()
        ramadan_days["holiday"] = "ramadan"

        hajj_days = time_dim[time_dim["is_hajj"] == True][["date"]].copy()
        hajj_days["holiday"] = "hajj"

        holidays_df = pd.concat([ramadan_days, hajj_days], ignore_index=True)
        holidays_df.rename(columns={"date": "ds"}, inplace=True)

        print(f"✅ Holidays loaded: Ramadan={len(ramadan_days)} days, Hajj={len(hajj_days)} days")

    else:
        print(f"⚠️ time_dimension.csv not found at {time_dim_path}. Training without holiday signals.")

    # Train models
    forecaster = MC4ForecastModel(model_dir=model_dir)

    # Attach holidays into forecaster so Prophet can use them
    forecaster.holidays_df = holidays_df

    sku_list = df_train["sku_id"].unique()
    forecaster.train_all_skus(df_train, sku_list)

    return forecaster


if __name__ == "__main__":
    # Train models
    forecaster = train_and_save_models()
    print("✅ Model training complete!")
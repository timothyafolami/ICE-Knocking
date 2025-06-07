import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from statsmodels.tsa.statespace.sarimax import SARIMAX
from datetime import timedelta
import logging
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s'
)
logger = logging.getLogger(__name__)

# Create output directories
def create_output_dirs():
    """Create directory structure for outputs"""
    dirs = ['outputs', 'outputs/predictions', 'outputs/forecasts', 'outputs/feature_importance', 'outputs/models', 'outputs/scalers']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

class FeatureEngineering:
    def __init__(self):
        self.scalers = {}

    def transform_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df_processed = df.copy()
        df_processed['Timestamp'] = pd.to_datetime(df_processed['Timestamp'])
        df_processed['hour'] = df_processed['Timestamp'].dt.hour
        df_processed['day_of_week'] = df_processed['Timestamp'].dt.dayofweek
        df_processed['is_weekend'] = df_processed['Timestamp'].dt.dayofweek.isin([5, 6]).astype(int)
        return df_processed

    def transform_target(self, feature_name: str, y: pd.Series) -> np.ndarray:
        return y.values

    def inverse_transform_target(self, feature_name: str, y: np.ndarray) -> np.ndarray:
        return y

class ModelTrainer:
    def __init__(self):
        self.models = {}
        self.feature_names = {}

    def train_models(self, X: pd.DataFrame, y: pd.Series, feature_name: str):
        logger.info(f"Training SARIMAX for {feature_name}")
        order = (1, 1, 1)
        seasonal_order = (1, 1, 1, 24)
        try:
            sarimax_model = SARIMAX(
                y,
                exog=X,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            sarimax_fit = sarimax_model.fit(disp=False)
            self.models[feature_name] = {'sarimax': sarimax_fit}
            self.feature_names[feature_name] = X.columns.tolist()
            logger.info(f"Completed training SARIMAX for {feature_name}")
        except Exception as e:
            logger.error(f"Training failed for SARIMAX on {feature_name}: {str(e)}")
            raise

    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series, feature_name: str, feature_eng: 'FeatureEngineering'):
        logger.info(f"Evaluating models for {feature_name}")
        model = self.models[feature_name]['sarimax']
        feature_cols = self.feature_names[feature_name]
        predictions = model.predict(start=0, end=len(y_test)-1, exog=X_test[feature_cols])
        
        y_test_orig = feature_eng.inverse_transform_target(feature_name, y_test)
        pred_orig = feature_eng.inverse_transform_target(feature_name, predictions)
        
        metrics = {
            'r2_score': r2_score(y_test_orig, pred_orig),
            'mse': mean_squared_error(y_test_orig, pred_orig),
            'mae': mean_absolute_error(y_test_orig, pred_orig)
        }
        
        logger.info(f"Metrics for {feature_name} (SARIMAX):")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")
        
        return predictions, metrics

    def generate_forecast(self, X: pd.DataFrame, y: pd.Series, feature_name: str, 
                        last_timestamp: pd.Timestamp, feature_eng: 'FeatureEngineering',
                        n_steps: int = 24*14) -> pd.DataFrame:
        logger.info(f"Generating {n_steps}-step forecast for {feature_name}")
        
        model = self.models[feature_name]['sarimax']
        feature_cols = self.feature_names[feature_name]
        
        max_lag = max([int(col.split('_')[-1]) for col in feature_cols if 'lag_' in col] or [0])
        max_window = max([int(col.split('_')[-1]) for col in feature_cols if 'rolling_' in col] or [0])
        N = max(max_lag, max_window, 1)
        
        historical_df = pd.concat([X, y.rename(feature_name)], axis=1)
        buffer_df = historical_df.iloc[-N:].copy()
        
        future_timestamps = pd.date_range(
            start=last_timestamp + timedelta(hours=1),
            periods=n_steps,
            freq='H'
        )
        
        forecast_df = pd.DataFrame({'Timestamp': future_timestamps})
        predictions = []
        
        for step in range(n_steps):
            next_timestamp = future_timestamps[step]
            
            next_exog = {
                'hour': next_timestamp.hour,
                'day_of_week': next_timestamp.dayofweek,
                'is_weekend': int(next_timestamp.dayofweek in [5, 6]),
            }
            
            for lag in [1, 2, 3, 24]:
                lag_col = f'{feature_name}_lag_{lag}'
                if lag_col in feature_cols:
                    lag_value = (buffer_df[feature_name].iloc[-lag] 
                                if len(buffer_df) >= lag 
                                else buffer_df[feature_name].iloc[0])
                    next_exog[lag_col] = lag_value
            
            for window in [3, 6]:
                mean_col = f'{feature_name}_rolling_mean_{window}'
                if mean_col in feature_cols:
                    rolling_mean = buffer_df[feature_name].tail(window).mean()
                    next_exog[mean_col] = rolling_mean
            
            diff_col = f'{feature_name}_diff_1'
            if diff_col in feature_cols:
                diff_1 = (buffer_df[feature_name].iloc[-1] - buffer_df[feature_name].iloc[-2] 
                        if len(buffer_df) >= 2 else 0)
                next_exog[diff_col] = diff_1
            
            next_exog_df = pd.DataFrame([next_exog])
            next_exog_array = next_exog_df[feature_cols].to_numpy()
            
            pred = model.forecast(steps=1, exog=next_exog_array)
            
            if isinstance(pred, pd.Series):
                pred_value = pred.iloc[0] if not pred.empty else 0.0
            elif isinstance(pred, np.ndarray):
                pred_value = pred[0] if pred.size > 0 else 0.0
            else:
                pred_value = float(pred) if pred is not None else 0.0
            
            predictions.append(pred_value)
            
            new_row = {'Timestamp': next_timestamp, feature_name: pred_value}
            new_row.update(next_exog)
            buffer_df = pd.concat([buffer_df, pd.DataFrame([new_row])], ignore_index=True)
            buffer_df = buffer_df.iloc[-N:]
        
        original_predictions = feature_eng.inverse_transform_target(feature_name, np.array(predictions))
        forecast_df['sarimax_forecast'] = original_predictions
        
        if not np.any(np.isnan(original_predictions)) and not np.any(np.isinf(original_predictions)):
            logger.info(f"Forecast statistics for SARIMAX (original scale):")
            logger.info(f"  Mean: {np.mean(original_predictions):.2f}")
            logger.info(f"  Std: {np.std(original_predictions):.2f}")
            logger.info(f"  Min: {np.min(original_predictions):.2f}")
            logger.info(f"  Max: {np.max(original_predictions):.2f}")
        
        plt.figure(figsize=(15, 8))
        plt.plot(forecast_df['Timestamp'], forecast_df['sarimax_forecast'], 
                label='SARIMAX Forecast', linestyle='--', alpha=0.7)
        plt.title(f'{feature_name} - 1-Day Forecast (Original Scale)')
        plt.xlabel('Timestamp')
        plt.ylabel(feature_name)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'outputs/forecasts/{feature_name}_forecast.png')
        plt.close()
        
        return forecast_df

def prepare_features(df: pd.DataFrame, target_feature: str, feature_eng: 'FeatureEngineering') -> tuple:
    df_processed = feature_eng.transform_features(df)
    
    for lag in [1, 2, 3, 24]:
        df_processed[f'{target_feature}_lag_{lag}'] = df_processed[target_feature].shift(lag)
    
    for window in [3, 6]:
        df_processed[f'{target_feature}_rolling_mean_{window}'] = df_processed[target_feature].rolling(window=window).mean()
    
    df_processed[f'{target_feature}_diff_1'] = df_processed[target_feature].diff(1)
    
    feature_columns = ['hour', 'day_of_week', 'is_weekend'] + \
                     [col for col in df_processed.columns if col.startswith(f'{target_feature}_')]
    
    df_clean = df_processed.dropna()
    logger.info(f"Dropped {len(df_processed) - len(df_clean)} rows due to NaN values for {target_feature}")
    
    X = df_clean[feature_columns]
    y = df_clean[target_feature]
    timestamps = df_clean['Timestamp']
    
    return X, y, timestamps

def plot_predictions(timestamps_test: pd.Series, y_test: pd.Series, predictions: dict, feature_name: str):
    plt.figure(figsize=(15, 8))
    plt.plot(timestamps_test, y_test, label='Actual', alpha=0.7)
    plt.plot(timestamps_test, predictions['sarimax'], label='SARIMAX Prediction', linestyle='--', alpha=0.7)
    plt.title(f'{feature_name} Predictions (Original Scale)')
    plt.xlabel('Timestamp')
    plt.ylabel(feature_name)
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f'outputs/predictions/{feature_name}_predictions.png')
    plt.close()

def main():
    create_output_dirs()
    logger.info("Loading data from CSV file")
    file_path = './data/engine_knock_data_hourly.csv'
    df = pd.read_csv(file_path)
    logger.info(f"Loaded data shape: {df.shape}")
    
    feature_eng = FeatureEngineering()
    trainer = ModelTrainer()
    features = ['RPM', 'CylinderPressure', 'BurnRate', 'Vibration', 'EGOVoltage', 'TempSensor']
    test_size = 0.1
    
    for feature in features:
        logger.info(f"\nProcessing feature: {feature}")
        
        X, y, timestamps = prepare_features(df, feature, feature_eng)
        X_train, X_test, y_train, y_test, timestamps_train, timestamps_test = train_test_split(
            X, y, timestamps, test_size=test_size, shuffle=False
        )
        
        trainer.train_models(X_train, y_train, feature)
        predictions, metrics = trainer.evaluate_models(X_test, y_test, feature, feature_eng)
        
        logger.info(f"Prediction statistics for {feature} (original scale):")
        logger.info("  sarimax:")
        for metric, value in metrics.items():
            logger.info(f"    {metric}: {value:.4f}")
        
        plot_predictions(timestamps_test, y_test, {'sarimax': predictions}, feature)
        
        last_timestamp = timestamps.iloc[-1]
        forecast_df = trainer.generate_forecast(X, y, feature, last_timestamp, feature_eng)
        forecast_df.to_csv(f'outputs/forecasts/{feature}_forecast.csv', index=False)
        logger.info(f"Forecast saved to outputs/forecasts/{feature}_forecast.csv")

if __name__ == "__main__":
    main()
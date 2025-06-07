import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import warnings
from loguru import logger
import sys
import joblib
import os
from datetime import timedelta
from typing import Dict, List, Tuple
import uuid

# Configure logger
logger.remove()
logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")
logger.add("ml_engine_knock_analysis.log", rotation="500 MB")

# Suppress warnings
warnings.filterwarnings('ignore')

# Create output directories
def create_output_dirs():
    """Create directory structure for outputs"""
    dirs = ['outputs', 'outputs/predictions', 'outputs/forecasts', 'outputs/feature_importance', 'outputs/models', 'outputs/scalers']
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)

class FeatureEngineering:
    def __init__(self):
        self.scalers = {}  # Store scalers for each feature
        
    def prepare_features(self, df: pd.DataFrame, target_feature: str) -> tuple:
        """Prepare features for modeling using only the target feature's lags and rolling statistics"""
        # Create a copy of the dataframe
        df_processed = df.copy()
        
        # Create time-based features
        df_processed['hour'] = df_processed['Timestamp'].dt.hour
        df_processed['day_of_week'] = df_processed['Timestamp'].dt.dayofweek
        df_processed['month'] = df_processed['Timestamp'].dt.month
        df_processed['is_weekend'] = df_processed['day_of_week'].isin([5, 6]).astype(int)
        
        # Scale the target feature
        self.scalers[target_feature] = MinMaxScaler()
        df_processed[target_feature] = self.scalers[target_feature].fit_transform(df_processed[[target_feature]].values)
        
        # Create lag features for the target feature only
        for lag in range(1, 4):
            df_processed[f'{target_feature}_lag_{lag}'] = df_processed[target_feature].shift(lag)
        
        # Create rolling features for the target feature only
        for window in [10, 30, 60]:
            df_processed[f'{target_feature}_rolling_mean_{window}'] = df_processed[target_feature].rolling(window=window).mean()
            df_processed[f'{target_feature}_rolling_std_{window}'] = df_processed[target_feature].rolling(window=window).std()
        
        # Drop NaN values
        df_processed = df_processed.dropna()
        
        # Prepare X (features) and y (target)
        feature_columns = ['hour', 'day_of_week', 'month', 'is_weekend'] + \
                          [col for col in df_processed.columns if col.startswith(f'{target_feature}_lag_') or col.startswith(f'{target_feature}_rolling_')]
        X = df_processed[feature_columns]
        y = df_processed[target_feature]
        
        # Save scaler for the target feature
        joblib.dump(self.scalers[target_feature], f'outputs/scalers/{target_feature}_scaler.joblib')
        
        return X, y, df_processed['Timestamp']
    
    def inverse_transform_target(self, target_feature: str, values: np.ndarray) -> np.ndarray:
        """Inverse transform the target feature values"""
        if target_feature in self.scalers:
            return self.scalers[target_feature].inverse_transform(values.reshape(-1, 1)).ravel()
        logger.warning(f"No scaler found for {target_feature}, returning original values")
        return values
    
    def load_scaler(self, target_feature: str):
        """Load scaler for a specific feature"""
        try:
            self.scalers[target_feature] = joblib.load(f'outputs/scalers/{target_feature}_scaler.joblib')
        except FileNotFoundError:
            logger.error(f"Scaler file for {target_feature} not found")
            raise

class ModelTrainer:
    def __init__(self, top_n_models: int = 2):
        self.models = {}
        self.top_n_models = top_n_models
        self.model_metrics = {}
        self.feature_names = {}
        
    def get_models(self) -> Dict:
        """Get dictionary of models to train with default parameters except random_state"""
        return {
            'xgb': XGBRegressor(random_state=42),
            'rf': RandomForestRegressor(random_state=42),
            'lr': LinearRegression()
        }
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series, feature_name: str):
        """Train multiple models for a feature and select the best ones"""
        logger.info(f"Training models for {feature_name}")
        
        self.feature_names[feature_name] = X_train.columns.tolist()
        X_train_array = X_train.to_numpy()
        y_train_array = y_train.to_numpy()
        
        self.models[feature_name] = {}
        self.model_metrics[feature_name] = {}
        
        for model_name, model in self.get_models().items():
            logger.info(f"Training {model_name} for {feature_name}")
            model.fit(X_train_array, y_train_array)
            self.models[feature_name][model_name] = model
            
            if model_name in ['xgb', 'rf']:
                importance = dict(zip(self.feature_names[feature_name], model.feature_importances_))
                importance = dict(sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10])
                
                plt.figure(figsize=(12, 6))
                plt.bar(importance.keys(), importance.values())
                plt.title(f'{feature_name} - Top 10 Feature Importance ({model_name.upper()})')
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(f'outputs/feature_importance/{feature_name}_{model_name}_importance.png')
                plt.close()
        
        logger.info(f"Completed training all models for {feature_name}")
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series, feature_name: str, feature_eng: FeatureEngineering) -> Dict:
        """Evaluate all models and select the best ones"""
        results = {}
        
        if feature_name in self.models:
            X_test_array = X_test.to_numpy()
            y_test_array = y_test.to_numpy()
            
            for model_name, model in self.models[feature_name].items():
                pred = model.predict(X_test_array)
                pred_original = feature_eng.inverse_transform_target(feature_name, pred)
                y_test_original = feature_eng.inverse_transform_target(feature_name, y_test_array)
                
                metrics = {
                    'r2_score': r2_score(y_test_original, pred_original),
                    'mse': mean_squared_error(y_test_original, pred_original),
                    'mae': mean_absolute_error(y_test_original, pred_original)
                }
                results[model_name] = metrics
                self.model_metrics[feature_name][model_name] = metrics
        
        sorted_models = sorted(
            self.model_metrics[feature_name].items(),
            key=lambda x: x[1]['mse']
        )[:self.top_n_models]
        
        self.models[feature_name] = {
            model_name: self.models[feature_name][model_name]
            for model_name, _ in sorted_models
        }
        
        logger.info(f"Selected top {self.top_n_models} models for {feature_name}:")
        for model_name, metrics in sorted_models:
            logger.info(f"  {model_name}: MSE = {metrics['mse']:.4f}, R² = {metrics['r2_score']:.4f}")
        
        return results

    def generate_forecast(self, X: pd.DataFrame, feature_name: str, 
                         last_timestamp: pd.Timestamp, feature_eng: FeatureEngineering,
                         n_steps: int = 20160) -> pd.DataFrame:
        """Generate forecast for the next n_steps (2 weeks for minute data)"""
        logger.info(f"Generating {n_steps}-step forecast for {feature_name}")
        
        future_timestamps = pd.date_range(
            start=last_timestamp + timedelta(minutes=1),
            periods=n_steps,
            freq='min'
        )
        
        forecast_df = pd.DataFrame({'Timestamp': future_timestamps})
        
        buffer_size = max([int(col.split('_')[-1]) for col in X.columns if 'rolling_' in col] + [3])
        buffer_df = X.iloc[-buffer_size:].copy()
        
        if feature_name not in buffer_df.columns:
            original_df = pd.read_csv("./data/engine_knock_data_minute.csv")
            original_df['Timestamp'] = pd.to_datetime(original_df['Timestamp'])
            buffer_df[feature_name] = feature_eng.inverse_transform_target(
                feature_name, 
                feature_eng.scalers[feature_name].transform(
                    original_df[feature_name].iloc[-buffer_size:].values.reshape(-1, 1)
                )
            )
        
        feature_cols = self.feature_names[feature_name]
        
        for model_name, model in self.models[feature_name].items():
            logger.info(f"Generating forecast using {model_name} model")
            scaled_predictions = []
            
            for step in range(n_steps):
                try:
                    current_features = buffer_df[feature_cols].iloc[-1:].to_numpy()
                    pred = float(model.predict(current_features)[0])
                    pred = np.clip(pred, 0, 1)
                    scaled_predictions.append(pred)
                    
                    new_row = buffer_df.iloc[-1].copy()
                    new_row[feature_name] = pred
                    
                    for lag in range(1, 4):
                        lag_col = f'{feature_name}_lag_{lag}'
                        if lag == 1:
                            new_row[lag_col] = pred
                        else:
                            new_row[lag_col] = buffer_df[feature_name].iloc[-(lag-1)]
                    
                    for col in feature_cols:
                        if 'rolling_mean' in col:
                            window = int(col.split('_')[-1])
                            new_row[col] = buffer_df[feature_name].rolling(window).mean().iloc[-1]
                        elif 'rolling_std' in col:
                            window = int(col.split('_')[-1])
                            new_row[col] = buffer_df[feature_name].rolling(window).std().iloc[-1]
                    
                    buffer_df = pd.concat([buffer_df, pd.DataFrame([new_row])], ignore_index=True)
                    buffer_df = buffer_df.iloc[-buffer_size:]
                    
                    buffer_df.replace([np.inf, -np.inf], np.nan, inplace=True)
                    buffer_df.fillna(method='ffill', inplace=True)
                    buffer_df.fillna(method='bfill', inplace=True)
                    
                except Exception as e:
                    logger.error(f"Forecast error at step {step} for {model_name}: {e}")
                    fallback = scaled_predictions[-1] if scaled_predictions else buffer_df[feature_name].iloc[-1]
                    scaled_predictions.append(fallback)
            
            original_predictions = feature_eng.inverse_transform_target(
                feature_name,
                np.array(scaled_predictions)
            )
            
            forecast_df[f'{model_name}_forecast'] = original_predictions
            
            if not np.any(np.isnan(original_predictions)) and not np.any(np.isinf(original_predictions)):
                logger.info(f"Forecast statistics for {model_name} (original scale):")
                logger.info(f"  Mean: {np.mean(original_predictions):.2f}")
                logger.info(f"  Std: {np.std(original_predictions):.2f}")
                logger.info(f"  Min: {np.min(original_predictions):.2f}")
                logger.info(f"  Max: {np.max(original_predictions):.2f}")
        
        plt.figure(figsize=(15, 8))
        for model_name in self.models[feature_name]:
            if not np.any(np.isnan(forecast_df[f'{model_name}_forecast'])) and \
               not np.any(np.isinf(forecast_df[f'{model_name}_forecast'])):
                plt.plot(forecast_df['Timestamp'], 
                        forecast_df[f'{model_name}_forecast'],
                        label=f'{model_name.upper()} Forecast',
                        linestyle='--',
                        alpha=0.7)
        
        plt.title(f'{feature_name} - 2-Week Forecast (Original Scale)')
        plt.xlabel('Timestamp')
        plt.ylabel(feature_name)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'outputs/forecasts/{feature_name}_forecast.png')
        plt.close()
        
        return forecast_df

    def plot_predictions(self, y_true: pd.Series, predictions: Dict[str, np.ndarray], 
                        feature_name: str, timestamps: pd.Series, feature_eng: FeatureEngineering):
        """Plot predictions from all models in original scale"""
        plt.figure(figsize=(15, 8))
        
        y_true_array = y_true.to_numpy() if isinstance(y_true, pd.Series) else y_true
        y_true_original = feature_eng.inverse_transform_target(feature_name, y_true_array)
        
        plt.plot(timestamps, y_true_original, label='Actual', alpha=0.7)
        
        for model_name, pred in predictions.items():
            pred_original = feature_eng.inverse_transform_target(feature_name, pred)
            plt.plot(timestamps, pred_original, label=f'{model_name.upper()}', linestyle='--', alpha=0.7)
        
        plt.title(f'{feature_name} - Actual vs Predicted (Original Scale)')
        plt.xlabel('Timestamp')
        plt.ylabel(feature_name)
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f'outputs/predictions/{feature_name}_predictions.png')
        plt.close()
        
        logger.info(f"Prediction statistics for {feature_name} (original scale):")
        for model_name, pred in predictions.items():
            pred_original = feature_eng.inverse_transform_target(feature_name, pred)
            y_true_original = feature_eng.inverse_transform_target(feature_name, y_true_array)
            metrics = {
                'r2_score': r2_score(y_true_original, pred_original),
                'mse': mean_squared_error(y_true_original, pred_original),
                'mae': mean_absolute_error(y_true_original, pred_original)
            }
            logger.info(f"  {model_name}:")
            for metric_name, value in metrics.items():
                logger.info(f"    {metric_name}: {value:.4f}")

def main():
    create_output_dirs()
    
    logger.info("Loading data from CSV file")
    try:
        df = pd.read_csv("./data/engine_knock_data_minute.csv")
    except FileNotFoundError:
        logger.error("Data file not found at './data/engine_knock_data_minute.csv'")
        sys.exit(1)
    logger.info(f"Loaded data shape: {df.shape}")
    
    df = df.drop(columns=['Knock', 'IgnitionTiming'])
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    feature_eng = FeatureEngineering()
    trainer = ModelTrainer(top_n_models=2)
    
    features = ['RPM', 'CylinderPressure', 'BurnRate', 'Vibration', 'EGOVoltage', 'TempSensor']
    results = {}
    forecasts = {}
    
    for feature in features:
        logger.info(f"\nProcessing feature: {feature}")
        
        X, y, timestamps = feature_eng.prepare_features(df, feature)
        
        train_size = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
        y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
        test_timestamps = timestamps.iloc[train_size:]
        
        trainer.train_models(X_train, y_train, feature)
        feature_results = trainer.evaluate_models(X_test, y_test, feature, feature_eng)
        results[feature] = feature_results
        
        predictions = {}
        for model_name, model in trainer.models[feature].items():
            pred_scaled = model.predict(X_test.to_numpy())
            predictions[model_name] = feature_eng.inverse_transform_target(feature, pred_scaled)
        
        trainer.plot_predictions(y_test, predictions, feature, test_timestamps, feature_eng)
        
        last_timestamp = timestamps.iloc[-1]
        forecast_df = trainer.generate_forecast(X, feature, last_timestamp, feature_eng)
        forecast_df.to_csv(f'outputs/forecasts/{feature}_forecast.csv', index=False)
        
        logger.info(f"Results for {feature}:")
        for model_name, metrics in feature_results.items():
            logger.info(f"  {model_name}:")
            for metric_name, value in metrics.items():
                logger.info(f"    {metric_name}: {value:.4f}")
    
    joblib.dump(trainer.models, 'outputs/models/ml_models.joblib')
    logger.info("Saved all models")

if __name__ == "__main__":
    main()
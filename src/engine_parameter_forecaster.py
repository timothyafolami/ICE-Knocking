import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input, Attention, LayerNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib
import warnings
from typing import Dict, Tuple, List
from datetime import datetime, timedelta
import os

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure GPU automatically
def configure_gpu():
    """Configure GPU for optimal performance"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Enable memory growth to avoid allocating all GPU memory at once
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU configured: {len(gpus)} GPU(s) available")
            print(f"   GPU Names: {[gpu.name for gpu in gpus]}")
        except RuntimeError as e:
            print(f"⚠️ GPU configuration error: {e}")
    else:
        print("⚠️ No GPU found, using CPU")
    
    return len(gpus) > 0

# Configure GPU at startup
GPU_AVAILABLE = configure_gpu()

class EnginePhysicsDerivation:
    """
    Physics-based derivation of secondary engine parameters from primary parameters
    """
    
    def __init__(self):
        # Engine specifications (matching realistic generator)
        self.ENGINE_DISPLACEMENT = 1.4
        self.CYLINDERS = 4
        self.COMPRESSION_RATIO = 10.0
        self.RPM_IDLE = 800
        self.RPM_MAX = 6500
        
        # Physics constants
        self.VIBRATION_BASELINE = 0.1
        self.EGO_BASE_VOLTAGE = 0.45
        
    def derive_throttle_position(self, load: np.ndarray, noise_std: float = 2.0) -> np.ndarray:
        """
        Derive throttle position from engine load with realistic noise
        """
        throttle = load + noise_std * np.random.standard_normal(len(load))
        return np.clip(throttle, 0, 100)
    
    def derive_ignition_timing(self, rpm: np.ndarray, load: np.ndarray, 
                             noise_std: float = 0.5) -> np.ndarray:
        """
        Derive ignition timing using realistic engine maps
        """
        # Base timing curve (advances with RPM)
        base_timing = 10 + 15 * (rpm - self.RPM_IDLE) / (self.RPM_MAX - self.RPM_IDLE)
        
        # Load compensation (retard timing under high load to prevent knock)
        load_compensation = -0.1 * (load - 50)
        
        # Add realistic variation
        timing_noise = noise_std * np.random.standard_normal(len(rpm))
        
        ignition_timing = base_timing + load_compensation + timing_noise
        return np.clip(ignition_timing, 5, 35)
    
    def derive_cylinder_pressure(self, rpm: np.ndarray, load: np.ndarray, 
                               ignition_timing: np.ndarray, noise_std: float = 1.0) -> np.ndarray:
        """
        Derive cylinder pressure from combustion physics
        """
        # Base pressure from compression
        compression_pressure = 12 + 0.002 * load
        
        # Combustion pressure rise based on load and timing
        combustion_pressure = load * 0.3 * (1 + 0.1 * np.sin(ignition_timing * np.pi / 180))
        
        # RPM effect on pressure dynamics
        rpm_effect = 0.002 * (rpm - 1000)
        
        # Total cylinder pressure
        cylinder_pressure = compression_pressure + combustion_pressure + rpm_effect
        
        # Add realistic noise
        pressure_noise = noise_std * np.random.standard_normal(len(rpm))
        
        return np.maximum(cylinder_pressure + pressure_noise, 8.0)
    
    def derive_burn_rate(self, rpm: np.ndarray, load: np.ndarray, 
                        ignition_timing: np.ndarray) -> np.ndarray:
        """
        Derive burn rate using Wiebe function
        """
        # Wiebe function parameters
        a = 5.0
        n = 3.0
        
        # Crank angle simulation
        theta_0 = 10  # Start of combustion
        delta_theta = 40 + 20 * load / 100  # Burn duration increases with load
        
        # Simulate crank angle progression
        theta = theta_0 + delta_theta * np.random.random(len(rpm))
        
        # Wiebe burn rate
        x = np.clip((theta - theta_0) / delta_theta, 0, 1)
        burn_rate = 1 - np.exp(-a * x**n)
        
        return burn_rate
    
    def derive_vibration_sensor(self, rpm: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
        """
        Derive vibration sensor output with realistic frequency content
        """
        # Firing frequency for 4-stroke engine
        firing_freq = rpm / 60 * (self.CYLINDERS / 2)
        
        # Create time vector (assuming 1Hz sampling)
        t = np.arange(len(rpm))
        
        # Primary vibration from engine firing
        primary_vib = 0.05 * np.sin(2 * np.pi * firing_freq * t / 1.0)  # 1Hz sampling
        
        # Secondary harmonics
        harmonic_vib = 0.02 * np.sin(4 * np.pi * firing_freq * t / 1.0)
        
        # Random mechanical vibration
        mechanical_noise = self.VIBRATION_BASELINE * np.random.standard_normal(len(rpm))
        
        # Additional sensor noise
        sensor_noise = noise_std * np.random.standard_normal(len(rpm))
        
        vibration = primary_vib + harmonic_vib + mechanical_noise + sensor_noise
        
        return vibration
    
    def derive_ego_voltage(self, load: np.ndarray, noise_std: float = 0.01) -> np.ndarray:
        """
        Derive EGO sensor voltage with realistic lambda response
        """
        # Load affects mixture (higher load = richer mixture)
        mixture_effect = -0.05 * (load - 50) / 50
        
        # Oscillation around stoichiometric due to closed-loop control
        oscillation = 0.03 * np.sin(2 * np.pi * np.arange(len(load)) / 10)
        
        # EGO voltage calculation
        ego_voltage = self.EGO_BASE_VOLTAGE + mixture_effect + oscillation
        
        # Add sensor noise
        sensor_noise = noise_std * np.random.standard_normal(len(load))
        
        return np.clip(ego_voltage + sensor_noise, 0.1, 0.9)
    
    def derive_all_parameters(self, rpm: np.ndarray, load: np.ndarray, 
                            temperature: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Derive all secondary parameters from primary parameters
        """
        print("🔧 Deriving secondary parameters from primary forecasts...")
        
        # Derive all secondary parameters
        throttle_position = self.derive_throttle_position(load)
        ignition_timing = self.derive_ignition_timing(rpm, load)
        cylinder_pressure = self.derive_cylinder_pressure(rpm, load, ignition_timing)
        burn_rate = self.derive_burn_rate(rpm, load, ignition_timing)
        vibration = self.derive_vibration_sensor(rpm)
        ego_voltage = self.derive_ego_voltage(load)
        
        derived_params = {
            'ThrottlePosition': throttle_position,
            'IgnitionTiming': ignition_timing,
            'CylinderPressure': cylinder_pressure,
            'BurnRate': burn_rate,
            'Vibration': vibration,
            'EGOVoltage': ego_voltage
        }
        
        print(f"✅ Derived {len(derived_params)} secondary parameters")
        return derived_params

class PrimaryParameterForecaster:
    """
    LSTM-based forecasting for primary engine parameters (RPM, Load, Temperature)
    """
    
    def __init__(self, sequence_length: int = 3600, forecast_horizon: int = 86400):
        self.sequence_length = sequence_length  # 1 hour of history
        self.forecast_horizon = forecast_horizon  # 1 day forecast
        self.scalers = {}
        self.models = {}
        self.model_configs = {
            'RPM': {
                'lstm_units': [64, 32],  # Reduced for memory efficiency
                'dropout_rate': 0.2,
                'learning_rate': 0.001,
                'batch_size': 64  # Increased for efficiency
            },
            'Load': {
                'lstm_units': [48, 24],  # Reduced for memory efficiency
                'dropout_rate': 0.3,
                'learning_rate': 0.001,
                'batch_size': 64
            },
            'TempSensor': {
                'lstm_units': [32, 16],  # Reduced for memory efficiency
                'dropout_rate': 0.1,
                'learning_rate': 0.0005,
                'batch_size': 32
            }
        }
        
    def create_sequences(self, data: np.ndarray, target_data: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences for LSTM training
        """
        if target_data is None:
            target_data = data
            
        X, y = [], []
        for i in range(len(data) - self.sequence_length):
            X.append(data[i:i + self.sequence_length])
            y.append(target_data[i + self.sequence_length])
        
        return np.array(X), np.array(y)
    
    def build_lstm_model(self, input_shape: Tuple, config: Dict) -> Model:
        """
        Build LSTM model with attention mechanism
        """
        inputs = Input(shape=input_shape)
        
        # First LSTM layer
        lstm1 = LSTM(config['lstm_units'][0], 
                    return_sequences=True, 
                    dropout=config['dropout_rate'])(inputs)
        lstm1 = LayerNormalization()(lstm1)
        
        # Second LSTM layer
        lstm2 = LSTM(config['lstm_units'][1], 
                    return_sequences=False, 
                    dropout=config['dropout_rate'])(lstm1)
        lstm2 = LayerNormalization()(lstm2)
        
        # Dense layers
        dense1 = Dense(32, activation='relu')(lstm2)
        dense1 = Dropout(config['dropout_rate'])(dense1)
        
        dense2 = Dense(16, activation='relu')(dense1)
        outputs = Dense(1, activation='linear')(dense2)
        
        model = Model(inputs=inputs, outputs=outputs)
        
        optimizer = Adam(learning_rate=config['learning_rate'])
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        return model
    
    def prepare_data(self, df: pd.DataFrame, parameter: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare data for training with memory optimization
        """
        print(f"📊 Preparing data for {parameter} forecasting...")
        
        # Use ALL data for training (no sampling)
        print(f"📊 Using all {len(df):,} records for training")
        df_sample = df.copy()
        
        # Create feature matrix (parameter + time features)
        df_features = df_sample[['Timestamp']].copy()
        df_features['hour'] = df_sample['Timestamp'].dt.hour
        df_features['day_of_week'] = df_sample['Timestamp'].dt.dayofweek
        df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
        
        # Add parameter-specific features
        if parameter == 'RPM':
            df_features['RPM'] = df_sample['RPM']
            feature_cols = ['RPM', 'hour', 'day_of_week', 'is_weekend']
        elif parameter == 'Load':
            df_features['Load'] = df_sample['Load']
            df_features['RPM'] = df_sample['RPM']
            feature_cols = ['Load', 'RPM', 'hour', 'day_of_week', 'is_weekend']
        else:  # TempSensor
            df_features['TempSensor'] = df_sample['TempSensor']
            df_features['Load'] = df_sample['Load']
            df_features['RPM'] = df_sample['RPM']
            feature_cols = ['TempSensor', 'Load', 'RPM', 'hour', 'day_of_week', 'is_weekend']
        
        # Scale features with memory efficiency
        scaler = MinMaxScaler()
        feature_data = df_features[feature_cols].values
        scaled_features = scaler.fit_transform(feature_data)
        self.scalers[parameter] = scaler
        
        # Clear intermediate data to free memory
        del df_features, feature_data, df_sample
        
        # Use reasonable sequence length for pattern recognition
        sequence_length = 3600  # 1 hour sequences for good pattern capture
        print(f"📏 Using sequence length: {sequence_length} seconds (1 hour)")
            
        target_idx = feature_cols.index(parameter)
        X, y = self.create_sequences_optimized(scaled_features, scaled_features[:, target_idx], sequence_length)
        
        # Clear scaled_features to free memory
        del scaled_features
        
        # Train/test split (80/20)
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]
        
        # Convert to float32 to save memory
        X_train = X_train.astype(np.float32)
        X_test = X_test.astype(np.float32)
        y_train = y_train.astype(np.float32)
        y_test = y_test.astype(np.float32)
        
        print(f"✅ Data prepared: {X_train.shape[0]} train, {X_test.shape[0]} test sequences")
        print(f"📏 Sequence shape: {X_train.shape}, Memory usage: ~{X_train.nbytes / 1024**2:.1f} MB")
        return X_train, X_test, y_train, y_test
    
    def create_sequences_optimized(self, data: np.ndarray, target_data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create sequences with memory optimization
        """
        if len(data) < seq_length + 1000:  # Need minimum data for meaningful sequences
            raise ValueError(f"Insufficient data: {len(data)} samples, need at least {seq_length + 1000}")
            
        # Pre-allocate arrays for efficiency
        num_sequences = len(data) - seq_length
        X = np.empty((num_sequences, seq_length, data.shape[1]), dtype=np.float32)
        y = np.empty(num_sequences, dtype=np.float32)
        
        # Fill arrays in chunks to manage memory
        chunk_size = 10000
        for i in range(0, num_sequences, chunk_size):
            end_idx = min(i + chunk_size, num_sequences)
            for j in range(i, end_idx):
                X[j] = data[j:j + seq_length]
                y[j] = target_data[j + seq_length]
        
        return X, y
    
    def train_parameter_model(self, parameter: str, X_train: np.ndarray, 
                            y_train: np.ndarray, X_test: np.ndarray, y_test: np.ndarray):
        """
        Train LSTM model for a specific parameter with memory optimization
        """
        print(f"🚀 Training {parameter} forecasting model...")
        
        config = self.model_configs[parameter]
        
        # Clear GPU memory before training
        if GPU_AVAILABLE:
            tf.keras.backend.clear_session()
            
        # Use mixed precision if GPU is available (set once globally)
        if GPU_AVAILABLE:
            try:
                policy = tf.keras.mixed_precision.Policy('mixed_float16')
                tf.keras.mixed_precision.set_global_policy(policy)
                print(f"🚀 Using mixed precision training on GPU")
            except Exception as e:
                print(f"⚠️ Mixed precision setup failed: {e}, using float32")
        
        # Build model
        model = self.build_lstm_model(
            input_shape=(X_train.shape[1], X_train.shape[2]),
            config=config
        )
        
        print(f"📊 Model parameters: {model.count_params():,}")
        
        # Memory-efficient callbacks
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=15,  # Increased patience for stability
            restore_best_weights=True,
            verbose=1,
            min_delta=1e-4
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,  # Less aggressive LR reduction
            patience=8,  # Increased patience
            min_lr=1e-6,
            verbose=1
        )
        
        # Create data generators for memory efficiency
        def data_generator(X, y, batch_size, shuffle=True):
            """Generator for memory-efficient training"""
            indices = np.arange(len(X))
            while True:
                if shuffle:
                    np.random.shuffle(indices)
                for i in range(0, len(indices), batch_size):
                    batch_indices = indices[i:i + batch_size]
                    yield X[batch_indices], y[batch_indices]
        
        # Calculate steps per epoch
        steps_per_epoch = len(X_train) // config['batch_size']
        validation_steps = len(X_test) // config['batch_size']
        
        print(f"📈 Training with {steps_per_epoch} steps per epoch, {validation_steps} validation steps")
        
        # Train model with generator for memory efficiency
        try:
            history = model.fit(
                data_generator(X_train, y_train, config['batch_size'], shuffle=True),
                steps_per_epoch=steps_per_epoch,
                validation_data=data_generator(X_test, y_test, config['batch_size'], shuffle=False),
                validation_steps=validation_steps,
                epochs=5,  # Fast training with 5 epochs
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
        except Exception as e:
            print(f"⚠️ Generator training failed: {e}")
            print("🔄 Falling back to standard training with reduced batch size...")
            
            # Fallback to standard training with smaller batch size
            small_batch_size = config['batch_size'] // 2
            history = model.fit(
                X_train, y_train,
                validation_data=(X_test, y_test),
                epochs=5,
                batch_size=small_batch_size,
                callbacks=[early_stopping, reduce_lr],
                verbose=1
            )
        
        # Evaluate model on small batches to avoid memory issues
        try:
            train_loss = model.evaluate(X_train[:1000], y_train[:1000], verbose=0, batch_size=32)
            test_loss = model.evaluate(X_test[:500], y_test[:500], verbose=0, batch_size=32)
            
            print(f"✅ {parameter} model trained:")
            print(f"   Train Loss (sample): {train_loss[0]:.6f}, Train MAE: {train_loss[1]:.6f}")
            print(f"   Test Loss (sample): {test_loss[0]:.6f}, Test MAE: {test_loss[1]:.6f}")
        except Exception as e:
            print(f"⚠️ Evaluation failed: {e}, but model training completed")
        
        self.models[parameter] = model
        
        # Clear memory after training
        if GPU_AVAILABLE:
            tf.keras.backend.clear_session()
            
        return model, history
    
    def forecast_parameter(self, parameter: str, last_sequence: np.ndarray, 
                         steps: int) -> np.ndarray:
        """
        Generate multi-step forecast for a parameter
        """
        model = self.models[parameter]
        scaler = self.scalers[parameter]
        
        predictions = []
        current_sequence = last_sequence.copy()
        
        for step in range(steps):
            # Predict next step
            pred = model.predict(current_sequence.reshape(1, *current_sequence.shape), verbose=0)
            predictions.append(pred[0, 0])
            
            # Update sequence for next prediction
            # Roll the sequence and add prediction
            current_sequence = np.roll(current_sequence, -1, axis=0)
            
            # Update the target parameter in the last position
            if parameter == 'RPM':
                param_idx = 0
            elif parameter == 'Load':
                param_idx = 0  # Load is first in its feature set
            else:  # TempSensor
                param_idx = 0  # TempSensor is first in its feature set
            
            current_sequence[-1, param_idx] = pred[0, 0]
            
            # Update time features if this is a daily boundary
            if step % 3600 == 0:  # Every hour, update time features
                hour_idx = -3 if parameter == 'RPM' else -3
                current_sequence[-1, hour_idx] = (step // 3600) % 24 / 23.0  # Normalized hour
        
        return np.array(predictions)
    
    def inverse_transform_predictions(self, parameter: str, predictions: np.ndarray) -> np.ndarray:
        """
        Inverse transform predictions to original scale
        """
        scaler = self.scalers[parameter]
        
        # Create dummy array for inverse transform
        dummy = np.zeros((len(predictions), scaler.n_features_in_))
        
        # Set the parameter column
        if parameter == 'RPM':
            param_idx = 0
        elif parameter == 'Load':
            param_idx = 0
        else:  # TempSensor
            param_idx = 0
            
        dummy[:, param_idx] = predictions
        
        # Inverse transform and extract the parameter column
        original_scale = scaler.inverse_transform(dummy)[:, param_idx]
        
        return original_scale

class EngineForecastingPipeline:
    """
    Complete engine parameter forecasting pipeline
    """
    
    def __init__(self, sequence_length: int = 3600, forecast_horizon: int = 86400):
        self.primary_forecaster = PrimaryParameterForecaster(sequence_length, forecast_horizon)
        self.physics_engine = EnginePhysicsDerivation()
        self.sequence_length = sequence_length
        self.forecast_horizon = forecast_horizon
        
    def train_all_models(self, df: pd.DataFrame):
        """
        Train all primary parameter forecasting models
        """
        print("🎯 Training Primary Parameter Forecasting Models")
        print("=" * 60)
        
        primary_params = ['RPM', 'Load', 'TempSensor']
        
        for param in primary_params:
            print(f"\n📈 Training {param} Model:")
            print("-" * 30)
            
            # Prepare data
            X_train, X_test, y_train, y_test = self.primary_forecaster.prepare_data(df, param)
            
            # Train model
            model, history = self.primary_forecaster.train_parameter_model(
                param, X_train, y_train, X_test, y_test
            )
            
            # Save model in Keras 3 compatible formats
            model_keras_path = f'outputs/models/{param}_forecaster.keras'
            model_h5_path = f'outputs/models/{param}_forecaster.h5'
            
            # Save in Keras 3 native format
            model.save(model_keras_path)
            print(f"✅ {param} model saved to {model_keras_path}")
            
            # Also save as H5 for backward compatibility
            model.save(model_h5_path)
            
            # Save scaler
            scaler_path = f'outputs/scalers/{param}_forecaster_scaler.joblib'
            joblib.dump(self.primary_forecaster.scalers[param], scaler_path)
            print(f"✅ {param} scaler saved to {scaler_path}")
            
            # Save training history
            history_path = f'outputs/models/{param}_training_history.joblib'
            joblib.dump(history.history, history_path)
            print(f"✅ {param} training history saved to {history_path}")
            
        print("\n✅ All primary parameter models trained successfully!")
    
    def load_trained_models(self):
        """
        Load previously trained models and scalers
        """
        print("📂 Loading trained models...")
        primary_params = ['RPM', 'Load', 'TempSensor']
        
        for param in primary_params:
            try:
                # Try to load Keras 3 format first, then H5 format
                model_keras_path = f'outputs/models/{param}_forecaster.keras'
                model_h5_path = f'outputs/models/{param}_forecaster.h5'
                
                if os.path.exists(model_keras_path):
                    model = tf.keras.models.load_model(model_keras_path)
                elif os.path.exists(model_h5_path):
                    model = tf.keras.models.load_model(model_h5_path)
                else:
                    raise FileNotFoundError(f"No model found for {param}")
                
                self.primary_forecaster.models[param] = model
                
                # Load scaler
                scaler_path = f'outputs/scalers/{param}_forecaster_scaler.joblib'
                scaler = joblib.load(scaler_path)
                self.primary_forecaster.scalers[param] = scaler
                
                print(f"✅ {param} model and scaler loaded successfully")
                
            except Exception as e:
                print(f"❌ Failed to load {param} model: {e}")
                raise
        
        print("✅ All models loaded successfully!")
    
    def generate_complete_forecast(self, df: pd.DataFrame, start_time: pd.Timestamp = None) -> pd.DataFrame:
        """
        Generate complete forecast for all engine parameters
        """
        if start_time is None:
            start_time = df['Timestamp'].iloc[-1] + pd.Timedelta(seconds=1)
            
        print(f"🔮 Generating {self.forecast_horizon//3600}h forecast starting from {start_time}")
        
        # Create forecast timestamps
        forecast_timestamps = pd.date_range(
            start=start_time,
            periods=self.forecast_horizon,
            freq='1S'
        )
        
        # Get last sequences for each primary parameter
        primary_params = ['RPM', 'Load', 'TempSensor']
        primary_forecasts = {}
        
        for param in primary_params:
            print(f"🎯 Forecasting {param}...")
            
            # Get feature columns for this parameter
            df_features = df.copy()
            df_features['hour'] = df_features['Timestamp'].dt.hour
            df_features['day_of_week'] = df_features['Timestamp'].dt.dayofweek
            df_features['is_weekend'] = df_features['day_of_week'].isin([5, 6]).astype(int)
            
            if param == 'RPM':
                feature_cols = ['RPM', 'hour', 'day_of_week', 'is_weekend']
            elif param == 'Load':
                feature_cols = ['Load', 'RPM', 'hour', 'day_of_week', 'is_weekend']
            else:  # TempSensor
                feature_cols = ['TempSensor', 'Load', 'RPM', 'hour', 'day_of_week', 'is_weekend']
            
            # Scale and get last sequence
            scaler = self.primary_forecaster.scalers[param]
            scaled_features = scaler.transform(df_features[feature_cols].iloc[-self.sequence_length:])
            last_sequence = scaled_features
            
            # Generate forecast
            predictions_scaled = self.primary_forecaster.forecast_parameter(
                param, last_sequence, self.forecast_horizon
            )
            
            # Inverse transform
            predictions_original = self.primary_forecaster.inverse_transform_predictions(
                param, predictions_scaled
            )
            
            primary_forecasts[param] = predictions_original
        
        # Derive secondary parameters using physics
        derived_params = self.physics_engine.derive_all_parameters(
            primary_forecasts['RPM'],
            primary_forecasts['Load'],
            primary_forecasts['TempSensor']
        )
        
        # Create complete forecast DataFrame
        forecast_df = pd.DataFrame({
            'Timestamp': forecast_timestamps,
            'RPM': primary_forecasts['RPM'].round(1),
            'Load': primary_forecasts['Load'].round(1),
            'TempSensor': primary_forecasts['TempSensor'].round(1),
            'ThrottlePosition': derived_params['ThrottlePosition'].round(1),
            'IgnitionTiming': derived_params['IgnitionTiming'].round(2),
            'CylinderPressure': derived_params['CylinderPressure'].round(2),
            'BurnRate': derived_params['BurnRate'].round(4),
            'Vibration': derived_params['Vibration'].round(4),
            'EGOVoltage': derived_params['EGOVoltage'].round(3)
        })
        
        print(f"✅ Complete forecast generated with {len(forecast_df)} data points")
        print("📋 Forecast includes ALL engine parameters EXCEPT Knock variable:")
        print("   🤖 ML Predicted: RPM, Load, TempSensor")
        print("   ⚙️ Physics Derived: ThrottlePosition, IgnitionTiming, CylinderPressure, BurnRate, Vibration, EGOVoltage")
        print("   🎯 Ready for knock detection model input!")
        
        return forecast_df
    
    def plot_forecast_comparison(self, historical_df: pd.DataFrame, forecast_df: pd.DataFrame):
        """
        Plot historical vs forecasted data for visualization
        """
        print("📊 Generating forecast visualization...")
        
        # Plot last 2 hours of historical + 4 hours of forecast for visualization
        hist_sample = historical_df.iloc[-7200:]  # Last 2 hours
        forecast_sample = forecast_df.iloc[:14400]  # First 4 hours of forecast
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle('Engine Parameter Forecasting Results', fontsize=14)
        
        parameters = ['RPM', 'Load', 'TempSensor', 'CylinderPressure', 'Vibration', 'EGOVoltage']
        
        for i, param in enumerate(parameters):
            row, col = i // 2, i % 2
            ax = axes[row, col]
            
            # Plot historical data
            ax.plot(hist_sample['Timestamp'], hist_sample[param], 
                   'b-', label='Historical', alpha=0.7)
            
            # Plot forecast
            ax.plot(forecast_sample['Timestamp'], forecast_sample[param], 
                   'r--', label='Forecast', alpha=0.8)
            
            # Mark the transition point
            transition_time = forecast_df['Timestamp'].iloc[0]
            ax.axvline(transition_time, color='green', linestyle=':', alpha=0.7, label='Forecast Start')
            
            ax.set_title(f'{param} Forecast')
            ax.set_ylabel(param)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.savefig('outputs/forecasts/engine_parameter_forecast_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("✅ Forecast visualization saved")

def main(train_new_models: bool = True):
    """
    Main function to run complete forecasting pipeline
    
    Args:
        train_new_models: If True, train new models. If False, load existing models.
    """
    print("🔧 ENGINE PARAMETER FORECASTING SYSTEM")
    print("=" * 60)
    
    # Create output directories
    os.makedirs('outputs/models', exist_ok=True)
    os.makedirs('outputs/scalers', exist_ok=True)
    os.makedirs('outputs/forecasts', exist_ok=True)
    
    # Load realistic engine data
    print("📊 Loading realistic engine data...")
    df = pd.read_csv('data/realistic_engine_knock_data_week.csv')
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    
    print(f"✅ Loaded {len(df):,} records from {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    
    # Initialize forecasting pipeline
    pipeline = EngineForecastingPipeline(
        sequence_length=3600,    # 1 hour history
        forecast_horizon=86400   # 1 day forecast
    )
    
    # Check if models exist and user preference
    models_exist = all(
        os.path.exists(f'outputs/models/{param}_forecaster.keras') or 
        os.path.exists(f'outputs/models/{param}_forecaster.h5')
        for param in ['RPM', 'Load', 'TempSensor']
    )
    
    if train_new_models or not models_exist:
        if not models_exist:
            print("⚠️ No existing models found, training new models...")
        else:
            print("🔄 Training new models as requested...")
        
        # Train all models
        pipeline.train_all_models(df)
    else:
        print("📂 Loading existing trained models...")
        pipeline.load_trained_models()
    
    # Generate forecast for next day
    print("\n🔮 Generating Next Day Forecast...")
    print("-" * 40)
    
    forecast_df = pipeline.generate_complete_forecast(df)
    
    # Save forecast with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    forecast_filename = f'outputs/forecasts/next_day_engine_forecast_{timestamp}.csv'
    forecast_df.to_csv(forecast_filename, index=False)
    
    # Also save as latest for easy access
    latest_filename = 'outputs/forecasts/next_day_engine_forecast_latest.csv'
    forecast_df.to_csv(latest_filename, index=False)
    
    print(f"💾 Forecast saved to: {forecast_filename}")
    print(f"💾 Latest forecast: {latest_filename}")
    
    # Generate visualization
    pipeline.plot_forecast_comparison(df, forecast_df)
    
    # Print forecast statistics
    print("\n📈 FORECAST STATISTICS:")
    print("-" * 30)
    for param in ['RPM', 'Load', 'TempSensor', 'CylinderPressure', 'EGOVoltage']:
        mean_val = forecast_df[param].mean()
        std_val = forecast_df[param].std()
        min_val = forecast_df[param].min()
        max_val = forecast_df[param].max()
        print(f"{param:15}: Mean={mean_val:6.1f}, Std={std_val:5.1f}, Range=[{min_val:6.1f}, {max_val:6.1f}]")
    
    # Print GPU info if available
    if GPU_AVAILABLE:
        print(f"\n🚀 Training completed using GPU acceleration")
    else:
        print(f"\n💻 Training completed using CPU")
    
    print(f"\n✅ Forecasting pipeline completed successfully!")
    print(f"📁 Next step: Use forecast_df for knock detection modeling")
    
    return forecast_df

if __name__ == "__main__":
    forecast_df = main()
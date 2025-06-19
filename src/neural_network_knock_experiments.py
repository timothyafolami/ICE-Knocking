"""
Neural Network Knock Detection Experiments
==========================================

Focused experimentation on Neural Network architectures for knock detection.
Tests multiple architectures and logs performance to find the optimal configuration.

Key Features:
- Multiple NN architectures optimized for imbalanced knock detection
- Comprehensive performance logging and comparison
- Advanced techniques: focal loss, ensemble, attention mechanisms
- Hyperparameter optimization for each architecture
- Best model selection based on multiple metrics

Author: Generated for knock detection optimization
Date: 2025-01-19
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           f1_score, precision_score, recall_score, average_precision_score)
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization, 
                                   Input, Concatenate, Add, Multiply,
                                   LayerNormalization, Attention)
from tensorflow.keras.optimizers import Adam, RMSprop, AdamW
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau, 
                                      ModelCheckpoint, LearningRateScheduler)
from tensorflow.keras.regularizers import l1_l2
import tensorflow.keras.backend as K
import joblib
import warnings
import json
import os
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Tuple
import gc

warnings.filterwarnings('ignore')

# Set seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Configure TensorFlow for optimal performance
def configure_tensorflow():
    """Configure TensorFlow for optimal performance"""
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✅ GPU configured: {len(gpus)} GPU(s) available")
        except RuntimeError as e:
            print(f"⚠️ GPU configuration error: {e}")
    else:
        print("💻 Using CPU (no GPU detected)")
    
    # Enable mixed precision for faster training
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("✅ Mixed precision enabled")
    except:
        print("⚠️ Mixed precision not available")

configure_tensorflow()

class FocalLoss:
    """Focal Loss for handling imbalanced datasets"""
    
    def __init__(self, alpha=0.25, gamma=2.0):
        self.alpha = alpha
        self.gamma = gamma
    
    def __call__(self, y_true, y_pred):
        # Clip predictions to prevent NaN values
        epsilon = K.epsilon()
        y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
        
        # Calculate focal loss
        alpha_t = y_true * self.alpha + (1 - y_true) * (1 - self.alpha)
        p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
        focal_loss = -alpha_t * K.pow((1 - p_t), self.gamma) * K.log(p_t)
        
        return K.mean(focal_loss)

class KnockDataProcessor:
    """Handles data loading and preprocessing for knock detection"""
    
    def __init__(self):
        self.scaler = None
        self.feature_columns = [
            'RPM', 'Load', 'TempSensor', 'ThrottlePosition', 
            'IgnitionTiming', 'CylinderPressure', 'BurnRate', 
            'Vibration', 'EGOVoltage'
        ]
        
    def load_and_prepare_data(self, data_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Load and prepare data for neural network training"""
        print("📊 Loading knock detection data...")
        
        # Load data
        df = pd.read_csv(data_path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        print(f"✅ Loaded {len(df):,} records")
        
        # Analyze knock distribution
        knock_events = (df['Knock'] > 0).sum()
        knock_rate = knock_events / len(df) * 100
        
        print(f"🎯 Knock Analysis:")
        print(f"   Total records: {len(df):,}")
        print(f"   Knock events: {knock_events:,} ({knock_rate:.3f}%)")
        print(f"   Class imbalance ratio: 1:{(len(df) - knock_events) / max(knock_events, 1):.1f}")
        
        # Prepare features and target
        X = df[self.feature_columns].copy()
        y = (df['Knock'] > 0).astype(int).values
        
        # Enhanced feature engineering
        X = self.create_enhanced_features(df, X)
        
        return df, X.values, y
    
    def create_enhanced_features(self, df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced features for better knock detection"""
        print("🔧 Creating enhanced features...")
        
        X_enhanced = X.copy()
        
        # Temporal features
        X_enhanced['hour'] = df['Timestamp'].dt.hour / 23.0  # Normalized
        X_enhanced['day_of_week'] = df['Timestamp'].dt.dayofweek / 6.0
        X_enhanced['is_weekend'] = (df['Timestamp'].dt.dayofweek >= 5).astype(float)
        X_enhanced['minute_of_day'] = (df['Timestamp'].dt.hour * 60 + df['Timestamp'].dt.minute) / 1439.0
        
        # Rolling statistics (5-minute windows)
        for col in ['RPM', 'Load', 'CylinderPressure', 'TempSensor']:
            X_enhanced[f'{col}_rolling_mean_5'] = X[col].rolling(window=5, min_periods=1).mean()
            X_enhanced[f'{col}_rolling_std_5'] = X[col].rolling(window=5, min_periods=1).std().fillna(0)
            X_enhanced[f'{col}_rolling_max_5'] = X[col].rolling(window=5, min_periods=1).max()
            X_enhanced[f'{col}_rolling_min_5'] = X[col].rolling(window=5, min_periods=1).min()
        
        # Rate of change features
        for col in ['RPM', 'Load', 'CylinderPressure']:
            X_enhanced[f'{col}_diff'] = X[col].diff().fillna(0)
            X_enhanced[f'{col}_diff_abs'] = X_enhanced[f'{col}_diff'].abs()
            X_enhanced[f'{col}_acceleration'] = X_enhanced[f'{col}_diff'].diff().fillna(0)
        
        # Physics-based interaction features (known knock risk factors)
        X_enhanced['load_rpm_interaction'] = X['Load'] * X['RPM'] / 100000  # Normalized
        X_enhanced['pressure_timing_interaction'] = X['CylinderPressure'] * X['IgnitionTiming'] / 1000
        X_enhanced['high_load_high_rpm'] = ((X['Load'] > 80) & (X['RPM'] > 3500)).astype(float)
        X_enhanced['advanced_timing'] = (X['IgnitionTiming'] > 25).astype(float)
        X_enhanced['high_pressure'] = (X['CylinderPressure'] > 40).astype(float)
        X_enhanced['high_temp'] = (X['TempSensor'] > 100).astype(float)
        
        # Engine stress indicators
        X_enhanced['engine_stress'] = (
            (X['Load'] / 100) * 0.4 + 
            (X['RPM'] / 6500) * 0.3 + 
            (X['CylinderPressure'] / 60) * 0.3
        )
        
        # Vibration patterns
        X_enhanced['vibration_intensity'] = X['Vibration'].abs()
        X_enhanced['vibration_squared'] = X['Vibration'] ** 2
        
        # Temperature-Load correlation
        X_enhanced['temp_load_ratio'] = X['TempSensor'] / (X['Load'] + 1)  # +1 to avoid division by zero
        
        print(f"✅ Enhanced features: {X_enhanced.shape[1]} total features")
        return X_enhanced
    
    def scale_features(self, X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features using robust scaling"""
        print("⚖️ Scaling features...")
        
        self.scaler = RobustScaler()
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✅ Features scaled: {X_train_scaled.shape[1]} features")
        return X_train_scaled, X_test_scaled

class NeuralNetworkArchitectures:
    """Collection of neural network architectures for knock detection"""
    
    def __init__(self, input_dim: int):
        self.input_dim = input_dim
        
    def deep_dense_network(self, dropout_rate: float = 0.3) -> Model:
        """Deep dense network with batch normalization"""
        inputs = Input(shape=(self.input_dim,), name='input_features')
        
        # First block
        x = Dense(512, activation='relu', name='dense_1')(inputs)
        x = BatchNormalization(name='bn_1')(x)
        x = Dropout(dropout_rate, name='dropout_1')(x)
        
        # Second block
        x = Dense(256, activation='relu', name='dense_2')(x)
        x = BatchNormalization(name='bn_2')(x)
        x = Dropout(dropout_rate, name='dropout_2')(x)
        
        # Third block
        x = Dense(128, activation='relu', name='dense_3')(x)
        x = BatchNormalization(name='bn_3')(x)
        x = Dropout(dropout_rate * 0.8, name='dropout_3')(x)
        
        # Fourth block
        x = Dense(64, activation='relu', name='dense_4')(x)
        x = BatchNormalization(name='bn_4')(x)
        x = Dropout(dropout_rate * 0.6, name='dropout_4')(x)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid', dtype='float32', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='DeepDenseNetwork')
        return model
    
    def residual_network(self, dropout_rate: float = 0.25) -> Model:
        """Residual network with skip connections"""
        inputs = Input(shape=(self.input_dim,), name='input_features')
        
        # Initial transformation
        x = Dense(256, activation='relu', name='initial_dense')(inputs)
        x = BatchNormalization(name='initial_bn')(x)
        
        # Residual block 1
        residual_1 = x
        x = Dense(256, activation='relu', name='res1_dense1')(x)
        x = BatchNormalization(name='res1_bn1')(x)
        x = Dropout(dropout_rate, name='res1_dropout1')(x)
        x = Dense(256, activation='relu', name='res1_dense2')(x)
        x = BatchNormalization(name='res1_bn2')(x)
        x = Add(name='res1_add')([x, residual_1])
        x = Dropout(dropout_rate, name='res1_dropout2')(x)
        
        # Residual block 2
        x = Dense(128, activation='relu', name='transition1')(x)
        x = BatchNormalization(name='transition1_bn')(x)
        residual_2 = x
        x = Dense(128, activation='relu', name='res2_dense1')(x)
        x = BatchNormalization(name='res2_bn1')(x)
        x = Dropout(dropout_rate, name='res2_dropout1')(x)
        x = Dense(128, activation='relu', name='res2_dense2')(x)
        x = BatchNormalization(name='res2_bn2')(x)
        x = Add(name='res2_add')([x, residual_2])
        x = Dropout(dropout_rate, name='res2_dropout2')(x)
        
        # Final layers
        x = Dense(64, activation='relu', name='final_dense')(x)
        x = BatchNormalization(name='final_bn')(x)
        x = Dropout(dropout_rate * 0.5, name='final_dropout')(x)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid', dtype='float32', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='ResidualNetwork')
        return model
    
    def attention_network(self, dropout_rate: float = 0.2) -> Model:
        """Network with self-attention mechanism"""
        inputs = Input(shape=(self.input_dim,), name='input_features')
        
        # Feature transformation
        x = Dense(256, activation='relu', name='transform_1')(inputs)
        x = BatchNormalization(name='transform_bn1')(x)
        x = Dropout(dropout_rate, name='transform_dropout1')(x)
        
        x = Dense(128, activation='relu', name='transform_2')(x)
        x = BatchNormalization(name='transform_bn2')(x)
        
        # Reshape for attention (treat features as sequence)
        x_reshaped = tf.expand_dims(x, axis=1)  # (batch, 1, 128)
        
        # Self-attention layer
        attention_output = Attention(name='self_attention')([x_reshaped, x_reshaped])
        attention_output = tf.squeeze(attention_output, axis=1)  # (batch, 128)
        
        # Combine original features with attention
        x = Add(name='attention_add')([x, attention_output])
        x = LayerNormalization(name='attention_ln')(x)
        x = Dropout(dropout_rate, name='attention_dropout')(x)
        
        # Final processing
        x = Dense(64, activation='relu', name='final_dense1')(x)
        x = BatchNormalization(name='final_bn1')(x)
        x = Dropout(dropout_rate, name='final_dropout1')(x)
        
        x = Dense(32, activation='relu', name='final_dense2')(x)
        x = Dropout(dropout_rate * 0.5, name='final_dropout2')(x)
        
        # Output layer
        outputs = Dense(1, activation='sigmoid', dtype='float32', name='output')(x)
        
        model = Model(inputs=inputs, outputs=outputs, name='AttentionNetwork')
        return model
    
    def wide_and_deep_network(self, dropout_rate: float = 0.3) -> Model:
        """Wide & Deep network architecture"""
        inputs = Input(shape=(self.input_dim,), name='input_features')
        
        # Wide component (linear)
        wide = Dense(1, activation='linear', name='wide_component')(inputs)
        
        # Deep component
        deep = Dense(256, activation='relu', name='deep_1')(inputs)
        deep = BatchNormalization(name='deep_bn1')(deep)
        deep = Dropout(dropout_rate, name='deep_dropout1')(deep)
        
        deep = Dense(128, activation='relu', name='deep_2')(deep)
        deep = BatchNormalization(name='deep_bn2')(deep)
        deep = Dropout(dropout_rate * 0.8, name='deep_dropout2')(deep)
        
        deep = Dense(64, activation='relu', name='deep_3')(deep)
        deep = BatchNormalization(name='deep_bn3')(deep)
        deep = Dropout(dropout_rate * 0.6, name='deep_dropout3')(deep)
        
        deep = Dense(32, activation='relu', name='deep_4')(deep)
        deep = Dropout(dropout_rate * 0.4, name='deep_dropout4')(deep)
        
        deep = Dense(1, activation='linear', name='deep_output')(deep)
        
        # Combine wide and deep
        combined = Add(name='wide_deep_combine')([wide, deep])
        outputs = tf.nn.sigmoid(combined, name='output')
        
        model = Model(inputs=inputs, outputs=outputs, name='WideAndDeepNetwork')
        return model
    
    def ensemble_network(self, dropout_rate: float = 0.25) -> Model:
        """Ensemble of multiple sub-networks"""
        inputs = Input(shape=(self.input_dim,), name='input_features')
        
        # Network 1: Focus on basic features
        net1 = Dense(128, activation='relu', name='net1_dense1')(inputs)
        net1 = BatchNormalization(name='net1_bn1')(net1)
        net1 = Dropout(dropout_rate, name='net1_dropout1')(net1)
        net1 = Dense(64, activation='relu', name='net1_dense2')(net1)
        net1 = Dense(1, activation='linear', name='net1_output')(net1)
        
        # Network 2: Focus on interaction features
        net2 = Dense(96, activation='relu', name='net2_dense1')(inputs)
        net2 = BatchNormalization(name='net2_bn1')(net2)
        net2 = Dropout(dropout_rate, name='net2_dropout1')(net2)
        net2 = Dense(48, activation='relu', name='net2_dense2')(net2)
        net2 = Dense(1, activation='linear', name='net2_output')(net2)
        
        # Network 3: Focus on temporal features
        net3 = Dense(64, activation='relu', name='net3_dense1')(inputs)
        net3 = BatchNormalization(name='net3_bn1')(net3)
        net3 = Dropout(dropout_rate, name='net3_dropout1')(net3)
        net3 = Dense(32, activation='relu', name='net3_dense2')(net3)
        net3 = Dense(1, activation='linear', name='net3_output')(net3)
        
        # Ensemble combination
        ensemble = Add(name='ensemble_add')([net1, net2, net3])
        ensemble = Dense(16, activation='relu', name='ensemble_dense')(ensemble)
        outputs = Dense(1, activation='sigmoid', dtype='float32', name='output')(ensemble)
        
        model = Model(inputs=inputs, outputs=outputs, name='EnsembleNetwork')
        return model

class ExperimentTracker:
    """Tracks and logs experiment results"""
    
    def __init__(self, log_dir: str = 'outputs/knock_experiments'):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.results = []
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def log_experiment(self, experiment_name: str, config: Dict, results: Dict):
        """Log experiment configuration and results"""
        experiment_data = {
            'experiment_id': self.experiment_id,
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'config': config,
            'results': results
        }
        
        self.results.append(experiment_data)
        
        # Save individual experiment
        exp_file = os.path.join(self.log_dir, f'{experiment_name}_{self.experiment_id}.json')
        with open(exp_file, 'w') as f:
            json.dump(experiment_data, f, indent=2, default=str)
        
        print(f"✅ Experiment logged: {experiment_name}")
    
    def save_summary(self):
        """Save summary of all experiments"""
        summary_file = os.path.join(self.log_dir, f'experiment_summary_{self.experiment_id}.json')
        with open(summary_file, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Create performance comparison
        self.create_performance_comparison()
        
        print(f"📊 Experiment summary saved: {summary_file}")
    
    def create_performance_comparison(self):
        """Create performance comparison visualization"""
        if not self.results:
            return
        
        # Extract metrics for comparison
        experiment_names = []
        roc_aucs = []
        precisions = []
        recalls = []
        f1_scores = []
        
        for result in self.results:
            experiment_names.append(result['experiment_name'])
            roc_aucs.append(result['results']['roc_auc'])
            precisions.append(result['results']['precision'])
            recalls.append(result['results']['recall'])
            f1_scores.append(result['results']['f1_score'])
        
        # Create comparison plot
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Neural Network Architecture Comparison - Knock Detection', fontsize=16, fontweight='bold')
        
        # ROC-AUC
        bars1 = ax1.bar(experiment_names, roc_aucs, color='skyblue')
        ax1.set_title('ROC-AUC Comparison')
        ax1.set_ylabel('ROC-AUC')
        ax1.tick_params(axis='x', rotation=45)
        ax1.set_ylim(0, 1)
        for i, v in enumerate(roc_aucs):
            ax1.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
        
        # Precision
        bars2 = ax2.bar(experiment_names, precisions, color='lightgreen')
        ax2.set_title('Precision Comparison')
        ax2.set_ylabel('Precision')
        ax2.tick_params(axis='x', rotation=45)
        ax2.set_ylim(0, 1)
        for i, v in enumerate(precisions):
            ax2.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
        
        # Recall
        bars3 = ax3.bar(experiment_names, recalls, color='lightcoral')
        ax3.set_title('Recall Comparison')
        ax3.set_ylabel('Recall')
        ax3.tick_params(axis='x', rotation=45)
        ax3.set_ylim(0, 1)
        for i, v in enumerate(recalls):
            ax3.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
        
        # F1-Score
        bars4 = ax4.bar(experiment_names, f1_scores, color='orange')
        ax4.set_title('F1-Score Comparison')
        ax4.set_ylabel('F1-Score')
        ax4.tick_params(axis='x', rotation=45)
        ax4.set_ylim(0, 1)
        for i, v in enumerate(f1_scores):
            ax4.text(i, v + 0.01, f'{v:.4f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        
        # Save plot
        plot_file = os.path.join(self.log_dir, f'architecture_comparison_{self.experiment_id}.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📈 Performance comparison plot saved: {plot_file}")
    
    def get_best_experiment(self) -> Dict:
        """Get the best performing experiment based on ROC-AUC"""
        if not self.results:
            return None
        
        best_experiment = max(self.results, key=lambda x: x['results']['roc_auc'])
        return best_experiment

class KnockNeuralNetworkExperiments:
    """Main class for running neural network experiments"""
    
    def __init__(self, data_path: str):
        self.data_path = data_path
        self.processor = KnockDataProcessor()
        self.tracker = ExperimentTracker()
        self.architectures = None
        
        # Load and prepare data
        self.df, self.X, self.y = self.processor.load_and_prepare_data(data_path)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42, stratify=self.y
        )
        
        # Scale features
        self.X_train_scaled, self.X_test_scaled = self.processor.scale_features(
            self.X_train, self.X_test
        )
        
        # Initialize architectures
        self.architectures = NeuralNetworkArchitectures(self.X_train_scaled.shape[1])
        
        # Calculate class weights
        self.class_weights = compute_class_weight('balanced', classes=np.unique(self.y_train), y=self.y_train)
        self.class_weight_dict = {0: self.class_weights[0], 1: self.class_weights[1]}
        
        print(f"📊 Data prepared:")
        print(f"   Training samples: {len(self.X_train):,}")
        print(f"   Testing samples: {len(self.X_test):,}")
        print(f"   Features: {self.X_train_scaled.shape[1]}")
        print(f"   Class weights: {self.class_weight_dict}")
    
    def train_and_evaluate_model(self, model: Model, experiment_name: str, 
                                config: Dict) -> Dict:
        """Train and evaluate a single model"""
        print(f"\n🧠 Training {experiment_name}")
        print("=" * 60)
        
        # Compile model
        optimizer_name = config.get('optimizer', 'adam')
        learning_rate = config.get('learning_rate', 0.001)
        loss_type = config.get('loss', 'binary_crossentropy')
        
        if optimizer_name == 'adam':
            optimizer = Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999)
        elif optimizer_name == 'adamw':
            optimizer = AdamW(learning_rate=learning_rate, weight_decay=0.01)
        elif optimizer_name == 'rmsprop':
            optimizer = RMSprop(learning_rate=learning_rate)
        
        if loss_type == 'focal':
            loss = FocalLoss(alpha=0.25, gamma=2.0)
        else:
            loss = 'binary_crossentropy'
        
        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy', 'precision', 'recall', 'AUC']
        )
        
        print(f"📊 Model parameters: {model.count_params():,}")
        
        # Callbacks
        early_stopping = EarlyStopping(
            monitor='val_AUC',
            mode='max',
            patience=config.get('patience', 15),
            restore_best_weights=True,
            verbose=1,
            min_delta=0.0001
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.7,
            patience=config.get('lr_patience', 5),
            min_lr=1e-7,
            verbose=1
        )
        
        checkpoint_path = f'outputs/knock_experiments/checkpoints/{experiment_name}_best.keras'
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        checkpoint = ModelCheckpoint(
            checkpoint_path,
            monitor='val_AUC',
            mode='max',
            save_best_only=True,
            verbose=0
        )
        
        # Train model
        history = model.fit(
            self.X_train_scaled, self.y_train,
            validation_data=(self.X_test_scaled, self.y_test),
            epochs=config.get('epochs', 100),
            batch_size=config.get('batch_size', 32),
            class_weight=self.class_weight_dict,
            callbacks=[early_stopping, reduce_lr, checkpoint],
            verbose=1,
            shuffle=True
        )
        
        # Evaluate model
        y_pred_proba = model.predict(self.X_test_scaled, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Calculate metrics
        accuracy = (self.y_test == y_pred).mean()
        precision = precision_score(self.y_test, y_pred, zero_division=0)
        recall = recall_score(self.y_test, y_pred, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(self.y_test, y_pred_proba)
        avg_precision = average_precision_score(self.y_test, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        results = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'specificity': float(specificity),
            'f1_score': float(f1),
            'roc_auc': float(roc_auc),
            'avg_precision': float(avg_precision),
            'confusion_matrix': cm.tolist(),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
            'training_epochs': len(history.history['loss']),
            'best_val_auc': float(max(history.history.get('val_AUC', [0]))),
            'model_params': int(model.count_params())
        }
        
        # Print results
        print(f"📊 {experiment_name} Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall: {recall:.4f}")
        print(f"   Specificity: {specificity:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   ROC-AUC: {roc_auc:.4f}")
        print(f"   Avg Precision: {avg_precision:.4f}")
        print(f"   Training epochs: {len(history.history['loss'])}")
        print(f"   Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        # Log experiment
        self.tracker.log_experiment(experiment_name, config, results)
        
        # Clean up memory
        del model
        gc.collect()
        K.clear_session()
        
        return results
    
    def run_all_experiments(self):
        """Run all neural network architecture experiments"""
        print("🚀 Starting Neural Network Architecture Experiments")
        print("=" * 80)
        
        experiments = [
            # Deep Dense Network experiments
            {
                'name': 'DeepDense_Baseline',
                'model_func': self.architectures.deep_dense_network,
                'config': {
                    'dropout_rate': 0.3,
                    'optimizer': 'adam',
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 100,
                    'loss': 'binary_crossentropy'
                }
            },
            {
                'name': 'DeepDense_FocalLoss',
                'model_func': self.architectures.deep_dense_network,
                'config': {
                    'dropout_rate': 0.3,
                    'optimizer': 'adam',
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 100,
                    'loss': 'focal'
                }
            },
            {
                'name': 'DeepDense_LowDropout',
                'model_func': self.architectures.deep_dense_network,
                'config': {
                    'dropout_rate': 0.15,
                    'optimizer': 'adam',
                    'learning_rate': 0.0008,
                    'batch_size': 24,
                    'epochs': 120,
                    'loss': 'binary_crossentropy'
                }
            },
            
            # Residual Network experiments
            {
                'name': 'Residual_Baseline',
                'model_func': self.architectures.residual_network,
                'config': {
                    'dropout_rate': 0.25,
                    'optimizer': 'adam',
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 100,
                    'loss': 'binary_crossentropy'
                }
            },
            {
                'name': 'Residual_AdamW',
                'model_func': self.architectures.residual_network,
                'config': {
                    'dropout_rate': 0.2,
                    'optimizer': 'adamw',
                    'learning_rate': 0.0008,
                    'batch_size': 28,
                    'epochs': 120,
                    'loss': 'binary_crossentropy'
                }
            },
            
            # Attention Network experiments
            {
                'name': 'Attention_Baseline',
                'model_func': self.architectures.attention_network,
                'config': {
                    'dropout_rate': 0.2,
                    'optimizer': 'adam',
                    'learning_rate': 0.0012,
                    'batch_size': 32,
                    'epochs': 100,
                    'loss': 'binary_crossentropy'
                }
            },
            {
                'name': 'Attention_FocalLoss',
                'model_func': self.architectures.attention_network,
                'config': {
                    'dropout_rate': 0.25,
                    'optimizer': 'adam',
                    'learning_rate': 0.001,
                    'batch_size': 28,
                    'epochs': 120,
                    'loss': 'focal'
                }
            },
            
            # Wide & Deep experiments
            {
                'name': 'WideDeep_Baseline',
                'model_func': self.architectures.wide_and_deep_network,
                'config': {
                    'dropout_rate': 0.3,
                    'optimizer': 'adam',
                    'learning_rate': 0.001,
                    'batch_size': 32,
                    'epochs': 100,
                    'loss': 'binary_crossentropy'
                }
            },
            
            # Ensemble experiments
            {
                'name': 'Ensemble_Baseline',
                'model_func': self.architectures.ensemble_network,
                'config': {
                    'dropout_rate': 0.25,
                    'optimizer': 'adam',
                    'learning_rate': 0.0015,
                    'batch_size': 32,
                    'epochs': 100,
                    'loss': 'binary_crossentropy'
                }
            },
            {
                'name': 'Ensemble_FocalLoss',
                'model_func': self.architectures.ensemble_network,
                'config': {
                    'dropout_rate': 0.2,
                    'optimizer': 'adamw',
                    'learning_rate': 0.001,
                    'batch_size': 24,
                    'epochs': 120,
                    'loss': 'focal'
                }
            }
        ]
        
        # Run experiments
        all_results = {}
        for experiment in experiments:
            try:
                model = experiment['model_func'](**{k: v for k, v in experiment['config'].items() 
                                                   if k in ['dropout_rate']})
                results = self.train_and_evaluate_model(
                    model, experiment['name'], experiment['config']
                )
                all_results[experiment['name']] = results
                
            except Exception as e:
                print(f"❌ Experiment {experiment['name']} failed: {str(e)}")
                continue
        
        # Save summary and find best model
        self.tracker.save_summary()
        best_experiment = self.tracker.get_best_experiment()
        
        if best_experiment:
            print(f"\n🏆 BEST PERFORMING MODEL: {best_experiment['experiment_name']}")
            print(f"   ROC-AUC: {best_experiment['results']['roc_auc']:.4f}")
            print(f"   Precision: {best_experiment['results']['precision']:.4f}")
            print(f"   Recall: {best_experiment['results']['recall']:.4f}")
            print(f"   F1-Score: {best_experiment['results']['f1_score']:.4f}")
        
        return all_results, best_experiment

def main():
    """Main function to run neural network experiments"""
    print("🧠 NEURAL NETWORK KNOCK DETECTION EXPERIMENTS")
    print("=" * 80)
    
    # Create output directories
    os.makedirs('outputs/knock_experiments', exist_ok=True)
    os.makedirs('outputs/knock_experiments/checkpoints', exist_ok=True)
    
    # Initialize experiments
    data_path = 'data/realistic_engine_knock_data_week_minute.csv'
    experiments = KnockNeuralNetworkExperiments(data_path)
    
    # Run all experiments
    results, best_experiment = experiments.run_all_experiments()
    
    print(f"\n✅ All experiments completed!")
    print(f"📁 Results saved to: outputs/knock_experiments/")
    
    return experiments, results, best_experiment

if __name__ == "__main__":
    experiments, results, best_experiment = main()
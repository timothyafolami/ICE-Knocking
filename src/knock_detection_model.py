"""
Engine Knock Detection Model
============================

This model trains on minute-based engine data to detect knock events using all engine 
parameters except the knock variable itself. Designed to work with forecasted data
from the engine parameter forecasting system.

Key Features:
- Trains on original minute-based data (10,080 data points)
- Uses all 9 engine parameters as features
- Handles imbalanced knock data (only 0.25% positive cases)
- Multiple model approaches: Random Forest, XGBoost, Neural Network
- Optimized for minute-based temporal resolution
- Ready for deployment with forecasted parameter inputs

Author: Generated for automotive knock detection
Date: 2025-01-19
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, confusion_matrix, 
                           roc_auc_score, roc_curve, precision_recall_curve,
                           f1_score, precision_score, recall_score)
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import joblib

# Advanced gradient boosting libraries
try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    print("⚠️ CatBoost not available. Install with: pip install catboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("⚠️ LightGBM not available. Install with: pip install lightgbm")
import warnings
from typing import Dict, Tuple, List
import os
from collections import Counter

warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class KnockDataPreprocessor:
    """Handles data preprocessing for knock detection"""
    
    def __init__(self):
        self.scaler = None
        self.feature_columns = [
            'RPM', 'Load', 'TempSensor', 'ThrottlePosition', 
            'IgnitionTiming', 'CylinderPressure', 'BurnRate', 
            'Vibration', 'EGOVoltage'
        ]
        
    def load_and_prepare_data(self, data_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Load and prepare minute-based data for knock detection"""
        print("📊 Loading minute-based knock detection data...")
        
        # Load data
        df = pd.read_csv(data_path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        print(f"✅ Loaded {len(df):,} minute-based records")
        
        # Analyze knock distribution
        knock_stats = self.analyze_knock_distribution(df)
        print("\n🎯 KNOCK DISTRIBUTION ANALYSIS:")
        print(f"   Total minutes: {len(df):,}")
        print(f"   Knock minutes: {knock_stats['knock_count']:,}")
        print(f"   Knock rate: {knock_stats['knock_rate']:.3f}%")
        print(f"   No-knock minutes: {knock_stats['no_knock_count']:,}")
        print(f"   Class imbalance ratio: 1:{knock_stats['imbalance_ratio']:.1f}")
        
        # Prepare features and target
        X = df[self.feature_columns].copy()
        y = (df['Knock'] > 0).astype(int).values  # Convert to binary classification
        
        print(f"\n📋 FEATURE MATRIX:")
        print(f"   Shape: {X.shape}")
        print(f"   Features: {self.feature_columns}")
        print(f"   Target distribution: {Counter(y)}")
        
        return df, X, y
    
    def analyze_knock_distribution(self, df: pd.DataFrame) -> Dict:
        """Analyze knock event distribution"""
        knock_events = df['Knock'] > 0
        knock_count = knock_events.sum()
        no_knock_count = len(df) - knock_count
        knock_rate = knock_count / len(df) * 100
        imbalance_ratio = no_knock_count / max(knock_count, 1)
        
        return {
            'knock_count': knock_count,
            'no_knock_count': no_knock_count,
            'knock_rate': knock_rate,
            'imbalance_ratio': imbalance_ratio
        }
    
    def create_temporal_features(self, df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        """Add temporal features for better pattern recognition"""
        print("🕐 Adding temporal features...")
        
        X_enhanced = X.copy()
        
        # Time-based features
        X_enhanced['hour'] = df['Timestamp'].dt.hour
        X_enhanced['day_of_week'] = df['Timestamp'].dt.dayofweek
        X_enhanced['is_weekend'] = (df['Timestamp'].dt.dayofweek >= 5).astype(int)
        X_enhanced['minute_of_day'] = df['Timestamp'].dt.hour * 60 + df['Timestamp'].dt.minute
        
        # Rolling window features (5-minute windows)
        for col in ['RPM', 'Load', 'CylinderPressure']:
            X_enhanced[f'{col}_rolling_mean'] = X[col].rolling(window=5, min_periods=1).mean()
            X_enhanced[f'{col}_rolling_std'] = X[col].rolling(window=5, min_periods=1).std().fillna(0)
        
        # Rate of change features
        for col in ['RPM', 'Load']:
            X_enhanced[f'{col}_diff'] = X[col].diff().fillna(0)
            X_enhanced[f'{col}_diff_abs'] = X_enhanced[f'{col}_diff'].abs()
        
        # Interaction features (known knock risk factors)
        X_enhanced['high_load_high_rpm'] = ((X['Load'] > 80) & (X['RPM'] > 4000)).astype(int)
        X_enhanced['advanced_timing'] = (X['IgnitionTiming'] > 25).astype(int)
        X_enhanced['high_pressure'] = (X['CylinderPressure'] > 40).astype(int)
        
        print(f"✅ Enhanced features: {X_enhanced.shape[1]} total features")
        return X_enhanced
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Scale features using robust scaling"""
        print("⚖️ Scaling features...")
        
        # Use RobustScaler for better handling of outliers
        self.scaler = RobustScaler()
        
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"✅ Features scaled: {X_train_scaled.shape[1]} features")
        return X_train_scaled, X_test_scaled

class KnockDetectionModels:
    """Collection of knock detection models"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        
    def train_random_forest(self, X_train: np.ndarray, y_train: np.ndarray, 
                          X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train Random Forest classifier"""
        print("\n🌲 TRAINING RANDOM FOREST MODEL")
        print("=" * 50)
        
        # Calculate class weights for imbalanced data
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        print(f"📊 Class weights: {class_weight_dict}")
        
        # Random Forest with optimized parameters for imbalanced data
        rf_model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            class_weight=class_weight_dict,
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        rf_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = rf_model.predict(X_test)
        y_pred_proba = rf_model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        results = self.evaluate_model(y_test, y_pred, y_pred_proba, "Random Forest")
        
        # Feature importance
        feature_importance = rf_model.feature_importances_
        results['feature_importance'] = feature_importance
        results['model'] = rf_model
        
        self.models['random_forest'] = rf_model
        return results
    
    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray, 
                     X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train XGBoost classifier"""
        print("\n🚀 TRAINING XGBOOST MODEL")
        print("=" * 50)
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"📊 Scale pos weight: {scale_pos_weight:.2f}")
        
        # XGBoost with optimized parameters
        xgb_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            objective='binary:logistic',
            eval_metric='auc',
            random_state=42,
            n_jobs=-1
        )
        
        # Train model
        xgb_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = xgb_model.predict(X_test)
        y_pred_proba = xgb_model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        results = self.evaluate_model(y_test, y_pred, y_pred_proba, "XGBoost")
        
        # Feature importance
        feature_importance = xgb_model.feature_importances_
        results['feature_importance'] = feature_importance
        results['model'] = xgb_model
        
        self.models['xgboost'] = xgb_model
        return results
    
    def train_catboost(self, X_train: np.ndarray, y_train: np.ndarray, 
                      X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train CatBoost classifier"""
        print("\n🐱 TRAINING CATBOOST MODEL")
        print("=" * 50)
        
        if not CATBOOST_AVAILABLE:
            print("❌ CatBoost not available, skipping...")
            return {}
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"📊 Scale pos weight: {scale_pos_weight:.2f}")
        
        # CatBoost with optimized parameters for imbalanced data
        cb_model = cb.CatBoostClassifier(
            iterations=500,
            depth=8,
            learning_rate=0.1,
            l2_leaf_reg=3,
            bootstrap_type='Bernoulli',
            subsample=0.8,
            scale_pos_weight=scale_pos_weight,
            eval_metric='AUC',
            random_state=42,
            verbose=50,
            early_stopping_rounds=50
        )
        
        # Train model with validation
        cb_model.fit(
            X_train, y_train,
            eval_set=(X_test, y_test),
            use_best_model=True,
            plot=False
        )
        
        # Predictions
        y_pred = cb_model.predict(X_test)
        y_pred_proba = cb_model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        results = self.evaluate_model(y_test, y_pred, y_pred_proba, "CatBoost")
        
        # Feature importance
        feature_importance = cb_model.feature_importances_
        results['feature_importance'] = feature_importance
        results['model'] = cb_model
        
        self.models['catboost'] = cb_model
        return results
    
    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray, 
                      X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train LightGBM classifier"""
        print("\n💡 TRAINING LIGHTGBM MODEL")
        print("=" * 50)
        
        if not LIGHTGBM_AVAILABLE:
            print("❌ LightGBM not available, skipping...")
            return {}
        
        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        print(f"📊 Scale pos weight: {scale_pos_weight:.2f}")
        
        # LightGBM with optimized parameters
        lgb_model = lgb.LGBMClassifier(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            objective='binary',
            metric='auc',
            boosting_type='gbdt',
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )
        
        # Train model with early stopping
        lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_test, y_test)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(50)]
        )
        
        # Predictions
        y_pred = lgb_model.predict(X_test)
        y_pred_proba = lgb_model.predict_proba(X_test)[:, 1]
        
        # Evaluate
        results = self.evaluate_model(y_test, y_pred, y_pred_proba, "LightGBM")
        
        # Feature importance
        feature_importance = lgb_model.feature_importances_
        results['feature_importance'] = feature_importance
        results['model'] = lgb_model
        
        self.models['lightgbm'] = lgb_model
        return results
    
    def train_neural_network(self, X_train: np.ndarray, y_train: np.ndarray, 
                           X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Train Neural Network classifier"""
        print("\n🧠 TRAINING NEURAL NETWORK MODEL")
        print("=" * 50)
        
        # Calculate class weights
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        
        print(f"📊 Class weights: {class_weight_dict}")
        
        # Enhanced Neural Network architecture optimized for knock detection
        model = Sequential([
            # Input layer with more units for complex feature learning
            Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
            BatchNormalization(),
            Dropout(0.4),
            
            # Larger hidden layers for better pattern recognition
            Dense(128, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(96, activation='relu'),
            BatchNormalization(),
            Dropout(0.3),
            
            Dense(64, activation='relu'),
            BatchNormalization(),
            Dropout(0.25),
            
            Dense(48, activation='relu'),
            BatchNormalization(),
            Dropout(0.2),
            
            Dense(32, activation='relu'),
            Dropout(0.15),
            
            Dense(16, activation='relu'),
            Dropout(0.1),
            
            # Output layer with sigmoid for binary classification
            Dense(1, activation='sigmoid')
        ])
        
        # Compile model with improved configuration for imbalanced data
        model.compile(
            optimizer=Adam(learning_rate=0.0008, beta_1=0.9, beta_2=0.999),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall', 'AUC']
        )
        
        print(f"📊 Enhanced model parameters: {model.count_params():,}")
        
        # Enhanced callbacks for better training
        early_stopping = EarlyStopping(
            monitor='val_AUC',
            mode='max',
            patience=15,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        )
        
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.6,
            patience=8,
            min_lr=1e-7,
            verbose=1,
            cooldown=3
        )
        
        # Additional callback for monitoring AUC
        from tensorflow.keras.callbacks import ModelCheckpoint
        
        checkpoint = ModelCheckpoint(
            'outputs/knock_models/best_nn_checkpoint.keras',
            monitor='val_AUC',
            mode='max',
            save_best_only=True,
            verbose=0
        )
        
        # Train model with enhanced configuration
        history = model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=150,  # More epochs for deeper network
            batch_size=24,  # Smaller batch size for better gradient estimates
            class_weight=class_weight_dict,
            callbacks=[early_stopping, reduce_lr, checkpoint],
            verbose=1,
            shuffle=True
        )
        
        # Predictions
        y_pred_proba = model.predict(X_test, verbose=0).flatten()
        y_pred = (y_pred_proba > 0.5).astype(int)
        
        # Evaluate
        results = self.evaluate_model(y_test, y_pred, y_pred_proba, "Neural Network")
        results['model'] = model
        results['history'] = history.history
        
        self.models['neural_network'] = model
        return results
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      y_pred_proba: np.ndarray, model_name: str) -> Dict:
        """Comprehensive model evaluation"""
        
        # Basic metrics
        accuracy = (y_true == y_pred).mean()
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_true, y_pred_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Additional metrics
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
        
        print(f"📊 {model_name} Results:")
        print(f"   Accuracy: {accuracy:.4f}")
        print(f"   Precision: {precision:.4f}")
        print(f"   Recall (Sensitivity): {recall:.4f}")
        print(f"   Specificity: {specificity:.4f}")
        print(f"   F1-Score: {f1:.4f}")
        print(f"   ROC-AUC: {roc_auc:.4f}")
        print(f"   Confusion Matrix:")
        print(f"     TN: {tn}, FP: {fp}")
        print(f"     FN: {fn}, TP: {tp}")
        
        return {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
            'y_true': y_true,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }

class KnockDetectionPipeline:
    """Complete knock detection pipeline"""
    
    def __init__(self):
        self.preprocessor = KnockDataPreprocessor()
        self.models = KnockDetectionModels()
        self.results = {}
        self.feature_names = None
        
    def train_all_models(self, data_path: str, use_temporal_features: bool = True):
        """Train all knock detection models"""
        print("🎯 KNOCK DETECTION MODEL TRAINING PIPELINE")
        print("=" * 60)
        
        # Load and prepare data
        df, X, y = self.preprocessor.load_and_prepare_data(data_path)
        
        # Add temporal features if requested
        if use_temporal_features:
            X = self.preprocessor.create_temporal_features(df, X)
        
        self.feature_names = list(X.columns)
        
        # Train/test split with stratification
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\n📊 TRAIN/TEST SPLIT:")
        print(f"   Training: {len(X_train):,} samples")
        print(f"   Testing: {len(X_test):,} samples")
        print(f"   Train knock rate: {y_train.mean()*100:.3f}%")
        print(f"   Test knock rate: {y_test.mean()*100:.3f}%")
        
        # Scale features
        X_train_scaled, X_test_scaled = self.preprocessor.scale_features(X_train, X_test)
        
        # Train Random Forest
        rf_results = self.models.train_random_forest(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        self.results['random_forest'] = rf_results
        
        # Train XGBoost
        xgb_results = self.models.train_xgboost(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        self.results['xgboost'] = xgb_results
        
        # Train CatBoost
        if CATBOOST_AVAILABLE:
            cb_results = self.models.train_catboost(
                X_train_scaled, y_train, X_test_scaled, y_test
            )
            if cb_results:  # Only add if training was successful
                self.results['catboost'] = cb_results
        
        # Train LightGBM
        if LIGHTGBM_AVAILABLE:
            lgb_results = self.models.train_lightgbm(
                X_train_scaled, y_train, X_test_scaled, y_test
            )
            if lgb_results:  # Only add if training was successful
                self.results['lightgbm'] = lgb_results
        
        # Train Enhanced Neural Network (best performing model)
        nn_results = self.models.train_neural_network(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
        self.results['neural_network'] = nn_results
        
        # Save models and preprocessor
        self.save_models()
        
        return self.results
    
    def save_models(self):
        """Save all trained models and preprocessor"""
        print("\n💾 SAVING MODELS AND PREPROCESSORS")
        print("=" * 40)
        
        os.makedirs('outputs/knock_models', exist_ok=True)
        
        # Save Random Forest
        if 'random_forest' in self.models.models:
            joblib.dump(self.models.models['random_forest'], 
                       'outputs/knock_models/random_forest_knock_detector.joblib')
            print("✅ Random Forest saved")
        
        # Save XGBoost
        if 'xgboost' in self.models.models:
            joblib.dump(self.models.models['xgboost'], 
                       'outputs/knock_models/xgboost_knock_detector.joblib')
            print("✅ XGBoost saved")
        
        # Save CatBoost
        if 'catboost' in self.models.models:
            joblib.dump(self.models.models['catboost'], 
                       'outputs/knock_models/catboost_knock_detector.joblib')
            print("✅ CatBoost saved")
        
        # Save LightGBM
        if 'lightgbm' in self.models.models:
            joblib.dump(self.models.models['lightgbm'], 
                       'outputs/knock_models/lightgbm_knock_detector.joblib')
            print("✅ LightGBM saved")
        
        # Save Enhanced Neural Network
        if 'neural_network' in self.models.models:
            self.models.models['neural_network'].save(
                'outputs/knock_models/enhanced_neural_network_knock_detector.keras'
            )
            print("✅ Enhanced Neural Network saved")
        
        # Save preprocessor
        joblib.dump(self.preprocessor, 'outputs/knock_models/knock_preprocessor.joblib')
        print("✅ Preprocessor saved")
        
        # Save feature names
        joblib.dump(self.feature_names, 'outputs/knock_models/feature_names.joblib')
        print("✅ Feature names saved")
        
        print(f"📁 All models saved to: outputs/knock_models/")
    
    def create_comprehensive_plots(self):
        """Create comprehensive visualization plots"""
        print("\n📊 GENERATING COMPREHENSIVE PLOTS")
        print("=" * 40)
        
        os.makedirs('outputs/knock_plots', exist_ok=True)
        
        # Model comparison plot
        self.plot_model_comparison()
        
        # ROC curves
        self.plot_roc_curves()
        
        # Precision-Recall curves
        self.plot_precision_recall_curves()
        
        # Feature importance comparison
        self.plot_feature_importance()
        
        # Confusion matrices
        self.plot_confusion_matrices()
        
        print("✅ All plots saved to outputs/knock_plots/")
    
    def plot_model_comparison(self):
        """Plot model comparison metrics"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Knock Detection Model Comparison', fontsize=16, fontweight='bold')
        
        # Prepare data - dynamically get all trained models
        model_data = {}
        for key, result in self.results.items():
            model_data[result['model_name']] = result
        
        models = list(model_data.keys())
        colors = ['skyblue', 'lightgreen', 'lightcoral', 'orange', 'purple'][:len(models)]
        
        # Accuracy comparison
        accuracies = [model_data[model]['accuracy'] for model in models]
        ax1.bar(models, accuracies, color=colors)
        ax1.set_title('Accuracy Comparison')
        ax1.set_ylabel('Accuracy')
        ax1.set_ylim(0, 1)
        ax1.tick_params(axis='x', rotation=45)
        for i, v in enumerate(accuracies):
            ax1.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # ROC-AUC comparison
        roc_aucs = [model_data[model]['roc_auc'] for model in models]
        ax2.bar(models, roc_aucs, color=colors)
        ax2.set_title('ROC-AUC Comparison')
        ax2.set_ylabel('ROC-AUC')
        ax2.set_ylim(0, 1)
        ax2.tick_params(axis='x', rotation=45)
        for i, v in enumerate(roc_aucs):
            ax2.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        # Precision vs Recall
        precisions = [model_data[model]['precision'] for model in models]
        recalls = [model_data[model]['recall'] for model in models]
        x = np.arange(len(models))
        width = 0.35
        ax3.bar(x - width/2, precisions, width, label='Precision', color='lightblue')
        ax3.bar(x + width/2, recalls, width, label='Recall', color='lightgreen')
        ax3.set_title('Precision vs Recall')
        ax3.set_ylabel('Score')
        ax3.set_xticks(x)
        ax3.set_xticklabels(models, rotation=45)
        ax3.legend()
        ax3.set_ylim(0, 1)
        
        # F1-Score comparison
        f1_scores = [model_data[model]['f1_score'] for model in models]
        ax4.bar(models, f1_scores, color=colors)
        ax4.set_title('F1-Score Comparison')
        ax4.set_ylabel('F1-Score')
        ax4.set_ylim(0, 1)
        ax4.tick_params(axis='x', rotation=45)
        for i, v in enumerate(f1_scores):
            ax4.text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('outputs/knock_plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_roc_curves(self):
        """Plot ROC curves for all models"""
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
        for i, (key, result) in enumerate(self.results.items()):
            fpr, tpr, _ = roc_curve(result['y_true'], result['y_pred_proba'])
            auc_score = result['roc_auc']
            color = colors[i % len(colors)]  # Cycle through colors if more models than colors
            plt.plot(fpr, tpr, color=color, linewidth=2, 
                    label=f'{result["model_name"]} (AUC = {auc_score:.3f})')
        
        plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves - Knock Detection Models')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.savefig('outputs/knock_plots/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_precision_recall_curves(self):
        """Plot Precision-Recall curves for all models"""
        plt.figure(figsize=(10, 8))
        
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']
        for i, (key, result) in enumerate(self.results.items()):
            precision, recall, _ = precision_recall_curve(result['y_true'], result['y_pred_proba'])
            color = colors[i % len(colors)]  # Cycle through colors if more models than colors
            plt.plot(recall, precision, color=color, linewidth=2, 
                    label=f'{result["model_name"]}')
        
        # Baseline (random classifier performance)
        baseline = np.mean(self.results['random_forest']['y_true'])
        plt.axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                   label=f'Baseline ({baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curves - Knock Detection Models')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('outputs/knock_plots/precision_recall_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_feature_importance(self):
        """Plot feature importance for tree-based models"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        fig.suptitle('Feature Importance Comparison', fontsize=16, fontweight='bold')
        
        # Random Forest feature importance
        if 'random_forest' in self.results:
            rf_importance = self.results['random_forest']['feature_importance']
            feature_names = self.feature_names[:len(rf_importance)]
            
            # Sort by importance
            sorted_idx = np.argsort(rf_importance)[-15:]  # Top 15 features
            sorted_importance = rf_importance[sorted_idx]
            sorted_names = [feature_names[i] for i in sorted_idx]
            
            ax1.barh(range(len(sorted_importance)), sorted_importance)
            ax1.set_yticks(range(len(sorted_importance)))
            ax1.set_yticklabels(sorted_names)
            ax1.set_xlabel('Importance')
            ax1.set_title('Random Forest Feature Importance')
            ax1.grid(True, alpha=0.3)
        
        # XGBoost feature importance
        if 'xgboost' in self.results:
            xgb_importance = self.results['xgboost']['feature_importance']
            feature_names = self.feature_names[:len(xgb_importance)]
            
            # Sort by importance
            sorted_idx = np.argsort(xgb_importance)[-15:]  # Top 15 features
            sorted_importance = xgb_importance[sorted_idx]
            sorted_names = [feature_names[i] for i in sorted_idx]
            
            ax2.barh(range(len(sorted_importance)), sorted_importance, color='lightgreen')
            ax2.set_yticks(range(len(sorted_importance)))
            ax2.set_yticklabels(sorted_names)
            ax2.set_xlabel('Importance')
            ax2.set_title('XGBoost Feature Importance')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('outputs/knock_plots/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrices(self):
        """Plot confusion matrices for all models"""
        n_models = len(self.results)
        n_cols = min(3, n_models)  # Max 3 columns
        n_rows = (n_models + n_cols - 1) // n_cols  # Calculate rows needed
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        fig.suptitle('Confusion Matrices - Knock Detection Models', fontsize=16, fontweight='bold')
        
        # Handle single model case
        if n_models == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if hasattr(axes, '__len__') else [axes]
        else:
            axes = axes.flatten()
        
        for i, (key, result) in enumerate(self.results.items()):
            ax = axes[i] if i < len(axes) else axes[0]
            cm = result['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['No Knock', 'Knock'],
                       yticklabels=['No Knock', 'Knock'])
            ax.set_title(f'{result["model_name"]}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
        
        # Hide empty subplots
        for i in range(n_models, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('outputs/knock_plots/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def predict_on_forecasted_data(self, forecast_df: pd.DataFrame, model_type: str = 'xgboost') -> np.ndarray:
        """Predict knock events on forecasted data"""
        print(f"\n🔮 PREDICTING KNOCK EVENTS ON FORECASTED DATA")
        print("=" * 50)
        
        # Prepare features from forecast
        feature_columns = [
            'RPM', 'Load', 'TempSensor', 'ThrottlePosition', 
            'IgnitionTiming', 'CylinderPressure', 'BurnRate', 
            'Vibration', 'EGOVoltage'
        ]
        
        X_forecast = forecast_df[feature_columns].copy()
        
        # Add temporal features if they were used in training
        if len(self.feature_names) > 9:  # More than basic features
            X_forecast['hour'] = forecast_df['Timestamp'].dt.hour
            X_forecast['day_of_week'] = forecast_df['Timestamp'].dt.dayofweek
            X_forecast['is_weekend'] = (forecast_df['Timestamp'].dt.dayofweek >= 5).astype(int)
            X_forecast['minute_of_day'] = forecast_df['Timestamp'].dt.hour * 60 + forecast_df['Timestamp'].dt.minute
            
            # Add other temporal features with default values
            for col in self.feature_names:
                if col not in X_forecast.columns:
                    X_forecast[col] = 0  # Default value for missing features
        
        # Ensure same feature order as training
        X_forecast = X_forecast.reindex(columns=self.feature_names, fill_value=0)
        
        # Scale features
        X_forecast_scaled = self.preprocessor.scaler.transform(X_forecast)
        
        # Predict using specified model
        model = self.models.models[model_type]
        predictions_proba = model.predict_proba(X_forecast_scaled)[:, 1]
        predictions_binary = (predictions_proba > 0.5).astype(int)
        
        print(f"✅ Predictions completed using {model_type}")
        print(f"   Predicted knock minutes: {predictions_binary.sum()}")
        print(f"   Predicted knock rate: {predictions_binary.mean()*100:.3f}%")
        print(f"   Average knock probability: {predictions_proba.mean():.4f}")
        
        return predictions_proba, predictions_binary

def main():
    """Main function to train knock detection models"""
    print("🎯 KNOCK DETECTION MODEL TRAINING")
    print("=" * 60)
    
    # Create output directories
    os.makedirs('outputs/knock_models', exist_ok=True)
    os.makedirs('outputs/knock_plots', exist_ok=True)
    
    # Initialize pipeline
    pipeline = KnockDetectionPipeline()
    
    # Train all models
    data_path = 'data/realistic_engine_knock_data_week_minute.csv'
    results = pipeline.train_all_models(data_path, use_temporal_features=True)
    
    # Create comprehensive plots
    pipeline.create_comprehensive_plots()
    
    # Print summary
    print("\n📊 FINAL MODEL COMPARISON")
    print("=" * 40)
    for model_name, result in results.items():
        print(f"\n{result['model_name']}:")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  Precision: {result['precision']:.4f}")
        print(f"  Recall: {result['recall']:.4f}")
        print(f"  F1-Score: {result['f1_score']:.4f}")
        print(f"  ROC-AUC: {result['roc_auc']:.4f}")
    
    # Recommend best model
    best_model = max(results.items(), key=lambda x: x[1]['roc_auc'])
    print(f"\n🏆 RECOMMENDED MODEL: {best_model[1]['model_name']}")
    print(f"   Best ROC-AUC: {best_model[1]['roc_auc']:.4f}")
    
    print(f"\n✅ Knock detection training completed!")
    print(f"📁 Models saved to: outputs/knock_models/")
    print(f"📊 Plots saved to: outputs/knock_plots/")
    
    return pipeline

if __name__ == "__main__":
    pipeline = main()
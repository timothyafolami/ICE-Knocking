"""
Knock Detection Inference Engine
================================

Production-ready inference script for detecting knock events using the best trained model.
Loads the optimal Neural Network model and predicts knocks on engine data.

Key Features:
- Load best performing model (Ensemble Network)
- Real-time knock detection on engine data
- Comprehensive prediction analysis and statistics
- Confidence scoring for each prediction
- Detailed reporting of detected knock events

Author: Generated for knock detection inference
Date: 2025-01-19
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import joblib
import warnings
import os
from datetime import datetime
from typing import Tuple, Dict, List
import json

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class KnockInferenceEngine:
    """Production inference engine for knock detection"""
    
    def __init__(self, model_path: str = None, experiment_summary_path: str = None):
        self.model = None
        self.scaler = None
        self.feature_columns = [
            'RPM', 'Load', 'TempSensor', 'ThrottlePosition', 
            'IgnitionTiming', 'CylinderPressure', 'BurnRate', 
            'Vibration', 'EGOVoltage'
        ]
        self.best_model_info = None
        
        # Auto-detect best model if paths not provided
        if model_path is None or experiment_summary_path is None:
            self.auto_detect_best_model()
        else:
            self.model_path = model_path
            self.experiment_summary_path = experiment_summary_path
    
    def auto_detect_best_model(self):
        """Automatically detect the best performing model from experiments"""
        print("🔍 Auto-detecting best model from experiments...")
        
        # Look for experiment summaries
        experiment_dir = 'outputs/knock_experiments'
        if not os.path.exists(experiment_dir):
            raise FileNotFoundError(f"Experiment directory not found: {experiment_dir}")
        
        # Find the latest experiment summary
        summary_files = [f for f in os.listdir(experiment_dir) if f.startswith('experiment_summary_')]
        if not summary_files:
            raise FileNotFoundError("No experiment summary files found")
        
        latest_summary = sorted(summary_files)[-1]
        self.experiment_summary_path = os.path.join(experiment_dir, latest_summary)
        
        # Load experiment summary and find best model
        with open(self.experiment_summary_path, 'r') as f:
            experiments = json.load(f)
        
        # Find best model by ROC-AUC
        best_experiment = max(experiments, key=lambda x: x['results']['roc_auc'])
        self.best_model_info = best_experiment
        
        # Construct model path
        experiment_id = best_experiment['experiment_id']
        model_name = best_experiment['experiment_name']
        self.model_path = f'outputs/knock_experiments/checkpoints/{model_name}_best.keras'
        
        print(f"✅ Best model identified: {model_name}")
        print(f"   ROC-AUC: {best_experiment['results']['roc_auc']:.4f}")
        print(f"   Recall: {best_experiment['results']['recall']:.4f}")
        print(f"   Model path: {self.model_path}")
    
    def load_model_and_preprocessor(self):
        """Load the trained model and preprocessing components"""
        print("📦 Loading trained model and preprocessor...")
        
        # Load the model
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
        
        self.model = tf.keras.models.load_model(self.model_path)
        print(f"✅ Model loaded: {self.model.count_params():,} parameters")
        
        # Initialize and fit scaler (we'll refit on the data)
        self.scaler = RobustScaler()
        print("✅ Preprocessor initialized")
    
    def create_enhanced_features(self, df: pd.DataFrame, X: pd.DataFrame) -> pd.DataFrame:
        """Create enhanced features matching training pipeline"""
        print("🔧 Creating enhanced features for inference...")
        
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
        
        # Physics-based interaction features
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
        X_enhanced['temp_load_ratio'] = X['TempSensor'] / (X['Load'] + 1)
        
        print(f"✅ Enhanced features created: {X_enhanced.shape[1]} features")
        return X_enhanced
    
    def load_and_prepare_data(self, data_path: str) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
        """Load and prepare data for inference"""
        print(f"📊 Loading data from: {data_path}")
        
        # Load data
        df = pd.read_csv(data_path)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'])
        
        print(f"✅ Loaded {len(df):,} records")
        
        # Prepare features
        X = df[self.feature_columns].copy()
        y_true = (df['Knock'] > 0).astype(int).values if 'Knock' in df.columns else None
        
        # Create enhanced features
        X_enhanced = self.create_enhanced_features(df, X)
        
        # Fit scaler on the data
        X_scaled = self.scaler.fit_transform(X_enhanced.values)
        
        return df, X_scaled, y_true
    
    def predict_knocks(self, X: np.ndarray, confidence_threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        """Predict knock events with confidence scores"""
        print(f"🔮 Running knock detection inference...")
        print(f"   Confidence threshold: {confidence_threshold}")
        
        # Get model predictions
        knock_probabilities = self.model.predict(X, verbose=0).flatten()
        knock_predictions = (knock_probabilities > confidence_threshold).astype(int)
        
        predicted_knocks = np.sum(knock_predictions)
        avg_confidence = np.mean(knock_probabilities)
        max_confidence = np.max(knock_probabilities)
        
        print(f"✅ Predictions completed:")
        print(f"   Total predictions: {len(knock_predictions):,}")
        print(f"   Predicted knocks: {predicted_knocks:,}")
        print(f"   Knock rate: {predicted_knocks/len(knock_predictions)*100:.3f}%")
        print(f"   Average confidence: {avg_confidence:.4f}")
        print(f"   Max confidence: {max_confidence:.4f}")
        
        return knock_predictions, knock_probabilities
    
    def analyze_predictions(self, df: pd.DataFrame, knock_predictions: np.ndarray, 
                          knock_probabilities: np.ndarray, y_true: np.ndarray = None) -> Dict:
        """Comprehensive analysis of predictions"""
        print("\n📈 ANALYZING KNOCK DETECTION RESULTS")
        print("=" * 60)
        
        analysis = {}
        
        # Basic prediction statistics
        total_samples = len(knock_predictions)
        predicted_knocks = np.sum(knock_predictions)
        knock_rate = predicted_knocks / total_samples * 100
        
        analysis['total_samples'] = total_samples
        analysis['predicted_knocks'] = predicted_knocks
        analysis['knock_rate_percent'] = knock_rate
        analysis['avg_confidence'] = float(np.mean(knock_probabilities))
        analysis['max_confidence'] = float(np.max(knock_probabilities))
        analysis['min_confidence'] = float(np.min(knock_probabilities))
        
        print(f"📊 PREDICTION SUMMARY:")
        print(f"   Total samples analyzed: {total_samples:,}")
        print(f"   Predicted knock events: {predicted_knocks:,}")
        print(f"   Predicted knock rate: {knock_rate:.3f}%")
        print(f"   Confidence range: {analysis['min_confidence']:.4f} - {analysis['max_confidence']:.4f}")
        
        # If ground truth is available, calculate accuracy metrics
        if y_true is not None:
            true_knocks = np.sum(y_true)
            true_knock_rate = true_knocks / total_samples * 100
            
            # Calculate metrics
            accuracy = (y_true == knock_predictions).mean()
            roc_auc = roc_auc_score(y_true, knock_probabilities)
            
            # Confusion matrix
            cm = confusion_matrix(y_true, knock_predictions)
            tn, fp, fn, tp = cm.ravel()
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            analysis.update({
                'true_knocks': true_knocks,
                'true_knock_rate_percent': true_knock_rate,
                'accuracy': float(accuracy),
                'roc_auc': float(roc_auc),
                'precision': float(precision),
                'recall': float(recall),
                'f1_score': float(f1),
                'specificity': float(specificity),
                'confusion_matrix': cm.tolist(),
                'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
            })
            
            print(f"\n🎯 ACCURACY METRICS (vs Ground Truth):")
            print(f"   True knock events: {true_knocks:,} ({true_knock_rate:.3f}%)")
            print(f"   Accuracy: {accuracy:.4f}")
            print(f"   ROC-AUC: {roc_auc:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall: {recall:.4f} ({tp}/{true_knocks} knocks detected)")
            print(f"   F1-Score: {f1:.4f}")
            print(f"   Specificity: {specificity:.4f}")
            print(f"   Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
        
        # Analyze knock event patterns
        knock_indices = np.where(knock_predictions == 1)[0]
        if len(knock_indices) > 0:
            knock_times = df.iloc[knock_indices]['Timestamp']
            knock_confidences = knock_probabilities[knock_indices]
            
            print(f"\n🕐 KNOCK EVENT TIMING ANALYSIS:")
            print(f"   First knock detected: {knock_times.iloc[0]}")
            print(f"   Last knock detected: {knock_times.iloc[-1]}")
            print(f"   Average time between knocks: {(knock_times.iloc[-1] - knock_times.iloc[0]) / len(knock_times)}")
            
            # High confidence detections
            high_conf_threshold = 0.8
            high_conf_knocks = np.sum(knock_confidences > high_conf_threshold)
            print(f"   High confidence knocks (>{high_conf_threshold}): {high_conf_knocks}")
            
            analysis['knock_timing'] = {
                'first_knock': str(knock_times.iloc[0]),
                'last_knock': str(knock_times.iloc[-1]),
                'high_confidence_knocks': int(high_conf_knocks)
            }
        
        return analysis
    
    def create_detailed_visualizations(self, df: pd.DataFrame, knock_predictions: np.ndarray, 
                                     knock_probabilities: np.ndarray, y_true: np.ndarray = None):
        """Create comprehensive visualizations of knock detection results"""
        print("\n📊 Generating detailed visualizations...")
        
        # Create output directory
        os.makedirs('outputs/knock_inference', exist_ok=True)
        
        # Figure 1: Time series of predictions
        fig, axes = plt.subplots(4, 1, figsize=(16, 12))
        fig.suptitle('Knock Detection Inference Results - Time Series Analysis', fontsize=16, fontweight='bold')
        
        # Sample first 1000 points for readability
        sample_size = min(1000, len(df))
        sample_idx = np.linspace(0, len(df)-1, sample_size, dtype=int)
        
        df_sample = df.iloc[sample_idx]
        pred_sample = knock_predictions[sample_idx]
        prob_sample = knock_probabilities[sample_idx]
        
        # Plot 1: Engine parameters
        axes[0].plot(df_sample['Timestamp'], df_sample['RPM'], 'b-', alpha=0.7, label='RPM')
        axes[0].set_ylabel('RPM', color='b')
        axes[0].tick_params(axis='y', labelcolor='b')
        ax0_twin = axes[0].twinx()
        ax0_twin.plot(df_sample['Timestamp'], df_sample['Load'], 'r-', alpha=0.7, label='Load (%)')
        ax0_twin.set_ylabel('Load (%)', color='r')
        ax0_twin.tick_params(axis='y', labelcolor='r')
        axes[0].set_title('Engine Parameters')
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Cylinder pressure and vibration
        axes[1].plot(df_sample['Timestamp'], df_sample['CylinderPressure'], 'g-', alpha=0.7, label='Pressure')
        axes[1].set_ylabel('Cylinder Pressure (bar)', color='g')
        axes[1].tick_params(axis='y', labelcolor='g')
        ax1_twin = axes[1].twinx()
        ax1_twin.plot(df_sample['Timestamp'], df_sample['Vibration'], 'purple', alpha=0.7, label='Vibration')
        ax1_twin.set_ylabel('Vibration (m/s²)', color='purple')
        ax1_twin.tick_params(axis='y', labelcolor='purple')
        axes[1].set_title('Pressure and Vibration')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Knock probabilities
        axes[2].plot(df_sample['Timestamp'], prob_sample, 'orange', alpha=0.8, linewidth=1)
        axes[2].axhline(y=0.5, color='red', linestyle='--', alpha=0.7, label='Threshold (0.5)')
        axes[2].fill_between(df_sample['Timestamp'], prob_sample, alpha=0.3, color='orange')
        axes[2].set_ylabel('Knock Probability')
        axes[2].set_title('Knock Detection Confidence')
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        # Plot 4: Binary predictions vs ground truth (if available)
        if y_true is not None:
            y_true_sample = y_true[sample_idx]
            axes[3].scatter(df_sample['Timestamp'], y_true_sample + 0.05, c='red', alpha=0.8, s=20, label='True Knocks', marker='^')
        
        axes[3].scatter(df_sample['Timestamp'], pred_sample - 0.05, c='blue', alpha=0.8, s=20, label='Predicted Knocks', marker='v')
        axes[3].set_ylabel('Knock Events')
        axes[3].set_title('Knock Event Detection')
        axes[3].set_ylim(-0.2, 1.2)
        axes[3].grid(True, alpha=0.3)
        axes[3].legend()
        
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.savefig('outputs/knock_inference/knock_detection_timeseries.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Distribution analysis
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Knock Detection Distribution Analysis', fontsize=16, fontweight='bold')
        
        # Confidence distribution
        ax1.hist(knock_probabilities, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        ax1.axvline(x=0.5, color='red', linestyle='--', label='Threshold')
        ax1.set_xlabel('Knock Probability')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Confidence Score Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Knock predictions by hour
        df['hour'] = df['Timestamp'].dt.hour
        df['knock_pred'] = knock_predictions
        hourly_knocks = df.groupby('hour')['knock_pred'].sum()
        ax2.bar(hourly_knocks.index, hourly_knocks.values, alpha=0.7, color='lightcoral')
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('Predicted Knocks')
        ax2.set_title('Knock Events by Hour')
        ax2.grid(True, alpha=0.3)
        
        # Engine conditions during predicted knocks
        knock_mask = knock_predictions == 1
        if np.sum(knock_mask) > 0:
            ax3.scatter(df.loc[knock_mask, 'RPM'], df.loc[knock_mask, 'Load'], 
                       c=knock_probabilities[knock_mask], cmap='Reds', alpha=0.8, s=50)
            ax3.set_xlabel('RPM')
            ax3.set_ylabel('Load (%)')
            ax3.set_title('Engine Conditions During Predicted Knocks')
            ax3.grid(True, alpha=0.3)
            cbar = plt.colorbar(ax3.collections[0], ax=ax3)
            cbar.set_label('Knock Confidence')
        
        # Confusion matrix (if ground truth available)
        if y_true is not None:
            cm = confusion_matrix(y_true, knock_predictions)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax4,
                       xticklabels=['No Knock', 'Knock'],
                       yticklabels=['No Knock', 'Knock'])
            ax4.set_title('Confusion Matrix')
            ax4.set_xlabel('Predicted')
            ax4.set_ylabel('Actual')
        else:
            ax4.text(0.5, 0.5, 'No Ground Truth\nAvailable', ha='center', va='center', 
                    transform=ax4.transAxes, fontsize=14)
            ax4.set_title('Confusion Matrix (N/A)')
        
        plt.tight_layout()
        plt.savefig('outputs/knock_inference/knock_detection_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("✅ Visualizations saved to outputs/knock_inference/")
    
    def generate_detailed_report(self, analysis: Dict, output_path: str = None):
        """Generate a detailed inference report"""
        if output_path is None:
            output_path = 'outputs/knock_inference/knock_detection_report.json'
        
        # Add model information to report
        report = {
            'inference_timestamp': datetime.now().isoformat(),
            'model_info': self.best_model_info,
            'analysis_results': analysis
        }
        
        # Save report
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        print(f"📄 Detailed report saved: {output_path}")
        
        return report
    
    def run_complete_inference(self, data_path: str, confidence_threshold: float = 0.5) -> Dict:
        """Run complete knock detection inference pipeline"""
        print("🚀 KNOCK DETECTION INFERENCE ENGINE")
        print("=" * 80)
        
        # Load model and preprocessor
        self.load_model_and_preprocessor()
        
        # Load and prepare data
        df, X_scaled, y_true = self.load_and_prepare_data(data_path)
        
        # Run predictions
        knock_predictions, knock_probabilities = self.predict_knocks(X_scaled, confidence_threshold)
        
        # Analyze results
        analysis = self.analyze_predictions(df, knock_predictions, knock_probabilities, y_true)
        
        # Create visualizations
        self.create_detailed_visualizations(df, knock_predictions, knock_probabilities, y_true)
        
        # Generate report
        report = self.generate_detailed_report(analysis)
        
        print(f"\n✅ INFERENCE COMPLETED SUCCESSFULLY!")
        print(f"📁 Results saved to: outputs/knock_inference/")
        
        return analysis

def main():
    """Main function to run knock detection inference"""
    print("🔮 KNOCK DETECTION INFERENCE ENGINE")
    print("=" * 80)
    
    # Initialize inference engine
    engine = KnockInferenceEngine()
    
    # Run inference on the FORECASTED data (next day predictions)
    forecast_path = 'outputs/forecasts/next_day_engine_forecast_minute_latest.csv'
    print(f"🔮 Testing knock detection on FORECASTED engine data:")
    print(f"   Forecast file: {forecast_path}")
    
    results = engine.run_complete_inference(forecast_path, confidence_threshold=0.5)
    
    print("\n🎯 FINAL SUMMARY - KNOCK DETECTION ON FORECASTED DATA:")
    print(f"   Model: {engine.best_model_info['experiment_name']}")
    print(f"   Forecast samples: {results['total_samples']:,}")
    print(f"   Predicted knocks in forecast: {results['predicted_knocks']:,}")
    print(f"   Forecast knock detection rate: {results['knock_rate_percent']:.3f}%")
    print(f"   Note: This is prediction on FUTURE engine conditions!")
    
    return engine, results

if __name__ == "__main__":
    engine, results = main()
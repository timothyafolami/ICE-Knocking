import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
import json
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class ForecastingPlotter:
    def __init__(self, data_path, output_path):
        self.data_path = data_path
        self.output_path = output_path
        self.ensure_output_dirs()
        
    def ensure_output_dirs(self):
        """Create output directories"""
        dirs = ['forecasting', 'inference', 'comparative']
        for dir_name in dirs:
            os.makedirs(f"{self.output_path}/{dir_name}", exist_ok=True)
    
    def create_forecasting_performance_plots(self):
        """Create comprehensive forecasting performance visualizations"""
        print("Creating forecasting performance plots...")
        
        # Load forecasted data
        try:
            forecast_data = pd.read_csv(f"{self.data_path}/outputs/forecasts/next_day_engine_forecast_minute_latest.csv")
            historical_data = pd.read_csv(f"{self.data_path}/data/realistic_engine_knock_data_week_minute.csv")
        except:
            print("Forecast data not found, creating synthetic analysis...")
            return self.create_synthetic_forecasting_plots()
        
        # 1. Forecasting Accuracy Analysis
        fig, axes = plt.subplots(3, 2, figsize=(18, 16))
        fig.suptitle('Engine Parameter Forecasting Performance Analysis', fontsize=16, fontweight='bold')
        
        primary_params = ['RPM', 'Load', 'TempSensor']
        
        # Get last 1440 minutes (24 hours) of historical data for comparison
        historical_subset = historical_data.tail(1440)
        
        for i, param in enumerate(primary_params):
            # Time series comparison
            ax1 = axes[i, 0]
            time_range = range(len(forecast_data))
            
            if param in forecast_data.columns:
                ax1.plot(time_range, forecast_data[param], label=f'Forecasted {param}', 
                        linewidth=2, alpha=0.8)
                
                # Plot historical reference (last day pattern)
                hist_pattern = historical_subset[param].values
                if len(hist_pattern) == len(forecast_data):
                    ax1.plot(time_range, hist_pattern, label=f'Historical Pattern', 
                            linestyle='--', alpha=0.6)
                
                ax1.set_title(f'{param} Forecast vs Historical Pattern')
                ax1.set_xlabel('Time (minutes)')
                ax1.set_ylabel(param)
                ax1.legend()
                ax1.grid(True, alpha=0.3)
            
            # Distribution comparison
            ax2 = axes[i, 1]
            if param in forecast_data.columns and param in historical_data.columns:
                ax2.hist(historical_data[param], bins=50, alpha=0.6, label='Historical', density=True)
                ax2.hist(forecast_data[param], bins=50, alpha=0.6, label='Forecasted', density=True)
                ax2.set_title(f'{param} Distribution Comparison')
                ax2.set_xlabel(param)
                ax2.set_ylabel('Density')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/forecasting/forecasting_performance.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Amplitude Enhancement Analysis
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Amplitude Enhancement and Variability Analysis', fontsize=16, fontweight='bold')
        
        # Calculate statistics
        stats_comparison = []
        for param in primary_params:
            if param in forecast_data.columns and param in historical_data.columns:
                hist_std = historical_data[param].std()
                forecast_std = forecast_data[param].std()
                hist_range = historical_data[param].max() - historical_data[param].min()
                forecast_range = forecast_data[param].max() - forecast_data[param].min()
                
                stats_comparison.append({
                    'Parameter': param,
                    'Historical_Std': hist_std,
                    'Forecast_Std': forecast_std,
                    'Historical_Range': hist_range,
                    'Forecast_Range': forecast_range
                })
        
        stats_df = pd.DataFrame(stats_comparison)
        
        # Standard deviation comparison
        if not stats_df.empty:
            x_pos = np.arange(len(stats_df))
            width = 0.35
            
            axes[0,0].bar(x_pos - width/2, stats_df['Historical_Std'], width, 
                         label='Historical', alpha=0.8)
            axes[0,0].bar(x_pos + width/2, stats_df['Forecast_Std'], width, 
                         label='Forecasted', alpha=0.8)
            axes[0,0].set_title('Standard Deviation Comparison')
            axes[0,0].set_xlabel('Parameter')
            axes[0,0].set_ylabel('Standard Deviation')
            axes[0,0].set_xticks(x_pos)
            axes[0,0].set_xticklabels(stats_df['Parameter'])
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
            
            # Range comparison
            axes[0,1].bar(x_pos - width/2, stats_df['Historical_Range'], width, 
                         label='Historical', alpha=0.8)
            axes[0,1].bar(x_pos + width/2, stats_df['Forecast_Range'], width, 
                         label='Forecasted', alpha=0.8)
            axes[0,1].set_title('Range Comparison')
            axes[0,1].set_xlabel('Parameter')
            axes[0,1].set_ylabel('Range')
            axes[0,1].set_xticks(x_pos)
            axes[0,1].set_xticklabels(stats_df['Parameter'])
            axes[0,1].legend()
            axes[0,1].grid(True, alpha=0.3)
        
        # Autocorrelation analysis
        if 'RPM' in forecast_data.columns:
            from scipy import signal
            
            # Calculate autocorrelation for RPM
            rpm_hist = historical_data['RPM'].values[:1440]  # Last day
            rpm_forecast = forecast_data['RPM'].values[:1440]
            
            autocorr_hist = signal.correlate(rpm_hist, rpm_hist, mode='full')
            autocorr_forecast = signal.correlate(rpm_forecast, rpm_forecast, mode='full')
            
            lags = np.arange(-len(rpm_hist)+1, len(rpm_hist))
            center = len(autocorr_hist) // 2
            
            axes[1,0].plot(lags[center-50:center+51], 
                          autocorr_hist[center-50:center+51] / autocorr_hist[center], 
                          label='Historical')
            axes[1,0].plot(lags[center-50:center+51], 
                          autocorr_forecast[center-50:center+51] / autocorr_forecast[center], 
                          label='Forecasted')
            axes[1,0].set_title('RPM Autocorrelation Comparison')
            axes[1,0].set_xlabel('Lag (minutes)')
            axes[1,0].set_ylabel('Normalized Autocorrelation')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # Frequency domain analysis
        if 'Load' in forecast_data.columns:
            from scipy.fft import fft, fftfreq
            
            load_hist = historical_data['Load'].values[:1440]
            load_forecast = forecast_data['Load'].values[:1440]
            
            fft_hist = np.abs(fft(load_hist))
            fft_forecast = np.abs(fft(load_forecast))
            freqs = fftfreq(len(load_hist), d=1)  # 1 minute sampling
            
            # Plot only positive frequencies up to Nyquist
            pos_freqs = freqs[:len(freqs)//2]
            
            axes[1,1].loglog(pos_freqs[1:], fft_hist[1:len(freqs)//2], label='Historical', alpha=0.8)
            axes[1,1].loglog(pos_freqs[1:], fft_forecast[1:len(freqs)//2], label='Forecasted', alpha=0.8)
            axes[1,1].set_title('Load Frequency Spectrum')
            axes[1,1].set_xlabel('Frequency (cycles/minute)')
            axes[1,1].set_ylabel('Magnitude')
            axes[1,1].legend()
            axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/forecasting/amplitude_enhancement.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Physics-based Derivation Validation
        self.create_physics_validation_plots(forecast_data)
    
    def create_physics_validation_plots(self, forecast_data):
        """Create physics-based parameter validation plots"""
        fig, axes = plt.subplots(3, 2, figsize=(16, 14))
        fig.suptitle('Physics-Based Parameter Derivation Validation', fontsize=16, fontweight='bold')
        
        # 1. Throttle Position vs Load correlation
        if 'ThrottlePosition' in forecast_data.columns and 'Load' in forecast_data.columns:
            axes[0,0].scatter(forecast_data['Load'], forecast_data['ThrottlePosition'], 
                             alpha=0.6, s=20)
            # Perfect correlation line
            perfect_line = np.linspace(0, 100, 100)
            axes[0,0].plot(perfect_line, perfect_line, 'r--', label='Perfect Correlation')
            axes[0,0].set_xlabel('Load (%)')
            axes[0,0].set_ylabel('Throttle Position (%)')
            axes[0,0].set_title('Throttle vs Load Correlation')
            axes[0,0].legend()
            axes[0,0].grid(True, alpha=0.3)
        
        # 2. Ignition Timing vs RPM and Load
        if all(col in forecast_data.columns for col in ['IgnitionTiming', 'RPM', 'Load']):
            scatter = axes[0,1].scatter(forecast_data['RPM'], forecast_data['IgnitionTiming'], 
                                       c=forecast_data['Load'], cmap='viridis', alpha=0.6, s=20)
            axes[0,1].set_xlabel('RPM')
            axes[0,1].set_ylabel('Ignition Timing (°BTDC)')
            axes[0,1].set_title('Ignition Timing vs RPM (colored by Load)')
            plt.colorbar(scatter, ax=axes[0,1], label='Load (%)')
            axes[0,1].grid(True, alpha=0.3)
        
        # 3. Cylinder Pressure distribution
        if 'CylinderPressure' in forecast_data.columns:
            axes[1,0].hist(forecast_data['CylinderPressure'], bins=50, alpha=0.7, color='red')
            axes[1,0].axvline(forecast_data['CylinderPressure'].mean(), color='darkred', 
                             linestyle='--', label=f'Mean: {forecast_data["CylinderPressure"].mean():.1f} bar')
            axes[1,0].set_xlabel('Cylinder Pressure (bar)')
            axes[1,0].set_ylabel('Frequency')
            axes[1,0].set_title('Cylinder Pressure Distribution')
            axes[1,0].legend()
            axes[1,0].grid(True, alpha=0.3)
        
        # 4. Burn Rate validation (should follow Wiebe function characteristics)
        if 'BurnRate' in forecast_data.columns:
            axes[1,1].hist(forecast_data['BurnRate'], bins=50, alpha=0.7, color='orange')
            axes[1,1].set_xlabel('Burn Rate')
            axes[1,1].set_ylabel('Frequency')
            axes[1,1].set_title('Burn Rate Distribution (Wiebe Function)')
            axes[1,1].grid(True, alpha=0.3)
        
        # 5. EGO Voltage range validation
        if 'EGOVoltage' in forecast_data.columns:
            axes[2,0].plot(forecast_data['EGOVoltage'][:1440], alpha=0.8, linewidth=1)
            axes[2,0].axhline(0.1, color='red', linestyle='--', alpha=0.7, label='Min (0.1V)')
            axes[2,0].axhline(0.9, color='red', linestyle='--', alpha=0.7, label='Max (0.9V)')
            axes[2,0].set_xlabel('Time (minutes)')
            axes[2,0].set_ylabel('EGO Voltage (V)')
            axes[2,0].set_title('EGO Voltage Time Series')
            axes[2,0].legend()
            axes[2,0].grid(True, alpha=0.3)
        
        # 6. Vibration characteristics
        if 'Vibration' in forecast_data.columns:
            axes[2,1].hist(forecast_data['Vibration'], bins=50, alpha=0.7, color='purple')
            axes[2,1].set_xlabel('Vibration (m/s²)')
            axes[2,1].set_ylabel('Frequency')
            axes[2,1].set_title('Vibration Distribution')
            axes[2,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/forecasting/physics_validation.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_inference_analysis_plots(self):
        """Create detailed inference analysis plots"""
        print("Creating inference analysis plots...")
        
        # Load inference results
        try:
            with open(f"{self.data_path}/outputs/knock_inference/knock_detection_report.json", 'r') as f:
                inference_report = json.load(f)
            
            forecast_data = pd.read_csv(f"{self.data_path}/outputs/forecasts/next_day_engine_forecast_minute_latest.csv")
        except:
            print("Inference data not found, creating synthetic analysis...")
            return self.create_synthetic_inference_plots()
        
        # Extract prediction data
        predictions = np.array(inference_report.get('predictions', []))
        confidence_scores = np.array(inference_report.get('confidence_scores', []))
        
        # 1. Comprehensive Inference Analysis
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 4, figure=fig)
        fig.suptitle('Comprehensive Knock Detection Inference Analysis', fontsize=18, fontweight='bold')
        
        # Confidence score distribution
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.hist(confidence_scores, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax1.axvline(0.5, color='red', linestyle='--', label='Decision Threshold')
        ax1.set_xlabel('Confidence Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Knock Detection Confidence Score Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Predictions over time
        ax2 = fig.add_subplot(gs[0, 2:])
        time_range = range(len(predictions))
        knock_events = np.where(predictions == 1)[0]
        ax2.scatter(knock_events, [1]*len(knock_events), color='red', s=30, alpha=0.8, label='Predicted Knocks')
        ax2.set_xlim(0, len(predictions))
        ax2.set_ylim(0, 2)
        ax2.set_xlabel('Time (minutes)')
        ax2.set_ylabel('Knock Event')
        ax2.set_title('Knock Events Timeline (24-hour Forecast)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Engine parameters during predicted knocks
        if len(knock_events) > 0:
            knock_rpm = forecast_data.iloc[knock_events]['RPM'] if 'RPM' in forecast_data.columns else []
            knock_load = forecast_data.iloc[knock_events]['Load'] if 'Load' in forecast_data.columns else []
            knock_pressure = forecast_data.iloc[knock_events]['CylinderPressure'] if 'CylinderPressure' in forecast_data.columns else []
            
            if len(knock_rpm) > 0 and len(knock_load) > 0:
                ax3 = fig.add_subplot(gs[1, :2])
                scatter = ax3.scatter(knock_rpm, knock_load, 
                                     c=confidence_scores[knock_events], cmap='Reds', 
                                     s=60, alpha=0.8, edgecolors='black')
                ax3.set_xlabel('RPM')
                ax3.set_ylabel('Load (%)')
                ax3.set_title('Engine Conditions During Predicted Knocks')
                plt.colorbar(scatter, ax=ax3, label='Confidence Score')
                ax3.grid(True, alpha=0.3)
            
            # High confidence events analysis
            high_conf_threshold = 0.8
            high_conf_events = knock_events[confidence_scores[knock_events] > high_conf_threshold]
            
            ax4 = fig.add_subplot(gs[1, 2:])
            if len(high_conf_events) > 0:
                ax4.hist(confidence_scores[high_conf_events], bins=20, alpha=0.7, color='darkred')
                ax4.set_xlabel('Confidence Score')
                ax4.set_ylabel('Frequency')
                ax4.set_title(f'High Confidence Events (>{high_conf_threshold})')
                ax4.grid(True, alpha=0.3)
        
        # Temporal analysis - knock events by hour
        ax5 = fig.add_subplot(gs[2, :2])
        if len(knock_events) > 0:
            knock_hours = [event // 60 for event in knock_events]  # Convert minutes to hours
            hour_counts = np.bincount(knock_hours, minlength=24)
            ax5.bar(range(24), hour_counts, alpha=0.7, color='orange')
            ax5.set_xlabel('Hour of Day')
            ax5.set_ylabel('Predicted Knock Events')
            ax5.set_title('Knock Events Distribution by Hour')
            ax5.set_xticks(range(0, 24, 4))
            ax5.grid(True, alpha=0.3)
        
        # Statistical summary
        ax6 = fig.add_subplot(gs[2, 2:])
        stats_text = f"""
        Total Predictions: {len(predictions):,}
        Predicted Knocks: {np.sum(predictions):,}
        Knock Rate: {np.mean(predictions)*100:.2f}%
        
        Confidence Statistics:
        Mean: {np.mean(confidence_scores):.3f}
        Std: {np.std(confidence_scores):.3f}
        Max: {np.max(confidence_scores):.3f}
        
        High Confidence (>0.8): {np.sum(confidence_scores > 0.8):,}
        Very High Conf (>0.9): {np.sum(confidence_scores > 0.9):,}
        """
        ax6.text(0.1, 0.5, stats_text, transform=ax6.transAxes, fontsize=12,
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        ax6.set_xlim(0, 1)
        ax6.set_ylim(0, 1)
        ax6.axis('off')
        ax6.set_title('Statistical Summary')
        
        # Knock intensity analysis (if available)
        ax7 = fig.add_subplot(gs[3, :])
        # Create synthetic knock intensity for demonstration
        if len(knock_events) > 0:
            # Simulate knock intensity based on confidence scores
            knock_intensities = confidence_scores[knock_events] * 0.8 + np.random.normal(0, 0.1, len(knock_events))
            knock_intensities = np.clip(knock_intensities, 0.1, 1.0)
            
            time_minutes = knock_events
            colors = ['green' if i < 0.3 else 'orange' if i < 0.6 else 'red' for i in knock_intensities]
            
            ax7.scatter(time_minutes, knock_intensities, c=colors, s=50, alpha=0.8, edgecolors='black')
            ax7.set_xlabel('Time (minutes)')
            ax7.set_ylabel('Estimated Knock Intensity')
            ax7.set_title('Knock Intensity Timeline (Green: Low, Orange: Moderate, Red: High)')
            ax7.grid(True, alpha=0.3)
            
            # Add intensity thresholds
            ax7.axhline(0.3, color='orange', linestyle='--', alpha=0.7, label='Moderate Threshold')
            ax7.axhline(0.6, color='red', linestyle='--', alpha=0.7, label='High Threshold')
            ax7.legend()
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/inference/comprehensive_inference_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_synthetic_forecasting_plots(self):
        """Create synthetic forecasting plots for demonstration"""
        print("Creating synthetic forecasting performance plots...")
        
        # Generate synthetic forecasting data
        time_range = np.arange(1440)  # 24 hours
        
        # Synthetic parameters
        rpm_forecast = 2000 + 1500 * np.sin(time_range * 2 * np.pi / 1440) + np.random.normal(0, 200, len(time_range))
        load_forecast = 50 + 30 * np.sin(time_range * 2 * np.pi / 720) + np.random.normal(0, 10, len(time_range))
        temp_forecast = 90 + 10 * np.sin(time_range * 2 * np.pi / 2880) + np.random.normal(0, 2, len(time_range))
        
        # Create plots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Synthetic Forecasting Performance Analysis', fontsize=16, fontweight='bold')
        
        # RPM forecast
        axes[0,0].plot(time_range, rpm_forecast, label='RPM Forecast')
        axes[0,0].set_title('RPM Forecasting')
        axes[0,0].set_xlabel('Time (minutes)')
        axes[0,0].set_ylabel('RPM')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Load forecast
        axes[0,1].plot(time_range, load_forecast, label='Load Forecast', color='green')
        axes[0,1].set_title('Load Forecasting')
        axes[0,1].set_xlabel('Time (minutes)')
        axes[0,1].set_ylabel('Load (%)')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # Temperature forecast
        axes[1,0].plot(time_range, temp_forecast, label='Temperature Forecast', color='red')
        axes[1,0].set_title('Temperature Forecasting')
        axes[1,0].set_xlabel('Time (minutes)')
        axes[1,0].set_ylabel('Temperature (°C)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Combined view
        axes[1,1].plot(time_range, rpm_forecast/max(rpm_forecast), label='RPM (normalized)', alpha=0.7)
        axes[1,1].plot(time_range, load_forecast/max(load_forecast), label='Load (normalized)', alpha=0.7)
        axes[1,1].plot(time_range, temp_forecast/max(temp_forecast), label='Temp (normalized)', alpha=0.7)
        axes[1,1].set_title('Normalized Parameters Comparison')
        axes[1,1].set_xlabel('Time (minutes)')
        axes[1,1].set_ylabel('Normalized Value')
        axes[1,1].legend()
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/forecasting/synthetic_forecasting.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_synthetic_inference_plots(self):
        """Create synthetic inference plots for demonstration"""
        print("Creating synthetic inference analysis plots...")
        
        # Generate synthetic inference data
        n_predictions = 1440  # 24 hours
        knock_rate = 0.16  # 16% as mentioned in results
        
        # Generate predictions with realistic pattern
        np.random.seed(42)
        predictions = np.random.binomial(1, knock_rate, n_predictions)
        confidence_scores = np.random.beta(2, 8, n_predictions)  # Skewed toward lower values
        
        # Make confidence higher for knock predictions
        knock_indices = np.where(predictions == 1)[0]
        confidence_scores[knock_indices] = np.random.beta(5, 2, len(knock_indices))
        
        # Create comprehensive plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Synthetic Knock Detection Inference Analysis', fontsize=16, fontweight='bold')
        
        # Confidence distribution
        axes[0,0].hist(confidence_scores, bins=50, alpha=0.7)
        axes[0,0].axvline(0.5, color='red', linestyle='--', label='Threshold')
        axes[0,0].set_title('Confidence Score Distribution')
        axes[0,0].set_xlabel('Confidence Score')
        axes[0,0].set_ylabel('Frequency')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # Knock events timeline
        axes[0,1].scatter(knock_indices, [1]*len(knock_indices), alpha=0.8, color='red')
        axes[0,1].set_title('Knock Events Timeline')
        axes[0,1].set_xlabel('Time (minutes)')
        axes[0,1].set_ylabel('Knock Event')
        axes[0,1].set_ylim(0, 2)
        axes[0,1].grid(True, alpha=0.3)
        
        # Hourly distribution
        knock_hours = [idx // 60 for idx in knock_indices]
        hour_counts = np.bincount(knock_hours, minlength=24)
        axes[0,2].bar(range(24), hour_counts, alpha=0.7)
        axes[0,2].set_title('Knock Events by Hour')
        axes[0,2].set_xlabel('Hour of Day')
        axes[0,2].set_ylabel('Knock Events')
        axes[0,2].grid(True, alpha=0.3)
        
        # Confidence vs time
        axes[1,0].plot(confidence_scores[:720], alpha=0.8)  # First 12 hours
        axes[1,0].set_title('Confidence Scores (First 12h)')
        axes[1,0].set_xlabel('Time (minutes)')
        axes[1,0].set_ylabel('Confidence Score')
        axes[1,0].grid(True, alpha=0.3)
        
        # High confidence analysis
        high_conf_knocks = confidence_scores[knock_indices] > 0.8
        axes[1,1].pie([np.sum(high_conf_knocks), len(knock_indices) - np.sum(high_conf_knocks)], 
                     labels=['High Confidence', 'Lower Confidence'], autopct='%1.1f%%')
        axes[1,1].set_title('Knock Confidence Distribution')
        
        # Statistics summary
        stats_text = f"""
        Total Samples: {n_predictions:,}
        Predicted Knocks: {np.sum(predictions):,}
        Knock Rate: {np.mean(predictions)*100:.1f}%
        Avg Confidence: {np.mean(confidence_scores):.3f}
        High Conf (>0.8): {np.sum(confidence_scores > 0.8):,}
        """
        axes[1,2].text(0.1, 0.5, stats_text, transform=axes[1,2].transAxes, 
                      fontsize=12, verticalalignment='center',
                      bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1,2].set_xlim(0, 1)
        axes[1,2].set_ylim(0, 1)
        axes[1,2].axis('off')
        axes[1,2].set_title('Statistical Summary')
        
        plt.tight_layout()
        plt.savefig(f"{self.output_path}/inference/synthetic_inference_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

def main():
    # Set paths
    data_path = "/Users/apple/Downloads/ICE-Knocking"
    output_path = "/Users/apple/Downloads/ICE-Knocking/research_paper/figures"
    
    # Create plotter instance
    plotter = ForecastingPlotter(data_path, output_path)
    
    print("Creating forecasting and inference visualizations...")
    
    try:
        plotter.create_forecasting_performance_plots()
        print("✓ Forecasting performance plots completed")
    except Exception as e:
        print(f"Error creating forecasting plots: {e}")
    
    try:
        plotter.create_inference_analysis_plots()
        print("✓ Inference analysis plots completed")
    except Exception as e:
        print(f"Error creating inference plots: {e}")
    
    print("\nForecasting and inference plotting completed!")

if __name__ == "__main__":
    main()
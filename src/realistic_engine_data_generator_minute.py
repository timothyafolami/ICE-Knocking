import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION PARAMETERS
# =============================================================================

# Time configuration
START_DATE = pd.Timestamp("2025-01-01 00:00:00")
DURATION_DAYS = 7
SAMPLE_RATE_MINUTES = 1  # 1-minute intervals
TOTAL_MINUTES = DURATION_DAYS * 24 * 60
SEED = 42

# Engine specifications (realistic 1.4L turbocharged engine)
ENGINE_DISPLACEMENT = 1.4  # Liters
MAX_POWER_KW = 110         # kW at 5500 RPM
MAX_TORQUE_NM = 200        # Nm at 1500-4000 RPM
COMPRESSION_RATIO = 10.0   # Modern turbo engine
CYLINDERS = 4

# Realistic operating ranges
RPM_IDLE = 800
RPM_MAX = 6500
RPM_REDLINE = 6000
LOAD_MIN = 0.0    # 0% throttle
LOAD_MAX = 100.0  # 100% throttle

# Knock occurrence probability (realistic: ~1% for performance/stress conditions)
BASE_KNOCK_PROBABILITY = 0.008  # 0.8% base rate per minute (more realistic for performance driving)
HIGH_LOAD_THRESHOLD = 70.0      # % load where knock risk increases (lowered threshold)
HIGH_RPM_THRESHOLD = 3500       # RPM where knock risk increases (lowered threshold)

# Sensor specifications
KNOCK_SENSOR_FREQ_RANGE = (5000, 15000)  # Hz, typical automotive knock sensor
VIBRATION_BASELINE = 0.1  # m/s² baseline vibration
TEMP_OPERATING_RANGE = (85, 105)  # °C normal operating temperature

# Random number generator
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

class RealisticEngineSimulatorMinute:
    """
    Realistic engine simulation optimized for minute-based data generation
    """
    
    def __init__(self):
        self.time_vector = None
        self.engine_state = {}
        self.sensors = {}
        
    def generate_realistic_driving_cycle_minute(self, duration_minutes: int) -> Dict[str, np.ndarray]:
        """
        Generate realistic driving patterns optimized for minute intervals
        """
        time_array = np.arange(duration_minutes)
        
        # Create mixed driving conditions optimized for minute resolution
        # 40% city driving (low RPM, frequent stops)
        # 35% highway driving (steady medium-high RPM)
        # 20% suburban driving (mixed conditions)
        # 5% performance driving (high load, potential knock conditions)
        
        # Base driving pattern - daily and weekly cycles
        daily_pattern = np.sin(2 * np.pi * time_array / (24 * 60))  # Daily cycle (1440 minutes)
        weekly_pattern = np.sin(2 * np.pi * time_array / (7 * 24 * 60))  # Weekly cycle
        
        # RPM generation with realistic patterns for minute data
        rpm_base = 1200 + 800 * (0.5 + 0.3 * daily_pattern + 0.1 * weekly_pattern)
        
        # Add driving events every ~15-30 minutes
        for i in range(0, duration_minutes, rng.integers(15, 30)):
            if rng.random() < 0.7:  # 70% chance of driving activity
                duration = rng.integers(5, 60)  # 5-60 minutes
                end_idx = min(i + duration, duration_minutes)
                
                # City driving (frequent stops and starts)
                if rng.random() < 0.4:
                    # Add stop-and-go pattern
                    for j in range(i, end_idx, 3):  # Every 3 minutes
                        stop_duration = min(2, end_idx - j)
                        rpm_base[j:j+stop_duration] = RPM_IDLE + 200 * rng.random(stop_duration)
                        if j + stop_duration < end_idx:
                            accel_duration = min(1, end_idx - j - stop_duration)
                            rpm_base[j+stop_duration:j+stop_duration+accel_duration] += 800 + 400 * rng.random(accel_duration)
                
                # Highway driving (steady speed)
                elif rng.random() < 0.6:
                    steady_rpm = 2500 + 300 * rng.random()
                    rpm_base[i:end_idx] = steady_rpm + 100 * rng.standard_normal(end_idx - i)
                
                # Performance driving (rare, high RPM)
                else:
                    performance_rpm = 3500 + 1000 * rng.random()
                    rpm_base[i:end_idx] = performance_rpm + 200 * rng.standard_normal(end_idx - i)
        
        # Apply noise and clip to realistic range
        rpm_noise = 100 * rng.standard_normal(duration_minutes)
        rpm = np.clip(rpm_base + rpm_noise, RPM_IDLE, RPM_MAX)
        
        # Engine load based on RPM and driving conditions
        # More load during acceleration, less during cruise
        load_base = 15 + 50 * (rpm - RPM_IDLE) / (RPM_MAX - RPM_IDLE)
        
        # Add load variations for realistic driving
        load_variation = 25 * rng.standard_normal(duration_minutes)
        
        # Simulate gear changes and load demands
        for i in range(0, duration_minutes, rng.integers(2, 8)):
            if rng.random() < 0.3:  # 30% chance of load change (acceleration/deceleration)
                change_duration = min(rng.integers(1, 4), duration_minutes - i)
                load_delta = 30 * (rng.random() - 0.5)  # ±15% load change
                load_base[i:i+change_duration] += load_delta
        
        load = np.clip(load_base + load_variation, LOAD_MIN, LOAD_MAX)
        
        # Throttle position correlates with load but with some independence
        throttle_position = load + 8 * rng.standard_normal(duration_minutes)
        throttle_position = np.clip(throttle_position, 0, 100)
        
        return {
            'rpm': rpm,
            'load': load,
            'throttle_position': throttle_position
        }
    
    def calculate_ignition_timing_minute(self, rpm: np.ndarray, load: np.ndarray) -> np.ndarray:
        """
        Calculate realistic ignition timing for minute-based data
        """
        # Base timing curve (degrees BTDC) - advances with RPM for efficiency
        base_timing = 10 + 15 * (rpm - RPM_IDLE) / (RPM_MAX - RPM_IDLE)
        
        # Load compensation (retard timing under high load to prevent knock)
        load_compensation = -0.15 * (load - 50)  # More aggressive retard for minute data
        
        # Add realistic ECU variation
        timing_noise = 1.0 * rng.standard_normal(len(rpm))
        
        ignition_timing = base_timing + load_compensation + timing_noise
        return np.clip(ignition_timing, 5, 35)  # Realistic timing range
    
    def calculate_cylinder_pressure_minute(self, rpm: np.ndarray, load: np.ndarray, 
                                         ignition_timing: np.ndarray) -> np.ndarray:
        """
        Calculate cylinder pressure optimized for minute-based resolution
        """
        # Base pressure from compression ratio and ambient conditions
        compression_pressure = 12 + 1 + 0.002 * load  # Slightly higher base for minute data
        
        # Combustion pressure rise based on load and timing
        combustion_pressure = load * 0.35 * (1 + 0.1 * np.sin(ignition_timing * np.pi / 180))
        
        # RPM effect on pressure dynamics (higher RPM = more dynamic pressure)
        rpm_effect = 0.003 * (rpm - 1000)
        
        # Total cylinder pressure
        cylinder_pressure = compression_pressure + combustion_pressure + rpm_effect
        
        # Add realistic pressure variation for minute data
        pressure_noise = 2.0 * rng.standard_normal(len(rpm))
        
        return np.maximum(cylinder_pressure + pressure_noise, 8.0)  # Minimum atmospheric pressure
    
    def calculate_knock_probability_minute(self, rpm: np.ndarray, load: np.ndarray, 
                                         cylinder_pressure: np.ndarray, 
                                         ignition_timing: np.ndarray) -> np.ndarray:
        """
        Calculate realistic knock probability for minute-based data
        """
        # Base probability (per minute instead of per second)
        knock_prob = np.full(len(rpm), BASE_KNOCK_PROBABILITY)
        
        # High load increases knock probability significantly
        high_load_mask = load > HIGH_LOAD_THRESHOLD
        knock_prob[high_load_mask] *= 3.0  # Moderate multiplier for realistic conditions
        
        # High RPM with high load is very dangerous
        high_rpm_mask = rpm > HIGH_RPM_THRESHOLD
        knock_prob[high_load_mask & high_rpm_mask] *= 2.0
        
        # Advanced ignition timing increases knock risk
        advanced_timing_mask = ignition_timing > 22  # Lower threshold for more realistic timing
        knock_prob[advanced_timing_mask] *= 1.8
        
        # Very high cylinder pressure increases knock risk
        high_pressure_mask = cylinder_pressure > 35  # Lower threshold
        knock_prob[high_pressure_mask] *= 1.5
        
        # Temperature effect (assume higher temp with sustained high load)
        temp_effect = 1 + 0.04 * np.maximum(0, load - 40) / 60  # More aggressive temperature effect
        knock_prob *= temp_effect
        
        # Performance driving conditions (random stress events)
        performance_events = rng.random(len(rpm)) < 0.002  # 0.2% chance of performance stress
        knock_prob[performance_events] *= 5.0  # Significantly higher risk during performance events
        
        return np.clip(knock_prob, 0, 0.15)  # Max 15% probability per minute for extreme conditions
    
    def generate_knock_events_minute(self, knock_probability: np.ndarray) -> np.ndarray:
        """
        Generate knock events for minute-based data (can be fractional for risk levels)
        """
        # For minute data, we can have fractional knock values representing risk levels
        random_values = rng.random(len(knock_probability))
        
        # Generate both binary events and risk levels
        binary_knocks = random_values < knock_probability
        
        # For minutes with knock events, add intensity based on conditions
        knock_values = binary_knocks.astype(float)
        
        # Add knock intensity for detected events (0.1 to 1.0)
        knock_indices = np.where(binary_knocks)[0]
        if len(knock_indices) > 0:
            # Higher intensity for higher probability conditions
            intensities = 0.1 + 0.9 * (knock_probability[knock_indices] / knock_probability.max())
            knock_values[knock_indices] = intensities
        
        return knock_values
    
    def calculate_burn_rate_minute(self, rpm: np.ndarray, load: np.ndarray, 
                                 ignition_timing: np.ndarray, knock_events: np.ndarray) -> np.ndarray:
        """
        Calculate burn rate using Wiebe function for minute data
        """
        # Wiebe function parameters
        a = 5.0  # Shape parameter
        n = 3.0  # Shape parameter
        
        # Crank angle simulation (averaged over the minute)
        theta_0 = 10  # Start of combustion (degrees ATDC)
        delta_theta = 40 + 25 * load / 100  # Burn duration increases with load
        
        # For minute data, simulate average burn characteristics
        theta = theta_0 + delta_theta * (0.3 + 0.4 * rng.random(len(rpm)))  # More consistent for minute data
        
        # Wiebe burn rate
        x = np.clip((theta - theta_0) / delta_theta, 0, 1)
        burn_rate = 1 - np.exp(-a * x**n)
        
        # Knock affects burn rate (incomplete combustion)
        # For minute data, knock_events can be fractional
        burn_rate *= (1 - 0.2 * knock_events)  # Up to 20% reduction with intense knock
        
        return np.clip(burn_rate, 0, 1)
    
    def calculate_vibration_sensor_minute(self, rpm: np.ndarray, knock_events: np.ndarray) -> np.ndarray:
        """
        Calculate vibration sensor output optimized for minute-based data
        """
        # For minute data, we represent average vibration characteristics
        
        # Base engine vibration (averaged firing frequency effects)
        firing_freq_avg = rpm / 60 * (CYLINDERS / 2)  # Average firing frequency
        
        # Create time vector for minute data
        t = np.arange(len(rpm))
        
        # Primary vibration from engine firing (averaged over minute)
        primary_vib = 0.05 * np.sin(2 * np.pi * firing_freq_avg * t / (60 * SAMPLE_RATE_MINUTES))
        
        # Secondary harmonics (also averaged)
        harmonic_vib = 0.02 * np.sin(4 * np.pi * firing_freq_avg * t / (60 * SAMPLE_RATE_MINUTES))
        
        # Random mechanical vibration
        mechanical_noise = VIBRATION_BASELINE * rng.standard_normal(len(rpm))
        
        # Base vibration
        vibration = primary_vib + harmonic_vib + mechanical_noise
        
        # Knock-induced vibration (proportional to knock intensity)
        knock_amplitude = 0.6  # Base knock vibration amplitude
        vibration += knock_amplitude * knock_events * (1 + 0.3 * rng.standard_normal(len(rpm)))
        
        return vibration
    
    def calculate_ego_voltage_minute(self, load: np.ndarray, knock_events: np.ndarray) -> np.ndarray:
        """
        Calculate EGO (oxygen) sensor voltage for minute-based data
        """
        # Base voltage around stoichiometric (lambda = 1)
        base_voltage = 0.45  # Volts at lambda = 1
        
        # Load affects mixture (higher load = richer mixture, typically)
        mixture_effect = -0.06 * (load - 50) / 50  # ±0.06V swing
        
        # Oscillation around stoichiometric due to closed-loop control
        # For minute data, represent average oscillation
        oscillation = 0.02 * np.sin(2 * np.pi * np.arange(len(load)) / 15)  # 15-minute cycle
        
        # Knock events cause lean condition (incomplete combustion)
        # Proportional to knock intensity
        ego_voltage = base_voltage + mixture_effect + oscillation
        ego_voltage += 0.12 * knock_events  # Lean spike proportional to knock intensity
        
        # Add sensor noise
        sensor_noise = 0.015 * rng.standard_normal(len(load))
        
        return np.clip(ego_voltage + sensor_noise, 0.1, 0.9)
    
    def calculate_temperature_sensor_minute(self, load: np.ndarray, rpm: np.ndarray, 
                                          knock_events: np.ndarray) -> np.ndarray:
        """
        Calculate engine temperature sensor with thermal dynamics for minute data
        """
        # Base operating temperature
        base_temp = 90  # °C
        
        # Load effect on temperature (sustained load increases temperature)
        load_effect = 18 * load / 100  # Up to 18°C increase at full load
        
        # RPM effect (higher RPM = better cooling but more friction)
        rpm_effect = 8 * (rpm - 2000) / 4000  # ±8°C variation
        
        # Thermal inertia (temperature changes slowly) - more relevant for minute data
        temp_signal = base_temp + load_effect + rpm_effect
        
        # Apply thermal filtering with appropriate time constant for minute data
        temperature = np.zeros_like(temp_signal)
        temperature[0] = temp_signal[0]
        tau_minutes = 5  # 5-minute time constant (more appropriate for minute data)
        alpha = 1 / (tau_minutes + 1)
        
        for i in range(1, len(temp_signal)):
            temperature[i] = alpha * temp_signal[i] + (1 - alpha) * temperature[i-1]
        
        # Knock events increase temperature (proportional to intensity)
        temperature += 10 * knock_events * (1 + 0.2 * rng.standard_normal(len(knock_events)))
        
        # Add sensor noise
        temp_noise = 2.5 * rng.standard_normal(len(load))
        
        return np.clip(temperature + temp_noise, 70, 130)
    
    def generate_complete_dataset_minute(self) -> pd.DataFrame:
        """
        Generate complete realistic minute-based engine dataset
        """
        print("🚗 Generating realistic minute-based engine data...")
        print(f"📊 Duration: {DURATION_DAYS} days ({TOTAL_MINUTES:,} minutes)")
        
        # Generate time vector (minute intervals)
        time_stamps = pd.date_range(
            start=START_DATE,
            periods=TOTAL_MINUTES,
            freq='1T'  # 1-minute frequency
        )
        
        # Generate driving cycle
        print("🛣️  Generating minute-based driving cycle...")
        driving_data = self.generate_realistic_driving_cycle_minute(TOTAL_MINUTES)
        
        # Calculate engine parameters
        print("⚙️  Calculating engine parameters...")
        ignition_timing = self.calculate_ignition_timing_minute(
            driving_data['rpm'], driving_data['load']
        )
        
        cylinder_pressure = self.calculate_cylinder_pressure_minute(
            driving_data['rpm'], driving_data['load'], ignition_timing
        )
        
        # Calculate knock probability and events
        print("💥 Calculating knock events...")
        knock_probability = self.calculate_knock_probability_minute(
            driving_data['rpm'], driving_data['load'], 
            cylinder_pressure, ignition_timing
        )
        
        knock_events = self.generate_knock_events_minute(knock_probability)
        knock_count = np.sum(knock_events > 0)
        knock_rate = knock_count / TOTAL_MINUTES * 100
        
        print(f"🎯 Knock events: {knock_count:,} minutes ({knock_rate:.3f}%)")
        
        # Calculate all sensor outputs
        print("📡 Calculating sensor outputs...")
        burn_rate = self.calculate_burn_rate_minute(
            driving_data['rpm'], driving_data['load'], ignition_timing, knock_events
        )
        
        vibration = self.calculate_vibration_sensor_minute(
            driving_data['rpm'], knock_events
        )
        
        ego_voltage = self.calculate_ego_voltage_minute(
            driving_data['load'], knock_events
        )
        
        temperature = self.calculate_temperature_sensor_minute(
            driving_data['load'], driving_data['rpm'], knock_events
        )
        
        # Create DataFrame
        df = pd.DataFrame({
            'Timestamp': time_stamps,
            'Knock': knock_events.round(4),  # Keep fractional values for risk levels
            'RPM': driving_data['rpm'].round(1),
            'Load': driving_data['load'].round(1),
            'ThrottlePosition': driving_data['throttle_position'].round(1),
            'IgnitionTiming': ignition_timing.round(2),
            'CylinderPressure': cylinder_pressure.round(2),
            'BurnRate': burn_rate.round(4),
            'Vibration': vibration.round(4),
            'EGOVoltage': ego_voltage.round(3),
            'TempSensor': temperature.round(1)
        })
        
        return df
    
    def plot_sample_data_minute(self, df: pd.DataFrame, sample_hours: int = 4):
        """
        Plot sample data for minute-based visualization
        """
        # Sample first N hours (in minutes)
        sample_size = sample_hours * 60
        df_sample = df.iloc[:sample_size]
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Realistic Minute-Based Engine Data Sample (First {sample_hours} Hours)', fontsize=14)
        
        # RPM and Load
        axes[0, 0].plot(df_sample['Timestamp'], df_sample['RPM'], 'b-', alpha=0.8, linewidth=1)
        axes[0, 0].set_title('Engine RPM (Minute-Based)')
        axes[0, 0].set_ylabel('RPM')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(df_sample['Timestamp'], df_sample['Load'], 'g-', alpha=0.8, linewidth=1)
        axes[0, 1].set_title('Engine Load (Minute-Based)')
        axes[0, 1].set_ylabel('Load (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cylinder Pressure and Vibration
        axes[1, 0].plot(df_sample['Timestamp'], df_sample['CylinderPressure'], 'r-', alpha=0.8, linewidth=1)
        axes[1, 0].set_title('Cylinder Pressure (Minute-Based)')
        axes[1, 0].set_ylabel('Pressure (bar)')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(df_sample['Timestamp'], df_sample['Vibration'], 'purple', alpha=0.8, linewidth=1)
        axes[1, 1].set_title('Vibration Sensor (Minute-Based)')
        axes[1, 1].set_ylabel('Vibration (m/s²)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # EGO Voltage and Temperature
        axes[2, 0].plot(df_sample['Timestamp'], df_sample['EGOVoltage'], 'orange', alpha=0.8, linewidth=1)
        axes[2, 0].set_title('EGO Voltage (Minute-Based)')
        axes[2, 0].set_ylabel('Voltage (V)')
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].plot(df_sample['Timestamp'], df_sample['TempSensor'], 'brown', alpha=0.8, linewidth=1)
        axes[2, 1].set_title('Temperature Sensor (Minute-Based)')
        axes[2, 1].set_ylabel('Temperature (°C)')
        axes[2, 1].grid(True, alpha=0.3)
        
        # Mark knock risk events (fractional values > 0.1)
        high_knock_times = df_sample[df_sample['Knock'] > 0.1]['Timestamp']
        for ax in axes.flat:
            for knock_time in high_knock_times:
                ax.axvline(knock_time, color='red', linestyle='--', alpha=0.6, linewidth=1)
        
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.show()
        
        return fig

def main():
    """
    Main function to generate realistic minute-based engine data
    """
    print("=" * 60)
    print("🔧 REALISTIC MINUTE-BASED ENGINE DATA GENERATOR")
    print("=" * 60)
    
    # Create simulator
    simulator = RealisticEngineSimulatorMinute()
    
    # Generate dataset
    df = simulator.generate_complete_dataset_minute()
    
    # Save to CSV
    filename = 'data/realistic_engine_knock_data_week_minute.csv'
    df.to_csv(filename, index=False)
    
    # Print statistics
    print("\n📈 DATASET STATISTICS:")
    print(f"Total records: {len(df):,} minutes")
    print(f"Time span: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    
    # Knock statistics
    binary_knocks = (df['Knock'] > 0).sum()
    high_risk_knocks = (df['Knock'] > 0.1).sum()
    max_knock_intensity = df['Knock'].max()
    
    print(f"Knock minutes: {binary_knocks:,} ({binary_knocks/len(df)*100:.3f}%)")
    print(f"High-risk knock minutes: {high_risk_knocks:,} ({high_risk_knocks/len(df)*100:.3f}%)")
    print(f"Max knock intensity: {max_knock_intensity:.3f}")
    
    # Parameter ranges
    print(f"RPM range: {df['RPM'].min():.0f} - {df['RPM'].max():.0f}")
    print(f"Load range: {df['Load'].min():.1f}% - {df['Load'].max():.1f}%")
    print(f"Cylinder pressure range: {df['CylinderPressure'].min():.1f} - {df['CylinderPressure'].max():.1f} bar")
    print(f"Temperature range: {df['TempSensor'].min():.1f} - {df['TempSensor'].max():.1f} °C")
    
    print(f"\n💾 Dataset saved to: {filename}")
    
    # Generate sample plot
    print("\n📊 Generating sample visualization...")
    simulator.plot_sample_data_minute(df, sample_hours=4)
    
    # Efficiency comparison
    equivalent_seconds = len(df) * 60
    print(f"\n⚡ EFFICIENCY COMPARISON:")
    print(f"   Minute-based data: {len(df):,} points")
    print(f"   Equivalent second-based: {equivalent_seconds:,} points")
    print(f"   Data reduction: {equivalent_seconds/len(df):.0f}x fewer points")
    print(f"   Storage efficiency: ~{equivalent_seconds/len(df):.0f}x smaller files")
    print(f"   Processing speed: ~{equivalent_seconds/len(df):.0f}x faster")
    
    return df

if __name__ == "__main__":
    df = main()
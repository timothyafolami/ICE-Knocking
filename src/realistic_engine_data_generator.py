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
SAMPLE_RATE_HZ = 1  # 1 second intervals
TOTAL_SECONDS = DURATION_DAYS * 24 * 3600
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

# Knock occurrence probability (realistic: <0.5% under normal conditions)
BASE_KNOCK_PROBABILITY = 0.002  # 0.2% base rate
HIGH_LOAD_THRESHOLD = 80.0      # % load where knock risk increases
HIGH_RPM_THRESHOLD = 4000       # RPM where knock risk increases

# Sensor specifications
KNOCK_SENSOR_FREQ_RANGE = (5000, 15000)  # Hz, typical automotive knock sensor
VIBRATION_BASELINE = 0.1  # m/s² baseline vibration
TEMP_OPERATING_RANGE = (85, 105)  # °C normal operating temperature

# Random number generator
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

class RealisticEngineSimulator:
    """
    Realistic engine simulation based on automotive engineering principles
    """
    
    def __init__(self):
        self.time_vector = None
        self.engine_state = {}
        self.sensors = {}
        
    def generate_realistic_driving_cycle(self, duration_seconds: int) -> Dict[str, np.ndarray]:
        """
        Generate realistic driving patterns based on WLTP/FTP75 cycles
        """
        time_array = np.arange(duration_seconds)
        
        # Create mixed driving conditions
        # 40% city driving (low RPM, frequent stops)
        # 35% highway driving (steady medium-high RPM)
        # 20% suburban driving (mixed conditions)
        # 5% performance driving (high load, potential knock conditions)
        
        # Base driving pattern - sinusoidal with noise
        daily_pattern = np.sin(2 * np.pi * time_array / (24 * 3600))  # Daily cycle
        weekly_pattern = np.sin(2 * np.pi * time_array / (7 * 24 * 3600))  # Weekly cycle
        
        # RPM generation with realistic patterns
        rpm_base = 1200 + 800 * (0.5 + 0.3 * daily_pattern + 0.1 * weekly_pattern)
        
        # Add driving events
        for i in range(0, duration_seconds, 3600):  # Every hour
            if rng.random() < 0.7:  # 70% chance of driving activity
                duration = rng.integers(300, 3600)  # 5-60 minutes
                end_idx = min(i + duration, duration_seconds)
                
                # City driving
                if rng.random() < 0.4:
                    rpm_base[i:end_idx] += 500 * rng.random(end_idx - i) + 200
                # Highway driving
                elif rng.random() < 0.6:
                    rpm_base[i:end_idx] += 1500 + 300 * rng.random(end_idx - i)
                # Performance driving (rare)
                else:
                    rpm_base[i:end_idx] += 2000 + 1000 * rng.random(end_idx - i)
        
        # Apply noise and clip to realistic range
        rpm_noise = 50 * rng.standard_normal(duration_seconds)
        rpm = np.clip(rpm_base + rpm_noise, RPM_IDLE, RPM_MAX)
        
        # Engine load based on RPM and driving conditions
        load_base = 20 + 60 * (rpm - RPM_IDLE) / (RPM_MAX - RPM_IDLE)
        load_variation = 20 * rng.standard_normal(duration_seconds)
        load = np.clip(load_base + load_variation, LOAD_MIN, LOAD_MAX)
        
        # Throttle position correlates with load
        throttle_position = load + 5 * rng.standard_normal(duration_seconds)
        throttle_position = np.clip(throttle_position, 0, 100)
        
        return {
            'rpm': rpm,
            'load': load,
            'throttle_position': throttle_position
        }
    
    def calculate_ignition_timing(self, rpm: np.ndarray, load: np.ndarray) -> np.ndarray:
        """
        Calculate realistic ignition timing based on engine maps
        Modern engines use advanced timing at low load, retarded at high load
        """
        # Base timing curve (degrees BTDC)
        base_timing = 10 + 15 * (rpm - RPM_IDLE) / (RPM_MAX - RPM_IDLE)
        
        # Load compensation (retard timing under high load)
        load_compensation = -0.1 * (load - 50)  # Retard 0.1° per % load above 50%
        
        # Add realistic variation
        timing_noise = 1.0 * rng.standard_normal(len(rpm))
        
        ignition_timing = base_timing + load_compensation + timing_noise
        return np.clip(ignition_timing, 5, 35)  # Realistic timing range
    
    def calculate_cylinder_pressure(self, rpm: np.ndarray, load: np.ndarray, 
                                  ignition_timing: np.ndarray) -> np.ndarray:
        """
        Calculate cylinder pressure based on combustion physics
        """
        # Base pressure from compression ratio
        compression_pressure = 12 + 2 * rng.standard_normal(len(rpm))
        
        # Combustion pressure rise based on load and timing
        combustion_pressure = load * 0.3 * (1 + 0.1 * np.sin(ignition_timing * np.pi / 180))
        
        # RPM effect on pressure dynamics
        rpm_effect = 0.002 * (rpm - 1000)
        
        # Total cylinder pressure
        cylinder_pressure = compression_pressure + combustion_pressure + rpm_effect
        
        # Add realistic noise
        pressure_noise = 1.5 * rng.standard_normal(len(rpm))
        
        return np.maximum(cylinder_pressure + pressure_noise, 8.0)  # Minimum atmospheric + some
    
    def calculate_knock_probability(self, rpm: np.ndarray, load: np.ndarray, 
                                  cylinder_pressure: np.ndarray, 
                                  ignition_timing: np.ndarray) -> np.ndarray:
        """
        Calculate realistic knock probability based on engine conditions
        """
        # Base probability
        knock_prob = np.full(len(rpm), BASE_KNOCK_PROBABILITY)
        
        # High load increases knock probability
        high_load_mask = load > HIGH_LOAD_THRESHOLD
        knock_prob[high_load_mask] *= 3.0
        
        # High RPM with high load is dangerous
        high_rpm_mask = rpm > HIGH_RPM_THRESHOLD
        knock_prob[high_load_mask & high_rpm_mask] *= 2.0
        
        # Advanced ignition timing increases knock risk
        advanced_timing_mask = ignition_timing > 25
        knock_prob[advanced_timing_mask] *= 1.5
        
        # Very high cylinder pressure increases knock risk
        high_pressure_mask = cylinder_pressure > 40
        knock_prob[high_pressure_mask] *= 2.0
        
        # Temperature effect (assume higher temp with high load)
        temp_effect = 1 + 0.02 * load  # 2% increase per % load
        knock_prob *= temp_effect
        
        return np.clip(knock_prob, 0, 0.05)  # Max 5% probability
    
    def generate_knock_events(self, knock_probability: np.ndarray) -> np.ndarray:
        """
        Generate knock events based on probability
        """
        random_values = rng.random(len(knock_probability))
        return random_values < knock_probability
    
    def calculate_burn_rate(self, rpm: np.ndarray, load: np.ndarray, 
                          knock_events: np.ndarray) -> np.ndarray:
        """
        Calculate burn rate using Wiebe function with knock effects
        """
        # Wiebe function parameters
        a = 5.0  # Shape parameter
        n = 3.0  # Shape parameter
        
        # Crank angle simulation (simplified)
        theta_0 = 10  # Start of combustion (degrees ATDC)
        delta_theta = 40 + 20 * load / 100  # Burn duration increases with load
        
        # Simulate crank angle progression
        theta = theta_0 + delta_theta * rng.random(len(rpm))
        
        # Wiebe burn rate
        x = np.clip((theta - theta_0) / delta_theta, 0, 1)
        burn_rate = 1 - np.exp(-a * x**n)
        
        # Knock affects burn rate (incomplete combustion)
        burn_rate[knock_events] *= 0.85  # 15% reduction in burn efficiency
        
        return burn_rate
    
    def calculate_vibration_sensor(self, rpm: np.ndarray, knock_events: np.ndarray) -> np.ndarray:
        """
        Calculate vibration sensor output with realistic frequency content
        """
        # Base engine vibration (firing frequency and harmonics)
        firing_freq = rpm / 60 * (CYLINDERS / 2)  # 4-stroke engine
        
        # Primary vibration from engine firing
        t = np.arange(len(rpm))
        primary_vib = 0.05 * np.sin(2 * np.pi * firing_freq * t / SAMPLE_RATE_HZ)
        
        # Secondary harmonics
        harmonic_vib = 0.02 * np.sin(4 * np.pi * firing_freq * t / SAMPLE_RATE_HZ)
        
        # Random mechanical vibration
        mechanical_noise = VIBRATION_BASELINE * rng.standard_normal(len(rpm))
        
        # Base vibration
        vibration = primary_vib + harmonic_vib + mechanical_noise
        
        # Knock-induced vibration spikes (high frequency)
        knock_amplitude = 0.5  # Significant spike during knock
        vibration[knock_events] += knock_amplitude * (1 + 0.3 * rng.standard_normal(np.sum(knock_events)))
        
        return vibration
    
    def calculate_ego_voltage(self, load: np.ndarray, knock_events: np.ndarray) -> np.ndarray:
        """
        Calculate EGO (oxygen) sensor voltage with realistic lambda response
        """
        # Base voltage around stoichiometric (lambda = 1)
        base_voltage = 0.45  # Volts at lambda = 1
        
        # Load affects mixture (higher load = richer mixture)
        mixture_effect = -0.05 * (load - 50) / 50  # ±0.05V swing
        
        # Oscillation around stoichiometric due to closed-loop control
        oscillation = 0.03 * np.sin(2 * np.pi * np.arange(len(load)) / 10)
        
        # Knock events cause lean condition (incomplete combustion)
        ego_voltage = base_voltage + mixture_effect + oscillation
        ego_voltage[knock_events] += 0.1  # Lean spike during knock
        
        # Add sensor noise
        sensor_noise = 0.01 * rng.standard_normal(len(load))
        
        return np.clip(ego_voltage + sensor_noise, 0.1, 0.9)
    
    def calculate_temperature_sensor(self, load: np.ndarray, rpm: np.ndarray, 
                                   knock_events: np.ndarray) -> np.ndarray:
        """
        Calculate engine temperature sensor with thermal dynamics
        """
        # Base operating temperature
        base_temp = 90  # °C
        
        # Load effect on temperature
        load_effect = 15 * load / 100  # Up to 15°C increase at full load
        
        # RPM effect (higher RPM = better cooling but more friction)
        rpm_effect = 5 * (rpm - 2000) / 4000  # ±5°C variation
        
        # Thermal inertia (temperature changes slowly)
        temp_signal = base_temp + load_effect + rpm_effect
        
        # Apply thermal filtering (1st order lag)
        temperature = np.zeros_like(temp_signal)
        temperature[0] = temp_signal[0]
        tau = 30  # 30-second time constant
        alpha = 1 / (tau * SAMPLE_RATE_HZ + 1)
        
        for i in range(1, len(temp_signal)):
            temperature[i] = alpha * temp_signal[i] + (1 - alpha) * temperature[i-1]
        
        # Knock events increase temperature locally
        temperature[knock_events] += 8 * (1 + 0.2 * rng.standard_normal(np.sum(knock_events)))
        
        # Add sensor noise
        temp_noise = 2.0 * rng.standard_normal(len(load))
        
        return np.clip(temperature + temp_noise, 70, 130)
    
    def generate_complete_dataset(self) -> pd.DataFrame:
        """
        Generate complete realistic engine dataset
        """
        print("🚗 Generating realistic engine data...")
        print(f"📊 Duration: {DURATION_DAYS} days ({TOTAL_SECONDS:,} seconds)")
        
        # Generate time vector
        time_stamps = pd.date_range(
            start=START_DATE,
            periods=TOTAL_SECONDS,
            freq='1S'
        )
        
        # Generate driving cycle
        print("🛣️  Generating driving cycle...")
        driving_data = self.generate_realistic_driving_cycle(TOTAL_SECONDS)
        
        # Calculate engine parameters
        print("⚙️  Calculating engine parameters...")
        ignition_timing = self.calculate_ignition_timing(
            driving_data['rpm'], driving_data['load']
        )
        
        cylinder_pressure = self.calculate_cylinder_pressure(
            driving_data['rpm'], driving_data['load'], ignition_timing
        )
        
        # Calculate knock probability and events
        print("💥 Calculating knock events...")
        knock_probability = self.calculate_knock_probability(
            driving_data['rpm'], driving_data['load'], 
            cylinder_pressure, ignition_timing
        )
        
        knock_events = self.generate_knock_events(knock_probability)
        knock_count = np.sum(knock_events)
        knock_rate = knock_count / TOTAL_SECONDS * 100
        
        print(f"🎯 Knock events: {knock_count:,} ({knock_rate:.3f}%)")
        
        # Calculate all sensor outputs
        print("📡 Calculating sensor outputs...")
        burn_rate = self.calculate_burn_rate(
            driving_data['rpm'], driving_data['load'], knock_events
        )
        
        vibration = self.calculate_vibration_sensor(
            driving_data['rpm'], knock_events
        )
        
        ego_voltage = self.calculate_ego_voltage(
            driving_data['load'], knock_events
        )
        
        temperature = self.calculate_temperature_sensor(
            driving_data['load'], driving_data['rpm'], knock_events
        )
        
        # Create DataFrame
        df = pd.DataFrame({
            'Timestamp': time_stamps,
            'Knock': knock_events.astype(int),
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
    
    def plot_sample_data(self, df: pd.DataFrame, sample_hours: int = 2):
        """
        Plot sample data for visualization
        """
        # Sample first N hours
        sample_size = sample_hours * 3600
        df_sample = df.iloc[:sample_size]
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))
        fig.suptitle(f'Realistic Engine Data Sample (First {sample_hours} Hours)', fontsize=14)
        
        # RPM and Load
        axes[0, 0].plot(df_sample['Timestamp'], df_sample['RPM'], 'b-', alpha=0.7)
        axes[0, 0].set_title('Engine RPM')
        axes[0, 0].set_ylabel('RPM')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].plot(df_sample['Timestamp'], df_sample['Load'], 'g-', alpha=0.7)
        axes[0, 1].set_title('Engine Load')
        axes[0, 1].set_ylabel('Load (%)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Cylinder Pressure and Vibration
        axes[1, 0].plot(df_sample['Timestamp'], df_sample['CylinderPressure'], 'r-', alpha=0.7)
        axes[1, 0].set_title('Cylinder Pressure')
        axes[1, 0].set_ylabel('Pressure (bar)')
        axes[1, 0].grid(True, alpha=0.3)
        
        axes[1, 1].plot(df_sample['Timestamp'], df_sample['Vibration'], 'purple', alpha=0.7)
        axes[1, 1].set_title('Vibration Sensor')
        axes[1, 1].set_ylabel('Vibration (m/s²)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # EGO Voltage and Temperature
        axes[2, 0].plot(df_sample['Timestamp'], df_sample['EGOVoltage'], 'orange', alpha=0.7)
        axes[2, 0].set_title('EGO Voltage')
        axes[2, 0].set_ylabel('Voltage (V)')
        axes[2, 0].grid(True, alpha=0.3)
        
        axes[2, 1].plot(df_sample['Timestamp'], df_sample['TempSensor'], 'brown', alpha=0.7)
        axes[2, 1].set_title('Temperature Sensor')
        axes[2, 1].set_ylabel('Temperature (°C)')
        axes[2, 1].grid(True, alpha=0.3)
        
        # Mark knock events
        knock_times = df_sample[df_sample['Knock'] == 1]['Timestamp']
        for ax in axes.flat:
            for knock_time in knock_times:
                ax.axvline(knock_time, color='red', linestyle='--', alpha=0.5, linewidth=0.5)
        
        plt.tight_layout()
        plt.xticks(rotation=45)
        plt.show()
        
        return fig

def main():
    """
    Main function to generate realistic engine data
    """
    print("=" * 60)
    print("🔧 REALISTIC ENGINE DATA GENERATOR")
    print("=" * 60)
    
    # Create simulator
    simulator = RealisticEngineSimulator()
    
    # Generate dataset
    df = simulator.generate_complete_dataset()
    
    # Save to CSV
    filename = 'data/realistic_engine_knock_data_week.csv'
    df.to_csv(filename, index=False)
    
    # Print statistics
    print("\n📈 DATASET STATISTICS:")
    print(f"Total records: {len(df):,}")
    print(f"Time span: {df['Timestamp'].min()} to {df['Timestamp'].max()}")
    print(f"Knock events: {df['Knock'].sum():,} ({df['Knock'].mean()*100:.3f}%)")
    print(f"RPM range: {df['RPM'].min():.0f} - {df['RPM'].max():.0f}")
    print(f"Load range: {df['Load'].min():.1f}% - {df['Load'].max():.1f}%")
    print(f"Cylinder pressure range: {df['CylinderPressure'].min():.1f} - {df['CylinderPressure'].max():.1f} bar")
    print(f"Temperature range: {df['TempSensor'].min():.1f} - {df['TempSensor'].max():.1f} °C")
    
    print(f"\n💾 Dataset saved to: {filename}")
    
    # Generate sample plot
    print("\n📊 Generating sample visualization...")
    simulator.plot_sample_data(df, sample_hours=2)
    
    return df

if __name__ == "__main__":
    df = main()
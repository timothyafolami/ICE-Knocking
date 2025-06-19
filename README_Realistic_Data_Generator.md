# Realistic Engine Data Generator

## Overview

This document provides a comprehensive explanation of the `realistic_engine_data_generator.py` script, which generates physics-based, automotive-grade engine data for knock detection research. The generator creates realistic second-by-second engine operating data over a one-week period, incorporating proper automotive engineering principles and standards.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Engine Specifications](#engine-specifications)
3. [Physics-Based Modeling](#physics-based-modeling)  
4. [Sensor Simulation](#sensor-simulation)
5. [Data Generation Process](#data-generation-process)
6. [Code Structure](#code-structure)
7. [Usage Instructions](#usage-instructions)
8. [Output Data Format](#output-data-format)

## Architecture Overview

The generator is built around the `RealisticEngineSimulator` class, which implements a comprehensive engine simulation based on:

- **Automotive Engineering Standards**: Real-world engine specifications and operating parameters
- **Physics-Based Modeling**: Combustion physics, thermodynamics, and sensor response characteristics
- **Realistic Driving Patterns**: WLTP/FTP75-inspired driving cycles with mixed operating conditions
- **Proper Sensor Correlations**: Realistic sensor responses with appropriate noise and dynamics

## Engine Specifications

The simulator models a modern 1.4L turbocharged 4-cylinder engine with the following specifications:

```python
ENGINE_DISPLACEMENT = 1.4      # Liters
MAX_POWER_KW = 110            # kW at 5500 RPM  
MAX_TORQUE_NM = 200           # Nm at 1500-4000 RPM
COMPRESSION_RATIO = 10.0      # Modern turbo engine
CYLINDERS = 4                 # 4-cylinder configuration
```

### Operating Ranges
- **RPM Range**: 800 (idle) to 6500 (max), with 6000 redline
- **Load Range**: 0% to 100% throttle position
- **Temperature Range**: 85°C to 118°C operating temperature
- **Pressure Range**: 8 to 58 bar cylinder pressure

## Physics-Based Modeling

### 1. Driving Cycle Generation

The `generate_realistic_driving_cycle()` method creates realistic driving patterns based on:

- **Daily Patterns**: Sinusoidal variation representing typical daily driving
- **Weekly Patterns**: Long-term variation over the week
- **Mixed Driving Conditions**:
  - 40% City driving (low RPM, frequent stops)
  - 35% Highway driving (steady medium-high RPM)
  - 20% Suburban driving (mixed conditions)
  - 5% Performance driving (high load, potential knock conditions)

```python
# Base driving pattern with daily and weekly cycles
daily_pattern = np.sin(2 * np.pi * time_array / (24 * 3600))
weekly_pattern = np.sin(2 * np.pi * time_array / (7 * 24 * 3600))
rpm_base = 1200 + 800 * (0.5 + 0.3 * daily_pattern + 0.1 * weekly_pattern)
```

### 2. Ignition Timing Calculation

The `calculate_ignition_timing()` method implements realistic ignition timing based on:

- **Base Timing Curve**: Advances with RPM for optimal efficiency
- **Load Compensation**: Retards timing under high load to prevent knock
- **Realistic Variation**: Adds appropriate noise for real-world conditions

```python
base_timing = 10 + 15 * (rpm - RPM_IDLE) / (RPM_MAX - RPM_IDLE)
load_compensation = -0.1 * (load - 50)  # Retard 0.1° per % load above 50%
```

### 3. Cylinder Pressure Modeling

The `calculate_cylinder_pressure()` method calculates pressure based on:

- **Compression Pressure**: Base pressure from compression ratio
- **Combustion Pressure**: Load-dependent pressure rise
- **RPM Effects**: Pressure dynamics variation with engine speed
- **Timing Effects**: Ignition timing influence on peak pressure

### 4. Knock Probability Calculation

The `calculate_knock_probability()` method implements realistic knock probability based on:

- **Base Rate**: 0.2% baseline knock probability
- **High Load Factor**: 3x increase above 80% load
- **High RPM Factor**: 2x increase above 4000 RPM with high load
- **Advanced Timing**: 1.5x increase with timing >25° BTDC
- **High Pressure**: 2x increase with pressure >40 bar
- **Temperature Effect**: 2% increase per % load (thermal effect)

```python
knock_prob = np.full(len(rpm), BASE_KNOCK_PROBABILITY)  # 0.2% base
high_load_mask = load > HIGH_LOAD_THRESHOLD
knock_prob[high_load_mask] *= 3.0
```

## Sensor Simulation

### 1. Vibration Sensor

The `calculate_vibration_sensor()` method models realistic vibration patterns:

- **Firing Frequency**: Primary vibration from engine firing (RPM/60 * 2 for 4-stroke)
- **Harmonics**: Secondary vibrations at 2x firing frequency
- **Mechanical Noise**: Random vibration from engine components
- **Knock Spikes**: Significant amplitude increase during knock events (0.5 m/s²)

```python
firing_freq = rpm / 60 * (CYLINDERS / 2)  # 4-stroke engine
primary_vib = 0.05 * np.sin(2 * np.pi * firing_freq * t / SAMPLE_RATE_HZ)
vibration[knock_events] += knock_amplitude * (1 + 0.3 * rng.standard_normal(...))
```

### 2. EGO (Oxygen) Sensor

The `calculate_ego_voltage()` method simulates lambda sensor behavior:

- **Stoichiometric Operation**: Base voltage around 0.45V (lambda = 1)
- **Load Effects**: Richer mixture under high load
- **Closed-Loop Oscillation**: Realistic control system oscillation
- **Knock Response**: Lean condition during knock (incomplete combustion)

### 3. Temperature Sensor

The `calculate_temperature_sensor()` method includes:

- **Thermal Dynamics**: First-order lag with 30-second time constant
- **Load Effects**: Temperature rise with engine load
- **RPM Effects**: Cooling vs. friction balance
- **Knock Response**: Local temperature increase during knock events

```python
# Thermal filtering (1st order lag)
tau = 30  # 30-second time constant
alpha = 1 / (tau * SAMPLE_RATE_HZ + 1)
temperature[i] = alpha * temp_signal[i] + (1 - alpha) * temperature[i-1]
```

### 4. Burn Rate Calculation

The `calculate_burn_rate()` method uses the Wiebe function:

- **Wiebe Parameters**: Industry-standard combustion modeling (a=5, n=3)
- **Load Dependency**: Burn duration increases with load
- **Knock Effects**: 15% reduction in burn efficiency during knock

```python
# Wiebe burn rate
x = np.clip((theta - theta_0) / delta_theta, 0, 1)
burn_rate = 1 - np.exp(-a * x**n)
burn_rate[knock_events] *= 0.85  # Knock reduces efficiency
```

## Data Generation Process

### Step 1: Time Vector Generation
- Creates 604,800 timestamps (1 second intervals for 7 days)
- Uses pandas date_range for proper datetime handling

### Step 2: Driving Cycle Generation
- Generates realistic RPM, load, and throttle position profiles
- Incorporates mixed driving conditions and random events

### Step 3: Engine Parameter Calculation
- Calculates ignition timing based on RPM and load
- Determines cylinder pressure from combustion physics
- Computes knock probability from multiple factors

### Step 4: Knock Event Generation
- Uses probabilistic approach to generate knock events
- Typically results in 0.3-0.5% knock occurrence

### Step 5: Sensor Output Calculation
- Computes all sensor responses with proper physics
- Applies realistic noise and dynamics
- Correlates sensor responses with knock events

### Step 6: Data Assembly and Export
- Creates comprehensive DataFrame with all parameters
- Exports to CSV with proper formatting
- Generates visualization plots

## Code Structure

### Main Class: `RealisticEngineSimulator`

```python
class RealisticEngineSimulator:
    def __init__(self):
        self.time_vector = None
        self.engine_state = {}
        self.sensors = {}
```

### Key Methods:

1. **`generate_realistic_driving_cycle()`**: Creates driving patterns
2. **`calculate_ignition_timing()`**: Computes timing based on engine maps
3. **`calculate_cylinder_pressure()`**: Physics-based pressure calculation
4. **`calculate_knock_probability()`**: Multi-factor knock risk assessment
5. **`generate_knock_events()`**: Probabilistic knock event generation
6. **`calculate_burn_rate()`**: Wiebe function combustion modeling
7. **`calculate_vibration_sensor()`**: Frequency-domain vibration modeling
8. **`calculate_ego_voltage()`**: Lambda sensor simulation
9. **`calculate_temperature_sensor()`**: Thermal dynamics modeling
10. **`generate_complete_dataset()`**: Master data generation method
11. **`plot_sample_data()`**: Visualization generation

## Usage Instructions

### Basic Usage

```python
from realistic_engine_data_generator import RealisticEngineSimulator

# Create simulator instance
simulator = RealisticEngineSimulator()

# Generate complete dataset
df = simulator.generate_complete_dataset()

# Generate visualization
simulator.plot_sample_data(df, sample_hours=2)
```

### Command Line Usage

```bash
python src/realistic_engine_data_generator.py
```

### Configuration Options

Modify the configuration parameters at the top of the script:

```python
# Time configuration
DURATION_DAYS = 7              # Change duration
SAMPLE_RATE_HZ = 1            # Change sampling rate

# Engine specifications
ENGINE_DISPLACEMENT = 1.4      # Engine size
MAX_POWER_KW = 110            # Maximum power
MAX_TORQUE_NM = 200           # Maximum torque

# Knock parameters
BASE_KNOCK_PROBABILITY = 0.002 # Base knock rate
HIGH_LOAD_THRESHOLD = 80.0     # High load threshold
```

## Output Data Format

The generator produces a CSV file with the following columns:

| Column | Description | Unit | Range |
|--------|-------------|------|-------|
| `Timestamp` | Date and time | ISO format | 7 days |
| `Knock` | Knock event flag | Binary | 0 or 1 |
| `RPM` | Engine speed | RPM | 800-6500 |
| `Load` | Engine load | Percentage | 0-100 |
| `ThrottlePosition` | Throttle position | Percentage | 0-100 |
| `IgnitionTiming` | Ignition timing | Degrees BTDC | 5-35 |
| `CylinderPressure` | Cylinder pressure | Bar | 8-58 |
| `BurnRate` | Burn rate fraction | Dimensionless | 0-1 |
| `Vibration` | Vibration amplitude | m/s² | Variable |
| `EGOVoltage` | Lambda sensor voltage | Volts | 0.1-0.9 |
| `TempSensor` | Engine temperature | Celsius | 85-118 |

### Typical Output Statistics

- **Total Records**: 604,800 (1 week at 1 Hz)
- **Knock Events**: ~2,400 (0.3-0.5%)
- **File Size**: ~50MB
- **Memory Usage**: ~200MB during generation

## Performance Considerations

### Memory Usage
- Peak memory usage: ~200MB during generation
- Optimized for single-pass processing
- Efficient NumPy operations throughout

### Processing Time
- Generation time: ~30-60 seconds on modern hardware
- Dominated by loop-based thermal calculations
- Parallelizable for longer datasets

### Scalability
- Linear scaling with duration
- Configurable sampling rate
- Modular design for easy extension

## Validation and Testing

The generator includes built-in validation:

- **Range Checking**: All parameters within realistic bounds
- **Correlation Validation**: Sensor responses correlate with knock events
- **Statistical Validation**: Knock rates match automotive standards
- **Physics Validation**: Combustion parameters follow known relationships

## Future Enhancements

Potential improvements for future versions:

1. **Multi-Cylinder Modeling**: Individual cylinder variations
2. **Fuel Quality Effects**: Octane rating impact on knock
3. **Environmental Conditions**: Temperature, humidity, altitude effects
4. **Advanced Knock Models**: Detailed combustion chemistry
5. **Real-Time Capability**: Streaming data generation
6. **Calibration Tools**: Parameter tuning utilities

## References

- Automotive Engineering Standards (SAE International)
- Engine Combustion Modeling (Wiebe Function)
- WLTP/FTP75 Driving Cycle Standards
- Knock Sensor Frequency Response Characteristics
- Modern Engine Management Systems

## License

This code is provided for research and educational purposes. Please refer to the main project license for usage terms.
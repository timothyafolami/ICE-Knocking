import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %% 1. Time Setup
start_date = pd.Timestamp('2024-01-01 00:00:00')
end_date = pd.Timestamp('2025-04-01 00:00:00')

# Generate hourly timestamps
time_stamp = pd.date_range(start=start_date, end=end_date - pd.Timedelta(hours=1), freq='H')
N = len(time_stamp)  # Should be 744 (1/60th of minute data)
t_sec = np.arange(N) * 3600  # Time vector in seconds, multiplied by 3600 for hourly intervals

# %% 2. Engine Operational Parameters
# RPM [800 – 4400 rpm]
np.random.seed(42)  # For reproducibility
rpm = 3000 + 500 * np.sin(2 * np.pi * (t_sec / 86400)) + 100 * np.random.randn(N)
rpm = np.clip(rpm, 800, 4400)

# Ignition Timing [° BTDC] – constant for 1GR-FE
ignition_timing = np.full(N, 10.0)

# %% 3. Combustion via Wiebe Function
def burn_rate(theta, theta0, delta_theta, a, n):
    return 1 - np.exp(-a * ((theta - theta0) / delta_theta) ** n)

# Simulated crank angle between 10°–50° each hour
theta = 10 + 40 * np.random.rand(N)

# Peak pressure contribution (arb. units)
P_max = 10  # tuned lower for realistic 13–20 bar total

P_normal = P_max * burn_rate(theta, 10, 40, 5, 3)

# Cylinder Pressure: 13.3 bar baseline + combustion + ±3 bar noise
cylinder_pressure = 13.3 + P_normal + 3 * np.random.randn(N)

# %% 4. Evenly-Spaced Knock Indicator (0.5% of data)
knock = np.zeros(N, dtype=bool)
step = round(1 / 0.005)  # 200
knock[::step] = True  # Every 200 points

# %% 5. Apply Knock Spikes
# Pressure spike: +5 bar ±2 bar noise
knock_indices = np.where(knock)[0]
cylinder_pressure[knock_indices] += 5 + (-2 + 4 * np.random.rand(len(knock_indices)))

# Adjust burn-rate feature
burn_rate_val = P_normal.copy()
burn_rate_val[knock_indices] += 0.15 * np.random.rand(len(knock_indices))

# %% 6. Knock-Sensitive Sensors
# 6.1 Base engine vibration
vib_base = 0.1 * np.sin(2 * np.pi * (rpm / 60) * t_sec) + 0.02 * np.random.randn(N)

# 6.2 Knock-induced vibration bursts
vib_knock = np.zeros(N)
vib_knock[knock_indices] = 0.5

# Combine vibration components
vibration_sensor = vib_base + vib_knock + 0.01 * np.random.randn(N)

# 6.3 EGO (O₂) Sensor Voltage
ego_voltage = 0.45 + 0.05 * np.sin(2 * np.pi * 0.1 * t_sec)
ego_voltage[knock_indices] -= 0.15

# 6.4 Auxiliary Temperature Sensor
temp_sensor = 80 + 30 * np.random.rand(N) + 1 * np.random.randn(N)
temp_sensor[knock_indices] += 8

# %% 7. Assemble & Export
df = pd.DataFrame({
    'Timestamp': time_stamp,
    'Knock': knock,
    'RPM': rpm,
    'IgnitionTiming': ignition_timing,
    'CylinderPressure': cylinder_pressure,
    'BurnRate': burn_rate_val,
    'Vibration': vibration_sensor,
    'EGOVoltage': ego_voltage,
    'TempSensor': temp_sensor
})

# %% 8. Write Data to CSV
csv_filename = 'data/engine_knock_data_hourly.csv'
df.to_csv(csv_filename, index=False)
print(f'CSV file "{csv_filename}" created successfully.')

# %% 9. Generate Diagnostic Plots
sample_range = slice(0, 24)  # First day (24 hourly samples)

plt.figure(figsize=(12, 8))

# RPM Plot
plt.subplot(3, 1, 1)
plt.plot(df['Timestamp'].iloc[sample_range], df['RPM'].iloc[sample_range])
plt.title('Engine RPM (First Day)')
plt.ylabel('RPM')

# Cylinder Pressure Plot
plt.subplot(3, 1, 2)
plt.plot(df['Timestamp'].iloc[sample_range], df['CylinderPressure'].iloc[sample_range])
plt.title('Cylinder Pressure (First Day)')
plt.ylabel('Pressure (bar)')

# Vibration Sensor Plot
plt.subplot(3, 1, 3)
plt.plot(df['Timestamp'].iloc[sample_range], df['Vibration'].iloc[sample_range])
plt.title('Vibration Sensor (First Day)')
plt.ylabel('Vibration')
plt.xlabel('Time')

plt.tight_layout()
plt.show()
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ------------------------------------------------------------------
# CONFIGURATION -----------------------------------------------------
# ------------------------------------------------------------------
START_DATE   = pd.Timestamp("2025-01-01 00:00:00")
END_DATE     = pd.Timestamp("2025-01-08 00:00:00")      # exclusive
TARGET_KNOCK_RATE = 0.30                               # 30 %
RESAMPLE_STR = "30S"                                   # 30-second means
CSV_FILE     = "engine_knock_week_raw.csv"
SEED         = 42
np.random.seed(SEED)
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------
# 1. TIME VECTOR ----------------------------------------------------
time_stamp = pd.date_range(
    start=START_DATE,
    end=END_DATE - pd.Timedelta(seconds=1),  # half-open [start, end)
    freq="S",
    name="Timestamp",
)
N      = len(time_stamp)                    # 2 678 400 rows
t_sec  = np.arange(N)
t_day  = t_sec / 86_400.0                   # fraction of a day

# ------------------------------------------------------------------
# 2. BASE ENGINE SIGNALS -------------------------------------------
rpm = 3000 + 500 * np.sin(2 * np.pi * t_day) + 30 * rng.standard_normal(N)
rpm = np.clip(rpm, 800, 4400)

ignition_timing = 10 + 2 * rng.standard_normal(N)

def wiebe(theta, theta0=10, delta_theta=40, a=5, n=3):
    x = np.clip((theta - theta0) / delta_theta, 0, 1)
    return 1 - np.exp(-a * x**n)

theta        = 10 + 40 * rng.random(N)
P_peak       = 0.006 * rpm
cyl_pressure = 13.3 + P_peak * wiebe(theta) + 1.0 * rng.standard_normal(N)
burn_rate = wiebe(theta)
vib_base  = 0.1 * np.sin(2 * np.pi * rpm / 60 * t_sec) + 0.02 * rng.standard_normal(N)
ego_volt  = 0.45 + 0.05 * np.sin(2 * np.pi * 0.1 * t_sec)  # 0.40–0.50 V swing
temp_sens = 80 + 15 * rng.standard_normal(N)

# ------------------------------------------------------------------
# 3. KNOCK LABELLING WITH 30 % TARGET -------------------------------
#    Pre-conditions: high RPM, lean mix, hot cylinder
# ------------------------------------------------------------------
high_rpm = rpm > 3200
lean_mix = ego_volt < 0.44         # widened threshold so some rows qualify
hot_cyl  = temp_sens > 95

eligible_mask = high_rpm & lean_mix & hot_cyl
eligible_idx  = np.where(eligible_mask)[0]

# If not enough eligible rows, drop the hot-cyl condition
desired_knock_rows = int(TARGET_KNOCK_RATE * N)
if len(eligible_idx) < desired_knock_rows:
    print(f"⚠️  Only {len(eligible_idx):,} rows meet all 3 conditions; "
          "relaxing temperature requirement.")
    eligible_mask = high_rpm & lean_mix
    eligible_idx  = np.where(eligible_mask)[0]

# Sample without replacement to hit the target rate (or max available)
knock_idx = rng.choice(
    eligible_idx,
    size=min(desired_knock_rows, len(eligible_idx)),
    replace=False,
)

knock = np.zeros(N, dtype=bool)
knock[knock_idx] = True
print(f"✅ Knock rows: {knock.sum():,}  "
      f"({100 * knock.mean():.2f} % of {N:,})")

# ------------------------------------------------------------------
# 4. SENSOR EFFECTS AFTER LABELLING --------------------------------
cyl_pressure[knock_idx] += 5 + rng.normal(0, 2, len(knock_idx))

vib_knock = np.zeros(N)
vib_knock[knock_idx] = 0.4
vibration = vib_base + vib_knock + 0.005 * rng.standard_normal(N)

ego_volt[knock_idx]  -= 0.12
temp_sens[knock_idx] += 6

# ------------------------------------------------------------------
# 5. BUILD DATAFRAME  & SAVE ---------------------------------------
df_raw = pd.DataFrame({
    "Timestamp":        time_stamp,
    "Knock":            knock.astype(int),
    "RPM":              rpm,
    "IgnitionTiming":   ignition_timing,
    "CylinderPressure": cyl_pressure,
    "BurnRate":         burn_rate,          # ← new column
    "Vibration":        vibration,
    "EGOVoltage":       ego_volt,
    "TempSensor":       temp_sens,
})
df_raw.to_csv(CSV_FILE, index=False)
print(f"📄 CSV saved → {CSV_FILE}")



# ------------------------------------------------------------------
# 8. OPTIONAL PLOT (first 5 minutes) -------------------------------
if __name__ == "__main__":
    sl = slice(0, 300)   # 300 s = 5 min
    plt.figure(figsize=(12, 7))
    plt.subplot(3, 1, 1)
    plt.plot(df_raw["Timestamp"].iloc[sl], df_raw["RPM"].iloc[sl])
    plt.title("RPM – first 5 min"); plt.ylabel("RPM")

    plt.subplot(3, 1, 2)
    plt.plot(df_raw["Timestamp"].iloc[sl],
             df_raw["CylinderPressure"].iloc[sl])
    plt.title("Cylinder pressure"); plt.ylabel("bar")

    plt.subplot(3, 1, 3)
    plt.plot(df_raw["Timestamp"].iloc[sl], df_raw["Vibration"].iloc[sl])
    plt.title("Vibration"); plt.ylabel("g (arb)"); plt.tight_layout(); plt.show()

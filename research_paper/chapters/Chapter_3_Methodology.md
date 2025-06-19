# Chapter 3: Methodology and System Design

## 3.1 Introduction

This chapter presents the comprehensive methodology and system design for the intelligent engine knock detection and predictive maintenance system. The proposed approach integrates advanced neural network architectures with physics-based modeling to create a production-ready solution capable of both forecasting engine parameters and detecting knock events in real-time automotive applications.

## 3.2 System Architecture Overview

### 3.2.1 Integrated Pipeline Design

The proposed system implements a novel two-tier hybrid architecture that combines machine learning-based forecasting with physics-based derivation and intelligent knock detection:

**Tier 1: Predictive Engine Parameter Forecasting**
- Primary parameter forecasting using LSTM neural networks (RPM, Load, TempSensor)
- Physics-based derivation of secondary parameters (ThrottlePosition, IgnitionTiming, CylinderPressure, BurnRate, Vibration, EGOVoltage)
- Amplitude enhancement for realistic operational variability

**Tier 2: Intelligent Knock Detection**
- Advanced neural network architectures for imbalanced classification
- Real-time inference on forecasted engine conditions
- Comprehensive performance evaluation and confidence scoring

### 3.2.2 Data Flow Architecture

```
Historical Engine Data (7 days, 10,080 minutes)
        ↓
Enhanced Feature Engineering (48 features from 9 original)
        ↓
Neural Network Training (5 architectures, 10 experiments)
        ↓
Model Selection and Optimization
        ↓
Forecasted Parameter Generation (24-hour predictions)
        ↓
Real-time Knock Detection and Analysis
        ↓
Predictive Maintenance Recommendations
```

## 3.3 Data Generation and Preprocessing

### 3.3.1 Realistic Engine Data Simulation

To ensure reproducible research and comprehensive validation, a physics-based engine data generator was developed to create realistic minute-based engine datasets:

**Engine Specifications:**
- Type: 1.4L Turbocharged 4-cylinder
- Power: 110 kW @ 5500 RPM
- Torque: 200 Nm @ 1500-4000 RPM
- Compression Ratio: 10.0:1

**Operational Parameters:**
- Temporal Resolution: 1-minute intervals
- Duration: 7 days (10,080 data points)
- RPM Range: 800-6500 (idle to redline)
- Load Range: 0-100% throttle
- Temperature Range: 85-105°C

**Knock Event Modeling:**
The knock probability calculation incorporates realistic automotive physics:

```python
def calculate_knock_probability(rpm, load, cylinder_pressure, ignition_timing):
    # Base probability (0.8% per minute for realistic conditions)
    knock_prob = BASE_KNOCK_PROBABILITY
    
    # High load increases knock risk
    if load > 70.0:
        knock_prob *= 3.0
    
    # High RPM with high load is critical
    if rpm > 3500 and load > 70.0:
        knock_prob *= 2.0
    
    # Advanced timing increases risk
    if ignition_timing > 22:
        knock_prob *= 1.8
    
    # High pressure conditions
    if cylinder_pressure > 35:
        knock_prob *= 1.5
    
    return knock_prob
```

**Resulting Dataset Characteristics:**
- Total samples: 10,080 minutes
- Knock events: 144 (1.429% occurrence rate)
- Class imbalance ratio: 1:69 (realistic for automotive applications)

### 3.3.2 Feature Engineering Strategy

The system implements comprehensive feature engineering to transform raw engine parameters into informative features for neural network training:

**Original Parameters (9 features):**
- RPM, Load, TempSensor, ThrottlePosition, IgnitionTiming
- CylinderPressure, BurnRate, Vibration, EGOVoltage

**Enhanced Feature Categories:**

1. **Temporal Features (4 features):**
   ```python
   hour = timestamp.hour / 23.0  # Normalized hour
   day_of_week = timestamp.dayofweek / 6.0
   is_weekend = (timestamp.dayofweek >= 5).astype(float)
   minute_of_day = (hour * 60 + minute) / 1439.0
   ```

2. **Rolling Statistics (16 features):**
   - 5-minute rolling mean, std, max, min for critical parameters
   - Captures short-term trend patterns

3. **Rate of Change Features (9 features):**
   ```python
   rpm_diff = rpm.diff().fillna(0)
   rpm_diff_abs = abs(rpm_diff)
   rpm_acceleration = rpm_diff.diff().fillna(0)
   ```

4. **Physics-Based Interactions (6 features):**
   ```python
   load_rpm_interaction = (Load * RPM) / 100000
   pressure_timing_interaction = (CylinderPressure * IgnitionTiming) / 1000
   high_load_high_rpm = ((Load > 80) & (RPM > 3500)).astype(float)
   ```

5. **Engine Stress Indicators (4 features):**
   ```python
   engine_stress = (Load/100)*0.4 + (RPM/6500)*0.3 + (CylinderPressure/60)*0.3
   vibration_intensity = abs(Vibration)
   temp_load_ratio = TempSensor / (Load + 1)
   ```

**Total Enhanced Features: 48** (comprehensive representation for neural network input)

### 3.3.3 Data Preprocessing Pipeline

**Scaling Strategy:**
```python
scaler = RobustScaler()  # Better handling of outliers than StandardScaler
X_scaled = scaler.fit_transform(X_enhanced)
```

**Train-Test Split:**
- Training: 8,064 samples (80%)
- Testing: 2,016 samples (20%)
- Stratified split to maintain class distribution
- Temporal ordering preserved

## 3.4 Neural Network Architectures

### 3.4.1 Architecture Design Philosophy

Five distinct neural network architectures were designed to explore different approaches to the imbalanced knock detection problem:

1. **Deep Dense Network**: Maximum capacity for complex pattern learning
2. **Residual Network**: Skip connections for gradient flow optimization
3. **Attention Network**: Focus mechanism for critical feature importance
4. **Wide & Deep Network**: Combination of memorization and generalization
5. **Ensemble Network**: Multiple specialized sub-networks

### 3.4.2 Deep Dense Network Architecture

**Design Rationale:** Maximize learning capacity through deep architecture with extensive regularization.

```python
model = Sequential([
    Dense(512, activation='relu', input_shape=(48,)),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    
    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.25),
    
    Dense(64, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),
    
    Dense(1, activation='sigmoid', dtype='float32')
])
```

**Key Features:**
- Progressive layer size reduction
- Extensive batch normalization for training stability
- Graduated dropout rates to prevent overfitting
- Mixed precision for computational efficiency

### 3.4.3 Residual Network Architecture

**Design Rationale:** Address vanishing gradient problem while enabling deeper networks.

```python
def residual_block(x, units, dropout_rate):
    residual = x
    x = Dense(units, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(units, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Add()([x, residual])  # Skip connection
    return x

# Implementation with multiple residual blocks
inputs = Input(shape=(48,))
x = Dense(256, activation='relu')(inputs)
x = residual_block(x, 256, 0.25)
x = residual_block(x, 128, 0.25)
outputs = Dense(1, activation='sigmoid')(x)
```

### 3.4.4 Attention Network Architecture

**Design Rationale:** Enable the model to focus on the most relevant features for knock detection.

```python
def attention_mechanism(x):
    # Self-attention implementation
    x_reshaped = tf.expand_dims(x, axis=1)
    attention_output = Attention()([x_reshaped, x_reshaped])
    attention_output = tf.squeeze(attention_output, axis=1)
    return attention_output

inputs = Input(shape=(48,))
x = Dense(256, activation='relu')(inputs)
x = BatchNormalization()(x)

# Apply attention mechanism
x_attention = attention_mechanism(x)
x = Add()([x, x_attention])  # Combine with original features
x = LayerNormalization()(x)

outputs = Dense(1, activation='sigmoid')(x)
```

### 3.4.5 Wide & Deep Network Architecture

**Design Rationale:** Combine memorization of specific patterns with generalization capabilities.

```python
inputs = Input(shape=(48,))

# Wide component (linear memorization)
wide = Dense(1, activation='linear')(inputs)

# Deep component (feature learning)
deep = Dense(256, activation='relu')(inputs)
deep = BatchNormalization()(deep)
deep = Dropout(0.3)(deep)
deep = Dense(128, activation='relu')(deep)
deep = Dense(1, activation='linear')(deep)

# Combine wide and deep
combined = Add()([wide, deep])
outputs = tf.nn.sigmoid(combined)
```

### 3.4.6 Ensemble Network Architecture

**Design Rationale:** Combine multiple specialized networks for robust performance.

```python
def create_ensemble_network(input_dim):
    inputs = Input(shape=(input_dim,))
    
    # Network 1: Focus on basic engine parameters
    net1 = Dense(128, activation='relu')(inputs)
    net1 = BatchNormalization()(net1)
    net1 = Dropout(0.25)(net1)
    net1 = Dense(64, activation='relu')(net1)
    net1_output = Dense(1, activation='linear')(net1)
    
    # Network 2: Focus on interaction features
    net2 = Dense(96, activation='relu')(inputs)
    net2 = BatchNormalization()(net2)
    net2 = Dropout(0.25)(net2)
    net2 = Dense(48, activation='relu')(net2)
    net2_output = Dense(1, activation='linear')(net2)
    
    # Network 3: Focus on temporal features
    net3 = Dense(64, activation='relu')(inputs)
    net3 = BatchNormalization()(net3)
    net3 = Dropout(0.25)(net3)
    net3 = Dense(32, activation='relu')(net3)
    net3_output = Dense(1, activation='linear')(net3)
    
    # Ensemble combination
    ensemble = Add()([net1_output, net2_output, net3_output])
    ensemble = Dense(16, activation='relu')(ensemble)
    outputs = Dense(1, activation='sigmoid', dtype='float32')(ensemble)
    
    return Model(inputs=inputs, outputs=outputs)
```

## 3.5 Imbalanced Learning Techniques

### 3.5.1 Class Weighting Strategy

Given the severe class imbalance (1:69 ratio), advanced weighting techniques were implemented to address the minority class under-representation. The balanced class weighting strategy computes weights inversely proportional to class frequencies:

**Mathematical Formulation:**
For a binary classification problem with classes {0, 1}, the class weights are computed as:

$$w_i = \frac{n}{k \cdot n_i}$$

where:
- $w_i$ = weight for class $i$
- $n$ = total number of samples
- $k$ = number of classes (2 for binary classification)
- $n_i$ = number of samples in class $i$

For our knock detection problem with class distribution $(n_0 = 7949, n_1 = 115)$:

$$w_0 = \frac{8064}{2 \times 7949} = 0.507$$

$$w_1 = \frac{8064}{2 \times 115} = 35.061$$

This weighting strategy effectively amplifies the learning signal from minority class samples (knock events) during training.

### 3.5.2 Focal Loss Mathematical Framework

Focal Loss addresses class imbalance by dynamically down-weighting easy examples and focusing learning on hard-to-classify cases. The mathematical formulation extends binary cross-entropy with a modulating factor:

**Standard Binary Cross-Entropy:**
$$BCE(p_t) = -\log(p_t)$$

**Focal Loss Definition:**
$$FL(p_t) = -\alpha_t(1-p_t)^{\gamma}\log(p_t)$$

where:
- $p_t$ = model's estimated probability for the true class
- $\alpha_t$ = class-dependent weighting factor
- $\gamma$ = focusing parameter (typically 2.0)

**Class-dependent probability calculation:**
$$p_t = \begin{cases}
p & \text{if } y = 1 \text{ (knock event)} \\
1-p & \text{if } y = 0 \text{ (normal operation)}
\end{cases}$$

**Weighting factor:**
$$\alpha_t = \begin{cases}
\alpha & \text{if } y = 1 \\
1-\alpha & \text{if } y = 0
\end{cases}$$

**Key Properties:**
1. **Easy example down-weighting**: When $p_t \to 1$ (confident correct prediction), $(1-p_t)^{\gamma} \to 0$
2. **Hard example emphasis**: When $p_t \to 0$ (uncertain prediction), $(1-p_t)^{\gamma} \to 1$
3. **Class balance**: $\alpha$ parameter balances positive/negative examples

**Implementation with numerical stability:**
```python
def focal_loss(y_true, y_pred, alpha=0.25, gamma=2.0):
    epsilon = K.epsilon()
    y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)
    
    alpha_t = y_true * alpha + (1 - y_true) * (1 - alpha)
    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    focal_loss = -alpha_t * K.pow((1 - p_t), gamma) * K.log(p_t)
    
    return K.mean(focal_loss)
```

### 3.5.3 Advanced Optimization Techniques

**Optimizer Selection:**
- **Adam**: Adaptive learning rate with momentum
- **AdamW**: Adam with weight decay for better generalization
- **RMSprop**: Alternative adaptive optimizer

**Learning Rate Scheduling:**
```python
reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.7,
    patience=5,
    min_lr=1e-7,
    verbose=1
)
```

**Early Stopping:**
```python
early_stopping = EarlyStopping(
    monitor='val_AUC',
    mode='max',
    patience=15,
    restore_best_weights=True,
    min_delta=0.0001
)
```

## 3.6 Engine Parameter Forecasting System

### 3.6.1 LSTM Architecture for Forecasting

The forecasting component uses specialized LSTM networks for each primary engine parameter:

**RPM Forecasting Model:**
```python
model = Sequential([
    LSTM(64, return_sequences=True, dropout=0.1, input_shape=(60, 4)),
    LayerNormalization(),
    LSTM(32, return_sequences=False, dropout=0.1),
    LayerNormalization(),
    Dense(32, activation='relu'),
    Dropout(0.1),
    Dense(16, activation='relu'),
    Dense(1, activation='linear')
])
```

**Multi-Step Forecasting Strategy:**
```python
def forecast_parameter(self, parameter, last_sequence, steps=1440):
    predictions = []
    current_sequence = last_sequence.copy()
    
    for step in range(steps):
        # Predict next minute
        pred = self.model.predict(current_sequence.reshape(1, 60, -1))
        
        # Add controlled noise for variability
        noise = np.random.normal(0, self.config['noise_std'])
        enhanced_pred = pred[0, 0] + noise
        
        predictions.append(enhanced_pred)
        
        # Update sequence (rolling window)
        current_sequence = np.roll(current_sequence, -1, axis=0)
        current_sequence[-1, 0] = enhanced_pred
    
    return np.array(predictions)
```

### 3.6.2 Physics-Based Parameter Derivation

Secondary engine parameters are derived using automotive engineering equations:

**Ignition Timing Derivation:**
```python
def derive_ignition_timing(self, rpm, load):
    # Base timing curve (advances with RPM)
    base_timing = 10 + 15 * (rpm - 800) / (6500 - 800)
    
    # Load compensation (retard under high load)
    load_compensation = -0.1 * (load - 50)
    
    # ECU variation
    timing_noise = 0.5 * np.random.standard_normal(len(rpm))
    
    ignition_timing = base_timing + load_compensation + timing_noise
    return np.clip(ignition_timing, 5, 35)
```

**Cylinder Pressure Derivation:**
```python
def derive_cylinder_pressure(self, rpm, load, ignition_timing):
    # Compression pressure
    compression_pressure = 12 + 0.002 * load
    
    # Combustion pressure (load and timing dependent)
    combustion_pressure = load * 0.3 * (1 + 0.1 * np.sin(ignition_timing * π / 180))
    
    # RPM effect
    rpm_effect = 0.002 * (rpm - 1000)
    
    cylinder_pressure = compression_pressure + combustion_pressure + rpm_effect
    return np.maximum(cylinder_pressure + noise, 8.0)
```

### 3.6.3 Amplitude Enhancement Algorithm

To address LSTM's tendency toward mean-reverting predictions, an amplitude enhancement technique was developed:

```python
def enhance_forecast_amplitude(self, historical_data, forecast_values, parameter):
    # Use recent 24 hours for relevance
    recent_data = historical_data[parameter].iloc[-1440:]
    
    # Calculate statistics
    hist_mean = recent_data.mean()
    hist_std = recent_data.std()
    forecast_std = np.std(forecast_values)
    
    # Enhancement parameters
    target_std = hist_std * 0.8  # Target 80% of historical variability
    scale_factor = min(target_std / forecast_std, 4.0)  # Cap at 4x
    
    # Apply enhancement
    centered_forecast = forecast_values - np.mean(forecast_values)
    scaled_forecast = centered_forecast * scale_factor
    enhanced_forecast = scaled_forecast + hist_mean
    
    # Ensure realistic bounds
    enhanced_forecast = np.clip(enhanced_forecast, 
                               recent_data.min() * 0.7,
                               recent_data.max() * 1.3)
    
    return enhanced_forecast
```

## 3.7 Experimental Design

### 3.7.1 Comprehensive Experiment Matrix

Ten distinct experiments were designed to systematically evaluate different architectures and configurations:

| Experiment | Architecture | Loss Function | Optimizer | Learning Rate | Batch Size | Dropout |
|------------|-------------|---------------|-----------|---------------|------------|---------|
| DeepDense_Baseline | Deep Dense | Binary CE | Adam | 0.001 | 32 | 0.3 |
| DeepDense_FocalLoss | Deep Dense | Focal | Adam | 0.001 | 32 | 0.3 |
| DeepDense_LowDropout | Deep Dense | Binary CE | Adam | 0.0008 | 24 | 0.15 |
| Residual_Baseline | Residual | Binary CE | Adam | 0.001 | 32 | 0.25 |
| Residual_AdamW | Residual | Binary CE | AdamW | 0.0008 | 28 | 0.2 |
| Attention_Baseline | Attention | Binary CE | Adam | 0.0012 | 32 | 0.2 |
| Attention_FocalLoss | Attention | Focal | Adam | 0.001 | 28 | 0.25 |
| WideDeep_Baseline | Wide & Deep | Binary CE | Adam | 0.001 | 32 | 0.3 |
| Ensemble_Baseline | Ensemble | Binary CE | Adam | 0.0015 | 32 | 0.25 |
| Ensemble_FocalLoss | Ensemble | Focal | AdamW | 0.001 | 24 | 0.2 |

### 3.7.2 Evaluation Metrics

**Primary Metrics for Imbalanced Classification:**
- **ROC-AUC**: Overall discriminative performance
- **Precision**: Accuracy of positive predictions
- **Recall**: Sensitivity to minority class (critical for safety)
- **F1-Score**: Harmonic mean of precision and recall
- **Specificity**: True negative rate

**Additional Analysis:**
- **Confusion Matrix**: Detailed classification breakdown
- **Precision-Recall Curves**: Performance across thresholds
- **Feature Importance**: Model interpretability analysis

### 3.7.3 Validation Strategy

**Cross-Validation Approach:**
- Stratified train-test split (80:20)
- Temporal ordering preserved
- Class distribution maintained across splits

**Model Selection Criteria:**
1. Primary: ROC-AUC (overall performance)
2. Secondary: Recall (safety-critical metric)
3. Tertiary: Computational efficiency (production readiness)

## 3.8 Production Deployment Considerations

### 3.8.1 Computational Efficiency

**Model Size Optimization:**
- Parameter count optimization (target <100K parameters)
- Mixed precision training for inference speed
- Efficient architecture selection

**Memory Requirements:**
- Embedded automotive ECU constraints
- Real-time processing capabilities
- Buffer management for streaming data

### 3.8.2 Real-Time Inference Pipeline

**Inference Architecture:**
```python
class KnockInferenceEngine:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = RobustScaler()
        self.feature_buffer = CircularBuffer(size=5)  # For rolling features
    
    def predict_knock(self, engine_data):
        # Real-time feature engineering
        features = self.create_enhanced_features(engine_data)
        
        # Scale features
        features_scaled = self.scaler.transform(features)
        
        # Model inference
        knock_probability = self.model.predict(features_scaled)
        knock_prediction = (knock_probability > 0.5).astype(int)
        
        return knock_prediction, knock_probability
```

### 3.8.3 Integration with Vehicle Systems

**CAN Bus Integration:**
- Standard automotive communication protocols
- Real-time data streaming from engine sensors
- Integration with existing ECU systems

**Failsafe Mechanisms:**
- Fallback to traditional threshold methods
- Model health monitoring
- Graceful degradation strategies

## 3.9 Summary

This methodology chapter presents a comprehensive approach to intelligent engine knock detection and predictive maintenance, integrating advanced neural network architectures with physics-based modeling. The system addresses critical challenges in automotive fault detection, including severe class imbalance, real-time processing requirements, and production deployment constraints.

Key methodological contributions include:

1. **Novel Ensemble Architecture**: Specialized sub-networks for different aspects of engine behavior
2. **Comprehensive Feature Engineering**: 48 enhanced features from 9 original parameters
3. **Hybrid ML-Physics Integration**: Combining data-driven and knowledge-based approaches
4. **Production-Ready Design**: Computationally efficient models suitable for automotive deployment
5. **Systematic Experimental Framework**: Comprehensive evaluation across multiple architectures and configurations

The next chapter presents detailed experimental results and performance analysis, validating the effectiveness of the proposed methodology for real-world automotive applications.
# Nomenclature

## Mathematical Symbols

### Engine Parameters
- **ρ** : Engine load percentage [%]
- **ω** : Engine rotational speed [RPM]
- **T** : Engine coolant temperature [°C]
- **p_c** : Cylinder pressure [bar]
- **θ_ig** : Ignition timing advance [°BTDC]
- **V_EGO** : Exhaust Gas Oxygen sensor voltage [V]
- **a_vib** : Engine block vibration acceleration [m/s²]

### Neural Network Variables
- **X** : Input feature matrix ∈ ℝ^(n×d)
- **y** : Target knock labels ∈ {0,1}^n
- **f(·)** : Neural network function mapping
- **W** : Weight matrices
- **b** : Bias vectors
- **α** : Learning rate
- **λ** : Regularization parameter

### Performance Metrics
- **TPR** : True Positive Rate (Recall/Sensitivity)
- **FPR** : False Positive Rate
- **PPV** : Positive Predictive Value (Precision)
- **NPV** : Negative Predictive Value
- **AUC** : Area Under ROC Curve
- **F₁** : F1-Score (harmonic mean of precision and recall)

### Statistical Parameters
- **μ** : Population mean
- **σ** : Population standard deviation
- **p** : Probability value
- **CI** : Confidence Interval
- **SE** : Standard Error
- **d** : Cohen's effect size

### Optimization Variables
- **L** : Loss function
- **∇** : Gradient operator
- **γ** : Focal loss focusing parameter
- **α_t** : Class-dependent weighting factor
- **β₁, β₂** : Adam optimizer momentum parameters

## Abbreviations and Acronyms

### Automotive Technology
- **ICE** : Internal Combustion Engine
- **ECU** : Electronic Control Unit
- **CAN** : Controller Area Network
- **OBD** : On-Board Diagnostics
- **BTDC** : Before Top Dead Center
- **MAPO** : Maximum Amplitude Pressure Oscillation
- **EGR** : Exhaust Gas Recirculation
- **VVT** : Variable Valve Timing

### Machine Learning and AI
- **ML** : Machine Learning
- **AI** : Artificial Intelligence
- **DL** : Deep Learning
- **NN** : Neural Network
- **CNN** : Convolutional Neural Network
- **LSTM** : Long Short-Term Memory
- **GRU** : Gated Recurrent Unit
- **RNN** : Recurrent Neural Network
- **ANN** : Artificial Neural Network
- **GAN** : Generative Adversarial Network

### Algorithm and Methods
- **SGD** : Stochastic Gradient Descent
- **Adam** : Adaptive Moment Estimation
- **AdamW** : Adam with Weight Decay
- **RMSprop** : Root Mean Square Propagation
- **BCE** : Binary Cross-Entropy
- **ROC** : Receiver Operating Characteristic
- **PR** : Precision-Recall
- **CV** : Cross-Validation
- **PCA** : Principal Component Analysis

### Data Processing
- **FFT** : Fast Fourier Transform
- **STFT** : Short-Time Fourier Transform
- **DWT** : Discrete Wavelet Transform
- **PSD** : Power Spectral Density
- **AR** : Autoregressive
- **ARIMA** : Autoregressive Integrated Moving Average

### Performance and Evaluation
- **RMSE** : Root Mean Square Error
- **MAE** : Mean Absolute Error
- **MAPE** : Mean Absolute Percentage Error
- **MSE** : Mean Square Error
- **R²** : Coefficient of Determination
- **KS** : Kolmogorov-Smirnov (test)

### Standards and Organizations
- **IEEE** : Institute of Electrical and Electronics Engineers
- **SAE** : Society of Automotive Engineers
- **ISO** : International Organization for Standardization
- **EPA** : Environmental Protection Agency
- **CARB** : California Air Resources Board
- **EU** : European Union

### Computing and Hardware
- **CPU** : Central Processing Unit
- **GPU** : Graphics Processing Unit
- **TPU** : Tensor Processing Unit
- **RAM** : Random Access Memory
- **IoT** : Internet of Things
- **API** : Application Programming Interface

## Units and Dimensions

### Physical Units
- **[RPM]** : Revolutions Per Minute
- **[bar]** : Pressure unit (1 bar = 10⁵ Pa)
- **[°C]** : Degrees Celsius
- **[°BTDC]** : Degrees Before Top Dead Center
- **[V]** : Volts
- **[Hz]** : Hertz (frequency)
- **[kW]** : Kilowatts (power)
- **[Nm]** : Newton-meters (torque)
- **[m/s²]** : Meters per second squared (acceleration)

### Computational Units
- **[ms]** : Milliseconds
- **[MB]** : Megabytes
- **[FLOPS]** : Floating Point Operations Per Second
- **[params]** : Model parameters count

### Statistical Units
- **[%]** : Percentage
- **[p.p.]** : Percentage points
- **[σ]** : Standard deviations

## Mathematical Notation Conventions

### Sets and Spaces
- **ℝ** : Real numbers
- **ℕ** : Natural numbers
- **ℝ^n** : n-dimensional real vector space
- **∈** : Element of
- **⊆** : Subset of
- **∪** : Union
- **∩** : Intersection

### Operations
- **∇** : Gradient (nabla operator)
- **∂** : Partial derivative
- **∑** : Summation
- **∏** : Product
- **argmax** : Argument of maximum
- **argmin** : Argument of minimum
- **E[·]** : Expected value
- **Var[·]** : Variance
- **Cov[·,·]** : Covariance

### Probability and Statistics
- **P(·)** : Probability measure
- **p(·)** : Probability density function
- **P(A|B)** : Conditional probability
- **~** : Distributed as
- **≈** : Approximately equal
- **≡** : Defined as/equivalent to

### Matrix and Vector Notation
- **x** : Vector (lowercase, bold)
- **X** : Matrix (uppercase, bold)
- **X^T** : Matrix transpose
- **X^(-1)** : Matrix inverse
- **||x||** : Vector norm
- **⟨x,y⟩** : Inner product
- **⊗** : Kronecker product

## Subscripts and Superscripts

### Temporal Notation
- **t** : Time index
- **i** : Sample index
- **j** : Feature index
- **k** : Class index
- **n** : Total number of samples
- **d** : Feature dimension

### Model Components
- **train** : Training set
- **test** : Test set
- **val** : Validation set
- **pred** : Predicted values
- **true** : True values
- **opt** : Optimal values
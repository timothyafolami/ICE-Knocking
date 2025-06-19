# Installation Guide

## Overview

This guide provides comprehensive installation instructions for the ICE-Knocking engine parameter forecasting and knock detection system.

## System Requirements

### Minimum Requirements
- **Python**: 3.8 or higher (3.9+ recommended)
- **RAM**: 8GB (16GB+ recommended for large datasets)
- **Storage**: 2GB free space
- **OS**: Windows 10+, macOS 10.15+, or Linux (Ubuntu 18.04+)

### Recommended Requirements
- **Python**: 3.10+
- **RAM**: 16GB+ 
- **GPU**: NVIDIA GPU with CUDA 11.2+ (optional but recommended for training)
- **Storage**: 5GB+ free space

## Installation Steps

### 1. Clone or Download the Repository

```bash
git clone <repository-url>
cd ICE-Knocking
```

### 2. Create Virtual Environment (Recommended)

```bash
# Using conda (recommended)
conda create -n ice-knocking python=3.10
conda activate ice-knocking

# OR using venv
python -m venv ice-knocking-env
source ice-knocking-env/bin/activate  # On Windows: ice-knocking-env\Scripts\activate
```

### 3. Install Dependencies

#### Option A: Standard Installation (CPU Only)
```bash
pip install -r requirements.txt
```

#### Option B: GPU Installation (NVIDIA GPUs)
```bash
# Install CUDA-enabled TensorFlow
pip install tensorflow[and-cuda]==2.13.0
pip install -r requirements.txt
```

#### Option C: Development Installation
```bash
pip install -r requirements.txt
pip install -e .  # Install in editable mode
```

### 4. Verify Installation

```bash
# Check basic imports
python -c "import numpy, pandas, sklearn, tensorflow; print('✅ Core libraries installed')"

# Check GPU availability (if applicable)
python run_forecasting.py --check-gpu
```

## GPU Setup (Optional but Recommended)

### NVIDIA GPU Setup

1. **Install NVIDIA Drivers**
   - Download from [NVIDIA Driver Downloads](https://www.nvidia.com/drivers/)
   - Ensure driver version 450.80.02+ for Linux or 451.82+ for Windows

2. **Install CUDA Toolkit**
   ```bash
   # Check CUDA version
   nvidia-smi
   
   # Install CUDA 11.2+ (required for TensorFlow 2.13)
   # Download from https://developer.nvidia.com/cuda-downloads
   ```

3. **Install cuDNN**
   - Download from [NVIDIA cuDNN](https://developer.nvidia.com/cudnn)
   - Extract and copy files to CUDA installation directory

4. **Verify GPU Setup**
   ```bash
   python -c "import tensorflow as tf; print('GPU Available:', tf.config.list_physical_devices('GPU'))"
   ```

### Apple Silicon (M1/M2) Setup

```bash
# Install TensorFlow for Apple Silicon
pip install tensorflow-macos==2.13.0
pip install tensorflow-metal==1.0.1  # For GPU acceleration
```

## Quick Start Test

### 1. Generate Realistic Data
```bash
python src/realistic_engine_data_generator.py
```

### 2. Run Forecasting Pipeline
```bash
python run_forecasting.py
```

### 3. Expected Output
```
🔧 ENGINE PARAMETER FORECASTING SYSTEM
============================================================
✅ GPU configured: 1 GPU(s) available
📊 Loading realistic engine data...
✅ Loaded 604,800 records from 2025-01-01 00:00:00 to 2025-01-07 23:59:59
🎯 Training Primary Parameter Forecasting Models
...
✅ Forecasting pipeline completed successfully!
```

## Troubleshooting

### Common Issues

#### 1. TensorFlow Installation Issues
```bash
# Error: Could not load dynamic library 'libcudart.so.11.0'
# Solution: Install correct CUDA version
conda install cudatoolkit=11.2 cudnn=8.1.0 -c conda-forge
```

#### 2. Memory Issues
```bash
# Error: ResourceExhaustedError (OOM when allocating tensor)
# Solution: Reduce batch size or enable memory growth
export TF_FORCE_GPU_ALLOW_GROWTH=true
```

#### 3. Import Errors
```bash
# Error: No module named 'tensorflow'
# Solution: Ensure virtual environment is activated
conda activate ice-knocking
pip install tensorflow==2.13.0
```

#### 4. CUDA Compatibility Issues
```bash
# Check CUDA and TensorFlow compatibility
python -c "import tensorflow as tf; print(tf.test.is_built_with_cuda())"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

### Performance Optimization

#### 1. Enable Mixed Precision (Automatic in our code)
```python
# Already implemented in engine_parameter_forecaster.py
policy = tf.keras.mixed_precision.Policy('mixed_float16')
```

#### 2. Optimize Data Loading
```bash
# Set environment variables for better performance
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=4
```

#### 3. Memory Management
```bash
# For large datasets, consider using data generators
# Already implemented in our pipeline for efficiency
```

## Version Compatibility

### Tested Combinations

| Python | TensorFlow | CUDA | cuDNN | Status |
|--------|------------|------|-------|--------|
| 3.10   | 2.13.0     | 11.8 | 8.6   | ✅ Recommended |
| 3.9    | 2.13.0     | 11.2 | 8.1   | ✅ Supported |
| 3.8    | 2.13.0     | 11.2 | 8.1   | ✅ Minimum |

### Package Dependencies

The system has been tested with the specific versions in `requirements.txt`. While newer versions may work, we recommend using the specified versions for stability.

## Docker Installation (Alternative)

### Create Dockerfile
```dockerfile
FROM tensorflow/tensorflow:2.13.0-gpu-jupyter

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["python", "run_forecasting.py"]
```

### Build and Run
```bash
docker build -t ice-knocking .
docker run --gpus all -v $(pwd)/outputs:/app/outputs ice-knocking
```

## Development Setup

### For Contributors

1. **Install Development Dependencies**
   ```bash
   pip install -r requirements.txt
   pip install pre-commit black flake8 pytest
   ```

2. **Setup Pre-commit Hooks**
   ```bash
   pre-commit install
   ```

3. **Run Tests**
   ```bash
   pytest tests/
   ```

4. **Code Formatting**
   ```bash
   black src/
   flake8 src/
   ```

## Environment Variables

### Optional Configuration
```bash
# Performance tuning
export TF_CPP_MIN_LOG_LEVEL=2           # Reduce TensorFlow logging
export CUDA_VISIBLE_DEVICES=0          # Specify GPU device
export TF_FORCE_GPU_ALLOW_GROWTH=true  # Enable memory growth

# Model configuration
export ENGINE_SEQUENCE_LENGTH=3600      # 1 hour sequence length
export ENGINE_FORECAST_HORIZON=86400    # 1 day forecast horizon
```

## Support

### Getting Help

1. **Check the documentation**: README files in the repository
2. **Review error messages**: Most errors include helpful suggestions
3. **Check system requirements**: Ensure your system meets minimum requirements
4. **Verify installation**: Run the verification steps above

### Common Commands Summary

```bash
# Basic installation
pip install -r requirements.txt

# Check GPU
python run_forecasting.py --check-gpu

# Generate data
python src/realistic_engine_data_generator.py

# Train models
python run_forecasting.py --train

# Load existing models
python run_forecasting.py --load

# Auto-detect mode
python run_forecasting.py
```

## Next Steps

After successful installation:

1. **Generate realistic data**: `python src/realistic_engine_data_generator.py`
2. **Train forecasting models**: `python run_forecasting.py --train`
3. **Generate forecasts**: `python run_forecasting.py --load`
4. **Develop knock detection**: Use forecasted data for knock detection modeling

For detailed usage instructions, see the README files in the repository.
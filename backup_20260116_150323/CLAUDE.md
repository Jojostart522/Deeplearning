# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **MISO (Multi-Input Single-Output) FIR Wiener Filter** implementation for Active Noise Control (ANC) analysis. The project estimates microphone signals from accelerometer sensor data using Wiener filtering.

**Key Achievement**: Python implementation has been meticulously corrected to produce **identical results** to the original MATLAB implementation (BeforeANC: 28.81 dBA, AfterANC: 21.16 dBA, ANC reduction: -7.65 dBA).

## Quick Start

### Running the Analysis

```bash
python main_analysis.py
```

This will:
1. Load 32-channel sensor data from `data/260107_ddr_32x5_05.dat`
2. Apply A-weighting filter
3. Compute PSD for all channels
4. Calculate MISO Wiener filter (18 accelerometer inputs → 1 microphone target)
5. Generate result plots in `results/` directory

### Key Parameters

- **Sampling rate**: 3000 Hz
- **Data duration**: 5 seconds (15000 samples)
- **Filter length**: 300 (critical - must match MATLAB)
- **Input channels**: Accelerometers (Ch 7-24, indices 6-23, total 18 channels)
- **Target channel**: Microphone (Ch 25, index 24)

## Critical Implementation Details

### 1. MATLAB-Python Compatibility: The `reshape` Order Issue

**This is the most critical aspect of the codebase.** MATLAB's `reshape` uses **column-major (Fortran) order** by default, while NumPy uses **row-major (C) order**. All `reshape` operations MUST use `order='F'` to match MATLAB behavior.

**Correct pattern:**
```python
# WRONG - will produce different results
R = R.reshape((N+1, M, M))

# CORRECT - matches MATLAB
R = R.reshape((N+1, M, M), order='F')
```

**All reshape operations in the codebase use `order='F'`:**
- R matrix reshaping (3 times)
- P vector reshaping (2 times)
- B vector reshaping in Block Levinson algorithm (2 times)
- W vector reshaping for filter coefficient extraction (2 times)

### 2. A-Weighting Application Order

**Critical**: A-weighting is applied AFTER computing the Wiener filter, not before.

**Correct workflow:**
```python
# 1. Use ORIGINAL data to compute filter
input_signals = data[acc_indices, :]  # NOT data_weighted
target_signal = data[mic_index, :]    # NOT data_weighted

# 2. Compute filter and apply
filters = compute_wiener_filter(input_signals, target_signal, filter_length=300)
estimated_mic = apply_miso_filter(input_signals, filters)
error_signal = target_signal - estimated_mic

# 3. Apply A-weighting to RESULTS
target_signal_weighted = apply_a_weighting(target_signal)
error_signal_weighted = apply_a_weighting(error_signal)
```

### 3. Filter Length

Must be **exactly 300** to match MATLAB (`FL = 300`, where `N = FL - 1 = 299`).

## Code Architecture

### Main Analysis Pipeline (`main_analysis.py`)

The file is organized into clear sections:

1. **Constants** (lines 19-37): A-weighting filter coefficients
2. **Data I/O** (lines 43-68): Binary data loading
3. **Signal Processing** (lines 75-231): A-weighting, PSD, RMS, dB calculations
4. **Analysis Functions** (lines 238-299): Multi-channel PSD/dB computation
5. **Visualization** (lines 306-376): PSD plotting by sensor type
6. **Wiener Filter Core** (lines 383-620):
   - `compute_wiener_filter()`: Implements MATLAB's `miso_firwiener2.m` with Block Levinson algorithm
   - `apply_miso_filter()`: Applies computed filters to input signals
7. **Main Workflow** (lines 628-781): Orchestrates the complete analysis

### Key Functions

- **`compute_wiener_filter(inputs, target, filter_length)`**: Core algorithm that computes MISO FIR Wiener filter coefficients. Implements Block Levinson algorithm for solving the Wiener-Hopf equations. All reshape operations use `order='F'`.

- **`apply_miso_filter(inputs, filters)`**: Applies the computed filter bank to input signals using `signal.lfilter()` (equivalent to MATLAB's `filter()`).

- **`apply_a_weighting(data)`**: Applies A-weighting filter using predefined IIR coefficients.

- **`compute_psd(signal_data, fs, win_length, noverlap)`**: Computes power spectral density using Welch's method.

## Data Format

- **Input file**: Binary `.dat` file containing 32-channel float32 data
- **Layout**: Interleaved channels, reshaped to `(32, 15000)`
- **Channel mapping**:
  - Ch 1-6: Unused/other sensors
  - Ch 7-24: Accelerometers (18 channels)
  - Ch 25-26: Microphones
  - Ch 27-32: Unused/other sensors

## Dependencies

```python
numpy
scipy
matplotlib
```

## Verification Against MATLAB

The Python implementation has been verified against MATLAB by:

1. Saving intermediate matrices from MATLAB (R, P, B, W matrices)
2. Loading them in Python using `scipy.io.loadmat()`
3. Comparing numerical differences (all < 1e-6)
4. Verifying final dB values match exactly

See `MATLAB_Python_对比修正记录.md` for detailed correction history.

## Common Pitfalls When Modifying

1. **Never remove `order='F'` from any reshape operation** - this will break MATLAB compatibility
2. **Don't apply A-weighting before computing the filter** - it must be applied to results only
3. **Don't change filter length from 300** - this is calibrated to match MATLAB
4. **Index conversion**: Remember Python uses 0-based indexing while MATLAB uses 1-based
5. **Vector orientation**: MATLAB defaults to column vectors; ensure proper reshaping when converting

## Output Files

Generated in `results/` directory:
- `psd_accelerometers.png`: PSD plot for accelerometer channels (Ch 7-24)
- `psd_microphones.png`: PSD plot for microphone channels (Ch 25-26)
- `wiener_filter_result.png`: Time-domain and frequency-domain comparison of original vs. filtered signals

## Related MATLAB Code

The `Wiener/` subdirectory contains the original MATLAB implementation:
- `z_importBASE.m`: Main MATLAB script
- `miso_firwiener2.m`: Core Wiener filter algorithm (Block Levinson)
- `test_wiener.m`: Filter application and error computation
- `aweighting.m`: A-weighting filter
- `psdcomp.m`: PSD comparison and plotting

## Performance Notes

- The Block Levinson algorithm is used for solving large linear systems efficiently
- For 18 input channels and filter length 300, the R matrix is 5400×18
- Direct solve is attempted first; Block Levinson is fallback for ill-conditioned matrices

## Deep Learning Alternative

### Overview

In addition to the classical Wiener filter approach, this project includes a **deep learning implementation** (`dl_anc_model.py`) that uses neural networks to learn the mapping from accelerometer signals to microphone signals.

### Quick Start - Deep Learning

```bash
# Install additional dependencies
pip install -r requirements_dl.txt

# Train the deep learning model
python dl_anc_model.py

# Compare Wiener filter vs Deep Learning
python compare_methods.py
```

### Deep Learning Models

Two neural network architectures are available:

1. **CNN-LSTM Hybrid (default)**:
   - 1D convolutional layers for local feature extraction
   - LSTM layers for temporal dependency modeling
   - Best for capturing both spatial and temporal patterns

2. **TCN (Temporal Convolutional Network)**:
   - Causal and dilated convolutions
   - Large receptive field
   - Faster training, good for long-range dependencies

### Key Files

- **`dl_anc_model.py`**: Main deep learning implementation with training pipeline
- **`compare_methods.py`**: Side-by-side comparison of Wiener filter and deep learning
- **`DL_README.md`**: Detailed documentation for deep learning approach
- **`requirements_dl.txt`**: Additional dependencies (PyTorch)

### Training Configuration

Default parameters (can be adjusted in `dl_anc_model.py`):
- Window size: 300 samples (matches Wiener filter length)
- Stride: 10 (for sliding window data augmentation)
- Batch size: 64
- Epochs: 100
- Learning rate: 0.001
- Train/test split: 80/20

### Performance Target

The deep learning model aims to achieve **≤ -7.65 dBA** ANC reduction (matching or exceeding Wiener filter performance).

### Advantages of Deep Learning Approach

- **Non-linear modeling**: Can capture complex non-linear relationships
- **Adaptive**: Can potentially generalize better to different noise conditions
- **Feature learning**: Automatically learns relevant features from data

### Limitations

- **Data hungry**: Requires more training data than Wiener filter
- **Computational cost**: Slower training and inference
- **Less interpretable**: Black-box model vs. analytical Wiener solution

### Model Output

After training, the following files are generated in `results/`:
- `dl_anc_model.pth`: Trained model weights and normalization parameters
- `dl_training_curve.png`: Training and validation loss curves
- `dl_anc_result.png`: Time-domain and frequency-domain comparison
- `comparison_wiener_vs_dl.png`: Side-by-side comparison (from `compare_methods.py`)

### Using Pre-trained Model

```python
import torch
from dl_anc_model import ANCModel_CNN_LSTM, predict_full_signal

# Load model
checkpoint = torch.load('results/dl_anc_model.pth')
model = ANCModel_CNN_LSTM(num_channels=18, window_size=300)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Load normalization parameters
input_mean = checkpoint['input_mean']
input_std = checkpoint['input_std']

# Predict on new data
# new_input: (18, num_samples)
new_input_norm = (new_input - input_mean) / input_std
predictions = predict_full_signal(model, new_input_norm, window_size=300)
```

### Hyperparameter Tuning

To improve performance, try adjusting:
- `stride`: Smaller values (e.g., 1-5) create more training samples
- `num_epochs`: Increase for better convergence
- `hidden_size`: Increase LSTM hidden size (e.g., 256)
- Model architecture: Try TCN or add more layers
- Learning rate: Use learning rate scheduling

See `DL_README.md` for detailed optimization strategies.

# CUDA Control Flow: Warp Divergence & Occupancy Analysis

A GPU computing educational project that explores the performance impact of warp divergence and thread block size optimization on NVIDIA GPUs.

## Overview

This project demonstrates three critical CUDA optimization concepts:

- **Warp Divergence** - How conditional branching serializes execution within warps
- **Thread Block Size Optimization** - Impact of block dimensions on occupancy and performance
- **GPU Occupancy** - Maximizing resource utilization for latency hiding

## Project Structure

```
├── notebooks/
│   ├── A05_Warp_Divergence_Analysis.ipynb   # Main analysis notebook
│   └── a05.ipynb                            # Secondary notebook
├── reports/
│   └── analysis.md                          # Q&A analysis document
├── images/
│   ├── Block Size vs Execution.png          # Performance visualization
│   └── Warp Divergence Performance Penalty.png
└── README.md
```

## Prerequisites

- Google account (for Google Colab)
- Or local setup with:
  - Python 3.7+
  - NVIDIA GPU with CUDA support
  - CUDA Toolkit installed

## Running the Project

### Option 1: Google Colab (Recommended)

This project was developed and tested on Google Colab with GPU runtime.

1. Upload the notebook to Google Colab
2. Enable GPU runtime: **Runtime → Change runtime type → GPU**
3. Run cells sequentially with Shift+Enter

Colab provides free access to NVIDIA Tesla T4 GPUs with all required packages pre-installed.

### Option 2: Local Jupyter Notebook

```bash
# Install required packages
pip install numba numpy matplotlib pandas jupyter

# Verify CUDA installation
python3 -c "from numba import cuda; print(f'CUDA Available: {cuda.is_available()}')"

# Start Jupyter
jupyter notebook

# Open notebooks/A05_Warp_Divergence_Analysis.ipynb
```

## Experiments

### Experiment 1: Warp Divergence Testing

Compares execution time across three data patterns:

| Pattern | Description | Branch Behavior |
|---------|-------------|-----------------|
| Divergent (50/50) | Half above threshold | 50% each branch |
| Non-Divergent (All TRUE) | All above threshold | 100% same path |
| Non-Divergent (All FALSE) | All below threshold | 100% same path |

**Key Kernel:**
```python
@cuda.jit
def threshold_kernel(input_array, output_array, threshold):
    idx = cuda.grid(1)
    if idx < input_array.size:
        if input_array[idx] > threshold:
            output_array[idx] = 1.0  # TRUE branch
        else:
            output_array[idx] = 0.0  # FALSE branch
```

### Experiment 2: Block Size Impact on Performance

Tests block sizes: 32, 64, 128, 256, 512, 1024 threads per block

**Results Summary:**

| Block Size | Warps/Block | Divergent Time | Non-Div Time | Penalty |
|------------|-------------|----------------|--------------|---------|
| 32 | 1 | 0.1445 ms | 0.1409 ms | +2.48% |
| 64 | 2 | 0.1026 ms | 0.1032 ms | -0.58% |
| **128** | **4** | **0.0963 ms** | **0.0972 ms** | **-0.94%** |
| **256** | **8** | **0.0972 ms** | **0.0976 ms** | **-0.41%** |
| 512 | 16 | 0.1056 ms | 0.1061 ms | -0.47% |
| 1024 | 32 | 0.1329 ms | 0.1300 ms | +2.23% |

**Optimal Configuration:** 128-256 threads per block

## Key Findings

### Warp Divergence
- When threads within a warp take different branch paths, execution is serialized
- Both branches execute sequentially with inactive threads masked
- Maximum divergence penalty observed: ~2.5%

### Block Size Optimization
- **Too small (32):** High launch overhead, only 1 warp per block
- **Optimal (128-256):** Best balance of warps per block and total blocks
- **Too large (1024):** Fewer total blocks, potential load imbalance

### Occupancy & Latency Hiding
- Higher occupancy enables more warps to switch during memory stalls
- Tesla T4 with 40 SMs benefits from having sufficient warps to hide 200-800 cycle memory latencies

## Technical Configuration

```python
ARRAY_SIZE = 1024 * 1024    # 1M elements
THRESHOLD = ARRAY_SIZE / 2   # Creates 50/50 split
NUM_TIMING_RUNS = 10         # Statistical averaging
```

## Hardware Tested

- **GPU:** NVIDIA Tesla T4
- **Streaming Multiprocessors:** 40
- **Warp Size:** 32 threads

## Troubleshooting

### CUDA Not Found
```bash
# Ensure CUDA Toolkit is installed
nvidia-smi
nvcc --version
```

### Insufficient GPU Memory
Reduce `ARRAY_SIZE` in the notebook if encountering memory errors.

### Numba Compatibility
Ensure Numba version is compatible with your CUDA Toolkit version:
```bash
pip install --upgrade numba
```

## Author

**Kenneth Peter Fernandes**

- Course: CISC 701 - GPU Computing
- Assignment: A05 - Control Flow, Warp Divergence & Occupancy
- Term: Fall 2025

## License

This project is for educational purposes as part of coursework at Harrisburg University.

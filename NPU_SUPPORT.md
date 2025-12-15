# NPU Support Guide

This document describes the NPU (Neural Processing Unit) support implementation in RecBole.

## Overview

RecBole now fully supports NPU devices (primarily Huawei Ascend NPU) in addition to GPU and CPU. The implementation automatically detects and uses NPU when available, with fallback to GPU or CPU.

## Key Changes

### 1. Device Management Module
**File**: `recbole/utils/device_utils.py` (new file)

Provides unified device detection and management:
- Automatic device type detection (NPU > GPU > CPU)
- Environment variable setup (ASCEND_RT_VISIBLE_DEVICES for NPU, CUDA_VISIBLE_DEVICES for GPU)
- Device creation and management
- Distributed backend selection (HCCL for NPU, NCCL for GPU)

### 2. Configuration Module
**File**: `recbole/config/configurator.py`

- Uses `device_utils` for device initialization
- Automatically sets environment variables
- Selects correct distributed backend (hccl/nccl)

### 3. Trainer Module
**File**: `recbole/trainer/trainer.py`

- NPU-aware accelerator detection
- Supports NPU mixed precision training (torch.amp)
- Compatible with NPU GradScaler

### 4. Utility Functions
**File**: `recbole/utils/utils.py`

- NPU seed initialization support
- NPU memory usage query
- Device-aware environment detection

### 5. Model Code
- All models use `.to(device)` which automatically works with NPU
- Fixed hardcoded CUDA checks in `kd_dagfm.py`

## Usage

### Basic Usage (Auto-detection)
```python
from recbole.quick_start import run_recbole

# Automatically detects NPU (if available), otherwise uses GPU or CPU
run_recbole(model='BPR', dataset='ml-100k', config_file_list=['config.yaml'])
```

### Configuration File
```yaml
# config.yaml - No changes needed, uses existing format
gpu_id: '0'  # Will use NPU device 0 if NPU is available
use_gpu: True
```

## Installation

For Huawei Ascend NPU:
```bash
pip install torch-npu
```

## Environment Variables

The code automatically sets:
- NPU: `ASCEND_RT_VISIBLE_DEVICES`
- GPU: `CUDA_VISIBLE_DEVICES`

## Distributed Training

- NPU uses **HCCL** backend
- GPU uses **NCCL** backend
- Automatically selected based on device type

## Compatibility

- ✅ Fully backward compatible with GPU
- ✅ Fully backward compatible with CPU
- ✅ All 142+ models automatically support NPU
- ✅ Robust fallback logic if device_utils unavailable

## Modified Files

1. **New**: `recbole/utils/device_utils.py`
2. `recbole/config/configurator.py`
3. `recbole/trainer/trainer.py`
4. `recbole/utils/utils.py`
5. `recbole/model/context_aware_recommender/kd_dagfm.py`
6. `recbole/model/general_recommender/ncl.py` (comment update)
7. `recbole/model/knowledge_aware_recommender/kgat.py` (comment update)

## Testing

After installation, verify NPU availability:
```python
import torch_npu
print(torch_npu.npu.is_available())  # Should return True
```

Then run your training as usual - NPU will be automatically used if available.

## Notes

- Mixed precision training is fully supported on NPU
- All model code uses device-agnostic methods (`.to(device)`)
- No configuration file changes required


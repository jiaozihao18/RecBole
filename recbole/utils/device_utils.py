# @Time   : 2024
# @Author : NPU Support
# @Description: Device utility functions for GPU and NPU support

"""
Device utility module for supporting both GPU and NPU devices.
"""

import torch
import os

# Try to import torch_npu for Huawei Ascend NPU support
try:
    import torch_npu
    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False
    torch_npu = None


def get_device_type():
    """
    Detect available device type (npu, cuda, or cpu).
    
    Returns:
        str: Device type ('npu', 'cuda', or 'cpu')
    """
    if NPU_AVAILABLE and torch_npu.npu.is_available():
        return 'npu'
    elif torch.cuda.is_available():
        return 'cuda'
    else:
        return 'cpu'


def is_accelerator_available():
    """
    Check if any accelerator (GPU or NPU) is available.
    
    Returns:
        tuple: (bool, str) - (is_available, device_type)
    """
    device_type = get_device_type()
    return device_type != 'cpu', device_type


def set_device_visible_devices(device_ids, device_type=None):
    """
    Set visible devices environment variable.
    
    Args:
        device_ids (str): Device IDs string (e.g., "0,1,2")
        device_type (str): Device type ('npu' or 'cuda'). If None, auto-detect.
    """
    if device_type is None:
        device_type = get_device_type()
    
    if device_type == 'npu':
        # Huawei Ascend NPU uses ASCEND_RT_VISIBLE_DEVICES
        os.environ["ASCEND_RT_VISIBLE_DEVICES"] = device_ids
    elif device_type == 'cuda':
        os.environ["CUDA_VISIBLE_DEVICES"] = device_ids
    # For CPU, no environment variable needed


def create_device(device_type=None, device_id=None):
    """
    Create a torch device object.
    
    Args:
        device_type (str): Device type ('npu', 'cuda', or 'cpu'). If None, auto-detect.
        device_id (int): Device ID for the specific device. If None, use default.
    
    Returns:
        torch.device: The device object
    """
    if device_type is None:
        device_type = get_device_type()
    
    if device_type == 'cpu':
        return torch.device('cpu')
    elif device_type == 'npu':
        if device_id is not None:
            return torch.device(f'npu:{device_id}')
        else:
            return torch.device('npu')
    elif device_type == 'cuda':
        if device_id is not None:
            return torch.device(f'cuda:{device_id}')
        else:
            return torch.device('cuda')
    else:
        return torch.device('cpu')


def set_current_device(device_id, device_type=None):
    """
    Set current device for the process.
    
    Args:
        device_id (int): Device ID
        device_type (str): Device type ('npu' or 'cuda'). If None, auto-detect.
    """
    if device_type is None:
        device_type = get_device_type()
    
    if device_type == 'npu' and NPU_AVAILABLE:
        torch_npu.npu.set_device(device_id)
    elif device_type == 'cuda':
        torch.cuda.set_device(device_id)


def get_distributed_backend(device_type=None):
    """
    Get distributed backend name for the device type.
    
    Args:
        device_type (str): Device type ('npu' or 'cuda'). If None, auto-detect.
    
    Returns:
        str: Backend name ('hccl' for NPU, 'nccl' for CUDA)
    """
    if device_type is None:
        device_type = get_device_type()
    
    if device_type == 'npu':
        return 'hccl'  # Huawei Communication Control Library
    elif device_type == 'cuda':
        return 'nccl'  # NVIDIA Collective Communications Library
    else:
        return 'gloo'  # Default backend for CPU


def is_device_available(device_type=None):
    """
    Check if the specified device type is available.
    
    Args:
        device_type (str): Device type to check ('npu' or 'cuda'). If None, auto-detect.
    
    Returns:
        bool: True if device is available
    """
    if device_type is None:
        device_type = get_device_type()
    
    if device_type == 'npu':
        return NPU_AVAILABLE and torch_npu.npu.is_available()
    elif device_type == 'cuda':
        return torch.cuda.is_available()
    else:
        return False


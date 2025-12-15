# NPU支持快速修改指南

## 核心修改点总结

### 1. 设备初始化 (`recbole/config/configurator.py`)

**位置**: `_init_device()` 方法 (约476-515行)

**需要修改的地方**:
- ✅ `CUDA_VISIBLE_DEVICES` → 使用`set_device_visible_devices()`自动适配
- ✅ `torch.cuda.is_available()` → `is_device_available(device_type)`
- ✅ `torch.device("cuda")` → `create_device(device_type)`
- ✅ `backend="nccl"` → `backend=get_distributed_backend(device_type)` (hccl for NPU)
- ✅ `torch.cuda.set_device()` → `set_current_device(..., device_type)`

### 2. Trainer设备检查 (`recbole/trainer/trainer.py`)

**位置**: `__init__` 方法 (约126-130行)

**需要修改的地方**:
- ✅ `torch.cuda.is_available()` → `is_device_available(device_type)`
- ✅ `import torch.cuda.amp` → `from torch.amp import GradScaler`

### 3. 工具函数 (`recbole/utils/utils.py`)

**位置1**: `init_seed()` 方法 (约188-205行)
- ✅ 添加NPU随机种子设置
- ✅ CUDNN设置仅对CUDA生效

**位置2**: `get_gpu_usage()` 方法 (约235-247行)
- ✅ 添加NPU内存查询支持

**位置3**: `get_environment()` 方法 (约418-422行)
- ✅ 使用设备工具函数检查加速器

### 4. 新增工具模块 (`recbole/utils/device_utils.py`)

**新增文件**: 提供统一的设备管理接口
- ✅ 自动检测设备类型（npu/cuda/cpu）
- ✅ 统一的环境变量设置
- ✅ 统一的设备创建接口
- ✅ 分布式backend选择

## 代码变化对比

### 设备初始化对比

**修改前 (GPU only)**:
```python
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
if torch.cuda.is_available():
    device = torch.device("cuda")
backend = "nccl"
torch.cuda.set_device(local_rank)
```

**修改后 (GPU + NPU)**:
```python
device_type = get_device_type()  # 'npu', 'cuda', or 'cpu'
set_device_visible_devices(gpu_id, device_type)
if is_device_available(device_type):
    device = create_device(device_type)
backend = get_distributed_backend(device_type)  # 'hccl' for NPU, 'nccl' for CUDA
set_current_device(local_rank, device_type)
```

### 设备检查对比

**修改前**:
```python
gpu_available = torch.cuda.is_available() and config["use_gpu"]
```

**修改后**:
```python
device_type = get_device_type()
accelerator_available = is_device_available(device_type) and config["use_gpu"]
```

### 混合精度对比

**修改前**:
```python
import torch.cuda.amp as amp
scaler = amp.GradScaler()
with amp.autocast():
    ...
```

**修改后**:
```python
from torch.amp import GradScaler
scaler = GradScaler()
with torch.autocast(device_type=device.type):  # 自动适配npu或cuda
    ...
```

## 环境变量映射

| 设备类型 | 环境变量 | Backend |
|---------|---------|---------|
| CUDA    | CUDA_VISIBLE_DEVICES | nccl |
| NPU (昇腾) | ASCEND_RT_VISIBLE_DEVICES | hccl |
| CPU     | - | gloo |

## 验证清单

- [ ] 安装torch-npu包
- [ ] 验证NPU是否可用: `python -c "import torch_npu; print(torch_npu.npu.is_available())"`
- [ ] 运行单卡训练测试
- [ ] 运行多卡分布式训练测试（如果支持）
- [ ] 测试混合精度训练
- [ ] 验证模型保存和加载
- [ ] 检查内存使用情况显示

## 常见问题

**Q: 如何强制使用GPU而不是NPU？**
A: 如果NPU可用，代码会优先使用NPU。可以通过环境变量禁用NPU或卸载torch-npu。

**Q: 如何确认代码使用的是NPU？**
A: 检查`config["device"]`，应该显示`npu`或`npu:0`。

**Q: 分布式训练是否支持？**
A: 支持，代码会自动使用HCCL backend（华为NPU）或NCCL backend（NVIDIA GPU）。

**Q: 其他NPU厂商支持吗？**
A: 当前代码主要针对华为昇腾NPU。如需支持其他NPU，需要修改`device_utils.py`中的相关函数。


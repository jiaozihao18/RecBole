# RecBole GPU到NPU迁移说明

本文档说明如何将RecBole代码从GPU支持修改为NPU支持。

## 需要修改的主要文件

### 1. `recbole/config/configurator.py` - 设备初始化（最重要）

**修改位置**: `_init_device()` 方法 (约476-515行)

**需要修改的内容**:
- `CUDA_VISIBLE_DEVICES` 环境变量 → `ASCEND_RT_VISIBLE_DEVICES`（华为NPU）或对应的NPU环境变量
- `torch.cuda.is_available()` → `torch_npu.npu.is_available()`（需要先导入torch_npu）
- `torch.device("cuda")` → `torch.device("npu")`
- `torch.cuda.set_device()` → `torch_npu.npu.set_device()`
- 分布式训练backend: `"nccl"` → `"hccl"`（华为NPU使用HCCL）

### 2. `recbole/trainer/trainer.py` - Trainer类

**修改位置1**: `__init__` 方法 (约126-130行)
- `torch.cuda.is_available()` → `torch_npu.npu.is_available()`

**修改位置2**: `_init_amp_scaler` 相关代码（如果存在）
- 混合精度训练的device_type可能需要调整

**注意**: `torch.autocast(device_type=self.device.type)` 应该自动适配，但需要确保device.type为"npu"

### 3. `recbole/utils/utils.py` - 工具函数

**修改位置1**: `init_seed()` 方法 (约198-199行)
- `torch.cuda.manual_seed()` 和 `torch.cuda.manual_seed_all()` → NPU对应的随机种子设置方法
- `torch.backends.cudnn` → 可能需要移除或替换（NPU不需要cudnn）

**修改位置2**: `get_gpu_usage()` 方法 (约235-247行)
- `torch.cuda.max_memory_reserved()` → `torch_npu.npu.max_memory_reserved()`（如果支持）
- `torch.cuda.get_device_properties()` → `torch_npu.npu.get_device_properties()`（如果支持）
- 函数名可以改为 `get_npu_usage()` 或保持兼容性

**修改位置3**: `get_environment()` 方法 (约421行)
- `torch.cuda.is_available()` → `torch_npu.npu.is_available()`

### 4. 配置文件修改

**`recbole/properties/overall.yaml`**
- `gpu_id` 参数名可以保持不变（兼容性），或添加 `npu_id` 参数

## 具体修改步骤

### 步骤1: 在文件开头添加NPU支持检查

在相关文件顶部添加：
```python
try:
    import torch_npu
    NPU_AVAILABLE = True
except ImportError:
    NPU_AVAILABLE = False
```

### 步骤2: 创建设备类型抽象

建议创建一个统一的设备检测函数，例如：
```python
def is_accelerator_available():
    """检查可用的加速器（GPU或NPU）"""
    if NPU_AVAILABLE and torch_npu.npu.is_available():
        return True, 'npu'
    elif torch.cuda.is_available():
        return True, 'cuda'
    else:
        return False, 'cpu'
```

### 步骤3: 环境变量映射

根据不同的NPU厂商，环境变量可能不同：
- 华为昇腾NPU: `ASCEND_RT_VISIBLE_DEVICES`
- 其他NPU: 请参考对应厂商文档

## 注意事项

1. **混合精度训练**: NPU的混合精度支持可能与CUDA不同，需要验证AMP是否正常工作

2. **分布式训练**: 
   - NCCL → HCCL（华为NPU）
   - 其他NPU厂商可能有不同的backend

3. **依赖安装**: 需要安装对应NPU的PyTorch插件
   - 华为昇腾: `torch-npu`
   - 其他NPU: 参考厂商文档

4. **测试**: 修改后需要充分测试：
   - 单卡训练
   - 多卡分布式训练（如果支持）
   - 混合精度训练
   - 模型保存和加载

5. **向后兼容**: 建议保持对GPU的支持，通过配置或环境变量切换

## 推荐修改策略

**方案1: 条件编译**（推荐）
- 根据环境变量或配置自动选择GPU或NPU
- 保持代码兼容性

**方案2: 完全替换**
- 将所有GPU相关代码替换为NPU
- 仅支持NPU运行

建议采用方案1，这样可以同时支持GPU和NPU。


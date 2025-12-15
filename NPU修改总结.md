# RecBole GPU到NPU迁移修改总结

## 已完成的修改

### 1. 新增文件

#### `recbole/utils/device_utils.py` (新增)
这是一个新的工具模块，提供了统一的设备管理接口，支持GPU和NPU的自动检测和切换。

**主要功能**:
- `get_device_type()`: 自动检测可用的设备类型（npu/cuda/cpu）
- `is_device_available()`: 检查指定设备是否可用
- `set_device_visible_devices()`: 设置可见设备环境变量
- `create_device()`: 创建torch.device对象
- `set_current_device()`: 设置当前进程的设备
- `get_distributed_backend()`: 获取分布式训练backend（nccl/hccl）

### 2. 修改的文件

#### `recbole/config/configurator.py`

**修改内容**:
- 导入新的`device_utils`模块
- 修改`_init_device()`方法:
  - 使用`get_device_type()`自动检测设备类型
  - 使用`set_device_visible_devices()`设置环境变量（支持CUDA和NPU）
  - 使用`create_device()`创建设备对象
  - 使用`get_distributed_backend()`获取正确的backend（nccl → hccl for NPU）
  - 使用`set_current_device()`设置当前设备

**关键变更**:
```python
# 原代码：
os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
torch.device("cuda")
backend="nccl"
torch.cuda.set_device(...)

# 修改后：
set_device_visible_devices(gpu_id, device_type)  # 自动适配CUDA/NPU
create_device(device_type)  # 自动创建npu或cuda设备
backend=get_distributed_backend(device_type)  # nccl for CUDA, hccl for NPU
set_current_device(..., device_type)  # 自动适配
```

#### `recbole/trainer/trainer.py`

**修改内容**:
- 修改`__init__`方法中的设备检查逻辑
- 将`torch.cuda.is_available()`改为使用`device_utils`的检查函数
- 修改混合精度训练的导入：
  - 从`torch.cuda.amp`改为使用`torch.amp`（兼容NPU和CUDA）
  - 保持`GradScaler`的使用方式

**关键变更**:
```python
# 原代码：
self.gpu_available = torch.cuda.is_available() and config["use_gpu"]
import torch.cuda.amp as amp

# 修改后：
accelerator_available = is_device_available(device_type) and config["use_gpu"]
from torch.amp import GradScaler  # 兼容CUDA和NPU
```

#### `recbole/utils/utils.py`

**修改内容**:
1. **`init_seed()`函数**:
   - 添加NPU的随机种子设置
   - 保留CUDA的随机种子设置
   - CUDNN设置仅在CUDA可用时生效

2. **`get_gpu_usage()`函数**:
   - 添加NPU内存查询支持
   - 根据设备类型自动选择查询方法（CUDA或NPU）

3. **`get_environment()`函数**:
   - 使用`device_utils`检查加速器可用性
   - 支持GPU和NPU的设备使用情况显示

**关键变更**:
```python
# init_seed - 添加NPU支持
if torch_npu.npu.is_available():
    torch_npu.npu.manual_seed(seed)

# get_gpu_usage - 添加NPU内存查询
if device_type == 'npu':
    reserved = torch_npu.npu.max_memory_reserved(device) / 1024**3
    total = torch_npu.npu.get_device_properties(device).total_memory / 1024**3
```

## 使用方法

### 1. 安装NPU支持

对于华为昇腾NPU，需要安装torch-npu：
```bash
# 根据您的PyTorch版本和NPU型号安装对应的torch-npu包
pip install torch-npu
```

### 2. 配置使用

代码会自动检测可用的设备类型。如果有NPU可用，将优先使用NPU；否则使用GPU；如果都没有，则使用CPU。

配置文件中的`gpu_id`参数可以继续使用（保持向后兼容），用于指定设备ID：
```yaml
gpu_id: '0'  # 使用设备0（可以是GPU或NPU）
use_gpu: True
```

### 3. 环境变量

代码会自动设置正确的环境变量：
- CUDA: `CUDA_VISIBLE_DEVICES`
- NPU (华为昇腾): `ASCEND_RT_VISIBLE_DEVICES`

## 注意事项

1. **分布式训练**: 
   - GPU使用NCCL backend
   - 华为NPU使用HCCL backend
   - 代码会自动选择合适的backend

2. **混合精度训练**: 
   - NPU支持torch.amp的autocast和GradScaler
   - device_type参数会自动设置为'npu'

3. **向后兼容性**: 
   - 代码保持了对GPU的完全支持
   - 如果NPU不可用，会自动回退到GPU或CPU
   - 配置文件格式无需修改

4. **其他NPU厂商**: 
   - 如果需要支持其他NPU（非华为昇腾），需要修改`device_utils.py`中的环境变量和API调用
   - 主要需要适配：
     - 环境变量名称
     - 设备检查API
     - 内存查询API
     - 分布式backend名称

## 测试建议

修改后建议测试以下场景：

1. **单卡训练**:
   ```bash
   python run_recbole.py --model=BPR --dataset=ml-100k --gpu_id=0
   ```

2. **多卡分布式训练**:
   ```bash
   python run_recbole.py --model=BPR --dataset=ml-100k --gpu_id=0,1,2,3 --nproc=4
   ```

3. **混合精度训练**:
   在配置文件中设置`enable_amp: True`

4. **模型保存和加载**:
   验证checkpoint的保存和加载是否正常

## 可能需要的额外修改

如果在实际使用中发现以下问题，可能需要额外修改：

1. **模型特定代码**: 某些模型可能包含硬编码的CUDA调用，需要检查并修改
2. **自定义Trainer**: 如果有自定义的Trainer类，可能需要类似的修改
3. **数据加载**: DataLoader的num_workers设置可能需要针对NPU调整
4. **性能优化**: NPU可能需要不同的优化策略（如算子融合等）

## 参考文档

- 华为昇腾NPU PyTorch支持: https://www.hiascend.com/document
- PyTorch AMP文档: https://pytorch.org/docs/stable/amp.html


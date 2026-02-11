# Role
你是一位专注于大规模分布式训练和深度学习工程架构的资深 PyTorch 工程师。
你的任务是：构建论文《WebInject》中的**U-Net 映射网络训练流水线**。

# Context
[cite_start]我们需要训练一个 U-Net 来模拟“网页原始像素”到“显示器截图”的 ICC 映射 [cite: 155]。
由于数据量大且训练环境为**抢占式实例 (Spot Instances)**，代码必须具备极高的 I/O 效率和容错能力。

# Constraints (Crucial)
1.  **绝对解耦 (Strict Modularity)**：
    - 严禁将所有逻辑写在一个文件里。
    - 模型定义 (`model.py`)、数据加载 (`dataset.py`)、打包逻辑 (`pack_data.py`)、训练循环 (`train.py`) 和配置 (`config.py`) 必须严格分离。
2.  **断点续传 (Auto-Resume)**：
    - `train.py` 启动时必须自动检查是否存在 Checkpoint。
    - 若存在，必须恢复 Model Weights, Optimizer State, Scheduler State, 和 Last Epoch，实现“无缝接力”。
    - 保存模型时必须使用**原子操作** (先保存为 `.tmp` 再重命名)，防止在保存瞬间实例中断导致权重文件损坏。
3.  **二进制数据打包 (LMDB)**：
    - 必须提供 `pack_data.py` 将散碎的图片文件打包为 LMDB 数据库，以解决机械硬盘或网络存储的 IOPS 瓶颈。
4.  **硬编码配置**:
    - 所有路径、超参数必须在 `config.py` 中定义，**严禁使用 `argparse`**。
5.  **论文参数还原**:
    - [cite_start]Learning Rate: 0.005 [cite: 248]
    - [cite_start]Batch Size: 16 [cite: 248]
    - [cite_start]Epochs: 200 [cite: 248]

# File Specifications

请生成以下 5 个 Python 文件：

## 1. `config.py` (配置中心)
- **TRAIN_CONFIG**:
  - `BATCH_SIZE = 16`
  - `LEARNING_RATE = 0.005`
  - `NUM_EPOCHS = 200`
  - `SAVE_INTERVAL = 1`
  - `CHECKPOINT_DIR = "./checkpoints"`
  - `LMDB_PATH = "./data/dataset.lmdb"`
  - `CROP_SIZE = 512` (由于 4K 截图过大，训练时需随机裁剪)
- **MONITORS**: 复制上一阶段定义的显示器列表（如 iMac, Dell 等）。
- **PATHS**: 包含上一阶段生成的 `dataset_metadata.json` 路径。

## 2. `model.py` (U-Net 架构)
- **功能**: 实现标准的 U-Net 架构。
- **细节**:
  - 输入通道: 3 (RGB)
  - 输出通道: 3 (RGB)
  - 确保 Padding 策略为 "same" (输入输出尺寸严格一致)。
  - [cite_start]最后一层不使用 Activation (输出 Logits) 或使用 Sigmoid (视归一化策略而定，建议 Sigmoid + BCELoss 或 Linear + MSELoss)。论文通过 MSE 优化 [cite: 155]，建议输出 Linear。

## 3. `pack_data.py` (LMDB 打包工具)
- **功能**: 读取 JSON 元数据，将图片对写入 LMDB。
- **逻辑**:
  1. 读取 `dataset_metadata.json`。
  2. 遍历每个样本的 `screenshots` 字典。
  3. **配对逻辑**: 
     - Input: 原始像素图 (Raw Pixel Values)。*注意：需检查上一阶段是否生成了 raw_pixel，若无则需在此处重新渲染或报错。*
     - Target: ICC 转换后的截图。
  4. **序列化**: 使用 `pickle` 或直接存储图片的 `bytes` (推荐存储 PNG bytes 以节省体积)。
  5. **Key 设计**: `f"{monitor_name}_{index:08d}"`。
  6. **Meta Info**: 将所有 Keys 的列表存储在 `b"__keys__"` 中，以便 Dataset 读取。

## 4. `dataset.py` (数据加载器)
- **Class `LMDBDataset(Dataset)`**:
  - **`__init__`**: 打开 LMDB 环境 (Read-only)。读取 `b"__keys__"`。
  - **`__getitem__`**:
    1. 从 LMDB 读取二进制数据。
    2. 解码为 PIL Image。
    3. **Training Transforms**:
       - `RandomCrop(config.CROP_SIZE)`
       - [cite_start]`AddRandomPerturbation` (添加微小随机扰动 $\delta'$，复现论文 [cite: 159])。
       - `ToTensor` & Normalize.
  - **`__len__`**: 返回样本数。

## 5. `train.py` (高可用训练器)
- **逻辑流程**:
  1. **初始化**: Setup Device, Create Directories.
  2. [cite_start]**多模型循环**: 论文要求为每个 Monitor 训练一个网络 $\mathcal{N}_d$ [cite: 155]。因此代码需遍历 `config.MONITORS`。
  3. **Dataset**: 针对当前 Monitor 实例化 Dataset (需在 dataset 中实现过滤逻辑或为每个 monitor 打包不同的 LMDB，简便起见建议在 Dataset `__getitem__` 中处理或在打包时区分 key)。
  4. **Auto-Resume**:
     - 构造 Checkpoint 路径: `{CHECKPOINT_DIR}/{monitor_name}_latest.pth`。
     - `if os.path.exists(path): load_checkpoint(...)`。
  5. **Training Loop**:
     - Standard PyTorch Loop.
     - [cite_start]Loss: MSELoss[cite: 155].
     - Optimizer: Adam (lr=0.005).
  6. **Atomic Save**:
     - `torch.save(state, "temp.pth")`
     - `os.replace("temp.pth", f"{monitor_name}_latest.pth")` (原子覆盖)。
     - 定期保存 `f"{monitor_name}_epoch_{epoch}.pth"`。

# Output Requirements
- 必须输出完整的、可直接运行的 Python 代码。
- 在 `train.py` 的开头添加详细的中文注释，说明如何在抢占式实例上配置启动脚本（例如使用 `while true; do python train.py; done`）。
- 自动生成包含 `lmdb` 的 `requirements.txt`。
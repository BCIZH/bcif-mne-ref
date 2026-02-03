# BCIF-NN 模块设计文档

> **bcif-nn**: BCIF 框架的神经网络插件模块
> 支持 EEGNet、ShallowConvNet 等轻量级 BCI 分类模型的训练与推理

---

## 目录

1. [背景调研](#1-背景调研)
2. [EEGNet 架构详解](#2-eegnet-架构详解)
3. [输入输出规格](#3-输入输出规格)
4. [BCIF 数据转换](#4-bcif-数据转换)
5. [模块架构设计](#5-模块架构设计)
6. [核心 Trait 设计](#6-核心-trait-设计)
7. [EEGNet 实现设计](#7-eegnet-实现设计)
8. [训练接口设计](#8-训练接口设计)
9. [完整使用流程](#9-完整使用流程)
10. [实现路径建议](#10-实现路径建议)

---

## 1. 背景调研

### 1.1 MNE-LSL 现状

**mne-lsl** 是 MNE-Python 的实时流处理扩展，但**不包含任何神经网络实现**。

```
┌─────────────────────────────────────────────────────────┐
│                    mne-lsl 架构                          │
├─────────────────────────────────────────────────────────┤
│  lsl/                                                   │
│  ├── StreamInlet   ← 接收 LSL 数据流                    │
│  ├── StreamOutlet  ← 发送 LSL 数据流                    │
│  └── StreamInfo    ← 流元数据                           │
├─────────────────────────────────────────────────────────┤
│  stream/                                                │
│  ├── StreamLSL     ← 高级连续数据流接口                 │
│  ├── EpochsStream  ← 事件相关的 Epoch 提取              │
│  └── _filters.py   ← 实时 IIR 滤波                      │
├─────────────────────────────────────────────────────────┤
│  player/                                                │
│  └── PlayerLSL     ← 回放 MNE Raw 文件为 LSL 流         │
├─────────────────────────────────────────────────────────┤
│  examples/                                              │
│  ├── bandpower.py      ← 带通功率特征提取               │
│  ├── decode.py         ← LogisticRegression (sklearn)   │
│  └── peak_detection.py ← R波检测                        │
└─────────────────────────────────────────────────────────┘
```

### 1.2 为什��需要 bcif-nn

- MNE/mne-lsl 不提供深度学习模型
- EEGNet 等轻量级 CNN 适合边缘 AI 部署
- 需要与 BCIF 预处理流水线无缝对接

---

## 2. EEGNet 架构详解

**EEGNet** 是由 Army Research Laboratory 提出的轻量级 EEG 分类模型（Lawhern et al., 2018），不属于 MNE 生态系统。

### 2.1 网络结构

```
输入: [batch, 1, channels, samples]
      例如: [32, 1, 64, 128]  (64通道, 128采样点)

┌─────────────────────────────────────────────────────────┐
│ Block 1: Temporal Convolution                           │
│ ├── Conv2D(F1, (1, kernel_length))  # 时间卷积          │
│ ├── BatchNorm2D                                         │
│ └── 输出: [batch, F1, channels, samples]                │
├─────────────────────────────────────────────────────────┤
│ Block 2: Depthwise Convolution (空间滤波)               │
│ ├── DepthwiseConv2D((channels, 1), depth_multiplier=D)  │
│ ├── BatchNorm2D                                         │
│ ├── ELU activation                                      │
│ ├── AvgPool2D((1, 4))                                   │
│ ├── Dropout(p)                                          │
│ └── 输出: [batch, F1*D, 1, samples/4]                   │
├─────────────────────────────────────────────────────────┤
│ Block 3: Separable Convolution                          │
│ ├── SeparableConv2D(F2, (1, 16))                        │
│ ├── BatchNorm2D                                         │
│ ├── ELU activation                                      │
│ ├── AvgPool2D((1, 8))                                   │
│ ├── Dropout(p)                                          │
│ └── 输出: [batch, F2, 1, samples/32]                    │
├─────────────────────────────────────────────────────────┤
│ Block 4: Classification                                 │
│ ├── Flatten                                             │
│ └── Dense(n_classes, softmax)                           │
└─────────────────────────────────────────────────────────┘
```

### 2.2 典型超参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| F1 | 8 | 第一层时间滤波器数 |
| D | 2 | 深度乘数 |
| F2 | F1 × D = 16 | 第三层滤波器数 |
| kernel_length | 64 | 时间卷积核长度 (对应 250ms @ 256Hz) |
| dropout | 0.5 | Dropout 概率 |

---

## 3. 输入输出规格

### 3.1 输入张量

```
Shape: [batch_size, 1, n_channels, n_samples]

示例 (Motor Imagery 4-class):
┌─────────────────────────────────────────────────────────┐
│ batch_size  = 32      # 批次大小                        │
│ 1           = 1       # 固定为1 (模拟图像的单通道)      │
│ n_channels  = 22      # EEG 电极数 (如 BCI Competition) │
│ n_samples   = 1000    # 采样点 (4秒 × 250Hz)            │
└─────────────────────────────────────────────────────────┘

数据类型: float32
数值范围: 建议标准化到 [-1, 1] 或 z-score
```

### 3.2 输出张量

```
Shape: [batch_size, n_classes]

示例:
┌─────────────────────────────────────────────────────────┐
│ [0.05, 0.85, 0.07, 0.03]  # 4类概率 (softmax 输出)      │
│  ↑      ↑     ↑     ↑                                   │
│ Left  Right  Feet Tongue                                │
└─────────────────────────────────────────────────────────┘
```

---

## 4. BCIF 数据转换

### 4.1 数据流转换

```
BCIF Layer 1: Epochs
┌─────────────────────────────────────────────────────────┐
│ struct Epochs {                                         │
│     data: Array3<f64>,  // [n_epochs, n_channels, n_samples]
│     info: Info,         // 元数据                       │
│     events: Array2<i32>,// [n_events, 3]                │
│     labels: Vec<i32>,   // 每个 epoch 的标签            │
│ }                                                       │
└─────────────────────────────────────────────────────────┘
                            ↓
                   【转换函数】
                            ↓
bcif-nn 输入格式
┌─────────────────────────────────────────────────────────┐
│ struct NNInput {                                        │
│     data: Array4<f32>,  // [batch, 1, channels, samples]│
│     labels: Array1<i64>,// [batch] class index          │
│ }                                                       │
└─────────────────────────────────────────────────────────┘
```

### 4.2 转换代码

```rust
/// Convert BCIF Epochs to neural network input format.
/// 将 BCIF Epochs 转换为神经网络输入格式。
fn epochs_to_nn_input(epochs: &Epochs) -> NNInput {
    let (n_epochs, n_ch, n_samples) = epochs.data.dim();

    // 1. f64 → f32 类型转换
    // 2. 添加维度 [N,C,T] → [N,1,C,T]
    // 3. 标准化 (z-score per channel)

    let data_f32 = epochs.data.mapv(|x| x as f32);
    let data_4d = data_f32
        .insert_axis(Axis(1));  // [N,C,T] → [N,1,C,T]

    NNInput {
        data: standardize_per_channel(data_4d),
        labels: Array1::from_vec(epochs.labels.clone()),
    }
}
```

---

## 5. 模块架构设计

```
bcif-nn/
├── Cargo.toml
├── src/
│   ├── lib.rs              # 模块入口
│   │
│   ├── data/               # 数据处理层
│   │   ├── mod.rs
│   │   ├── dataset.rs      # Dataset trait + EEGDataset
│   │   ├── transform.rs    # 数据增强 (时移/噪声/缩放)
│   │   └── loader.rs       # DataLoader (批次/打乱/并行)
│   │
│   ├── models/             # 神经网络模型
│   │   ├── mod.rs
│   │   ├── eegnet.rs       # EEGNet 实现
│   │   ├── shallownet.rs   # ShallowConvNet
│   │   ├── deepnet.rs      # DeepConvNet
│   │   └── traits.rs       # Model trait 定义
│   │
│   ├── layers/             # 神经网络层 (可选自实现)
│   │   ├── mod.rs
│   │   ├── conv2d.rs       # 2D 卷积
│   │   ├── depthwise.rs    # Depthwise 卷积
│   │   ├── separable.rs    # Separable 卷积
│   │   ├── batchnorm.rs    # Batch Normalization
│   │   ├── dropout.rs      # Dropout
│   │   ├── pooling.rs      # AvgPool / MaxPool
│   │   └── activation.rs   # ELU / ReLU / Softmax
│   │
│   ├── training/           # 训练逻辑
│   │   ├── mod.rs
│   │   ├── trainer.rs      # Trainer 主循环
│   │   ├── optimizer.rs    # Adam / SGD
│   │   ├── loss.rs         # CrossEntropy / NLLLoss
│   │   ├── metrics.rs      # Accuracy / F1 / Kappa
│   │   └── callbacks.rs    # EarlyStopping / Checkpoint
│   │
│   ├── inference/          # 推理逻辑
│   │   ├── mod.rs
│   │   ├── predictor.rs    # 批量/单样本预测
│   │   ├── onnx.rs         # ONNX 模型加载
│   │   └── quantized.rs    # 量化推理 (嵌入式)
│   │
│   └── export/             # 模型导出
│       ├── mod.rs
│       ├── onnx.rs         # 导出为 ONNX
│       └── weights.rs      # 自定义权重格式
│
└── examples/
    ├── train_eegnet.rs     # 训练示例
    ├── inference_onnx.rs   # ONNX 推理示例
    └── realtime_bci.rs     # 实时 BCI 示例
```

---

## 6. 核心 Trait 设计

### 6.1 Model Trait

```rust
// bcif-nn/src/models/traits.rs
// 神经网络模型统一接口
// Unified interface for neural network models

/// Neural network model trait.
/// 神经网络模型 trait。
pub trait Model: Send + Sync {
    /// Forward pass for inference.
    /// 前向传播（推理）。
    fn forward(&self, input: &Array4<f32>) -> Array2<f32>;

    /// Forward pass for training (returns intermediate activations).
    /// 前向传播（训练，返回中间激活值）。
    fn forward_train(&mut self, input: &Array4<f32>) -> (Array2<f32>, Cache);

    /// Backward pass (compute gradients).
    /// 反向传播（计算梯度）。
    fn backward(&mut self, grad_output: &Array2<f32>, cache: &Cache) -> Gradients;

    /// Get model parameters.
    /// 获取模型参数。
    fn parameters(&self) -> Vec<&Array<f32, IxDyn>>;

    /// Get mutable model parameters.
    /// 获取可变模型参数。
    fn parameters_mut(&mut self) -> Vec<&mut Array<f32, IxDyn>>;

    /// Number of trainable parameters.
    /// 可训练参数数量。
    fn num_parameters(&self) -> usize;

    /// Set training mode (enables dropout, etc.).
    /// 设置训练模式（启用 dropout 等）。
    fn train_mode(&mut self, training: bool);
}
```

### 6.2 Dataset Trait

```rust
/// Dataset trait for EEG data.
/// EEG 数据集 trait。
pub trait Dataset {
    /// Number of samples in dataset.
    /// 数据集中的样本数。
    fn len(&self) -> usize;

    /// Get a single sample by index.
    /// 按索引获取单个样本。
    fn get(&self, index: usize) -> (Array3<f32>, i64);  // [1, C, T], label

    /// Get a batch of samples.
    /// 获取一批样本。
    fn get_batch(&self, indices: &[usize]) -> (Array4<f32>, Array1<i64>);

    /// Input shape [1, n_channels, n_samples].
    /// 输入形状。
    fn input_shape(&self) -> (usize, usize, usize);

    /// Number of classes.
    /// 类别数。
    fn num_classes(&self) -> usize;
}
```

---

## 7. EEGNet 实现设计

### 7.1 配置结构

```rust
// bcif-nn/src/models/eegnet.rs
// EEGNet implementation (Lawhern et al., 2018)

/// EEGNet configuration.
/// EEGNet 配置。
#[derive(Debug, Clone)]
pub struct EEGNetConfig {
    /// Number of input channels (electrodes).
    /// 输入通道数（电极数）。
    pub n_channels: usize,

    /// Number of time samples per epoch.
    /// 每个 epoch 的采样点数。
    pub n_samples: usize,

    /// Number of output classes.
    /// 输出类别数。
    pub n_classes: usize,

    /// Number of temporal filters (F1).
    /// 时间滤波器数量 (F1)。
    pub f1: usize,

    /// Depth multiplier for depthwise conv (D).
    /// Depthwise 卷积的深度乘数 (D)。
    pub depth_mult: usize,

    /// Temporal kernel length (samples).
    /// 时间卷积核长度（采样点）。
    /// 建议: sampling_rate / 2 (如 250Hz → 125)
    pub kernel_length: usize,

    /// Dropout probability.
    /// Dropout 概率。
    pub dropout_rate: f32,
}
```

### 7.2 预设配置

```rust
impl EEGNetConfig {
    /// Motor Imagery (BCI Competition IV 2a).
    /// 运动想象。
    pub fn motor_imagery() -> Self {
        Self {
            n_channels: 22,
            n_samples: 1000,   // 4s @ 250Hz
            n_classes: 4,      // Left/Right/Feet/Tongue
            f1: 8,
            depth_mult: 2,
            kernel_length: 125, // 500ms @ 250Hz
            dropout_rate: 0.5,
        }
    }

    /// P300 Speller.
    /// P300 拼写器。
    pub fn p300() -> Self {
        Self {
            n_channels: 8,
            n_samples: 256,    // 1s @ 256Hz
            n_classes: 2,      // Target / Non-target
            f1: 8,
            depth_mult: 2,
            kernel_length: 128,
            dropout_rate: 0.25,
        }
    }

    /// SSVEP (Steady-State Visual Evoked Potential).
    /// 稳态视觉诱发电位。
    pub fn ssvep() -> Self {
        Self {
            n_channels: 9,
            n_samples: 1024,   // 4s @ 256Hz
            n_classes: 12,     // 12 frequencies
            f1: 8,
            depth_mult: 2,
            kernel_length: 256,
            dropout_rate: 0.5,
        }
    }
}
```

### 7.3 模型结构

```rust
/// EEGNet model structure.
/// EEGNet 模型结构。
pub struct EEGNet {
    config: EEGNetConfig,

    // Block 1: Temporal Convolution
    conv_temporal: Conv2D,      // (1, kernel_length) → F1 filters
    bn_temporal: BatchNorm2D,

    // Block 2: Depthwise Convolution (Spatial Filter)
    conv_depthwise: DepthwiseConv2D,  // (n_channels, 1) → F1*D filters
    bn_depthwise: BatchNorm2D,
    pool_depthwise: AvgPool2D,        // (1, 4)
    dropout_depthwise: Dropout,

    // Block 3: Separable Convolution
    conv_separable: SeparableConv2D,  // (1, 16) → F2 filters
    bn_separable: BatchNorm2D,
    pool_separable: AvgPool2D,        // (1, 8)
    dropout_separable: Dropout,

    // Block 4: Classification
    flatten: Flatten,
    dense: Dense,                      // → n_classes

    // Training state
    training: bool,
}

impl EEGNet {
    /// Create new EEGNet from config.
    /// 从配置创建新的 EEGNet。
    pub fn new(config: EEGNetConfig) -> Self { ... }

    /// Load from ONNX file.
    /// 从 ONNX 文件加载。
    pub fn from_onnx(path: &Path) -> Result<Self, NNError> { ... }

    /// Export to ONNX file.
    /// 导出为 ONNX 文件。
    pub fn to_onnx(&self, path: &Path) -> Result<(), NNError> { ... }
}
```

---

## 8. 训练接口设计

### 8.1 训练配置

```rust
// bcif-nn/src/training/trainer.rs

/// Training configuration.
/// 训练配置。
#[derive(Debug, Clone)]
pub struct TrainConfig {
    /// Number of epochs.
    /// 训练轮数。
    pub epochs: usize,

    /// Batch size.
    /// 批次大小。
    pub batch_size: usize,

    /// Learning rate.
    /// 学习率。
    pub learning_rate: f32,

    /// Weight decay (L2 regularization).
    /// 权重衰减（L2 正则化）。
    pub weight_decay: f32,

    /// Validation split ratio.
    /// 验证集比例。
    pub val_split: f32,

    /// Early stopping patience.
    /// 早停耐心值。
    pub patience: usize,

    /// Random seed for reproducibility.
    /// 随机种子（可复现性）。
    pub seed: u64,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            epochs: 500,
            batch_size: 32,
            learning_rate: 1e-3,
            weight_decay: 1e-4,
            val_split: 0.2,
            patience: 50,
            seed: 42,
        }
    }
}
```

### 8.2 Trainer 结构

```rust
/// Trainer for neural network models.
/// 神经网络模型训练器。
pub struct Trainer<M: Model> {
    model: M,
    optimizer: Adam,
    loss_fn: CrossEntropyLoss,
    config: TrainConfig,
}

impl<M: Model> Trainer<M> {
    /// Create new trainer.
    /// 创建新的训练器。
    pub fn new(model: M, config: TrainConfig) -> Self { ... }

    /// Train the model on dataset.
    /// 在数据集上训练模型。
    ///
    /// # Arguments
    /// * `train_data` - Training dataset (BCIF Epochs)
    /// * `val_data` - Optional validation dataset
    ///
    /// # Returns
    /// * Training history (loss, accuracy per epoch)
    pub fn fit(
        &mut self,
        train_data: &Epochs,
        val_data: Option<&Epochs>,
    ) -> TrainHistory { ... }

    /// Evaluate model on dataset.
    /// 在数据集上评估模型。
    pub fn evaluate(&self, data: &NNInput) -> (f32, f32) { ... }

    /// Get trained model.
    /// 获取训练后的模型。
    pub fn into_model(self) -> M { ... }
}
```

---

## 9. 完整使用流程

### 9.1 BCI 流水线架构

```
┌─────────────────────────────────────────────────────────────────┐
│                    完整 BCI 流水线                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ 硬件设备  │───▶│ mne-lsl  │───▶│  BCIF    │───▶│ 控制输出  │  │
│  │ (EEG)    │    │ (采集)   │    │ (处理)   │    │ (BCI)    │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                       │               │                         │
│                       ▼               ▼                         │
│              ┌─────────────┐  ┌─────────────────┐               │
│              │ LSL Stream  │  │ bcif-nn (新增)  │               │
│              │ [C, T] f64  │  │ ├── EEGNet      │               │
│              └─────────────┘  │ ├── ShallowNet  │               │
│                               │ └── ONNX推理    │               │
│                               └─────────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

### 9.2 训练阶段代码示例

```rust
// 1. 数据加载
let raw = bcif_io::read_xdf("motor_imagery.xdf")?;
let events = bcif_core::find_events(&raw, "stim")?;

// 2. 预处理 (BCIF Layer 2)
let raw = bcif_dsp::filter(&raw, 1.0, 40.0, "butter", 4)?;
let raw = bcif_dsp::notch_filter(&raw, 50.0)?;
let raw = bcif_algo::ica_remove_artifacts(&raw, 20)?;

// 3. 分段 (BCIF Layer 1)
let epochs = bcif_core::Epochs::from_events(
    &raw, &events,
    tmin: -0.5, tmax: 4.0,  // 时间窗口
    baseline: Some((-0.5, 0.0)),
)?;
// epochs.data: [n_epochs, n_channels, n_samples]
// epochs.labels: [769, 770, 771, 772, ...]

// 4. 训练 EEGNet (bcif-nn)
let config = EEGNetConfig::motor_imagery();
let model = EEGNet::new(config);

let train_config = TrainConfig {
    epochs: 500,
    batch_size: 32,
    learning_rate: 1e-3,
    ..Default::default()
};

let mut trainer = Trainer::new(model, train_config);
let history = trainer.fit(&epochs, None);

// 5. 导出模型
let model = trainer.into_model();
model.to_onnx("eegnet_mi.onnx")?;
```

### 9.3 推理阶段代码示例

```rust
// 加载训练好的模型
let model = EEGNet::from_onnx("eegnet_mi.onnx")?;
let predictor = Predictor::new(model);

// 实时循环
loop {
    // 从 LSL 获取数据
    let chunk = lsl_inlet.pull_chunk()?;

    // 预处理
    let filtered = bcif_dsp::filter(&chunk, 1.0, 40.0)?;

    // 推理
    let probs = predictor.predict(&filtered)?;
    let class = probs.argmax();

    // 输出控制信号
    match class {
        0 => send_command("LEFT"),
        1 => send_command("RIGHT"),
        2 => send_command("FEET"),
        3 => send_command("TONGUE"),
        _ => {}
    }
}
```

---

## 10. 实现路径建议

| 阶段 | 任务 | 工具/Crate |
|------|------|------------|
| 1 | 训练 EEGNet | Python + PyTorch/TensorFlow |
| 2 | 导出模型 | ONNX format |
| 3 | Rust 推理 | `ort` (ONNX Runtime) 或 `candle` |
| 4 | 集成 BCIF | bcif-nn 模块 |
| 5 | 嵌入式部署 | 定点数量化 + no_std 推理 |

### 推荐 Rust Crate

| 功能 | Crate | 说明 |
|------|-------|------|
| ONNX 推理 | `ort` | ONNX Runtime 官方绑定 |
| 纯 Rust 推理 | `candle` | Hugging Face 出品 |
| PyTorch 绑定 | `tch-rs` | 需要 libtorch |
| 多后端推理 | `burn` | 支持 WGPU/Candle/LibTorch |
| 张量运算 | `ndarray` | 与 BCIF 其他模块一致 |

---

## 参考文献

1. Lawhern, V. J., et al. (2018). EEGNet: A compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering*, 15(5), 056013.

2. Schirrmeister, R. T., et al. (2017). Deep learning with convolutional neural networks for EEG decoding and visualization. *Human Brain Mapping*, 38(11), 5391-5420.

# Brainstorm3 架构概述

## 1. 项目简介

Brainstorm3 是一个用于MEG/EEG脑电信号分析的开源MATLAB应用程序，专注于提供用户友好的图形界面和完整的分析工具链。

**核心特点：**
- 支持多种电生理数据：MEG、EEG、fNIRS、ECoG、深部电极和动物电生理
- 无需MATLAB许可证即可运行（提供编译版本）
- 丰富的图形界面和批处理脚本支持
- 超过35,000注册用户

## 2. 数据模型架构

### 2.1 核心数据结构（定义于 `toolbox/db/db_template.m`）

Brainstorm采用层次化的数据组织模型，主要包括以下核心结构：

#### **协议（Protocol）结构**
```matlab
ProtocolInfo
├── SUBJECTS/           % 受试者数据目录
├── STUDIES/            % 研究数据目录  
├── iStudy              % 研究索引
├── UseDefaultAnat      % 是否使用默认解剖
└── UseDefaultChannel   % 是否使用默认通道
```

#### **受试者（Subject）结构**
```matlab
Subject
├── Name                % 受试者名称
├── Anatomy[]           % 解剖结构数组（MRI等）
│   ├── Cube           % MRI体素数据
│   ├── Voxsize        % 体素尺寸
│   ├── SCS/NCS        % 坐标系统
│   └── History        % 操作历史
├── Surface[]           % 表面网格数组
│   ├── Vertices       % 顶点坐标
│   ├── Faces          % 面片索引
│   ├── Atlas          % 脑区图谱
│   └── Curvature      % 曲率信息
└── iCortex/iScalp     % 各类表面索引
```

#### **研究（Study）结构**
```matlab
Study
├── Channel[]           % 通道/传感器信息
│   ├── Channel[]      % 传感器阵列
│   ├── MegRefCoef     % MEG参考系数
│   └── Projector      % SSP投影矩阵
├── Data[]             % 记录数据
│   ├── F              % 信号矩阵 [nChannels × nTime]
│   ├── Time           % 时间向量
│   ├── Events[]       % 事件标记
│   └── ChannelFlag    % 通道标记
├── HeadModel[]        % 头模型
│   ├── Gain           % 增益矩阵 [nChannels × 3*nSources]
│   ├── GridLoc        % 源点位置 [nSources × 3]
│   └── GridOrient     % 源点方向
├── Result[]           % 源重建结果
│   ├── ImageGridAmp   % 源信号 [nSources × nTime]
│   ├── ImagingKernel  % 成像核
│   └── Function       % 估计方法
├── Timefreq[]         % 时频分析结果
│   ├── TF             % 时频矩阵 [nRows × nFreqs × nTime]
│   ├── Freqs          % 频率向量
│   └── Method         % 分析方法
├── Stat[]             % 统计分析结果
└── Matrix[]           % 通用矩阵数据
```

#### **事件（Event）结构**
```matlab
Event
├── label              % 事件标签
├── color              % 显示颜色 [R,G,B]
├── times              % 时间点 [1或2 × nOccur]
├── epochs             % 试次索引
├── channels           % 关联通道
└── hedTags            % HED标签
```

### 2.2 数据文件格式

所有Brainstorm数据以MATLAB `.mat`格式存储，采用统一的文件命名和结构规范：

- **data_*.mat**: 原始或处理后的记录数据
- **headmodel_*.mat**: 头模型/正向模型
- **results_*.mat**: 源重建结果
- **timefreq_*.mat**: 时频分析结果
- **noisecov_*.mat**: 噪声协方差矩阵
- **matrix_*.mat**: 通用矩阵数据

## 3. 模块组成

Brainstorm采用模块化设计，toolbox下分为多个功能模块：

### 3.1 核心模块（`toolbox/core/`）

**核心功能：**
- `bst_get.m` / `bst_set.m`: 全局状态管理（GlobalData）
- `bst_memory.m`: 内存管理，加载/卸载数据
- `bst_process.m`: 处理管道引擎
- `bst_figures.m`: 图形窗口管理
- `bst_plugin.m`: 插件系统管理

**关键设计模式：**
```matlab
% GlobalData结构 - 单例模式的全局状态
GlobalData
├── DataBase           % 数据库信息
│   ├── ProtocolInfo   % 协议配置
│   ├── ProtocolSubjects
│   └── ProtocolStudies
├── Preferences        % 用户偏好设置
├── Program            % 程序状态
└── MemoryLogs        % 加载的数据缓存
```

### 3.2 数据库模块（`toolbox/db/`）

- `db_template.m`: 定义所有数据结构模板
- `db_add_*.m`: 向数据库添加各类数据
- `db_reload_*.m`: 重新加载数据库元素
- `db_save.m`: 保存协议信息

### 3.3 I/O模块（`toolbox/io/`）

采用统一的命名规范：

**输入函数（220+个）：**
- `in_fopen_*.m`: 打开原始数据文件（200+种格式）
  - EEG: `in_fopen_edf.m`, `in_fopen_cnt.m`, `in_fopen_eeglab.m`
  - MEG: `in_fopen_ctf.m`, `in_fopen_4d.m`, `in_fopen_fif.m`
  - 其他: `in_fopen_blackrock.m`, `in_fopen_nsx.m`
- `in_fread_*.m`: 读取数据块
- `in_channel_*.m`: 读取通道/传感器信息
- `in_events_*.m`: 导入事件标记
- `in_tess_*.m`: 导入网格表面
- `in_mri_*.m`: 导入MRI数据

**输出函数（70+个）：**
- `out_fwrite_*.m`: 写入原始数据
- `out_fieldtrip_*.m`: 导出到FieldTrip格式
- `out_mne_*.m`: 导出到MNE-Python格式
- `out_events_*.m`: 导出事件
- `out_tess_*.m`: 导出表面网格

**设计模式：**
```matlab
% 文件句柄结构（sFile）
sFile
├── filename          % 完整路径
├── format            % 文件格式标识
├── device            % 采集设备
├── header            % 原始头信息
├── channelmat        % 通道结构
└── epochs[]          % 试次/分段信息
```

### 3.4 处理模块（`toolbox/process/`）

**处理框架：**
- `bst_process.m`: 处理管道核心引擎
- `bst_report.m`: 生成处理报告
- `panel_process_select.m`: GUI选择界面

**处理函数（257+个）在 `functions/` 子目录：**

分类体系：
1. **预处理** (40+)
   - `process_bandpass.m`: 带通滤波
   - `process_notch.m`: 陷波滤波
   - `process_resample.m`: 重采样
   - `process_detectbad.m`: 坏通道检测
   
2. **事件处理** (30+)
   - `process_evt_*.m`: 事件操作
   - `process_import_events.m`: 导入事件
   
3. **源估计** (20+)
   - `process_inverse.m`: 源重建（主函数）
   - `process_dipole_scanning.m`: 偶极子扫描
   
4. **时频分析** (15+)
   - `process_timefreq.m`: 时频分解
   - `process_hilbert.m`: Hilbert变换
   - `process_psd.m`: 功率谱密度
   
5. **连接性分析** (25+)
   - `process_corr1n.m`: 相关性
   - `process_cohere1n.m`: 相干性
   - `process_granger1n.m`: Granger因果
   - `process_plv1n.m`: 相位锁定值
   
6. **统计分析** (20+)
   - `process_test_parametric*.m`: 参数检验
   - `process_test_permutation*.m`: 置换检验
   
7. **机器学习** (10+)
   - `process_decoding.m`: 分类解码
   - `process_spikesorting_*.m`: 尖峰分类

**处理函数接口规范：**
```matlab
function varargout = process_example(varargin)
    eval(macro_method);
end

% 必需方法：
function sProcess = GetDescription()
    sProcess.Comment     = '功能描述';
    sProcess.Category    = '分类';
    sProcess.InputTypes  = {'data', 'raw'};
    sProcess.OutputTypes = {'data'};
    sProcess.options     = struct(...);  % 参数定义
end

function OutputFiles = Run(sProcess, sInputs)
    % 实际处理逻辑
end
```

### 3.5 正向模型模块（`toolbox/forward/`）

**头模型计算：**
- `bst_headmodeler.m`: 头模型主控
- `bst_openmeeg.m`: OpenMEEG BEM求解器
- `bst_duneuro.m`: DUNEuro FEM求解器
- `bst_eeg_sph.m`: EEG球形头模型
- `bst_meg_sph.m`: MEG球形头模型
- `bst_sourcegrid.m`: 源空间网格生成
- `bst_gain_orient.m`: 增益矩阵方向处理

### 3.6 逆向求解模块（`toolbox/inverse/`）

**源重建算法：**
- 最小范数估计（MNE）
- dSPM（动态统计参数映射）
- sLORETA（标准化低分辨率脑电磁断层成像）
- LCMV波束形成器
- SAM（合成孔径磁强计）

### 3.7 解剖学模块（`toolbox/anatomy/`）

- `import_anatomy_*.m`: 导入各种解剖格式
  - FreeSurfer, CAT12, BrainSuite, BrainVISA
- `bst_normalize_mni.m`: MNI标准化
- `tess_*.m`: 表面网格操作
- 解剖配准和坐标系转换

### 3.8 时频分析模块（`toolbox/timefreq/`）

- `bst_timefreq.m`: 时频分解主函数
  - Morlet小波
  - Hilbert变换
  - 短时傅里叶变换（STFT）
- `bst_psd.m`: Welch功率谱密度
- `bst_pac.m`: 相位-幅值耦合

### 3.9 连接性模块（`toolbox/connectivity/`）

- `bst_mvar.m`: 多元自回归（MVAR）建模
- Granger因果分析
- 相位相干性、PLV、AEC等

### 3.10 数学/信号处理模块（`toolbox/math/`）

通用算法：
- 滤波器设计
- 统计函数
- 插值和平滑
- 矩阵运算优化

### 3.11 GUI模块（`toolbox/gui/`）

- `panel_*.m`: 各类GUI面板（80+个）
- 3D可视化（基于MATLAB图形和OpenGL）
- 交互式数据浏览器

### 3.12 传感器模块（`toolbox/sensors/`）

- 通道位置配准
- 传感器坐标系统转换
- 标准电极帽定义

### 3.13 实时处理模块（`toolbox/realtime/`）

支持实时数据流处理和神经反馈

## 4. 核心算法

### 4.1 信号处理算法

**滤波：**
- FIR/IIR数字滤波器
- 陷波滤波（去除电源线干扰）
- 高通/低通/带通滤波

**去噪：**
- SSP（信号空间投影）- 去除伪迹
- ICA（独立成分分析）
- PCA（主成分分析）

**时频分析：**
- 连续小波变换（CWT）
- Hilbert-Huang变换
- 多taper方法

### 4.2 源定位算法

**线性逆问题求解：**
```
最小化: ||Gain × Sources - Data||² + λ × ||L × Sources||²

其中:
- Gain: 增益矩阵（正向模型）
- Sources: 源活动
- Data: 观测数据
- L: 正则化矩阵
- λ: 正则化参数
```

**算法分类：**
1. **分布源模型：**
   - MNE（最小范数估计）
   - dSPM（深度加权）
   - sLORETA（零定位误差）

2. **自适应波束形成器：**
   - LCMV（线性约束最小方差）
   - SAM（合成孔径磁强计）

3. **偶极子拟合：**
   - 等效电流偶极子（ECD）
   - MUSIC算法

### 4.3 连接性算法

**功能连接性：**
- 相关/协方差
- 相干性（Coherence）
- 相位锁定值（PLV）
- 幅值包络相关（AEC）

**有效连接性：**
- Granger因果
- 部分定向相干（PDC）
- 定向传递函数（DTF）
- 相位传递熵（PTE）

**动态连接性：**
- 滑窗分析
- 时变相干性

### 4.4 统计分析

**单样本/双样本检验：**
- t检验
- 配对t检验
- 单因素/双因素ANOVA

**多重比较校正：**
- FDR（错误发现率）
- Bonferroni校正
- 最大统计量置换检验
- 聚类统计

## 5. 模块协作机制

### 5.1 处理流水线

```
原始数据导入 (I/O模块)
    ↓
预处理 (Process模块 + Math模块)
    ↓
试次提取/平均
    ↓
[分支1: 传感器空间分析]     [分支2: 源空间分析]
    ↓                           ↓
时频分析                    正向模型 (Forward模块)
    ↓                           ↓
连接性分析                  源重建 (Inverse模块)
    ↓                           ↓
统计检验 ←──────────────── 时频/连接性分析
    ↓
可视化 (GUI模块) / 导出 (I/O模块)
```

### 5.2 数据流

```matlab
% 典型的处理流程
1. bst_process('CallProcess', 'process_import_data_raw', ...)
   → I/O模块读取原始数据
   → DB模块注册到数据库
   
2. bst_process('CallProcess', 'process_bandpass', ...)
   → Memory模块加载数据
   → Math模块执行滤波
   → 保存新文件
   
3. bst_process('CallProcess', 'process_inverse', ...)
   → Forward模块加载头模型
   → Inverse模块计算源信号
   → DB模块保存结果
   
4. GUI模块可视化
   → Figures模块管理窗口
   → 从Memory读取数据
   → 渲染到屏幕
```

### 5.3 插件系统

**插件管理（`bst_plugin.m`）：**
- 动态加载外部工具箱
- 版本管理和自动更新
- 依赖解析

**支持的主要插件：**
- **FieldTrip**: MEG/EEG分析工具箱
- **SPM12**: 统计参数映射
- **MNE-Python**: Python集成
- **CAT12**: 解剖分割
- **OpenMEEG**: BEM求解器
- **DUNEuro**: FEM求解器
- **ISO2MESH/Brain2Mesh**: 网格生成

### 5.4 Python集成

通过MATLAB的Python引擎实现与MNE-Python的互操作：

```matlab
% Python初始化
bst_python_init('Initialize', 1);

% 导出到MNE对象
pyRaw = out_mne_data(DataFile, 'Raw');

% 调用Python函数
pyRaw_sss = py.mne.preprocessing.maxwell_filter(pyRaw);

% 导入回Brainstorm
DataFile = in_fread_mne(pyRaw_sss);
```

### 5.5 批处理脚本

**脚本模板（`toolbox/script/`）：**
- `tutorial_*.m`: 教程脚本
- 标准化的分析流程
- 可重复的研究范式

## 6. 坐标系统和配准

### 6.1 坐标系统

**多坐标系支持：**
1. **设备坐标系**: 原始采集坐标
2. **头部坐标系（SCS）**: Brainstorm标准
   - 基于解剖标志点（NAS、LPA、RPA）
3. **MNI坐标系**: 标准化脑空间
4. **Talairach坐标系**: 老式标准

**坐标变换链：**
```
设备坐标 → 头部坐标(SCS) → MRI体素坐标 → MNI坐标
          ↑
    TransfMeg/TransfEeg矩阵
```

### 6.2 配准方法

- **MRI-通道配准**: 基于标志点的刚体变换
- **MNI标准化**: 
  - 线性（affine）: SPM maff8
  - 非线性: SPM segment（变形场）
- **表面配准**: FreeSurfer/BrainVISA

## 7. 文件组织

```
brainstorm3/
├── brainstorm.m              # 启动脚本
├── defaults/                 # 默认模板
│   ├── eeg/                 # EEG电极模板
│   └── meg/                 # MEG通道模板
├── deploy/                   # 编译部署
├── doc/                      # 文档
├── external/                 # 外部依赖（200+工具）
│   ├── mne/                 # MNE MATLAB工具
│   ├── fieldtrip/           # FieldTrip接口
│   ├── spm/                 # SPM函数
│   └── ...
├── java/                     # Java GUI组件
├── python/                   # Python集成示例
└── toolbox/                  # 核心工具箱
    ├── anatomy/             # 解剖学处理
    ├── connectivity/        # 连接性分析
    ├── core/               # 核心系统
    ├── db/                 # 数据库管理
    ├── forward/            # 正向模型
    ├── gui/                # 图形界面
    ├── inverse/            # 逆问题求解
    ├── io/                 # 输入输出
    ├── math/               # 数学算法
    ├── misc/               # 辅助函数
    ├── process/            # 处理框架
    ├── realtime/           # 实时处理
    ├── script/             # 脚本模板
    ├── sensors/            # 传感器处理
    ├── timefreq/           # 时频分析
    └── tree/               # 数据树GUI
```

## 8. 编程范式

### 8.1 MATLAB特性使用

**面向对象：**
- 结构体（struct）作为主要数据容器
- 通过函数式接口操作数据（而非类）

**函数式编程：**
- 大量使用函数句柄
- 匿名函数用于回调
- varargin/varargout实现灵活接口

**eval/macro模式：**
```matlab
function varargout = my_function(varargin)
    eval(macro_method);  % 动态调度子方法
end
```

### 8.2 内存管理策略

**延迟加载：**
- 数据库只存储文件路径
- 仅在需要时加载数据到内存

**缓存机制：**
```matlab
GlobalData.MemoryLogs{i}
├── DataFile        % 文件路径
├── DataMat         % 加载的数据
└── UnloadTimer    % 自动卸载计时器
```

### 8.3 错误处理

- `bst_error.m`: 统一错误报告
- `bst_report.m`: 处理日志和报告生成
- Try-catch块捕获并记录错误

## 9. 扩展性设计

### 9.1 添加新的数据格式

1. 在 `toolbox/io/` 创建 `in_fopen_newformat.m`
2. 实现标准的sFile结构返回
3. 创建 `in_fread_newformat.m` 读取数据块
4. 在GUI中注册新格式

### 9.2 添加新的处理算法

1. 在 `toolbox/process/functions/` 创建 `process_newalgo.m`
2. 实现 `GetDescription()` 定义接口
3. 实现 `Run()` 执行算法
4. 自动集成到GUI处理面板

### 9.3 添加新的可视化

1. 在 `toolbox/gui/` 创建 `view_*.m` 或 `panel_*.m`
2. 使用 `bst_figures.m` 注册新视图类型
3. 实现标准的回调接口

## 10. 性能优化

### 10.1 并行处理

- 支持MATLAB Parallel Computing Toolbox
- 在process选项中启用parallel标志
- 自动分配试次到并行池

### 10.2 内存优化

- 分块读取大数据文件
- 稀疏矩阵用于大规模源重建
- 自动清理未使用的缓存

### 10.3 计算优化

- MEX编译关键算法（C/C++）
- 向量化MATLAB代码
- GPU加速（某些算法）

## 11. 总结

Brainstorm3采用了高度模块化的架构设计，其特点包括：

1. **清晰的数据模型**: 层次化的Subject-Study-Data结构
2. **统一的接口规范**: in_/out_/process_前缀标识功能
3. **可扩展的处理框架**: 基于插件的算法集成
4. **丰富的I/O支持**: 200+种数据格式
5. **完整的分析流程**: 从原始数据到统计推断
6. **混合语言支持**: MATLAB核心，Python/Java集成
7. **用户友好**: GUI和脚本双重接口

这种设计使得Brainstorm既适合交互式探索，也适合大规模批处理，成为神经电生理研究的强大工具。

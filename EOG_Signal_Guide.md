# EOG 信号技术指南
# EOG (Electrooculography) Signal Technical Guide

> **文档目标**: 理解EEG采集中的EOG信号本质及其在BCI系统中的应用
> **适用项目**: BCIF (Brain-Computer Interface Framework)
> **创建日期**: 2026-02-06

---

## 目录

1. [EOG信号的本质](#1-eog信号的本质)
2. [EOG信号特征](#2-eog信号特征)
3. [EOG的双重角色](#3-eog的双重角色)
4. [应用场景详解](#4-应用场景详解)
5. [技术对比](#5-技术对比)
6. [BCIF实现建议](#6-bcif实现建议)

---

## 1. EOG信号的本质

### 1.1 生理原理

**EOG不是"眼球追踪"，而是测量角膜-视网膜电位差**

```
眼球的电偶极子模型：
    角膜 (+)  ←→  视网膜 (-)
    电位差: 0.4-1.0 mV
```

**工作机制**：
- 眼球像一个"电池"：角膜带正电（+），视网膜带负电（-）
- 当眼球转动或眨眼时，这个电偶极子的方向改变
- 在头皮电极上产生可测量的电位变化
- 眨眼时眼睑遮挡角膜，产生**大幅度的垂直EOG信号**（通常>100 μV）

### 1.2 信号幅度对比

| 信号类型 | 典型幅度 | 频率范围 |
|---------|---------|---------|
| **EEG（脑电）** | 10-100 μV | 0.5-100 Hz |
| **EOG（眼电）** | 100-500 μV | 0.1-30 Hz |
| **EMG（肌电）** | 50-5000 μV | 20-500 Hz |
| **ECG（心电）** | 1000-3000 μV | 0.5-40 Hz |

**关键问题**：EOG幅度是EEG的**5-10倍**，容易污染脑电信号！

### 1.3 电极配置

**标准EOG电极位置**：

```
垂直EOG (VEOG)：检测眨眼和垂直��动
    - 上电极：眉毛上方（Fp1/Fp2附近）
    - 下电极：眼睛下方
    - 参考：耳垂或鼻尖

水平EOG (HEOG)：检测水平眼动
    - 左电极：左眼外眦（外眼角）
    - 右电极：右眼外眦
    - 参考：耳垂或鼻尖
```

**最小配置**：
- 2个电极：单通道VEOG（检测眨眼）
- 4个电极：双通道VEOG + HEOG（检测眨眼和眼动方向）

---

## 2. EOG信号特征

### 2.1 眨眼信号特征

| 眨眼类型 | 持续时间 | 幅度 | 频率 | 波形 |
|---------|---------|------|------|------|
| **自然眨眼** | 100-150 ms | 100-200 μV | 15-20次/分钟 | 单峰，对称 |
| **主动眨眼** | 150-400 ms | 200-500 μV | 不规律 | 可能双峰，不对称 |
| **疲劳眨眼** | >300 ms | 150-300 μV | >25次/分钟 | 持续时间延长 |

### 2.2 眼动信号特征

| 眼动类型 | 持续时间 | 幅度 | 波形 |
|---------|---------|------|------|
| **扫视（Saccade）** | 20-100 ms | 20-80 μV | 阶跃函数 |
| **平滑追踪** | 持续 | 10-50 μV | 平滑曲线 |
| **眨眼** | 100-400 ms | >100 μV | 尖峰 |

### 2.3 MNE-Python检测算法

**核心流程**（基于 `mne/preprocessing/eog.py`）：

```python
# 1. 带通滤波（去除DC漂移，突出眨眼）
eog_filtered = filter_data(eog, sfreq, l_freq=1.0, h_freq=10.0)

# 2. 自动阈值（信号峰峰值的1/4）
thresh = (max(eog_filtered) - min(eog_filtered)) / 4

# 3. 峰值检测
if abs(max(eog_filtered)) > abs(min(eog_filtered)):
    peaks = peak_finder(eog_filtered, thresh, extrema=1)  # 正峰
else:
    peaks = peak_finder(eog_filtered, thresh, extrema=-1)  # 负峰
```

**峰值检测算法**（`_peak_finder.py`）：
- 基于导数变化检测极值点
- 噪声容忍：要求峰值高于周围至少 `thresh`
- 避免重复检测：相邻峰值必须有足够间隔

---

## 3. EOG的双重角色

### 3.1 作为"伪迹"（Artifact）

**问题**：EOG污染EEG信号

```
真实情况：
EEG信号：  ~~~~ (10-50 μV, 包含认知信息)
EOG伪迹：  ^^^^^ (100-300 μV, 眨眼/眼动)
记录信号：  ^^^^^ (EOG淹没了EEG)
```

**影响的分析**：
- ❌ **ERP分析**：P300、N400等成分被掩盖
- ❌ **频谱分析**：低频段（<4 Hz）被污染
- ❌ **源定位**：错误定位到额叶
- ❌ **连接性分析**：虚假的额叶-其他区域连接

**解决方法**：ICA去除EOG成分

```python
from sklearn.decomposition import FastICA

# 1. 运行ICA分解
ica = ICA(n_components=20, method='fastica')
ica.fit(raw)

# 2. 自动识别EOG成分（与EOG通道相关性高）
eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name='EOG')

# 3. 排除EOG成分，重建干净的EEG
ica.exclude = eog_indices
raw_clean = ica.apply(raw)
```

### 3.2 作为"控制信号"（Feature）

**优势**：
- ✅ **简单可靠**：不需要复杂的机器学习，阈值判断即可
- ✅ **响应快速**：延迟<100ms，接近自然反应
- ✅ **训练时间短**：5-10分钟即可上手（vs EEG-BCI需要数小时）
- ✅ **成本低廉**：只需2-4个电极（vs EEG需要16-64个）
- ✅ **适合重度残疾**：眼球运动是ALS晚期患者唯一可控的身体部位

---

## 4. 应用场景详解

### 4.1 辅助通信：ALS患者拼写器

**目标用户**：渐冻症（ALS）晚期患者
- 无法说话、无法移动
- 但眼球运动保留到最后

**系统设计**：

```rust
struct EogSpeller {
    keyboard: Vec<Vec<char>>,  // 虚拟键盘
    current_pos: (usize, usize),
    dwell_time: f64,  // 停留时间（秒）
}

impl EogSpeller {
    fn process_eog(&mut self, heog: f64, veog: f64, dt: f64) -> Option<char> {
        let movement = self.detect_movement(heog, veog);

        match movement {
            EyeMovement::Up => {
                self.current_pos.0 = self.current_pos.0.saturating_sub(1);
                self.dwell_time = 0.0;
            }
            EyeMovement::Down => {
                self.current_pos.0 = (self.current_pos.0 + 1).min(4);
                self.dwell_time = 0.0;
            }
            EyeMovement::Left => {
                self.current_pos.1 = self.current_pos.1.saturating_sub(1);
                self.dwell_time = 0.0;
            }
            EyeMovement::Right => {
                self.current_pos.1 = (self.current_pos.1 + 1).min(4);
                self.dwell_time = 0.0;
            }
            EyeMovement::None => {
                // 停留在当前位置：累计时间
                self.dwell_time += dt;

                // 停留超过1.5秒 = 选择该字母
                if self.dwell_time > 1.5 {
                    let (row, col) = self.current_pos;
                    self.dwell_time = 0.0;
                    return Some(self.keyboard[row][col]);
                }
            }
        }

        None
    }

    fn detect_movement(&self, heog: f64, veog: f64) -> EyeMovement {
        const THRESHOLD: f64 = 60.0;  // μV

        if veog > THRESHOLD {
            EyeMovement::Up
        } else if veog < -THRESHOLD {
            EyeMovement::Down
        } else if heog > THRESHOLD {
            EyeMovement::Right
        } else if heog < -THRESHOLD {
            EyeMovement::Left
        } else {
            EyeMovement::None
        }
    }
}
```

**性能指标**：
- 打字速度：5-10字/分钟
- 准确率：90-95%
- 学习时间：<10分钟

### 4.2 智能轮椅控制

**目标用户**：
- 高位截瘫患者（C1-C4脊髓损伤）
- 脑瘫患者（无法控制手部）
- 多发性硬化症患者

**控制方案**：

```rust
struct EogWheelchair {
    speed: f64,           // 当前速度 (m/s)
    direction: f64,       // 当前方向 (度)
    max_speed: f64,       // 最大速度
}

impl EogWheelchair {
    fn process_command(&mut self, heog: f64, veog: f64) -> WheelchairAction {
        // 安全检查：双眨眼 = 紧急停止
        if self.detect_double_blink(veog) {
            self.speed = 0.0;
            return WheelchairAction::EmergencyStop;
        }

        const TURN_THRESHOLD: f64 = 80.0;
        const SPEED_THRESHOLD: f64 = 100.0;

        if heog > TURN_THRESHOLD {
            self.direction += 5.0;  // 向右转
            WheelchairAction::TurnRight
        } else if heog < -TURN_THRESHOLD {
            self.direction -= 5.0;  // 向左转
            WheelchairAction::TurnLeft
        } else if veog > SPEED_THRESHOLD {
            self.speed = (self.speed + 0.1).min(self.max_speed);
            WheelchairAction::Accelerate
        } else if veog < -SPEED_THRESHOLD {
            self.speed = (self.speed - 0.1).max(0.0);
            WheelchairAction::Decelerate
        } else {
            WheelchairAction::Maintain
        }
    }
}
```

**安全特性**：
- 双眨眼紧急停止
- 障碍物检测（超声波/激光雷达）
- 速度限制（根据环境动态调整）

### 4.3 驾驶员疲劳监测（商业化最成功）

**市场规模**：
- 全球每年因疲劳驾驶导致的事故：数十万起
- 商用车（卡车、客车）强制安装疲劳监测系统（部分国家）

**疲劳指标**：

```rust
struct FatigueMonitor {
    blink_history: VecDeque<BlinkEvent>,
    window_size: Duration,  // 分析窗口（如60秒）
}

impl FatigueMonitor {
    fn assess_fatigue(&self) -> FatigueLevel {
        let recent_blinks = self.get_recent_blinks();

        // 指标1：眨眼频率
        let blink_rate = recent_blinks.len() as f64 / 60.0;  // 次/分钟

        // 指标2：平均眨眼持续时间
        let avg_duration: f64 = recent_blinks.iter()
            .map(|b| b.duration)
            .sum::<f64>() / recent_blinks.len() as f64;

        // 指标3：PERCLOS（长时间闭眼比例）
        let long_closures = recent_blinks.iter()
            .filter(|b| b.duration > 0.5)
            .count();
        let perclos = long_closures as f64 / recent_blinks.len() as f64;

        // 综合判断
        let fatigue_score = self.calculate_score(blink_rate, avg_duration, perclos);

        match fatigue_score {
            s if s > 0.8 => FatigueLevel::Critical,  // 立即警报
            s if s > 0.6 => FatigueLevel::High,      // 建议休息
            s if s > 0.4 => FatigueLevel::Medium,    // 提醒注意
            _ => FatigueLevel::Low,                  // 正常
        }
    }

    fn calculate_score(&self, blink_rate: f64, avg_duration: f64, perclos: f64) -> f64 {
        // 正常状态：15-20次/分钟，150ms持续时间，<10% PERCLOS
        // 疲劳状态：>25次/分钟，>300ms持续时间，>20% PERCLOS

        let rate_score = ((blink_rate - 15.0) / 10.0).clamp(0.0, 1.0);
        let duration_score = ((avg_duration - 0.15) / 0.15).clamp(0.0, 1.0);
        let perclos_score = (perclos / 0.2).clamp(0.0, 1.0);

        // 加权平均
        0.3 * rate_score + 0.3 * duration_score + 0.4 * perclos_score
    }
}
```

**商业产品**：
- **奔驰 ATTENTION ASSIST**：基于方向盘转动+眨眼模式
- **沃尔沃 Driver Alert Control**：摄像头+EOG算法
- **Seeing Machines**：专业驾驶员监测系统（商用车）

### 4.4 睡眠监测与分期

**应用**：
- 多导睡眠图（PSG）：医院睡眠实验室
- 家用睡眠监测：智能手环/头带
- 睡眠研究：REM睡眠、梦境研究

**睡眠分期规则**：

| 睡眠阶段 | EOG特征 | EEG特征 |
|---------|---------|---------|
| **清醒** | 频繁眨眼，快速眼动 | Alpha波（8-12 Hz） |
| **N1（浅睡）** | 慢速眼动（SEM） | Theta波（4-8 Hz） |
| **N2/N3（深睡）** | 无眼动 | Delta波（0.5-4 Hz） |
| **REM睡眠** | 快速眼动（REM） | 低幅混合频率 |

```rust
fn classify_sleep_stage(
    eog: &[f64],
    eeg: &[f64],
    emg: &[f64],
) -> SleepStage {
    let eog_activity = calculate_eog_activity(eog);
    let eeg_delta_power = calculate_band_power(eeg, 0.5, 4.0);
    let eeg_theta_power = calculate_band_power(eeg, 4.0, 8.0);
    let emg_tone = calculate_muscle_tone(emg);

    // AASM标准
    if eog_activity > 50.0 && emg_tone < 10.0 {
        SleepStage::REM  // 快速眼动 + 低肌张力
    } else if eeg_delta_power > 75.0 {
        SleepStage::N3  // 深睡眠
    } else if eeg_theta_power > 50.0 {
        SleepStage::N2  // 浅睡眠
    } else if eog_activity > 20.0 {
        SleepStage::N1  // 入睡期
    } else {
        SleepStage::Wake  // 清醒
    }
}
```

### 4.5 VR/AR交互

**应用场景**：
- VR游戏：眼神瞄准、菜单选择
- AR工业：免提操作（外科手术、维修）
- 军事训练：飞行员头盔显示器

**EOG vs 摄像头Eye-tracking**：

| 特性 | EOG | 摄像头Eye-tracking |
|------|-----|-------------------|
| **VR头显集成** | 容易（小电极） | 困难（需要内置摄像头）|
| **成本** | 低 | 高 |
| **精度** | 低（方向级） | 高（像素级）|
| **适用场景** | 粗略选择 | 精确瞄准 |

### 4.6 智能家居控制

**应用场景**：
- 残疾人家居控制（灯光、窗帘、电视）
- 老年人辅助生活
- 免提家居控制（烹饪时、抱孩子时）

**系统设计**：

```rust
struct EogSmartHome {
    devices: HashMap<String, Device>,
    current_selection: Option<String>,
    dwell_timer: f64,
}

impl EogSmartHome {
    fn process_eog(&mut self, heog: f64, veog: f64, dt: f64) -> Option<HomeCommand> {
        // 1. 眼动选择设备
        let device_direction = self.detect_direction(heog, veog);

        match device_direction {
            Direction::Up => {
                self.current_selection = Some("light".to_string());
                self.dwell_timer = 0.0;
            }
            Direction::Down => {
                self.current_selection = Some("tv".to_string());
                self.dwell_timer = 0.0;
            }
            Direction::Left => {
                self.current_selection = Some("curtain".to_string());
                self.dwell_timer = 0.0;
            }
            Direction::Right => {
                self.current_selection = Some("ac".to_string());
                self.dwell_timer = 0.0;
            }
            Direction::None => {
                // ���留选择
                if let Some(device) = &self.current_selection {
                    self.dwell_timer += dt;

                    if self.dwell_timer > 2.0 {
                        // 停留2秒 = 切换设备状态
                        return Some(HomeCommand::Toggle(device.clone()));
                    }
                }
            }
        }

        None
    }

    fn detect_blink_command(&self, veog: f64) -> Option<HomeCommand> {
        // 单次眨眼：确认
        // 双次眨眼：取消
        // 三次眨眼：紧急呼叫
        if veog.abs() > 150.0 {
            Some(HomeCommand::Confirm)
        } else {
            None
        }
    }
}

enum HomeCommand {
    Toggle(String),      // 切换设备状态
    Confirm,             // 确认操作
    Cancel,              // 取消操作
    EmergencyCall,       // 紧急呼叫
}
```

**控制界面**：

```
屏幕显示：
┌─────────────────────────────┐
│   向上看 → 💡 灯光           │
│   向下看 → 📺 电视           │
│   向左看 → 🪟 窗帘           │
│   向右看 → ❄️ 空调           │
│                             │
│   停留2秒 = 切换开关         │
│   眨眼 = 确认               │
└─────────────────────────────┘
```

**优势**：
- 完全免提操作
- 适合行动不便的用户
- 成本低于语音控制（无需麦克风阵列）

### 4.7 医疗诊断辅助

**应用场景**：
- 神经系统疾病诊断
- 眼肌功能评估
- 注意力缺陷多动障碍（ADHD）评估

#### 4.7.1 眼肌麻痹检测

```rust
struct OcularMotilityTest {
    test_positions: Vec<GazePosition>,
    current_position: usize,
}

impl OcularMotilityTest {
    fn run_test(&mut self, heog: f64, veog: f64) -> TestResult {
        // 要求患者看向不同方向
        let target = self.test_positions[self.current_position];

        // 测量实际眼动幅度
        let actual_heog = heog;
        let actual_veog = veog;

        // 计算眼动范围
        let horizontal_range = self.calculate_range(actual_heog);
        let vertical_range = self.calculate_range(actual_veog);

        // 正常范围：±30度（约±100 μV）
        let is_normal = horizontal_range > 80.0 && vertical_range > 80.0;

        TestResult {
            position: target,
            horizontal_range,
            vertical_range,
            is_normal,
        }
    }
}

struct TestResult {
    position: GazePosition,
    horizontal_range: f64,  // μV
    vertical_range: f64,    // μV
    is_normal: bool,
}
```

**诊断标准**：

| 疾病 | EOG特征 |
|------|---------|
| **眼肌麻痹** | 某个方向眼动幅度减小（<50 μV） |
| **重症肌无力** | 持续注视时眼动幅度逐渐减小 |
| **帕金森病** | 扫视速度减慢，眨眼频率降低 |
| **进行性核上性麻痹** | 垂直眼动受限 |

#### 4.7.2 ADHD评估

```rust
struct AdhdAssessment {
    fixation_duration: Vec<f64>,  // 注视持续时间
    saccade_count: usize,         // 眼跳次数
    blink_rate: f64,              // 眨眼频率
}

impl AdhdAssessment {
    fn assess(&self) -> AdhdScore {
        // ADHD患者特征：
        // 1. 注视持续时间短（难以集中注意力）
        // 2. 眼跳频繁（容易分心）
        // 3. 眨眼频率高（焦虑）

        let avg_fixation = self.fixation_duration.iter().sum::<f64>()
                          / self.fixation_duration.len() as f64;

        let adhd_score = if avg_fixation < 0.5 && self.saccade_count > 100 {
            AdhdScore::High  // 高风险
        } else if avg_fixation < 1.0 && self.saccade_count > 50 {
            AdhdScore::Medium  // 中等风险
        } else {
            AdhdScore::Low  // 低风险
        };

        adhd_score
    }
}
```

### 4.8 阅读研究与教育

**应用场景**：
- 阅读障碍（Dyslexia）研究
- 阅读效率评估
- 在线教育注意力监测

#### 4.8.1 阅读模式分析

```rust
struct ReadingAnalyzer {
    fixations: Vec<Fixation>,      // 注视点
    saccades: Vec<Saccade>,        // 眼跳
    regressions: usize,            // 回视次数
}

#[derive(Debug)]
struct Fixation {
    duration: f64,      // 注视持续时间（ms）
    position: (f64, f64),  // 注视位置
}

impl ReadingAnalyzer {
    fn analyze_reading_pattern(&self) -> ReadingMetrics {
        // 计算阅读指标
        let avg_fixation_duration = self.fixations.iter()
            .map(|f| f.duration)
            .sum::<f64>() / self.fixations.len() as f64;

        let saccade_amplitude = self.saccades.iter()
            .map(|s| s.amplitude)
            .sum::<f64>() / self.saccades.len() as f64;

        let regression_rate = self.regressions as f64 / self.fixations.len() as f64;

        ReadingMetrics {
            avg_fixation_duration,  // 正常：200-250ms
            saccade_amplitude,      // 正常：7-9个字符
            regression_rate,        // 正常：10-15%
            reading_speed: self.calculate_reading_speed(),
        }
    }

    fn detect_dyslexia(&self) -> bool {
        let metrics = self.analyze_reading_pattern();

        // 阅读障碍特征：
        // 1. 注视时间长（>300ms）
        // 2. 眼跳幅度小（<5个字符）
        // 3. 回视频繁（>20%）

        metrics.avg_fixation_duration > 300.0
            && metrics.saccade_amplitude < 5.0
            && metrics.regression_rate > 0.2
    }
}

struct ReadingMetrics {
    avg_fixation_duration: f64,  // ms
    saccade_amplitude: f64,      // 字符数
    regression_rate: f64,        // 回视率
    reading_speed: f64,          // 字/分钟
}
```

#### 4.8.2 在线教育注意力监测

```rust
struct OnlineLearningMonitor {
    on_screen_time: f64,      // 看屏幕的时间
    off_screen_time: f64,     // 看其他地方的时间
    blink_rate: f64,          // 眨眼频率
}

impl OnlineLearningMonitor {
    fn assess_engagement(&self) -> EngagementLevel {
        // 计算注意力集中度
        let attention_ratio = self.on_screen_time / (self.on_screen_time + self.off_screen_time);

        // 眨眼频率：专注时减少
        let focus_score = if self.blink_rate < 10.0 {
            1.0  // 高度专注
        } else if self.blink_rate < 15.0 {
            0.7  // 中等专注
        } else {
            0.3  // 分心
        };

        let engagement = attention_ratio * focus_score;

        match engagement {
            e if e > 0.8 => EngagementLevel::High,
            e if e > 0.5 => EngagementLevel::Medium,
            _ => EngagementLevel::Low,
        }
    }

    fn generate_report(&self) -> LearningReport {
        LearningReport {
            total_time: self.on_screen_time + self.off_screen_time,
            attention_time: self.on_screen_time,
            distraction_count: self.count_distractions(),
            engagement_level: self.assess_engagement(),
            recommendation: self.get_recommendation(),
        }
    }
}
```

### 4.9 工业与军事应用

**应用场景**：
- 飞行员/宇航员状态监测
- 手术室免提控制
- 工业检测员注意力监测
- 无人机操作员疲劳检测

#### 4.9.1 飞行员监测系统

```rust
struct PilotMonitoringSystem {
    fatigue_monitor: FatigueMonitor,
    attention_monitor: AttentionMonitor,
    workload_estimator: WorkloadEstimator,
}

impl PilotMonitoringSystem {
    fn assess_pilot_state(&self) -> PilotState {
        let fatigue = self.fatigue_monitor.assess_fatigue();
        let attention = self.attention_monitor.get_attention_score();
        let workload = self.workload_estimator.estimate_workload();

        // 综合评估
        if fatigue == FatigueLevel::Critical {
            PilotState::Unfit  // 不适合飞行
        } else if attention < 0.6 || workload > 0.9 {
            PilotState::Warning  // 需要警告
        } else {
            PilotState::Normal  // 正常
        }
    }
}

struct WorkloadEstimator {
    scan_pattern: Vec<GazePosition>,  // 扫视模式
    fixation_distribution: HashMap<String, f64>,  // 注视分布
}

impl WorkloadEstimator {
    fn estimate_workload(&self) -> f64 {
        // 高工作负荷特征：
        // 1. 扫视频率增加
        // 2. 注视时间缩短
        // 3. 注视分布更分散

        let scan_frequency = self.scan_pattern.len() as f64 / 60.0;  // 次/秒
        let fixation_entropy = self.calculate_entropy();

        // 正常：5-10次/秒，熵<2.0
        // 高负荷：>15次/秒，熵>3.0

        let workload = (scan_frequency / 15.0).min(1.0) * 0.5
                     + (fixation_entropy / 3.0).min(1.0) * 0.5;

        workload
    }
}
```

**应用价值**：
- 🛫 **航空安全**：实时监测飞行员状态，预防疲劳驾驶
- 🚀 **太空任务**：长时间任务中的宇航员状态监测
- ⚠️ **事故预防**：在危险状态前发出警报

#### 4.9.2 手术室免提控制

```rust
struct SurgicalEogController {
    current_view: MedicalImage,
    zoom_level: f64,
    selected_tool: Option<SurgicalTool>,
}

impl SurgicalEogController {
    fn process_surgeon_gaze(&mut self, heog: f64, veog: f64) -> SurgicalCommand {
        // 外科医生通过眼动控制医学影像

        if veog > 100.0 {
            // 向上看：放大影像
            self.zoom_level *= 1.2;
            SurgicalCommand::ZoomIn
        } else if veog < -100.0 {
            // 向下看：缩小影像
            self.zoom_level /= 1.2;
            SurgicalCommand::ZoomOut
        } else if heog > 100.0 {
            // 向右看：下一张影像
            SurgicalCommand::NextImage
        } else if heog < -100.0 {
            // 向左看：上一张影像
            SurgicalCommand::PreviousImage
        } else {
            SurgicalCommand::None
        }
    }
}

enum SurgicalCommand {
    ZoomIn,
    ZoomOut,
    NextImage,
    PreviousImage,
    RotateImage,
    None,
}
```

**优势**：
- ✅ **无菌操作**：无需触摸屏幕或设备
- ✅ **实时响应**：延迟<100ms
- ✅ **自然交互**：符合外科医生的工作流程

### 4.10 游戏与娱乐

**应用场景**：
- 眼控游戏
- 无障碍游戏（残疾人）
- VR游戏增强交互

#### 4.10.1 眼控射击游戏

```rust
struct EogShooterGame {
    crosshair_position: (f64, f64),
    sensitivity: f64,
}

impl EogShooterGame {
    fn update(&mut self, heog: f64, veog: f64, dt: f64) -> GameAction {
        // 眼动控制准星移动
        self.crosshair_position.0 += heog * self.sensitivity * dt;
        self.crosshair_position.1 += veog * self.sensitivity * dt;

        // 限制在屏幕范围内
        self.crosshair_position.0 = self.crosshair_position.0.clamp(0.0, 1920.0);
        self.crosshair_position.1 = self.crosshair_position.1.clamp(0.0, 1080.0);

        GameAction::MoveCrosshair(self.crosshair_position)
    }

    fn detect_shoot(&self, veog: f64) -> bool {
        // 眨眼 = 射击
        veog.abs() > 150.0
    }
}

enum GameAction {
    MoveCrosshair((f64, f64)),
    Shoot,
    Reload,
    None,
}
```

#### 4.10.2 无障碍游戏设计

```rust
struct AccessibleGameController {
    control_mode: ControlMode,
    difficulty: DifficultyLevel,
}

enum ControlMode {
    EyeOnly,           // 纯眼控
    EyePlusBlink,      // 眼动+眨眼
    EyePlusVoice,      // 眼动+语音
}

impl AccessibleGameController {
    fn adapt_difficulty(&mut self, player_performance: f64) {
        // 根据玩家表现自动调整难度
        if player_performance < 0.3 {
            self.difficulty = DifficultyLevel::Easy;
        } else if player_performance > 0.7 {
            self.difficulty = DifficultyLevel::Hard;
        }
    }
}
```

**游戏类型适配**：

| 游戏类型 | EOG控制方案 | 适用性 |
|---------|------------|--------|
| **射击游戏** | 眼动瞄准+眨眼射击 | ⭐⭐⭐ |
| **策略游戏** | 眼动选择+停留确认 | ⭐⭐⭐⭐⭐ |
| **赛车游戏** | 眼动转向+眨眼加速 | ⭐⭐ |
| **解谜游戏** | 眼动选择物品 | ⭐⭐⭐⭐ |
| **RPG游戏** | 眼动导航+眨眼交互 | ⭐⭐⭐⭐ |

### 4.11 认知负荷与情绪识别

**应用场景**：
- 用户体验（UX）研究
- 广告效果评估
- 情绪计算

#### 4.11.1 认知负荷评估

```rust
struct CognitiveLoadEstimator {
    pupil_diameter: Vec<f64>,     // 瞳孔直径（需要额外传感器）
    blink_rate: f64,              // 眨眼频率
    fixation_duration: Vec<f64>,  // 注视持续时间
}

impl CognitiveLoadEstimator {
    fn estimate_cognitive_load(&self) -> CognitiveLoad {
        // 认知负荷指标：
        // 1. 瞳孔扩大（高负荷）
        // 2. 眨眼减少（高负荷）
        // 3. 注视时间延长（高负荷）

        let avg_fixation = self.fixation_duration.iter().sum::<f64>()
                          / self.fixation_duration.len() as f64;

        let load_score = if self.blink_rate < 10.0 && avg_fixation > 300.0 {
            CognitiveLoad::High
        } else if self.blink_rate < 15.0 && avg_fixation > 200.0 {
            CognitiveLoad::Medium
        } else {
            CognitiveLoad::Low
        };

        load_score
    }
}

enum CognitiveLoad {
    Low,     // 任务简单
    Medium,  // 任务适中
    High,    // 任务困难/过载
}
```

#### 4.11.2 情绪识别

```rust
struct EmotionRecognizer {
    blink_pattern: Vec<BlinkEvent>,
    gaze_pattern: Vec<GazePosition>,
}

impl EmotionRecognizer {
    fn recognize_emotion(&self) -> Emotion {
        // 情绪与眼动的关系：
        // - 焦虑：眨眼频繁，眼动不规律
        // - 兴趣：眨眼减少，注视集中
        // - 疲劳：眨眼持续时间延长
        // - 惊讶：眨眼暂停，眼睛睁大

        let blink_rate = self.blink_pattern.len() as f64 / 60.0;
        let gaze_stability = self.calculate_gaze_stability();

        if blink_rate > 25.0 && gaze_stability < 0.5 {
            Emotion::Anxious
        } else if blink_rate < 10.0 && gaze_stability > 0.8 {
            Emotion::Interested
        } else {
            Emotion::Neutral
        }
    }
}

enum Emotion {
    Neutral,
    Interested,
    Anxious,
    Tired,
    Surprised,
}
```

---

## 5. 技术对比

### 5.1 EOG vs Eye-tracking

| 特性 | EOG（眼电图） | Eye-tracking（眼动追踪） |
|------|--------------|------------------------|
| **测量方式** | 电极（角膜-视网膜电位） | 摄像头（瞳孔/角膜反射） |
| **空间精度** | 低（只能检测方向） | 高（精确到0.5°视角） |
| **时间精度** | 高（1000 Hz+） | 中（60-1000 Hz） |
| **成本** | 低（几百元） | 高（几千到几万元） |
| **便携性** | 高（小电极） | 低（需要摄像头） |
| **适用场景** | BCI控制、睡眠监测 | 阅读研究、UI测试 |

### 5.2 EOG-BCI vs 其他BCI技术

| BCI类型 | 学习时间 | 准确率 | 速度 | 成本 | 适用场景 |
|---------|---------|--------|------|------|---------|
| **EOG-BCI** | 5-10分钟 | 90-95% | 快 | $100-500 | 重度残疾、轮椅、疲劳监测 |
| **P300-BCI** | 数小时 | 70-85% | 慢 | $1000-3000 | 拼写器、选择任务 |
| **SSVEP-BCI** | 30分钟 | 85-95% | 快 | $1000-3000 | 高速打字、游戏 |
| **MI-BCI** | 数天 | 60-80% | 慢 | $2000-5000 | 运动康复、轮椅 |
| **侵入式BCI** | 数月 | 95-99% | 很快 | $50000+ | 研究、高端医疗 |

**EOG-BCI的"甜蜜点"**：
- ✅ **最适合**：重度残疾患者的日常辅助（轮椅、通信）
- ✅ **商业化成功**：驾驶员疲劳监测（已有产品）
- ⚠️ **不适合**：需要高精度控制的任务（如机械臂）
- ⚠️ **竞争对手**：摄像头eye-tracking（精度更高但成本也高）

---

## 6. BCIF实现建议

### 6.1 模块架构

```rust
// bcif-eog crate
pub mod detection {
    // 底层：信号检测
    pub fn find_blinks(eog: &[f64], sfreq: f64) -> Vec<BlinkEvent>;
    pub fn find_saccades(heog: &[f64], veog: &[f64]) -> Vec<SaccadeEvent>;
    pub fn classify_blink_type(blink: &BlinkEvent) -> BlinkType;
}

pub mod removal {
    // 中层：伪迹去除
    pub fn ica_remove_eog(eeg: &Array2<f64>, eog: &[f64]) -> Array2<f64>;
    pub fn regression_remove_eog(eeg: &Array2<f64>, eog: &[f64]) -> Array2<f64>;
}

pub mod features {
    // 中层：特征提取
    pub fn calculate_fatigue_score(blinks: &[BlinkEvent]) -> f64;
    pub fn decode_eye_command(heog: f64, veog: f64) -> EyeCommand;
    pub fn calculate_attention_score(blinks: &[BlinkEvent], saccades: &[SaccadeEvent]) -> f64;
}

pub mod applications {
    // 高层：应用接口
    pub struct EogSpeller;
    pub struct EogWheelchair;
    pub struct FatigueMonitor;
    pub struct SleepStageClassifier;
}
```

### 6.2 核心数据结构

```rust
#[derive(Debug, Clone)]
pub struct BlinkEvent {
    pub sample_idx: usize,      // 峰值位置
    pub timestamp: f64,         // 时间戳（秒）
    pub amplitude: f64,         // 幅度（μV）
    pub duration: f64,          // 持续时间（秒）
    pub blink_type: BlinkType,  // 眨眼类型
}

#[derive(Debug, Clone, Copy)]
pub enum BlinkType {
    Natural,    // 自然眨眼
    Voluntary,  // 主动眨眼
    Fatigue,    // 疲劳眨眼
}

#[derive(Debug, Clone)]
pub struct SaccadeEvent {
    pub start_idx: usize,
    pub end_idx: usize,
    pub direction: SaccadeDirection,
    pub amplitude: f64,  // 眼动幅度（度）
}

#[derive(Debug, Clone, Copy)]
pub enum SaccadeDirection {
    Left,
    Right,
    Up,
    Down,
}

#[derive(Debug, Clone, Copy)]
pub enum EyeCommand {
    Left,
    Right,
    Up,
    Down,
    Blink,
    DoubleBlink,
    None,
}
```

### 6.3 依赖的Rust Crate

| 功能 | Rust Crate | 用途 |
|------|-----------|------|
| **滤波** | `idsp` | 带通滤波（1-10 Hz） |
| **FFT** | `realfft` | 频谱分析（区分眨眼和肌电） |
| **ICA** | `petal-decomposition` | 去除EOG伪迹 |
| **统计** | `statrs` | 计算均值、标准差、百分位数 |
| **数组** | `ndarray` | 多维数组操作 |

### 6.4 设计权衡

**三种使用模式**：

1. **纯EEG-BCI**（P300/SSVEP）
   - EOG是伪迹，必须去除
   - 使用 `bcif-eog::removal` 模块

2. **混合BCI**（EEG+EOG）
   - EOG是额外的控制通道
   - 同时使用 `removal` 和 `features` 模块

3. **纯EOG-BCI**
   - 只用EOG，不需要复杂的EEG分析
   - 使用 `detection` + `features` + `applications` 模块

**BCIF建议**：
- **Layer 2（预处理）**：提供ICA去除EOG的功能
- **Layer 4（应用）**：提供EOG-BCI控制接口
- **让用户根据应用场景选择是"去除"还是"利用"EOG**

---

## 参考文献

1. MNE-Python Documentation: https://mne.tools/stable/index.html
2. Brainstorm3 Preprocessing Guide
3. AASM睡眠分期标准
4. Seeing Machines Driver Monitoring Systems

---

## 总结

**EOG信号的价值取决于应用目标**：

- 🧠 **认知神经科学研究**（分析ERP）→ EOG是需要去除的噪声
- ♿ **辅助技术**（帮助残疾人控制设备）→ EOG是宝贵的控制信号
- 🚗 **疲劳监测**（驾驶安全）→ EOG是关键的状态指标
- 😴 **睡眠研究**（REM检测）→ EOG是必需的生理标记

**BCIF项目应该同时支持这两种用途，让用户根据需求选择。**

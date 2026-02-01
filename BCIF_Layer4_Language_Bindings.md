# BCIF Layer 4: Language Bindings 架构设计

> **Status**: ✅ 设计决策已完成
> **Version**: 0.1.0
> **Date**: 2026-02-02
> **Purpose**: Layer 4 语言绑定层的详细架构与设计规范

---

## 1. Layer 4 在整体架构中的位置

```
┌─────────────────────────────────────────────────────────────────┐
│ Layer 5: Applications (应用层) 【独立 Repo】                     │
│   Desktop GUI, Mobile App, Web App, CLI                         │
└─────────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────────┐
│ Layer 4: Language Bindings (语言绑定层) ★ 当前设计层 ★           │
│                                                                 │
│   ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐       │
│   │bcif-ffi│ │bcif-py │ │bcif-   │ │bcif-   │ │bcif-kt │       │
│   │ C FFI  │ │ Python │ │ wasm   │ │ sharp  │ │ Kotlin │       │
│   └────────┘ └────────┘ └────────┘ └────────┘ └────────┘       │
│                                                                 │
│   核心职责：让其他语言能调用 BCIF Core                           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────────┐
│ Layer 0-3: BCIF Core (Rust)                                     │
│   Data I/O, Core Numerics, Algorithm, Pipeline                  │
└──────────────────────────────────────────────────────────���──────┘
```

---

## 2. 设计决策汇总

| # | 决策项 | 决策结果 | 理由 |
|---|--------|----------|------|
| 1 | 基础接口 | C FFI | 所有语言都能调用的"通用接口" |
| 2 | Python 绑定 | PyO3 直接绑定 | 性能好，numpy 零拷贝 |
| 3 | Node.js 绑定 | napi-rs 直接绑定 | 原生性能，TS 支持 |
| 4 | WASM 绑定 | wasm-bindgen | 浏览器唯一方式 |
| 5 | C# 绑定 | P/Invoke + 封装 | GUI 需要 |
| 6 | Kotlin 绑定 | JNI + cinterop | KMP 移动端需要 |
| 7 | API 风格 | 极简声明式 | 3 个核心函数 + 可组合步骤 |
| 8 | CLI/TUI | 放 Layer 5 | 是应用，不是绑定 |

---

## 3. 各绑定调用方式

| 绑定 | 目标平台 | 技术 | 产物 |
|------|----------|------|------|
| bcif-ffi | 所有 | C FFI | .dll / .so / .dylib / .a |
| bcif-py | Python | PyO3 | pip install bcif |
| bcif-wasm | 浏览器 | wasm-bindgen | npm @bcif/wasm |
| bcif-node | Node.js | napi-rs | npm @bcif/node |
| bcif-sharp | C# (.NET) | P/Invoke | NuGet BcifSharp |
| bcif-kt | Kotlin (KMP) | JNI + cinterop | Maven bcif-kt |

---

## 4. 架构图

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                   应用层 (Layer 5)                               │
│                                                                                 │
│   ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐        │
│   │ C# GUI    │ │ Python    │ │ Web App   │ │ Node.js   │ │ Mobile    │        │
│   │ Avalonia  │ │ 脚本/分析 │ │ 浏览器    │ │ 服务端    │ │ KMP+CMP   │        │
│   └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘        │
│         │             │             │             │             │              │
└─────────┼─────────────┼─────────────┼─────────────┼─────────────┼──────────────┘
          │             │             │             │             │
          ▼             ▼             ▼             ▼             ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              语言绑定层 (Layer 4)                                │
│                                                                                 │
│   ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐ ┌───────────┐        │
│   │bcif-sharp │ │ bcif-py   │ │ bcif-wasm │ │ bcif-node │ │ bcif-kt   │        │
│   │ C# 封装   │ │ PyO3      │ │ WASM      │ │ napi-rs   │ │ Kotlin    │        │
│   └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘ └─────┬─────┘        │
│         │             │             │             │             │              │
│    P/Invoke       直接绑定     wasm-bindgen    直接绑定     JNI/cinterop       │
│         │             │             │             │             │              │
│         ▼             │             │             │             ▼              │
│   ┌─────────────────────────────────────────────────────────────────────────┐   │
│   │                         bcif-ffi (C FFI 基础层)                          │   │
│   │                                                                          │   │
│   │   输出: bcif.dll (Win) / libbcif.so (Linux) / libbcif.dylib (macOS)     │   │
│   │         libbcif.a (iOS/Android 静态库)                                   │   │
│   └─────────────────────────────────────────────────────────────────────────┘   │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                                        │
                                        ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           BCIF Core (Rust) Layer 0-3                             │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 5. API 设计风格：极简声明式

### 5.1 核心原则

1. **极简 (Minimal)**
   - 能用 1 个参数解决的，不用 2 个
   - 能省略的参数，提供合理默认值

2. **可猜测 (Guessable)**
   - 用户不看文档也能猜到 API 名称
   - 命名符合领域惯例

3. **一致 (Consistent)**
   - 相似操作用相似的 API 模式
   - 参数顺序一致

4. **渐进式复杂度 (Progressive Complexity)**
   - 简单任务用简单 API
   - 复杂需求时才暴露高级选项

### 5.2 统一 API 模式（所有语言一致）

**核心 API（3 个函数搞定 80% 场景）**

```
bcif.read(path)                    → 读取文件，返回 (data, info)
bcif.process(data, info, steps)    → 处理数据，返回处理后的 data
bcif.stream(name)                  → 连接实时流，返回 stream 对象
```

**处理步骤（声明式，可组合）**

```
滤波：
├── bcif.bandpass(low, high)       带通滤波
├── bcif.highpass(freq)            高通滤波
├── bcif.lowpass(freq)             低通滤波
└── bcif.notch(freq)               陷波滤波

通道：
├── bcif.pick(channels)            选择通道
└── bcif.rereference(type)         重参考

其他：
└── bcif.resample(rate)            重采样
```

### 5.3 字符串 DSL 支持

**格式**: `"操作名(参数1, 参数2, ...)"`

**示例**:
- `"bandpass(1, 40)"` → `bcif.bandpass(1, 40)`
- `"notch(50)"` → `bcif.notch(50)`
- `"pick(C3, C4, O1)"` → `bcif.pick(["C3", "C4", "O1"])`
- `"rereference(average)"` → `bcif.rereference("average")`

**优点**:
- 极简：一行字符串描述整个处理流程
- 可序列化：可以保存到配置文件
- GUI 友好：用户在界面输入，直接执行
- 跨语言一致：所有语言用相同的字符串格式

---

## 6. 各语言 API 示例

### 6.1 Python

```python
import bcif

# 最简用法（一行处理）
data, info = bcif.read("file.edf")
data = bcif.process(data, info, ["bandpass(1,40)", "notch(50)"])

# 标准用法
data = bcif.process(data, info, [
    bcif.bandpass(1, 40),
    bcif.notch(50),
    bcif.pick(["C3", "C4"]),
])

# 实时流
stream = bcif.stream("EEG").apply(bcif.bandpass(8, 30)).start()
while True:
    data = stream.get(seconds=1)
    # 处理...
```

### 6.2 C#

```csharp
using BcifSharp;

// 最简用法
var (data, info) = Bcif.Read("file.edf");
data = Bcif.Process(data, info, "bandpass(1,40)", "notch(50)");

// 标准用法
data = Bcif.Process(data, info,
    Bcif.Bandpass(1, 40),
    Bcif.Notch(50),
    Bcif.Pick("C3", "C4")
);

// 实时流（GUI 场景）
using var stream = Bcif.Stream("EEG")
    .Apply(Bcif.Bandpass(8, 30))
    .Build();

stream.OnData += (data) => waveformView.Update(data);
stream.Start();
```

### 6.3 TypeScript (Node.js)

```typescript
import * as bcif from '@bcif/node';

// 最简用法
const { data, info } = bcif.read('file.edf');
const processed = bcif.process(data, info, ['bandpass(1,40)', 'notch(50)']);

// 标准用法
const processed = bcif.process(data, info, [
    bcif.bandpass(1, 40),
    bcif.notch(50),
]);
```

### 6.4 TypeScript (浏览器 WASM)

```typescript
import init, { read, process, bandpass, notch } from '@bcif/wasm';

// 初始化 WASM
await init();

// 读取文件
const file = await fetch('data.edf');
const buffer = await file.arrayBuffer();
const { data, info } = read(new Uint8Array(buffer));

// 处理
const processed = process(data, info, [bandpass(1, 40), notch(50)]);
```

### 6.5 Kotlin

```kotlin
import com.bcif.*

// 最简用法
val (data, info) = Bcif.read("file.edf")
val processed = Bcif.process(data, info, "bandpass(1,40)", "notch(50)")

// 标准用法
val processed = Bcif.process(data, info,
    bandpass(1.0, 40.0),
    notch(50.0),
    pick("C3", "C4")
)
```

---

## 7. bcif-ffi (C FFI) 设计

### 7.1 命名规范

```
bcif_<模块>_<操作>
```

**示例**:
- `bcif_read()`
- `bcif_process()`
- `bcif_stream_new()`
- `bcif_stream_start()`
- `bcif_stream_get()`
- `bcif_stream_free()`
- `bcif_get_error()`

### 7.2 核心函数

```c
// === 文件读取 ===
int bcif_read(
    const char* path,
    double** out_data,
    int* out_n_channels,
    int* out_n_samples,
    BcifInfo** out_info
);

// === 数据处理 ===
int bcif_process(
    double* data,
    int n_channels,
    int n_samples,
    const BcifInfo* info,
    const char* steps_json  // JSON 格式的处理步骤
);

// === 实时流 ===
BcifStream* bcif_stream_new(const char* name, double buffer_seconds);
int bcif_stream_apply(BcifStream* stream, const char* step_json);
int bcif_stream_start(BcifStream* stream);
int bcif_stream_get(BcifStream* stream, double seconds, double** out_data, int* out_samples);
void bcif_stream_free(BcifStream* stream);

// === 错误处理 ===
const char* bcif_get_error();

// === 内存管理 ===
void bcif_free_data(double* data);
void bcif_free_info(BcifInfo* info);
```

### 7.3 设计原则

1. 所有函数返回 int (0=成功, 非0=错误码)
2. 输出参数用指针
3. 字符串用 `const char*`
4. 复杂参数用 JSON 字符串
5. 资源用 `_new`/`_free` 配对
6. 错误信息通过 `bcif_get_error()` 获取

---

## 8. Kotlin + KMP 调用架构

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                    bcif-mobile (KMP + CMP)                                       │
│                                                                                 │
│   ┌─────────────────┐              ┌─────────────────┐                          │
│   │ Android App     │              │ iOS App         │                          │
│   │ (Compose)       │              │ (Compose)       │                          │
│   └────────┬────────┘              └────────┬────────┘                          │
│            │                                │                                    │
│            └────────────┬───────────────────┘                                    │
│                         │                                                        │
│                         ▼                                                        │
│            ┌─────────────────────────┐                                          │
│            │ Kotlin Common Code      │                                          │
│            │ (共享业务逻辑)           │                                          │
│            └────────────┬────────────┘                                          │
│                         │                                                        │
└─────────────────────────┼────────────────────────────────────────────────────────┘
                          │ 调用
                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         bcif-kt                                                  │
│                                                                                 │
│   ┌─────────────────┐              ┌─────────────────┐                          │
│   │ Android         │              │ iOS             │                          │
│   │ JNI → libbcif.so│              │ cinterop →      │                          │
│   │                 │              │ libbcif.a       │                          │
│   └─────────────────┘              └─────────────────┘                          │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────────────────────┐
│                         bcif-ffi (C FFI)                                         │
│                                                                                 │
│   Android: libbcif.so (ARM64, ARMv7, x86_64)                                    │
│   iOS: libbcif.a (ARM64, x86_64 simulator)                                      │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
```

---

## 9. 项目结构

```
bcif/
├── bcif-core/              # Rust 核心 (Layer 0-3)
│
├── bcif-ffi/               # C FFI 导出
│   ├── src/
│   │   ├── lib.rs          # FFI 函数实现
│   │   ├── types.rs        # C 兼容类型
│   │   └── error.rs        # 错误处理
│   ├── include/
│   │   └── bcif.h          # C 头文件
│   └── Cargo.toml
│
├── bcif-py/                # Python 绑定 (PyO3)
│   ├── src/
│   │   ├── lib.rs
│   │   └── ...
│   ├── bcif.pyi            # 类型提示
│   └── pyproject.toml
│
├── bcif-wasm/              # WASM 绑定
│   ├── src/
│   └── Cargo.toml
│
├── bcif-node/              # Node.js 绑定 (napi-rs)
│   ├── src/
│   ├── index.d.ts
│   └── package.json
│
├── bindings/
│   ├── csharp/             # C# 封装
│   │   ├── BcifSharp/
│   │   │   ├── Bcif.cs
│   │   │   ├── Native.cs   # P/Invoke 声明
│   │   │   └── ...
│   │   └── BcifSharp.sln
│   │
│   └── kotlin/             # Kotlin 封装
│       ├── bcif-kt/
│       │   ├── src/
│       │   │   ├── commonMain/   # KMP 共享代码
│       │   │   ├── androidMain/  # Android JNI
│       │   │   └── iosMain/      # iOS cinterop
│       │   └── build.gradle.kts
│       └── settings.gradle.kts
│
└── bcif-cli/               # CLI 工具 (Layer 5，但在主 Repo)
    ├── src/
    └── Cargo.toml
```

---

## 10. 实现优先级

**Phase 1: 基础（必须先完成）**
- bcif-core (Layer 0-3)
- bcif-ffi (C FFI)

**Phase 2: 核心绑定**
- bcif-sharp (C#) ← Desktop GUI 需要
- bcif-py (Python) ← 数据科学家需要

**Phase 3: CLI**
- bcif-cli ← 批处理脚本需要

**Phase 4: 扩展绑定**
- bcif-kt (Kotlin) ← Mobile App 需要
- bcif-wasm (浏览器) ← Web App 需要
- bcif-node (Node.js) ← 服务端需要

---

## 11. 与 Layer 5 的关系

**Layer 4 (语言绑定) 提供：**
- 库/包（开发者使用）
- 无 UI
- 在 BCIF 主 Repo

**Layer 5 (应用) 使用：**
- 可执行程序（最终用户使用）
- 有 UI（GUI/CLI/Mobile）
- 独立 Repo

**Layer 5 应用列表：**
| 应用 | 描述 | 位置 |
|------|------|------|
| bcif-studio | Desktop GUI (C# + Avalonia) | 独立 Repo |
| bcif-mobile | Mobile App (Kotlin + KMP + CMP) | 独立 Repo |
| bcif-web | Web App (TypeScript + React) | 独立 Repo |
| bcif-cli | CLI 工具 | 主 Repo 内 |

---

## 12. 代码量估算

| 模块 | 估算行数 | 说明 |
|------|----------|------|
| bcif-ffi | ~500 行 | C FFI 导出 |
| bcif-py | ~400 行 | PyO3 绑定 |
| bcif-wasm | ~300 行 | WASM 绑定 |
| bcif-node | ~300 行 | napi-rs 绑定 |
| bcif-sharp | ~400 行 | C# 封装 |
| bcif-kt | ~500 行 | Kotlin 封装 (含 KMP) |
| bcif-cli | ~300 行 | CLI 工具 |
| **总计** | **~2700 行** | |

---

## 13. 参考资料

| 资源 | 说明 |
|------|------|
| PyO3 | https://pyo3.rs |
| wasm-bindgen | https://rustwasm.github.io/wasm-bindgen |
| napi-rs | https://napi.rs |
| Kotlin/Native cinterop | https://kotlinlang.org/docs/native-c-interop.html |
| uniffi (备选) | https://mozilla.github.io/uniffi-rs |

---

**Document Version**: 0.1.0
**Last Updated**: 2026-02-02
**Status**: ✅ 设计决策已完成

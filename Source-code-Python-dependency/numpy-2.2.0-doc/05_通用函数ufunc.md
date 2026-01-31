# NumPy 通用函数 (ufunc) 详解

## 1. 概述

通用函数（universal function，简称 ufunc）是 NumPy 的核心特性之一，提供对数组元素逐个操作的高效向量化函数。

### 1.1 什么是 ufunc

ufunc 是一个对 `ndarray` 进行元素级操作的函数，具有以下特性：

1. **向量化**: 自动应用于数组的每个元素
2. **广播**: 支持不同形状数组的操作
3. **类型转换**: 自动处理类型转换和提升
4. **快速**: C 语言实现，高度优化
5. **输出灵活**: 支持多种输出选项

### 1.2 ufunc vs 普通函数

```python
import numpy as np

# 普通 Python 函数（慢）
def python_add(a, b):
    result = []
    for i in range(len(a)):
        result.append(a[i] + b[i])
    return result

# ufunc（快）
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.add(a, b)  # [5, 7, 9]
```

## 2. ufunc 的类型

### 2.1 算术运算

```python
np.add(x, y)          # x + y
np.subtract(x, y)     # x - y
np.multiply(x, y)     # x * y
np.divide(x, y)       # x / y
np.true_divide(x, y)  # x / y (真除法)
np.floor_divide(x, y) # x // y (向下取整除法)
np.mod(x, y)          # x % y (取模)
np.remainder(x, y)    # 余数
np.divmod(x, y)       # (x // y, x % y)
np.power(x, y)        # x ** y
np.float_power(x, y)  # x ** y (浮点结果)
```

**示例**:
```python
a = np.array([10, 20, 30])
b = np.array([3, 7, 11])

print(np.add(a, b))         # [13 27 41]
print(np.divide(a, b))      # [3.33... 2.85... 2.72...]
print(np.floor_divide(a, b)) # [3 2 2]
print(np.mod(a, b))         # [1 6 8]
```

### 2.2 比较运算

```python
np.equal(x, y)         # x == y
np.not_equal(x, y)     # x != y
np.less(x, y)          # x < y
np.less_equal(x, y)    # x <= y
np.greater(x, y)       # x > y
np.greater_equal(x, y) # x >= y
```

**返回值**: 布尔数组

```python
a = np.array([1, 2, 3, 4])
b = np.array([1, 3, 2, 4])

print(np.equal(a, b))        # [True False False True]
print(np.greater(a, b))      # [False False True False]
```

### 2.3 逻辑运算

```python
np.logical_and(x, y)   # x & y (逻辑与)
np.logical_or(x, y)    # x | y (逻辑或)
np.logical_xor(x, y)   # x ^ y (逻辑异或)
np.logical_not(x)      # ~x (逻辑非)
```

**示例**:
```python
a = np.array([True, True, False, False])
b = np.array([True, False, True, False])

print(np.logical_and(a, b))  # [True False False False]
print(np.logical_or(a, b))   # [True True True False]
print(np.logical_xor(a, b))  # [False True True False]
```

### 2.4 位运算

```python
np.bitwise_and(x, y)      # x & y
np.bitwise_or(x, y)       # x | y
np.bitwise_xor(x, y)      # x ^ y
np.bitwise_not(x)         # ~x
np.left_shift(x, y)       # x << y
np.right_shift(x, y)      # x >> y
np.invert(x)              # ~x (等价于 bitwise_not)
```

**示例**:
```python
a = np.array([0b1100, 0b1010], dtype=np.uint8)
b = np.array([0b1010, 0b0101], dtype=np.uint8)

print(np.bitwise_and(a, b))   # [8 0] (0b1000, 0b0000)
print(np.bitwise_or(a, b))    # [14 15] (0b1110, 0b1111)
print(np.left_shift(a, 1))    # [24 20] (左移1位)
```

### 2.5 三角函数

```python
# 基础三角函数
np.sin(x)       # 正弦
np.cos(x)       # 余弦
np.tan(x)       # 正切

# 反三角函数
np.arcsin(x)    # 反正弦
np.arccos(x)    # 反余弦
np.arctan(x)    # 反正切
np.arctan2(y, x) # atan2(y, x)

# 双曲三角函数
np.sinh(x)      # 双曲正弦
np.cosh(x)      # 双曲余弦
np.tanh(x)      # 双曲正切

# 反双曲三角函数
np.arcsinh(x)   # 反双曲正弦
np.arccosh(x)   # 反双曲余弦
np.arctanh(x)   # 反双曲正切
```

**示例**:
```python
angles = np.array([0, np.pi/6, np.pi/4, np.pi/3, np.pi/2])
print(np.sin(angles))
# [0.  0.5  0.707...  0.866...  1.]

# 弧度和角度转换
np.deg2rad(180)  # π
np.rad2deg(np.pi)  # 180
```

### 2.6 指数和对数

```python
# 指数
np.exp(x)       # e^x
np.exp2(x)      # 2^x
np.expm1(x)     # e^x - 1 (精确计算小x)

# 对数
np.log(x)       # ln(x) (自然对数)
np.log2(x)      # log2(x)
np.log10(x)     # log10(x)
np.log1p(x)     # ln(1+x) (精确计算小x)

# 特殊指数/对数函数
np.logaddexp(x, y)   # log(exp(x) + exp(y))
np.logaddexp2(x, y)  # log2(2^x + 2^y)
```

**示例**:
```python
x = np.array([1, 2, np.e, 10])
print(np.log(x))     # [0.  0.693...  1.  2.302...]
print(np.log10(x))   # [0.  0.301...  0.434...  1.]

# 避免精度问题
small = 1e-10
print(np.log(1 + small))   # 可能不精确
print(np.log1p(small))     # 更精确
```

### 2.7 开方和幂函数

```python
np.sqrt(x)      # √x
np.cbrt(x)      # ∛x (立方根)
np.square(x)    # x²
np.reciprocal(x) # 1/x
```

### 2.8 取整函数

```python
np.floor(x)     # 向下取整
np.ceil(x)      # 向上取整
np.trunc(x)     # 截断到零
np.rint(x)      # 四舍五入到最近整数
np.round(x)     # 四舍五入（可指定小数位）
```

**示例**:
```python
x = np.array([-2.7, -1.5, -0.2, 0.2, 1.5, 2.7])
print(np.floor(x))   # [-3. -2. -1.  0.  1.  2.]
print(np.ceil(x))    # [-2. -1. -0.  1.  2.  3.]
print(np.trunc(x))   # [-2. -1. -0.  0.  1.  2.]
print(np.rint(x))    # [-3. -2. -0.  0.  2.  3.]
```

### 2.9 符号和绝对值

```python
np.abs(x)       # |x| (绝对值)
np.absolute(x)  # |x| (等价于 abs)
np.fabs(x)      # |x| (浮点绝对值)
np.sign(x)      # 符号函数 (-1, 0, 1)
np.copysign(x, y) # x 的值，y 的符号
```

### 2.10 最大最小函数

```python
np.maximum(x, y)    # 元素级最大值
np.minimum(x, y)    # 元素级最小值
np.fmax(x, y)       # 最大值（忽略 NaN）
np.fmin(x, y)       # 最小值（忽略 NaN）
```

**示例**:
```python
a = np.array([1, 5, 3])
b = np.array([4, 2, 6])
print(np.maximum(a, b))  # [4 5 6]

# 处理 NaN
a = np.array([1, np.nan, 3])
b = np.array([4, 2, np.nan])
print(np.maximum(a, b))  # [4. nan nan]
print(np.fmax(a, b))     # [4. 2. 3.]
```

### 2.11 浮点运算

```python
np.isfinite(x)  # 是否有限
np.isinf(x)     # 是否无穷
np.isnan(x)     # 是否 NaN
np.signbit(x)   # 符号位是否设置
np.copysign(x, y) # 复制符号
np.nextafter(x, y) # 下一个浮点数
np.spacing(x)   # 与最近浮点数的距离
np.ldexp(x, i)  # x * 2^i
np.frexp(x)     # (m, e) 使得 x = m * 2^e
np.modf(x)      # (小数部分, 整数部分)
```

### 2.12 其他数学函数

```python
np.hypot(x, y)      # √(x² + y²)
np.heaviside(x, h0) # Heaviside 阶跃函数
np.convolve(a, v)   # 卷积
np.clip(a, min, max) # 限制在范围内
np.real(x)          # 实部
np.imag(x)          # 虚部
np.conj(x)          # 共轭
np.angle(x)         # 相位角
```

## 3. ufunc 的方法

每个 ufunc 都有以下方法：

### 3.1 reduce

沿指定轴聚合数组：

```python
# 求和
np.add.reduce(arr)  # 等价于 arr.sum()

# 求积
np.multiply.reduce(arr)  # 等价于 arr.prod()

# 沿轴操作
arr = np.array([[1, 2, 3], [4, 5, 6]])
np.add.reduce(arr, axis=0)  # [5 7 9] (列和)
np.add.reduce(arr, axis=1)  # [6 15] (行和)
```

**示例**:
```python
a = np.array([1, 2, 3, 4])
print(np.add.reduce(a))       # 10
print(np.multiply.reduce(a))  # 24
print(np.maximum.reduce(a))   # 4
```

### 3.2 accumulate

累积操作：

```python
a = np.array([1, 2, 3, 4])
print(np.add.accumulate(a))       # [1 3 6 10] (累加)
print(np.multiply.accumulate(a))  # [1 2 6 24] (累乘)
```

等价于：
```python
np.cumsum(a)   # 累加
np.cumprod(a)  # 累乘
```

### 3.3 reduceat

在指定的索引位置进行 reduce：

```python
a = np.array([0, 1, 2, 3, 4, 5, 6, 7])
indices = [0, 3, 5]
result = np.add.reduceat(a, indices)
# [3, 12, 18]
# [0+1+2, 3+4+5+6, 7]
```

### 3.4 outer

计算外积：

```python
a = np.array([1, 2, 3])
b = np.array([10, 20, 30, 40])
result = np.multiply.outer(a, b)
# [[ 10  20  30  40]
#  [ 20  40  60  80]
#  [ 30  60  90 120]]
```

等价于：
```python
result = a[:, np.newaxis] * b
```

### 3.5 at

在指定索引处执行原地操作：

```python
a = np.array([1, 2, 3, 4, 5])
indices = [0, 2, 4]
np.add.at(a, indices, 10)
print(a)  # [11  2 13  4 15]

# 处理重复索引
a = np.zeros(5)
indices = [0, 0, 1, 1, 1]
np.add.at(a, indices, 1)
print(a)  # [2. 3. 0. 0. 0.]
```

## 4. ufunc 的属性

### 4.1 基本属性

```python
ufunc = np.add

# 输入数量
print(ufunc.nin)      # 2 (二元函数)

# 输出数量
print(ufunc.nout)     # 1

# 参数总数
print(ufunc.nargs)    # 3 (nin + nout)

# 函数名
print(ufunc.__name__) # 'add'

# 身份元素（用于 reduce）
print(ufunc.identity) # 0 (对于 add)
```

### 4.2 类型信息

```python
# 支持的类型
print(np.add.types)
# ['??->?', 'bb->b', 'BB->B', 'hh->h', ...]

# 每种类型的签名
```

## 5. 广播（Broadcasting）

### 5.1 广播规则

NumPy 在执行 ufunc 时自动广播数组：

1. 如果数组维度不同，在较小数组的形状前面填充 1
2. 对于每个维度，如果大小相同或其中一个为 1，则兼容
3. 大小为 1 的维度会被"拉伸"以匹配另一个数组

**示例**:
```python
# 标量广播
a = np.array([1, 2, 3])
result = a + 10  # [11, 12, 13]

# 一维到二维
a = np.array([[1, 2, 3],
              [4, 5, 6]])  # shape (2, 3)
b = np.array([10, 20, 30])  # shape (3,)
result = a + b
# [[11 22 33]
#  [14 25 36]]

# 二维到二维
a = np.array([[1], [2], [3]])  # shape (3, 1)
b = np.array([[10, 20, 30]])   # shape (1, 3)
result = a + b
# [[11 21 31]
#  [12 22 32]
#  [13 23 33]]
```

### 5.2 检查广播兼容性

```python
# 获取广播后的形状
shape = np.broadcast_shapes((3, 1), (1, 4))
# (3, 4)

# 创建广播迭代器
a = np.array([[1], [2], [3]])
b = np.array([[10, 20]])
bc = np.broadcast(a, b)
print(bc.shape)  # (3, 2)
```

### 5.3 显式广播

```python
# broadcast_to
a = np.array([1, 2, 3])
broadcasted = np.broadcast_to(a, (4, 3))
# [[1 2 3]
#  [1 2 3]
#  [1 2 3]
#  [1 2 3]]

# broadcast_arrays
a = np.array([[1], [2], [3]])
b = np.array([10, 20, 30])
a_bc, b_bc = np.broadcast_arrays(a, b)
# a_bc.shape = (3, 3)
# b_bc.shape = (3, 3)
```

## 6. 输出参数

### 6.1 out 参数

将结果写入预分配的数组：

```python
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = np.empty(3)

np.add(a, b, out=result)
print(result)  # [5. 7. 9.]
```

**优势**:
- 减少内存分配
- 原地操作
- 性能提升

### 6.2 where 参数

条件性应用 ufunc：

```python
a = np.array([1, 2, 3, 4, 5])
b = np.array([10, 20, 30, 40, 50])
condition = np.array([True, False, True, False, True])
result = np.zeros(5)

np.add(a, b, out=result, where=condition)
print(result)
# [11.  0. 33.  0. 55.]
# 只在 condition 为 True 的位置执行加法
```

### 6.3 casting 参数

控制类型转换：

```python
a = np.array([1.5, 2.7], dtype=np.float64)
result = np.empty(2, dtype=np.int32)

# 不安全转换（需要显式允许）
np.floor(a, out=result, casting='unsafe')
print(result)  # [1 2]
```

**casting 选项**:
- `'no'`: 不允许转换
- `'equiv'`: 只允许字节序变化
- `'safe'`: 只允许安全转换
- `'same_kind'`: 同类转换
- `'unsafe'`: 允许任何转换

## 7. 创建自定义 ufunc

### 7.1 使用 frompyfunc

```python
# Python 函数
def my_func(x, y):
    return x + 2 * y

# 转换为 ufunc
my_ufunc = np.frompyfunc(my_func, nin=2, nout=1)

# 使用
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = my_ufunc(a, b)
print(result)  # [9 12 15]
```

**限制**:
- 返回对象数组
- 性能不如 C 实现的 ufunc

### 7.2 使用 vectorize

```python
@np.vectorize
def my_func(x, y):
    return x + 2 * y

result = my_func([1, 2, 3], [4, 5, 6])
```

**更灵活的 vectorize**:
```python
def my_func(x, y):
    return x + 2 * y

# 指定输出类型
vfunc = np.vectorize(my_func, otypes=[np.float64])
result = vfunc(a, b)
```

### 7.3 使用 Numba (推荐)

对于高性能自定义 ufunc，使用 Numba：

```python
from numba import vectorize, float64

@vectorize([float64(float64, float64)])
def my_ufunc(x, y):
    return x + 2 * y

# 编译为真正的 ufunc
a = np.array([1.0, 2.0, 3.0])
b = np.array([4.0, 5.0, 6.0])
result = my_ufunc(a, b)
```

## 8. 性能考虑

### 8.1 向量化 vs 循环

```python
# 慢（Python 循环）
def slow_sum(arr):
    total = 0
    for x in arr:
        total += x
    return total

# 快（ufunc）
def fast_sum(arr):
    return np.add.reduce(arr)
```

### 8.2 内存布局

```python
# C 连续数组更快
arr_c = np.ascontiguousarray(arr)
result = np.sin(arr_c)

# 避免不必要的拷贝
result = np.sin(arr, out=arr)  # 原地操作
```

### 8.3 减少临时数组

```python
# 低效（多个临时数组）
result = a * b + c * d

# 高效（使用 out 参数）
temp = np.multiply(a, b)
np.multiply(c, d, out=result)
np.add(temp, result, out=result)

# 或使用 einsum
result = np.einsum('i,i,j,j->', a, b, c, d)
```

## 9. 常见用例

### 9.1 条件替换

```python
arr = np.array([1, -2, 3, -4, 5])

# 将负数替换为 0
arr_pos = np.where(arr < 0, 0, arr)
# [1 0 3 0 5]

# 或使用 clip
arr_clipped = np.clip(arr, 0, None)
```

### 9.2 元素级比较

```python
a = np.array([1, 2, 3, 4, 5])
b = np.array([1, 3, 2, 4, 6])

# 找出相同的元素
same = np.equal(a, b)
same_indices = np.where(same)[0]
```

### 9.3 复杂运算

```python
# 计算两点之间的距离
x1, y1 = np.array([0, 1, 2]), np.array([0, 1, 2])
x2, y2 = np.array([3, 4, 5]), np.array([4, 5, 6])

distances = np.hypot(x2 - x1, y2 - y1)
# 或
distances = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
```

## 10. 总结

ufunc 是 NumPy 性能的核心：

1. **高效**: C 实现，向量化操作
2. **灵活**: 支持广播、类型转换、条件操作
3. **丰富**: 涵盖各种数学运算
4. **可扩展**: 可创建自定义 ufunc

掌握 ufunc 对于高效使用 NumPy 至关重要。

---

**相关文档**:
- [02_核心模块详解.md](02_核心模块详解.md)
- [04_数据类型系统.md](04_数据类型系统.md)
- [07_内存管理与性能.md](07_内存管理与性能.md)

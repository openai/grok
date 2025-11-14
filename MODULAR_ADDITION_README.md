# Modular Addition Grokking Experiments

这个模块提供了完整的工具集来研究模加法任务中的 **Grokking 现象**，特别关注训练数据比例 α 对 Grokking 的影响。

## 📋 目录

- [问题定义](#问题定义)
- [快速开始](#快速开始)
- [模块组成](#模块组成)
- [详细使用指南](#详细使用指南)
- [实验示例](#实验示例)
- [结果分析](#结果分析)
- [参数说明](#参数说明)

## 🎯 问题定义

### 模加法任务

我们研究以下形式的模加法问题：

```
(x, y) → (x + y) mod p
```

其中：
- `x, y ∈ Z_p` (整数模 p)
- `p` 是素数
- 数据集大小 = p²（所有可能的 (x, y) 组合）

### Grokking 现象

**Grokking** 是指模型在训练集上快速达到高精度，但在验证集上的泛化能力延迟出现的现象。典型特征：

1. **快速记忆**：训练准确率快速上升到接近 100%
2. **延迟泛化**：验证准确率在很长时间内保持较低水平
3. **突然泛化**：验证准确率突然跃升到接近 100%

### 研究目标

研究训练数据比例 α 如何影响 Grokking：

```
α = 训练数据大小 / 总数据大小 = |训练集| / p²
```

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install torch pytorch-lightning numpy pandas matplotlib sympy
```

### 2. 测试数据生成

```bash
python -m grok.modular_arithmetic
```

输出示例：
```
================================================================================
Modular Addition Dataset Example
================================================================================

Creating dataset with:
  - Modulus (p): 97
  - Train fraction (α): 0.3

Dataset Info:
  modulus: 97
  train_fraction: 0.3
  total_size: 9409
  train_size: 2822
  val_size: 6587
  vocab_size: 100
  sequence_length: 11

Example equations:
  <|eos|> 45 + 23 = 68 <|eos|>
  <|eos|> 12 + 89 = 4 <|eos|>
  ...
```

### 3. 运行单个实验

```bash
python -m grok.train_modular_addition \
    --modulus 97 \
    --train_fraction 0.3 \
    --max_steps 10000 \
    --gpu 0
```

### 4. 运行多个 α 实验

```bash
python scripts/run_alpha_experiments.py \
    --modulus 97 \
    --alpha_values 0.1 0.2 0.3 0.4 0.5 \
    --max_steps 50000 \
    --gpu 0
```

### 5. 可视化结果

```bash
python scripts/visualize_grokking.py \
    --log_dir logs/alpha_experiments \
    --modulus 97
```

## 📦 模块组成

### 1. 数据生成模块 (`grok/modular_arithmetic.py`)

提供模加法数据生成和处理功能：

**核心类：**
- `ModularAdditionTokenizer`: 词元化器
- `ModularAdditionDataset`: 数据集生成器
- `ModularAdditionIterator`: 批次迭代器

**功能：**
- 生成所有可能的模加法方程式
- 按比例划分训练/验证集
- 支持批次处理和洗牌

### 2. 训练模块 (`grok/train_modular_addition.py`)

基于 PyTorch Lightning 的训练框架：

**核心类：**
- `ModularAdditionTransformer`: 训练模型封装

**功能：**
- 集成 Transformer 模型
- AdamW 优化器
- 学习率预热调度
- 自动日志记录（CSV 格式）

### 3. 实验运行器 (`scripts/run_alpha_experiments.py`)

批量运行实验的脚本：

**功能：**
- 自动运行多个 α 值的实验
- 支持多个素数 p
- 支持多个随机种子
- 生成实验摘要

### 4. 可视化工具 (`scripts/visualize_grokking.py`)

分析和可视化 Grokking 现象：

**功能：**
- 绘制训练/验证曲线
- 比较不同 α 值
- 检测 Grokking 延迟
- 生成统计摘要

## 📖 详细使用指南

### 单个实验训练

#### 基本命令

```bash
python -m grok.train_modular_addition \
    --modulus 97 \
    --train_fraction 0.3 \
    --max_steps 50000
```

#### 完整参数示例

```bash
python -m grok.train_modular_addition \
    --modulus 97 \                      # 素数 p
    --train_fraction 0.3 \               # 训练比例 α = 30%
    --num_layers 2 \                     # Transformer 层数
    --num_heads 4 \                      # 注意力头数
    --d_model 128 \                      # 模型维度
    --learning_rate 1e-3 \               # 学习率
    --weight_decay 1.0 \                 # 权重衰减
    --batch_size 512 \                   # 批次大小
    --max_steps 50000 \                  # 最大训练步数
    --warmup_steps 50 \                  # 预热步数
    --seed 0 \                           # 随机种子
    --gpu 0 \                            # GPU 设备
    --log_dir logs/my_experiment         # 日志目录
```

### 批量实验

#### 1. 研究不同 α 值（固定 p）

```bash
python scripts/run_alpha_experiments.py \
    --modulus 97 \
    --alpha_values 0.05 0.1 0.2 0.3 0.4 0.5 0.7 0.9 \
    --max_steps 50000 \
    --log_dir logs/alpha_sweep_p97
```

#### 2. 研究不同 p 值

```bash
python scripts/run_alpha_experiments.py \
    --modulus 59 97 113 \
    --alpha_values 0.2 0.4 0.6 \
    --max_steps 50000 \
    --log_dir logs/modulus_comparison
```

#### 3. 多种子重复实验

```bash
python scripts/run_alpha_experiments.py \
    --modulus 97 \
    --alpha_values 0.3 \
    --seeds 0 1 2 3 4 \
    --max_steps 50000 \
    --log_dir logs/multi_seed
```

#### 4. 快速测试（小规模）

```bash
python scripts/run_alpha_experiments.py \
    --modulus 59 \
    --alpha_values 0.2 0.4 \
    --max_steps 10000 \
    --log_dir logs/quick_test
```

### 结果可视化

#### 1. 可视化单个 p 的所有实验

```bash
python scripts/visualize_grokking.py \
    --log_dir logs/alpha_experiments \
    --modulus 97
```

生成的图表：
- `p97_alpha0.300.png`: 单个实验的训练曲线
- `alpha_comparison_p97.png`: 不同 α 值的比较
- `grokking_analysis_p97.png`: Grokking 现象分析

#### 2. 可视化所有实验

```bash
python scripts/visualize_grokking.py \
    --log_dir logs/alpha_experiments
```

## 💡 实验示例

### 示例 1: 基础 Grokking 观察

**目标**：观察基本的 Grokking 现象

```bash
# 运行实验
python -m grok.train_modular_addition \
    --modulus 97 \
    --train_fraction 0.3 \
    --max_steps 100000 \
    --log_dir logs/basic_grokking

# 可视化
python scripts/visualize_grokking.py \
    --log_dir logs/basic_grokking
```

**预期结果**：
- 训练准确率在 ~1000 步达到 95%+
- 验证准确率在 ~20000 步突然跃升到 95%+
- Grokking 延迟约 19000 步

### 示例 2: α 对 Grokking 的影响

**目标**：研究训练数据比例如何影响 Grokking

```bash
# 运行实验（8 个不同的 α 值）
python scripts/run_alpha_experiments.py \
    --modulus 97 \
    --alpha_values 0.05 0.1 0.2 0.3 0.4 0.5 0.7 0.9 \
    --max_steps 100000 \
    --log_dir logs/alpha_study

# 可视化比较
python scripts/visualize_grokking.py \
    --log_dir logs/alpha_study \
    --modulus 97
```

**预期观察**：
- **小 α (0.05-0.2)**：Grokking 延迟更长，泛化更困难
- **中 α (0.3-0.5)**：明显的 Grokking 现象
- **大 α (0.7-0.9)**：Grokking 延迟减少，可能无明显 Grokking

### 示例 3: 问题难度对比

**目标**：比较不同素数大小的影响

```bash
# 小素数（简单）
python scripts/run_alpha_experiments.py \
    --modulus 59 \
    --alpha_values 0.2 0.4 \
    --max_steps 50000 \
    --log_dir logs/p59

# 中素数（中等）
python scripts/run_alpha_experiments.py \
    --modulus 97 \
    --alpha_values 0.2 0.4 \
    --max_steps 50000 \
    --log_dir logs/p97

# 大素数（困难）
python scripts/run_alpha_experiments.py \
    --modulus 113 \
    --alpha_values 0.2 0.4 \
    --max_steps 100000 \
    --log_dir logs/p113
```

**预期观察**：
- 更大的 p → 更难的问题 → 更长的 Grokking 延迟

## 📊 结果分析

### 日志文件结构

```
logs/
└── alpha_experiments/
    ├── experiment_summary.json          # 所有实验的摘要
    ├── p97_alpha0.300_layers2_heads4_d128_seed0/
    │   ├── hparams.json                 # 超参数
    │   ├── lightning_logs/
    │   │   └── version_0/
    │   │       └── metrics.csv          # 训练指标
    │   └── checkpoints/                 # 模型检查点
    └── plots/                           # 生成的图表
        ├── p97_alpha0.300.png
        ├── alpha_comparison_p97.png
        └── grokking_analysis_p97.png
```

### 关键指标

在 `metrics.csv` 中记录：

- `step`: 训练步数
- `train_loss`: 训练损失
- `train_acc`: 训练准确率
- `val_loss`: 验证损失
- `val_acc`: 验证准确率
- `full_train_acc`: 完整训练集准确率
- `learning_rate`: 当前学习率

### Grokking 检测标准

脚本自动检测 Grokking：

```python
# 定义"解决"标准：准确率 > 95%
train_solve_step = 训练准确率首次超过 95% 的步数
val_solve_step = 验证准确率首次超过 95% 的步数

# Grokking 延迟
grokking_delay = val_solve_step - train_solve_step

# 判断是否出现 Grokking
grokking_detected = (grokking_delay > 100)
```

## 🔧 参数说明

### 数据参数

| 参数 | 说明 | 默认值 | 推荐范围 |
|------|------|--------|----------|
| `--modulus` | 素数 p | 97 | 59, 97, 113 |
| `--train_fraction` | 训练比例 α | 0.3 | 0.05 - 0.9 |
| `--seed` | 随机种子 | 0 | 任意整数 |

### 模型参数

| 参数 | 说明 | 默认值 | 推荐范围 |
|------|------|--------|----------|
| `--num_layers` | Transformer 层数 | 2 | 1 - 4 |
| `--num_heads` | 注意力头数 | 4 | 2 - 8 |
| `--d_model` | 模型维度 | 128 | 64 - 256 |
| `--dropout` | Dropout 率 | 0.0 | 0.0 - 0.1 |

### 训练参数

| 参数 | 说明 | 默认值 | 推荐范围 |
|------|------|--------|----------|
| `--learning_rate` | 学习率 | 1e-3 | 1e-4 - 1e-2 |
| `--weight_decay` | 权重衰减 | 1.0 | 0.0 - 10.0 |
| `--batch_size` | 批次大小 | 512 | 256 - 1024 |
| `--max_steps` | 最大步数 | 50000 | 10000 - 200000 |
| `--warmup_steps` | 预热步数 | 50 | 10 - 1000 |

### 重要参数说明

#### 1. `train_fraction` (α)

- **核心研究参数**
- 控制用于训练的数据比例
- 对 Grokking 现象影响最大
- 建议测试范围：0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9

#### 2. `weight_decay`

- 对 Grokking 有重要影响
- 更大的权重衰减通常加速 Grokking
- 论文中常用值：1.0
- 建议测试：0.0, 0.1, 1.0, 10.0

#### 3. `max_steps`

- Grokking 可能需要很长时间
- α 较小时需要更多步数
- 建议：
  - 快速测试：10,000
  - 标准实验：50,000
  - 完整观察：100,000+

## 📈 预期实验结果

### 典型 Grokking 曲线

```
准确率 (%)
100 |                                        ╭────────
    |                                    ╭───╯
    |              ╭────────────────────╯
    |          ╭───╯
 50 |      ╭───╯                          验证准确率
    |  ╭───╯
    |──╯                                  训练准确率
  0 |
    +-------------------------------------------------> 训练步数
      0     5k    10k    15k    20k    25k    30k
          ↑                    ↑
      记忆阶段            泛化阶段
      (训练快速收敛)      (验证突然跃升)
```

### α 对 Grokking 的影响

| α 范围 | 训练收敛 | 验证收敛 | Grokking 延迟 | 特征 |
|--------|----------|----------|---------------|------|
| 0.05-0.1 | 快 (~1k步) | 很慢 (>50k步) | 非常长 | 明显 Grokking |
| 0.2-0.3 | 快 (~1k步) | 慢 (~20k步) | 长 | 典型 Grokking |
| 0.4-0.5 | 快 (~1k步) | 中速 (~10k步) | 中等 | 轻微 Grokking |
| 0.7-0.9 | 中速 | 快 | 短或无 | 几乎无 Grokking |

## 🔬 进阶实验

### 1. 权重衰减研究

```bash
for wd in 0.0 0.1 1.0 10.0; do
    python -m grok.train_modular_addition \
        --modulus 97 \
        --train_fraction 0.3 \
        --weight_decay $wd \
        --experiment_name wd_${wd} \
        --max_steps 100000
done
```

### 2. 模型规模研究

```bash
# 小模型
python -m grok.train_modular_addition \
    --modulus 97 --train_fraction 0.3 \
    --num_layers 1 --d_model 64 \
    --experiment_name small_model

# 大模型
python -m grok.train_modular_addition \
    --modulus 97 --train_fraction 0.3 \
    --num_layers 4 --d_model 256 \
    --experiment_name large_model
```

### 3. 学习率影响

```bash
for lr in 1e-4 5e-4 1e-3 5e-3; do
    python -m grok.train_modular_addition \
        --modulus 97 \
        --train_fraction 0.3 \
        --learning_rate $lr \
        --experiment_name lr_${lr}
done
```

## 🐛 常见问题

### 1. CUDA 内存不足

**解决方案**：
```bash
# 减小批次大小
python -m grok.train_modular_addition \
    --batch_size 256 \
    --val_batch_size 256
```

### 2. 训练太慢

**解决方案**：
```bash
# 使用更小的 p 进行测试
python -m grok.train_modular_addition \
    --modulus 59 \
    --max_steps 10000
```

### 3. 无法观察到 Grokking

**可能原因**：
- α 太大（尝试 α < 0.3）
- 训练步数不够（增加 max_steps）
- 权重衰减太小（尝试 weight_decay=1.0）

## 📚 参考资料

### 论文

- **Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets**
  - Alethea Power et al., 2022
  - [arXiv:2201.02177](https://arxiv.org/abs/2201.02177)

### 相关工作

- Modular arithmetic as a testbed for studying generalization
- Phase transitions in neural network training
- Implicit regularization in deep learning

## 📝 实验记录模板

建议创建实验记录文件记录您的观察：

```markdown
# 实验记录

## 实验 1: 基础 Grokking 观察
- **日期**: 2024-XX-XX
- **参数**: p=97, α=0.3
- **观察**:
  - 训练收敛步数: ~1000
  - 验证收敛步数: ~18000
  - Grokking 延迟: ~17000 步
- **结论**: 明显的 Grokking 现象

## 实验 2: α 影响研究
- **日期**: 2024-XX-XX
- **参数**: p=97, α∈{0.1, 0.3, 0.5, 0.7}
- **观察**:
  - α=0.1: 延迟 >40000 步
  - α=0.3: 延迟 ~18000 步
  - α=0.5: 延迟 ~8000 步
  - α=0.7: 几乎无延迟
- **结论**: α 越小，Grokking 延迟越长
```

## 🎓 总结

这套工具提供了完整的框架来研究模加法中的 Grokking 现象。主要特点：

✅ **易用性**：简单的命令行接口
✅ **灵活性**：支持多种参数配置
✅ **可重复性**：固定随机种子
✅ **可视化**：自动生成分析图表
✅ **批量处理**：支持并行实验

祝研究顺利！如有问题，请查看日志文件或提交 Issue。

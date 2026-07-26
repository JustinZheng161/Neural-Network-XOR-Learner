# 神经网络学习 XOR 问题 —— 反向传播可视化（终极版）

本项目完整实现了多层感知机（MLP）神经网络，可学习 XOR 等逻辑门问题，并配有丰富的实时可视化功能，直观展示反向传播、决策边界形成、权重演化等核心过程。

---

## 📌 项目简介

XOR（异或）是经典的线性不可分问题，单层感知机无法解决，而多层神经网络通过**反向传播算法**可以完美拟合。本项目从零实现了一个可配置的神经网络系统，支持：

- 多种激活函数、损失函数、优化器
- 学习率调度、正则化、Dropout
- 实时动画可视化（网络结构、损失曲线、决策边界、权重分布等）
- 模型保存与加载
- 评估指标（准确率、精确率、召回率、F1、混淆矩阵）

通过可视化，你可以直观地看到网络如何从随机权重一步步调整，最终学会拟合目标函数。

---

## ✨ 核心特性

- **完整的反向传播**：手动实现梯度计算，无第三方深度学习库依赖。
- **丰富的激活函数**：Sigmoid、ReLU、Tanh、Leaky ReLU、ELU。
- **多种优化器**：SGD、Momentum、Adam。
- **学习率调度**：常数衰减、指数衰减、余弦退火、阶梯衰减。
- **正则化支持**：L1、L2、Dropout。
- **损失函数**：MSE、二元交叉熵。
- **数据集**：XOR、AND、OR、NAND、XNOR、Circle、Spiral。
- **高级可视化**：
  - 网络结构动态展示（神经元激活值、连接权重）
  - 训练损失/准确率曲线
  - 实时决策边界绘制
  - 权重分布直方图
  - 学习率变化曲线
  - 预测结果表格
- **模型持久化**：保存/加载模型权重及优化器状态。
- **早停机制**：防止过拟合。

---

## 🛠 依赖环境

- Python 3.7+
- NumPy
- Matplotlib

安装依赖：

```bash
pip install numpy matplotlib
```

---

## 🚀 使用方法

### 1. 直接运行

在终端中执行：

```bash
python neural_network_xor.py
```

默认使用 XOR 数据集，网络结构 `[2, 8, 8, 1]`，激活函数 `sigmoid`，优化器 `adam`，训练 1000 个 epoch。训练过程中会弹出可视化窗口，实时展示学习过程。

### 2. 自定义配置

修改 `main()` 函数中的 `config` 字典即可调整参数：

```python
config = {
    'dataset': 'xor',           # 数据集：xor, and, or, nand, xnor, circle, spiral
    'network_structure': [2, 8, 8, 1],  # 输入层→隐藏层→输出层神经元数
    'activation': 'sigmoid',     # 激活函数：sigmoid, relu, tanh, leaky_relu, elu
    'learning_rate': 0.5,        # 初始学习率
    'scheduler_type': 'cosine',  # 调度器：constant, exponential, cosine, step
    'optimizer_type': 'adam',    # 优化器：sgd, momentum, adam
    'loss_type': 'mse',          # 损失函数：mse, binary_crossentropy
    'reg_type': 'none',          # 正则化：none, l1, l2, dropout
    'reg_lambda': 0.01,          # 正则化强度
    'dropout_rate': 0.0,         # Dropout 概率（仅当 reg_type='dropout'）
    'epochs': 1000,              # 总训练轮数
    'steps_per_frame': 10,       # 每帧动画执行的训练步数
    'interval': 50               # 帧间隔（毫秒）
}
```

### 3. 加载已保存的模型

```python
from ModelPersistence import ModelPersistence
nn = ModelPersistence.load_model("trained_model.npz")
# 使用 nn.forward(X) 进行预测
```

---

## 📊 可视化界面说明

运行后将出现一个包含多个子图的窗口：

| 子图区域 | 内容 |
|---------|------|
| 左上（大） | **神经网络结构**：节点颜色代表激活值，连线颜色代表权重正负，粗细代表权重大小 |
| 右上（左） | **训练损失曲线**：实时显示损失下降 |
| 右上（右） | **准确率曲线** |
| 中右 | **决策边界**：背景色表示预测概率，散点为训练样本 |
| 左下 | **权重分布直方图**：各层权重的分布情况 |
| 中下（右） | **学习率变化曲线** |
| 底部 | **预测结果表格**：每个样本的输入、预期值、预测值、误差和正确性 |

---

## 📁 文件结构

```
.
├── neural_network_xor.py    # 主程序（包含所有实现）
├── trained_model.npz        # 训练结束后自动保存的模型（可选）
└── README.md                # 本文件
```

> 所有代码都整合在一个脚本中，便于阅读和实验。

---

## 🔬 技术原理简述

### 反向传播核心步骤

1. **前向传播**：逐层计算线性变换 `z = W·a + b` 和激活 `a = σ(z)`。
2. **计算损失**：使用 MSE 或交叉熵。
3. **反向传播**：从输出层开始，利用链式法则计算每层的误差项 `δ`，并得到梯度 `∂L/∂W` 和 `∂L/∂b`。
4. **参数更新**：使用优化器（SGD/Momentum/Adam）更新权重和偏置。

### 为什么 XOR 需要多层网络？

XOR 的数据在二维空间中是线性不可分的，单层感知机只能学习线性决策边界，而多层网络通过非线性的激活函数可以组合出任意复杂的边界，从而完美分类。

---

## 📈 示例输出（训练 XOR）

```
配置: [2, 8, 8, 1], sigmoid, adam, mse
...
最终 loss: 0.000123
准确率: 100.0%
[混淆矩阵]
  TP: 2   FP: 0
  FN: 0   TN: 2
```

可视化窗口将展示网络如何从随机状态逐渐收敛到正确的映射。

---

## 📝 许可证

本项目仅供学习和研究使用，无特定许可证限制。欢迎自由修改和分享。

---

## 🤝 贡献

如果你有任何改进建议或发现了 bug，欢迎提交 Issue 或 Pull Request。

---




**Enjoy exploring the beauty of neural networks! 🧠✨**

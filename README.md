


```markdown
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




**Enjoy exploring the beauty of neural networks! 🧠✨**

---

---

# Neural Network Learning XOR Problem – Backpropagation Visualization (Ultimate Edition)

This project implements a complete multi-layer perceptron (MLP) neural network capable of learning XOR and other logic gates, with rich real‑time visualizations that intuitively display backpropagation, decision boundary formation, weight evolution, and more.

---

## 📌 Introduction

XOR (exclusive OR) is a classic linearly inseparable problem that a single‑layer perceptron cannot solve. A multi‑layer neural network, however, can perfectly fit it using the **backpropagation algorithm**. This project builds a configurable neural network system from scratch, supporting:

- Multiple activation functions, loss functions, and optimizers
- Learning rate scheduling, regularization, and Dropout
- Real‑time animated visualizations (network structure, loss curves, decision boundaries, weight distributions, etc.)
- Model saving and loading
- Evaluation metrics (accuracy, precision, recall, F1, confusion matrix)

Through visualization, you can intuitively observe how the network adjusts from random weights step by step until it successfully fits the target function.

---

## ✨ Key Features

- **Complete backpropagation** – manual gradient computation without relying on deep learning frameworks.
- **Rich activation functions**: Sigmoid, ReLU, Tanh, Leaky ReLU, ELU.
- **Multiple optimizers**: SGD, Momentum, Adam.
- **Learning rate schedulers**: constant, exponential decay, cosine annealing, step decay.
- **Regularization**: L1, L2, Dropout.
- **Loss functions**: MSE, binary cross‑entropy.
- **Datasets**: XOR, AND, OR, NAND, XNOR, Circle, Spiral.
- **Advanced visualizations**:
  - Dynamic network structure (neuron activations, connection weights)
  - Training loss / accuracy curves
  - Real‑time decision boundary plot
  - Weight distribution histograms
  - Learning rate curve
  - Prediction result table
- **Model persistence**: save/load weights and optimizer state.
- **Early stopping** to prevent overfitting.

---

## 🛠 Dependencies

- Python 3.7+
- NumPy
- Matplotlib

Install dependencies:

```bash
pip install numpy matplotlib
```

---

## 🚀 Usage

### 1. Run directly

In your terminal:

```bash
python neural_network_xor.py
```

By default, it uses the XOR dataset, network structure `[2, 8, 8, 1]`, activation `sigmoid`, optimizer `adam`, and trains for 1000 epochs. A visualization window will appear and update in real time.

### 2. Custom configuration

Modify the `config` dictionary inside the `main()` function:

```python
config = {
    'dataset': 'xor',           # Options: xor, and, or, nand, xnor, circle, spiral
    'network_structure': [2, 8, 8, 1],  # Input → hidden → output layer sizes
    'activation': 'sigmoid',     # sigmoid, relu, tanh, leaky_relu, elu
    'learning_rate': 0.5,
    'scheduler_type': 'cosine',  # constant, exponential, cosine, step
    'optimizer_type': 'adam',    # sgd, momentum, adam
    'loss_type': 'mse',          # mse, binary_crossentropy
    'reg_type': 'none',          # none, l1, l2, dropout
    'reg_lambda': 0.01,
    'dropout_rate': 0.0,         # Only used when reg_type='dropout'
    'epochs': 1000,
    'steps_per_frame': 10,
    'interval': 50               # milliseconds
}
```

### 3. Load a saved model

```python
from ModelPersistence import ModelPersistence
nn = ModelPersistence.load_model("trained_model.npz")
# Use nn.forward(X) for prediction
```

---

## 📊 Visualization Panels

When you run the program, a window with multiple subplots appears:

| Panel | Content |
|-------|---------|
| Top‑left (large) | **Neural Network Structure**: node colors represent activations, connection colors indicate weight sign, thickness indicates magnitude |
| Top‑right (left) | **Training Loss Curve** |
| Top‑right (right) | **Accuracy Curve** |
| Middle‑right | **Decision Boundary**: background shows prediction probability, points are training samples |
| Bottom‑left | **Weight Distribution Histograms** per layer |
| Bottom‑right | **Learning Rate Curve** |
| Bottom | **Prediction Table**: input, expected, predicted, error, and correctness for each sample |

---

## 📁 File Structure

```
.
├── neural_network_xor.py    # Main script (all implementations included)
├── trained_model.npz        # Automatically saved model after training (optional)
└── README.md                # This file
```

> All code is contained in a single script for easy reading and experimentation.

---

## 🔬 Brief Technical Principles

### Core Steps of Backpropagation

1. **Forward pass**: compute linear transformations `z = W·a + b` and activations `a = σ(z)` layer by layer.
2. **Compute loss**: using MSE or cross‑entropy.
3. **Backward pass**: starting from the output layer, use the chain rule to compute the error term `δ` for each layer, then obtain gradients `∂L/∂W` and `∂L/∂b`.
4. **Parameter update**: update weights and biases using the chosen optimizer (SGD, Momentum, or Adam).

### Why XOR Needs a Multi‑Layer Network?

XOR data is linearly inseparable in 2D space. A single‑layer perceptron can only learn linear decision boundaries. A multi‑layer network with nonlinear activations can combine multiple linear boundaries to form arbitrarily complex decision regions, thus perfectly classifying XOR.

---

## 📈 Example Output (Training XOR)

```
Config: [2, 8, 8, 1], sigmoid, adam, mse
...
Final loss: 0.000123
Accuracy: 100.0%
[Confusion Matrix]
  TP: 2   FP: 0
  FN: 0   TN: 2
```

The visualization window will show how the network gradually converges from random to correct mappings.

---

## 📝 License

This project is for educational and research purposes only. No specific license restrictions. Feel free to modify and share.

---

## 🤝 Contributing

If you have suggestions or find bugs, please open an Issue or submit a Pull Request.



**Enjoy exploring the beauty of neural networks! 🧠✨**
```

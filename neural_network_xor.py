"""
Neural Network Learning XOR Problem - Showcase of Code Beauty (Ultimate Version)
================================================================================

This program implements a complete neural network system capable of learning
the XOR problem and other logic gates. XOR (exclusive or) is a classic
nonlinear problem that single-layer perceptrons cannot solve and requires
a multi-layer neural network to learn.

Key Features:
1. Complete backpropagation algorithm implementation
2. Multiple activation function support (Sigmoid, ReLU, Tanh, Leaky ReLU, ELU)
3. Learning rate scheduler (exponential decay, cosine annealing, step decay)
4. Multiple optimizers (SGD, Momentum, Adam)
5. Multiple loss functions (MSE, Binary Cross-Entropy)
6. Regularization techniques (L1, L2, Dropout)
7. Evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix)
8. Model persistence (save/load trained models)
9. Early stopping mechanism to prevent overfitting
10. Real-time visualization of neural network structure and learning process
11. Dynamic display of decision boundary formation
12. Weight distribution and gradient flow visualization
13. Multiple logic gate dataset support (XOR, AND, OR, NAND, XNOR)
14. Advanced datasets (Circle, Spiral classification)

Author: Trae AI
Date: 2026-07-24
Version: 3.0 - Ultimate Edition
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# [Core Module 1] Activation Function Library - Source of Neural Network Nonlinearity
# ============================================================================

class ActivationFunctions:
    """
    Activation Function Library

    [Core Principle]
    Activation functions are the key to enabling neural networks to learn
    nonlinear patterns. Without activation functions, a multi-layer network
    is equivalent to a single-layer linear transformation.

    Each activation function has two methods:
    1. forward: Forward propagation
    2. derivative: Derivative (used for backpropagation)
    """

    @staticmethod
    def sigmoid(x):
        """
        [Classic] Sigmoid Activation Function

        Mathematical formula: σ(x) = 1 / (1 + e^(-x))

        Advantages:
        1. Output range (0,1), suitable for probability interpretation
        2. Simple derivative: σ'(x) = σ(x) * (1 - σ(x))
        3. Smooth and differentiable everywhere

        Disadvantages:
        1. Vanishing gradient: when x is very large or very small, gradient approaches 0
        2. Output is not zero-centered
        3. Expensive exponential computation
        """
        x = np.clip(x, -500, 500)
        return 1.0 / (1.0 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        """Sigmoid derivative: σ'(x) = σ(x) * (1 - σ(x))"""
        return x * (1 - x)

    @staticmethod
    def relu(x):
        """
        [Modern] ReLU Activation Function (Rectified Linear Unit)

        Mathematical formula: f(x) = max(0, x)

        Advantages:
        1. Simple computation, fast convergence
        2. Solves vanishing gradient problem (gradient is constant 1 in positive region)
        3. Sparse activation, closer to biological neurons

        Disadvantages:
        1. Dead ReLU: gradient is 0 in negative region, neurons may "die"
        2. Output is not zero-centered
        """
        return np.maximum(0, x)

    @staticmethod
    def relu_derivative(x):
        """ReLU derivative: f'(x) = 1 if x > 0, else 0"""
        return (x > 0).astype(float)

    @staticmethod
    def tanh(x):
        """
        [Classic] Tanh Activation Function

        Mathematical formula: f(x) = (e^x - e^(-x)) / (e^x + e^(-x))

        Advantages:
        1. Zero-centered output, (-1, 1)
        2. Simple derivative: f'(x) = 1 - f(x)²

        Disadvantages:
        1. Vanishing gradient problem (better than Sigmoid)
        2. Expensive exponential computation
        """
        return np.tanh(x)

    @staticmethod
    def tanh_derivative(x):
        """Tanh derivative: f'(x) = 1 - f(x)²"""
        return 1 - x * x

    @staticmethod
    def leaky_relu(x, alpha=0.01):
        """
        [Improved] Leaky ReLU Activation Function

        Mathematical formula: f(x) = x if x > 0, else α*x

        Advantages:
        1. Solves the Dead ReLU problem
        2. Small non-zero gradient in the negative region
        """
        return np.where(x > 0, x, alpha * x)

    @staticmethod
    def leaky_relu_derivative(x, alpha=0.01):
        """Leaky ReLU derivative"""
        return np.where(x > 0, 1, alpha)

    @staticmethod
    def elu(x, alpha=1.0):
        """
        [Improved] ELU Activation Function (Exponential Linear Unit)

        Mathematical formula: f(x) = x if x > 0, else α*(e^x - 1)

        Advantages:
        1. Output is approximately zero-centered
        2. Non-zero gradient in the negative region
        3. More robust to noise
        """
        return np.where(x > 0, x, alpha * (np.exp(x) - 1))

    @staticmethod
    def elu_derivative(x, alpha=1.0):
        """ELU derivative"""
        return np.where(x > 0, 1, alpha * np.exp(x))

    @classmethod
    def get_activation(cls, name):
        """Get activation function and its derivative"""
        activations = {
            'sigmoid': (cls.sigmoid, cls.sigmoid_derivative),
            'relu': (cls.relu, cls.relu_derivative),
            'tanh': (cls.tanh, cls.tanh_derivative),
            'leaky_relu': (cls.leaky_relu, cls.leaky_relu_derivative),
            'elu': (cls.elu, cls.elu_derivative)
        }

        if name not in activations:
            raise ValueError(f"Unknown activation: {name}. Choose from {list(activations.keys())}")

        return activations[name]


# ============================================================================
# [Core Module 2] Learning Rate Scheduler - Intelligent Adjustment During Training
# ============================================================================

class LearningRateScheduler:
    """
    Learning Rate Scheduler

    [Core Principle]
    Learning rate is one of the most important hyperparameters for training
    neural networks:
    - Too large: unstable training, loss oscillation
    - Too slow: convergence too slow, may get stuck in local optima

    The scheduler dynamically adjusts the learning rate, taking large steps
    for exploration early on and fine-tuning in later stages.
    """

    def __init__(self, initial_lr, scheduler_type='exponential', **kwargs):
        """
        Initialize learning rate scheduler

        Args:
            initial_lr: Initial learning rate
            scheduler_type: Scheduler type
                - 'constant': Fixed learning rate
                - 'exponential': Exponential decay
                - 'cosine': Cosine annealing
                - 'step': Step decay
            **kwargs: Additional parameters
        """
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.scheduler_type = scheduler_type
        self.epoch = 0

        # Exponential decay parameters
        self.decay_rate = kwargs.get('decay_rate', 0.99)

        # Cosine annealing parameters
        self.min_lr = kwargs.get('min_lr', initial_lr * 0.01)
        self.T_max = kwargs.get('T_max', 1000)

        # Step decay parameters
        self.step_size = kwargs.get('step_size', 200)
        self.gamma = kwargs.get('gamma', 0.5)

    def step(self):
        """Update learning rate"""
        self.epoch += 1

        if self.scheduler_type == 'constant':
            # Fixed learning rate
            pass

        elif self.scheduler_type == 'exponential':
            # Exponential decay: lr = lr0 * decay_rate^epoch
            self.current_lr = self.initial_lr * (self.decay_rate ** self.epoch)

        elif self.scheduler_type == 'cosine':
            # Cosine annealing: lr = min_lr + 0.5*(lr0 - min_lr)*(1 + cos(π*epoch/T_max))
            self.current_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * \
                             (1 + np.cos(np.pi * self.epoch / self.T_max))

        elif self.scheduler_type == 'step':
            # Step decay: every step_size epochs, multiply learning rate by gamma
            if self.epoch % self.step_size == 0:
                self.current_lr *= self.gamma

        return self.current_lr

    def get_lr(self):
        """Get current learning rate"""
        return self.current_lr


# ============================================================================
# [Core Module 2.5] Loss Functions - Measuring Prediction Error
# ============================================================================

class LossFunctions:
    """
    Loss Function Library
    
    Loss functions measure how far predictions are from true values.
    Different loss functions are suited for different tasks:
    - MSE: Regression, simple binary classification
    - Cross-Entropy: Classification tasks (better gradient properties)
    """
    
    @staticmethod
    def mse(y_true, y_pred):
        """Mean Squared Error: L = (1/n) * Σ(y_true - y_pred)²"""
        return np.mean(np.square(y_true - y_pred))
    
    @staticmethod
    def mse_derivative(y_true, y_pred):
        """MSE derivative: dL/dy_pred = 2 * (y_pred - y_true) / n"""
        return 2 * (y_pred - y_true) / y_true.shape[0]
    
    @staticmethod
    def binary_crossentropy(y_true, y_pred):
        """
        Binary Cross-Entropy Loss
        
        Formula: L = -[y*log(p) + (1-y)*log(1-p)]
        
        Better than MSE for classification because:
        1. Stronger gradients when predictions are wrong
        2. Convex loss surface for logistic regression
        """
        epsilon = 1e-15  # Avoid log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    
    @staticmethod
    def binary_crossentropy_derivative(y_true, y_pred):
        """Binary cross-entropy derivative"""
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return (y_pred - y_true) / (y_pred * (1 - y_pred)) / y_true.shape[0]
    
    @classmethod
    def get_loss(cls, name):
        """Get loss function and its derivative"""
        losses = {
            'mse': (cls.mse, cls.mse_derivative),
            'binary_crossentropy': (cls.binary_crossentropy, cls.binary_crossentropy_derivative)
        }
        if name not in losses:
            raise ValueError(f"Unknown loss: {name}. Choose from {list(losses.keys())}")
        return losses[name]


# ============================================================================
# [Core Module 2.6] Optimizers - Advanced Weight Update Strategies
# ============================================================================

class Optimizer:
    """
    Optimizer Library
    
    [Core Principle]
    Optimizers determine how weights are updated during training.
    Simple gradient descent can be slow and get stuck in local minima.
    Advanced optimizers use momentum and adaptive learning rates.
    
    Supported optimizers:
    - SGD: Basic stochastic gradient descent
    - SGD with Momentum: Uses velocity to accelerate convergence
    - Adam: Adaptive learning rates with momentum (most popular)
    """
    
    def __init__(self, optimizer_type='sgd', learning_rate=0.01, **kwargs):
        """
        Initialize optimizer
        
        Args:
            optimizer_type: 'sgd', 'momentum', or 'adam'
            learning_rate: Step size for weight updates
            **kwargs: Additional parameters (beta, epsilon, etc.)
        """
        self.optimizer_type = optimizer_type
        self.lr = learning_rate
        self.t = 0  # Time step
        
        # Momentum parameters
        self.beta1 = kwargs.get('beta1', 0.9)  # For momentum
        
        # Adam parameters
        self.beta2 = kwargs.get('beta2', 0.999)
        self.epsilon = kwargs.get('epsilon', 1e-8)
        
        # Velocity and squared gradient accumulators (initialized on first use)
        self.velocity_w = None
        self.velocity_b = None
        self.sq_grad_w = None
        self.sq_grad_b = None
    
    def update(self, weights, biases, weight_gradients, bias_gradients):
        """
        Update weights and biases using the selected optimizer

        Standard gradient descent: w = w - lr * gradient
        Gradients are computed as dL/dw (positive direction of loss increase),
        so we SUBTRACT to minimize loss.

        Args:
            weights: List of weight matrices
            biases: List of bias vectors
            weight_gradients: List of weight gradient matrices (dL/dw)
            bias_gradients: List of bias gradient vectors (dL/db)

        Returns:
            Updated weights and biases
        """
        if self.optimizer_type == 'sgd':
            # Basic SGD: w = w - lr * gradient
            for i in range(len(weights)):
                weights[i] -= self.lr * weight_gradients[i]
                biases[i] -= self.lr * bias_gradients[i]

        elif self.optimizer_type == 'momentum':
            # SGD with Momentum
            # v = beta * v + gradient
            # w = w - lr * v
            if self.velocity_w is None:
                self.velocity_w = [np.zeros_like(w) for w in weights]
                self.velocity_b = [np.zeros_like(b) for b in biases]

            for i in range(len(weights)):
                self.velocity_w[i] = self.beta1 * self.velocity_w[i] + weight_gradients[i]
                self.velocity_b[i] = self.beta1 * self.velocity_b[i] + bias_gradients[i]
                weights[i] -= self.lr * self.velocity_w[i]
                biases[i] -= self.lr * self.velocity_b[i]

        elif self.optimizer_type == 'adam':
            # Adam (Adaptive Moment Estimation)
            # m = beta1 * m + (1 - beta1) * gradient          (first moment)
            # v = beta2 * v + (1 - beta2) * gradient²         (second moment)
            # m_hat = m / (1 - beta1^t)                        (bias correction)
            # v_hat = v / (1 - beta2^t)                        (bias correction)
            # w = w - lr * m_hat / (sqrt(v_hat) + epsilon)

            self.t += 1

            if self.velocity_w is None:
                self.velocity_w = [np.zeros_like(w) for w in weights]
                self.velocity_b = [np.zeros_like(b) for b in biases]
                self.sq_grad_w = [np.zeros_like(w) for w in weights]
                self.sq_grad_b = [np.zeros_like(b) for b in biases]

            for i in range(len(weights)):
                # Update first moment (momentum)
                self.velocity_w[i] = self.beta1 * self.velocity_w[i] + (1 - self.beta1) * weight_gradients[i]
                self.velocity_b[i] = self.beta1 * self.velocity_b[i] + (1 - self.beta1) * bias_gradients[i]

                # Update second moment (RMSprop)
                self.sq_grad_w[i] = self.beta2 * self.sq_grad_w[i] + (1 - self.beta2) * np.square(weight_gradients[i])
                self.sq_grad_b[i] = self.beta2 * self.sq_grad_b[i] + (1 - self.beta2) * np.square(bias_gradients[i])

                # Bias correction
                m_hat_w = self.velocity_w[i] / (1 - self.beta1 ** self.t)
                m_hat_b = self.velocity_b[i] / (1 - self.beta1 ** self.t)
                v_hat_w = self.sq_grad_w[i] / (1 - self.beta2 ** self.t)
                v_hat_b = self.sq_grad_b[i] / (1 - self.beta2 ** self.t)

                # Update weights
                weights[i] -= self.lr * m_hat_w / (np.sqrt(v_hat_w) + self.epsilon)
                biases[i] -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.epsilon)

        else:
            raise ValueError(f"Unknown optimizer: {self.optimizer_type}")

        return weights, biases

    def reset_state(self):
        """Reset optimizer internal state (velocity, squared gradients, time step)"""
        self.t = 0
        self.velocity_w = None
        self.velocity_b = None
        self.sq_grad_w = None
        self.sq_grad_b = None


# ============================================================================
# [Core Module 2.7] Regularization - Preventing Overfitting
# ============================================================================

class Regularization:
    """
    Regularization Techniques
    
    [Core Principle]
    Regularization prevents overfitting by adding constraints to the model.
    Overfitting occurs when the model memorizes training data instead of
    learning general patterns.
    
    Techniques:
    - L1 (Lasso): Adds |w| penalty, encourages sparsity
    - L2 (Ridge): Adds w² penalty, encourages small weights
    - Dropout: Randomly deactivates neurons during training
    """
    
    def __init__(self, reg_type='none', reg_lambda=0.01, dropout_rate=0.5):
        """
        Initialize regularization
        
        Args:
            reg_type: 'none', 'l1', 'l2', or 'dropout'
            reg_lambda: Regularization strength (for L1/L2)
            dropout_rate: Probability of dropping a neuron (for dropout)
        """
        self.reg_type = reg_type
        self.reg_lambda = reg_lambda
        self.dropout_rate = dropout_rate
        self.dropout_masks = []
    
    def l1_penalty(self, weights):
        """L1 regularization penalty: λ * Σ|w|"""
        penalty = 0
        for w in weights:
            penalty += np.sum(np.abs(w))
        return self.reg_lambda * penalty
    
    def l2_penalty(self, weights):
        """L2 regularization penalty: λ * Σw²"""
        penalty = 0
        for w in weights:
            penalty += np.sum(np.square(w))
        return self.reg_lambda * penalty
    
    def l1_gradient(self, weight):
        """L1 gradient: λ * sign(w)"""
        return self.reg_lambda * np.sign(weight)
    
    def l2_gradient(self, weight):
        """L2 gradient: 2λ * w"""
        return 2 * self.reg_lambda * weight
    
    def apply_dropout(self, activations, training=True):
        """
        Apply dropout to activations
        
        During training: randomly set neurons to 0 and scale remaining
        During inference: use all neurons (no dropout)
        """
        if not training or self.dropout_rate == 0:
            return activations
        
        self.dropout_masks = []
        masked_activations = []
        
        for a in activations:
            # Create mask: 1 with probability (1 - dropout_rate)
            mask = (np.random.rand(*a.shape) > self.dropout_rate).astype(float)
            
            # Scale by (1 - dropout_rate) to maintain expected value
            # This is "inverted dropout"
            mask = mask / (1 - self.dropout_rate)
            
            self.dropout_masks.append(mask)
            masked_activations.append(a * mask)
        
        return masked_activations
    
    def backward_dropout(self, deltas):
        """Apply dropout mask to gradients during backpropagation"""
        if self.dropout_rate == 0 or not self.dropout_masks:
            return deltas
        
        masked_deltas = []
        for i, delta in enumerate(deltas):
            if i < len(self.dropout_masks):
                masked_deltas.append(delta * self.dropout_masks[i])
            else:
                masked_deltas.append(delta)
        
        return masked_deltas
    
    def compute_penalty(self, weights):
        """Compute total regularization penalty"""
        if self.reg_type == 'l1':
            return self.l1_penalty(weights)
        elif self.reg_type == 'l2':
            return self.l2_penalty(weights)
        return 0
    
    def compute_gradients(self, weights):
        """Compute regularization gradients for weight updates"""
        grads = []
        for w in weights:
            if self.reg_type == 'l1':
                grads.append(self.l1_gradient(w))
            elif self.reg_type == 'l2':
                grads.append(self.l2_gradient(w))
            else:
                grads.append(np.zeros_like(w))
        return grads


# ============================================================================
# [Core Module 2.8] Evaluation Metrics - Measuring Model Performance
# ============================================================================

class EvaluationMetrics:
    """
    Evaluation Metrics Library
    
    [Core Principle]
    Accuracy alone doesn't tell the full story. We need multiple metrics:
    - Precision: Of all positive predictions, how many are correct?
    - Recall: Of all actual positives, how many did we find?
    - F1 Score: Harmonic mean of precision and recall
    
    Confusion Matrix shows:
    - True Positives (TP): Correctly predicted positive
    - True Negatives (TN): Correctly predicted negative
    - False Positives (FP): Incorrectly predicted positive (Type I error)
    - False Negatives (FN): Incorrectly predicted negative (Type II error)
    """
    
    @staticmethod
    def confusion_matrix(y_true, y_pred, threshold=0.5):
        """
        Compute confusion matrix
        
        Returns:
            dict with TP, TN, FP, FN counts
        """
        y_pred_binary = (y_pred >= threshold).astype(int)
        y_true_binary = y_true.astype(int)
        
        tp = np.sum((y_pred_binary == 1) & (y_true_binary == 1))
        tn = np.sum((y_pred_binary == 0) & (y_true_binary == 0))
        fp = np.sum((y_pred_binary == 1) & (y_true_binary == 0))
        fn = np.sum((y_pred_binary == 0) & (y_true_binary == 1))
        
        return {'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn}
    
    @staticmethod
    def precision(y_true, y_pred, threshold=0.5):
        """
        Precision = TP / (TP + FP)
        
        "Of all positive predictions, how many are actually positive?"
        High precision = few false positives
        """
        cm = EvaluationMetrics.confusion_matrix(y_true, y_pred, threshold)
        tp, fp = cm['TP'], cm['FP']
        if tp + fp == 0:
            return 0.0
        return tp / (tp + fp)
    
    @staticmethod
    def recall(y_true, y_pred, threshold=0.5):
        """
        Recall = TP / (TP + FN)
        
        "Of all actual positives, how many did we find?"
        High recall = few false negatives
        """
        cm = EvaluationMetrics.confusion_matrix(y_true, y_pred, threshold)
        tp, fn = cm['TP'], cm['FN']
        if tp + fn == 0:
            return 0.0
        return tp / (tp + fn)
    
    @staticmethod
    def f1_score(y_true, y_pred, threshold=0.5):
        """
        F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
        
        Harmonic mean of precision and recall.
        Good for imbalanced datasets.
        """
        p = EvaluationMetrics.precision(y_true, y_pred, threshold)
        r = EvaluationMetrics.recall(y_true, y_pred, threshold)
        if p + r == 0:
            return 0.0
        return 2 * p * r / (p + r)
    
    @staticmethod
    def accuracy(y_true, y_pred, threshold=0.5):
        """Accuracy = (TP + TN) / Total"""
        y_pred_binary = (y_pred >= threshold).astype(int)
        return np.mean(y_pred_binary == y_true.astype(int))
    
    @classmethod
    def all_metrics(cls, y_true, y_pred, threshold=0.5):
        """Compute all metrics at once"""
        return {
            'accuracy': cls.accuracy(y_true, y_pred, threshold),
            'precision': cls.precision(y_true, y_pred, threshold),
            'recall': cls.recall(y_true, y_pred, threshold),
            'f1': cls.f1_score(y_true, y_pred, threshold),
            'confusion_matrix': cls.confusion_matrix(y_true, y_pred, threshold)
        }


# ============================================================================
# [Core Module 2.9] Model Persistence - Save and Load Models
# ============================================================================

class ModelPersistence:
    """
    Model Persistence Utility
    
    Save trained models to disk and load them later.
    Uses numpy's .npz format for efficient storage.
    """
    
    @staticmethod
    def save_model(nn, filepath):
        """
        Save neural network model to file

        Saves:
        - Weights and biases for each layer
        - Network architecture (layer sizes)
        - Optimizer state (for momentum/Adam continuation)
        - Training history
        """
        save_dict = {
            'layer_sizes': np.array(nn.layer_sizes),
            'activation': nn.activation_name,
            'learning_rate': nn.learning_rate,
            'optimizer_type': nn.optimizer_type,
            'loss_type': nn.loss_type,
        }

        # Save weights and biases
        for i, (w, b) in enumerate(zip(nn.weights, nn.biases)):
            save_dict[f'weight_{i}'] = w
            save_dict[f'bias_{i}'] = b

        # Save optimizer state
        opt = nn.optimizer
        save_dict['optimizer_t'] = np.array(opt.t)
        if opt.velocity_w is not None:
            for i, (vw, vb) in enumerate(zip(opt.velocity_w, opt.velocity_b)):
                save_dict[f'opt_vw_{i}'] = vw
                save_dict[f'opt_vb_{i}'] = vb
        if opt.sq_grad_w is not None:
            for i, (sw, sb) in enumerate(zip(opt.sq_grad_w, opt.sq_grad_b)):
                save_dict[f'opt_sw_{i}'] = sw
                save_dict[f'opt_sb_{i}'] = sb

        # Save training history
        if nn.training_history['loss']:
            save_dict['history_loss'] = np.array(nn.training_history['loss'])
            save_dict['history_accuracy'] = np.array(nn.training_history['accuracy'])

        np.savez(filepath, **save_dict)
        print(f"Model saved to {filepath}")

    @staticmethod
    def load_model(filepath):
        """
        Load neural network model from file

        Returns:
            NeuralNetwork instance with loaded weights and optimizer state
        """
        data = np.load(filepath, allow_pickle=True)

        # Reconstruct network
        layer_sizes = data['layer_sizes'].tolist()
        activation = str(data['activation'])
        learning_rate = float(data['learning_rate'])
        optimizer_type = str(data.get('optimizer_type', 'sgd'))
        loss_type = str(data.get('loss_type', 'mse'))

        nn = NeuralNetwork(
            layer_sizes, activation, learning_rate,
            optimizer_type=optimizer_type, loss_type=loss_type
        )

        # Load weights and biases
        for i in range(len(layer_sizes) - 1):
            nn.weights[i] = data[f'weight_{i}']
            nn.biases[i] = data[f'bias_{i}']

        # Restore optimizer state
        nn.optimizer.t = int(data.get('optimizer_t', 0))
        n_layers = len(layer_sizes) - 1
        if 'opt_vw_0' in data:
            nn.optimizer.velocity_w = [data[f'opt_vw_{i}'] for i in range(n_layers)]
            nn.optimizer.velocity_b = [data[f'opt_vb_{i}'] for i in range(n_layers)]
        if 'opt_sw_0' in data:
            nn.optimizer.sq_grad_w = [data[f'opt_sw_{i}'] for i in range(n_layers)]
            nn.optimizer.sq_grad_b = [data[f'opt_sb_{i}'] for i in range(n_layers)]

        # Load training history if available
        if 'history_loss' in data:
            nn.training_history['loss'] = data['history_loss'].tolist()
            nn.training_history['accuracy'] = data['history_accuracy'].tolist()

        print(f"Model loaded from {filepath}")
        return nn


# ============================================================================
# [Core Module 3] Neural Network Class - Complete Forward and Backward Propagation
# ============================================================================

class NeuralNetwork:
    """
    Multi-layer Perceptron Neural Network Implementation (Enhanced Version)

    Core Algorithm:
    1. Forward propagation: compute network output
    2. Backpropagation: compute gradients and update weights
    3. Multiple activation function support
    4. Learning rate scheduling
    5. Early stopping mechanism

    Network structure: customizable
    """

    def __init__(self, layer_sizes, activation='sigmoid', learning_rate=0.1,
                 scheduler_type='constant', optimizer_type='sgd',
                 loss_type='mse', reg_type='none', reg_lambda=0.01,
                 dropout_rate=0.0, **kwargs):
        """
        Initialize neural network

        Args:
            layer_sizes: List specifying the number of neurons in each layer
                        e.g.: [2, 4, 1] means 2 input, 4 hidden, 1 output
            activation: Activation function type (sigmoid, relu, tanh, leaky_relu, elu)
            learning_rate: Initial learning rate
            scheduler_type: Learning rate scheduler type (constant, exponential, cosine, step)
            optimizer_type: Optimizer type (sgd, momentum, adam)
            loss_type: Loss function type (mse, binary_crossentropy)
            reg_type: Regularization type (none, l1, l2, dropout)
            reg_lambda: Regularization strength
            dropout_rate: Dropout probability
            **kwargs: Additional parameters for scheduler and optimizer
        """
        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes)

        # [Core] Initialize weights and biases
        # Using Xavier initialization, which helps gradient propagation
        self.weights = []
        self.biases = []

        for i in range(self.num_layers - 1):
            # Xavier initialization: weights sampled from normal distribution with std = sqrt(2/n)
            # This is a commonly used initialization method in deep learning
            # that helps avoid vanishing/exploding gradients
            scale = np.sqrt(2.0 / layer_sizes[i])
            w = np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * scale
            b = np.zeros((1, layer_sizes[i + 1]))

            self.weights.append(w)
            self.biases.append(b)

        # Store activation values for each layer, used in backpropagation
        self.activations = []

        # Store z values (linear transformation results) for each layer, used to compute gradients
        self.z_values = []

        # Store gradients for each layer, used for visualization
        self.weight_gradients = []
        self.bias_gradients = []

        # Set activation function
        self.activation_func, self.activation_derivative = \
            ActivationFunctions.get_activation(activation)
        self.activation_name = activation

        # Set learning rate scheduler
        self.scheduler = LearningRateScheduler(
            learning_rate, scheduler_type, **kwargs
        )
        self.learning_rate = learning_rate

        # [New] Set optimizer (SGD, Momentum, Adam)
        self.optimizer = Optimizer(optimizer_type, learning_rate, **kwargs)
        self.optimizer_type = optimizer_type

        # [New] Set loss function
        self.loss_func, self.loss_derivative = LossFunctions.get_loss(loss_type)
        self.loss_type = loss_type

        # [New] Set regularization
        self.regularization = Regularization(reg_type, reg_lambda, dropout_rate)
        self.reg_type = reg_type

        # Training history records
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'learning_rate': [],
            'weight_stats': []  # Weight statistics
        }

        # Early stopping related
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.patience = 50  # Early stopping patience
        self.min_loss_change = 1e-6  # Minimum loss change
        self.early_stopped = False  # Flag to signal early stopping

    def forward(self, X, training=True):
        """
        [Core] Forward Propagation Algorithm

        Mathematical principle:
        For each layer l:
            z^(l) = W^(l) * a^(l-1) + b^(l)  (linear transformation)
            a^(l) = σ(z^(l))                   (activation function)

        Where:
        - W^(l) is the weight matrix of layer l
        - a^(l-1) is the activation value of the previous layer (input layer is raw data)
        - b^(l) is the bias vector of layer l
        - σ is the activation function

        This process is like dominoes, data flows from input layer to output layer.
        """
        self.activations = [X]  # Store input as layer 0 activation
        self.z_values = []
        current_input = X

        # Compute layer by layer
        for layer_idx in range(len(self.weights)):
            # Linear transformation: z = W * a + b
            z = np.dot(current_input, self.weights[layer_idx]) + self.biases[layer_idx]
            self.z_values.append(z)

            # Activation function: a = σ(z)
            # Output layer: use sigmoid for binary classification (ensures output in [0,1])
            # Hidden layers: use the configured activation function
            if layer_idx == len(self.weights) - 1:
                activation = ActivationFunctions.sigmoid(z)
                self._output_activation_derivative = ActivationFunctions.sigmoid_derivative
            else:
                activation = self.activation_func(z)

            # Store activation values for subsequent backpropagation
            self.activations.append(activation)

            # Current layer's output becomes the next layer's input
            current_input = activation

        # Apply dropout to hidden layer activations (not input or output)
        if training and self.reg_type == 'dropout' and self.regularization.dropout_rate > 0:
            # Only dropout hidden layers (indices 1 to len-2 in activations)
            hidden_activations = self.activations[1:-1]
            if hidden_activations:
                masked = self.regularization.apply_dropout(hidden_activations, training=True)
                for i, m in enumerate(masked):
                    self.activations[i + 1] = m

        return current_input

    def backward(self, X, y):
        """
        [Core] Backpropagation Algorithm - The Core of Neural Network Learning!

        This method only computes gradients. Weight updates are handled by
        the optimizer in train_step() to avoid double-updating.

        Mathematical principle:
        1. Compute output layer error: δ^(L) = (a^(L) - y) ⊙ σ'(z^(L))
           For BCE + Sigmoid: δ^(L) = a^(L) - y (sigmoid derivative cancels out)
        2. Backpropagate error: δ^(l) = (W^(l+1))^T * δ^(l+1) ⊙ σ'(z^(l))
        3. Compute gradients:
           ∂C/∂W^(l) = δ^(l) * (a^(l-1))^T
           ∂C/∂b^(l) = δ^(l)

        Where:
        - ⊙ denotes element-wise multiplication (Hadamard product)
        - C is the loss function
        """
        m = X.shape[0]  # Number of samples

        # Store gradients
        self.weight_gradients = []
        self.bias_gradients = []

        # [Step 1] Compute output layer delta
        # For MSE + Sigmoid: delta = (a - y) * sigmoid'(a) = (a - y) * a * (1-a)
        # For BCE + Sigmoid: delta = a - y (sigmoid derivative cancels in BCE gradient)
        output_error = self.activations[-1] - y  # Note: a - y, not y - a

        if self.loss_type == 'binary_crossentropy':
            # BCE gradient with sigmoid output: delta = a - y
            output_delta = output_error
        else:
            # MSE gradient: delta = (a - y) * output_activation'(a)
            output_delta = output_error * self._output_activation_derivative(self.activations[-1])

        # Store deltas for all layers
        deltas = [output_delta]

        # [Step 2] Backpropagate error to hidden layers
        # Start from the second-to-last layer and propagate forward
        for layer_idx in range(len(self.weights) - 1, 0, -1):
            # Current layer error = next layer delta dot product with current layer weight transpose
            error = deltas[-1].dot(self.weights[layer_idx].T)

            # Apply dropout mask to hidden layer deltas
            hidden_idx = layer_idx - 1  # index into dropout_masks
            if self.reg_type == 'dropout' and self.regularization.dropout_rate > 0 and \
               hidden_idx < len(self.regularization.dropout_masks):
                error = error * self.regularization.dropout_masks[hidden_idx]

            # Multiply by the derivative of the current layer's activation function
            # For ReLU/Leaky ReLU/ELU: derivative depends on pre-activation value z
            # For Sigmoid/Tanh: derivative can be computed from post-activation value a
            if self.activation_name in ('relu', 'leaky_relu', 'elu'):
                delta = error * self.activation_derivative(self.z_values[layer_idx - 1])
            else:
                delta = error * self.activation_derivative(self.activations[layer_idx])

            deltas.append(delta)

        # Reverse the deltas list to match the layer order
        deltas.reverse()

        # [Step 3] Compute and store gradients (do NOT update weights here)
        # Weight updates are handled by the optimizer in train_step()
        for layer_idx in range(len(self.weights)):
            # Weight gradient = current layer activation transpose * current layer delta
            weight_gradient = self.activations[layer_idx].T.dot(deltas[layer_idx])

            # Bias gradient = column sum of delta
            bias_gradient = np.sum(deltas[layer_idx], axis=0, keepdims=True)

            # Store gradients for optimizer and visualization
            self.weight_gradients.append(weight_gradient)
            self.bias_gradients.append(bias_gradient)

    def compute_loss(self, y_true, y_pred):
        """
        Compute loss using selected loss function

        Supports:
        - MSE: Mean Squared Error (1/n) * Σ(y_true - y_pred)²
        - Binary Cross-Entropy: -[y*log(p) + (1-y)*log(1-p)]
        """
        return self.loss_func(y_true, y_pred)

    def compute_accuracy(self, y_true, y_pred, threshold=0.5):
        """Compute accuracy"""
        predictions = (y_pred >= threshold).astype(int)
        return np.mean(predictions == y_true)

    def get_weight_statistics(self):
        """Get weight statistics for visualization"""
        stats = []
        for i, w in enumerate(self.weights):
            stats.append({
                'layer': i,
                'mean': np.mean(w),
                'std': np.std(w),
                'min': np.min(w),
                'max': np.max(w),
                'norm': np.linalg.norm(w)
            })
        return stats

    def train_step(self, X, y):
        """
        Execute one training step

        Returns:
            loss: Current loss
            accuracy: Current accuracy
            lr: Current learning rate
        """
        # Forward propagation (with dropout during training)
        output = self.forward(X, training=True)

        # Compute loss and accuracy
        loss = self.compute_loss(y, output)
        accuracy = self.compute_accuracy(y, output)

        # Update learning rate from scheduler
        lr = self.scheduler.step()

        # Sync optimizer learning rate with scheduler
        self.optimizer.lr = lr

        # Backpropagation (only computes gradients, does NOT update weights)
        self.backward(X, y)

        # Add regularization gradients if applicable
        if self.reg_type in ('l1', 'l2'):
            reg_grads = self.regularization.compute_gradients(self.weights)
            for i in range(len(self.weight_gradients)):
                self.weight_gradients[i] += reg_grads[i]

        # Use optimizer to update weights (single point of weight update)
        self.weights, self.biases = self.optimizer.update(
            self.weights, self.biases,
            self.weight_gradients, self.bias_gradients
        )

        # Record training history
        self.training_history['loss'].append(loss)
        self.training_history['accuracy'].append(accuracy)
        self.training_history['learning_rate'].append(lr)
        self.training_history['weight_stats'].append(self.get_weight_statistics())

        return loss, accuracy, lr

    def early_stop(self, loss):
        """
        Early stopping mechanism

        Stop training when the loss no longer decreases significantly,
        to prevent overfitting.
        """
        if self.early_stopped:
            return True
        if loss < self.best_loss - self.min_loss_change:
            self.best_loss = loss
            self.patience_counter = 0
            return False
        else:
            self.patience_counter += 1
            if self.patience_counter >= self.patience:
                self.early_stopped = True
                return True
            return False


# ============================================================================
# [Core Module 4] Dataset Factory - Multiple Logic Gate Datasets
# ============================================================================

class DatasetFactory:
    """
    Dataset Factory

    Provides multiple classic logic gate datasets for testing neural networks.
    Each logic gate has different linear separability:
    - AND, OR, NAND, NOR are linearly separable
    - XOR, XNOR are linearly non-separable
    """

    @staticmethod
    def create_xor():
        """
        XOR (Exclusive OR) Dataset

        Truth table:
        Input1  Input2  Output
          0       0      0
          0       1      1
          1       0      1
          1       1      0

        This is a classic linearly non-separable problem!
        """
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])
        return X, y, "XOR"

    @staticmethod
    def create_and():
        """
        AND Dataset

        Truth table:
        Input1  Input2  Output
          0       0      0
          0       1      0
          1       0      0
          1       1      1

        This is a linearly separable problem.
        """
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [0], [0], [1]])
        return X, y, "AND"

    @staticmethod
    def create_or():
        """
        OR Dataset

        Truth table:
        Input1  Input2  Output
          0       0      0
          0       1      1
          1       0      1
          1       1      1

        This is a linearly separable problem.
        """
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [1]])
        return X, y, "OR"

    @staticmethod
    def create_nand():
        """
        NAND (NOT AND) Dataset

        Truth table:
        Input1  Input2  Output
          0       0      1
          0       1      1
          1       0      1
          1       1      0

        This is a linearly separable problem.
        """
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[1], [1], [1], [0]])
        return X, y, "NAND"

    @staticmethod
    def create_xnor():
        """
        XNOR (NOT XOR) Dataset

        Truth table:
        Input1  Input2  Output
          0       0      1
          0       1      0
          1       0      0
          1       1      1

        This is a linearly non-separable problem (negation of XOR).
        """
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[1], [0], [0], [1]])
        return X, y, "XNOR"

    @staticmethod
    def create_circle(num_points=100):
        """
        Circle Dataset

        Generates a circular classification problem:
        - Points inside the circle are labeled 1
        - Points outside the circle are labeled 0

        This is a nonlinear classification problem that requires the neural
        network to learn a circular boundary.
        """
        # Generate random points
        X = np.random.randn(num_points, 2) * 0.5

        # Compute distance to origin
        distances = np.sqrt(X[:, 0]**2 + X[:, 1]**2)

        # Label: distance < 0.5 is 1, otherwise 0
        y = (distances < 0.5).astype(float).reshape(-1, 1)

        return X, y, "Circle Classification"

    @staticmethod
    def create_spiral(num_points=100):
        """
        Spiral Dataset

        Generates two intertwined spiral lines, a very challenging classification problem.
        """
        X = np.zeros((num_points * 2, 2))
        y = np.zeros((num_points * 2, 1))

        for i in range(num_points):
            # First spiral
            r = i / num_points * 5
            t = 1.75 * i / num_points * 2 * np.pi
            X[i] = [r * np.sin(t), r * np.cos(t)]
            y[i] = 0

            # Second spiral
            X[i + num_points] = [r * np.sin(t + np.pi), r * np.cos(t + np.pi)]
            y[i + num_points] = 1

        # Add noise
        X += np.random.randn(*X.shape) * 0.1

        return X, y, "Spiral Classification"

    @classmethod
    def get_dataset(cls, name):
        """Get the specified dataset"""
        datasets = {
            'xor': cls.create_xor,
            'and': cls.create_and,
            'or': cls.create_or,
            'nand': cls.create_nand,
            'xnor': cls.create_xnor,
            'circle': cls.create_circle,
            'spiral': cls.create_spiral
        }

        if name not in datasets:
            raise ValueError(f"Unknown dataset: {name}. Choose from {list(datasets.keys())}")

        return datasets[name]()


# ============================================================================
# [Core Module 5] Enhanced Visualization System - Comprehensive Learning Process Display
# ============================================================================

class EnhancedVisualizer:
    """
    Enhanced Neural Network Visualizer

    Features:
    1. Real-time display of neural network structure
    2. Dynamic weight change visualization
    3. Decision boundary visualization
    4. Loss function curve plotting
    5. Prediction result display
    6. Weight distribution histogram
    7. Gradient flow visualization
    8. Learning rate change curve
    """

    def __init__(self, nn, X, y, dataset_name="XOR"):
        """
        Initialize visualizer

        Args:
            nn: Neural network instance
            X: Input data
            y: Expected output
            dataset_name: Dataset name
        """
        self.nn = nn
        self.X = X
        self.y = y
        self.dataset_name = dataset_name
        self.losses = []
        self.accuracies = []
        self.learning_rates = []

        # Set Chinese font
        plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans', 'sans-serif']
        plt.rcParams['axes.unicode_minus'] = False

        # Create figure window
        self.fig = plt.figure(figsize=(20, 14))
        self.fig.patch.set_facecolor('#F8F9FA')

        # Use GridSpec for flexible layout (4 rows x 4 columns)
        gs = GridSpec(4, 4, figure=self.fig, hspace=0.5, wspace=0.5)

        # Subplot 1: Neural network structure (top-left, 2x2)
        self.ax_network = self.fig.add_subplot(gs[0:2, 0:2])
        self.ax_network.set_title('Neural Network Structure', fontsize=14, fontweight='bold', pad=15)
        self.ax_network.set_facecolor('#FFFFFF')

        # Subplot 2: Loss curve (top-right)
        self.ax_loss = self.fig.add_subplot(gs[0, 2])
        self.ax_loss.set_title('Training Loss', fontsize=12, fontweight='bold')
        self.ax_loss.set_facecolor('#FFFFFF')

        # Subplot 3: Accuracy curve (top-right)
        self.ax_accuracy = self.fig.add_subplot(gs[0, 3])
        self.ax_accuracy.set_title('Accuracy', fontsize=12, fontweight='bold')
        self.ax_accuracy.set_facecolor('#FFFFFF')

        # Subplot 4: Decision boundary (middle-right)
        self.ax_boundary = self.fig.add_subplot(gs[1, 2:4])
        self.ax_boundary.set_title('Decision Boundary', fontsize=12, fontweight='bold')
        self.ax_boundary.set_facecolor('#FFFFFF')

        # Subplot 5: Weight distribution (bottom-left)
        self.ax_weights = self.fig.add_subplot(gs[2, 0:2])
        self.ax_weights.set_title('Weight Distribution', fontsize=12, fontweight='bold')
        self.ax_weights.set_facecolor('#FFFFFF')

        # Subplot 6: Learning rate curve (bottom-right)
        self.ax_lr = self.fig.add_subplot(gs[2, 2])
        self.ax_lr.set_title('Learning Rate', fontsize=12, fontweight='bold')
        self.ax_lr.set_facecolor('#FFFFFF')

        # Subplot 7: Prediction result table (bottom)
        self.ax_table = self.fig.add_subplot(gs[3, :])
        self.ax_table.set_title('Prediction Results Comparison', fontsize=12, fontweight='bold')
        self.ax_table.axis('off')

        # Set main title
        self.fig.suptitle(
            f'Neural Network Learning {dataset_name} Problem - Backpropagation Visualization\n'
            f'Network: {"→".join(map(str, nn.layer_sizes))} | '
            f'Activation: {nn.activation_name}',
            fontsize=18, fontweight='bold', y=0.98
        )

        # Track colorbar to prevent accumulation
        self.colorbar = None

    def draw_network_structure(self, epoch, total_epochs):
        """
        [Visualization Core 1] Draw Neural Network Structure

        Displayed content:
        1. Neuron nodes (color represents activation value)
        2. Connection lines (thickness and color represent weight magnitude and sign)
        3. Current epoch and loss value
        """
        self.ax_network.clear()
        self.ax_network.set_title(
            f'Neural Network Structure (Epoch: {epoch}/{total_epochs})',
            fontsize=14, fontweight='bold', pad=15
        )

        # Adjust layout based on network structure
        num_layers = len(self.nn.layer_sizes)
        x_spacing = 4.0 / (num_layers - 1) if num_layers > 1 else 4.0
        x_positions = [0.5 + i * x_spacing for i in range(num_layers)]

        # Compute y positions for neurons in each layer
        layer_positions = []
        for i, size in enumerate(self.nn.layer_sizes):
            x = x_positions[i]
            if size == 1:
                positions = [(x, 1.5)]
            else:
                y_spacing = 3.0 / (size - 1)
                positions = [(x, 0.0 + j * y_spacing) for j in range(size)]
            layer_positions.append(positions)

        # Set axis ranges
        self.ax_network.set_xlim(-0.3, 5.0)
        self.ax_network.set_ylim(-0.5, 3.5)
        self.ax_network.axis('off')

        # [Visualization] Draw connection lines
        for layer_idx in range(len(layer_positions) - 1):
            for i, pos1 in enumerate(layer_positions[layer_idx]):
                for j, pos2 in enumerate(layer_positions[layer_idx + 1]):
                    weight = self.nn.weights[layer_idx][i][j]

                    # Color: blue for positive weights, red for negative weights
                    color = '#2196F3' if weight >= 0 else '#F44336'

                    # Transparency: larger absolute weight means more opaque
                    alpha = min(0.9, abs(weight) * 0.8 + 0.2)

                    # Line width: larger absolute weight means thicker line
                    linewidth = abs(weight) * 4 + 0.5

                    # Draw connection line
                    self.ax_network.plot(
                        [pos1[0], pos2[0]],
                        [pos1[1], pos2[1]],
                        color=color,
                        alpha=alpha,
                        linewidth=linewidth,
                        solid_capstyle='round'
                    )

        # [Visualization] Draw neuron nodes
        for layer_idx, layer in enumerate(layer_positions):
            for neuron_idx, pos in enumerate(layer):
                # Get activation value
                if layer_idx < len(self.nn.activations):
                    activation = self.nn.activations[layer_idx][0][neuron_idx]
                else:
                    activation = 0.5

                # Color mapping
                color = plt.cm.RdYlBu_r(activation)

                # Draw circular node
                circle = plt.Circle(
                    pos, 0.2,
                    color=color,
                    ec='#333333',
                    linewidth=2,
                    zorder=10
                )
                self.ax_network.add_patch(circle)

                # Display activation value at node center
                text_color = 'white' if activation < 0.3 or activation > 0.7 else 'black'
                self.ax_network.text(
                    pos[0], pos[1],
                    f'{activation:.2f}',
                    fontsize=8,
                    fontweight='bold',
                    ha='center',
                    va='center',
                    color=text_color
                )

        # Add layer labels
        layer_names = ['Input'] + [f'Hidden {i+1}' for i in range(len(self.nn.layer_sizes)-2)] + ['Output']
        for i, (name, x_pos) in enumerate(zip(layer_names, x_positions)):
            self.ax_network.text(
                x_pos, -0.4,
                name,
                fontsize=10,
                fontweight='bold',
                ha='center'
            )

    def draw_loss_curve(self):
        """[Visualization Core 2] Draw loss function curve"""
        self.ax_loss.clear()
        self.ax_loss.set_title('Training Loss', fontsize=12, fontweight='bold')
        self.ax_loss.grid(True, alpha=0.3, linestyle='--')

        if len(self.losses) > 0:
            self.ax_loss.plot(self.losses, color='#FF5722', linewidth=2, alpha=0.8)
            self.ax_loss.fill_between(range(len(self.losses)), self.losses, alpha=0.3, color='#FF5722')

            # Display current loss value
            self.ax_loss.text(
                0.95, 0.95,
                f'Loss: {self.losses[-1]:.6f}',
                transform=self.ax_loss.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8)
            )

            max_loss = max(self.losses) if self.losses else 1
            self.ax_loss.set_ylim(0, max_loss * 1.1)
            self.ax_loss.set_xlim(0, len(self.losses))

    def draw_accuracy_curve(self):
        """Draw accuracy curve"""
        self.ax_accuracy.clear()
        self.ax_accuracy.set_title('Accuracy', fontsize=12, fontweight='bold')
        self.ax_accuracy.grid(True, alpha=0.3, linestyle='--')

        if len(self.accuracies) > 0:
            self.ax_accuracy.plot(self.accuracies, color='#4CAF50', linewidth=2, alpha=0.8)
            self.ax_accuracy.fill_between(range(len(self.accuracies)), self.accuracies, alpha=0.3, color='#4CAF50')

            # Display current accuracy
            self.ax_accuracy.text(
                0.95, 0.05,
                f'Acc: {self.accuracies[-1]:.1%}',
                transform=self.ax_accuracy.transAxes,
                fontsize=9,
                verticalalignment='bottom',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8)
            )

            self.ax_accuracy.set_ylim(0, 1.1)
            self.ax_accuracy.set_xlim(0, len(self.accuracies))

    def draw_decision_boundary(self):
        """[Visualization Core 3] Draw decision boundary"""
        # Remove previous colorbar to prevent accumulation
        if self.colorbar is not None:
            self.colorbar.remove()
            self.colorbar = None

        self.ax_boundary.clear()
        self.ax_boundary.set_title('Decision Boundary', fontsize=12, fontweight='bold')
        self.ax_boundary.set_xlabel('X1', fontsize=10)
        self.ax_boundary.set_ylabel('X2', fontsize=10)

        # Create grid points
        resolution = 100
        x_min, x_max = self.X[:, 0].min() - 0.5, self.X[:, 0].max() + 0.5
        y_min, y_max = self.X[:, 1].min() - 0.5, self.X[:, 1].max() + 0.5

        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, resolution),
            np.linspace(y_min, y_max, resolution)
        )

        # Use neural network to predict the class of each grid point
        grid_points = np.c_[xx.ravel(), yy.ravel()]
        predictions = self.nn.forward(grid_points, training=False)
        predictions = predictions.reshape(xx.shape)

        # Draw decision boundary
        contour = self.ax_boundary.contourf(
            xx, yy, predictions,
            levels=50,
            cmap='RdYlBu',
            alpha=0.8
        )

        # Draw decision boundary line
        self.ax_boundary.contour(
            xx, yy, predictions,
            levels=[0.5],
            colors='black',
            linewidths=3,
            linestyles='--'
        )

        # Add color bar (store reference to prevent accumulation)
        self.colorbar = plt.colorbar(contour, ax=self.ax_boundary)
        self.colorbar.set_label('Output Probability', fontsize=9)

        # Draw data points
        colors = ['#F44336' if label == 0 else '#2196F3' for label in self.y.flatten()]
        self.ax_boundary.scatter(
            self.X[:, 0], self.X[:, 1],
            c=colors,
            s=200,
            edgecolors='black',
            linewidths=2,
            zorder=10
        )

        self.ax_boundary.set_xlim(x_min, x_max)
        self.ax_boundary.set_ylim(y_min, y_max)

    def draw_weight_distribution(self):
        """[Visualization Core 4] Draw weight distribution"""
        self.ax_weights.clear()
        self.ax_weights.set_title('Weight Distribution', fontsize=12, fontweight='bold')

        # Draw weight distribution for each layer
        for i, w in enumerate(self.nn.weights):
            weights_flat = w.flatten()
            self.ax_weights.hist(
                weights_flat,
                bins=30,
                alpha=0.5,
                label=f'Layer {i+1}',
                density=True
            )

        self.ax_weights.set_xlabel('Weight Value', fontsize=10)
        self.ax_weights.set_ylabel('Density', fontsize=10)
        self.ax_weights.legend(fontsize=8)
        self.ax_weights.grid(True, alpha=0.3, linestyle='--')

    def draw_learning_rate_curve(self):
        """Draw learning rate change curve"""
        self.ax_lr.clear()
        self.ax_lr.set_title('Learning Rate', fontsize=12, fontweight='bold')
        self.ax_lr.grid(True, alpha=0.3, linestyle='--')

        if len(self.learning_rates) > 0:
            self.ax_lr.plot(self.learning_rates, color='#9C27B0', linewidth=2, alpha=0.8)

            # Display current learning rate
            self.ax_lr.text(
                0.95, 0.95,
                f'LR: {self.learning_rates[-1]:.6f}',
                transform=self.ax_lr.transAxes,
                fontsize=9,
                verticalalignment='top',
                horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='plum', alpha=0.8)
            )

            self.ax_lr.set_xlim(0, len(self.learning_rates))

    def draw_prediction_table(self):
        """[Visualization Core 5] Draw prediction result table"""
        self.ax_table.clear()
        self.ax_table.set_title('Prediction Results Comparison', fontsize=12, fontweight='bold')
        self.ax_table.axis('off')

        # Get prediction results
        predictions = self.nn.forward(self.X, training=False)

        # Prepare table data
        cell_text = []
        cell_colors = []

        for i in range(len(self.X)):
            x1, x2 = self.X[i]
            y_true = self.y[i][0]
            y_pred = predictions[i][0]
            error = abs(y_true - y_pred)
            is_correct = error < 0.1

            row = [
                f'({x1:.2f}, {x2:.2f})',
                f'{y_true:.0f}',
                f'{y_pred:.4f}',
                f'{error:.4f}',
                '✓' if is_correct else '✗'
            ]
            cell_text.append(row)

            if is_correct:
                cell_colors.append(['#E8F5E9', '#E8F5E9', '#E8F5E9', '#E8F5E9', '#C8E6C9'])
            else:
                cell_colors.append(['#FFEBEE', '#FFEBEE', '#FFEBEE', '#FFEBEE', '#FFCDD2'])

        # Create table
        table = self.ax_table.table(
            cellText=cell_text,
            colLabels=['Input', 'Expected', 'Predicted', 'Error', 'Status'],
            cellColours=cell_colors,
            colColours=['#3F51B5'] * 5,
            cellLoc='center',
            loc='center'
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.0, 1.8)

        # Set header style
        for j in range(5):
            cell = table[0, j]
            cell.set_text_props(color='white', fontweight='bold')

        # Compute accuracy
        accuracy = sum(1 for i in range(len(self.X))
                      if abs(predictions[i][0] - self.y[i][0]) < 0.1) / len(self.X)

        self.ax_table.text(
            0.5, -0.15,
            f'Accuracy: {accuracy:.1%} | Dataset: {self.dataset_name}',
            transform=self.ax_table.transAxes,
            fontsize=12,
            fontweight='bold',
            ha='center',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.8)
        )

    def update(self, frame, epochs, steps_per_frame=10):
        """
        Update function - called once per frame for animation
        """
        # Skip training if early stopping already triggered
        if self.nn.early_stopped:
            return []

        for _ in range(steps_per_frame):
            loss, accuracy, lr = self.nn.train_step(self.X, self.y)
            self.losses.append(loss)
            self.accuracies.append(accuracy)
            self.learning_rates.append(lr)

            # Check early stopping after each step
            if self.nn.early_stop(loss):
                print(f"Early stopping triggered at step {frame * steps_per_frame + _}")
                break

        # Update all visualizations
        current_epoch = frame * steps_per_frame
        self.draw_network_structure(current_epoch, epochs)
        self.draw_loss_curve()
        self.draw_accuracy_curve()
        self.draw_decision_boundary()
        self.draw_weight_distribution()
        self.draw_learning_rate_curve()
        self.draw_prediction_table()

        # Update main title
        progress = min(100.0, current_epoch / epochs * 100)
        self.fig.suptitle(
            f'Neural Network Learning {self.dataset_name} Problem - Backpropagation Visualization\n'
            f'Progress: {progress:.1f}% (Epoch: {current_epoch}/{epochs}) | '
            f'Loss: {self.losses[-1]:.6f} | Acc: {self.accuracies[-1]:.1%}',
            fontsize=18,
            fontweight='bold',
            y=0.98
        )

        return []

    def run_animation(self, epochs=1000, steps_per_frame=10, interval=50):
        """
        Run animation
        """
        # Initial drawing
        self.draw_network_structure(0, epochs)
        self.draw_loss_curve()
        self.draw_accuracy_curve()
        self.draw_decision_boundary()
        self.draw_weight_distribution()
        self.draw_learning_rate_curve()
        self.draw_prediction_table()

        # Create animation
        num_frames = epochs // steps_per_frame
        ani = FuncAnimation(
            self.fig,
            self.update,
            fargs=(epochs, steps_per_frame),
            frames=num_frames,
            interval=interval,
            repeat=False,
            blit=False
        )

        plt.tight_layout()
        plt.show()

        return ani


# ============================================================================
# [Core Module 6] Main Program - Assembling All Modules
# ============================================================================

def print_banner():
    """Print program banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║               Neural Network Learning System - Backpropagation Visualization ║
║                           Enhanced Neural Network v2.0                        ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""
    print(banner)


def print_dataset_info(dataset_name, X, y):
    """Print dataset information"""
    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")
    print(f"Number of samples: {len(X)}")
    print(f"Feature dimensions: {X.shape[1]}")
    print(f"\nTruth table:")
    print("-" * 40)
    for i in range(len(X)):
        print(f"Input: {X[i]} -> Output: {y[i][0]}")
    print("-" * 40)


def print_network_info(nn):
    """Print network information"""
    print(f"\n{'='*60}")
    print(f"Neural Network Configuration")
    print(f"{'='*60}")
    print(f"Network structure: {' → '.join(map(str, nn.layer_sizes))}")
    print(f"Activation function: {nn.activation_name}")
    print(f"Initial learning rate: {nn.learning_rate}")
    print(f"Learning rate scheduler: {nn.scheduler.scheduler_type}")
    print(f"Optimizer: {nn.optimizer_type}")
    print(f"Loss function: {nn.loss_type}")
    print(f"Regularization: {nn.reg_type}")
    print(f"Total parameters: {sum(w.size + b.size for w, b in zip(nn.weights, nn.biases))}")
    print(f"{'='*60}")


def print_training_result(nn, X, y, dataset_name):
    """Print training results"""
    print(f"\n{'='*60}")
    print(f"Training Complete - {dataset_name}")
    print(f"{'='*60}")

    predictions = nn.forward(X, training=False)

    print(f"\nPrediction Results:")
    print("-" * 50)
    for i in range(len(X)):
        x1, x2 = X[i]
        y_true = y[i][0]
        y_pred = predictions[i][0]
        error = abs(y_true - y_pred)
        status = "✓ Correct" if error < 0.1 else "✗ Wrong"

        print(f"Input: ({x1:.2f}, {x2:.2f}) -> "
              f"Predicted: {y_pred:.4f} "
              f"(Expected: {y_true:.0f}) "
              f"[{status}]")

    final_loss = nn.compute_loss(y, predictions)
    accuracy = sum(1 for i in range(len(X))
                  if abs(predictions[i][0] - y[i][0]) < 0.1) / len(X)

    print("-" * 50)
    print(f"Final loss: {final_loss:.6f}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"Total training epochs: {len(nn.training_history['loss'])}")

    # [New] Print evaluation metrics
    metrics = EvaluationMetrics.all_metrics(y, predictions)
    print(f"\n[Evaluation Metrics]")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall: {metrics['recall']:.4f}")
    print(f"  F1 Score: {metrics['f1']:.4f}")
    cm = metrics['confusion_matrix']
    print(f"\n[Confusion Matrix]")
    print(f"  TP: {cm['TP']}  FP: {cm['FP']}")
    print(f"  FN: {cm['FN']}  TN: {cm['TN']}")
    print(f"{'='*60}")


def main():
    """
    Main function - Demonstrates the complete process of neural network
    learning multiple logic gate problems

    Core value of this program:
    1. Complete implementation of the backpropagation algorithm
    2. Supports multiple activation functions and learning rate scheduling
    3. Makes abstract mathematics intuitive through visualization
    4. Demonstrates the essence of machine learning: learning patterns from data
    5. Implements complex functionality with concise code

    Historical significance:
    In 1969, Minsky and Papert proved that single-layer perceptrons cannot
    solve the XOR problem, leading to the first AI winter for neural network
    research. It was not until the backpropagation algorithm was proposed in
    1986 that the field was reignited. Today, deep learning based on
    backpropagation has changed the world.
    """

    print_banner()

    # Configuration parameters
    config = {
        'dataset': 'xor',           # Dataset: xor, and, or, nand, xnor, circle, spiral
        'network_structure': [2, 8, 8, 1],  # Network structure
        'activation': 'sigmoid',     # Activation function: sigmoid, relu, tanh, leaky_relu, elu
        'learning_rate': 0.5,         # Initial learning rate
        'scheduler_type': 'cosine',  # Learning rate scheduler: constant, exponential, cosine, step
        'optimizer_type': 'adam',    # Optimizer: sgd, momentum, adam
        'loss_type': 'mse',          # Loss function: mse, binary_crossentropy
        'reg_type': 'none',          # Regularization: none, l1, l2, dropout
        'reg_lambda': 0.01,          # Regularization strength
        'dropout_rate': 0.0,         # Dropout rate (only for dropout reg)
        'epochs': 1000,              # Training epochs
        'steps_per_frame': 10,       # Training steps per frame
        'interval': 50               # Animation frame interval (ms)
    }

    print("\n[Configuration Parameters]")
    for key, value in config.items():
        print(f"  {key}: {value}")

    # Create dataset
    X, y, dataset_name = DatasetFactory.get_dataset(config['dataset'])
    print_dataset_info(dataset_name, X, y)

    # Create neural network
    nn = NeuralNetwork(
        layer_sizes=config['network_structure'],
        activation=config['activation'],
        learning_rate=config['learning_rate'],
        scheduler_type=config['scheduler_type'],
        optimizer_type=config['optimizer_type'],
        loss_type=config['loss_type'],
        reg_type=config['reg_type'],
        reg_lambda=config['reg_lambda'],
        dropout_rate=config['dropout_rate']
    )
    print_network_info(nn)

    # Create visualizer
    visualizer = EnhancedVisualizer(nn, X, y, dataset_name)

    print("\nStarting training animation...")
    print("Please observe:")
    print("  1. Weight changes in the neural network structure")
    print("  2. The decline of the loss function curve")
    print("  3. The formation of the decision boundary")
    print("  4. Changes in weight distribution")
    print("  5. Dynamic adjustment of learning rate")
    print()

    # Run animation
    ani = visualizer.run_animation(
        epochs=config['epochs'],
        steps_per_frame=config['steps_per_frame'],
        interval=config['interval']
    )

    # Display final results after training
    print_training_result(nn, X, y, dataset_name)

    print("\n" + "=" * 60)
    print("[Summary]")
    print("=" * 60)
    print("This program demonstrates how neural networks learn through")
    print("the backpropagation algorithm.")
    print("From initially random weights to eventually correct predictions,")
    print("this process is the essence of machine learning:")
    print("automatically discovering patterns from data.")
    print()
    print("The beauty of code lies in:")
    print("  1. Mathematical beauty: matrix operations, chain rule, gradient descent")
    print("  2. Simplicity beauty: modular design, clear code structure")
    print("  3. Visualization beauty: making abstract algorithms visible and intuitive")
    print("  4. Practical beauty: this principle underpins today's AI revolution")
    print("=" * 60)

    # [New] Save the trained model
    ModelPersistence.save_model(nn, "trained_model.npz")

    return nn, visualizer


# ============================================================================
# Program Entry Point
# ============================================================================

if __name__ == "__main__":
    # Set random seed for reproducible results
    np.random.seed(42)

    # Run main program
    nn, visualizer = main()

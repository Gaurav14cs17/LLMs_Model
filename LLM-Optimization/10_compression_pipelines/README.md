# Module 10: Compression Pipelines

## 🎯 Overview

This module covers the mathematical framework for combining compression techniques and optimizing deployment.

---

## 📐 Mathematical Framework

### 1. Compression Composition

**Single technique:** $C: W \to \hat{W}$ with error $\epsilon$
$$\|W - \hat{W}\| \leq \epsilon$$

**Composed techniques:** $C_n \circ C_{n-1} \circ \ldots \circ C_1$

**Theorem 1 (Error Accumulation):**

For compression pipeline $C_1, C_2, \ldots, C_n$ with errors $\epsilon_1, \ldots, \epsilon_n$:

$$\|W - C_n \circ \ldots \circ C_1(W)\| \leq \sum_{i=1}^{n} \epsilon_i \prod_{j>i} L_j$$

Where $L_j$ is the Lipschitz constant of $C_j$.

**Special case (linear compressions):**
$$\|W - C_n \circ \ldots \circ C_1(W)\| \leq \sum_{i=1}^{n} \epsilon_i$$

### 2. Optimal Ordering

**Theorem 2:** For independent compressions, the optimal order is:
$$\text{Sort by } \frac{\epsilon_i}{1 - \rho_i}$$

Where $\rho_i$ = compression ratio.

**Proof:** Minimizes total error subject to target compression.

---

## 📊 Compression-Accuracy Trade-off

### Pareto Frontier

**Definition:** The Pareto frontier $\mathcal{P}$ is:
$$\mathcal{P} = \{(\rho, L) : \nexists (\rho', L') \text{ with } \rho' > \rho \text{ and } L' < L\}$$

Where $\rho$ = compression ratio, $L$ = loss.

### Theorem 3 (Convex Frontier)

For convex combination of compression techniques:
$$C_{\lambda} = \lambda C_1 + (1-\lambda) C_2$$

The frontier is convex:
$$L(C_{\lambda}) \leq \lambda L(C_1) + (1-\lambda) L(C_2)$$

### Optimal Compression Selection

**Problem:**
$$\min_{C \in \mathcal{C}} L(C(W)) \quad \text{s.t.} \quad \rho(C) \geq \rho_{target}$$

**Lagrangian:**
$$\mathcal{L}(C, \lambda) = L(C(W)) + \lambda(\rho_{target} - \rho(C))$$

---

## 📐 Calibration Theory

### Calibration Dataset Requirements

**Theorem 4 (Sample Complexity):**

For calibration dataset $D$ with $n$ samples:
$$\mathbb{P}\left[\left|\frac{1}{n}\sum_i f(x_i) - \mathbb{E}[f(x)]\right| > \epsilon\right] \leq 2\exp\left(-\frac{2n\epsilon^2}{R^2}\right)$$

Where $R$ = range of $f$.

**Implication:** Need $n = O(R^2/\epsilon^2)$ samples for $\epsilon$ accuracy.

### Calibration for Quantization

**Optimal scale estimation:**
$$\hat{s} = \arg\min_s \mathbb{E}[(X - Q_s(X))^2]$$

With $n$ samples, estimation error:
$$|\hat{s} - s^*| \leq O\left(\frac{\sigma}{\sqrt{n}}\right)$$

---

## 📊 Layer-wise Optimization

### Theorem 5 (Layer Sensitivity)

Define sensitivity of layer $l$:
$$S_l(\epsilon) = \frac{\partial \mathcal{L}}{\partial \|W_l - \hat{W}_l\|}\Big|_{\|\cdot\| = \epsilon}$$

**Optimal budget allocation:**
$$\epsilon_l^* \propto S_l^{-1}$$

Less sensitive layers get more compression.

### Fisher Information for Sensitivity

$$S_l \approx \text{tr}(F_l)$$

Where $F_l$ is Fisher information matrix:
$$F_l = \mathbb{E}\left[\nabla_{\theta_l} \log p(y|x) \nabla_{\theta_l} \log p(y|x)^T\right]$$

---

## 📐 Inference Optimization

### Batching Analysis

**Theorem 6 (Optimal Batch Size):**

For memory $M$, model size $S$, per-sample memory $m$:
$$B^* = \frac{M - S}{m}$$

**Throughput as function of batch:**
$$\text{Throughput}(B) = \frac{B}{\text{Latency}(B)}$$

### Memory-Latency Trade-off

**KV-cache memory:**
$$M_{KV} = 2 \cdot B \cdot S \cdot L \cdot h \cdot d \cdot b_{precision}$$

**Maximum sequence length:**
$$S_{max} = \frac{M_{available} - M_{model}}{2 \cdot B \cdot L \cdot h \cdot d \cdot b}$$

---

## 📊 Deployment Metrics

### Latency Decomposition

$$T_{total} = T_{load} + T_{compute} + T_{memory} + T_{network}$$

**Compute-bound:**
$$T_{compute} = \frac{\text{FLOPs}}{\text{GPU FLOPS}}$$

**Memory-bound:**
$$T_{memory} = \frac{\text{Bytes transferred}}{\text{Bandwidth}}$$

### Roofline Model

**Performance bound:**
$$\text{FLOPS}_{achieved} \leq \min\left(\text{Peak FLOPS}, \text{Bandwidth} \times \text{Arithmetic Intensity}\right)$$

**Arithmetic intensity:**
$$I = \frac{\text{FLOPs}}{\text{Bytes}}$$

For transformer: $I \approx 2d$ (for large batch).

---

## 📖 Summary

| Concept | Formula |
|---------|---------|
| Error accumulation | $\sum_i \epsilon_i \prod_{j>i} L_j$ |
| Sample complexity | $n = O(R^2/\epsilon^2)$ |
| Sensitivity | $S_l = \partial\mathcal{L}/\partial\|\Delta W_l\|$ |
| Optimal batch | $B^* = (M-S)/m$ |
| Roofline | $\min(\text{Peak}, BW \times I)$ |

---

## ➡️ Next Module

Continue to [Module 11: Tools](../11_tools/) for practical implementation guides.

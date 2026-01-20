# Module 10: Compression Pipelines

## 🎯 Overview

This module covers the mathematical framework for combining compression techniques and optimizing deployment.

---

## 📐 Mathematical Framework

### 1. Compression Composition

**Single technique:** $C: W \to \hat{W}$ with error $\epsilon$
```math
\|W - \hat{W}\| \leq \epsilon
```

**Composed techniques:** $C\_n \circ C\_{n-1} \circ \ldots \circ C\_1$

**Theorem 1 (Error Accumulation):**

For compression pipeline $C\_1, C\_2, \ldots, C\_n$ with errors $\epsilon\_1, \ldots, \epsilon\_n$:

```math
\|W - C_n \circ \ldots \circ C_1(W)\| \leq \sum_{i=1}^{n} \epsilon_i \prod_{j>i} L_j
```

Where $L\_j$ is the Lipschitz constant of $C\_j$.

**Special case (linear compressions):**
```math
\|W - C_n \circ \ldots \circ C_1(W)\| \leq \sum_{i=1}^{n} \epsilon_i
```

### 2. Optimal Ordering

**Theorem 2:** For independent compressions, the optimal order is:
```math
\text{Sort by } \frac{\epsilon_i}{1 - \rho_i}
```

Where $\rho\_i$ = compression ratio.

**Proof:** Minimizes total error subject to target compression.

---

## 📊 Compression-Accuracy Trade-off

### Pareto Frontier

**Definition:** The Pareto frontier $\mathcal{P}$ is:
```math
\mathcal{P} = \{(\rho, L) : \nexists (\rho', L') \text{ with } \rho' > \rho \text{ and } L' < L\}
```

Where $\rho$ = compression ratio, $L$ = loss.

### Theorem 3 (Convex Frontier)

For convex combination of compression techniques:
```math
C_{\lambda} = \lambda C_1 + (1-\lambda) C_2
```

The frontier is convex:
```math
L(C_{\lambda}) \leq \lambda L(C_1) + (1-\lambda) L(C_2)
```

### Optimal Compression Selection

**Problem:**
```math
\min_{C \in \mathcal{C}} L(C(W)) \quad \text{s.t.} \quad \rho(C) \geq \rho_{target}
```

**Lagrangian:**
```math
\mathcal{L}(C, \lambda) = L(C(W)) + \lambda(\rho_{target} - \rho(C))
```

---

## 📐 Calibration Theory

### Calibration Dataset Requirements

**Theorem 4 (Sample Complexity):**

For calibration dataset $D$ with $n$ samples:
```math
\mathbb{P}\left[\left|\frac{1}{n}\sum_i f(x_i) - \mathbb{E}[f(x)]\right| > \epsilon\right] \leq 2\exp\left(-\frac{2n\epsilon^2}{R^2}\right)
```

Where $R$ = range of $f$.

**Implication:** Need $n = O(R^2/\epsilon^2)$ samples for $\epsilon$ accuracy.

### Calibration for Quantization

**Optimal scale estimation:**
```math
\hat{s} = \arg\min_s \mathbb{E}[(X - Q_s(X))^2]
```

With $n$ samples, estimation error:
```math
|\hat{s} - s^*| \leq O\left(\frac{\sigma}{\sqrt{n}}\right)
```

---

## 📊 Layer-wise Optimization

### Theorem 5 (Layer Sensitivity)

Define sensitivity of layer $l$:
```math
S_l(\epsilon) = \frac{\partial \mathcal{L}}{\partial \|W_l - \hat{W}_l\|}\Big|_{\|\cdot\| = \epsilon}
```

**Optimal budget allocation:**
```math
\epsilon_l^* \propto S_l^{-1}
```

Less sensitive layers get more compression.

### Fisher Information for Sensitivity

```math
S_l \approx \text{tr}(F_l)
```

Where $F\_l$ is Fisher information matrix:
```math
F_l = \mathbb{E}\left[\nabla_{\theta_l} \log p(y|x) \nabla_{\theta_l} \log p(y|x)^T\right]
```

---

## 📐 Inference Optimization

### Batching Analysis

**Theorem 6 (Optimal Batch Size):**

For memory $M$, model size $S$, per-sample memory $m$:
```math
B^* = \frac{M - S}{m}
```

**Throughput as function of batch:**
```math
\text{Throughput}(B) = \frac{B}{\text{Latency}(B)}
```

### Memory-Latency Trade-off

**KV-cache memory:**
```math
M_{KV} = 2 \cdot B \cdot S \cdot L \cdot h \cdot d \cdot b_{precision}
```

**Maximum sequence length:**
```math
S_{max} = \frac{M_{available} - M_{model}}{2 \cdot B \cdot L \cdot h \cdot d \cdot b}
```

---

## 📊 Deployment Metrics

### Latency Decomposition

```math
T_{total} = T_{load} + T_{compute} + T_{memory} + T_{network}
```

**Compute-bound:**
```math
T_{compute} = \frac{\text{FLOPs}}{\text{GPU FLOPS}}
```

**Memory-bound:**
```math
T_{memory} = \frac{\text{Bytes transferred}}{\text{Bandwidth}}
```

### Roofline Model

**Performance bound:**
```math
\text{FLOPS}_{achieved} \leq \min\left(\text{Peak FLOPS}, \text{Bandwidth} \times \text{Arithmetic Intensity}\right)
```

**Arithmetic intensity:**
```math
I = \frac{\text{FLOPs}}{\text{Bytes}}
```

For transformer: $I \approx 2d$ (for large batch).

---

## 📖 Summary

| Concept | Formula |
|---------|---------|
| Error accumulation | $\sum\_i \epsilon\_i \prod\_{j>i} L\_j$ |
| Sample complexity | $n = O(R^2/\epsilon^2)$ |
| Sensitivity | $S\_l = \partial\mathcal{L}/\partial\|\Delta W\_l\|$ |
| Optimal batch | $B^* = (M-S)/m$ |
| Roofline | $\min(\text{Peak}, BW \times I)$ |

---

## ➡️ Next Module

Continue to [Module 11: Tools](../11_tools/) for practical implementation guides.

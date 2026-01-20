# Module 01: Introduction to LLM Optimization

## 🎯 Overview

Large Language Models (LLMs) have revolutionized NLP but come with significant computational costs. This module introduces the fundamental concepts, mathematics, and theory behind model compression.

---

## 📊 Mathematical Foundations

### 1. Model Size Analysis

A transformer-based LLM consists of several components. The total parameter count is:

```math
P_{total} = P_{embed} + L \cdot P_{layer} + P_{head}

```

Where:

- $P\_{embed}$ = Embedding parameters

- $L$ = Number of layers

- $P\_{layer}$ = Parameters per layer

- $P\_{head}$ = Output head parameters

#### Embedding Layer

```math
P_{embed} = V \cdot d_{model}

```

Where:

- $V$ = Vocabulary size (typically 32K-100K)

- $d\_{model}$ = Model hidden dimension

#### Transformer Layer
Each transformer layer contains:

**Self-Attention:**

```math
P_{attn} = 4 \cdot d_{model}^2

```

(For Q, K, V, and output projections)

**Feed-Forward Network:**

```math
P_{ffn} = 2 \cdot d_{model} \cdot d_{ff} = 2 \cdot d_{model} \cdot (4 \cdot d_{model}) = 8 \cdot d_{model}^2

```

**Layer Normalization:**

```math
P_{ln} = 4 \cdot d_{model}

```

**Total per layer:**

```math
P_{layer} = P_{attn} + P_{ffn} + P_{ln} = 12 \cdot d_{model}^2 + 4 \cdot d_{model}

```

#### Example: LLaMA-7B

- $V = 32,000$

- $d\_{model} = 4,096$

- $L = 32$

- $d\_{ff} = 11,008$

```math
P_{total} \approx 32,000 \times 4,096 + 32 \times 12 \times 4,096^2 \approx 7B

```

---

### 2. Memory Requirements

#### Inference Memory

For inference with batch size $B$ and sequence length $S$:

```math
M_{inference} = M_{weights} + M_{activations} + M_{KV-cache}

```

**Weights:**

```math
M_{weights} = P_{total} \times b_w

```

Where $b\_w$ = bytes per weight (4 for FP32, 2 for FP16, 0.5 for INT4)

**KV-Cache:**

```math
M_{KV} = 2 \times B \times S \times L \times d_{model} \times b_a

```

The factor of 2 accounts for both K and V.

**Example:** LLaMA-7B with B=1, S=2048, FP16:

```math
M_{KV} = 2 \times 1 \times 2048 \times 32 \times 4096 \times 2 = 1.07 \text{ GB}

```

#### Training Memory

Training requires additional memory for:

- Gradients: Same size as weights

- Optimizer states: 2× weights for Adam (momentum + variance)

- Activations: Depends on batch size and checkpointing

```math
M_{training} \approx M_{weights} \times (1 + 1 + 2) + M_{activations}
M_{training} \approx 4 \times M_{weights} + M_{activations}

```

---

### 3. Computational Complexity

#### FLOPs per Token

For a single forward pass with sequence length $S$:

**Self-Attention:**

```math
\text{FLOPs}_{attn} = 2 \times B \times S^2 \times d_{model} + 4 \times B \times S \times d_{model}^2

```

The $S^2$ term dominates for long sequences (attention computation).

**Feed-Forward:**

```math
\text{FLOPs}_{ffn} = 2 \times B \times S \times d_{model} \times d_{ff}

```

**Total per layer:**

```math
\text{FLOPs}_{layer} \approx 2BS^2d + 4BSd^2 + 4BSd \cdot d_{ff}

```

**Approximate total (ignoring attention for short sequences):**

```math
\text{FLOPs}_{total} \approx 2 \times P_{total} \times S

```

---

## 📈 Compression Theory

### 1. Compression Ratio

**Definition:** The compression ratio $\rho$ is defined as:

```math
\rho = \frac{|M_{original}|}{|M_{compressed}|}

```

Where $|M|$ denotes the size of model $M$ in bits or bytes.

### 2. Rate-Distortion Theory

Compression involves a fundamental trade-off between:

- **Rate (R):** Bits used to represent the model

- **Distortion (D):** Loss in model quality

The rate-distortion function $R(D)$ gives the minimum rate required to achieve distortion ≤ D:

```math
R(D) = \min_{p(\hat{w}|w): \mathbb{E}[d(w,\hat{w})] \leq D} I(W; \hat{W})

```

Where:

- $W$ = original weights

- $\hat{W}$ = compressed weights

- $d(w, \hat{w})$ = distortion measure

- $I(W; \hat{W})$ = mutual information

### 3. Information-Theoretic Lower Bounds

**Shannon's Source Coding Theorem:**

For lossless compression, the minimum expected code length satisfies:

```math
H(W) \leq \mathbb{E}[L(W)] < H(W) + 1

```

Where $H(W)$ is the entropy of the weight distribution.

**For neural networks:** If weights are approximately Gaussian with variance $\sigma^2$:

```math
H(W) = \frac{1}{2}\log_2(2\pi e \sigma^2) \text{ bits per weight}

```

---

## 📊 Quality Metrics

### 1. Perplexity

**Definition:** For a language model with probability distribution $p$:

```math
\text{PPL} = \exp\left(-\frac{1}{N}\sum_{i=1}^{N} \log p(x_i | x_{1:i-1})\right)

```

**Interpretation:**
- Lower perplexity = better model

- Perplexity of $k$ means the model is as uncertain as choosing uniformly among $k$ options

**Cross-Entropy Relationship:**

```math
\text{PPL} = 2^{H(p, q)} = \exp(CE)

```

Where $H(p, q)$ is cross-entropy between true distribution $p$ and model $q$.

### 2. Task-Specific Accuracy

```math
\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}

```

### 3. Compression Quality Trade-off

We define the quality retention ratio:

```math
Q = \frac{\text{Metric}_{compressed}}{\text{Metric}_{original}}

```

And compression efficiency:

```math
\eta = Q \times \rho

```

**Optimal compression** maximizes $\eta$ subject to $Q \geq Q\_{min}$.

---

## 🔬 Theoretical Framework for Compression

### 1. The Bias-Variance Trade-off in Compression

Compression introduces additional error that can be decomposed:

```math
\mathbb{E}[(f(x) - \hat{f}(x))^2] = \underbrace{\text{Bias}^2}_{\text{systematic error}} + \underbrace{\text{Variance}}_{\text{random error}} + \underbrace{\sigma^2}_{\text{irreducible}}

```

Compression typically increases bias but may reduce variance (regularization effect).

### 2. Lottery Ticket Hypothesis (Theoretical Foundation)

**Theorem (Frankle & Carlin, 2019):** A randomly-initialized, dense neural network contains a subnetwork that, when trained in isolation, can match the test accuracy of the original network after training for at most the same number of iterations.

**Mathematical Formulation:**

Let $f(x; \theta)$ be a network with parameters $\theta \in \mathbb{R}^d$.

There exists a mask $m \in \{0, 1\}^d$ with $\|m\|\_0 \ll d$ such that:

```math
\min_t \mathcal{L}(f(x; m \odot \theta_t)) \approx \min_t \mathcal{L}(f(x; \theta_t))

```

Where $\theta\_t$ denotes parameters at training step $t$.

### 3. Neural Network Pruning Theory

**Optimal Brain Damage (LeCun et al., 1990):**

The saliency of weight $w\_i$ is approximated by:

```math
s_i = \frac{1}{2} H_{ii} w_i^2

```

Where $H\_{ii}$ is the $i$-th diagonal element of the Hessian $H = \nabla^2 \mathcal{L}$.

**Proof sketch:**

Taylor expansion of loss around current weights:

```math
\mathcal{L}(\theta + \delta) \approx \mathcal{L}(\theta) + g^T\delta + \frac{1}{2}\delta^T H \delta

```

At a local minimum, $g \approx 0$, so:

```math
\Delta\mathcal{L} \approx \frac{1}{2}\delta^T H \delta

```

For pruning weight $i$ (setting $\delta\_i = -w\_i$):

```math
\Delta\mathcal{L}_i \approx \frac{1}{2} H_{ii} w_i^2

```

---

## 📐 Complexity Analysis

### Time Complexity of Compression Techniques

| Technique | Compression Time | Inference Time |
|-----------|-----------------|----------------|
| Quantization (PTQ) | $O(n)$ | $O(n/k)$ where $k$ = compression ratio |
| Pruning | $O(n \log n)$ for sorting | $O(n \cdot (1-s))$ where $s$ = sparsity |
| Distillation | $O(T \cdot n\_{teacher})$ | $O(n\_{student})$ |
| Low-rank factorization | $O(n \cdot r)$ | $O(n \cdot r / d)$ |

### Space Complexity

| Technique | Model Size | Additional Storage |
|-----------|------------|-------------------|
| INT8 Quantization | $n/4$ | Scales: $n/g$ |
| INT4 Quantization | $n/8$ | Scales + zeros: $2n/g$ |
| Pruning (sparse) | $n \cdot (1-s) + $ indices | Sparse format overhead |
| Low-rank | $(m + n) \cdot r$ | None |

Where $n$ = original parameters, $g$ = group size, $s$ = sparsity, $r$ = rank.

---

## 📚 Key Theorems and Results

### Theorem 1: Universal Approximation with Compressed Networks

**Statement:** For any $\epsilon > 0$ and any continuous function $f$ on a compact set, there exists a compressed network $\hat{f}$ with $\rho$ compression ratio such that $\|f - \hat{f}\|\_\infty < \epsilon$, provided:

```math
\rho < O\left(\frac{1}{\epsilon^{d/r}}\right)

```

Where $d$ = input dimension, $r$ = smoothness of $f$.

### Theorem 2: Quantization Error Bound

**Statement:** For uniform $b$-bit quantization of weights in range $[-M, M]$:

```math
\mathbb{E}[(w - Q(w))^2] \leq \frac{M^2}{3 \cdot 2^{2b}}

```

**Proof:**
The quantization step size is $\Delta = 2M / 2^b$.

For uniform distribution of quantization error in $[-\Delta/2, \Delta/2]$:

```math
\mathbb{E}[e^2] = \int_{-\Delta/2}^{\Delta/2} e^2 \cdot \frac{1}{\Delta} de = \frac{\Delta^2}{12} = \frac{M^2}{3 \cdot 2^{2b}}

```

---

## 📖 Summary of Mathematical Concepts

| Concept | Formula | Significance |
|---------|---------|--------------|
| Model size | $P = 12Ld^2 + Vd$ | Determines memory requirements |
| Compression ratio | $\rho = \|M\_{orig}\| / \|M\_{comp}\|$ | Measures compression effectiveness |
| Perplexity | $\text{PPL} = \exp(-\frac{1}{N}\sum \log p)$ | Measures language model quality |
| Quantization error | $\mathbb{E}[e^2] \leq \frac{M^2}{3 \cdot 2^{2b}}$ | Bounds precision loss |
| Saliency | $s\_i = \frac{1}{2}H\_{ii}w\_i^2$ | Guides pruning decisions |

---

## ➡️ Next Module

Continue to [Module 02: Quantization](../02_quantization/) for detailed mathematical treatment of quantization techniques.

# Module 05: Weight Sharing

## 🎯 Overview

Weight sharing reduces model size by reusing parameters across different parts of the network. This module provides mathematical foundations and proofs for various weight sharing techniques.

---

## 📐 Mathematical Foundations

### 1. Cross-Layer Parameter Sharing

#### ALBERT-Style Sharing

**Standard Transformer:**
$$h^{(l)} = \text{TransformerLayer}_l(h^{(l-1)}; \theta_l)$$

Each layer has its own parameters $\theta_l$.

**ALBERT (Shared):**
$$h^{(l)} = \text{TransformerLayer}(h^{(l-1)}; \theta)$$

All layers share parameters $\theta$.

**Parameter Reduction:**

| Component | Standard (L layers) | ALBERT |
|-----------|--------------------:|-------:|
| Attention | $4Ld^2$ | $4d^2$ |
| FFN | $8Ld^2$ | $8d^2$ |
| **Total** | $12Ld^2$ | $12d^2$ |

**Compression Ratio:** $L$ (e.g., 12× for 12 layers)

### 2. Universal Transformer Interpretation

**Theorem 1 (Recurrent View):**

ALBERT can be viewed as a recurrent network unrolled for $L$ steps:

$$h_t = f(h_{t-1}; \theta) \quad \text{for } t = 1, \ldots, L$$

**Fixed Point Analysis:**

If the transformation is contractive:
$$\|f(h_1; \theta) - f(h_2; \theta)\| \leq \gamma \|h_1 - h_2\|, \quad \gamma < 1$$

Then by Banach fixed-point theorem, there exists unique fixed point $h^*$:
$$h^* = f(h^*; \theta)$$

**Interpretation:** Deep enough ALBERT approximates finding this fixed point.

---

## 📊 Embedding Factorization

### Mathematical Formulation

**Standard Embedding:**
$$E \in \mathbb{R}^{V \times H}$$

Parameters: $V \times H$

**Factorized Embedding:**
$$E = E_1 \cdot E_2, \quad E_1 \in \mathbb{R}^{V \times E}, \quad E_2 \in \mathbb{R}^{E \times H}$$

Parameters: $V \times E + E \times H$

**Theorem 2 (Optimal Embedding Dimension):**

For vocabulary $V$, hidden size $H$, the optimal $E$ minimizing parameters while maintaining rank is:

$$E^* = \min\left(\sqrt{VH}, \text{rank}_{effective}(E_{full})\right)$$

**Proof:**

Parameters: $f(E) = VE + EH = E(V + H)$

Minimizing subject to $E \leq \min(V, H)$:
$$\frac{df}{dE} = V + H$$

This is always positive, so minimum is at smallest $E$ that preserves information.

**Practical choice:** $E = 128$ for $V = 30K$, $H = 768$:
- Standard: $30K \times 768 = 23M$
- Factorized: $30K \times 128 + 128 \times 768 = 3.9M$ (6× smaller)

---

## 📐 Multi-Query Attention (MQA)

### Standard Multi-Head Attention

$$\text{MHA}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)W^O$$

Where each head:
$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

**Parameters per layer:**
$$P_{MHA} = h \cdot d_k \cdot d + h \cdot d_k \cdot d + h \cdot d_v \cdot d + h \cdot d_v \cdot d = 4hd_k d$$

### Multi-Query Attention

**Key Insight:** Share K and V projections across all heads:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW^K, VW^V)$$

**Parameters:**
$$P_{MQA} = h \cdot d_k \cdot d + d_k \cdot d + d_v \cdot d + h \cdot d_v \cdot d = (h+2)d_k d$$

**Reduction ratio:**
$$\frac{P_{MHA}}{P_{MQA}} = \frac{4h}{h+2} \approx 4 \text{ for large } h$$

### KV-Cache Reduction

**Theorem 3 (KV-Cache Memory):**

For batch size $B$, sequence length $S$, layers $L$:

| Method | KV-Cache Size |
|--------|---------------|
| MHA | $2 \cdot B \cdot S \cdot L \cdot h \cdot d_k$ |
| MQA | $2 \cdot B \cdot S \cdot L \cdot d_k$ |

**Reduction:** $h \times$ (e.g., 32× for 32 heads)

**Proof:**

MHA stores K and V for each head: $2 \times h \times d_k$ per position.
MQA stores single K and V: $2 \times d_k$ per position.

---

## 📊 Grouped-Query Attention (GQA)

### Mathematical Formulation

**GQA:** Intermediate between MHA and MQA.

Let $g$ = number of KV head groups, $h$ = query heads:

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_{g(i)}^K, VW_{g(i)}^V)$$

Where $g(i) = \lfloor i \cdot g / h \rfloor$ maps query head to KV group.

**Parameters:**
$$P_{GQA} = h \cdot d_k \cdot d + g \cdot d_k \cdot d + g \cdot d_v \cdot d + h \cdot d_v \cdot d = (h+2g)d_k d$$

**Special Cases:**
- $g = h$: MHA
- $g = 1$: MQA

### Quality vs. Efficiency Trade-off

**Theorem 4:** The approximation error of GQA compared to MHA is:

$$\mathbb{E}[\|\text{MHA}(x) - \text{GQA}(x)\|^2] \leq \frac{h - g}{h} \cdot \sigma_{KV}^2$$

Where $\sigma_{KV}^2$ is variance in KV head outputs.

**Proof sketch:**

Grouping introduces error when heads in same group want different K/V.
Error proportional to within-group variance, which decreases with more groups.

---

## 📐 Weight Clustering

### K-Means Clustering of Weights

**Objective:** Cluster weights into $K$ centroids:

$$\min_{\{c_k\}, \{a_i\}} \sum_i (w_i - c_{a_i})^2$$

Where $a_i \in \{1, \ldots, K\}$ assigns weight $i$ to cluster.

**Storage:**
- $K$ centroids: $K \times 32$ bits (FP32)
- $n$ indices: $n \times \log_2(K)$ bits

**Total bits:** $32K + n\log_2(K)$

**Theorem 5 (Optimal Number of Clusters):**

For $n$ weights with target bits per weight $b$:

$$K^* = \frac{2^b \cdot n}{n + 32 \cdot 2^b / b} \approx 2^b$$

For large $n$.

### Quantization Error

**Theorem 6 (Lloyd-Max Clustering):**

For weights $W \sim \mathcal{N}(0, \sigma^2)$, optimal K-means MSE is:

$$\text{MSE}_K \approx \sigma^2 \cdot \frac{1}{12} \cdot \left(\frac{2\sqrt{3}\sigma}{K}\right)^2 = \frac{\sigma^4}{K^2}$$

For uniformly spaced centroids. Lloyd-Max optimal centroids achieve lower MSE.

---

## 📊 Tied Embeddings

### Input-Output Weight Tying

**Standard:**
- Input embedding: $E_{in} \in \mathbb{R}^{V \times d}$
- Output projection: $W_{out} \in \mathbb{R}^{d \times V}$
- Parameters: $2Vd$

**Tied:**
$$W_{out} = E_{in}^T$$
- Parameters: $Vd$ (2× reduction)

**Theorem 7 (Tying Justification):**

If input and output embeddings share semantic space:
$$\text{sim}(e_i^{in}, e_j^{out}) \propto P(j | i)$$

Then tying improves sample efficiency.

**Proof (Information-Theoretic):**

Tied weights enforce:
$$\langle e_i, e_j \rangle = \langle e_j, e_i \rangle$$

This symmetric similarity is appropriate for bidirectional relationships.

---

## 📐 Universal Function Approximation with Sharing

### Theorem 8 (Expressiveness of Shared Weights)

A transformer with shared weights across $L$ layers can approximate any continuous function $f: \mathbb{R}^{d_{in}} \to \mathbb{R}^{d_{out}}$ to arbitrary precision, given:

1. Sufficient hidden dimension $d$
2. Sufficient depth $L$
3. Skip connections

**Proof sketch:**

1. Each layer applies same nonlinear transformation
2. Skip connections allow information bypass
3. Composition of nonlinearities + skip = universal

This is analogous to the universal approximation theorem for deep networks.

### Practical Implications

**Trade-offs of Weight Sharing:**

| Aspect | Benefit | Cost |
|--------|---------|------|
| Parameters | $L\times$ reduction | None |
| Memory | $L\times$ reduction | None |
| Training | Regularization effect | Slightly harder optimization |
| Capacity | Implicit depth limit | Need more layers for same capacity |

---

## 📊 Information-Theoretic View

### Theorem 9 (Information Bottleneck)

Weight sharing creates an information bottleneck:

$$I(X; h^{(L)}) \leq I(X; \theta)$$

With shared weights, the mutual information is limited by the shared parameter capacity.

**Implication:** Shared models are more regularized.

### Rate-Distortion for Weight Sharing

**Definition:** The rate $R$ is the number of bits to describe the model:

For standard: $R_{std} = n \cdot 32$ bits
For shared: $R_{shared} = (n/L) \cdot 32$ bits

**Theorem 10:** Weight sharing is optimal when:

$$\text{Var}(\theta_l - \theta_{l'}) \ll \text{Var}(\theta_l)$$

i.e., layer weights are similar to each other.

---

## 📖 Summary of Key Formulas

| Concept | Formula |
|---------|---------|
| ALBERT compression | $L\times$ fewer parameters |
| Factorized embedding | $VE + EH$ vs $VH$ |
| MQA KV-cache | $h\times$ smaller |
| GQA parameters | $(h + 2g)d_k d$ |
| Cluster storage | $32K + n\log_2 K$ bits |
| Tied embeddings | $2\times$ reduction |

---

## ➡️ Next Module

Continue to [Module 06: Factorization](../06_factorization/) for low-rank decomposition mathematics.

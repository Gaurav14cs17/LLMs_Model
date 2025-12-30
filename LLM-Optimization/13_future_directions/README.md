# Module 13: Future Directions

## 🎯 Overview

This module covers emerging techniques with their mathematical foundations.

---

## 📐 Speculative Decoding

### Mathematical Framework

**Standard autoregressive generation:**
$$p(x_{1:T}) = \prod_{t=1}^{T} p(x_t | x_{<t})$$

Each token requires one forward pass of the large model.

**Speculative decoding:**
1. Draft model generates $k$ tokens: $\tilde{x}_1, \ldots, \tilde{x}_k$
2. Target model verifies in parallel
3. Accept valid prefix

### Theorem 1 (Speculative Decoding Correctness)

Using rejection sampling, speculative decoding produces samples from the exact target distribution.

**Acceptance criterion:**
$$\text{Accept } \tilde{x}_t \text{ with prob } \min\left(1, \frac{p(\tilde{x}_t | x_{<t})}{q(\tilde{x}_t | x_{<t})}\right)$$

Where $p$ = target, $q$ = draft.

**Proof:**

For accepted tokens:
$$P(\text{accept } x) = q(x) \cdot \min\left(1, \frac{p(x)}{q(x)}\right) = \min(q(x), p(x))$$

Rejection samples from:
$$p_{reject}(x) \propto \max(0, p(x) - q(x))$$

Combined: exact $p(x)$.

### Theorem 2 (Expected Tokens per Step)

Let $\alpha = \mathbb{E}_q[\min(1, p(x)/q(x))]$ be acceptance rate.

Expected accepted tokens per verification:
$$\mathbb{E}[n] = \frac{1 - \alpha^{k+1}}{1 - \alpha}$$

For $\alpha = 0.8$, $k = 5$: $\mathbb{E}[n] \approx 3.4$ tokens.

### Speedup Analysis

**Theorem 3 (Speculative Speedup):**

$$\text{Speedup} = \frac{T_{target}}{T_{draft} \cdot k + T_{target}} \cdot \mathbb{E}[n]$$

For $T_{draft}/T_{target} = 0.1$, $k = 5$, $\alpha = 0.8$:
$$\text{Speedup} \approx \frac{3.4}{0.5 + 1} = 2.3\times$$

---

## 📊 Mixture of Experts Scaling

### Scaling Laws

**Dense scaling law (Kaplan et al.):**
$$L(N) = \left(\frac{N_c}{N}\right)^{\alpha_N}$$

**MoE scaling law:**
$$L(N, E, k) = \left(\frac{N_c}{N_{active}}\right)^{\alpha_N}$$

Where $N_{active} = N/E \cdot k$ (active parameters).

### Theorem 4 (MoE Efficiency)

For same compute budget $C$:

MoE with $E$ experts achieves loss of dense model with $E^{\beta}$ more parameters.

**Empirically:** $\beta \approx 0.5$ to $0.7$.

### Load Balancing

**Auxiliary loss:**
$$\mathcal{L}_{aux} = \alpha \cdot E \cdot \sum_{i=1}^{E} f_i \cdot P_i$$

**Theorem 5:** Minimizing $\mathcal{L}_{aux}$ encourages:
$$f_i = P_i = \frac{1}{E} \quad \forall i$$

---

## 📐 State Space Models (Mamba)

### Mathematical Formulation

**Continuous state space:**
$$\frac{dh(t)}{dt} = Ah(t) + Bx(t)$$
$$y(t) = Ch(t) + Dx(t)$$

**Discretized (ZOH):**
$$h_k = \bar{A}h_{k-1} + \bar{B}x_k$$
$$y_k = Ch_k + Dx_k$$

Where:
$$\bar{A} = e^{\Delta A}, \quad \bar{B} = (\Delta A)^{-1}(e^{\Delta A} - I) \cdot \Delta B$$

### Theorem 6 (Linear Recurrence)

SSMs can be computed in $O(N)$ for sequence length $N$.

**Recurrent mode:** $O(N)$ sequential
**Parallel mode:** $O(N \log N)$ using parallel scan

### Selective State Spaces (Mamba)

**Key innovation:** Make $\Delta$, $B$, $C$ input-dependent:
$$\Delta_k = \text{softplus}(W_\Delta x_k)$$
$$B_k = W_B x_k$$
$$C_k = W_C x_k$$

**Theorem 7:** Selection allows content-based reasoning:
$$\frac{\partial h_k}{\partial x_j} \neq 0 \text{ for } j < k$$

(Unlike fixed SSM where dependencies are position-based)

---

## 📊 Extreme Quantization

### 1-Bit Networks (BitNet)

**Weight representation:**
$$W \in \{-1, +1\}^{m \times n}$$

**Forward pass:**
$$y = W \cdot x = \sum_j W_j \cdot x_j$$

All multiplications become additions/subtractions!

### Theorem 8 (BitNet Approximation)

For weights $W$ and binary approximation $\hat{W} \in \{-1, +1\}$:

$$\hat{W}_{ij} = \text{sign}(W_{ij})$$

Error bound:
$$\mathbb{E}[\|W - \hat{W}\|_F^2] = \sum_{ij} (|W_{ij}| - 1)^2$$

### Ternary Quantization

**Weights:** $W \in \{-\alpha, 0, +\alpha\}$

**Threshold-based:**
$$\hat{W}_{ij} = \begin{cases} +\alpha & W_{ij} > \Delta \\ 0 & |W_{ij}| \leq \Delta \\ -\alpha & W_{ij} < -\Delta \end{cases}$$

**Theorem 9 (Optimal Scale):**
$$\alpha^* = \frac{\sum_{|W_{ij}| > \Delta} |W_{ij}|}{|\{(i,j): |W_{ij}| > \Delta\}|}$$

---

## 📐 KV-Cache Compression

### Quantization

**Per-channel quantization:**
$$K^{quant} = Q(K), \quad V^{quant} = Q(V)$$

**Theorem 10:** INT8 KV-cache has error:
$$\mathbb{E}[\|KV - K^qV^q\|] \leq O(\Delta_K \|V\| + \Delta_V \|K\|)$$

### Eviction Policies

**Attention-based eviction:**
$$\text{Evict } k_i \text{ if } \sum_{j > i} A_{ji} < \tau$$

Low-attention keys are less important.

**Theorem 11 (Eviction Error):**
$$\|O - \hat{O}\| \leq \sum_{i \in \text{evicted}} A_i \cdot \|v_i\|$$

---

## 📊 Continuous Batching

### Mathematical Model

**Request arrival:** Poisson process with rate $\lambda$
**Service time:** Token generation time $\mu^{-1}$

**Theorem 12 (Throughput):**

Without continuous batching (static):
$$\text{Throughput} = \frac{B}{\max_i T_i}$$

With continuous batching:
$$\text{Throughput} = \mu \cdot B_{avg}$$

**Improvement:** Up to $\max_i T_i / \bar{T}$ where $\bar{T}$ = average completion time.

---

## 📖 Summary of Key Formulas

| Concept | Formula |
|---------|---------|
| Speculative acceptance | $\min(1, p(x)/q(x))$ |
| Expected tokens | $(1 - \alpha^{k+1})/(1 - \alpha)$ |
| MoE scaling | $L \propto N_{active}^{-\alpha}$ |
| SSM recurrence | $h_k = \bar{A}h_{k-1} + \bar{B}x_k$ |
| BitNet weights | $W \in \{-1, +1\}$ |
| KV eviction error | $\sum_{i \in evict} A_i \|v_i\|$ |

---

## 🎓 Course Complete!

Congratulations on completing the LLM Optimization course! You now have deep mathematical understanding of:

- ✅ Quantization theory and error bounds
- ✅ Pruning algorithms and saliency
- ✅ Knowledge distillation mathematics
- ✅ Weight sharing and factorization
- ✅ Sparse computation patterns
- ✅ PEFT methods (LoRA, QLoRA)
- ✅ Efficient attention (Flash Attention)
- ✅ Emerging techniques (speculative decoding, MoE, SSMs)

**Happy Optimizing! 🚀**

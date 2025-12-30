# Module 07: Sparsity

## 🎯 Overview

Sparsity exploits the fact that many computations involve zeros. This module provides mathematical foundations for sparse computation patterns.

---

## 📐 Mathematical Foundations

### 1. Sparse Matrix Representation

**Definition:** A matrix $W$ is $s$-sparse if:
$$\|W\|_0 = |\{(i,j) : W_{ij} \neq 0\}| \leq s$$

**Sparsity ratio:**
$$\text{sparsity} = 1 - \frac{\|W\|_0}{mn}$$

### Storage Formats

**Compressed Sparse Row (CSR):**
- Values: non-zero elements
- Column indices: column of each value
- Row pointers: start of each row

**Storage:** $O(nnz + n)$ vs $O(mn)$ for dense.

---

### 2. Sparse Matrix-Vector Multiplication

**Dense:** $y = Wx$

$$y_i = \sum_{j=1}^{n} W_{ij} x_j$$

FLOPs: $2mn$

**Sparse:** Only compute non-zero terms:
$$y_i = \sum_{j: W_{ij} \neq 0} W_{ij} x_j$$

FLOPs: $2 \cdot nnz$

**Speedup (theoretical):**
$$\text{Speedup} = \frac{mn}{nnz} = \frac{1}{1 - \text{sparsity}}$$

---

## 📊 N:M Structured Sparsity

### Definition

**N:M Sparsity:** In every group of M consecutive elements, exactly N are non-zero.

Common pattern: **2:4** (50% sparse)

### Mathematical Formulation

For weight vector $w = [w_1, \ldots, w_M]$:

Find optimal mask $m \in \{0, 1\}^M$ with $\|m\|_0 = N$:

$$\min_{m: \|m\|_0 = N} \|w - m \odot w\|_2^2 = \min_{m} \sum_{i: m_i = 0} w_i^2$$

**Solution:** Keep the $N$ largest magnitude elements.

### Theorem 1 (2:4 Sparsity Error)

For $w \sim \mathcal{N}(0, \sigma^2)^4$, the expected squared error is:

$$\mathbb{E}[\|w - m^* \odot w\|_2^2] = 2\mathbb{E}[w_{(1)}^2 + w_{(2)}^2]$$

Where $w_{(1)} \leq w_{(2)} \leq w_{(3)} \leq w_{(4)}$ are order statistics.

**For standard Gaussian:**
$$\mathbb{E}[w_{(1)}^2] = \sigma^2(1 - \frac{3}{\sqrt{\pi}}\int_0^{\infty} x\Phi(x)^3\phi(x)dx) \approx 0.318\sigma^2$$

**Total expected error:** $\approx \sigma^2$ (25% of total energy per group of 4)

### Hardware Acceleration

**Theorem 2 (Sparse Tensor Core Efficiency):**

For 2:4 sparse matrix multiplication:
$$\text{Speedup} = 2\times \text{ (theoretical)}$$

With overhead from index management: ~$1.8\times$ (practical).

---

## 📐 Mixture of Experts (MoE)

### Mathematical Formulation

**MoE Layer:**
$$y = \sum_{i=1}^{N} G(x)_i \cdot E_i(x)$$

Where:
- $E_i$ = Expert network $i$
- $G(x) = \text{softmax}(W_g x)$ = Gating/router function
- $G(x)_i$ = Probability of selecting expert $i$

### Top-K Routing

**Sparse Gating:**
$$G(x)_i = \begin{cases} \text{softmax}(W_g x)_i & i \in \text{TopK}(W_g x) \\ 0 & \text{otherwise} \end{cases}$$

**Theorem 3 (MoE Parameter vs. Compute):**

For MoE with $N$ experts, top-$k$ routing:
- Total parameters: $P_{total} = P_{router} + N \cdot P_{expert}$
- Active parameters: $P_{active} = P_{router} + k \cdot P_{expert}$
- Compute per token: $\propto P_{active}$

**Example:** Mixtral 8×7B:
- $N = 8$ experts, $k = 2$
- Total: ~47B parameters
- Active: ~13B parameters per token

### Load Balancing

**Problem:** Experts may receive uneven load.

**Auxiliary Loss:**
$$\mathcal{L}_{aux} = \alpha \cdot N \cdot \sum_{i=1}^{N} f_i \cdot P_i$$

Where:
- $f_i$ = Fraction of tokens routed to expert $i$
- $P_i$ = Average routing probability for expert $i$

**Theorem 4:** This loss encourages uniform expert utilization:
$$\min \mathcal{L}_{aux} \Rightarrow f_i = P_i = \frac{1}{N} \quad \forall i$$

**Proof:**

By Cauchy-Schwarz: $\sum_i f_i P_i \geq N (\prod_i f_i P_i)^{1/N}$

Equality when all $f_i P_i$ are equal. Given $\sum_i f_i = 1$ and $\sum_i P_i = 1$, minimum is at $f_i = P_i = 1/N$.

---

## 📊 Sparse Attention

### Standard Attention Complexity

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

**Complexity:**
- Time: $O(n^2 d)$
- Space: $O(n^2)$ for attention matrix

### Sparse Attention Patterns

**Definition:** Sparse attention uses mask $M \in \{0, 1\}^{n \times n}$:
$$A_{ij} = \begin{cases} \text{softmax}(QK^T/\sqrt{d})_{ij} & M_{ij} = 1 \\ 0 & M_{ij} = 0 \end{cases}$$

**Types of Patterns:**

1. **Local (Sliding Window):**
$$M_{ij} = \mathbf{1}[|i - j| \leq w/2]$$

2. **Strided:**
$$M_{ij} = \mathbf{1}[j \mod s = 0]$$

3. **Global + Local:**
$$M_{ij} = \mathbf{1}[|i - j| \leq w/2] \lor \mathbf{1}[i \in \mathcal{G}] \lor \mathbf{1}[j \in \mathcal{G}]$$

### Theorem 5 (Sparse Attention Complexity)

For sparsity pattern with $c$ connections per query:

$$\text{Time} = O(ncd), \quad \text{Space} = O(nc)$$

**For sliding window (width $w$):** $c = w$, giving $O(nwd)$ time.

**Speedup over full attention:**
$$\frac{n^2}{nw} = \frac{n}{w}$$

For $n = 4096$, $w = 256$: 16× speedup.

---

## 📐 Linear Attention

### Standard Attention (Quadratic)

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$$

### Linear Attention (Linear)

**Key Insight:** Replace softmax with kernel function:
$$\text{Attention}(Q, K, V) = \frac{\phi(Q)\phi(K)^T V}{\phi(Q)\phi(K)^T \mathbf{1}}$$

**Rearranging (associativity):**
$$= \frac{\phi(Q)(\phi(K)^T V)}{\phi(Q)(\phi(K)^T \mathbf{1})}$$

**Complexity:** $O(nd^2)$ instead of $O(n^2d)$.

### Theorem 6 (Kernel Approximation)

For feature map $\phi: \mathbb{R}^d \to \mathbb{R}^D$, the approximation error is:

$$\mathbb{E}[\|K_{exact} - \phi(Q)\phi(K)^T\|_F^2] \leq O\left(\frac{d}{D}\right)$$

**Common choices for $\phi$:**
- Random Fourier features
- ELU + 1
- Softmax approximation

---

## 📊 Activation Sparsity

### ReLU Sparsity

For hidden activations after ReLU:
$$h = \max(0, Wx + b)$$

**Theorem 7:** For $Wx + b \sim \mathcal{N}(\mu, \sigma^2)$:

$$\mathbb{P}[h = 0] = \Phi\left(-\frac{\mu}{\sigma}\right)$$

Where $\Phi$ is the standard normal CDF.

**For zero-centered pre-activations:** 50% sparsity.

### Exploiting Activation Sparsity

**Theorem 8 (Activation-Sparse Speedup):**

If $\mathbb{E}[\|h\|_0] = \alpha \cdot d$ for $\alpha < 1$:

Theoretical speedup in next layer: $1/\alpha$

---

## 📐 Sparse Training

### Theorem 9 (Sparse Training Convergence)

For training with sparsity constraint $\|W\|_0 \leq k$:

$$\mathcal{L}(W_t) - \mathcal{L}(W^*) \leq O\left(\frac{\|W_0 - W^*\|_F^2}{t} + \sigma_k(W^*)\right)$$

Where $\sigma_k(W^*)$ is the error from approximating optimal weights with $k$ non-zeros.

### Gradient Flow Through Sparse Masks

**Problem:** Gradient of sparsity constraint is zero almost everywhere.

**Solution (STE):**
$$\frac{\partial \mathcal{L}}{\partial W} = \frac{\partial \mathcal{L}}{\partial (M \odot W)} \odot M$$

**Theorem 10:** STE provides unbiased gradient estimate when:
$$\mathbb{E}[M | W] = \text{smooth function of } W$$

---

## 📖 Summary of Key Formulas

| Concept | Formula |
|---------|---------|
| Sparsity ratio | $1 - \|W\|_0 / mn$ |
| Sparse speedup | $mn / nnz$ |
| 2:4 error | $\approx \sigma^2$ per group |
| MoE compute | $P_{router} + k \cdot P_{expert}$ |
| Load balance loss | $\alpha N \sum_i f_i P_i$ |
| Sliding window complexity | $O(nwd)$ |
| Linear attention | $O(nd^2)$ |
| ReLU sparsity | $\Phi(-\mu/\sigma)$ |

---

## ➡️ Next Module

Continue to [Module 08: PEFT](../08_peft/) for mathematical foundations of parameter-efficient fine-tuning.

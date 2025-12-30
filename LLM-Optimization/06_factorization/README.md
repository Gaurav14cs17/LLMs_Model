# Module 06: Matrix Factorization

## 🎯 Overview

Matrix factorization decomposes large weight matrices into products of smaller matrices, enabling significant compression. This module provides rigorous mathematical foundations and proofs.

---

## 📐 Mathematical Foundations

### 1. Singular Value Decomposition (SVD)

**Theorem 1 (SVD Existence):**

Every matrix $W \in \mathbb{R}^{m \times n}$ can be decomposed as:

$$W = U \Sigma V^T$$

Where:
- $U \in \mathbb{R}^{m \times m}$ is orthogonal ($U^TU = I$)
- $\Sigma \in \mathbb{R}^{m \times n}$ is diagonal with singular values $\sigma_1 \geq \sigma_2 \geq \ldots \geq \sigma_r > 0$
- $V \in \mathbb{R}^{n \times n}$ is orthogonal ($V^TV = I$)
- $r = \text{rank}(W)$

**Proof (sketch):**

1. Consider $W^TW$ (symmetric positive semi-definite)
2. Eigendecomposition: $W^TW = V\Lambda V^T$ with eigenvalues $\lambda_i = \sigma_i^2$
3. Define $U_i = Wv_i / \sigma_i$
4. Verify $U^TU = I$ and $W = U\Sigma V^T$

---

### 2. Low-Rank Approximation

**Theorem 2 (Eckart-Young-Mirsky):**

The best rank-$k$ approximation to $W$ in Frobenius norm is:

$$W_k = \sum_{i=1}^{k} \sigma_i u_i v_i^T = U_k \Sigma_k V_k^T$$

With error:
$$\|W - W_k\|_F = \sqrt{\sum_{i=k+1}^{r} \sigma_i^2}$$

**Proof:**

Let $\tilde{W}$ be any rank-$k$ matrix. Then:

$$\|W - \tilde{W}\|_F^2 = \|U\Sigma V^T - \tilde{W}\|_F^2$$

By orthogonal invariance:
$$= \|\Sigma - U^T\tilde{W}V\|_F^2$$

Let $B = U^T\tilde{W}V$. Since $\text{rank}(B) = \text{rank}(\tilde{W}) \leq k$:

$$\|W - \tilde{W}\|_F^2 = \sum_{i=1}^{r} (\sigma_i - b_{ii})^2 + \sum_{i \neq j} b_{ij}^2$$

Minimizing: $b_{ii} = \sigma_i$ for $i \leq k$, zero otherwise.

$$\min_{\tilde{W}} \|W - \tilde{W}\|_F^2 = \sum_{i=k+1}^{r} \sigma_i^2$$

**Corollary (Spectral Norm):**
$$\|W - W_k\|_2 = \sigma_{k+1}$$

---

### 3. Energy Conservation

**Definition:** The energy captured by rank-$k$ approximation:

$$E_k = \frac{\sum_{i=1}^{k} \sigma_i^2}{\sum_{i=1}^{r} \sigma_i^2} = \frac{\|W_k\|_F^2}{\|W\|_F^2}$$

**Choosing Rank:** Select $k$ such that $E_k \geq \tau$ (e.g., $\tau = 0.99$).

**Theorem 3 (Energy Decay for Neural Networks):**

For trained neural network weight matrices, singular values typically decay as:

$$\sigma_i \approx c \cdot i^{-\alpha}, \quad \alpha \in [0.5, 2]$$

**Implication:** Many weight matrices are approximately low-rank.

---

## 📊 Factorized Linear Layers

### Standard vs. Factorized

**Standard:**
$$y = Wx, \quad W \in \mathbb{R}^{m \times n}$$
Parameters: $mn$

**Factorized:**
$$y = (UV)x = U(Vx), \quad U \in \mathbb{R}^{m \times r}, V \in \mathbb{R}^{r \times n}$$
Parameters: $r(m + n)$

**Compression Ratio:**
$$\rho = \frac{mn}{r(m+n)} = \frac{mn}{r(m+n)}$$

**When is this beneficial?**
$$r < \frac{mn}{m+n} = \frac{1}{1/m + 1/n}$$

For $m = n$: $r < n/2$

---

### 4. Error Analysis

**Theorem 4 (Output Error Bound):**

For factorized layer with approximation error $E = W - UV$:

$$\|Wx - UVx\|_2 = \|Ex\|_2 \leq \|E\|_2 \|x\|_2 = \sigma_{k+1} \|x\|_2$$

**For entire network with $L$ factorized layers:**

$$\|f(x) - \hat{f}(x)\|_2 \leq \sum_{l=1}^{L} \sigma_{k_l+1}^{(l)} \prod_{j>l} \|W_j\|_2 \cdot \text{Lip}(\phi)^{L-l} \|x\|_2$$

Where $\text{Lip}(\phi)$ is the Lipschitz constant of the activation function.

---

## 📐 Tucker Decomposition

### Definition

For tensor $\mathcal{W} \in \mathbb{R}^{I_1 \times I_2 \times \ldots \times I_N}$:

$$\mathcal{W} \approx \mathcal{G} \times_1 A^{(1)} \times_2 A^{(2)} \ldots \times_N A^{(N)}$$

Where:
- $\mathcal{G} \in \mathbb{R}^{R_1 \times R_2 \times \ldots \times R_N}$ is the core tensor
- $A^{(n)} \in \mathbb{R}^{I_n \times R_n}$ are factor matrices
- $\times_n$ is mode-$n$ product

**Mode-n Product:**
$$(\mathcal{G} \times_n A)_{i_1...i_{n-1}ji_{n+1}...i_N} = \sum_{k} \mathcal{G}_{i_1...i_{n-1}ki_{n+1}...i_N} A_{jk}$$

### Parameter Count

**Original:** $\prod_{n=1}^{N} I_n$

**Tucker:** $\prod_{n=1}^{N} R_n + \sum_{n=1}^{N} I_n R_n$

**For 4D Convolution** ($C_{out} \times C_{in} \times H \times W$):

Original: $C_{out} \cdot C_{in} \cdot H \cdot W$

Tucker: $R_1 R_2 R_3 R_4 + C_{out}R_1 + C_{in}R_2 + HR_3 + WR_4$

---

### Theorem 5 (Tucker Approximation Error)

The Tucker decomposition minimizes:
$$\|\mathcal{W} - \mathcal{G} \times_1 A^{(1)} \times_2 A^{(2)} \ldots \times_N A^{(N)}\|_F$$

**Solution via Higher-Order SVD (HOSVD):**

1. Compute mode-$n$ unfolding $W_{(n)}$
2. Compute SVD: $W_{(n)} = U^{(n)} \Sigma^{(n)} V^{(n)T}$
3. Set $A^{(n)} = U^{(n)}_{:,1:R_n}$
4. Compute core: $\mathcal{G} = \mathcal{W} \times_1 A^{(1)T} \times_2 A^{(2)T} \ldots$

**Error bound:**
$$\|\mathcal{W} - \text{Tucker}(\mathcal{W})\|_F^2 \leq \sum_{n=1}^{N} \sum_{i > R_n} (\sigma_i^{(n)})^2$$

---

## 📊 CP Decomposition

### Definition (CANDECOMP/PARAFAC)

$$\mathcal{W} \approx \sum_{r=1}^{R} \lambda_r \cdot a_r^{(1)} \circ a_r^{(2)} \circ \ldots \circ a_r^{(N)}$$

Where $\circ$ denotes outer product.

**Element-wise:**
$$\mathcal{W}_{i_1 i_2 \ldots i_N} \approx \sum_{r=1}^{R} \lambda_r \prod_{n=1}^{N} a_{i_n r}^{(n)}$$

### Parameters

Original: $\prod_n I_n$

CP with rank $R$: $R(1 + \sum_n I_n)$

**Extreme compression for high-order tensors.**

### Theorem 6 (CP Uniqueness)

The CP decomposition is essentially unique (up to permutation and scaling) when:

$$R \leq \frac{\prod_{n=1}^{N} I_n}{I_{\max}}$$

Where $I_{\max} = \max_n I_n$.

---

## 📐 LoRA as Low-Rank Factorization

### Mathematical Connection

LoRA update:
$$W' = W + BA$$

Where $B \in \mathbb{R}^{m \times r}$, $A \in \mathbb{R}^{r \times n}$.

**This is low-rank matrix perturbation:**
$$\text{rank}(BA) \leq r$$

### Theorem 7 (LoRA Expressiveness)

Any weight update $\Delta W$ can be approximated by LoRA with rank $r$ if:

$$\|\Delta W - BA\|_F \leq \sigma_{r+1}(\Delta W)$$

**Proof:** By Eckart-Young theorem.

### Optimal LoRA Initialization

**Theorem 8:** The optimal $B, A$ minimizing $\|W' - W_{new}\|_F$ are:

$$A = \sqrt{\Sigma_r} V_r^T, \quad B = U_r \sqrt{\Sigma_r}$$

Where $\Delta W = W_{new} - W = U_r \Sigma_r V_r^T$ is the rank-$r$ SVD.

---

## 📊 Spectral Analysis of Neural Networks

### Theorem 9 (Singular Value Distribution)

For randomly initialized neural networks:

$$\sigma_i \sim \sqrt{n} \cdot f(\lambda_i)$$

Where $\lambda_i$ follows the Marchenko-Pastur distribution.

### After Training

Training typically leads to:
1. **Larger leading singular values** (task-relevant directions)
2. **Smaller trailing singular values** (noise)
3. **Increased effective rank** initially, then decrease

**Implication:** Post-training factorization is often more effective than pre-training.

---

## 📐 Practical Considerations

### Choosing Rank

**Energy-based:**
$$r^* = \min\{k : E_k \geq \tau\}$$

**Error-based:**
$$r^* = \min\{k : \|W - W_k\|_F / \|W\|_F \leq \epsilon\}$$

**Compute-based:**
$$r^* = \arg\min_k \{r(m+n) : r(m+n) < mn / \rho_{target}\}$$

### Theorem 10 (Factorization with Fine-tuning)

Let $\mathcal{L}^*$ be optimal loss and $\mathcal{L}_k$ loss after rank-$k$ factorization:

After fine-tuning for $T$ steps with learning rate $\eta$:

$$\mathcal{L}_{k,T} \leq \mathcal{L}_k - T\eta\|\nabla\mathcal{L}\|^2 / 2$$

**Implication:** Fine-tuning can recover most accuracy lost to factorization.

---

## 📖 Summary of Key Formulas

| Concept | Formula |
|---------|---------|
| SVD | $W = U\Sigma V^T$ |
| Rank-k error | $\|W - W_k\|_F = \sqrt{\sum_{i>k}\sigma_i^2}$ |
| Energy ratio | $E_k = \sum_{i \leq k}\sigma_i^2 / \sum_i\sigma_i^2$ |
| Compression ratio | $\rho = mn / (r(m+n))$ |
| Tucker core | $\mathcal{G} = \mathcal{W} \times_1 A^{(1)T} \times_2 A^{(2)T}...$ |
| CP form | $\mathcal{W} \approx \sum_r \lambda_r \prod_n a_r^{(n)}$ |

---

## ➡️ Next Module

Continue to [Module 07: Sparsity](../07_sparsity/) for mathematical foundations of sparse computation.

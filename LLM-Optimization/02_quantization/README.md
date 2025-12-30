# Module 02: Quantization

<p align="center">
  <img src="../assets/quantization-overview.svg" alt="Quantization Overview" width="800"/>
</p>

## 🎯 Overview

Quantization reduces the numerical precision of model weights and activations. This module provides the mathematical foundations, proofs, and theoretical analysis of quantization techniques.

---

## 📐 Mathematical Foundations

### 1. Uniform Quantization

**Definition:** A uniform quantizer $Q: \mathbb{R} \rightarrow \mathcal{Q}$ maps continuous values to a discrete set $\mathcal{Q}$ with $2^b$ levels.

#### Affine (Asymmetric) Quantization

$$Q(x) = \text{round}\left(\frac{x - z}{s}\right), \quad \hat{x} = s \cdot Q(x) + z$$

Where:
- $s$ = scale (step size)
- $z$ = zero-point (offset)
- $b$ = bit-width

**Computing scale and zero-point:**

$$s = \frac{x_{max} - x_{min}}{2^b - 1}$$

$$z = x_{min} - s \cdot q_{min}$$

Where $q_{min} = 0$ for unsigned or $q_{min} = -2^{b-1}$ for signed.

#### Symmetric Quantization

For symmetric quantization around zero:

$$Q(x) = \text{round}\left(\frac{x}{s}\right), \quad s = \frac{\max(|x_{max}|, |x_{min}|)}{2^{b-1} - 1}$$

---

### 2. Quantization Error Analysis

**Theorem 1 (Quantization Noise):**

For uniform quantization with step size $\Delta$, the quantization error $e = x - Q(x)$ has:

$$\mathbb{E}[e] = 0, \quad \mathbb{E}[e^2] = \frac{\Delta^2}{12}$$

**Proof:**

Assuming the input is uniformly distributed within a quantization bin:
$$p(e) = \frac{1}{\Delta}, \quad e \in \left[-\frac{\Delta}{2}, \frac{\Delta}{2}\right]$$

Mean:
$$\mathbb{E}[e] = \int_{-\Delta/2}^{\Delta/2} e \cdot \frac{1}{\Delta} de = 0$$

Variance:
$$\mathbb{E}[e^2] = \int_{-\Delta/2}^{\Delta/2} e^2 \cdot \frac{1}{\Delta} de = \frac{1}{\Delta} \cdot \frac{e^3}{3}\Big|_{-\Delta/2}^{\Delta/2} = \frac{\Delta^2}{12}$$

**Corollary:** Signal-to-Quantization-Noise Ratio (SQNR):

$$\text{SQNR} = \frac{\sigma_x^2}{\sigma_e^2} = \frac{12\sigma_x^2}{\Delta^2}$$

In dB for full-range signal:
$$\text{SQNR}_{dB} \approx 6.02b + 1.76 \text{ dB}$$

**Each additional bit provides ~6 dB improvement.**

---

### 3. Optimal Quantization Theory

#### Lloyd-Max Quantizer

**Problem:** Find quantization levels $\{q_i\}$ and decision boundaries $\{d_i\}$ that minimize mean squared error.

**Optimality Conditions:**

1. **Nearest Neighbor:** Each input is quantized to the nearest level:
$$d_i = \frac{q_i + q_{i+1}}{2}$$

2. **Centroid:** Each level is the centroid of its region:
$$q_i = \frac{\int_{d_{i-1}}^{d_i} x \cdot p(x) dx}{\int_{d_{i-1}}^{d_i} p(x) dx}$$

**For Gaussian Distribution:**

The optimal quantization levels are not uniformly spaced. For $\mathcal{N}(0, 1)$:

| Bits | Optimal Levels | MSE |
|------|---------------|-----|
| 1 | ±0.7979 | 0.3634 |
| 2 | ±0.4528, ±1.5104 | 0.1175 |
| 3 | ±0.2451, ±0.7560, ±1.3439, ±2.1520 | 0.0344 |

---

### 4. NormalFloat Quantization (NF4)

**Motivation:** Neural network weights follow approximately Gaussian distributions.

**NF4 Construction:**

1. Assume weights $W \sim \mathcal{N}(0, \sigma^2)$
2. Normalize: $\tilde{W} = W/\sigma$
3. Find 16 quantization levels that minimize MSE for $\mathcal{N}(0, 1)$

**Optimal NF4 Levels (for $\mathcal{N}(0,1)$):**
$$\mathcal{Q}_{NF4} = \{-1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0, $$
$$0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0\}$$

**Theorem 2:** NF4 achieves lower MSE than uniform INT4 for Gaussian-distributed weights.

**Proof:**

For weights $W \sim \mathcal{N}(0, \sigma^2)$:

Uniform INT4 MSE:
$$\text{MSE}_{uniform} = \frac{\Delta^2}{12} \approx \frac{(2 \cdot 3\sigma / 15)^2}{12} = 0.0333\sigma^2$$

NF4 MSE (Lloyd-Max optimal):
$$\text{MSE}_{NF4} \approx 0.0220\sigma^2$$

**Improvement:** $\frac{0.0333 - 0.0220}{0.0333} \approx 34\%$ lower MSE.

---

## 📊 Post-Training Quantization (PTQ)

### 1. Min-Max Quantization

**Algorithm:**
1. Collect activation statistics over calibration data
2. Compute range: $[x_{min}, x_{max}]$
3. Compute scale and zero-point
4. Quantize weights/activations

**Error Analysis:**

If true range is $[\mu - k\sigma, \mu + k\sigma]$:
$$\text{Clipping Error} = \mathbb{E}[(x - \text{clip}(x))^2] = 2\int_{k\sigma}^{\infty} (x - k\sigma)^2 p(x) dx$$

For Gaussian, optimal $k \approx 2.83$ for 4-bit.

### 2. Percentile Quantization

Use percentiles instead of min/max to reduce outlier sensitivity:

$$x_{min} = \text{Percentile}(X, p), \quad x_{max} = \text{Percentile}(X, 100-p)$$

Typical $p = 0.1\%$ to $1\%$.

### 3. MSE-Optimal Quantization

**Objective:** Find $s, z$ that minimize:

$$\min_{s, z} \|W - Q(W; s, z)\|_F^2$$

**Solution:** Grid search or analytical approximation.

---

## 📈 GPTQ: Second-Order Quantization

### Mathematical Foundation

**Objective:** Quantize weights to minimize output error:

$$\min_{\hat{W}} \|WX - \hat{W}X\|_F^2$$

Where $X$ is the input activation matrix.

### Optimal Brain Quantization (OBQ)

**Theorem 3 (Weight Update Formula):**

When quantizing weight $w_q$ in column $q$, the optimal update to remaining weights is:

$$\delta_F = -\frac{w_q - \text{quant}(w_q)}{[H^{-1}]_{qq}} \cdot (H^{-1})_{:,q}$$

Where $H = 2X^TX$ is the Hessian of the squared error.

**Proof:**

The loss increase from quantizing $w_q$ is:
$$\Delta\mathcal{L} = \frac{(w_q - \text{quant}(w_q))^2}{2[H^{-1}]_{qq}}$$

Using Lagrange multipliers to minimize loss subject to $w_q$ being quantized:
$$\nabla_{w_F} \mathcal{L} + \lambda \nabla_{w_F}(w_q - c) = 0$$

Solving gives the update formula above.

### GPTQ Algorithm

GPTQ processes columns in order, updating the inverse Hessian efficiently:

**Inverse Update (Gaussian Elimination):**
$$[H^{-1}]_{F,F} \leftarrow [H^{-1}]_{F,F} - \frac{[H^{-1}]_{F,q}[H^{-1}]_{q,F}}{[H^{-1}]_{qq}}$$

**Complexity:** $O(d_{col} \cdot d_{row}^2)$ per layer.

---

## 📐 AWQ: Activation-Aware Quantization

### Mathematical Foundation

**Key Insight:** Not all weights are equally important. Weights corresponding to large activations matter more.

**Theorem 4 (Saliency-Weighted Quantization):**

The quantization error weighted by activation magnitude is:

$$\mathcal{L} = \sum_i s_i (w_i - \hat{w}_i)^2$$

Where $s_i = \mathbb{E}[|x_i|]$ is the average activation magnitude.

### Optimal Scaling

**Objective:** Find per-channel scales $\alpha$ that minimize weighted error:

$$\min_\alpha \mathbb{E}_x\left[\left\|Q\left(\frac{W}{\alpha}\right) \cdot \alpha \cdot x - W \cdot x\right\|^2\right]$$

**Solution:** Grid search over $\alpha \in [0, 1]$ with:
$$\alpha_j = s_j^\beta, \quad \beta \in [0, 1]$$

Where $s_j$ is the activation scale for channel $j$.

### Proof of Effectiveness

**Lemma:** For salient channels with high activation variance, scaling reduces quantization error.

Let $w$ be a weight with quantization error $e = w - Q(w)$.

After scaling by $\alpha < 1$:
$$e' = \alpha \cdot Q(w/\alpha) - w$$

For large $|w|$, the relative error decreases:
$$\frac{|e'|}{|w|} < \frac{|e|}{|w|}$$

---

## 📊 Group-wise Quantization

### Mathematical Formulation

Instead of per-tensor scales, use per-group scales:

$$Q(W) = \begin{bmatrix} s_1 \cdot Q(W_1/s_1) \\ s_2 \cdot Q(W_2/s_2) \\ \vdots \\ s_g \cdot Q(W_g/s_g) \end{bmatrix}$$

Where groups $W_1, \ldots, W_g$ partition the weight matrix.

### Error Analysis

**Theorem 5:** Group-wise quantization with group size $g$ reduces MSE by factor of approximately:

$$\frac{\text{MSE}_{per-tensor}}{\text{MSE}_{group}} \approx \frac{\text{Var}(W)}{\text{Var}(W|group)} \approx \sqrt{g}$$

**Proof sketch:**

Within-group variance is lower than full-tensor variance. For uniformly distributed group means:
$$\text{Var}(W|group) \approx \frac{\text{Var}(W)}{g^{1/2}}$$

### Storage Overhead

For $n$ weights with group size $g$:
- Scales: $n/g$ values (16 or 32 bits each)
- Overhead: $\frac{b_{scale}}{g \cdot b_{weight}}$ relative

**Example:** g=128, 4-bit weights, FP16 scales:
$$\text{Overhead} = \frac{16}{128 \times 4} = 3.1\%$$

---

## 📉 Quantization-Aware Training (QAT)

### Straight-Through Estimator (STE)

**Problem:** Quantization function has zero gradient almost everywhere.

**Solution:** Use STE for backpropagation:

$$\frac{\partial \mathcal{L}}{\partial w} = \frac{\partial \mathcal{L}}{\partial \hat{w}} \cdot \underbrace{\frac{\partial \hat{w}}{\partial w}}_{\approx 1}$$

### Mathematical Justification

**Theorem 6 (Bengio et al., 2013):** STE provides an unbiased gradient estimate when:
1. Quantization error is small relative to weight updates
2. Loss surface is relatively smooth

### QAT Loss Function

$$\mathcal{L}_{QAT} = \mathcal{L}_{task}(Q(W)) + \lambda \cdot R(W)$$

Where $R(W)$ is a regularizer encouraging quantization-friendly weights:
$$R(W) = \|W - Q(W)\|_F^2$$

---

## 📊 Theoretical Bounds

### Theorem 7 (Quantization Error Propagation)

For an $L$-layer network with per-layer quantization error $\epsilon_l$:

$$\|f(x) - \hat{f}(x)\| \leq \sum_{l=1}^{L} \epsilon_l \cdot \prod_{k=l+1}^{L} \|W_k\| \cdot \text{Lip}(\sigma)^{L-l}$$

Where $\text{Lip}(\sigma)$ is the Lipschitz constant of activation $\sigma$.

### Corollary: Depth Sensitivity

Deeper networks are more sensitive to quantization:
$$\text{Error} = O(\epsilon \cdot L \cdot \|W\|^L)$$

**Implication:** Early layers may need higher precision.

---

## 📖 Summary of Key Formulas

| Concept | Formula |
|---------|---------|
| Uniform quantization | $Q(x) = s \cdot \text{round}(x/s - z) + z$ |
| Quantization MSE | $\mathbb{E}[e^2] = \Delta^2/12$ |
| SQNR | $\approx 6.02b + 1.76$ dB |
| GPTQ weight update | $\delta_F = -\frac{w_q - Q(w_q)}{[H^{-1}]_{qq}} (H^{-1})_{:,q}$ |
| AWQ objective | $\min_\alpha \|Q(W/\alpha)\alpha x - Wx\|^2$ |
| Group overhead | $b_{scale}/(g \cdot b_{weight})$ |

---

## ➡️ Next Module

Continue to [Module 03: Pruning](../03_pruning/) for mathematical foundations of weight pruning.

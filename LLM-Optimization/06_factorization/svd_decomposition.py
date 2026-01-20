"""
Module 06: Matrix Factorization
================================

This script demonstrates matrix factorization techniques:
- SVD (Singular Value Decomposition)
- Low-rank approximation
- Tucker decomposition for tensors
- Practical application to neural network layers
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional
import math


def svd_decompose(matrix: torch.Tensor, rank: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform SVD and return low-rank factors.
    
    W ≈ U_r @ V_r  where U_r = U[:,:r] @ diag(S[:r])^0.5
                          V_r = diag(S[:r])^0.5 @ Vt[:r,:]
    """
    U, S, Vt = torch.linalg.svd(matrix, full_matrices=False)
    
    # Keep top-r components
    U_r = U[:, :rank]
    S_r = S[:rank]
    Vt_r = Vt[:rank, :]
    
    # Distribute singular values to both factors
    S_sqrt = torch.sqrt(S_r)
    
    # U_r: [m, r], V_r: [r, n]
    factor_U = U_r * S_sqrt.unsqueeze(0)
    factor_V = Vt_r * S_sqrt.unsqueeze(1)
    
    return factor_U, factor_V


def compute_reconstruction_error(original: torch.Tensor, 
                                  reconstructed: torch.Tensor) -> dict:
    """Compute various error metrics."""
    
    diff = original - reconstructed
    
    frobenius_error = torch.norm(diff, p='fro')
    frobenius_original = torch.norm(original, p='fro')
    relative_error = frobenius_error / frobenius_original
    
    mse = F.mse_loss(reconstructed, original)
    
    return {
        'frobenius_error': frobenius_error.item(),
        'relative_error': relative_error.item(),
        'mse': mse.item()
    }


def analyze_singular_values(matrix: torch.Tensor) -> dict:
    """Analyze singular value distribution."""
    
    _, S, _ = torch.linalg.svd(matrix, full_matrices=False)
    
    total_energy = (S ** 2).sum()
    cumulative_energy = torch.cumsum(S ** 2, dim=0) / total_energy
    
    # Find rank for different energy thresholds
    thresholds = [0.90, 0.95, 0.99, 0.999]
    ranks_for_threshold = {}
    
    for threshold in thresholds:
        rank = (cumulative_energy >= threshold).float().argmax().item() + 1
        ranks_for_threshold[threshold] = rank
    
    return {
        'singular_values': S,
        'total_energy': total_energy.item(),
        'cumulative_energy': cumulative_energy,
        'ranks_for_threshold': ranks_for_threshold,
        'max_rank': len(S)
    }


class FactorizedLinear(nn.Module):
    """
    Linear layer factorized as W = U @ V.
    
    Reduces parameters from m*n to m*r + r*n.
    """
    
    def __init__(self, in_features: int, out_features: int, rank: int,
                 bias: bool = True):
        super().__init__()
        
        self.U = nn.Linear(in_features, rank, bias=False)
        self.V = nn.Linear(rank, out_features, bias=bias)
        
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int) -> 'FactorizedLinear':
        """Create from existing Linear layer using SVD."""
        
        factorized = cls(
            linear.in_features, 
            linear.out_features, 
            rank,
            bias=linear.bias is not None
        )
        
        # SVD decomposition
        U, V = svd_decompose(linear.weight.data, rank)
        
        factorized.U.weight.data = U.T  # Linear expects [out, in]
        factorized.V.weight.data = V
        
        if linear.bias is not None:
            factorized.V.bias.data = linear.bias.data.clone()
        
        return factorized
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.V(self.U(x))
    
    def get_full_weight(self) -> torch.Tensor:
        """Reconstruct the full weight matrix."""
        return self.V.weight @ self.U.weight


def demonstrate_svd_basics():
    """Demonstrate basic SVD decomposition."""
    
    print("\n" + "=" * 60)
    print("SVD DECOMPOSITION BASICS")
    print("=" * 60)
    
    # Create a random weight matrix
    m, n = 512, 1024
    W = torch.randn(m, n)
    
    print(f"\nOriginal matrix: {m} × {n} = {m * n:,} elements")
    
    # Analyze singular values
    analysis = analyze_singular_values(W)
    
    print(f"\nSingular value analysis:")
    print(f"  Max singular value: {analysis['singular_values'][0]:.2f}")
    print(f"  Min singular value: {analysis['singular_values'][-1]:.4f}")
    print(f"  Ratio (condition): {analysis['singular_values'][0] / analysis['singular_values'][-1]:.1f}")
    
    print(f"\nRank needed for energy threshold:")
    for threshold, rank in analysis['ranks_for_threshold'].items():
        print(f"  {threshold:.1%} energy: rank {rank} ({rank}/{analysis['max_rank']})")


def demonstrate_low_rank_approximation():
    """Demonstrate low-rank matrix approximation."""
    
    print("\n" + "=" * 60)
    print("LOW-RANK APPROXIMATION")
    print("=" * 60)
    
    # Create matrix with known low-rank structure + noise
    m, n = 256, 512
    true_rank = 32
    
    # Low-rank + noise
    A = torch.randn(m, true_rank)
    B = torch.randn(true_rank, n)
    noise = torch.randn(m, n) * 0.1
    W = A @ B + noise
    
    print(f"\nMatrix: {m} × {n}")
    print(f"True rank: {true_rank} (with noise)")
    print(f"Original parameters: {m * n:,}")
    
    # Test different approximation ranks
    ranks = [8, 16, 32, 64, 128]
    
    print(f"\n{'Rank':>6} {'Parameters':>12} {'Compression':>12} {'Rel. Error':>12}")
    print("-" * 50)
    
    for rank in ranks:
        U, V = svd_decompose(W, rank)
        W_approx = U @ V
        
        params = m * rank + rank * n
        compression = (m * n) / params
        
        error = compute_reconstruction_error(W, W_approx)
        
        print(f"{rank:>6} {params:>12,} {compression:>12.1f}x "
              f"{error['relative_error']:>12.4f}")


def demonstrate_layer_factorization():
    """Demonstrate factorizing a neural network layer."""
    
    print("\n" + "=" * 60)
    print("NEURAL NETWORK LAYER FACTORIZATION")
    print("=" * 60)
    
    # Create a linear layer (simulating a transformer FFN)
    in_features = 768
    out_features = 3072
    
    original_layer = nn.Linear(in_features, out_features)
    original_params = sum(p.numel() for p in original_layer.parameters())
    
    print(f"\nOriginal Linear: {in_features} → {out_features}")
    print(f"Parameters: {original_params:,}")
    
    # Factorize at different ranks
    ranks = [64, 128, 256, 384]
    
    print(f"\n{'Rank':>6} {'Parameters':>12} {'Compression':>12} {'Output MSE':>12}")
    print("-" * 55)
    
    # Test input
    x = torch.randn(4, 32, in_features)
    
    with torch.no_grad():
        original_output = original_layer(x)
    
    for rank in ranks:
        factorized = FactorizedLinear.from_linear(original_layer, rank)
        factorized_params = sum(p.numel() for p in factorized.parameters())
        compression = original_params / factorized_params
        
        with torch.no_grad():
            factorized_output = factorized(x)
        
        mse = F.mse_loss(factorized_output, original_output).item()
        
        print(f"{rank:>6} {factorized_params:>12,} {compression:>12.1f}x "
              f"{mse:>12.6f}")


def demonstrate_tucker_decomposition():
    """Demonstrate Tucker decomposition concept for tensors."""
    
    print("\n" + "=" * 60)
    print("TUCKER DECOMPOSITION (CONCEPTUAL)")
    print("=" * 60)
    
    print("""
    Tucker decomposition for a 4D convolution filter:
    
    Original: W ∈ ℝ^(C_out × C_in × H × W)
    
    Tucker:   W ≈ G ×₁ A₁ ×₂ A₂ ×₃ A₃ ×₄ A₄
    
    Where:
    - G is the core tensor: ℝ^(R₁ × R₂ × R₃ × R₄)
    - A₁: ℝ^(C_out × R₁)
    - A₂: ℝ^(C_in × R₂)
    - A₃: ℝ^(H × R₃)
    - A₄: ℝ^(W × R₄)
    
    For 3×3 convolutions, typically R₃ = 3, R₄ = 3 (keep spatial).
    """)
    
    # Example calculation
    C_out, C_in, H, W = 256, 256, 3, 3
    original_params = C_out * C_in * H * W
    
    # Tucker ranks
    R1, R2, R3, R4 = 64, 64, 3, 3
    
    tucker_params = (
        R1 * R2 * R3 * R4 +  # Core
        C_out * R1 +          # A1
        C_in * R2 +           # A2
        H * R3 +              # A3
        W * R4                # A4
    )
    
    print(f"\nExample: Conv layer {C_out}×{C_in}×{H}×{W}")
    print(f"Original parameters: {original_params:,}")
    print(f"Tucker parameters (R=[{R1},{R2},{R3},{R4}]): {tucker_params:,}")
    print(f"Compression: {original_params / tucker_params:.1f}x")


def factorize_transformer_ffn():
    """Demonstrate factorizing a transformer FFN."""
    
    print("\n" + "=" * 60)
    print("FACTORIZING TRANSFORMER FFN")
    print("=" * 60)
    
    hidden_size = 768
    intermediate_size = 3072
    
    class StandardFFN(nn.Module):
        def __init__(self):
            super().__init__()
            self.up = nn.Linear(hidden_size, intermediate_size)
            self.down = nn.Linear(intermediate_size, hidden_size)
            self.act = nn.GELU()
        
        def forward(self, x):
            return self.down(self.act(self.up(x)))
    
    class FactorizedFFN(nn.Module):
        def __init__(self, rank: int):
            super().__init__()
            # Factorized up projection
            self.up_u = nn.Linear(hidden_size, rank, bias=False)
            self.up_v = nn.Linear(rank, intermediate_size)
            
            # Factorized down projection
            self.down_u = nn.Linear(intermediate_size, rank, bias=False)
            self.down_v = nn.Linear(rank, hidden_size)
            
            self.act = nn.GELU()
        
        def forward(self, x):
            x = self.up_v(self.up_u(x))
            x = self.act(x)
            x = self.down_v(self.down_u(x))
            return x
    
    standard = StandardFFN()
    standard_params = sum(p.numel() for p in standard.parameters())
    
    print(f"\nStandard FFN: {hidden_size} → {intermediate_size} → {hidden_size}")
    print(f"Parameters: {standard_params:,}")
    
    ranks = [128, 256, 384]
    
    print(f"\n{'Rank':>6} {'Parameters':>12} {'Compression':>12}")
    print("-" * 35)
    
    for rank in ranks:
        factorized = FactorizedFFN(rank)
        factorized_params = sum(p.numel() for p in factorized.parameters())
        compression = standard_params / factorized_params
        
        print(f"{rank:>6} {factorized_params:>12,} {compression:>12.1f}x")


def main():
    """Main demonstration of matrix factorization."""
    
    print("\n" + "=" * 60)
    print("   MODULE 06: MATRIX FACTORIZATION")
    print("=" * 60)
    
    # 1. SVD basics
    demonstrate_svd_basics()
    
    # 2. Low-rank approximation
    demonstrate_low_rank_approximation()
    
    # 3. Layer factorization
    demonstrate_layer_factorization()
    
    # 4. Tucker decomposition
    demonstrate_tucker_decomposition()
    
    # 5. FFN factorization
    factorize_transformer_ffn()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    MATRIX FACTORIZATION TECHNIQUES:
    
    1. SVD DECOMPOSITION:
       - W = U × Σ × V^T
       - Optimal low-rank approximation
       - Keep top-k singular values
    
    2. LOW-RANK APPROXIMATION:
       - W ≈ U × V where U:[m,r], V:[r,n]
       - Parameters: mr + rn << mn
       - Error controlled by rank
    
    3. TUCKER DECOMPOSITION:
       - For multi-dimensional tensors
       - Core tensor + factor matrices
       - Good for convolutions
    
    4. PRACTICAL TIPS:
       - Analyze singular value distribution
       - Many neural networks are approximately low-rank
       - Fine-tune after factorization
       - Combine with LoRA for fine-tuning
    
    5. COMPARISON WITH LoRA:
       - SVD: Compress existing weights
       - LoRA: Low-rank update during fine-tuning
       - Both based on low-rank principle
    """)
    print("=" * 60)


if __name__ == "__main__":
    main()


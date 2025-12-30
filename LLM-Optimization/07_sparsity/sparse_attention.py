"""
Module 07: Sparsity
===================

This script demonstrates sparsity techniques:
- N:M structured sparsity (2:4)
- Mixture of Experts (MoE)
- Sparse attention patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List
import math


def apply_2_4_sparsity(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply 2:4 structured sparsity.
    
    In every group of 4 consecutive values, keep the 2 largest
    and zero out the 2 smallest.
    
    Returns:
        sparse_tensor: Tensor with 2:4 sparsity applied
        mask: Boolean mask of kept values
    """
    original_shape = tensor.shape
    
    # Reshape to groups of 4
    tensor_flat = tensor.view(-1)
    num_groups = tensor_flat.numel() // 4
    tensor_grouped = tensor_flat[:num_groups * 4].view(-1, 4)
    
    # Find top-2 indices in each group
    _, top_indices = torch.topk(tensor_grouped.abs(), k=2, dim=1)
    
    # Create mask
    mask = torch.zeros_like(tensor_grouped, dtype=torch.bool)
    mask.scatter_(1, top_indices, True)
    
    # Apply mask
    sparse_tensor = tensor_grouped * mask.float()
    
    # Reshape back
    sparse_tensor = sparse_tensor.view(-1)
    
    # Handle remainder
    if tensor_flat.numel() % 4 != 0:
        remainder = tensor_flat[num_groups * 4:]
        sparse_tensor = torch.cat([sparse_tensor, remainder])
    
    return sparse_tensor.view(original_shape), mask


class SparseLinear2_4(nn.Module):
    """Linear layer with 2:4 structured sparsity."""
    
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.register_buffer('mask', torch.ones_like(self.weight, dtype=torch.bool))
    
    def apply_sparsity(self):
        """Apply 2:4 sparsity to weights."""
        with torch.no_grad():
            sparse_weight, mask = apply_2_4_sparsity(self.weight.data)
            self.weight.data = sparse_weight
            self.mask = mask.view(self.weight.shape)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # In practice, this would use sparse tensor cores
        # Here we simulate by masking
        sparse_weight = self.weight * self.mask.float()
        return F.linear(x, sparse_weight, self.bias)


class Expert(nn.Module):
    """Single expert (FFN) in MoE."""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.up = nn.Linear(hidden_size, intermediate_size)
        self.down = nn.Linear(intermediate_size, hidden_size)
        self.act = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(self.act(self.up(x)))


class Router(nn.Module):
    """Router/Gate for selecting experts."""
    
    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            router_probs: [batch, seq, num_experts]
            router_logits: [batch, seq, num_experts]
        """
        logits = self.gate(x)
        probs = F.softmax(logits, dim=-1)
        return probs, logits


class MoELayer(nn.Module):
    """
    Mixture of Experts layer.
    
    Routes each token to top-k experts and combines their outputs.
    """
    
    def __init__(self, hidden_size: int, intermediate_size: int,
                 num_experts: int = 8, top_k: int = 2):
        super().__init__()
        
        self.num_experts = num_experts
        self.top_k = top_k
        
        self.router = Router(hidden_size, num_experts)
        self.experts = nn.ModuleList([
            Expert(hidden_size, intermediate_size)
            for _ in range(num_experts)
        ])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        """
        Args:
            x: [batch, seq, hidden]
            
        Returns:
            output: [batch, seq, hidden]
            aux_info: Dictionary with routing statistics
        """
        batch_size, seq_len, hidden_size = x.shape
        
        # Get routing probabilities
        router_probs, router_logits = self.router(x)
        
        # Select top-k experts per token
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        
        # Normalize selected expert weights
        top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs (simplified - in practice would batch tokens per expert)
        output = torch.zeros_like(x)
        
        for i in range(self.top_k):
            expert_indices = top_k_indices[:, :, i]  # [batch, seq]
            expert_weights = top_k_probs[:, :, i:i+1]  # [batch, seq, 1]
            
            for expert_idx in range(self.num_experts):
                mask = (expert_indices == expert_idx)
                if mask.any():
                    expert_input = x[mask]
                    expert_output = self.experts[expert_idx](expert_input)
                    output[mask] += expert_weights[mask].squeeze(-1).unsqueeze(-1) * expert_output
        
        # Load balancing loss (encourage even expert usage)
        expert_usage = router_probs.mean(dim=(0, 1))
        load_balance_loss = self.num_experts * (expert_usage * router_probs.mean(dim=(0, 1))).sum()
        
        aux_info = {
            'load_balance_loss': load_balance_loss,
            'expert_usage': expert_usage,
            'top_expert': top_k_indices[:, :, 0]
        }
        
        return output, aux_info


class SlidingWindowAttention(nn.Module):
    """
    Sparse attention with sliding window pattern.
    
    Each token only attends to a local window of nearby tokens,
    reducing complexity from O(n²) to O(n × window_size).
    """
    
    def __init__(self, hidden_size: int, num_heads: int, window_size: int = 256):
        super().__init__()
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.window_size = window_size
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        
        # Create sliding window attention mask
        # Each position can only attend to positions within window_size
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device)
        for i in range(seq_len):
            start = max(0, i - self.window_size // 2)
            end = min(seq_len, i + self.window_size // 2 + 1)
            mask[i, start:end] = False
        
        # Compute attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = attn.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.o_proj(out)


def demonstrate_2_4_sparsity():
    """Demonstrate 2:4 structured sparsity."""
    
    print("\n" + "=" * 60)
    print("2:4 STRUCTURED SPARSITY")
    print("=" * 60)
    
    # Create sample tensor
    tensor = torch.randn(4, 8)
    print(f"\nOriginal tensor:")
    print(tensor)
    
    sparse_tensor, mask = apply_2_4_sparsity(tensor)
    
    print(f"\nSparse tensor (2:4 pattern):")
    print(sparse_tensor)
    
    print(f"\nMask (1 = kept, 0 = zeroed):")
    print(mask.view(4, 8).int())
    
    sparsity = (sparse_tensor == 0).float().mean()
    print(f"\nActual sparsity: {sparsity:.1%}")
    
    # Demonstrate on linear layer
    print("\n" + "-" * 40)
    print("Applying to Linear Layer")
    print("-" * 40)
    
    layer = SparseLinear2_4(256, 512)
    original_density = (layer.weight != 0).float().mean()
    
    layer.apply_sparsity()
    final_density = (layer.weight != 0).float().mean()
    
    print(f"Before sparsification: {original_density:.1%} non-zero")
    print(f"After sparsification: {final_density:.1%} non-zero")
    
    # Theoretical speedup
    print(f"\nTheoretical speedup on Ampere GPU: ~2x")


def demonstrate_moe():
    """Demonstrate Mixture of Experts."""
    
    print("\n" + "=" * 60)
    print("MIXTURE OF EXPERTS (MoE)")
    print("=" * 60)
    
    hidden_size = 512
    intermediate_size = 2048
    num_experts = 8
    top_k = 2
    
    moe = MoELayer(hidden_size, intermediate_size, num_experts, top_k)
    
    # Count parameters
    total_params = sum(p.numel() for p in moe.parameters())
    single_expert_params = sum(p.numel() for p in moe.experts[0].parameters())
    router_params = sum(p.numel() for p in moe.router.parameters())
    
    print(f"\nMoE Configuration:")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Intermediate size: {intermediate_size}")
    print(f"  Number of experts: {num_experts}")
    print(f"  Top-k selection: {top_k}")
    
    print(f"\nParameter count:")
    print(f"  Per expert: {single_expert_params:,}")
    print(f"  All experts: {single_expert_params * num_experts:,}")
    print(f"  Router: {router_params:,}")
    print(f"  Total: {total_params:,}")
    
    # Compare to dense layer
    dense_params = hidden_size * intermediate_size + intermediate_size * hidden_size
    print(f"\nDense equivalent: {dense_params:,}")
    print(f"MoE total params: {total_params:,} ({total_params / dense_params:.1f}x)")
    print(f"MoE active params: {single_expert_params * top_k + router_params:,}")
    
    # Forward pass
    x = torch.randn(2, 32, hidden_size)
    output, aux_info = moe(x)
    
    print(f"\nExpert usage distribution:")
    usage = aux_info['expert_usage']
    for i, u in enumerate(usage):
        bar = "█" * int(u * 50)
        print(f"  Expert {i}: {bar} {u:.2%}")
    
    print(f"\nLoad balance loss: {aux_info['load_balance_loss']:.4f}")


def demonstrate_sparse_attention():
    """Demonstrate sparse attention patterns."""
    
    print("\n" + "=" * 60)
    print("SPARSE ATTENTION PATTERNS")
    print("=" * 60)
    
    print("""
    Common Sparse Attention Patterns:
    
    1. SLIDING WINDOW (Local)
       - Each token attends to W nearby tokens
       - Complexity: O(n × W) instead of O(n²)
       - Used in: Longformer, BigBird
    
    2. GLOBAL TOKENS
       - Some tokens (e.g., [CLS]) attend to all
       - Others use local attention
       - Used in: Longformer, BigBird
    
    3. DILATED SLIDING WINDOW
       - Multiple windows with different dilation rates
       - Captures both local and long-range patterns
       - Used in: Longformer
    
    4. BLOCK SPARSE
       - Attention computed in blocks
       - Random or learned block patterns
       - Used in: BigBird, Sparse Transformer
    """)
    
    # Demonstrate sliding window
    hidden_size = 256
    num_heads = 4
    window_size = 64
    
    attn = SlidingWindowAttention(hidden_size, num_heads, window_size)
    
    seq_len = 512
    x = torch.randn(1, seq_len, hidden_size)
    
    # Standard full attention FLOPs
    full_attn_flops = seq_len * seq_len * hidden_size
    
    # Sliding window FLOPs
    sparse_attn_flops = seq_len * window_size * hidden_size
    
    print(f"\nComparison (seq_len={seq_len}, window={window_size}):")
    print(f"  Full attention FLOPs: {full_attn_flops:,}")
    print(f"  Sliding window FLOPs: {sparse_attn_flops:,}")
    print(f"  Speedup: {full_attn_flops / sparse_attn_flops:.1f}x")
    
    # Memory comparison
    full_attn_memory = seq_len * seq_len * 4  # bytes
    sparse_attn_memory = seq_len * window_size * 4
    
    print(f"\n  Full attention memory: {full_attn_memory / 1024:.1f} KB")
    print(f"  Sliding window memory: {sparse_attn_memory / 1024:.1f} KB")


def compare_sparsity_methods():
    """Compare different sparsity approaches."""
    
    print("\n" + "=" * 60)
    print("SPARSITY METHODS COMPARISON")
    print("=" * 60)
    
    comparison = """
    ┌──────────────────────┬────────────────┬──────────────┬──────────────┐
    │ Method               │ Speedup        │ Memory       │ Hardware     │
    ├──────────────────────┼────────────────┼──────────────┼──────────────┤
    │ Unstructured (90%)   │ 1.5-3x*        │ 10x smaller  │ Specialized  │
    │ 2:4 Structured       │ 2x             │ 2x smaller   │ Ampere GPU   │
    │ MoE (8 experts, k=2) │ Same compute   │ 4x capacity  │ Standard     │
    │ Sparse Attention     │ n/window       │ n/window     │ Standard     │
    └──────────────────────┴────────────────┴──────────────┴──────────────┘
    
    * Unstructured sparsity speedup requires special hardware/libraries
    
    RECOMMENDATIONS:
    
    1. For INFERENCE on Ampere+:
       → Use 2:4 sparsity for guaranteed 2x speedup
    
    2. For MORE CAPACITY (same compute):
       → Use MoE to scale parameters without FLOPs
    
    3. For LONG SEQUENCES:
       → Use sparse attention (sliding window, etc.)
    
    4. For MAXIMUM COMPRESSION:
       → Combine pruning + quantization + 2:4 sparsity
    """
    print(comparison)


def main():
    """Main demonstration of sparsity techniques."""
    
    print("\n" + "=" * 60)
    print("   MODULE 07: SPARSITY")
    print("=" * 60)
    
    # 1. 2:4 sparsity
    demonstrate_2_4_sparsity()
    
    # 2. MoE
    demonstrate_moe()
    
    # 3. Sparse attention
    demonstrate_sparse_attention()
    
    # 4. Comparison
    compare_sparsity_methods()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    SPARSITY TECHNIQUES:
    
    1. 2:4 STRUCTURED SPARSITY:
       - 50% sparsity with hardware acceleration
       - ~2x speedup on Ampere GPUs
       - Fine-tune to recover accuracy
    
    2. MIXTURE OF EXPERTS (MoE):
       - Scale parameters without compute
       - 8x experts, 2x active = 8x capacity, same FLOPs
       - Requires load balancing
       - Examples: Mixtral, Switch Transformer
    
    3. SPARSE ATTENTION:
       - Essential for long sequences
       - O(n × w) instead of O(n²)
       - Examples: Longformer, BigBird
    
    4. MODERN USAGE:
       - Mixtral 8x7B: MoE with 8 experts
       - Mistral: Sliding window attention
       - LLaMA 2 Long: Extended context with sparse
    """)
    print("=" * 60)


if __name__ == "__main__":
    main()


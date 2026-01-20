"""
Module 09: Efficient Architectures
===================================

This script demonstrates efficient attention mechanisms:
- Standard attention memory analysis
- Flash Attention concepts
- Memory and speed comparisons
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math
import time


class StandardAttention(nn.Module):
    """
    Standard scaled dot-product attention.
    
    Memory complexity: O(N²) for the attention matrix
    """
    
    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project Q, K, V
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention: [batch, heads, seq, dim]
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores: [batch, heads, seq, seq] <- O(N²) memory!
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        if attention_mask is not None:
            attn_scores = attn_scores + attention_mask
        
        # Softmax
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attn_probs, v)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        return self.o_proj(output)


def calculate_attention_memory(batch_size: int, num_heads: int, 
                                seq_len: int, head_dim: int) -> dict:
    """Calculate memory requirements for attention."""
    
    # Q, K, V tensors
    qkv_memory = 3 * batch_size * num_heads * seq_len * head_dim * 4  # bytes
    
    # Attention scores: [batch, heads, seq, seq]
    attn_scores_memory = batch_size * num_heads * seq_len * seq_len * 4
    
    # Attention probs (after softmax)
    attn_probs_memory = batch_size * num_heads * seq_len * seq_len * 4
    
    # Output
    output_memory = batch_size * num_heads * seq_len * head_dim * 4
    
    total_standard = qkv_memory + attn_scores_memory + attn_probs_memory + output_memory
    
    # Flash Attention: O(N) - no full attention matrix
    # Only stores tiles in SRAM
    tile_size = 128  # typical tile size
    flash_memory = qkv_memory + batch_size * num_heads * seq_len * head_dim * 4
    
    return {
        'qkv_mb': qkv_memory / 1e6,
        'attn_scores_mb': attn_scores_memory / 1e6,
        'attn_probs_mb': attn_probs_memory / 1e6,
        'output_mb': output_memory / 1e6,
        'total_standard_mb': total_standard / 1e6,
        'total_flash_mb': flash_memory / 1e6,
        'savings_ratio': total_standard / flash_memory
    }


def demonstrate_memory_comparison():
    """Compare memory requirements for different sequence lengths."""
    
    print("\n" + "=" * 70)
    print("ATTENTION MEMORY COMPARISON")
    print("=" * 70)
    
    batch_size = 1
    num_heads = 32
    head_dim = 128  # Typical for LLaMA-7B
    
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768]
    
    print(f"\nConfiguration: batch={batch_size}, heads={num_heads}, head_dim={head_dim}")
    print(f"\n{'Seq Len':>10} {'Standard':>15} {'Flash':>15} {'Savings':>12}")
    print("-" * 55)
    
    for seq_len in seq_lengths:
        mem = calculate_attention_memory(batch_size, num_heads, seq_len, head_dim)
        
        standard_gb = mem['total_standard_mb'] / 1024
        flash_gb = mem['total_flash_mb'] / 1024
        
        if standard_gb < 1:
            standard_str = f"{mem['total_standard_mb']:.1f} MB"
        else:
            standard_str = f"{standard_gb:.2f} GB"
        
        if flash_gb < 1:
            flash_str = f"{mem['total_flash_mb']:.1f} MB"
        else:
            flash_str = f"{flash_gb:.2f} GB"
        
        print(f"{seq_len:>10} {standard_str:>15} {flash_str:>15} {mem['savings_ratio']:>10.1f}x")
    
    print("\n* Standard attention scales O(N²), Flash Attention scales O(N)")


def explain_flash_attention_algorithm():
    """Explain the Flash Attention algorithm."""
    
    print("\n" + "=" * 70)
    print("FLASH ATTENTION ALGORITHM")
    print("=" * 70)
    
    explanation = """
    Flash Attention Key Ideas:
    
    1. TILING: Divide Q, K, V into blocks that fit in SRAM
    
    2. RECOMPUTATION: Trade compute for memory
       - Don't store attention matrix
       - Recompute softmax statistics incrementally
    
    3. IO-AWARENESS: Minimize HBM (GPU memory) accesses
       - Load blocks to SRAM, compute there
       - Write only final output to HBM
    
    Algorithm (simplified):
    ┌────────────────────────────────────────────────────────────────┐
    │                                                                 │
    │  for each block of Q (size Br × d):                            │
    │      Load Q_block to SRAM                                       │
    │      Initialize: O = 0, ℓ = 0, m = -∞                          │
    │                                                                 │
    │      for each block of K, V (size Bc × d):                     │
    │          Load K_block, V_block to SRAM                          │
    │          Compute S_block = Q_block @ K_block.T                  │
    │                                                                 │
    │          # Online softmax update                                │
    │          m_new = max(m, rowmax(S_block))                        │
    │          P_block = exp(S_block - m_new)                         │
    │          ℓ_new = exp(m - m_new) × ℓ + rowsum(P_block)          │
    │                                                                 │
    │          # Update output                                        │
    │          O = (ℓ/ℓ_new) × O + P_block @ V_block / ℓ_new         │
    │          m, ℓ = m_new, ℓ_new                                   │
    │                                                                 │
    │      Write O to HBM                                             │
    │                                                                 │
    └────────────────────────────────────────────────────────────────┘
    
    Key insight: Online softmax allows computing attention without 
    ever materializing the full N×N attention matrix!
    """
    print(explanation)


def demonstrate_online_softmax():
    """Demonstrate the online softmax trick used in Flash Attention."""
    
    print("\n" + "=" * 70)
    print("ONLINE SOFTMAX (Flash Attention Key Trick)")
    print("=" * 70)
    
    print("""
    Problem: Standard softmax requires knowing all values first
    
    softmax(x_i) = exp(x_i) / Σ exp(x_j)
                            ↑ need to see all x first!
    
    Solution: Online softmax with running statistics
    
    For each new block of values:
    1. Track running maximum: m = max(m_old, max(x_new))
    2. Track running sum: ℓ = ℓ_old × exp(m_old - m) + Σ exp(x_new - m)
    3. Update output: O = O × exp(m_old - m) × (ℓ_old/ℓ) + ...
    """)
    
    # Demonstrate with example
    torch.manual_seed(42)
    
    # Full attention (standard)
    x = torch.randn(1, 8)  # 8 values
    softmax_full = F.softmax(x, dim=-1)
    
    print(f"\nExample: Computing softmax of 8 values in 2 blocks of 4")
    print(f"Input: {x[0].tolist()}")
    print(f"Standard softmax: {softmax_full[0].tolist()}")
    
    # Online computation (2 blocks of 4)
    block1, block2 = x[0, :4], x[0, 4:]
    
    # Process block 1
    m1 = block1.max()
    l1 = torch.exp(block1 - m1).sum()
    o1 = torch.exp(block1 - m1) / l1  # partial softmax
    
    # Process block 2, updating statistics
    m2 = max(m1, block2.max())
    # Update l with correction factor
    l2 = l1 * torch.exp(m1 - m2) + torch.exp(block2 - m2).sum()
    
    # Final softmax values
    final_block1 = torch.exp(block1 - m2) / l2
    final_block2 = torch.exp(block2 - m2) / l2
    online_result = torch.cat([final_block1, final_block2])
    
    print(f"Online softmax: {online_result.tolist()}")
    print(f"Difference: {(softmax_full[0] - online_result).abs().max().item():.2e}")


def flash_attention_usage():
    """Show how to use Flash Attention in practice."""
    
    print("\n" + "=" * 70)
    print("USING FLASH ATTENTION IN PRACTICE")
    print("=" * 70)
    
    code_example = '''
# Method 1: Hugging Face Transformers (easiest)
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    torch_dtype=torch.float16,
    attn_implementation="flash_attention_2",  # Enable Flash Attention 2
    device_map="auto"
)

# Method 2: flash-attn library directly
from flash_attn import flash_attn_func

# Inputs: [batch, seqlen, heads, headdim]
q = torch.randn(2, 1024, 32, 64, dtype=torch.float16, device='cuda')
k = torch.randn(2, 1024, 32, 64, dtype=torch.float16, device='cuda')
v = torch.randn(2, 1024, 32, 64, dtype=torch.float16, device='cuda')

output = flash_attn_func(q, k, v, causal=True)

# Method 3: PyTorch 2.0+ scaled_dot_product_attention
# Automatically uses Flash Attention when possible
output = F.scaled_dot_product_attention(
    q.transpose(1, 2),  # [batch, heads, seq, dim]
    k.transpose(1, 2),
    v.transpose(1, 2),
    is_causal=True
)

# Method 4: xformers
from xformers.ops import memory_efficient_attention

output = memory_efficient_attention(q, k, v)
'''
    
    print(code_example)


def compare_efficient_attention_methods():
    """Compare different efficient attention methods."""
    
    print("\n" + "=" * 70)
    print("EFFICIENT ATTENTION METHODS COMPARISON")
    print("=" * 70)
    
    comparison = """
    ┌──────────────────────┬────────────┬────────────┬────────────────────┐
    │ Method               │ Memory     │ Speed      │ Notes              │
    ├──────────────────────┼────────────┼────────────┼────────────────────┤
    │ Standard Attention   │ O(N²)      │ 1x         │ Baseline           │
    ├──────────────────────┼────────────┼────────────┼────────────────────┤
    │ Flash Attention      │ O(N)       │ 2-4x       │ Exact, IO-aware    │
    ├──────────────────────┼────────────┼────────────┼────────────────────┤
    │ Flash Attention 2    │ O(N)       │ 4-8x       │ Better parallelism │
    ├──────────────────────┼────────────┼────────────┼────────────────────┤
    │ xFormers             │ O(N)       │ 2-4x       │ Similar to Flash   │
    ├──────────────────────┼────────────┼────────────┼────────────────────┤
    │ PyTorch SDPA         │ O(N)*      │ 2-4x       │ Auto-selects best  │
    ├──────────────────────┼────────────┼────────────┼────────────────────┤
    │ Linear Attention     │ O(N)       │ ~1-2x      │ Approximate        │
    ├──────────────────────┼────────────┼────────────┼────────────────────┤
    │ Sparse (Longformer)  │ O(N×W)     │ Variable   │ Local + global     │
    └──────────────────────┴────────────┴────────────┴────────────────────┘
    
    * PyTorch SDPA (scaled_dot_product_attention) automatically selects:
      - Flash Attention when available
      - Memory-efficient attention as fallback
      - Standard attention as last resort
    
    Requirements for Flash Attention:
    - NVIDIA GPU (Ampere or newer recommended)
    - CUDA 11.6+
    - FP16 or BF16 (not FP32)
    - Head dimension ≤ 256
    """
    print(comparison)


def main():
    """Main demonstration of efficient architectures."""
    
    print("\n" + "=" * 70)
    print("   MODULE 09: EFFICIENT ARCHITECTURES")
    print("=" * 70)
    
    # 1. Memory comparison
    demonstrate_memory_comparison()
    
    # 2. Flash Attention algorithm
    explain_flash_attention_algorithm()
    
    # 3. Online softmax
    demonstrate_online_softmax()
    
    # 4. Usage examples
    flash_attention_usage()
    
    # 5. Comparison
    compare_efficient_attention_methods()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    EFFICIENT ARCHITECTURES KEY POINTS:
    
    1. FLASH ATTENTION:
       - IO-aware algorithm: optimizes memory access
       - Tiling: compute in blocks that fit in SRAM
       - Online softmax: never materialize full attention matrix
       - Result: O(N) memory instead of O(N²)
    
    2. BENEFITS:
       - 2-4x faster training/inference
       - Enables much longer sequences
       - No approximation (exact attention)
    
    3. USAGE:
       - Hugging Face: attn_implementation="flash_attention_2"
       - PyTorch 2.0+: F.scaled_dot_product_attention
       - flash-attn library for direct control
    
    4. REQUIREMENTS:
       - Modern NVIDIA GPU (Ampere+ best)
       - FP16/BF16 precision
       - CUDA 11.6+
    
    5. COMBINE WITH:
       - GQA/MQA for KV cache savings
       - Quantization for memory
       - Efficient decoding (speculative, etc.)
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()


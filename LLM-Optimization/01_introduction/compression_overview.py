"""
Module 01: Introduction to LLM Compression
==========================================

This script demonstrates fundamental concepts of model compression:
- Measuring model size and memory footprint
- Understanding compression ratios
- Comparing different precision formats
"""

import torch
import torch.nn as nn
from typing import Dict, Tuple
import sys


def calculate_model_size(model: nn.Module, precision: str = "fp32") -> Dict[str, float]:
    """
    Calculate the memory footprint of a PyTorch model.
    
    Args:
        model: PyTorch model
        precision: One of 'fp32', 'fp16', 'int8', 'int4'
    
    Returns:
        Dictionary with size information
    """
    bytes_per_param = {
        "fp32": 4,
        "fp16": 2,
        "bf16": 2,
        "int8": 1,
        "int4": 0.5
    }
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    size_bytes = total_params * bytes_per_param.get(precision, 4)
    size_mb = size_bytes / (1024 ** 2)
    size_gb = size_bytes / (1024 ** 3)
    
    return {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "precision": precision,
        "size_bytes": size_bytes,
        "size_mb": size_mb,
        "size_gb": size_gb
    }


def compare_precisions(model: nn.Module) -> None:
    """Compare model sizes across different precisions."""
    precisions = ["fp32", "fp16", "int8", "int4"]
    
    print("\n" + "=" * 60)
    print("MODEL SIZE COMPARISON ACROSS PRECISIONS")
    print("=" * 60)
    
    base_size = None
    for precision in precisions:
        info = calculate_model_size(model, precision)
        if base_size is None:
            base_size = info["size_mb"]
        
        compression_ratio = base_size / info["size_mb"]
        
        print(f"\n{precision.upper():>6}: {info['size_mb']:>10.2f} MB | "
              f"Compression: {compression_ratio:.1f}x")
    
    print("\n" + "=" * 60)


class SimpleLLMBlock(nn.Module):
    """A simplified transformer block for demonstration."""
    
    def __init__(self, hidden_size: int = 768, num_heads: int = 12, 
                 intermediate_size: int = 3072):
        super().__init__()
        
        # Self-attention
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        # Feed-forward network
        self.up_proj = nn.Linear(hidden_size, intermediate_size)
        self.down_proj = nn.Linear(intermediate_size, hidden_size)
        
        # Layer norms
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention
        residual = x
        x = self.ln1(x)
        q, k, v = self.q_proj(x), self.k_proj(x), self.v_proj(x)
        # Simplified attention (not full implementation)
        attn_out = self.o_proj(v)
        x = residual + attn_out
        
        # Feed-forward
        residual = x
        x = self.ln2(x)
        x = self.up_proj(x)
        x = torch.relu(x)
        x = self.down_proj(x)
        x = residual + x
        
        return x


class SimpleLLM(nn.Module):
    """A simplified LLM architecture for demonstration."""
    
    def __init__(self, vocab_size: int = 32000, hidden_size: int = 768,
                 num_layers: int = 12, num_heads: int = 12,
                 intermediate_size: int = 3072):
        super().__init__()
        
        self.embed_tokens = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            SimpleLLMBlock(hidden_size, num_heads, intermediate_size)
            for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits


def simulate_llm_sizes():
    """Simulate memory requirements for popular LLM sizes."""
    
    llm_configs = {
        "GPT-2 Small": {"params": 124_000_000},
        "GPT-2 Medium": {"params": 355_000_000},
        "GPT-2 Large": {"params": 774_000_000},
        "GPT-2 XL": {"params": 1_500_000_000},
        "LLaMA-7B": {"params": 7_000_000_000},
        "LLaMA-13B": {"params": 13_000_000_000},
        "LLaMA-33B": {"params": 33_000_000_000},
        "LLaMA-65B": {"params": 65_000_000_000},
        "LLaMA-2-70B": {"params": 70_000_000_000},
        "Falcon-180B": {"params": 180_000_000_000},
    }
    
    print("\n" + "=" * 80)
    print("POPULAR LLM MEMORY REQUIREMENTS")
    print("=" * 80)
    print(f"{'Model':<20} {'Params':>12} {'FP32':>12} {'FP16':>12} {'INT8':>12} {'INT4':>12}")
    print("-" * 80)
    
    for name, config in llm_configs.items():
        params = config["params"]
        fp32_gb = (params * 4) / (1024 ** 3)
        fp16_gb = (params * 2) / (1024 ** 3)
        int8_gb = (params * 1) / (1024 ** 3)
        int4_gb = (params * 0.5) / (1024 ** 3)
        
        params_str = f"{params / 1e9:.1f}B" if params >= 1e9 else f"{params / 1e6:.0f}M"
        
        print(f"{name:<20} {params_str:>12} {fp32_gb:>10.1f}GB {fp16_gb:>10.1f}GB "
              f"{int8_gb:>10.1f}GB {int4_gb:>10.1f}GB")
    
    print("=" * 80)


def compression_ratio_analysis():
    """Analyze compression ratios and their implications."""
    
    print("\n" + "=" * 60)
    print("COMPRESSION RATIO ANALYSIS")
    print("=" * 60)
    
    # Example: LLaMA-7B
    original_size_gb = 28  # FP32
    
    techniques = [
        ("FP16 Conversion", 2.0, 0.0),
        ("INT8 Quantization", 4.0, 0.5),
        ("INT4 Quantization (GPTQ)", 8.0, 1.5),
        ("INT4 + Pruning (50%)", 16.0, 3.0),
        ("Distillation (to 1B)", 7.0, 5.0),
    ]
    
    print(f"\nStarting with LLaMA-7B at {original_size_gb} GB (FP32)\n")
    print(f"{'Technique':<30} {'Ratio':>10} {'New Size':>12} {'Perplexity Δ':>15}")
    print("-" * 70)
    
    for technique, ratio, ppl_delta in techniques:
        new_size = original_size_gb / ratio
        ppl_str = f"+{ppl_delta}%" if ppl_delta > 0 else "0%"
        print(f"{technique:<30} {ratio:>10.1f}x {new_size:>10.1f} GB {ppl_str:>15}")
    
    print("=" * 60)


def main():
    """Main demonstration of compression concepts."""
    
    print("\n" + "=" * 60)
    print("   LLM OPTIMIZATION: INTRODUCTION TO COMPRESSION")
    print("=" * 60)
    
    # 1. Create a simple model
    print("\n[1] Creating a simplified LLM (BERT-base scale)...")
    model = SimpleLLM(
        vocab_size=32000,
        hidden_size=768,
        num_layers=12,
        num_heads=12,
        intermediate_size=3072
    )
    
    info = calculate_model_size(model)
    print(f"    Total parameters: {info['total_params']:,}")
    print(f"    Model size (FP32): {info['size_mb']:.2f} MB")
    
    # 2. Compare different precisions
    print("\n[2] Comparing model size across precisions...")
    compare_precisions(model)
    
    # 3. Simulate popular LLM sizes
    print("\n[3] Popular LLM memory requirements...")
    simulate_llm_sizes()
    
    # 4. Compression ratio analysis
    print("\n[4] Compression ratio analysis...")
    compression_ratio_analysis()
    
    # 5. Key insights
    print("\n" + "=" * 60)
    print("KEY INSIGHTS")
    print("=" * 60)
    print("""
    1. PRECISION MATTERS: FP32 → INT4 gives 8x compression
    
    2. COMPOUND TECHNIQUES: Combining methods yields best results
       - Quantization + Pruning + Distillation = 10-100x compression
    
    3. TRADEOFFS EXIST: More compression = more accuracy loss
       - INT8: Usually < 1% perplexity increase
       - INT4: Usually 1-3% perplexity increase
       - Aggressive pruning: Can lose 5%+ accuracy
    
    4. DEPLOYMENT TARGET MATTERS:
       - Cloud: Can use larger models, focus on throughput
       - Edge: Must use aggressive compression
       - Mobile: Need INT4 or smaller models
    
    5. MEASURE EVERYTHING:
       - Memory usage
       - Inference latency
       - Throughput (tokens/second)
       - Task-specific accuracy
    """)
    print("=" * 60)


if __name__ == "__main__":
    main()


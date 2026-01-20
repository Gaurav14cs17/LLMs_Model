"""
Module 03: Structured Pruning
=============================

This script demonstrates structured pruning techniques:
- Channel/filter pruning
- Attention head pruning
- Layer pruning
- N:M sparsity patterns
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import copy


class ConvBlock(nn.Module):
    """Convolutional block for demonstrating channel pruning."""
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class SimpleCNN(nn.Module):
    """Simple CNN for demonstrating structured pruning."""
    
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv1 = ConvBlock(3, 64)
        self.conv2 = ConvBlock(64, 128)
        self.conv3 = ConvBlock(128, 256)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class MultiHeadAttention(nn.Module):
    """Multi-head attention for demonstrating head pruning."""
    
    def __init__(self, hidden_size: int = 512, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        # Head importance scores (learned or computed)
        self.head_mask = nn.Parameter(torch.ones(num_heads), requires_grad=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Project and reshape
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpose for attention: (batch, heads, seq, dim)
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        
        # Apply head mask
        attn = attn * self.head_mask.view(1, -1, 1, 1)
        
        # Combine heads
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.o_proj(out)


def compute_channel_importance(conv_layer: nn.Conv2d, 
                                bn_layer: Optional[nn.BatchNorm2d] = None,
                                method: str = "l1") -> torch.Tensor:
    """
    Compute importance scores for each output channel.
    
    Methods:
    - l1: L1 norm of weights
    - l2: L2 norm of weights
    - bn: BatchNorm gamma (scaling factor)
    """
    weight = conv_layer.weight.data  # [out, in, h, w]
    
    if method == "l1":
        # Sum of absolute values per channel
        importance = weight.abs().sum(dim=(1, 2, 3))
    elif method == "l2":
        # L2 norm per channel
        importance = weight.pow(2).sum(dim=(1, 2, 3)).sqrt()
    elif method == "bn" and bn_layer is not None:
        # BatchNorm gamma as importance
        importance = bn_layer.weight.data.abs()
    else:
        importance = weight.abs().sum(dim=(1, 2, 3))
    
    return importance


def prune_conv_channels(conv_layer: nn.Conv2d, 
                         bn_layer: nn.BatchNorm2d,
                         keep_indices: torch.Tensor) -> Tuple[nn.Conv2d, nn.BatchNorm2d]:
    """
    Prune output channels from a convolutional layer.
    
    Returns new layers with reduced channels.
    """
    new_out_channels = len(keep_indices)
    
    # Create new conv layer
    new_conv = nn.Conv2d(
        conv_layer.in_channels,
        new_out_channels,
        conv_layer.kernel_size,
        stride=conv_layer.stride,
        padding=conv_layer.padding,
        bias=conv_layer.bias is not None
    )
    
    # Copy selected weights
    new_conv.weight.data = conv_layer.weight.data[keep_indices].clone()
    if conv_layer.bias is not None:
        new_conv.bias.data = conv_layer.bias.data[keep_indices].clone()
    
    # Create new BN layer
    new_bn = nn.BatchNorm2d(new_out_channels)
    new_bn.weight.data = bn_layer.weight.data[keep_indices].clone()
    new_bn.bias.data = bn_layer.bias.data[keep_indices].clone()
    new_bn.running_mean = bn_layer.running_mean[keep_indices].clone()
    new_bn.running_var = bn_layer.running_var[keep_indices].clone()
    
    return new_conv, new_bn


def demonstrate_channel_pruning():
    """Demonstrate channel/filter pruning on a CNN."""
    
    print("\n" + "=" * 60)
    print("CHANNEL/FILTER PRUNING")
    print("=" * 60)
    
    # Create model
    model = SimpleCNN()
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nOriginal model parameters: {total_params:,}")
    
    # Analyze conv2 layer (128 output channels)
    conv = model.conv2.conv
    bn = model.conv2.bn
    
    importance = compute_channel_importance(conv, bn, method="l1")
    
    print(f"\nConv2 layer: {conv.in_channels} -> {conv.out_channels} channels")
    print(f"Channel importance scores (first 10): {importance[:10].tolist()}")
    
    # Prune 50% of channels
    num_keep = conv.out_channels // 2
    _, keep_indices = torch.topk(importance, num_keep)
    keep_indices = keep_indices.sort().values
    
    print(f"\nPruning 50% of channels ({conv.out_channels} -> {num_keep})")
    print(f"Keeping channel indices: {keep_indices[:10].tolist()}...")
    
    # Create pruned layers
    new_conv, new_bn = prune_conv_channels(conv, bn, keep_indices)
    
    print(f"\nPruned conv layer: {new_conv}")
    print(f"Pruned BN layer: {new_bn}")
    
    # Parameter reduction
    old_params = conv.weight.numel() + bn.weight.numel() + bn.bias.numel()
    new_params = new_conv.weight.numel() + new_bn.weight.numel() + new_bn.bias.numel()
    print(f"\nParameter reduction: {old_params:,} -> {new_params:,} ({new_params/old_params:.1%})")


def compute_head_importance(attention: MultiHeadAttention,
                            sample_input: torch.Tensor) -> torch.Tensor:
    """
    Compute importance scores for each attention head.
    
    Uses activation-based importance: heads with larger
    output magnitudes are considered more important.
    """
    attention.eval()
    
    with torch.no_grad():
        batch_size, seq_len, _ = sample_input.shape
        
        q = attention.q_proj(sample_input).view(
            batch_size, seq_len, attention.num_heads, attention.head_dim
        )
        k = attention.k_proj(sample_input).view(
            batch_size, seq_len, attention.num_heads, attention.head_dim
        )
        v = attention.v_proj(sample_input).view(
            batch_size, seq_len, attention.num_heads, attention.head_dim
        )
        
        q, k, v = [t.transpose(1, 2) for t in (q, k, v)]
        
        attn = torch.matmul(q, k.transpose(-2, -1)) / (attention.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        
        # Head importance = mean absolute activation
        importance = out.abs().mean(dim=(0, 2, 3))
    
    return importance


def demonstrate_head_pruning():
    """Demonstrate attention head pruning."""
    
    print("\n" + "=" * 60)
    print("ATTENTION HEAD PRUNING")
    print("=" * 60)
    
    # Create attention layer
    attn = MultiHeadAttention(hidden_size=512, num_heads=8)
    
    # Sample input
    sample_input = torch.randn(4, 32, 512)
    
    # Compute head importance
    importance = compute_head_importance(attn, sample_input)
    
    print(f"\nAttention layer: 8 heads, 64 dim each")
    print(f"Head importance scores: {importance.tolist()}")
    
    # Prune 50% of heads (keep top 4)
    num_keep = 4
    _, keep_indices = torch.topk(importance, num_keep)
    keep_indices = keep_indices.sort().values
    
    print(f"\nPruning 4 heads, keeping heads: {keep_indices.tolist()}")
    
    # Create mask
    head_mask = torch.zeros(8)
    head_mask[keep_indices] = 1.0
    attn.head_mask.data = head_mask
    
    print(f"Head mask: {head_mask.tolist()}")
    
    # In practice, you would also prune the projection weights
    # to actually reduce computation
    print(f"\nNote: For actual speedup, also prune Q/K/V/O projections")
    print(f"New hidden size would be: {num_keep * attn.head_dim} (was 512)")


def demonstrate_nm_sparsity():
    """Demonstrate N:M semi-structured sparsity."""
    
    print("\n" + "=" * 60)
    print("N:M SEMI-STRUCTURED SPARSITY (2:4)")
    print("=" * 60)
    
    print("""
    2:4 Sparsity Pattern:
    - Every 4 consecutive weights, keep only 2 (50% sparse)
    - Hardware-accelerated on NVIDIA Ampere+ GPUs
    - ~2x speedup with minimal accuracy loss
    """)
    
    # Create a sample weight tensor
    weight = torch.randn(8, 16)
    print(f"\nOriginal weight shape: {weight.shape}")
    print(f"Original weights (first row): {weight[0].tolist()}")
    
    # Apply 2:4 sparsity
    def apply_2_4_sparsity(weight: torch.Tensor) -> torch.Tensor:
        """Apply 2:4 sparsity pattern."""
        weight = weight.clone()
        
        # Process in groups of 4
        for i in range(0, weight.shape[1], 4):
            group = weight[:, i:i+4]
            
            # Find indices of 2 smallest values (to prune)
            _, indices = torch.topk(group.abs(), 2, dim=1, largest=False)
            
            # Create mask
            mask = torch.ones_like(group)
            mask.scatter_(1, indices, 0)
            
            # Apply mask
            weight[:, i:i+4] = group * mask
        
        return weight
    
    sparse_weight = apply_2_4_sparsity(weight)
    
    print(f"Sparse weights (first row): {sparse_weight[0].tolist()}")
    
    # Verify sparsity
    sparsity = (sparse_weight == 0).sum() / sparse_weight.numel()
    print(f"\nActual sparsity: {sparsity:.1%}")
    
    # Show pattern
    print(f"\nPattern visualization (0 = pruned):")
    mask = (sparse_weight[0] != 0).int()
    for i in range(0, 16, 4):
        print(f"  Group {i//4}: {mask[i:i+4].tolist()}")


def demonstrate_layer_pruning():
    """Demonstrate layer pruning (removing entire layers)."""
    
    print("\n" + "=" * 60)
    print("LAYER PRUNING")
    print("=" * 60)
    
    print("""
    Layer Pruning:
    - Remove entire transformer layers
    - Most aggressive compression
    - Often combined with knowledge distillation
    
    Research findings (BERT):
    - Can remove 30-40% of layers with <2% accuracy drop
    - Earlier layers more important for syntax
    - Later layers more important for semantics
    """)
    
    class TransformerEncoder(nn.Module):
        def __init__(self, num_layers: int = 12, hidden_size: int = 768):
            super().__init__()
            self.layers = nn.ModuleList([
                nn.TransformerEncoderLayer(hidden_size, nhead=12, batch_first=True)
                for _ in range(num_layers)
            ])
            self.layer_importance = nn.Parameter(
                torch.ones(num_layers), requires_grad=False
            )
        
        def forward(self, x):
            for i, layer in enumerate(self.layers):
                if self.layer_importance[i] > 0:
                    x = layer(x)
            return x
    
    model = TransformerEncoder(num_layers=12)
    
    # Simulate layer importance (in practice, computed from gradients)
    layer_importance = torch.tensor([
        1.0, 0.9, 0.7, 0.5, 0.4, 0.3,  # Earlier layers
        0.3, 0.4, 0.5, 0.7, 0.9, 1.0   # Later layers
    ])
    
    print(f"\nLayer importance scores:")
    for i, imp in enumerate(layer_importance):
        bar = "█" * int(imp * 10)
        print(f"  Layer {i:2d}: {bar:<10} {imp:.2f}")
    
    # Prune layers with importance < 0.5
    keep_mask = layer_importance >= 0.5
    num_kept = keep_mask.sum().item()
    
    print(f"\nKeeping layers with importance >= 0.5:")
    print(f"  Kept: {num_kept}/12 layers")
    print(f"  Kept indices: {torch.where(keep_mask)[0].tolist()}")
    print(f"  Compression: {12/num_kept:.1f}x fewer layers")


def main():
    """Main demonstration of structured pruning."""
    
    print("\n" + "=" * 60)
    print("   MODULE 03: STRUCTURED PRUNING")
    print("=" * 60)
    
    # 1. Channel pruning
    demonstrate_channel_pruning()
    
    # 2. Head pruning
    demonstrate_head_pruning()
    
    # 3. N:M sparsity
    demonstrate_nm_sparsity()
    
    # 4. Layer pruning
    demonstrate_layer_pruning()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    STRUCTURED PRUNING TECHNIQUES:
    
    1. CHANNEL/FILTER PRUNING:
       - Remove entire convolution filters
       - Works on standard hardware
       - Use L1/L2 norm or BN gamma for importance
    
    2. ATTENTION HEAD PRUNING:
       - Remove entire attention heads
       - LLMs are often over-parameterized in heads
       - Can prune 20-40% with minimal loss
    
    3. N:M SPARSITY (2:4):
       - Semi-structured pattern
       - Hardware accelerated on Ampere+
       - 50% sparsity with ~2x speedup
    
    4. LAYER PRUNING:
       - Remove entire transformer layers
       - Most aggressive compression
       - Often combined with distillation
    
    ADVANTAGES OF STRUCTURED PRUNING:
    - Works on standard hardware
    - Actual speedups (not just theoretical)
    - Easier to implement and deploy
    
    DISADVANTAGES:
    - Less flexible than unstructured
    - Harder to achieve high compression
    - May need architecture changes
    """)
    print("=" * 60)


if __name__ == "__main__":
    main()


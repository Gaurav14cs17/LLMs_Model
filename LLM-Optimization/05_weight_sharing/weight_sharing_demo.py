"""
Module 05: Weight Sharing
=========================

This script demonstrates weight sharing techniques:
- Cross-layer parameter sharing (ALBERT-style)
- Embedding factorization
- Multi-Query Attention (MQA)
- Grouped-Query Attention (GQA)
- Weight clustering
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class ALBERTEncoder(nn.Module):
    """
    ALBERT-style encoder with cross-layer parameter sharing.
    
    All transformer layers share the same weights, dramatically
    reducing the number of parameters.
    """
    
    def __init__(self, hidden_size: int = 768, num_layers: int = 12,
                 num_heads: int = 12, intermediate_size: int = 3072):
        super().__init__()
        
        # Single shared transformer layer
        self.shared_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=intermediate_size,
            batch_first=True
        )
        
        self.num_layers = num_layers
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply the same layer multiple times
        for _ in range(self.num_layers):
            x = self.shared_layer(x)
        return x


class StandardEncoder(nn.Module):
    """Standard encoder with separate parameters per layer."""
    
    def __init__(self, hidden_size: int = 768, num_layers: int = 12,
                 num_heads: int = 12, intermediate_size: int = 3072):
        super().__init__()
        
        # Separate layer for each
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=intermediate_size,
                batch_first=True
            ) for _ in range(num_layers)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class FactorizedEmbedding(nn.Module):
    """
    Factorized embedding layer (ALBERT-style).
    
    Instead of V × H, use V × E and E × H where E << H.
    """
    
    def __init__(self, vocab_size: int, hidden_size: int, 
                 embedding_size: int = 128):
        super().__init__()
        
        # Factorized embeddings
        self.word_embeddings = nn.Embedding(vocab_size, embedding_size)
        self.projection = nn.Linear(embedding_size, hidden_size)
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.word_embeddings(input_ids)
        return self.projection(embeddings)


class MultiQueryAttention(nn.Module):
    """
    Multi-Query Attention (MQA).
    
    All query heads share a single key and value head.
    Significantly faster inference, especially for long sequences.
    """
    
    def __init__(self, hidden_size: int = 512, num_heads: int = 8):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        
        # Multiple query heads
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        
        # Single key and value head (shared)
        self.k_proj = nn.Linear(hidden_size, self.head_dim)
        self.v_proj = nn.Linear(hidden_size, self.head_dim)
        
        self.o_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Queries: multiple heads
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)  # [batch, heads, seq, dim]
        
        # Keys and values: single head, broadcast to all
        k = self.k_proj(x).view(batch_size, seq_len, 1, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, 1, self.head_dim)
        k = k.transpose(1, 2)  # [batch, 1, seq, dim]
        v = v.transpose(1, 2)  # [batch, 1, seq, dim]
        
        # Attention (k, v are broadcast across heads)
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)  # [batch, heads, seq, dim]
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.o_proj(out)


class GroupedQueryAttention(nn.Module):
    """
    Grouped-Query Attention (GQA).
    
    Groups of query heads share key-value heads.
    Balance between MHA (quality) and MQA (speed).
    """
    
    def __init__(self, hidden_size: int = 512, num_heads: int = 8,
                 num_kv_heads: int = 2):
        super().__init__()
        
        assert num_heads % num_kv_heads == 0
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.heads_per_group = num_heads // num_kv_heads
        self.head_dim = hidden_size // num_heads
        
        # Query projection (all heads)
        self.q_proj = nn.Linear(hidden_size, hidden_size)
        
        # Key-Value projection (fewer heads)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * self.head_dim)
        
        self.o_proj = nn.Linear(hidden_size, hidden_size)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Queries: all heads
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        q = q.transpose(1, 2)
        
        # Keys and values: fewer heads
        k = self.k_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_kv_heads, self.head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Repeat k, v for each group
        k = k.repeat_interleave(self.heads_per_group, dim=1)
        v = v.repeat_interleave(self.heads_per_group, dim=1)
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        
        return self.o_proj(out)


def demonstrate_albert_sharing():
    """Demonstrate ALBERT-style parameter sharing."""
    
    print("\n" + "=" * 60)
    print("ALBERT-STYLE CROSS-LAYER PARAMETER SHARING")
    print("=" * 60)
    
    # Create both models
    standard = StandardEncoder(hidden_size=768, num_layers=12)
    albert = ALBERTEncoder(hidden_size=768, num_layers=12)
    
    # Count parameters
    standard_params = sum(p.numel() for p in standard.parameters())
    albert_params = sum(p.numel() for p in albert.parameters())
    
    print(f"\nStandard Encoder (12 layers):")
    print(f"  Parameters: {standard_params:,}")
    
    print(f"\nALBERT Encoder (12 layers, shared):")
    print(f"  Parameters: {albert_params:,}")
    
    print(f"\nParameter reduction: {standard_params / albert_params:.1f}x")
    
    # Test forward pass
    x = torch.randn(2, 32, 768)
    
    with torch.no_grad():
        standard_out = standard(x)
        albert_out = albert(x)
    
    print(f"\nOutput shapes: {standard_out.shape} (both identical)")


def demonstrate_embedding_factorization():
    """Demonstrate embedding factorization."""
    
    print("\n" + "=" * 60)
    print("EMBEDDING FACTORIZATION")
    print("=" * 60)
    
    vocab_size = 30000
    hidden_size = 768
    embedding_size = 128
    
    # Standard embedding
    standard_emb = nn.Embedding(vocab_size, hidden_size)
    standard_params = sum(p.numel() for p in standard_emb.parameters())
    
    # Factorized embedding
    factorized_emb = FactorizedEmbedding(vocab_size, hidden_size, embedding_size)
    factorized_params = sum(p.numel() for p in factorized_emb.parameters())
    
    print(f"\nVocab size: {vocab_size:,}, Hidden size: {hidden_size}")
    
    print(f"\nStandard Embedding (V × H):")
    print(f"  Parameters: {standard_params:,}")
    
    print(f"\nFactorized Embedding (V × E + E × H), E={embedding_size}:")
    print(f"  Word embeddings (V × E): {vocab_size * embedding_size:,}")
    print(f"  Projection (E × H): {embedding_size * hidden_size:,}")
    print(f"  Total: {factorized_params:,}")
    
    print(f"\nParameter reduction: {standard_params / factorized_params:.1f}x")


def demonstrate_mqa_gqa():
    """Demonstrate MQA and GQA."""
    
    print("\n" + "=" * 60)
    print("MULTI-QUERY & GROUPED-QUERY ATTENTION")
    print("=" * 60)
    
    hidden_size = 512
    num_heads = 8
    head_dim = hidden_size // num_heads
    
    # Standard MHA
    mha = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
    
    # MQA
    mqa = MultiQueryAttention(hidden_size, num_heads)
    
    # GQA (4 KV heads for 8 query heads)
    gqa = GroupedQueryAttention(hidden_size, num_heads, num_kv_heads=4)
    
    # Count parameters
    mha_params = sum(p.numel() for p in mha.parameters())
    mqa_params = sum(p.numel() for p in mqa.parameters())
    gqa_params = sum(p.numel() for p in gqa.parameters())
    
    print(f"\nHidden size: {hidden_size}, Heads: {num_heads}")
    
    print(f"\nMulti-Head Attention (MHA):")
    print(f"  Parameters: {mha_params:,}")
    print(f"  KV heads: {num_heads}")
    
    print(f"\nMulti-Query Attention (MQA):")
    print(f"  Parameters: {mqa_params:,}")
    print(f"  KV heads: 1")
    
    print(f"\nGrouped-Query Attention (GQA):")
    print(f"  Parameters: {gqa_params:,}")
    print(f"  KV heads: 4")
    
    # KV cache size comparison
    batch_size, seq_len = 1, 2048
    kv_cache_mha = 2 * batch_size * seq_len * num_heads * head_dim * 4  # 4 bytes per float
    kv_cache_mqa = 2 * batch_size * seq_len * 1 * head_dim * 4
    kv_cache_gqa = 2 * batch_size * seq_len * 4 * head_dim * 4
    
    print(f"\nKV Cache size (batch=1, seq=2048):")
    print(f"  MHA: {kv_cache_mha / 1024:.1f} KB")
    print(f"  MQA: {kv_cache_mqa / 1024:.1f} KB ({kv_cache_mha / kv_cache_mqa:.0f}x smaller)")
    print(f"  GQA: {kv_cache_gqa / 1024:.1f} KB ({kv_cache_mha / kv_cache_gqa:.0f}x smaller)")


def demonstrate_weight_clustering():
    """Demonstrate weight clustering/quantization."""
    
    print("\n" + "=" * 60)
    print("WEIGHT CLUSTERING")
    print("=" * 60)
    
    # Create sample weights
    weight = torch.randn(64, 64)
    
    print(f"\nOriginal weight: {weight.shape}")
    print(f"Unique values: {weight.numel()} (all unique)")
    
    # Cluster weights using k-means style
    num_clusters = 16
    
    # Simple clustering: find centroids
    flat_weight = weight.view(-1)
    
    # Initialize centroids uniformly
    min_val, max_val = flat_weight.min(), flat_weight.max()
    centroids = torch.linspace(min_val, max_val, num_clusters)
    
    # Assign each weight to nearest centroid
    distances = (flat_weight.unsqueeze(1) - centroids.unsqueeze(0)).abs()
    cluster_assignments = distances.argmin(dim=1)
    
    # Update centroids
    new_centroids = torch.zeros_like(centroids)
    for i in range(num_clusters):
        mask = cluster_assignments == i
        if mask.sum() > 0:
            new_centroids[i] = flat_weight[mask].mean()
        else:
            new_centroids[i] = centroids[i]
    
    # Reconstruct using centroids
    clustered_weight = new_centroids[cluster_assignments].view(weight.shape)
    
    # Calculate error
    mse = F.mse_loss(clustered_weight, weight)
    
    print(f"\nClustered weight:")
    print(f"  Number of clusters: {num_clusters}")
    print(f"  Unique values: {num_clusters}")
    print(f"  MSE: {mse:.6f}")
    
    # Storage analysis
    original_bits = weight.numel() * 32  # FP32
    clustered_bits = (
        num_clusters * 32 +  # Centroids (FP32)
        weight.numel() * math.ceil(math.log2(num_clusters))  # Indices (4 bits for 16 clusters)
    )
    
    print(f"\nStorage:")
    print(f"  Original: {original_bits / 8 / 1024:.2f} KB")
    print(f"  Clustered: {clustered_bits / 8 / 1024:.2f} KB")
    print(f"  Compression: {original_bits / clustered_bits:.1f}x")


def main():
    """Main demonstration of weight sharing techniques."""
    
    print("\n" + "=" * 60)
    print("   MODULE 05: WEIGHT SHARING")
    print("=" * 60)
    
    # 1. ALBERT-style sharing
    demonstrate_albert_sharing()
    
    # 2. Embedding factorization
    demonstrate_embedding_factorization()
    
    # 3. MQA and GQA
    demonstrate_mqa_gqa()
    
    # 4. Weight clustering
    demonstrate_weight_clustering()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    WEIGHT SHARING TECHNIQUES:
    
    1. CROSS-LAYER SHARING (ALBERT):
       - All layers share same weights
       - 12x parameter reduction
       - Slight quality trade-off
    
    2. EMBEDDING FACTORIZATION:
       - V×H → V×E + E×H
       - 5-6x smaller embeddings
       - Minimal quality impact
    
    3. MULTI-QUERY ATTENTION (MQA):
       - Single K/V head shared by all Q heads
       - 8x smaller KV cache
       - Major inference speedup
    
    4. GROUPED-QUERY ATTENTION (GQA):
       - Groups of Q heads share K/V heads
       - Balance quality vs speed
       - Used in Llama 2 70B
    
    5. WEIGHT CLUSTERING:
       - Group similar weights
       - Store centroids + indices
       - 4-8x compression
    
    MODERN LLMs:
    - LLaMA 2 7B/13B: MHA (standard)
    - LLaMA 2 70B: GQA (8 KV heads)
    - Falcon: MQA
    - Mistral: GQA
    """)
    print("=" * 60)


if __name__ == "__main__":
    main()


"""
Module 08: LoRA (Low-Rank Adaptation)
=====================================

This script demonstrates LoRA implementation and usage:
- Understanding low-rank adaptation
- Implementing LoRA from scratch
- Using the PEFT library
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Dict
import math


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation.
    
    Instead of updating W directly, we learn:
    W' = W + BA
    
    Where:
    - W: [out_features, in_features] (frozen)
    - B: [out_features, r] (trainable)
    - A: [r, in_features] (trainable)
    - r: rank (typically 8-64)
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.0,
        merge_weights: bool = False
    ):
        super().__init__()
        
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r
        self.merge_weights = merge_weights
        self.merged = False
        
        # Original linear layer (frozen)
        self.linear = nn.Linear(in_features, out_features)
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(r, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, r))
        
        # Dropout
        self.lora_dropout = nn.Dropout(p=lora_dropout) if lora_dropout > 0 else nn.Identity()
        
        # Initialize
        self.reset_lora_parameters()
        
        # Freeze original weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False
    
    def reset_lora_parameters(self):
        """Initialize LoRA parameters."""
        # A: Kaiming uniform initialization
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B: Zero initialization (start with original weights)
        nn.init.zeros_(self.lora_B)
    
    def merge(self):
        """Merge LoRA weights into the main weights."""
        if not self.merged:
            self.linear.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            self.merged = True
    
    def unmerge(self):
        """Unmerge LoRA weights from the main weights."""
        if self.merged:
            self.linear.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
            self.merged = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.linear(x)
        
        # Original output
        result = self.linear(x)
        
        # Add LoRA output
        lora_output = self.lora_dropout(x) @ self.lora_A.T @ self.lora_B.T
        result = result + lora_output * self.scaling
        
        return result
    
    @classmethod
    def from_linear(cls, linear: nn.Linear, r: int = 8, lora_alpha: int = 16,
                    lora_dropout: float = 0.0) -> 'LoRALinear':
        """Create LoRALinear from an existing Linear layer."""
        lora_linear = cls(
            linear.in_features,
            linear.out_features,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        lora_linear.linear.weight.data = linear.weight.data.clone()
        if linear.bias is not None:
            lora_linear.linear.bias.data = linear.bias.data.clone()
        return lora_linear


class SimpleTransformer(nn.Module):
    """Simple transformer for demonstrating LoRA."""
    
    def __init__(self, hidden_size: int = 768, num_heads: int = 12,
                 num_layers: int = 6, vocab_size: int = 32000):
        super().__init__()
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': nn.ModuleDict({
                    'q_proj': nn.Linear(hidden_size, hidden_size),
                    'k_proj': nn.Linear(hidden_size, hidden_size),
                    'v_proj': nn.Linear(hidden_size, hidden_size),
                    'o_proj': nn.Linear(hidden_size, hidden_size),
                }),
                'ffn': nn.ModuleDict({
                    'up_proj': nn.Linear(hidden_size, hidden_size * 4),
                    'down_proj': nn.Linear(hidden_size * 4, hidden_size),
                }),
                'ln1': nn.LayerNorm(hidden_size),
                'ln2': nn.LayerNorm(hidden_size),
            })
            for _ in range(num_layers)
        ])
        self.output = nn.Linear(hidden_size, vocab_size)
        
        self.hidden_size = hidden_size
        self.num_heads = num_heads
    
    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        
        for layer in self.layers:
            # Self-attention (simplified)
            residual = x
            x = layer['ln1'](x)
            q = layer['attention']['q_proj'](x)
            k = layer['attention']['k_proj'](x)
            v = layer['attention']['v_proj'](x)
            # Simplified attention
            attn = F.softmax(q @ k.transpose(-2, -1) / math.sqrt(self.hidden_size), dim=-1)
            x = layer['attention']['o_proj'](attn @ v)
            x = residual + x
            
            # FFN
            residual = x
            x = layer['ln2'](x)
            x = layer['ffn']['up_proj'](x)
            x = F.gelu(x)
            x = layer['ffn']['down_proj'](x)
            x = residual + x
        
        return self.output(x)


def apply_lora_to_model(model: nn.Module, 
                         target_modules: List[str],
                         r: int = 8,
                         lora_alpha: int = 16,
                         lora_dropout: float = 0.0) -> nn.Module:
    """Apply LoRA to specified modules in a model."""
    
    for name, module in model.named_modules():
        for target in target_modules:
            if target in name and isinstance(module, nn.Linear):
                # Get parent module
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = model
                    for part in parent_name.split('.'):
                        if part.isdigit():
                            parent = parent[int(part)]
                        else:
                            parent = getattr(parent, part)
                else:
                    parent = model
                
                # Replace with LoRA version
                lora_module = LoRALinear.from_linear(
                    module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout
                )
                
                if child_name.isdigit():
                    parent[int(child_name)] = lora_module
                else:
                    setattr(parent, child_name, lora_module)
    
    return model


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and trainable parameters."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable,
        'trainable_percent': trainable / total * 100
    }


def demonstrate_lora_math():
    """Demonstrate the mathematics of LoRA."""
    
    print("\n" + "=" * 60)
    print("LoRA MATHEMATICS")
    print("=" * 60)
    
    # Example dimensions
    d = 768  # hidden size
    k = 768  # same for q, k, v projections
    r = 16   # rank
    
    print(f"""
    Original Linear: y = Wx
    - W: [{d}, {k}] = {d * k:,} parameters
    
    LoRA Adaptation: y = Wx + BAx
    - W: [{d}, {k}] = {d * k:,} parameters (FROZEN)
    - B: [{d}, {r}] = {d * r:,} parameters (trainable)
    - A: [{r}, {k}] = {r * k:,} parameters (trainable)
    
    Total trainable: {d * r + r * k:,} ({(d * r + r * k) / (d * k) * 100:.2f}% of original)
    
    Why it works:
    - Weight updates during fine-tuning are often low-rank
    - ∆W = BA captures the essential update in low-rank form
    - Empirically: rank 8-64 works well for most tasks
    """)


def demonstrate_lora_forward():
    """Demonstrate LoRA forward pass."""
    
    print("\n" + "=" * 60)
    print("LoRA FORWARD PASS")
    print("=" * 60)
    
    # Create LoRA linear layer
    in_features = 768
    out_features = 768
    r = 16
    lora_alpha = 32
    
    lora_layer = LoRALinear(in_features, out_features, r=r, lora_alpha=lora_alpha)
    
    params = count_parameters(lora_layer)
    
    print(f"\nLoRALinear({in_features}, {out_features}, r={r})")
    print(f"  Total parameters: {params['total']:,}")
    print(f"  Trainable: {params['trainable']:,} ({params['trainable_percent']:.2f}%)")
    print(f"  Frozen: {params['frozen']:,}")
    
    # Forward pass
    x = torch.randn(2, 32, in_features)
    
    with torch.no_grad():
        output = lora_layer(x)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    
    # Demonstrate merging
    print("\n" + "-" * 40)
    print("Weight Merging")
    print("-" * 40)
    
    pre_merge_weight = lora_layer.linear.weight.clone()
    lora_layer.merge()
    post_merge_weight = lora_layer.linear.weight.clone()
    
    weight_diff = (post_merge_weight - pre_merge_weight).abs().mean()
    print(f"Weight change after merge: {weight_diff:.6f}")
    
    # After merging, forward is just a normal linear
    with torch.no_grad():
        merged_output = lora_layer(x)
    
    output_diff = (output - merged_output).abs().max()
    print(f"Output difference after merge: {output_diff:.6f}")


def demonstrate_lora_on_transformer():
    """Demonstrate applying LoRA to a transformer."""
    
    print("\n" + "=" * 60)
    print("LoRA ON TRANSFORMER")
    print("=" * 60)
    
    # Create model
    model = SimpleTransformer(hidden_size=768, num_heads=12, num_layers=6)
    
    original_params = count_parameters(model)
    print(f"\nOriginal model:")
    print(f"  Total parameters: {original_params['total']:,}")
    
    # Apply LoRA to attention projections
    target_modules = ['q_proj', 'v_proj']
    
    model = apply_lora_to_model(
        model, 
        target_modules=target_modules,
        r=16,
        lora_alpha=32
    )
    
    lora_params = count_parameters(model)
    print(f"\nAfter LoRA (target: {target_modules}):")
    print(f"  Total parameters: {lora_params['total']:,}")
    print(f"  Trainable: {lora_params['trainable']:,} ({lora_params['trainable_percent']:.2f}%)")
    print(f"  Frozen: {lora_params['frozen']:,}")
    
    # Also apply to FFN
    print("\n" + "-" * 40)
    
    model2 = SimpleTransformer(hidden_size=768, num_heads=12, num_layers=6)
    target_modules_full = ['q_proj', 'v_proj', 'k_proj', 'o_proj', 'up_proj', 'down_proj']
    
    model2 = apply_lora_to_model(
        model2,
        target_modules=target_modules_full,
        r=16,
        lora_alpha=32
    )
    
    full_lora_params = count_parameters(model2)
    print(f"LoRA on all projections ({target_modules_full}):")
    print(f"  Total parameters: {full_lora_params['total']:,}")
    print(f"  Trainable: {full_lora_params['trainable']:,} ({full_lora_params['trainable_percent']:.2f}%)")


def using_peft_library():
    """Show how to use the PEFT library."""
    
    print("\n" + "=" * 60)
    print("USING PEFT LIBRARY")
    print("=" * 60)
    
    code_example = '''
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, TaskType

# 1. Load base model
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# 2. Configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,                          # Rank
    lora_alpha=32,                 # Scaling
    lora_dropout=0.1,              # Dropout
    target_modules=[               # Which modules to adapt
        "q_proj",
        "v_proj",
        "k_proj",
        "o_proj",
        "up_proj",
        "down_proj"
    ],
    bias="none"                    # Don't train biases
)

# 3. Create PEFT model
peft_model = get_peft_model(model, lora_config)
peft_model.print_trainable_parameters()
# Output: trainable params: 4,194,304 || all params: 6,742,609,920 || trainable%: 0.0622

# 4. Train as usual
from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="./lora-llama",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    fp16=True,
    save_strategy="epoch",
)

trainer = Trainer(
    model=peft_model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
)

trainer.train()

# 5. Save adapter (small file!)
peft_model.save_pretrained("./lora-adapter")

# 6. Load and merge for inference
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
peft_model = PeftModel.from_pretrained(base_model, "./lora-adapter")

# Merge weights (no inference overhead)
merged_model = peft_model.merge_and_unload()

# Or keep separate (switch adapters dynamically)
output = peft_model.generate(**inputs)
'''
    
    print(code_example)


def main():
    """Main demonstration of LoRA."""
    
    print("\n" + "=" * 60)
    print("   MODULE 08: LoRA (Low-Rank Adaptation)")
    print("=" * 60)
    
    # 1. LoRA math
    demonstrate_lora_math()
    
    # 2. LoRA forward pass
    demonstrate_lora_forward()
    
    # 3. LoRA on transformer
    demonstrate_lora_on_transformer()
    
    # 4. PEFT library usage
    using_peft_library()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    LoRA KEY POINTS:
    
    1. CORE IDEA:
       - W' = W + BA (low-rank update)
       - Freeze W, only train B and A
       - 0.1-1% trainable parameters
    
    2. CONFIGURATION:
       - rank r: 8-64 typical (16 is good default)
       - lora_alpha: scaling factor (often 2*r)
       - target_modules: usually attention projections
    
    3. ADVANTAGES:
       - Massive memory savings
       - Fast training
       - Small checkpoints (~10-100 MB)
       - Can merge for inference (no overhead)
    
    4. BEST PRACTICES:
       - Start with r=16, target q_proj and v_proj
       - Add more modules if needed
       - Use higher r for complex tasks
       - Merge weights for deployment
    
    5. VARIATIONS:
       - QLoRA: LoRA + 4-bit quantization
       - LoRA+: Improved initialization
       - DoRA: Decomposed weight update
    """)
    print("=" * 60)


if __name__ == "__main__":
    main()


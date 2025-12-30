"""
Module 08: QLoRA (Quantized LoRA)
=================================

QLoRA enables fine-tuning of large language models on consumer hardware
by combining 4-bit quantization with LoRA.

Key innovations:
- 4-bit NormalFloat (NF4) quantization
- Double quantization
- Paged optimizers
"""

import torch
import torch.nn as nn
from typing import Dict, Optional
import math


def demonstrate_qlora_concept():
    """Explain the QLoRA concept."""
    
    print("\n" + "=" * 60)
    print("QLoRA CONCEPT")
    print("=" * 60)
    
    concept = """
    QLoRA: Quantized Low-Rank Adaptation
    
    ┌─────────────────────────────────────────────────────────┐
    │                    QLoRA Architecture                    │
    ├─────────────────────────────────────────────────────────┤
    │                                                          │
    │    ┌─────────────────┐    ┌─────────────────────────┐   │
    │    │   Base Model    │    │     LoRA Adapters       │   │
    │    │                 │    │                         │   │
    │    │  4-bit (NF4)    │  + │  16-bit (BF16/FP16)    │   │
    │    │   Quantized     │    │     Full precision     │   │
    │    │                 │    │                         │   │
    │    │   ~3.5 GB       │    │      ~100 MB           │   │
    │    │  (for 7B)       │    │    (trainable)         │   │
    │    └─────────────────┘    └─────────────────────────┘   │
    │                                                          │
    │    Computation: BF16 (dequantize on-the-fly)            │
    │    Gradients: Only for LoRA parameters                  │
    │    Optimizer states: Only for LoRA parameters           │
    │                                                          │
    └─────────────────────────────────────────────────────────┘
    
    Memory Comparison (LLaMA-7B):
    ┌────────────────────┬──────────────┐
    │ Method             │ GPU Memory   │
    ├────────────────────┼──────────────┤
    │ Full fine-tuning   │ ~56 GB       │
    │ LoRA (FP16)        │ ~16 GB       │
    │ QLoRA (4-bit)      │ ~6 GB        │
    └────────────────────┴──────────────┘
    """
    print(concept)


def explain_nf4_quantization():
    """Explain NF4 quantization used in QLoRA."""
    
    print("\n" + "=" * 60)
    print("NF4 (4-bit NormalFloat) QUANTIZATION")
    print("=" * 60)
    
    explanation = """
    Why NF4 instead of regular INT4?
    
    Neural network weights follow a normal distribution (roughly).
    NF4 places quantization levels to minimize error for normally
    distributed values.
    
    Regular INT4:  Uniform spacing between -8 and 7
                   [-8, -7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]
    
    NF4:           Non-uniform, optimized for N(0,1)
                   More levels near 0 (where most weights are)
    
    ┌─────────────────────────────────────────────────────────┐
    │               Weight Distribution                        │
    │                                                          │
    │                        ████                              │
    │                      ████████                            │
    │                    ████████████                          │
    │                  ████████████████                        │
    │                ████████████████████                      │
    │              ████████████████████████                    │
    │            ████████████████████████████                  │
    │          ████████████████████████████████                │
    │    ─────────────────────────────────────────────         │
    │    -3σ    -2σ    -1σ     0     1σ     2σ    3σ          │
    │                                                          │
    │    NF4: More quantization levels in the center          │
    │    INT4: Uniform levels waste precision at extremes     │
    └─────────────────────────────────────────────────────────┘
    """
    print(explanation)
    
    # Show NF4 quantization levels (normalized)
    nf4_levels = torch.tensor([
        -1.0, -0.6962, -0.5251, -0.3949, -0.2844, -0.1848, -0.0911, 0.0,
        0.0796, 0.1609, 0.2461, 0.3379, 0.4407, 0.5626, 0.7230, 1.0
    ])
    
    print("\nNF4 Quantization Levels (normalized to [-1, 1]):")
    for i, level in enumerate(nf4_levels):
        bar_pos = int((level + 1) * 25)
        bar = ' ' * bar_pos + '█'
        print(f"  {i:2d}: {level:+.4f} |{bar}")


def explain_double_quantization():
    """Explain double quantization in QLoRA."""
    
    print("\n" + "=" * 60)
    print("DOUBLE QUANTIZATION")
    print("=" * 60)
    
    explanation = """
    Problem: Storing quantization constants (scales) adds memory overhead.
    
    Single Quantization:
    - Weights: 4-bit
    - Scales: FP32 (one per block of 64 weights)
    - Overhead: 32 bits / 64 = 0.5 bits per weight
    
    Double Quantization:
    - Weights: 4-bit
    - Scales: 8-bit quantized (with their own FP32 scale)
    - Overhead: 8 bits / 64 + 32 / 256 = 0.125 + 0.125 = 0.25 bits per weight
    
    Memory savings: ~0.37 GB for a 65B parameter model
    
    ┌─────────────────────────────────────────────────────────┐
    │                  Double Quantization                     │
    │                                                          │
    │   Level 1: Quantize weights (64 weights per block)      │
    │   ┌────────────────────────────────────────────────┐    │
    │   │ W1 W2 W3 ... W64 │ scale_1 (FP32)              │    │
    │   │ W65 W66 ... W128 │ scale_2 (FP32)              │    │
    │   │ ...              │ ...                          │    │
    │   └────────────────────────────────────────────────┘    │
    │                                                          │
    │   Level 2: Quantize the scales (256 scales per block)   │
    │   ┌────────────────────────────────────────────────┐    │
    │   │ s1 s2 ... s256 (8-bit) │ meta_scale (FP32)     │    │
    │   └────────────────────────────────────────────────┘    │
    │                                                          │
    └─────────────────────────────────────────────────────────┘
    """
    print(explanation)


def calculate_memory_requirements():
    """Calculate memory requirements for different approaches."""
    
    print("\n" + "=" * 60)
    print("MEMORY REQUIREMENTS COMPARISON")
    print("=" * 60)
    
    models = {
        "LLaMA-7B": 7e9,
        "LLaMA-13B": 13e9,
        "LLaMA-33B": 33e9,
        "LLaMA-65B": 65e9,
    }
    
    print(f"\n{'Model':<15} {'Full FT':<12} {'LoRA FP16':<12} {'QLoRA 4-bit':<12}")
    print("-" * 55)
    
    for name, params in models.items():
        # Full fine-tuning: model (FP32) + optimizer states (2x for Adam) + gradients
        # Roughly 20 bytes per parameter
        full_ft_gb = params * 20 / 1e9
        
        # LoRA FP16: model (FP16) + small trainable params
        # Roughly 2 bytes per param + small overhead
        lora_fp16_gb = params * 2 / 1e9 + 0.5  # +0.5 for optimizer
        
        # QLoRA 4-bit: model (4-bit) + LoRA adapters (FP16) + optimizer
        # 0.5 bytes per param + small overhead
        qlora_gb = params * 0.5 / 1e9 + 1  # +1 for LoRA + optimizer
        
        print(f"{name:<15} {full_ft_gb:>10.1f} GB {lora_fp16_gb:>10.1f} GB {qlora_gb:>10.1f} GB")
    
    print("\n* Estimates; actual may vary based on batch size, sequence length, etc.")


def qlora_training_example():
    """Show QLoRA training code example."""
    
    print("\n" + "=" * 60)
    print("QLoRA TRAINING EXAMPLE")
    print("=" * 60)
    
    code_example = '''
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# 1. Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # Use 4-bit quantization
    bnb_4bit_quant_type="nf4",            # NF4 quantization type
    bnb_4bit_compute_dtype=torch.bfloat16, # Compute in BF16
    bnb_4bit_use_double_quant=True,       # Double quantization
)

# 2. Load model in 4-bit
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

# 3. Prepare model for k-bit training
model = prepare_model_for_kbit_training(model)

# 4. Configure LoRA
lora_config = LoraConfig(
    r=64,                      # Higher rank for QLoRA
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

# 5. Apply LoRA
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Output: trainable params: 33,554,432 || all params: 6,771,970,048 || trainable%: 0.4957

# 6. Training arguments
training_args = TrainingArguments(
    output_dir="./qlora-llama",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    gradient_checkpointing=True,       # Save memory
    optim="paged_adamw_32bit",         # Paged optimizer
    logging_steps=10,
    learning_rate=2e-4,
    bf16=True,                         # BF16 training
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    save_strategy="epoch",
)

# 7. Train with SFTTrainer (Supervised Fine-Tuning)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    peft_config=lora_config,
    tokenizer=tokenizer,
    args=training_args,
    dataset_text_field="text",
    max_seq_length=512,
)

trainer.train()

# 8. Save
model.save_pretrained("./qlora-adapter")

# 9. Inference
from peft import AutoPeftModelForCausalLM

model = AutoPeftModelForCausalLM.from_pretrained(
    "./qlora-adapter",
    load_in_4bit=True,
    device_map="auto",
)

# Generate
output = model.generate(**tokenizer("Hello", return_tensors="pt"), max_new_tokens=50)
print(tokenizer.decode(output[0]))
'''
    
    print(code_example)


def compare_lora_vs_qlora():
    """Compare LoRA and QLoRA."""
    
    print("\n" + "=" * 60)
    print("LoRA vs QLoRA COMPARISON")
    print("=" * 60)
    
    comparison = """
    ┌────────────────────────┬─────────────────────┬─────────────────────┐
    │ Aspect                 │ LoRA                │ QLoRA               │
    ├────────────────────────┼─────────────────────┼─────────────────────┤
    │ Base Model Precision   │ FP16/BF16 (16-bit)  │ NF4 (4-bit)         │
    ├────────────────────────┼─────────────────────┼─────────────────────┤
    │ Adapter Precision      │ FP16/BF16           │ FP16/BF16           │
    ├────────────────────────┼─────────────────────┼─────────────────────┤
    │ Compute Precision      │ FP16/BF16           │ BF16                │
    ├────────────────────────┼─────────────────────┼─────────────────────┤
    │ Memory (LLaMA-7B)      │ ~16 GB              │ ~6 GB               │
    ├────────────────────────┼─────────────────────┼─────────────────────┤
    │ Memory (LLaMA-65B)     │ ~120 GB             │ ~48 GB              │
    ├────────────────────────┼─────────────────────┼─────────────────────┤
    │ Training Speed         │ Faster              │ Slightly slower     │
    │                        │                     │ (dequant overhead)  │
    ├────────────────────────┼─────────────────────┼─────────────────────┤
    │ Quality                │ Best                │ Very close          │
    │                        │                     │ (~1% gap typically) │
    ├────────────────────────┼─────────────────────┼─────────────────────┤
    │ Hardware Required      │ A100 40GB           │ RTX 3090 24GB       │
    │ (for 7B model)         │ or similar          │ or similar          │
    ├────────────────────────┼─────────────────────┼─────────────────────┤
    │ Best Use Case          │ When you have       │ Consumer hardware   │
    │                        │ enough memory       │ or very large models│
    └────────────────────────┴─────────────────────┴─────────────────────┘
    
    Recommendation:
    - Have A100 80GB? → Use LoRA for fastest training
    - Have RTX 3090/4090? → Use QLoRA for 7B-13B models
    - Have RTX 3080/4080? → Use QLoRA for 7B models
    - Want to train 70B? → Use QLoRA on A100 or multi-GPU
    """
    print(comparison)


def main():
    """Main demonstration of QLoRA."""
    
    print("\n" + "=" * 60)
    print("   MODULE 08: QLoRA (Quantized LoRA)")
    print("=" * 60)
    
    # 1. QLoRA concept
    demonstrate_qlora_concept()
    
    # 2. NF4 explanation
    explain_nf4_quantization()
    
    # 3. Double quantization
    explain_double_quantization()
    
    # 4. Memory requirements
    calculate_memory_requirements()
    
    # 5. Training example
    qlora_training_example()
    
    # 6. Comparison
    compare_lora_vs_qlora()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    QLoRA KEY POINTS:
    
    1. CORE INNOVATION:
       - 4-bit NF4 quantization for base model
       - FP16 LoRA adapters for training
       - Double quantization for memory efficiency
    
    2. MEMORY SAVINGS:
       - ~4x reduction vs LoRA
       - Train 7B on 24GB GPU
       - Train 65B on 48GB GPU
    
    3. QUALITY:
       - Very close to full fine-tuning
       - ~1% gap in most benchmarks
       - No quality loss vs LoRA (higher rank compensates)
    
    4. BEST PRACTICES:
       - Use higher rank (r=64) for QLoRA
       - Enable gradient checkpointing
       - Use paged optimizers
       - BF16 compute dtype
    
    5. WHEN TO USE:
       - Limited GPU memory
       - Fine-tuning large models (13B+)
       - Consumer hardware (RTX 3090/4090)
    """)
    print("=" * 60)


if __name__ == "__main__":
    main()


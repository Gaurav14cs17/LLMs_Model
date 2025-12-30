"""
Module 04: Knowledge Distillation
=================================

This script demonstrates knowledge distillation techniques:
- Basic response-based distillation
- Feature/hidden state distillation
- Attention transfer
- Temperature scaling effects
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import math


class TeacherModel(nn.Module):
    """Large teacher model (simulating a larger LLM)."""
    
    def __init__(self, vocab_size: int = 10000, hidden_size: int = 768,
                 num_layers: int = 12, num_heads: int = 12):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size, 
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.output = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
    
    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        hidden_states = []
        
        x = self.embedding(x)
        hidden_states.append(x)
        
        for layer in self.layers:
            x = layer(x)
            hidden_states.append(x)
        
        logits = self.output(x)
        
        if return_hidden:
            return logits, hidden_states
        return logits


class StudentModel(nn.Module):
    """Smaller student model."""
    
    def __init__(self, vocab_size: int = 10000, hidden_size: int = 384,
                 num_layers: int = 6, num_heads: int = 6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size,
                nhead=num_heads,
                dim_feedforward=hidden_size * 4,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.output = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
    
    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        hidden_states = []
        
        x = self.embedding(x)
        hidden_states.append(x)
        
        for layer in self.layers:
            x = layer(x)
            hidden_states.append(x)
        
        logits = self.output(x)
        
        if return_hidden:
            return logits, hidden_states
        return logits


def softmax_with_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply softmax with temperature scaling."""
    return F.softmax(logits / temperature, dim=-1)


def distillation_loss(student_logits: torch.Tensor,
                      teacher_logits: torch.Tensor,
                      labels: torch.Tensor,
                      temperature: float = 4.0,
                      alpha: float = 0.5) -> Dict[str, torch.Tensor]:
    """
    Compute knowledge distillation loss.
    
    L = α * L_soft + (1 - α) * L_hard
    
    Args:
        student_logits: Student model output
        teacher_logits: Teacher model output (detached)
        labels: Ground truth labels
        temperature: Temperature for softening
        alpha: Weight for soft loss
    
    Returns:
        Dictionary with loss components
    """
    # Soft targets from teacher
    soft_targets = F.softmax(teacher_logits / temperature, dim=-1)
    
    # Soft predictions from student
    soft_predictions = F.log_softmax(student_logits / temperature, dim=-1)
    
    # KL divergence loss (soft loss)
    # Multiply by T^2 to maintain gradient magnitude
    soft_loss = F.kl_div(
        soft_predictions, 
        soft_targets, 
        reduction='batchmean'
    ) * (temperature ** 2)
    
    # Hard loss (cross-entropy with true labels)
    hard_loss = F.cross_entropy(
        student_logits.view(-1, student_logits.size(-1)),
        labels.view(-1)
    )
    
    # Combined loss
    total_loss = alpha * soft_loss + (1 - alpha) * hard_loss
    
    return {
        'total': total_loss,
        'soft': soft_loss,
        'hard': hard_loss
    }


def hidden_state_loss(student_hidden: List[torch.Tensor],
                      teacher_hidden: List[torch.Tensor],
                      projector: Optional[nn.Module] = None) -> torch.Tensor:
    """
    Compute hidden state distillation loss.
    
    Matches intermediate representations between teacher and student.
    """
    total_loss = 0.0
    
    # Map student layers to teacher layers
    # Student has fewer layers, so we sample from teacher
    student_layers = len(student_hidden)
    teacher_layers = len(teacher_hidden)
    
    # Layer mapping: evenly sample from teacher
    layer_mapping = [
        int(i * (teacher_layers - 1) / (student_layers - 1))
        for i in range(student_layers)
    ]
    
    for s_idx, t_idx in enumerate(layer_mapping):
        s_hidden = student_hidden[s_idx]
        t_hidden = teacher_hidden[t_idx]
        
        # Project student hidden to teacher dimension if needed
        if projector is not None:
            s_hidden = projector(s_hidden)
        
        # MSE loss between hidden states
        loss = F.mse_loss(s_hidden, t_hidden)
        total_loss += loss
    
    return total_loss / len(layer_mapping)


class HiddenStateProjector(nn.Module):
    """Projects student hidden states to teacher dimension."""
    
    def __init__(self, student_dim: int, teacher_dim: int):
        super().__init__()
        self.projector = nn.Linear(student_dim, teacher_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.projector(x)


def demonstrate_temperature_effect():
    """Visualize the effect of temperature on softmax distribution."""
    
    print("\n" + "=" * 60)
    print("TEMPERATURE EFFECT ON SOFTMAX")
    print("=" * 60)
    
    # Sample logits (imagine 5 classes)
    logits = torch.tensor([2.0, 1.5, 0.5, -0.5, -1.0])
    
    print(f"\nRaw logits: {logits.tolist()}")
    print(f"\n{'Temperature':>12} | Distribution")
    print("-" * 60)
    
    temperatures = [0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
    
    for temp in temperatures:
        probs = softmax_with_temperature(logits, temp)
        
        # Visual bar
        bars = ['█' * int(p * 30) for p in probs]
        prob_str = [f"{p:.3f}" for p in probs.tolist()]
        
        print(f"\nT = {temp:>5.1f}")
        for i, (prob, bar) in enumerate(zip(prob_str, bars)):
            print(f"  Class {i}: {prob} {bar}")
    
    print(f"\n→ Higher temperature = softer distribution")
    print(f"→ Lower temperature = sharper (more confident)")


def demonstrate_distillation_training():
    """Demonstrate a distillation training step."""
    
    print("\n" + "=" * 60)
    print("DISTILLATION TRAINING DEMONSTRATION")
    print("=" * 60)
    
    # Create models
    teacher = TeacherModel(vocab_size=1000, hidden_size=512, num_layers=6)
    student = StudentModel(vocab_size=1000, hidden_size=256, num_layers=3)
    
    # Count parameters
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    
    print(f"\nTeacher: {teacher_params:,} parameters")
    print(f"Student: {student_params:,} parameters")
    print(f"Compression: {teacher_params / student_params:.1f}x")
    
    # Freeze teacher
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    # Create projector for hidden state distillation
    projector = HiddenStateProjector(256, 512)
    
    # Sample data
    batch_size, seq_len = 4, 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    labels = torch.randint(0, 1000, (batch_size, seq_len))
    
    # Forward pass
    with torch.no_grad():
        teacher_logits, teacher_hidden = teacher(input_ids, return_hidden=True)
    
    student_logits, student_hidden = student(input_ids, return_hidden=True)
    
    # Compute losses
    kd_losses = distillation_loss(
        student_logits, teacher_logits, labels,
        temperature=4.0, alpha=0.7
    )
    
    hidden_loss = hidden_state_loss(student_hidden, teacher_hidden, projector)
    
    # Total loss
    total_loss = kd_losses['total'] + 0.5 * hidden_loss
    
    print(f"\nLoss components:")
    print(f"  Soft loss (KD): {kd_losses['soft']:.4f}")
    print(f"  Hard loss (CE): {kd_losses['hard']:.4f}")
    print(f"  Hidden loss:    {hidden_loss:.4f}")
    print(f"  Total loss:     {total_loss:.4f}")


def compare_distillation_strategies():
    """Compare different distillation strategies."""
    
    print("\n" + "=" * 60)
    print("DISTILLATION STRATEGIES COMPARISON")
    print("=" * 60)
    
    strategies = """
    ┌─────────────────────────────────────────────────────────────┐
    │                 DISTILLATION STRATEGIES                      │
    ├─────────────────────────────────────────────────────────────┤
    │                                                              │
    │  1. RESPONSE-BASED (Logit Distillation)                     │
    │     └─ Transfer: Final output probabilities                  │
    │     └─ Loss: KL divergence on softmax outputs               │
    │     └─ Simple, effective, most common                       │
    │                                                              │
    │  2. FEATURE-BASED (Hidden State Distillation)               │
    │     └─ Transfer: Intermediate layer representations         │
    │     └─ Loss: MSE between hidden states                      │
    │     └─ Requires dimension matching (projector)              │
    │     └─ Example: TinyBERT, MiniLM                            │
    │                                                              │
    │  3. ATTENTION-BASED                                          │
    │     └─ Transfer: Attention maps/patterns                    │
    │     └─ Loss: MSE/KL on attention weights                    │
    │     └─ Preserves "what to focus on"                         │
    │                                                              │
    │  4. RELATION-BASED                                           │
    │     └─ Transfer: Relationships between samples              │
    │     └─ Loss: Match similarity matrices                      │
    │     └─ Example: Relational Knowledge Distillation           │
    │                                                              │
    └─────────────────────────────────────────────────────────────┘
    """
    print(strategies)


def demonstrate_progressive_distillation():
    """Demonstrate progressive knowledge distillation."""
    
    print("\n" + "=" * 60)
    print("PROGRESSIVE DISTILLATION")
    print("=" * 60)
    
    print("""
    When the gap between teacher and student is large,
    use intermediate "teaching assistant" models:
    
    Teacher (7B) → TA (3B) → TA (1B) → Student (350M)
    
    Each step:
    - Smaller compression ratio
    - More effective knowledge transfer
    - Better final student performance
    """)
    
    sizes = [7000, 3000, 1000, 350]  # Million parameters
    
    print(f"\nProgressive distillation chain:")
    for i in range(len(sizes) - 1):
        teacher_size = sizes[i]
        student_size = sizes[i + 1]
        ratio = teacher_size / student_size
        print(f"  {teacher_size}M → {student_size}M ({ratio:.1f}x compression)")
    
    total_compression = sizes[0] / sizes[-1]
    print(f"\nTotal compression: {total_compression:.1f}x")
    print(f"Each step: ~{(total_compression ** (1/3)):.1f}x average")


def using_huggingface_distillation():
    """Show how to use Hugging Face for distillation."""
    
    print("\n" + "=" * 60)
    print("HUGGING FACE DISTILLATION EXAMPLE")
    print("=" * 60)
    
    code_example = '''
# Using the transformers library for distillation

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)
import torch
import torch.nn.functional as F

class DistillationTrainer(Trainer):
    def __init__(self, teacher_model, temperature=4.0, alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.teacher_model = teacher_model
        self.temperature = temperature
        self.alpha = alpha
        
        # Freeze teacher
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
    
    def compute_loss(self, model, inputs, return_outputs=False):
        # Student forward
        outputs = model(**inputs)
        student_logits = outputs.logits
        
        # Teacher forward
        with torch.no_grad():
            teacher_outputs = self.teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits
        
        # Distillation loss
        soft_targets = F.softmax(teacher_logits / self.temperature, dim=-1)
        soft_predictions = F.log_softmax(student_logits / self.temperature, dim=-1)
        soft_loss = F.kl_div(soft_predictions, soft_targets, reduction='batchmean')
        soft_loss = soft_loss * (self.temperature ** 2)
        
        # Hard loss
        hard_loss = outputs.loss  # Standard LM loss
        
        # Combined
        loss = self.alpha * soft_loss + (1 - self.alpha) * hard_loss
        
        return (loss, outputs) if return_outputs else loss

# Usage
teacher = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
student = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B")

trainer = DistillationTrainer(
    teacher_model=teacher,
    temperature=4.0,
    alpha=0.7,
    model=student,
    args=TrainingArguments(
        output_dir="./distilled_model",
        num_train_epochs=3,
        per_device_train_batch_size=4,
        learning_rate=5e-5,
    ),
    train_dataset=train_dataset,
)

trainer.train()
'''
    
    print(code_example)


def main():
    """Main demonstration of knowledge distillation."""
    
    print("\n" + "=" * 60)
    print("   MODULE 04: KNOWLEDGE DISTILLATION")
    print("=" * 60)
    
    # 1. Temperature effect
    demonstrate_temperature_effect()
    
    # 2. Distillation training
    demonstrate_distillation_training()
    
    # 3. Strategies comparison
    compare_distillation_strategies()
    
    # 4. Progressive distillation
    demonstrate_progressive_distillation()
    
    # 5. Hugging Face example
    using_huggingface_distillation()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    KEY TAKEAWAYS:
    
    1. TEMPERATURE SCALING:
       - Higher T (4-8) softens distribution
       - Reveals "dark knowledge" (class relationships)
       - Scale loss by T² to maintain gradients
    
    2. LOSS COMPONENTS:
       - Soft loss: KL div with teacher outputs
       - Hard loss: CE with true labels
       - α balances the two (typically 0.5-0.9)
    
    3. DISTILLATION TYPES:
       - Response-based: Output logits (simple)
       - Feature-based: Hidden states (powerful)
       - Attention-based: Focus patterns
    
    4. PRACTICAL TIPS:
       - Use teacher's training data or similar
       - Progressive distillation for large gaps
       - Combine with other compression methods
       - DistilBERT achieves 97% with 40% size
    
    5. FOR LLMs:
       - Works well for task-specific distillation
       - Harder for general capabilities
       - Often combined with quantization
    """)
    print("=" * 60)


if __name__ == "__main__":
    main()


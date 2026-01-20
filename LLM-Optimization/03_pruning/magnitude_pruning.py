"""
Module 03: Magnitude Pruning
============================

This script demonstrates magnitude-based pruning techniques:
- Global magnitude pruning
- Local (layer-wise) magnitude pruning
- Iterative pruning
- Lottery ticket hypothesis
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import copy
import time


class SimpleClassifier(nn.Module):
    """A simple neural network for pruning demonstration."""
    
    def __init__(self, input_size: int = 784, hidden_sizes: List[int] = [512, 256, 128],
                 num_classes: int = 10):
        super().__init__()
        
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.BatchNorm1d(hidden_size)
            ])
            prev_size = hidden_size
        
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(prev_size, num_classes)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.features(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> Dict[str, int]:
    """Count total and non-zero parameters."""
    total = 0
    nonzero = 0
    
    for param in model.parameters():
        total += param.numel()
        nonzero += (param != 0).sum().item()
    
    return {
        "total": total,
        "nonzero": nonzero,
        "sparsity": 1 - nonzero / total
    }


def global_magnitude_pruning(model: nn.Module, sparsity: float) -> nn.Module:
    """
    Apply global magnitude pruning.
    
    All weights across all layers are ranked together,
    and the smallest ones are pruned.
    """
    model = copy.deepcopy(model)
    
    # Collect all weights
    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            all_weights.append(param.data.abs().view(-1))
    
    all_weights = torch.cat(all_weights)
    
    # Find threshold
    k = int(sparsity * len(all_weights))
    if k > 0:
        threshold = torch.kthvalue(all_weights, k).values.item()
    else:
        threshold = 0
    
    # Apply pruning
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            mask = param.data.abs() > threshold
            param.data *= mask.float()
    
    return model


def local_magnitude_pruning(model: nn.Module, sparsity: float) -> nn.Module:
    """
    Apply local (layer-wise) magnitude pruning.
    
    Each layer is pruned independently to the target sparsity.
    """
    model = copy.deepcopy(model)
    
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            weights = param.data.abs().view(-1)
            k = int(sparsity * len(weights))
            
            if k > 0:
                threshold = torch.kthvalue(weights, k).values.item()
                mask = param.data.abs() > threshold
                param.data *= mask.float()
    
    return model


def iterative_pruning(model: nn.Module, 
                      target_sparsity: float,
                      num_iterations: int = 5,
                      fine_tune_fn: Optional[callable] = None) -> nn.Module:
    """
    Apply iterative magnitude pruning.
    
    Gradually increase sparsity over multiple iterations,
    optionally fine-tuning between iterations.
    """
    model = copy.deepcopy(model)
    current_sparsity = 0
    
    for i in range(num_iterations):
        # Calculate sparsity for this iteration (cubic schedule)
        progress = (i + 1) / num_iterations
        current_sparsity = target_sparsity * (1 - (1 - progress) ** 3)
        
        # Prune
        model = global_magnitude_pruning(model, current_sparsity)
        
        # Fine-tune if provided
        if fine_tune_fn is not None:
            model = fine_tune_fn(model)
        
        stats = count_parameters(model)
        print(f"  Iteration {i+1}: Sparsity = {stats['sparsity']:.1%}")
    
    return model


class LotteryTicketExperiment:
    """
    Implements the Lottery Ticket Hypothesis experiment.
    
    The hypothesis: Dense networks contain sparse subnetworks that,
    when trained in isolation from their original initialization,
    can match the full network's performance.
    """
    
    def __init__(self, model: nn.Module):
        # Save initial weights
        self.initial_weights = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }
        self.model = model
    
    def find_winning_ticket(self, 
                            trained_model: nn.Module,
                            sparsity: float) -> Tuple[nn.Module, Dict[str, torch.Tensor]]:
        """
        Find a winning ticket (sparse subnetwork).
        
        1. Use trained weights to determine pruning mask
        2. Apply mask to initial weights
        """
        # Get pruning masks from trained model
        masks = {}
        all_weights = []
        
        for name, param in trained_model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                all_weights.append(param.data.abs().view(-1))
        
        all_weights = torch.cat(all_weights)
        k = int(sparsity * len(all_weights))
        threshold = torch.kthvalue(all_weights, k).values.item() if k > 0 else 0
        
        for name, param in trained_model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                masks[name] = (param.data.abs() > threshold).float()
        
        # Create winning ticket: initial weights with masks
        winning_ticket = copy.deepcopy(self.model)
        for name, param in winning_ticket.named_parameters():
            if name in masks:
                param.data = self.initial_weights[name] * masks[name]
            else:
                param.data = self.initial_weights[name].clone()
        
        return winning_ticket, masks


def demonstrate_magnitude_pruning():
    """Demonstrate different magnitude pruning approaches."""
    
    print("\n" + "=" * 60)
    print("MAGNITUDE PRUNING DEMONSTRATION")
    print("=" * 60)
    
    # Create model
    model = SimpleClassifier()
    original_stats = count_parameters(model)
    
    print(f"\nOriginal model:")
    print(f"  Total parameters: {original_stats['total']:,}")
    print(f"  Non-zero parameters: {original_stats['nonzero']:,}")
    
    # Test different sparsity levels
    sparsity_levels = [0.5, 0.7, 0.9, 0.95]
    
    print("\n" + "-" * 60)
    print("GLOBAL vs LOCAL PRUNING")
    print("-" * 60)
    
    print(f"\n{'Sparsity':>10} | {'Global (actual)':>15} | {'Local (actual)':>15}")
    print("-" * 50)
    
    for target_sparsity in sparsity_levels:
        global_model = global_magnitude_pruning(model, target_sparsity)
        local_model = local_magnitude_pruning(model, target_sparsity)
        
        global_stats = count_parameters(global_model)
        local_stats = count_parameters(local_model)
        
        print(f"{target_sparsity:>10.0%} | {global_stats['sparsity']:>15.1%} | "
              f"{local_stats['sparsity']:>15.1%}")


def demonstrate_iterative_pruning():
    """Demonstrate iterative pruning with cubic schedule."""
    
    print("\n" + "=" * 60)
    print("ITERATIVE PRUNING (CUBIC SCHEDULE)")
    print("=" * 60)
    
    model = SimpleClassifier()
    
    print("\nPruning to 90% sparsity over 5 iterations:")
    pruned_model = iterative_pruning(model, target_sparsity=0.9, num_iterations=5)
    
    final_stats = count_parameters(pruned_model)
    print(f"\nFinal sparsity: {final_stats['sparsity']:.1%}")


def demonstrate_lottery_ticket():
    """Demonstrate the Lottery Ticket Hypothesis."""
    
    print("\n" + "=" * 60)
    print("LOTTERY TICKET HYPOTHESIS")
    print("=" * 60)
    
    # Create model
    model = SimpleClassifier()
    experiment = LotteryTicketExperiment(model)
    
    # Simulate training (just random weight changes for demo)
    trained_model = copy.deepcopy(model)
    for param in trained_model.parameters():
        param.data += torch.randn_like(param) * 0.1
    
    # Find winning ticket at 80% sparsity
    winning_ticket, masks = experiment.find_winning_ticket(trained_model, sparsity=0.8)
    
    stats = count_parameters(winning_ticket)
    
    print(f"""
    Lottery Ticket Experiment:
    
    1. Saved initial random weights W₀
    2. Trained model to get W_trained
    3. Found pruning mask M at 80% sparsity
    4. Created winning ticket: W₀ * M
    
    Winning ticket statistics:
    - Sparsity: {stats['sparsity']:.1%}
    - Non-zero parameters: {stats['nonzero']:,}
    
    In practice, the winning ticket can be retrained from scratch
    and achieve similar accuracy to the full dense network!
    """)


def analyze_weight_distribution():
    """Analyze weight distribution before and after pruning."""
    
    print("\n" + "=" * 60)
    print("WEIGHT DISTRIBUTION ANALYSIS")
    print("=" * 60)
    
    model = SimpleClassifier()
    
    # Collect all weights
    all_weights = []
    for param in model.parameters():
        if param.dim() >= 2:
            all_weights.append(param.data.view(-1))
    all_weights = torch.cat(all_weights)
    
    # Statistics
    print(f"\nWeight distribution (before pruning):")
    print(f"  Mean: {all_weights.mean():.4f}")
    print(f"  Std: {all_weights.std():.4f}")
    print(f"  Min: {all_weights.min():.4f}")
    print(f"  Max: {all_weights.max():.4f}")
    
    # Percentiles
    percentiles = [10, 25, 50, 75, 90]
    print(f"\n  Percentiles (absolute values):")
    abs_weights = all_weights.abs()
    for p in percentiles:
        value = torch.quantile(abs_weights, p/100).item()
        print(f"    {p}th: {value:.4f}")
    
    # After pruning
    pruned_model = global_magnitude_pruning(model, 0.9)
    
    pruned_weights = []
    for param in pruned_model.parameters():
        if param.dim() >= 2:
            nonzero = param.data[param.data != 0]
            if len(nonzero) > 0:
                pruned_weights.append(nonzero.view(-1))
    pruned_weights = torch.cat(pruned_weights)
    
    print(f"\nWeight distribution (after 90% pruning, non-zero only):")
    print(f"  Mean: {pruned_weights.mean():.4f}")
    print(f"  Std: {pruned_weights.std():.4f}")
    print(f"  Min (abs): {pruned_weights.abs().min():.4f}")
    print(f"  Max: {pruned_weights.max():.4f}")


def pruning_with_pytorch():
    """Show how to use PyTorch's built-in pruning."""
    
    print("\n" + "=" * 60)
    print("PYTORCH BUILT-IN PRUNING")
    print("=" * 60)
    
    code_example = '''
import torch.nn.utils.prune as prune

# Create a simple layer
linear = nn.Linear(100, 50)

# 1. L1 Unstructured Pruning (magnitude-based)
prune.l1_unstructured(linear, name='weight', amount=0.3)
# Creates: linear.weight_mask, linear.weight_orig

# 2. Random Unstructured Pruning
prune.random_unstructured(linear, name='weight', amount=0.3)

# 3. Ln Structured Pruning (prune entire neurons)
prune.ln_structured(linear, name='weight', amount=0.3, n=2, dim=0)

# 4. Global Pruning (across multiple layers)
parameters_to_prune = [
    (model.layer1, 'weight'),
    (model.layer2, 'weight'),
]
prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.5
)

# 5. Make pruning permanent
prune.remove(linear, 'weight')
# Now linear.weight contains the pruned weights

# 6. Check sparsity
sparsity = (linear.weight == 0).sum() / linear.weight.numel()
print(f"Sparsity: {sparsity:.1%}")
'''
    
    print(code_example)
    
    # Actually demonstrate it
    print("\nDemonstration:")
    import torch.nn.utils.prune as prune
    
    linear = nn.Linear(100, 50)
    original_nonzero = (linear.weight != 0).sum().item()
    
    prune.l1_unstructured(linear, name='weight', amount=0.5)
    prune.remove(linear, 'weight')
    
    pruned_nonzero = (linear.weight != 0).sum().item()
    
    print(f"  Original non-zero weights: {original_nonzero}")
    print(f"  After 50% pruning: {pruned_nonzero}")
    print(f"  Actual sparsity: {1 - pruned_nonzero/original_nonzero:.1%}")


def main():
    """Main demonstration of magnitude pruning."""
    
    print("\n" + "=" * 60)
    print("   MODULE 03: MAGNITUDE PRUNING")
    print("=" * 60)
    
    # 1. Basic magnitude pruning
    demonstrate_magnitude_pruning()
    
    # 2. Iterative pruning
    demonstrate_iterative_pruning()
    
    # 3. Lottery ticket
    demonstrate_lottery_ticket()
    
    # 4. Weight distribution analysis
    analyze_weight_distribution()
    
    # 5. PyTorch built-in pruning
    pruning_with_pytorch()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    KEY TAKEAWAYS:
    
    1. MAGNITUDE PRUNING: Remove weights with smallest absolute values
       - Global: All weights compete; consistent threshold
       - Local: Each layer pruned independently
    
    2. ITERATIVE PRUNING: Gradually increase sparsity
       - Better accuracy than one-shot
       - Cubic schedule works well
    
    3. LOTTERY TICKET HYPOTHESIS:
       - Sparse "winning tickets" exist in random init
       - Can match dense network performance
       - Theoretical foundation for pruning
    
    4. PRACTICAL TIPS:
       - Start with 50% sparsity
       - Use iterative pruning for high sparsity
       - Fine-tune after pruning
       - Combine with quantization
    """)
    print("=" * 60)


if __name__ == "__main__":
    main()


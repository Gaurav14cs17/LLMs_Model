"""
Module 02: INT8 Quantization
============================

This script demonstrates various INT8 quantization techniques:
- Dynamic quantization
- Static quantization
- Per-channel vs per-tensor quantization
"""

import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Tuple, Dict
import time


class SimpleTransformerBlock(nn.Module):
    """A simplified transformer block for quantization demonstration."""
    
    def __init__(self, hidden_size: int = 512):
        super().__init__()
        self.linear1 = nn.Linear(hidden_size, hidden_size * 4)
        self.linear2 = nn.Linear(hidden_size * 4, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.activation = nn.GELU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        return x + residual


class SimpleModel(nn.Module):
    """A simple model for quantization demonstration."""
    
    def __init__(self, input_size: int = 512, num_layers: int = 6):
        super().__init__()
        self.layers = nn.ModuleList([
            SimpleTransformerBlock(input_size) for _ in range(num_layers)
        ])
        self.output = nn.Linear(input_size, 1000)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


def get_model_size(model: nn.Module) -> float:
    """Calculate model size in MB."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return (param_size + buffer_size) / (1024 ** 2)


def measure_inference_time(model: nn.Module, input_tensor: torch.Tensor, 
                           num_runs: int = 100) -> float:
    """Measure average inference time."""
    model.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_tensor)
    
    # Measure
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(input_tensor)
    end = time.perf_counter()
    
    return (end - start) / num_runs * 1000  # ms


def dynamic_quantization(model: nn.Module) -> nn.Module:
    """
    Apply dynamic quantization to the model.
    
    Dynamic quantization:
    - Weights are quantized ahead of time
    - Activations are quantized dynamically at runtime
    - Good for RNNs and Transformers
    """
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear},  # Layers to quantize
        dtype=torch.qint8
    )
    return quantized_model


class QuantizableModel(nn.Module):
    """Model with quantization stubs for static quantization."""
    
    def __init__(self, input_size: int = 512, num_layers: int = 6):
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.dequant = torch.quantization.DeQuantStub()
        
        self.layers = nn.Sequential(*[
            nn.Sequential(
                nn.Linear(input_size, input_size * 4),
                nn.ReLU(),  # Use ReLU for quantization compatibility
                nn.Linear(input_size * 4, input_size)
            ) for _ in range(num_layers)
        ])
        self.output = nn.Linear(input_size, 1000)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        for layer in self.layers:
            x = layer(x) + x  # Residual
        x = self.output(x)
        x = self.dequant(x)
        return x


def static_quantization(model: nn.Module, calibration_data: torch.Tensor) -> nn.Module:
    """
    Apply static quantization to the model.
    
    Static quantization:
    - Both weights and activations are quantized
    - Requires calibration data to determine activation ranges
    - Better performance than dynamic quantization
    """
    model.eval()
    
    # Specify quantization config
    model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    
    # Prepare model for calibration
    model_prepared = torch.quantization.prepare(model)
    
    # Calibrate with representative data
    with torch.no_grad():
        model_prepared(calibration_data)
    
    # Convert to quantized model
    model_quantized = torch.quantization.convert(model_prepared)
    
    return model_quantized


def demonstrate_quantization_math():
    """Demonstrate the mathematics of quantization."""
    
    print("\n" + "=" * 60)
    print("QUANTIZATION MATHEMATICS")
    print("=" * 60)
    
    # Original FP32 tensor
    fp32_tensor = torch.randn(4, 4) * 2
    print(f"\nOriginal FP32 tensor:\n{fp32_tensor}")
    
    # Quantization parameters
    num_bits = 8
    qmin, qmax = -128, 127
    
    # Calculate scale and zero point
    x_min, x_max = fp32_tensor.min(), fp32_tensor.max()
    scale = (x_max - x_min) / (qmax - qmin)
    zero_point = qmin - x_min / scale
    zero_point = torch.clamp(torch.round(zero_point), qmin, qmax).int()
    
    print(f"\nQuantization parameters:")
    print(f"  Scale: {scale:.6f}")
    print(f"  Zero point: {zero_point}")
    
    # Quantize
    quantized = torch.clamp(
        torch.round(fp32_tensor / scale + zero_point.float()),
        qmin, qmax
    ).to(torch.int8)
    print(f"\nQuantized INT8 tensor:\n{quantized}")
    
    # Dequantize
    dequantized = (quantized.float() - zero_point.float()) * scale
    print(f"\nDequantized tensor:\n{dequantized}")
    
    # Calculate error
    error = torch.abs(fp32_tensor - dequantized)
    print(f"\nQuantization error:")
    print(f"  Mean absolute error: {error.mean():.6f}")
    print(f"  Max absolute error: {error.max():.6f}")
    print(f"  Relative error: {(error / fp32_tensor.abs().clamp(min=1e-7)).mean() * 100:.2f}%")


def compare_quantization_methods():
    """Compare different quantization approaches."""
    
    print("\n" + "=" * 60)
    print("COMPARING QUANTIZATION METHODS")
    print("=" * 60)
    
    # Create models
    original_model = SimpleModel(input_size=512, num_layers=6)
    original_model.eval()
    
    # Create test input
    test_input = torch.randn(1, 32, 512)
    
    # Get original metrics
    original_size = get_model_size(original_model)
    original_time = measure_inference_time(original_model, test_input)
    
    with torch.no_grad():
        original_output = original_model(test_input)
    
    print(f"\nOriginal Model (FP32):")
    print(f"  Size: {original_size:.2f} MB")
    print(f"  Inference time: {original_time:.2f} ms")
    
    # Dynamic quantization
    dynamic_model = dynamic_quantization(original_model)
    dynamic_size = get_model_size(dynamic_model)
    dynamic_time = measure_inference_time(dynamic_model, test_input)
    
    with torch.no_grad():
        dynamic_output = dynamic_model(test_input)
    
    dynamic_error = torch.abs(original_output - dynamic_output).mean().item()
    
    print(f"\nDynamic Quantization (INT8):")
    print(f"  Size: {dynamic_size:.2f} MB ({original_size/dynamic_size:.1f}x compression)")
    print(f"  Inference time: {dynamic_time:.2f} ms ({original_time/dynamic_time:.1f}x speedup)")
    print(f"  Mean output difference: {dynamic_error:.6f}")
    
    # Static quantization
    quantizable_model = QuantizableModel(input_size=512, num_layers=6)
    quantizable_model.eval()
    
    calibration_data = torch.randn(100, 32, 512)
    
    try:
        static_model = static_quantization(quantizable_model, calibration_data)
        static_size = get_model_size(static_model)
        static_time = measure_inference_time(static_model, test_input)
        
        print(f"\nStatic Quantization (INT8):")
        print(f"  Size: {static_size:.2f} MB")
        print(f"  Inference time: {static_time:.2f} ms")
    except Exception as e:
        print(f"\nStatic quantization requires compatible model architecture.")
        print(f"  Error: {e}")


def per_channel_vs_per_tensor():
    """Compare per-channel and per-tensor quantization."""
    
    print("\n" + "=" * 60)
    print("PER-CHANNEL vs PER-TENSOR QUANTIZATION")
    print("=" * 60)
    
    # Create a weight tensor with varying channel magnitudes
    weights = torch.randn(4, 8)
    weights[0] *= 0.1  # Small magnitudes
    weights[1] *= 1.0  # Normal magnitudes
    weights[2] *= 10.0  # Large magnitudes
    weights[3] *= 100.0  # Very large magnitudes
    
    print(f"\nOriginal weights (4 channels with different scales):")
    print(f"  Channel 0 range: [{weights[0].min():.3f}, {weights[0].max():.3f}]")
    print(f"  Channel 1 range: [{weights[1].min():.3f}, {weights[1].max():.3f}]")
    print(f"  Channel 2 range: [{weights[2].min():.3f}, {weights[2].max():.3f}]")
    print(f"  Channel 3 range: [{weights[3].min():.3f}, {weights[3].max():.3f}]")
    
    # Per-tensor quantization
    scale_tensor = (weights.max() - weights.min()) / 255
    quantized_tensor = torch.round(weights / scale_tensor).clamp(-128, 127)
    dequant_tensor = quantized_tensor * scale_tensor
    error_tensor = torch.abs(weights - dequant_tensor)
    
    print(f"\nPer-tensor quantization:")
    print(f"  Single scale for all: {scale_tensor:.6f}")
    print(f"  Mean error: {error_tensor.mean():.6f}")
    print(f"  Per-channel errors: {[f'{error_tensor[i].mean():.6f}' for i in range(4)]}")
    
    # Per-channel quantization
    errors_channel = []
    for i in range(4):
        scale = (weights[i].max() - weights[i].min()) / 255
        quantized = torch.round(weights[i] / scale).clamp(-128, 127)
        dequant = quantized * scale
        error = torch.abs(weights[i] - dequant).mean()
        errors_channel.append(error.item())
    
    print(f"\nPer-channel quantization:")
    print(f"  Separate scale per channel")
    print(f"  Mean error: {sum(errors_channel)/len(errors_channel):.6f}")
    print(f"  Per-channel errors: {[f'{e:.6f}' for e in errors_channel]}")
    
    print(f"\n→ Per-channel is {error_tensor.mean().item() / (sum(errors_channel)/len(errors_channel)):.1f}x more accurate!")


def main():
    """Main demonstration of INT8 quantization."""
    
    print("\n" + "=" * 60)
    print("   MODULE 02: INT8 QUANTIZATION")
    print("=" * 60)
    
    # 1. Demonstrate quantization math
    demonstrate_quantization_math()
    
    # 2. Compare quantization methods
    compare_quantization_methods()
    
    # 3. Per-channel vs per-tensor
    per_channel_vs_per_tensor()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("""
    KEY TAKEAWAYS:
    
    1. DYNAMIC QUANTIZATION:
       - Easy to apply (one line of code)
       - Weights quantized, activations quantized at runtime
       - ~4x size reduction, moderate speedup
    
    2. STATIC QUANTIZATION:
       - Requires calibration data
       - Both weights and activations quantized
       - Better performance but more setup
    
    3. PER-CHANNEL QUANTIZATION:
       - More accurate than per-tensor
       - Handles varying channel magnitudes
       - Standard for modern quantization
    
    4. BEST PRACTICES:
       - Start with dynamic quantization for quick wins
       - Use static quantization for production
       - Always validate on your specific task
    """)
    print("=" * 60)


if __name__ == "__main__":
    main()


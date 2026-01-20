"""
Module 11: ONNX Export and Optimization
========================================

Demonstrates exporting models to ONNX format for cross-platform deployment.
"""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple
import os


def demonstrate_onnx_export():
    """Show how to export models to ONNX."""
    
    print("\n" + "=" * 70)
    print("ONNX EXPORT GUIDE")
    print("=" * 70)
    
    export_code = '''
# ============== Method 1: Using Optimum (Recommended) ==============

from optimum.onnxruntime import ORTModelForCausalLM
from transformers import AutoTokenizer

# Export directly with optimization
model = ORTModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    export=True,  # Trigger export
    provider="CUDAExecutionProvider"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

# Save
model.save_pretrained("./llama-onnx")
tokenizer.save_pretrained("./llama-onnx")

# Use for inference
outputs = model.generate(**tokenizer("Hello", return_tensors="pt"))

# ============== Method 2: Command Line ==============

# Basic export
# optimum-cli export onnx --model meta-llama/Llama-2-7b-hf ./llama-onnx/

# With optimization
# optimum-cli export onnx \\
#     --model meta-llama/Llama-2-7b-hf \\
#     --task text-generation \\
#     --optimize O3 \\
#     --device cuda \\
#     ./llama-onnx-optimized/

# ============== Method 3: Manual PyTorch Export ==============

import torch

# For simple models
class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(768, 768)
    
    def forward(self, x):
        return self.linear(x)

model = SimpleModel()
dummy_input = torch.randn(1, 32, 768)

torch.onnx.export(
    model,
    dummy_input,
    "model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch", 1: "seq"},
        "output": {0: "batch", 1: "seq"}
    },
    opset_version=17
)
'''
    
    print(export_code)


def demonstrate_onnx_optimization():
    """Show ONNX optimization techniques."""
    
    print("\n" + "=" * 70)
    print("ONNX OPTIMIZATION TECHNIQUES")
    print("=" * 70)
    
    optimization_code = '''
# ============== Graph Optimization ==============

from onnxruntime.transformers import optimizer
from onnxruntime.transformers.fusion_options import FusionOptions

# Load and optimize
optimized_model = optimizer.optimize_model(
    "model.onnx",
    model_type="gpt2",  # or "bert", "t5", etc.
    num_heads=32,
    hidden_size=4096,
    optimization_options=FusionOptions("gpt2")
)

# Available optimizations:
# - Attention fusion
# - Layer normalization fusion
# - Skip layer normalization fusion
# - Gelu fusion
# - Bias fusion

optimized_model.save_model_to_file("model_optimized.onnx")

# ============== Quantization ==============

from onnxruntime.quantization import quantize_dynamic, QuantType

# Dynamic quantization (INT8)
quantize_dynamic(
    model_input="model_optimized.onnx",
    model_output="model_int8.onnx",
    weight_type=QuantType.QInt8
)

# Static quantization (requires calibration)
from onnxruntime.quantization import quantize_static, CalibrationDataReader

class MyCalibrationReader(CalibrationDataReader):
    def __init__(self, data):
        self.data = iter(data)
    
    def get_next(self):
        try:
            return next(self.data)
        except StopIteration:
            return None

quantize_static(
    model_input="model_optimized.onnx",
    model_output="model_int8_static.onnx",
    calibration_data_reader=MyCalibrationReader(calibration_data)
)

# ============== ONNX Runtime Inference ==============

import onnxruntime as ort
import numpy as np

# Create session with optimization
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

# GPU execution
session = ort.InferenceSession(
    "model_optimized.onnx",
    sess_options=session_options,
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
)

# Run inference
input_data = np.random.randn(1, 32, 768).astype(np.float32)
outputs = session.run(None, {"input": input_data})
'''
    
    print(optimization_code)


def show_onnx_benefits():
    """Show benefits of ONNX format."""
    
    print("\n" + "=" * 70)
    print("ONNX BENEFITS AND USE CASES")
    print("=" * 70)
    
    benefits = """
    ┌─────────────────────────────────────────────────────────────────────┐
    │                        ONNX ADVANTAGES                               │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  1. CROSS-PLATFORM DEPLOYMENT                                       │
    │     ├─ Same model works on: Windows, Linux, Mac, ARM               │
    │     ├─ CPU, GPU, NPU, FPGA support                                 │
    │     └─ Mobile (ONNX Runtime Mobile, CoreML via conversion)         │
    │                                                                      │
    │  2. FRAMEWORK AGNOSTIC                                              │
    │     ├─ Export from: PyTorch, TensorFlow, JAX                       │
    │     ├─ Import to: Any ONNX-compatible runtime                      │
    │     └─ No framework lock-in                                        │
    │                                                                      │
    │  3. OPTIMIZATION TOOLS                                              │
    │     ├─ Graph optimizations (fusion, constant folding)              │
    │     ├─ Quantization (dynamic, static)                              │
    │     └─ Hardware-specific optimizations                             │
    │                                                                      │
    │  4. CONSISTENT EXECUTION                                            │
    │     ├─ Deterministic results across platforms                      │
    │     ├─ Version-controlled model artifacts                          │
    │     └─ Easy model validation                                       │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
    
    WHEN TO USE ONNX:
    
    ✅ Cross-platform deployment required
    ✅ Need to run on diverse hardware
    ✅ Want framework independence
    ✅ CPU inference is acceptable
    ✅ Integration with enterprise systems
    
    ❌ Need maximum GPU performance (use TensorRT instead)
    ❌ LLM-specific optimizations (use vLLM, TensorRT-LLM)
    ❌ Rapid iteration in research (stay in PyTorch)
    """
    print(benefits)


def main():
    """Main demonstration of ONNX tools."""
    
    print("\n" + "=" * 70)
    print("   MODULE 11: ONNX EXPORT & OPTIMIZATION")
    print("=" * 70)
    
    # 1. Export guide
    demonstrate_onnx_export()
    
    # 2. Optimization
    demonstrate_onnx_optimization()
    
    # 3. Benefits
    show_onnx_benefits()
    
    print("\n" + "=" * 70)
    print("See tensorrt_optimize.py for TensorRT examples")
    print("=" * 70)


if __name__ == "__main__":
    main()


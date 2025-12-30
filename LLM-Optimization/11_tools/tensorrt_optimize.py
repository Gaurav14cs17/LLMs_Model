"""
Module 11: TensorRT and TensorRT-LLM
=====================================

Demonstrates using TensorRT for maximum GPU inference performance.
"""

import torch
from typing import Dict, List


def demonstrate_tensorrt_llm():
    """Show TensorRT-LLM usage."""
    
    print("\n" + "=" * 70)
    print("TENSORRT-LLM GUIDE")
    print("=" * 70)
    
    trt_llm_code = '''
# ============== Installation ==============

# pip install tensorrt-llm -U --pre --extra-index-url https://pypi.nvidia.com

# Or build from source:
# git clone https://github.com/NVIDIA/TensorRT-LLM
# cd TensorRT-LLM
# pip install .

# ============== Building a Model ==============

# Convert HuggingFace model to TensorRT-LLM format

# Step 1: Convert checkpoint
python convert_checkpoint.py \\
    --model_dir ./llama-2-7b-hf \\
    --output_dir ./tllm_checkpoint \\
    --dtype float16

# Step 2: Build TRT engine
trtllm-build \\
    --checkpoint_dir ./tllm_checkpoint \\
    --output_dir ./trt_engines \\
    --gemm_plugin float16 \\
    --gpt_attention_plugin float16 \\
    --max_batch_size 8 \\
    --max_input_len 2048 \\
    --max_output_len 512

# With quantization:
trtllm-build \\
    --checkpoint_dir ./tllm_checkpoint \\
    --output_dir ./trt_engines_int8 \\
    --gemm_plugin float16 \\
    --gpt_attention_plugin float16 \\
    --use_smooth_quant \\
    --per_token \\
    --per_channel

# ============== Running Inference ==============

from tensorrt_llm import LLM, SamplingParams

llm = LLM(model="./trt_engines")

prompts = [
    "What is machine learning?",
    "Explain quantum computing in simple terms.",
]

sampling_params = SamplingParams(
    temperature=0.7,
    top_p=0.95,
    max_tokens=256
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(f"Prompt: {output.prompt}")
    print(f"Generated: {output.outputs[0].text}")

# ============== Advanced: Multi-GPU ==============

# Build with tensor parallelism
trtllm-build \\
    --checkpoint_dir ./tllm_checkpoint \\
    --output_dir ./trt_engines_tp2 \\
    --gemm_plugin float16 \\
    --gpt_attention_plugin float16 \\
    --tp_size 2 \\
    --pp_size 1

# Run on 2 GPUs
mpirun -n 2 python run.py --engine_dir ./trt_engines_tp2
'''
    
    print(trt_llm_code)


def demonstrate_tensorrt_basic():
    """Show basic TensorRT usage."""
    
    print("\n" + "=" * 70)
    print("BASIC TENSORRT (Non-LLM Models)")
    print("=" * 70)
    
    trt_basic_code = '''
# ============== Convert ONNX to TensorRT ==============

import tensorrt as trt

def build_engine(onnx_path, engine_path):
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(
        1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    )
    parser = trt.OnnxParser(network, logger)
    
    # Parse ONNX
    with open(onnx_path, 'rb') as f:
        parser.parse(f.read())
    
    # Configure builder
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    
    # Enable FP16
    if builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    
    # Build engine
    engine = builder.build_serialized_network(network, config)
    
    # Save engine
    with open(engine_path, 'wb') as f:
        f.write(engine)
    
    return engine

# ============== Run Inference ==============

import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np

def run_inference(engine_path, input_data):
    logger = trt.Logger(trt.Logger.WARNING)
    runtime = trt.Runtime(logger)
    
    with open(engine_path, 'rb') as f:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    
    # Allocate memory
    input_idx = engine.get_binding_index("input")
    output_idx = engine.get_binding_index("output")
    
    d_input = cuda.mem_alloc(input_data.nbytes)
    d_output = cuda.mem_alloc(output_data.nbytes)
    
    # Copy to GPU
    cuda.memcpy_htod(d_input, input_data)
    
    # Execute
    context.execute_v2([int(d_input), int(d_output)])
    
    # Copy back
    output = np.empty_like(output_data)
    cuda.memcpy_dtoh(output, d_output)
    
    return output

# ============== Using torch-tensorrt (Easier) ==============

import torch_tensorrt

model = torch.load("model.pt").eval().cuda()
inputs = [torch_tensorrt.Input(
    min_shape=[1, 3, 224, 224],
    opt_shape=[8, 3, 224, 224],
    max_shape=[32, 3, 224, 224],
    dtype=torch.float16
)]

trt_model = torch_tensorrt.compile(
    model,
    inputs=inputs,
    enabled_precisions={torch.float16}
)

# Run inference
output = trt_model(input_tensor.half().cuda())
'''
    
    print(trt_basic_code)


def show_tensorrt_optimizations():
    """Show TensorRT optimization features."""
    
    print("\n" + "=" * 70)
    print("TENSORRT OPTIMIZATIONS")
    print("=" * 70)
    
    optimizations = """
    ┌─────────────────────────────────────────────────────────────────────┐
    │                    TENSORRT OPTIMIZATIONS                            │
    ├─────────────────────────────────────────────────────────────────────┤
    │                                                                      │
    │  1. LAYER FUSION                                                    │
    │     ├─ Conv + BatchNorm + ReLU → Single kernel                     │
    │     ├─ Attention fusion                                            │
    │     └─ Reduces memory bandwidth                                    │
    │                                                                      │
    │  2. PRECISION CALIBRATION                                           │
    │     ├─ FP32 → FP16: 2x speedup, minimal accuracy loss             │
    │     ├─ FP32 → INT8: 4x speedup, requires calibration              │
    │     ├─ FP8 (Hopper): Best perf/accuracy trade-off                 │
    │     └─ Mixed precision: Auto-select per layer                      │
    │                                                                      │
    │  3. KERNEL AUTO-TUNING                                              │
    │     ├─ Tests multiple kernel implementations                       │
    │     ├─ Selects fastest for your hardware                          │
    │     └─ Caches results for fast startup                            │
    │                                                                      │
    │  4. MEMORY OPTIMIZATION                                             │
    │     ├─ Tensor memory reuse                                         │
    │     ├─ Activation checkpointing                                    │
    │     └─ Minimizes GPU memory footprint                              │
    │                                                                      │
    │  5. LLM-SPECIFIC (TensorRT-LLM)                                     │
    │     ├─ Paged KV cache                                              │
    │     ├─ In-flight batching                                          │
    │     ├─ Custom attention kernels                                    │
    │     └─ Speculative decoding                                        │
    │                                                                      │
    └─────────────────────────────────────────────────────────────────────┘
    
    PERFORMANCE COMPARISON (LLaMA-7B, A100):
    
    ┌─────────────────────────┬────────────────┬────────────────┐
    │ Configuration           │ Latency (ms)   │ Throughput     │
    ├─────────────────────────┼────────────────┼────────────────┤
    │ PyTorch FP16            │ ~45            │ ~22 tok/s      │
    │ vLLM FP16               │ ~25            │ ~800 tok/s*    │
    │ TensorRT-LLM FP16       │ ~15            │ ~1200 tok/s*   │
    │ TensorRT-LLM FP8        │ ~10            │ ~1800 tok/s*   │
    │ TensorRT-LLM INT8       │ ~12            │ ~1500 tok/s*   │
    └─────────────────────────┴────────────────┴────────────────┘
    
    * Throughput with batching (batch_size=32)
    """
    print(optimizations)


def show_llama_cpp_guide():
    """Show llama.cpp usage guide."""
    
    print("\n" + "=" * 70)
    print("LLAMA.CPP QUICK GUIDE")
    print("=" * 70)
    
    llama_cpp_code = '''
# ============== Installation ==============

# Clone and build
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp

# CPU only
make

# With CUDA (NVIDIA GPU)
make LLAMA_CUDA=1

# With Metal (Apple Silicon)
make LLAMA_METAL=1

# ============== Convert Model to GGUF ==============

# Install dependencies
pip install -r requirements.txt

# Convert HuggingFace model
python convert_hf_to_gguf.py \\
    ./llama-2-7b-hf \\
    --outfile llama-2-7b.gguf \\
    --outtype f16

# Quantize
./llama-quantize llama-2-7b.gguf llama-2-7b-q4_k_m.gguf Q4_K_M

# ============== Run Inference ==============

# Interactive mode
./llama-cli -m llama-2-7b-q4_k_m.gguf -n 256 --interactive

# One-shot generation
./llama-cli -m llama-2-7b-q4_k_m.gguf \\
    -p "What is machine learning?" \\
    -n 256

# Server mode (OpenAI-compatible API)
./llama-server -m llama-2-7b-q4_k_m.gguf --port 8080

# ============== Python Bindings ==============

from llama_cpp import Llama

llm = Llama(
    model_path="./llama-2-7b-q4_k_m.gguf",
    n_ctx=4096,           # Context length
    n_gpu_layers=35,      # GPU offload (0 for CPU-only)
    n_threads=8           # CPU threads
)

output = llm(
    "What is artificial intelligence?",
    max_tokens=256,
    temperature=0.7,
    top_p=0.95,
    stop=["\\n\\n"]
)

print(output["choices"][0]["text"])

# Chat completions (OpenAI-compatible)
response = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is Python?"}
    ]
)
'''
    
    print(llama_cpp_code)


def main():
    """Main demonstration of TensorRT tools."""
    
    print("\n" + "=" * 70)
    print("   MODULE 11: TENSORRT & LLAMA.CPP")
    print("=" * 70)
    
    # 1. TensorRT-LLM
    demonstrate_tensorrt_llm()
    
    # 2. Basic TensorRT
    demonstrate_tensorrt_basic()
    
    # 3. Optimizations
    show_tensorrt_optimizations()
    
    # 4. llama.cpp
    show_llama_cpp_guide()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print("""
    TOOL SELECTION:
    
    Need MAX GPU perf?    → TensorRT-LLM
    Need high throughput? → vLLM
    Need cross-platform?  → ONNX Runtime
    Need CPU/Edge?        → llama.cpp
    Need easy local?      → Ollama
    
    TENSORRT-LLM BEST FOR:
    - Production NVIDIA GPU deployment
    - Maximum tokens/second
    - Multi-GPU inference
    - FP8/INT8 quantization
    
    LLAMA.CPP BEST FOR:
    - CPU inference
    - Mac (Metal)
    - Edge devices
    - Easy quantization (GGUF)
    """)
    print("=" * 70)


if __name__ == "__main__":
    main()


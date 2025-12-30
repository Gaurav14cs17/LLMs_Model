# Module 11: Tools & Frameworks

## 🎯 Overview

This module covers the essential tools and frameworks for LLM optimization and deployment.

---

## 🛠️ Tool Categories

### Quantization Tools

| Tool | Type | Best For |
|------|------|----------|
| **AutoGPTQ** | GPTQ quantization | High-quality INT4 |
| **AutoAWQ** | AWQ quantization | Fast, good quality |
| **bitsandbytes** | Dynamic quantization | Easy integration |
| **llama.cpp** | GGUF quantization | CPU/edge deployment |

### Inference Engines

| Tool | Best For | Key Feature |
|------|----------|-------------|
| **vLLM** | High throughput | PagedAttention |
| **TensorRT-LLM** | Max performance | NVIDIA optimization |
| **TGI** | Production serving | HF integration |
| **llama.cpp** | Edge/CPU | Portable |
| **Ollama** | Local deployment | Easy to use |

### Export/Conversion

| Tool | Format | Use Case |
|------|--------|----------|
| **Optimum** | ONNX | Cross-platform |
| **TensorRT** | TensorRT engines | NVIDIA GPUs |
| **llama.cpp** | GGUF | Portable inference |
| **ct2** | CTranslate2 | Fast CPU inference |

---

## 📦 TensorRT-LLM

NVIDIA's high-performance inference library.

```bash
# Install
pip install tensorrt-llm

# Build optimized model
python build.py \
    --model_dir ./llama-7b \
    --dtype float16 \
    --use_gpt_attention_plugin \
    --use_gemm_plugin \
    --output_dir ./trt-llama
```

Key features:
- FP8/INT8/INT4 quantization
- Tensor parallelism
- Inflight batching
- Paged KV cache

---

## 📦 ONNX Runtime

Cross-platform inference with optimizations.

```bash
# Export to ONNX
optimum-cli export onnx \
    --model meta-llama/Llama-2-7b-hf \
    --task text-generation \
    ./llama-onnx/

# Optimize
python -m onnxruntime.transformers.optimizer \
    --input llama-onnx/model.onnx \
    --output llama-onnx/model_optimized.onnx \
    --model_type gpt2
```

---

## 📦 llama.cpp

The most popular tool for local LLM deployment.

```bash
# Clone and build
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# Convert model to GGUF
python convert.py model_path --outtype q4_0 --outfile model.gguf

# Run inference
./main -m model.gguf -p "Hello!" -n 128

# Quantize further
./quantize model.gguf model_q4_k_m.gguf q4_k_m
```

### GGUF Quantization Types

| Type | Bits | Quality | Size (7B) |
|------|------|---------|-----------|
| Q8_0 | 8 | Best | ~7 GB |
| Q6_K | 6 | Excellent | ~5.5 GB |
| Q5_K_M | 5 | Great | ~4.5 GB |
| Q4_K_M | 4 | Good | ~4 GB |
| Q4_0 | 4 | Acceptable | ~3.8 GB |
| Q3_K_M | 3 | Lower | ~3 GB |
| Q2_K | 2 | Lowest | ~2.5 GB |

---

## 🧪 Hands-On Examples

```bash
cd 11_tools
python onnx_export.py
python tensorrt_optimize.py
```

---

## 💡 Tool Selection Guide

```
Need maximum GPU performance? → TensorRT-LLM
Need high throughput API? → vLLM or TGI
Need to run on CPU/Mac? → llama.cpp
Need easy local setup? → Ollama
Need cross-platform? → ONNX Runtime
```

---

## ➡️ Next Module

Continue to [Module 12: Case Studies](../12_case_studies/) for real-world examples.


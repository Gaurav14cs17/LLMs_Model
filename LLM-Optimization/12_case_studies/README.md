# Module 12: Case Studies

## 🎯 Overview

Real-world examples of LLM compression and optimization, showing the techniques applied in practice.

---

## 📊 Case Study 1: BERT Compression

### Original BERT-base
- Parameters: 110M
- Size: 440 MB (FP32)
- Inference: ~10ms (GPU)

### DistilBERT (Knowledge Distillation)
- Parameters: 66M (40% smaller)
- Size: 264 MB
- Performance: 97% of BERT
- Speed: 60% faster

### TinyBERT (Aggressive Distillation)
- Parameters: 14.5M (7.5x smaller)
- Size: 58 MB
- Performance: 96% of BERT
- Speed: 9x faster

### Quantized BERT-base (INT8)
- Size: 110 MB (4x smaller)
- Performance: 99.5% of original
- Speed: 2x faster

---

## 📊 Case Study 2: LLaMA Compression

### LLaMA-2-7B Original
- Parameters: 7B
- Size: 14 GB (FP16)
- VRAM: 14+ GB

### Compression Results

| Method | Size | VRAM | Perplexity | Notes |
|--------|------|------|------------|-------|
| FP16 (base) | 14 GB | 14 GB | 5.47 | Baseline |
| INT8 | 7 GB | 8 GB | 5.49 | +0.02 PPL |
| GPTQ INT4 | 3.9 GB | 5 GB | 5.63 | +0.16 PPL |
| AWQ INT4 | 3.9 GB | 5 GB | 5.60 | +0.13 PPL |
| GGUF Q4_K_M | 4.1 GB | 4.5 GB | ~5.7 | CPU-friendly |
| GGUF Q2_K | 2.5 GB | 3 GB | ~6.5 | Aggressive |

---

## 📊 Case Study 3: Production Deployment

### Scenario: Customer Service Chatbot

**Requirements:**
- 100 concurrent users
- <500ms latency (P95)
- Budget: $1000/month

**Solution:**
1. Model: LLaMA-2-7B → AWQ INT4
2. Server: 1x A10G (24GB)
3. Runtime: vLLM with continuous batching
4. Result: 200+ concurrent users supported

---

## 🧪 Hands-On Examples

```bash
cd 12_case_studies
python bert_compression.py
python llama_compression.py
```

---

## ➡️ Next Module

Continue to [Module 13: Future Directions](../13_future_directions/) for emerging trends.


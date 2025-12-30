<p align="center">
  <img src="https://img.shields.io/badge/🤖_LLMs-Complete_Guide-blueviolet?style=for-the-badge&logoColor=white" alt="LLMs Guide"/>
</p>

<h1 align="center">🚀 LLMS_MODEL</h1>

<p align="center">
  <b>Complete Guide to Large Language Models: Fine-Tuning & Optimization</b>
</p>

<p align="center">
  <a href="#-fine-tuning-guide"><img src="https://img.shields.io/badge/Fine--Tuning-Guide-4361ee?style=for-the-badge" alt="Fine-Tuning"/></a>
  <a href="#-optimization-guide"><img src="https://img.shields.io/badge/Optimization-Guide-7209b7?style=for-the-badge" alt="Optimization"/></a>
  <a href="https://colab.research.google.com/github/Gaurav14cs17/LLMs_Model/"><img src="https://img.shields.io/badge/Open%20in-Colab-F9AB00?style=for-the-badge&logo=googlecolab" alt="Colab"/></a>
</p>

<p align="center">
  <img src="https://img.shields.io/github/stars/Gaurav14cs17/LLMs_Model?style=social" alt="Stars"/>
  <img src="https://img.shields.io/github/forks/Gaurav14cs17/LLMs_Model?style=social" alt="Forks"/>
  <img src="https://img.shields.io/github/license/Gaurav14cs17/LLMs_Model" alt="License"/>
</p>

---

## 📚 Repository Overview

This repository provides **comprehensive guides** for working with Large Language Models (LLMs):

| Guide | Description | Topics |
|:------|:------------|:-------|
| 🎯 [Fine-Tuning Guide](./Fine-Tuning-LLMs-Guide/) | Complete fine-tuning workflow | LoRA, QLoRA, DPO, RLHF, Deployment |
| ⚡ [Optimization Guide](./LLM-Optimization/) | Model compression & efficiency | Quantization, Pruning, Distillation, Flash Attention |

---

## 🎯 Fine-Tuning Guide

<p align="center">
  <a href="./Fine-Tuning-LLMs-Guide/">
    <img src="./Fine-Tuning-LLMs-Guide/assets/svg/llm-fine-tuning-hero.svg" alt="Fine-Tuning Guide" width="700"/>
  </a>
</p>

### 📋 What's Included

| Module | Description |
|:-------|:------------|
| [01 Introduction](./Fine-Tuning-LLMs-Guide/01-Introduction/) | LLM fundamentals, transformer architecture |
| [02 Seven-Stage Pipeline](./Fine-Tuning-LLMs-Guide/02-Seven-Stage-Pipeline/) | Complete fine-tuning workflow |
| [03 Data Preparation](./Fine-Tuning-LLMs-Guide/03-Data-Preparation/) | Dataset creation & formatting |
| [04 Model Initialization](./Fine-Tuning-LLMs-Guide/04-Model-Initialization/) | Loading & configuring models |
| [05 Training Setup](./Fine-Tuning-LLMs-Guide/05-Training-Setup/) | Hyperparameters & optimizers |
| [06 Fine-Tuning Techniques](./Fine-Tuning-LLMs-Guide/06-Fine-Tuning-Techniques/) | LoRA, QLoRA, DoRA, DPO, PPO |
| [07 Evaluation](./Fine-Tuning-LLMs-Guide/07-Evaluation-Validation/) | Metrics & benchmarks |
| [08 Deployment](./Fine-Tuning-LLMs-Guide/08-Deployment/) | vLLM, TGI, Ollama |
| [09 Monitoring](./Fine-Tuning-LLMs-Guide/09-Monitoring-Maintenance/) | Production maintenance |

### 📓 Google Colab Notebooks

| Notebook | Description | Open |
|:---------|:------------|:----:|
| Basic Fine-Tuning | SFT with HuggingFace | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/LLMs_Model/blob/main/Fine-Tuning-LLMs-Guide/notebooks/01_basic_fine_tuning.ipynb) |
| LoRA Fine-Tuning | Parameter-efficient training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/LLMs_Model/blob/main/Fine-Tuning-LLMs-Guide/notebooks/02_lora_fine_tuning.ipynb) |
| QLoRA Fine-Tuning | 4-bit quantized training | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/LLMs_Model/blob/main/Fine-Tuning-LLMs-Guide/notebooks/03_qlora_fine_tuning.ipynb) |
| DPO Training | Preference alignment | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Gaurav14cs17/LLMs_Model/blob/main/Fine-Tuning-LLMs-Guide/notebooks/04_dpo_training.ipynb) |

---

## ⚡ Optimization Guide

<p align="center">
  <a href="./LLM-Optimization/">
    <img src="./LLM-Optimization/assets/banner.svg" alt="Optimization Guide" width="700"/>
  </a>
</p>

### 📋 What's Included

| Module | Description | Key Concepts |
|:-------|:------------|:-------------|
| [01 Introduction](./LLM-Optimization/01_introduction/) | Compression fundamentals | Theory, trade-offs |
| [02 Quantization](./LLM-Optimization/02_quantization/) | Precision reduction | INT8, INT4, GPTQ, AWQ |
| [03 Pruning](./LLM-Optimization/03_pruning/) | Weight removal | Magnitude, structured, N:M |
| [04 Knowledge Distillation](./LLM-Optimization/04_knowledge_distillation/) | Model compression | Teacher-student |
| [05 Weight Sharing](./LLM-Optimization/05_weight_sharing/) | Parameter reuse | ALBERT, MQA, GQA |
| [06 Factorization](./LLM-Optimization/06_factorization/) | Low-rank decomposition | SVD, Tucker |
| [07 Sparsity](./LLM-Optimization/07_sparsity/) | Sparse computation | MoE, 2:4 sparsity |
| [08 PEFT](./LLM-Optimization/08_peft/) | Efficient fine-tuning | LoRA, QLoRA, Adapters |
| [09 Efficient Architectures](./LLM-Optimization/09_efficient_architectures/) | Architecture innovations | Flash Attention |
| [10 Compression Pipelines](./LLM-Optimization/10_compression_pipelines/) | End-to-end workflows | Combined techniques |
| [11 Tools](./LLM-Optimization/11_tools/) | Optimization tools | TensorRT, ONNX, llama.cpp |
| [12 Case Studies](./LLM-Optimization/12_case_studies/) | Real examples | BERT, LLaMA |
| [13 Future Directions](./LLM-Optimization/13_future_directions/) | Emerging techniques | Speculative decoding |

### 📐 Mathematical Foundations

Each module includes detailed mathematical proofs and formulas:

```
📐 Quantization: Q(x) = round((x - z) / s)
📐 Pruning: saliency = ½ × Hᵢᵢ × wᵢ²
📐 Distillation: L = T² × KL(σ_T(z_T) ‖ σ_T(z_S))
📐 LoRA: W = W₀ + BA, rank(BA) ≤ r
📐 Flash Attention: O(Nd + N²d/M) IO complexity
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/Gaurav14cs17/LLMs_Model.git
cd LLMS_MODEL

# Install Fine-Tuning dependencies
pip install transformers datasets accelerate peft bitsandbytes trl

# Install Optimization dependencies  
pip install -r LLM-Optimization/requirements.txt
```

### Basic Usage

```python
# Fine-tune with LoRA
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
lora_config = LoraConfig(r=16, lora_alpha=32)
model = get_peft_model(model, lora_config)
# Train only 0.1% of parameters!
```

```python
# Quantize to 4-bit
from transformers import BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4"
)
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-hf",
    quantization_config=bnb_config
)
# 4x smaller model!
```

---

## 🏗️ Supported Models

| Model | Organization | Parameters |
|:------|:-------------|:-----------|
| LLaMA 2/3 | Meta | 7B - 70B |
| Mistral/Mixtral | Mistral AI | 7B - 8x7B |
| Phi-2/3 | Microsoft | 2.7B - 3.8B |
| Gemma | Google | 2B - 7B |
| Qwen/Qwen2 | Alibaba | 0.5B - 72B |
| ChatGLM/GLM-4 | Zhipu AI | 6B - 130B |
| DeepSeek | DeepSeek | 7B - 67B |
| Yi | 01.AI | 6B - 34B |
| Baichuan | Baichuan AI | 7B - 13B |
| InternLM | Shanghai AI Lab | 7B - 20B |

---

## 📖 References

### Key Papers

| Topic | Paper | Link |
|:------|:------|:-----|
| Fine-Tuning Survey | A Comprehensive Guide to Fine-Tuning LLMs | [arXiv:2408.13296](https://arxiv.org/abs/2408.13296) |
| LoRA | Low-Rank Adaptation of LLMs | [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) |
| QLoRA | Efficient Finetuning of Quantized LLMs | [arXiv:2305.14314](https://arxiv.org/abs/2305.14314) |
| DPO | Direct Preference Optimization | [arXiv:2305.18290](https://arxiv.org/abs/2305.18290) |
| GPTQ | Accurate Post-Training Quantization | [arXiv:2210.17323](https://arxiv.org/abs/2210.17323) |
| Flash Attention | Fast and Memory-Efficient Attention | [arXiv:2205.14135](https://arxiv.org/abs/2205.14135) |

### Tools & Libraries

| Tool | Description | Link |
|:-----|:------------|:-----|
| 🤗 Transformers | Model hub & training | [huggingface.co](https://huggingface.co/docs/transformers) |
| 🤗 PEFT | Parameter-efficient fine-tuning | [huggingface.co/peft](https://huggingface.co/docs/peft) |
| 🤗 TRL | Reinforcement learning | [huggingface.co/trl](https://huggingface.co/docs/trl) |
| vLLM | High-throughput serving | [github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) |
| llama.cpp | CPU inference | [github.com/ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) |

---

## ⭐ Star History

[![Star History Chart](https://api.star-history.com/svg?repos=Gaurav14cs17/LLMs_Model&type=Date)](https://star-history.com/#Gaurav14cs17/LLMs_Model&Date)

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](./LLM-Optimization/LICENSE) file for details.

---

<p align="center">
  <b>Made with ❤️ for the ML Community</b>
</p>

<p align="center">
  <a href="https://github.com/gaurav14cs17">
    <img src="https://img.shields.io/badge/GitHub-gaurav14cs17-181717?style=for-the-badge&logo=github" alt="GitHub"/>
  </a>
</p>


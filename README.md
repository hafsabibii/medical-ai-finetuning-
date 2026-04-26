# 🧠 Medical AI Fine-Tuning with QLoRA & Unsloth

> Fine-tuned LLaMA-3 8B on medical QA data — running entirely on a free Google Colab T4 GPU.

---

## 📌 Project Overview

Large Language Models are powerful — but adapting them for specific domains like healthcare usually requires expensive hardware and massive compute.

This project proves otherwise.

Using **QLoRA** and **Unsloth**, I fine-tuned **LLaMA-3 8B** on a medical question-answering dataset using nothing but a **free Google Colab GPU** — making domain-specific AI accessible without a budget.

---

## ⚙️ Tech Stack

| Tool | Purpose |
|------|---------|
| `unsloth/llama-3-8b-bnb-4bit` | Base model (4-bit quantized) |
| QLoRA | Memory-efficient fine-tuning |
| PEFT | Trains only select layers |
| Hugging Face Transformers | Model & tokenizer pipeline |
| TRL / SFTTrainer | Supervised fine-tuning |
| BitsAndBytes | 4-bit quantization backend |
| Google Colab (T4 GPU) | Free training environment |

---

## 🔍 Key Techniques Explained

**QLoRA (Quantized Low-Rank Adaptation)**
Reduces model weights to 4-bit precision, drastically cutting VRAM usage while preserving performance.

**PEFT (Parameter Efficient Fine-Tuning)**
Instead of updating all model parameters, only specific attention layers (`q_proj`, `v_proj`) are trained — faster and lighter.

**Gradient Accumulation**
Simulates a larger batch size on limited hardware by accumulating gradients over multiple steps before updating weights.

---

## 🗂️ Dataset Format

Training data should be in `.jsonl` format:

```json
{"prompt": "What is anemia?", "response": "Anemia is a condition where the blood lacks enough healthy red blood cells..."}
{"prompt": "What causes diabetes?", "response": "Diabetes is caused by insufficient insulin production or insulin resistance..."}
```

Two files needed:
- `medical_qa_train.jsonl` — training data
- `medical_qa_eval.jsonl` — validation data

---

## 🚀 How to Run

1. Open the notebook in **Google Colab**
2. Install dependencies (first cell)
3. Add your Hugging Face token when prompted — **do not hardcode it**
4. Upload your `.jsonl` dataset files
5. Run all cells — training takes ~15–30 mins on T4

---

## ⚠️ Important

- Never hardcode your HF token in the notebook
- Use `login()` interactively or store it in Colab Secrets
- Do not share your notebook publicly with the token visible

--

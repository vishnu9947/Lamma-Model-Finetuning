🦙 LLaMA Fine-Tuning with QLoRA

This project demonstrates how to fine-tune the LLaMA family of models efficiently using QLoRA (Quantized Low-Rank Adaptation).
QLoRA enables fine-tuning large language models on limited hardware by combining 4-bit quantization with LoRA adapters, drastically reducing memory usage while maintaining model performance.

🚀 Features

⚡ Efficient Fine-Tuning – Train LLaMA models with minimal GPU resources.

💾 4-bit Quantization – Memory-efficient model loading with bitsandbytes.

🔧 LoRA Adapters – Low-rank parameter updates without full fine-tuning.

📊 Evaluation & Inference – Test fine-tuned models on downstream tasks.

🔗 Hugging Face Integration – Use transformers and peft for seamless workflow.

🛠️ Tech Stack

Base Model: LLaMA (Meta’s LLaMA-2 or LLaMA-3 depending on access)

Quantization: bitsandbytes

Fine-Tuning Framework: PEFT (Parameter-Efficient Fine-Tuning)

Libraries:

PyTorch

Hugging Face Transformers

Datasets

Accelerate

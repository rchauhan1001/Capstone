# Distilling Multi-Agent Social Reasoning: Compressing MetaMind's Cognitive Architecture via Reinforcement Learning & Supervised Fine-Tuning

## Overview

**Theory of Mind (ToM)** — the human ability to infer others' unspoken intentions, emotions, and beliefs — has long been a challenge for large language models, which tend to struggle with emotional nuance and the asymmetric information inherent in social scenarios.

[Zhang et al. (2025)](https://arxiv.org/abs/2501.13177) demonstrated that a multi-agent framework (MetaMind) can emulate human-like social reasoning and meaningfully improve the social intelligence of LLMs. However, the vanila framework requires significant inference time and was tested upon top-notch large models. 

This project builds on that work aiming to reduce inference cost while preserving social reasoning quality by **1. distilling MetaMind's reasoning capability from a large model (GPT-4o, 120B+) into a small model (LLaMA 3 8B), and    2. re-frame MetaMind's framework for model calls to reduce multi-agent overhead.**

Our approach:
1. Run a minimized MetaMind structure on a large base model to generate chain-of-thought reasoning traces over social scenarios
2. Filter for correct answers to produce a high-quality distillation dataset
3. Supervised Fine-Tune (SFT) LLaMA 3.1 8B on these reasoning traces
4. Optionally apply Reinforcement Learning (RL) to further boost ToM performance post-SFT

---


## Dataset

We use **Social IQa** ([Sap et al., 2019](https://arxiv.org/abs/1904.09728)), a large-scale benchmark for commonsense reasoning about social situations.

| Property | Details |
|----------|---------|
| Size | ~38,000 multiple-choice questions |
| Format | Context + Question + 3 answer choices |
| Domain | Everyday social interactions |
| Task type | Multiple choice (A / B / C) |

Social IQa does not include reasoning chains behind its answers. Since explicit reasoning is known to benefit ToM tasks, we use MetaMind to **generate reasoning traces** for each question, then filter to keep only samples where the model answers correctly. This filtered, reasoning-augmented subset forms our **distillation training dataset**.

---


## Running the Data Generation Pipeline

Data generation runs on a compute cluster due to the high per-sample cost of the MetaMind framework.
The vanilla MetaMind framework is powerful but expensive. Running it on a 120B OpenAI model incurs **~50 seconds per sample** due to ~11 chained model calls per question. This took us a long time to prepare the dataset and also makes the vanilla framework impractical for real-time applications or resource-constrained settings.

```bash
python run_socialiqa_cluster.py \
  --dev_path "/scratch/<username>/socialiqa-train-dev/dev.jsonl" \
  --labels_path "/scratch/<username>/socialiqa-train-dev/dev-labels.lst" \
  --output_dir "/scratch/<username>/results" \
  --max_samples 1000
```

---

## Running SFT Training

A preliminary training pipeline is available under the training_phase folder:

```bash
python dataloader.py \
  --input_file "/scratch/<username>/results/socialiqa_results_20260304_111247.jsonl" \
  --output_file "/scratch/<username>/results/training_data.jsonl"
```

> **Status:** Full SFT training will begin once distillation data generation is complete. We intend too further improve it via reinforcement learning in later stages.

---


## References

- Zhang et al. (2025). *MetaMind: A Multi-Agent Framework for Social Reasoning in Large Language Models.* [arXiv](https://arxiv.org/abs/2501.13177)
- Sap et al. (2019). *Social IQa: Commonsense Reasoning about Social Interactions.* EMNLP 2019. [arXiv](https://arxiv.org/abs/1904.09728)

---

## Team

*Members: Eitan Laredo，Qihui Fan, Roshan Chouhan，and Tianrui Chen*

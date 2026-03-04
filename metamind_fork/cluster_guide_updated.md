# Cluster Setup Guide

## Prerequisites
- Northeastern username and password
- SSH access to Explorer cluster
- The metamind_fork repo cloned locally on your Mac

---

## Step 1: SSH into the Cluster
```bash
ssh <username>@login.explorer.northeastern.edu
# Or if you've set up ~/.ssh/config with alias:
ssh explorer
```
Verify you're connected — you should see the Explorer ASCII banner.

---

## Step 2: Upload MetaMind Code
From your Mac terminal (not the cluster):
```bash
scp -r '/path/to/metamind_fork/.' <username>@xfer.discovery.neu.edu:/scratch/<username>/MetaMind/
```
Verify on the cluster:
```bash
ls /scratch/<username>/MetaMind/
```
You should see `run_socialiqa_cluster.py`, `run_metamind.slurm`, `llm_interface/`, `agents/`, etc.

---

## Step 3: Upload Social IQA Dataset
From your Mac terminal:
```bash
scp -r '/path/to/socialiqa-train-dev' <username>@xfer.discovery.neu.edu:/scratch/<username>/
```

---

## Step 4: Download the Model
Get a compute node with internet access:
```bash
srun --partition=short --nodes=1 --cpus-per-task=4 --mem=16G --time=02:00:00 --pty /bin/bash
```

Once on the compute node, load modules and activate/create the conda env:
```bash
module load anaconda3/2024.06
conda create -c conda-forge python=3.10 -y --prefix /scratch/<username>/envs/inference
source activate /scratch/<username>/envs/inference
```

Install dependencies:
```bash
pip install vllm transformers accelerate huggingface_hub
```

Download the model (this takes ~15-20 minutes):
```bash
export HF_HOME=/scratch/<username>/.cache/huggingface
huggingface-cli download openai/gpt-oss-120b --exclude "original/*" --local-dir /scratch/<username>/models/gpt-oss-120b-hf
```

Verify:
```bash
ls /scratch/<username>/models/gpt-oss-120b-hf/
```
You should see `config.json`, `tokenizer.json`, and 15 model shards.

---

## Step 5: Update the Scripts
Exit the compute node:
```bash
exit
```

Update all scripts with your username:
```bash
sed -i 's|/scratch/laredo.ei|/scratch/<username>|g' /scratch/<username>/MetaMind/run_metamind.slurm
sed -i 's|/scratch/laredo.ei|/scratch/<username>|g' /scratch/<username>/MetaMind/run_socialiqa_cluster.py
```

Verify:
```bash
grep "scratch" /scratch/<username>/MetaMind/run_metamind.slurm | head -5
```
All paths should show your username.

---

## Step 6: Test Run (3 samples)
```bash
sbatch --export=ALL,START_SAMPLE=1,MAX_SAMPLES=3 /scratch/<username>/MetaMind/run_metamind.slurm
```

Monitor:
```bash
squeue --me
tail -f /scratch/<username>/logs/metamind_<JOBID>.out
```

Expected sequence:
- `[DirectVLLM] Loading model from: ...`
- Model loading messages (~3-5 min)
- `[DirectVLLM] Model loaded.`
- `[DirectVLLM] Call #1 completed in ...`
- Sample results with `Predicted / Gold / CORRECT or WRONG`

Check results:
```bash
wc -l /scratch/<username>/results/socialiqa_results_*.jsonl
cat /scratch/<username>/results/summary_*.json
```

---

## Step 7: Run Full Batches
Edit `launch_train_jobs.sh` to use your username:
```bash
sed -i 's|/scratch/laredo.ei|/scratch/<username>|g' /scratch/<username>/MetaMind/launch_train_jobs.sh
```

Update the start/end samples as needed, then launch:
```bash
bash /scratch/<username>/MetaMind/launch_train_jobs.sh
```

---

## Notes
- Max 6 concurrent GPU jobs per user
- Each job processes ~30 samples safely within 8hr limit (~200s per sample on H200)
- Model loads in ~3-5 min
- Results land in `/scratch/<username>/results/`
- If a job gets cancelled mid-run, check the last sample ID and resubmit with `--START_SAMPLE=<last+1>`
- All previous results before the DirectVLLM fix (pre March 3 2026) used a broken proxy and produced invalid data — do not use those results

---

## Architecture Notes
MetaMind uses **DirectVLLM** (`llm_interface/direct_vllm.py`) which calls vLLM's Python API directly using a chat format with a system prompt — no HTTP server, no proxy issues. This replaced the previous LocalVLLM approach which spun up a vLLM HTTP server and had persistent proxy/health-check problems on the cluster.

Each sample requires ~30 LLM calls across three agents:
- **ToMAgent**: generates 7 mental state hypotheses (`max_tokens=500`)
- **DomainAgent**: refines each hypothesis (`max_tokens=800`) and scores plausibility (`max_tokens=50`)
- **ResponseAgent**: synthesizes, validates, and optionally optimizes the final answer

The system prompt in `direct_vllm.py` is tuned for gpt-oss-120b instruction following:
```
"You are a precise reasoning engine. You receive tasks and produce only the requested output — nothing more. You never narrate, plan, or describe your process. You never begin with phrases like 'we need to', 'analysis', 'let me', or 'the task is'. You simply reason and output."
```

---

## config.py Notes
The `config.py` file contains `OPENAI_API_KEY` which will throw a `KeyError` on the cluster if set to `os.environ["OPENAI_API_KEY"]`. Use `os.environ.get("OPENAI_API_KEY", "not-needed")` instead — this works for both local and cluster environments.

---

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `KeyError: OPENAI_API_KEY` | config.py uses `os.environ[]` | Change to `os.environ.get("OPENAI_API_KEY", "not-needed")` |
| Model narrating instead of answering | gpt-oss-120b instruction following | System prompt in `direct_vllm.py` addresses this |
| `finish: length` on inference calls | Response hitting max token limit | Per-call `max_tokens` set in each agent |
| Slow inference (~900s/sample) | Prompt snowballing between agents | Regex parsing in `tom_agent.py` and `domain_agent.py` extracts clean fields |
| `ModuleNotFoundError: direct_vllm` | Missing export in `__init__.py` | Add `from .direct_vllm import DirectVLLM` to `llm_interface/__init__.py` |
| Unrecognized model / no model_type | Using `original/` format | Re-download without `original/*`, use HF format |
| Home directory full | Conda envs or HF cache in home | Move to scratch, set `HF_HOME` to scratch |
| scp nesting folders | `scp -r folder` creates subfolder | Use `scp -r folder/.` for contents only |
| SSH timeout kills session | Idle connection dropped | Use `ssh -o ServerAliveInterval=60` or use `sbatch` instead of `srun` |

---

## Key File Locations (for user laredo.ei)
```
/scratch/laredo.ei/
├── MetaMind/
│   ├── run_socialiqa_cluster.py     # Main runner (uses DirectVLLM)
│   ├── run_metamind.slurm           # SLURM job script
│   ├── launch_train_jobs.sh         # Batch launcher
│   ├── config.py                    # Agent configs (utility_threshold=0.8)
│   ├── llm_interface/
│   │   ├── direct_vllm.py           # Direct vLLM Python API (chat format)
│   │   └── __init__.py              # Must export DirectVLLM
│   ├── agents/
│   │   ├── tom_agent.py             # max_tokens=500, regex parsing
│   │   ├── domain_agent.py          # max_tokens=800/50, regex parsing
│   │   └── response_agent.py        # regex validation parsing
│   └── prompts/
│       └── prompt_templates.py      # IMPORTANT: directives on all output sections
├── models/
│   └── gpt-oss-120b-hf/             # HuggingFace format (used by vLLM)
├── socialiqa-train-dev/             # Social IQA dataset (33,410 samples)
├── envs/inference/                  # Conda environment
├── results/                         # Output files
├── logs/                            # SLURM job logs
└── .cache/huggingface/              # HuggingFace cache
```

# Useful Cluster Commands

## SSH
```bash
# Connect to cluster
ssh <username>@login.explorer.northeastern.edu

# Connect with keepalive to prevent timeout
ssh -o ServerAliveInterval=60 explorer

# Set up SSH alias in ~/.ssh/config:
# Host explorer
#     HostName login.explorer.northeastern.edu
#     User laredo.ei
```

## File Transfer (always from local Mac terminal)
```bash
# Upload a single file
scp "/path/with spaces/file.py" user@xfer.discovery.neu.edu:/scratch/user/dest/file.py

# Upload folder contents (use /. to avoid nesting)
scp -r '/path/to/folder/.' user@xfer.discovery.neu.edu:/scratch/user/dest/

# Upload dataset
scp -r '/path/to/socialiqa-train-dev' user@xfer.discovery.neu.edu:/scratch/user/
```

## SLURM Job Management
```bash
# Submit a job
sbatch /scratch/laredo.ei/MetaMind/run_metamind.slurm

# Submit with parameters
sbatch --export=ALL,START_SAMPLE=7001,MAX_SAMPLES=3 /scratch/laredo.ei/MetaMind/run_metamind.slurm

# Check your jobs
squeue --me

# Cancel a job
scancel <JOBID>

# Cancel multiple jobs
scancel <JOBID1> <JOBID2> <JOBID3>

# Check job history and status
sacct -u laredo.ei --format=JobID,JobName,State,Elapsed,Reason -n | tail -10

# Check only metamind jobs
sacct -u laredo.ei --format=JobID,JobName,State,Elapsed -n | grep metamind
```

## Monitoring Jobs
```bash
# Tail live job output
tail -f /scratch/laredo.ei/logs/metamind_<JOBID>.out

# Tail error log
cat /scratch/laredo.ei/logs/metamind_<JOBID>.err

# Check GPU usage on a running node
ssh <nodename> nvidia-smi
# e.g. ssh d4054 nvidia-smi
```

## Results Inspection
```bash
# List recent result files
ls -lt /scratch/laredo.ei/results/*.jsonl | head -10

# Check sample count in results
wc -l /scratch/laredo.ei/results/socialiqa_results_*.jsonl

# Check summary files
cat /scratch/laredo.ei/results/summary_*.json

# Quick sample overview
python3 -c "
import json
with open('/scratch/laredo.ei/results/<file>.jsonl') as f:
    for line in f:
        d = json.loads(line)
        print(f'Sample {d[\"sample_id\"]}: predicted={d[\"predicted_answer\"]} gold={d[\"gold_answer\"]} correct={d[\"correct\"]} elapsed={d[\"elapsed_seconds\"]}s')
"

# Check first sample ID in a file
head -1 /scratch/laredo.ei/results/<file>.jsonl | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print('start:', d['sample_id'])"

# Check last sample ID in a file
tail -1 /scratch/laredo.ei/results/<file>.jsonl | python3 -c "import json,sys; d=json.loads(sys.stdin.read()); print('last:', d['sample_id'])"

# Check inference call times
python3 -c "
import json
with open('/scratch/laredo.ei/results/inference_log_<timestamp>.jsonl') as f:
    for line in f:
        d = json.loads(line)
        print(f'Call #{d[\"call_id\"]}: {d[\"elapsed_seconds\"]}s | tokens: {d.get(\"usage\",{}).get(\"total_tokens\",\"?\")} | finish: {d.get(\"finish_reason\",\"?\")}')
"

# Check error rate in results
python3 -c "
import json
errors, real = 0, 0
with open('/scratch/laredo.ei/results/<file>.jsonl') as f:
    for line in f:
        d = json.loads(line)
        if 'Connection error' in str(d['final_response']) or 'Error generating' in str(d['final_response']):
            errors += 1
        else:
            real += 1
print(f'Real: {real} | Errors: {errors} | Error rate: {errors/(real+errors)*100:.1f}%')
"
```

## File Editing on Cluster
```bash
# Edit a file
nano /scratch/laredo.ei/MetaMind/agents/tom_agent.py

# Search for a string in a file
grep -n "enforce_eager\|max_tokens" /scratch/laredo.ei/MetaMind/llm_interface/direct_vllm.py

# Replace string in file (in-place)
sed -i 's|/scratch/laredo.ei|/scratch/<username>|g' /scratch/laredo.ei/MetaMind/run_metamind.slurm

# Replace specific line content
sed -i 's|old_string|new_string|' /path/to/file.py

# Add a new line after a match
sed -i 's|from .local_vllm import LocalVLLM|from .local_vllm import LocalVLLM\nfrom .direct_vllm import DirectVLLM|' file.py
```

## Storage Management
```bash
# Check disk usage in home
du -sh /home/laredo.ei/* /home/laredo.ei/.[^.]* 2>/dev/null | sort -rh

# Check scratch usage
du -sh /scratch/laredo.ei/*/

# List files by size
ls -lSh /scratch/laredo.ei/results/ | head -10
```

## Interactive Compute Node (for downloads/installs)
```bash
# Get a short partition node with internet
srun --partition=short --nodes=1 --cpus-per-task=4 --mem=16G --time=02:00:00 --pty /bin/bash

# Load modules
module purge
module load anaconda3/2024.06 cuda/12.1.1

# Activate conda env
source activate /scratch/laredo.ei/envs/inference

# Exit interactive node
exit
```

## Model Download
```bash
export HF_HOME=/scratch/laredo.ei/.cache/huggingface

# Download gpt-oss-120b (HF format only, ~61GB)
huggingface-cli download openai/gpt-oss-120b \
    --exclude "original/*" \
    --local-dir /scratch/laredo.ei/models/gpt-oss-120b-hf

# Verify model format
head -3 /scratch/laredo.ei/models/gpt-oss-120b-hf/config.json
# Should show: "architectures": ["GptOssForCausalLM"]
```

# HPC Cluster Training Guide

Run the TikTok recommendation training pipeline on the IE University HPC cluster.

## Cluster Info

- **Login node:** `rust` at `10.205.20.10`
- **Compute node:** `haskell` — 512 CPU, 503 GB RAM, 2x RTX 6000 Ada (48 GB)
- **Credentials:** `nlp03` / `Scitech2026!`
- **Partitions:** `gpu` (3-day max), `cpu` (7-day max)

## Quick Start

### 1. Connect

**Off-campus (Windows — must use WSL):**

```bash
# Terminal 1: start tunnel (keep open)
sudo sshuttle --dns -NHr nlp03@ssh.iesci.tech 0/0

# Terminal 2: SSH into cluster
ssh nlp03@10.205.20.10
```

**On-campus:**

```bash
ssh nlp03@10.205.20.10
```

### 2. Clone and setup (first time only)

```bash
git clone https://github.com/PredictiveSocialMedia/Tik-Tok-Recommendation-System.git
cd Tik-Tok-Recommendation-System
git checkout training_branch
bash hpc/setup_env.sh
```

### 3. Submit training job

```bash
cd ~/Tik-Tok-Recommendation-System
export DATABASE_URL="postgresql://user:pass@host:5432/dbname"  # ask team lead for the URL
sbatch --export=ALL hpc/train_pipeline.sh
```

### 4. Monitor

```bash
# Check job status
squeue -u $USER

# Watch live output
tail -f train_tiktok-pipeline_<JOBID>.log

# Check GPU availability
sinfo -o "%N %G %t"
```

### 5. When training finishes

Check the log for metrics:

```bash
cat train_tiktok-pipeline_<JOBID>.log | tail -20
```

Artifacts are auto-copied to `~/tiktok-artifacts/`. To get them to your local machine:

```bash
# From your local machine (with tunnel running):
scp -r nlp03@10.205.20.10:~/tiktok-artifacts/ ./artifacts-from-cluster/
```

## Useful Commands

```bash
squeue -u $USER                    # Your running/pending jobs
scancel <JOBID>                    # Cancel a job
sacct -u $USER --format=JobID,JobName,State,Elapsed,ExitCode  # Past jobs
sinfo                              # Cluster load
nvidia-smi                         # GPU status (on compute node only)
```

## Troubleshooting

- **Job stuck in PD (pending):** GPU may be in use. Check `sinfo -o "%N %G %t"`.
- **OOM killed:** Increase `--mem` in `train_pipeline.sh`.
- **DB connection error:** Cluster needs internet access to reach Supabase. Check with `curl -s https://mlmlcilyoqvbvgljsjtv.supabase.co`.
- **Conda not found:** Run `source ~/.bashrc` after initial setup.

#!/bin/bash
#SBATCH --job-name=pae_uccix_llama          # Job name
#SBATCH --output=logs/pae_uccix_llama_%j.out  # Standard output log
#SBATCH --partition=physical-gpu                        # Partition/queue name
#SBATCH --gres=gpu:1                           # Number of GPUs
#SBATCH --cpus-per-task=8                      # CPU cores per task
#SBATCH --mem=32G                              # Memory per node
#SBATCH --time=12:00:00                        # Maximum runtime (HH:MM:SS)

# --- Paths to models ---
BASE_MODEL_PATH="meta-llama/Llama-2-13b"
TARGET_MODEL_PATH="ReliableAI/UCCIX-Llama2-13B"
JSON_OUTPUT_DIR="../json_artifacts/pae_uccix_llama_${SLURM_JOB_ID}"

# --- Run Python script ---
python ../pa_experiment.py \
    --base-model-path "$BASE_MODEL_PATH" \
    --target-model-path "$TARGET_MODEL_PATH" \
    --json-artifact-out "$JSON_OUTPUT_DIR"

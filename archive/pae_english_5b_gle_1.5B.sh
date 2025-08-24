#!/bin/bash
#SBATCH --job-name=pae_english_5b_gle_1.5B          # Job name
#SBATCH --output=logs/pae_english_5b_gle_1.5B-%j.out  # Standard output log
#SBATCH --partition=physical-gpu                        # Partition/queue name
#SBATCH --gres=gpu:1                           # Number of GPUs
#SBATCH --cpus-per-task=8                      # CPU cores per task
#SBATCH --mem=32G                              # Memory per node
#SBATCH --time=12:00:00                        # Maximum runtime (HH:MM:SS)

# --- Paths to models ---
BASE_MODEL_PATH="/home/khanh/sla/sla_cpt/qwen2.5-0.5b_english_wiki/checkpoint-12197"
TARGET_MODEL_PATH="/home/khanh/sla/sla_cpt/qwen2.5-0.5b_english_wiki_irish_corpus_custom_sampler"
JSON_OUTPUT_DIR="../json_artifacts/pae_english_5b_gle_1.5B_${SLURM_JOB_ID}"

# --- Run Python script ---
python ../pa_experiment.py \
    --base-model-path "$BASE_MODEL_PATH" \
    --target-model-path "$TARGET_MODEL_PATH" \
    --json-artifact-out "$JSON_OUTPUT_DIR"

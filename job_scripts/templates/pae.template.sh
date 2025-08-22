#!/bin/bash
#SBATCH --job-name=parameter_analysis          # Job name
#SBATCH --output=logs/parameter_analysis-%j.out  # Standard output log
#SBATCH --partition=gpu                        # Partition/queue name
#SBATCH --gres=gpu:1                           # Number of GPUs
#SBATCH --cpus-per-task=8                      # CPU cores per task
#SBATCH --mem=32G                              # Memory per node
#SBATCH --time=12:00:00                        # Maximum runtime (HH:MM:SS)
#SBATCH --mail-type=END,FAIL                   # Email notifications (optional)
#SBATCH --mail-user=your.email@domain.com      # Your email address

# --- Paths to models ---
BASE_MODEL_PATH="/home/khanh/sla/sla_cpt/qwen2.5-0.5b_english_wiki/checkpoint-12197"
TARGET_MODEL_PATH="/path/to/target_model_folder"
JSON_OUTPUT_DIR="/path/to/json_artifacts"

# --- Run Python script ---
python /path/to/your_script.py \
    --base-model-path "$BASE_MODEL_PATH" \
    --target-model-path "$TARGET_MODEL_PATH" \
    --json-artifact-out "$JSON_OUTPUT_DIR"

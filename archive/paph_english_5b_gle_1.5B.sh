#!/bin/bash
#SBATCH --job-name=paph_english_5b_gle_1.5B       # Job name
#SBATCH --output=logs/paph_english_5b_gle_%j.out   # Standard output log
#SBATCH --partition=physical-gpu                # Partition name, adjust if CPU only
#SBATCH --gres=gpu:1                   # Number of GPUs (if needed)
#SBATCH --cpus-per-task=4              # CPU cores
#SBATCH --mem=16G                      # Memory per node
#SBATCH --time=01:00:00                # Time limit hh:mm:ss

# --- Run the script ---
JSON_ARTIFACT="../json_artifacts/pae_english_5b_gle_1.5B_677/20250822_211112_checkpoint-12197_qwen2.5-0.5b_english_wiki_irish_corpus_custom_sampler.json"
OUTPUT_DIR="../plot_artifacts/paph_english_5b_gle_1.5B_${SLURM_JOB_ID}/"


python ../pa_plot_heat.py --json $JSON_ARTIFACT --out-dir $OUTPUT_DIR

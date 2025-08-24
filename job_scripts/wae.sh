#!/bin/bash
#SBATCH --job-name=wae          # Job name
#SBATCH --output=logs/%j-wae.out  # Standard output log
#SBATCH --partition=physical-gpu                        # Partition/queue name
#SBATCH --cpus-per-task=32                      # CPU cores per task
#SBATCH --mem=32G                              # Memory per node

ENV_FILE=$1

if [ -z "$ENV_FILE" ]; then
    echo "Error: No environment file specified"
    exit 1
fi

if [ ! -f "$ENV_FILE" ]; then
    echo "Error: Environment file $ENV_FILE not found"
    exit 1
fi

# Load the specified environment file
source "$ENV_FILE"

# Create the artifact folder
mkdir -p "$ARTIFACT_FOLDER"

# --- Run Python script ---
python ../wa_experiment.py \
    --base-model-path "$BASE_MODEL_PATH" \
    --target-model-path "$TARGET_MODEL_PATH" \
    --artifact-out "$ARTIFACT_FOLDER"

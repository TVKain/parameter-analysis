# SLA Parameter Analysis Experiment

## Models
- https://huggingface.co/tktung/sla_cpt

## Usage

### Using python

```
usage: wa_experiment.py [-h] --base-model-path BASE_MODEL_PATH
                        --target-model-path TARGET_MODEL_PATH [--artifact-out ARTIFACT_OUT]

Compare models and plot heatmaps

options:
  -h, --help            show this help message and exit
  --base-model-path BASE_MODEL_PATH
                        Path to the base model
  --target-model-path TARGET_MODEL_PATH
                        Path to the target model folder
  --artifact-out ARTIFACT_OUT
                        Folder to save artifacts
```


### Using slurm

Run an experiment

```
cd job_scripts
sbatch wae.sh <path_to_env_file>
```

Example
```
sbatch sre.sh wae/qwen2.5-0.5B_5B-eng_1.5B-gle.sh
```

Run all experiments in `envs` folder

Note: need to configure paths for local models

```
cd job_scripts
sbatch wae_all.sh
```


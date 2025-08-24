"""
SLA Parameter Analysis

This script will:
- Load a base model and its checkpoint(s) during CPT to adapt for a target language
- Compare model components such as q_proj, v_proj, k_proj, gate_proj, up_proj, down_proj
- Compute 5 metrics for each component across layers:
    * cosine_sim
    * l1_norm
    * l1_norm_mean
    * l2_norm
    * l2_norm_mean
- Output:
    * A JSON file containing the differences between the base model and target model
    * Heatmap plot files for all components for each checkpoint
    * (Optional future) Embedding and LM head plots for the same metrics
"""

import os
import json
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from torch import Tensor
import torch.nn.functional as F
from transformers import AutoModelForCausalLM

# -----------------------------
# Configurable constants
# -----------------------------
COMPONENTS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
METRICS = ["cosine_sim", "l1_norm", "l1_norm_mean", "l2_norm", "l2_norm_mean"]

# -----------------------------
# Metric functions
# -----------------------------
def cosine_similarity(a: Tensor, b: Tensor) -> float:
    a_flat = a.flatten()
    b_flat = b.flatten()
    sim = F.cosine_similarity(a_flat, b_flat, dim=0).item()
    sim = max(min(sim, 1.0), -1.0)
    return (sim + 1) / 2

def l1_norm(a: Tensor, b: Tensor) -> float:
    return torch.abs(a - b).sum().item()

def l1_norm_mean(a: Tensor, b: Tensor) -> float:
    return torch.abs(a - b).mean().item()

def l2_norm(a: Tensor, b: Tensor) -> float:
    return torch.sqrt((a - b).pow(2).sum()).item()

def l2_norm_mean(a: Tensor, b: Tensor) -> float:
    return torch.sqrt((a - b).pow(2).mean()).item()

def compute_metrics(a: Tensor, b: Tensor) -> dict:
    return {
        "cosine_sim": cosine_similarity(a, b),
        "l1_norm": l1_norm(a, b),
        "l1_norm_mean": l1_norm_mean(a, b),
        "l2_norm": l2_norm(a, b),
        "l2_norm_mean": l2_norm_mean(a, b),
    }

# -----------------------------
# Model utility functions
# -----------------------------
def load_model(model_path: str, device_map="auto") -> AutoModelForCausalLM:
    print(f"[INFO] Loading model from {model_path}")
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )

def detect_checkpoints(model_dir: str) -> list[str]:
    import re
    pattern = re.compile(r"checkpoint-\d+")
    try:
        checkpoints = [os.path.join(model_dir, f) for f in os.listdir(model_dir) if pattern.match(f)]
    except Exception:
        return []
    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))
    return checkpoints

def get_model_name(model_path: str) -> str:
    return os.path.basename(os.path.normpath(model_path))

def generate_output_filename(base_model_path: str, target_model_path: str, output_dir: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = get_model_name(base_model_path)
    target_name = get_model_name(target_model_path)
    filename = f"{timestamp}_{base_name}_{target_name}.json"
    return os.path.join(output_dir, filename)

# -----------------------------
# Compare parameters and generate JSON
# -----------------------------
def compare_parameters(base_model_path: str, target_model_path: str, output_json: str):
    checkpoints = detect_checkpoints(target_model_path)
    if len(checkpoints) == 0:
        checkpoints = [target_model_path]

    base_model = load_model(base_model_path, device_map="auto")
    base_layers = base_model.model.layers
    base_embedding = base_model.model.embed_tokens.weight.detach().cpu()
    base_lm_head = base_model.lm_head.weight.detach().cpu()

    results = {
        "base_model": base_model_path,
        "target_model_folder": target_model_path,
        "checkpoints": {},
    }

    for checkpoint in checkpoints:
        print(f"[INFO] Comparing with {checkpoint}")
        target_model = load_model(checkpoint, device_map="cpu")
        target_layers = target_model.model.layers
        target_embedding = target_model.model.embed_tokens.weight.detach().cpu()
        target_lm_head = target_model.lm_head.weight.detach().cpu()

        checkpoint_results = {
            "embedding": compute_metrics(base_embedding, target_embedding),
            "lm_head": compute_metrics(base_lm_head, target_lm_head),
            "layers": {},
        }

        for layer_idx, (base_layer, target_layer) in enumerate(zip(base_layers, target_layers)):
            layer_results = {}
            for comp in COMPONENTS:
                if hasattr(base_layer.self_attn, comp):  # Attention
                    base_tensor = getattr(base_layer.self_attn, comp).weight.detach().cpu()
                    target_tensor = getattr(target_layer.self_attn, comp).weight.detach().cpu()
                elif hasattr(base_layer.mlp, comp):  # MLP
                    base_tensor = getattr(base_layer.mlp, comp).weight.detach().cpu()
                    target_tensor = getattr(target_layer.mlp, comp).weight.detach().cpu()
                else:
                    continue
                layer_results[comp] = compute_metrics(base_tensor, target_tensor)

            checkpoint_results["layers"][f"layer_{layer_idx}"] = layer_results

        results["checkpoints"][checkpoint] = checkpoint_results

        del target_model
        torch.cuda.empty_cache()

    with open(output_json, "w") as f:
        json.dump(results, f, indent=4)
    print(f"[INFO] Results saved to {output_json}")
    return results

# -----------------------------
# Plotting function
# -----------------------------
def plot_checkpoint_heatmaps(json_path: str, output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    checkpoints = data.get("checkpoints", {})

    for ckpt_path, ckpt_data in checkpoints.items():
        layers = ckpt_data.get("layers", {})
        layer_names = sorted(layers.keys(), key=lambda x: int(''.join(filter(str.isdigit, x))))
        num_layers = len(layer_names)

        metric_matrices = {metric: np.zeros((num_layers, len(COMPONENTS))) for metric in METRICS}
        reversed_layer_names = list(reversed(layer_names))

        for i, layer_name in enumerate(reversed_layer_names):
            layer = layers[layer_name]
            for j, comp in enumerate(COMPONENTS):
                comp_data = layer.get(comp, {})
                for metric in METRICS:
                    metric_matrices[metric][i, j] = comp_data.get(metric, np.nan)

        fig, axes = plt.subplots(1, 5, figsize=(64, max(10, num_layers * 0.6)))
        fig.suptitle(f"Metrics Heatmaps - Checkpoint: {os.path.basename(ckpt_path)}", fontsize=16)
        plt.subplots_adjust(wspace=0.5)

        for idx, (metric, data_matrix) in enumerate(metric_matrices.items()):
            vmin = np.nanmin(data_matrix)
            vmax = np.nanmax(data_matrix)

            sns.heatmap(
                data_matrix,
                annot=True,
                fmt=".6f",
                cmap="YlGnBu",
                xticklabels=COMPONENTS,
                yticklabels=reversed_layer_names if idx == 0 else False,
                vmin=vmin,
                vmax=vmax,
                ax=axes[idx],
                square=False,
                cbar_kws={'shrink': 0.6}
            )

            axes[idx].set_aspect(1.0 / 4.0)
            axes[idx].set_title(f"{metric}")
            axes[idx].set_xlabel("Component")
            if idx == 0:
                axes[idx].set_ylabel("Layer")
            axes[idx].tick_params(axis='x', rotation=45)

        plt.tight_layout()
        plot_file = os.path.join(output_dir, f"{os.path.basename(ckpt_path)}_heatmap_qkvo_mlp.png")
        plt.savefig(plot_file, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved combined heatmap to {plot_file}")
def plot_embedding_and_lmhead_separately(json_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    checkpoints = data.get("checkpoints", {})
    if not checkpoints:
        print("[WARN] No checkpoints found in JSON.")
        return

    # Extract checkpoint numbers from paths
    ckpt_paths = list(checkpoints.keys())
    ckpt_numbers = []
    ckpt_labels = []

    for path in ckpt_paths:
        base = os.path.basename(path)  # e.g., "checkpoint-1000"
        digits = ''.join(filter(str.isdigit, base))
        ckpt_number = int(digits) if digits else 0
        ckpt_numbers.append(ckpt_number)
        ckpt_labels.append(str(ckpt_number))  # string for x-axis labels

    # Sort all lists by checkpoint number
    ckpt_numbers, ckpt_paths, ckpt_labels = zip(*sorted(zip(ckpt_numbers, ckpt_paths, ckpt_labels)))

    # Prepare metric data for embedding and lm_head separately
    embedding_data = {metric: [] for metric in METRICS}
    lm_head_data = {metric: [] for metric in METRICS}

    for path in ckpt_paths:
        ckpt_data = checkpoints[path]
        embedding_metrics = ckpt_data.get("embedding", {})
        lm_head_metrics = ckpt_data.get("lm_head", {})

        for metric in METRICS:
            embedding_data[metric].append(embedding_metrics.get(metric, np.nan))
            lm_head_data[metric].append(lm_head_metrics.get(metric, np.nan))

    # Plot embedding metrics
    fig, axes = plt.subplots(1, 5, figsize=(25, 6))
    fig.suptitle("Embedding Metrics Across Checkpoints", fontsize=16)
    plt.subplots_adjust(wspace=0.4)

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        ax.plot(ckpt_labels, embedding_data[metric], marker='o', color='blue')
        ax.set_title(metric)
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Value")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    embedding_file = os.path.join(output_dir, "embedding.png")
    plt.savefig(embedding_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved embedding metrics plot to {embedding_file}")

    # Plot LM head metrics
    fig, axes = plt.subplots(1, 5, figsize=(25, 6))
    fig.suptitle("LM Head Metrics Across Checkpoints", fontsize=16)
    plt.subplots_adjust(wspace=0.4)

    for idx, metric in enumerate(METRICS):
        ax = axes[idx]
        ax.plot(ckpt_labels, lm_head_data[metric], marker='o', color='orange')
        ax.set_title(metric)
        ax.set_xlabel("Checkpoint")
        ax.set_ylabel("Value")
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    lmhead_file = os.path.join(output_dir, "lmhead.png")
    plt.savefig(lmhead_file, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[INFO] Saved LM Head metrics plot to {lmhead_file}")

def plot_l1mean_vs_cosine(json_path: str, output_dir: str = "plots"):
    import matplotlib.pyplot as plt
    import os
    import json
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    checkpoints = data.get("checkpoints", {})
    if not checkpoints:
        print("[WARN] No checkpoints found in JSON.")
        return

    # Extract checkpoint numbers
    ckpt_paths = list(checkpoints.keys())
    ckpt_numbers = []
    for path in ckpt_paths:
        base = os.path.basename(path)
        digits = ''.join(filter(str.isdigit, base))
        ckpt_numbers.append(int(digits) if digits else 0)

    # Sort checkpoints
    sorted_ckpt_data = sorted(zip(ckpt_numbers, ckpt_paths))
    ckpt_numbers, ckpt_paths = zip(*sorted_ckpt_data)

    # Loop over layers
    example_ckpt = checkpoints[ckpt_paths[0]]
    layer_keys = sorted(example_ckpt['layers'].keys(), key=lambda x: int(x.split('_')[-1]))

    for layer in layer_keys:
        fig, axes = plt.subplots(len(COMPONENTS), 1, figsize=(10, 3 * len(COMPONENTS)), sharex=True)
        fig.suptitle(f"{layer}: L1 Norm Mean vs Cosine Similarity", fontsize=16)

        for i, comp in enumerate(COMPONENTS):
            l1_mean_vals = []
            cos_vals = []

            for path in ckpt_paths:
                layer_data = checkpoints[path]['layers'][layer]
                comp_data = layer_data.get(comp, {})
                l1_mean_vals.append(comp_data.get('l1_norm_mean', float('nan')))
                cos_vals.append(comp_data.get('cosine_sim', float('nan')))

            ax = axes[i] if len(COMPONENTS) > 1 else axes
            ax.plot(ckpt_numbers, l1_mean_vals, marker='o', color='blue', label="L1 Norm Mean")
            ax.set_ylabel("L1 Norm Mean", color='blue')
            ax.tick_params(axis='y', labelcolor='blue')
            ax.grid(True)

            ax2 = ax.twinx()
            ax2.plot(ckpt_numbers, cos_vals, marker='x', color='orange', linestyle='--', label="Cosine Similarity")
            ax2.set_ylabel("Cosine Similarity", color='orange')
            ax2.tick_params(axis='y', labelcolor='orange')

            ax.set_title(comp)
            ax.set_xlabel("Checkpoint Step")

        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out_file = os.path.join(output_dir, f"{layer}_l1mean_vs_cosine.png")
        plt.savefig(out_file, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved plot {out_file}")

# -----------------------------
# Main entry point
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Compare models and plot heatmaps")
    parser.add_argument("--base-model-path", type=str, required=True, help="Path to the base model")
    parser.add_argument("--target-model-path", type=str, required=True, help="Path to the target model folder")
    parser.add_argument("--artifact-out", type=str, default="artifacts", help="Folder to save artifacts")
    args = parser.parse_args()

    os.makedirs(args.artifact_out, exist_ok=True)
    output_json = generate_output_filename(args.base_model_path, args.target_model_path, args.artifact_out)

    # Compare and save JSON
    compare_parameters(args.base_model_path, args.target_model_path, output_json)

    # Generate heatmap plots
    plot_checkpoint_heatmaps(output_json, args.artifact_out)

    # Generate embedding and ml head plots
    plot_embedding_and_lmhead_separately(output_json, args.artifact_out)

    # Generate L1 Norm mean vs Cosine similarity for QKVO MLP matrices
    plot_l1mean_vs_cosine(output_json, args.artifact_out)

if __name__ == "__main__":
    main()

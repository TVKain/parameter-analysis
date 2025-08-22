import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

COMPONENTS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
METRICS = ["cosine_sim", "l1_norm", "l1_norm_mean", "l2_norm", "l2_norm_mean"]

def plot_checkpoint_heatmaps(json_path: str, output_dir: str = "plots"):
    os.makedirs(output_dir, exist_ok=True)

    with open(json_path, "r") as f:
        data = json.load(f)

    checkpoints = data.get("checkpoints", {})

    for ckpt_path, ckpt_data in checkpoints.items():
        layers = ckpt_data.get("layers", {})
        layer_names = sorted(layers.keys(), key=lambda x: int(''.join(filter(str.isdigit, x))))
        num_layers = len(layer_names)

        # Prepare matrices per metric (reverse layer order)
        metric_matrices = {metric: np.zeros((num_layers, len(COMPONENTS))) for metric in METRICS}
        reversed_layer_names = list(reversed(layer_names))

        for i, layer_name in enumerate(reversed_layer_names):
            layer = layers[layer_name]
            for j, comp in enumerate(COMPONENTS):
                comp_data = layer.get(comp, {})
                for metric in METRICS:
                    metric_matrices[metric][i, j] = comp_data.get(metric, np.nan)

        # Create a single figure with 5 subplots (one for each metric)
        fig, axes = plt.subplots(1, 5, figsize=(64, max(10, num_layers*0.6)))
        fig.suptitle(f"Metrics Heatmaps - Checkpoint: {os.path.basename(ckpt_path)}", fontsize=16)
        
        # Add space between subplots
        plt.subplots_adjust(wspace=0.5)

        for idx, (metric, data_matrix) in enumerate(metric_matrices.items()):
            vmin = np.nanmin(data_matrix)
            vmax = np.nanmax(data_matrix)

            # Set cell aspect ratio to make cells wider
            cell_width = 4  # Make cells 1.5 times wider than tall
            cell_height = 1.0
            
            sns.heatmap(
                data_matrix,
                annot=True,
                fmt=".6f",  
                cmap="YlGnBu",
                xticklabels=COMPONENTS,
                yticklabels=reversed_layer_names if idx == 0 else False,  # Show y-labels only on first plot
                vmin=vmin,
                vmax=vmax,
                ax=axes[idx],
                square=False,  # Allow rectangular cells
                cbar_kws={'shrink': 0.6}  # Smaller colorbar
            )
            
            # Manually set the aspect ratio to make cells wider
            axes[idx].set_aspect(cell_height/cell_width)
            axes[idx].set_title(f"{metric}")
            axes[idx].set_xlabel("Component")
            if idx == 0:
                axes[idx].set_ylabel("Layer")
            
            # Rotate x-axis labels for better readability
            axes[idx].tick_params(axis='x', rotation=45)

        plt.tight_layout()

        plot_file = os.path.join(output_dir, f"{os.path.basename(ckpt_path)}_all_metrics.png")
        plt.savefig(plot_file, dpi=200, bbox_inches='tight')
        plt.close()
        print(f"[INFO] Saved combined heatmap to {plot_file}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Plot Qwen parameter heatmaps")
    parser.add_argument("--json", type=str, required=True, help="Path to JSON artifact")
    parser.add_argument("--out-dir", type=str, default="plots", help="Directory to save plots")
    args = parser.parse_args()

    plot_checkpoint_heatmaps(args.json, args.out_dir)
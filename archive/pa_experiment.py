"""
SLA Parameter Analysis

This script will load a base model and its checkpoint during CPT to adapt for a target language

The output of this script will be
- A JSON file containing all the difference of the model's components such as q_proj, v_proj, k_proj, mlp_gate, mlp_up, mlp_down
- Heatmap plot files for the q_proj, v_proj, k_proj, mlp_gate, mlp_up, mlp_down components for each checkpoint
- 1 input embedding plot for 5 metrics
- 1 ml head plot for 5 metrics  
"""

from datetime import datetime
import json
import os
import torch
from torch import Tensor

import torch.nn.functional as F

import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer


def load_model(model_path: str, device_map="auto") -> AutoModelForCausalLM:
    """Load a model"""

    print(f"Loading model from {model_path}")
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map=device_map,
    )


def cosine_similarity(a: Tensor, b: Tensor) -> float:
    """
    Flatten cosine similarity
    """
    a_flat = a.flatten()
    b_flat = b.flatten()

    sim = F.cosine_similarity(a_flat, b_flat, dim=0).item()

    sim = max(min(sim, 1.0), -1.0)

    return (sim + 1) / 2


def l1_norm(a: Tensor, b: Tensor) -> float:
    """
    L1 norm
    """
    return torch.abs(a - b).sum().item()


def l1_norm_mean(a: Tensor, b: Tensor) -> float:
    """
    L1 norm mean
    """
    return torch.abs(a - b).mean().item()


def l2_norm(a: Tensor, b: Tensor) -> float:
    """
    L2 norm
    """
    return torch.sqrt((a - b).pow(2).sum()).item()


def l2_norm_mean(a: Tensor, b: Tensor) -> float:
    """
    L2 norm mean
    """
    return torch.sqrt((a - b).pow(2).mean()).item()


def get_model_name(model_path: str) -> str:
    """Extract model name from path"""
    return os.path.basename(os.path.normpath(model_path))


def generate_output_filename(
    base_model_path: str, target_model_path: str, output_dir: str
) -> str:
    """Generate JSON filename with datetime and model names"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = get_model_name(base_model_path)
    target_name = get_model_name(target_model_path)
    filename = f"{timestamp}_{base_name}_{target_name}.json"
    return os.path.join(output_dir, filename)


def detect_check_points(model_dir: str) -> list[str]:
    """
    Detects all checkpoint files in the given model directory.

    Args:
        model_dir (str): The directory where the model checkpoints are stored.

    Returns:
        list[str]: A list of checkpoint file paths.
    """
    import os
    import re

    """
    Pattern is checkpoint-<epoch> folder
    Example: checkpoint-1000
    """

    # if not os.path.exists(model_dir):
    #     raise FileNotFoundError(f"The directory {model_dir} does not exist.", model_dir)
    # if not os.path.isdir(model_dir):
    #     raise NotADirectoryError(f"The path {model_dir} is not a directory.", model_dir)
    pattern = re.compile(r"checkpoint-\d+")

    try:
        checkpoints = [
            os.path.join(model_dir, f)
            for f in os.listdir(model_dir)
            if pattern.match(f)
        ]
    except Exception:
        # Hack
        return []

    checkpoints.sort(key=lambda x: int(x.split("-")[-1]))  # Sort by epoch number

    return checkpoints


def compute_metrics(a: Tensor, b: Tensor) -> dict:
    return {
        "cosine_sim": cosine_similarity(a, b),
        "l1_norm": l1_norm(a, b),
        "l1_norm_mean": l1_norm_mean(a, b),
        "l2_norm": l2_norm(a, b),
        "l2_norm_mean": l2_norm_mean(a, b),
    }


def compare_parameters(
    base_model_path: str, target_model_path: str, output_json: str = None
):
    """
    Compare input embeddings, LM head, and layer weights between base model and checkpoints in target_model_path.
    """

    # Detect checkpoints
    checkpoints = detect_check_points(target_model_path)

    if len(checkpoints) == 0:
        # make a list with the target_model_path
        # note: hack
        checkpoints = [target_model_path]

    # Load base model
    base_model = load_model(base_model_path, device_map="cpu")
    base_layers = base_model.model.layers
    base_embedding = base_model.model.embed_tokens.weight.detach().cpu()
    base_lm_head = base_model.lm_head.weight.detach().cpu()

    components = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "up_proj",
        "gate_proj",
        "down_proj",
    ]

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

        for layer_idx, (base_layer, target_layer) in enumerate(
            zip(base_layers, target_layers)
        ):
            layer_results = {}
            for comp in components:
                if hasattr(base_layer.self_attn, comp):  # Attention
                    base_tensor = (
                        getattr(base_layer.self_attn, comp).weight.detach().cpu()
                    )
                    target_tensor = (
                        getattr(target_layer.self_attn, comp).weight.detach().cpu()
                    )
                elif hasattr(base_layer.mlp, comp):  # MLP
                    base_tensor = getattr(base_layer.mlp, comp).weight.detach().cpu()
                    target_tensor = (
                        getattr(target_layer.mlp, comp).weight.detach().cpu()
                    )
                else:
                    continue

                layer_results[comp] = compute_metrics(base_tensor, target_tensor)

            checkpoint_results["layers"][f"layer_{layer_idx}"] = layer_results

        results["checkpoints"][checkpoint] = checkpoint_results

        # Free memory
        del target_model
        torch.cuda.empty_cache()

    # File name should be <datetime>_<base_model_name>_<target_model_name>.json

    # Save JSON artifact
    if output_json:
        os.makedirs(os.path.dirname(output_json), exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(results, f, indent=4)
        print(f"[INFO] Results saved to {output_json}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Sentence retrieval task")
    parser.add_argument(
        "--base-model-path", type=str, required=True, help="Path to the base model"
    )
    parser.add_argument(
        "--target-model-path",
        type=str,
        required=True,
        help="Path to the target model folder",
    )
    parser.add_argument(
        "--json-artifact-out",
        type=str,
        default="json_artifacts",
        help="Path to JSON accuracy artifacts folder",
    )
    args = parser.parse_args()

    os.makedirs(args.json_artifact_out, exist_ok=True)
    output_json = generate_output_filename(
        args.base_model_path, args.target_model_path, args.json_artifact_out
    )

    compare_parameters(args.base_model_path, args.target_model_path, output_json)


if __name__ == "__main__":
    main()

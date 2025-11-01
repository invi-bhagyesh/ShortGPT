import os
import datasets

# SET THIS FIRST - before any other imports or operations
datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import lm_eval
import argparse

from short_hf import ShortHFModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Prune layers from a language model and evaluate it."
    )
    
    parser.add_argument(
        "--model_name",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="HuggingFace model name or path (default: TinyLlama/TinyLlama-1.1B-Chat-v1.0)"
    )
    
    parser.add_argument(
        "--n_prune_layers",
        type=int,
        default=3,
        help="Number of top layers to prune (default: 3)"
    )
    
    parser.add_argument(
        "--eval_tasks",
        type=str,
        nargs="+",
        default=["truthfulqa_mc"],
        help="Evaluation tasks (space-separated list, e.g., 'truthfulqa_mc hellaswag piqa')"
    )
    
    parser.add_argument(
        "--layers_path",
        type=str,
        default="model.layers",
        help="Path to model layers attribute (default: model.layers)"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for evaluation (default: 1)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./pruned_model",
        help="Directory to save pruned model (default: ./pruned_model)"
    )
    
    parser.add_argument(
        "--num_fewshot",
        type=int,
        default=0,
        help="Number of few-shot examples for evaluation (default: 0)"
    )
    
    return parser.parse_args()


def main():
    # Parse command line arguments
    args = parse_args()
    
    print(f"Initializing {args.model_name}...")
    short_model = ShortHFModel(
        model_name=args.model_name,
        layers_path=args.layers_path,
        n_prune_layers=args.n_prune_layers
    )
    
    print(f"Total layers before pruning: {len(short_model.layers)}")
    
    # Remove top N layers WITHOUT computing block influence
    print(f"\nRemoving top {args.n_prune_layers} layers (no BI computation)...")
    # Specify which layers to remove - top N layers means the last N layers
    total_layers = len(short_model.layers)
    layers_to_remove = list(range(total_layers - args.n_prune_layers, total_layers))
    
    print(f"Removing layers: {layers_to_remove}")
    
    # Remove layers manually
    for layer_idx in sorted(layers_to_remove, reverse=True):
        del short_model.layers[layer_idx]
    
    print(f"Remaining layers: {len(short_model.layers)}")
    
    # Save the pruned model temporarily
    print(f"\nSaving pruned model to {args.output_dir}...")
    short_model.model.save_pretrained(args.output_dir)
    short_model.tokenizer.save_pretrained(args.output_dir)
    
    # Evaluate using the saved model path directly
    print(f"\nEvaluating pruned model on tasks: {', '.join(args.eval_tasks)}...")
    results = lm_eval.simple_evaluate(
        model="hf",
        model_args=f"pretrained={args.output_dir},dtype=float16",
        tasks=args.eval_tasks,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device="cuda"
    )
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS (Top N Layers Removed)")
    print("="*60)
    print(f"Model: {args.model_name}")
    print(f"Removed layers: {layers_to_remove}")
    print(f"Remaining layers: {len(short_model.layers)}")
    print(f"Tasks: {', '.join(args.eval_tasks)}")
    print("\nResults:")
    print(results)


if __name__ == "__main__":
    main()

import os
import datasets

# SET THIS FIRST - before any other imports or operations
datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import lm_eval
from lm_eval.utils import make_table  # Add this import
import argparse

from short_hf import ShortHFModel


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prune layers from a language model and evaluate it.")
    parser.add_argument("--model_name", type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0", help="HuggingFace model name or path")
    parser.add_argument("--n_prune_layers", type=int, default=3, help="Number of top layers to prune")
    parser.add_argument("--eval_tasks", type=str, nargs="+", default=["truthfulqa_mc"], help="Evaluation tasks (space-separated list)")
    parser.add_argument("--layers_path", type=str, default="model.layers", help="Path to model layers attribute")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for evaluation")
    parser.add_argument("--output_dir", type=str, default="./pruned_model", help="Directory to save pruned model")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples for evaluation")
    parser.add_argument("--dataset", type=str, default=None, help="dataset name for DPO like 'crows-pairs'")
    parser.add_argument("--dpo_path", type=str, default=None, help="path to an existing DPO fine-tuned model; if provided, DPO training will be skipped")
    return parser.parse_args()


def main():
    args = parse_args()

###########################################################################
    # ------------------ DPO fine-tuning ------------------ #
    if args.dataset == "crows-pairs":
        from transformers import AutoTokenizer, AutoModelForCausalLM
        from trl import DPOTrainer, DPOConfig
        import pandas as pd
        from datasets import load_dataset, Dataset

        if args.dpo_path:
            print(f"Loading existing DPO model from {args.dpo_path}...")
            model = AutoModelForCausalLM.from_pretrained(args.dpo_path)
            tokenizer = AutoTokenizer.from_pretrained(args.dpo_path)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
        else:
            print("Preparing crows-pairs dataset for DPO fine-tuning...")
            ds = load_dataset("BigScienceBiasEval/crows_pairs_multilingual")["test"]
            pairs = []
            for item in ds:
                chosen, rejected = item["sent_less"], item["sent_more"]
                if item["stereo_antistereo"] == "anti-stereotype":
                    chosen, rejected = rejected, chosen
                pairs.append({
                    "prompt": f"Bias type: {item['bias_type']}",
                    "chosen": chosen,
                    "rejected": rejected,
                    "bias_type": item["bias_type"]
                })

            pairs_df = pd.DataFrame(pairs)
            train_data = Dataset.from_list([{
                "prompt": row["prompt"],
                "chosen": row["chosen"],
                "rejected": row["rejected"]
            } for _, row in pairs_df.iterrows()])

            tokenizer = AutoTokenizer.from_pretrained(args.model_name)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            model = AutoModelForCausalLM.from_pretrained(args.model_name)

            def preprocess(example):
                return {
                    "prompt_ids": tokenizer(example["prompt"], truncation=True, padding="max_length", max_length=512)["input_ids"],
                    "chosen_ids": tokenizer(example["chosen"], truncation=True, padding="max_length", max_length=512)["input_ids"],
                    "rejected_ids": tokenizer(example["rejected"], truncation=True, padding="max_length", max_length=512)["input_ids"],
                }

            train_data = train_data.map(preprocess, batched=False)

            training_args = DPOConfig(
                output_dir=f"./dpo_finetuned_{args.eval_tasks}",
                num_train_epochs=1,
                per_device_train_batch_size=2,
                per_device_eval_batch_size=2,
                remove_unused_columns=False,
                logging_steps=10,
                gradient_accumulation_steps=1,
                learning_rate=5e-6,
                warmup_steps=2,
                fp16=False,
                save_steps=500,
                eval_strategy="no",
                report_to='none'
            )

            trainer = DPOTrainer(
                model=model,
                ref_model=None,
                args=training_args,
                processing_class=tokenizer,
                train_dataset=train_data,
                eval_dataset=None
            )

            print("Starting DPO fine-tuning...")
            trainer.train()
            model.save_pretrained(training_args.output_dir)
            tokenizer.save_pretrained(training_args.output_dir)
            print("DPO fine-tuning complete.")

    # ------------------ Layer pruning ------------------ #
    print(f"\nInitializing ShortHFModel for pruning...")
    short_model = ShortHFModel(
        model_name=training_args.output_dir if not args.dpo_path else args.dpo_path,
        layers_path=args.layers_path,
        n_prune_layers=args.n_prune_layers
    )
###########################################################################

    
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
    
    # Print formatted table with all metrics
    print("\n" + make_table(results))
    
    # Print detailed results for each task
    print("\nDetailed Metrics:")
    for task_name in args.eval_tasks:
        if task_name in results['results']:
            print(f"\n{task_name}:")
            task_results = results['results'][task_name]
            for metric, value in task_results.items():
                if not metric.endswith('_stderr'):
                    stderr = task_results.get(f"{metric}_stderr", None)
                    if stderr is not None:
                        print(f"  {metric}: {value:.4f} ± {stderr:.4f}")
                    else:
                        print(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import lm_eval

from short_hf import ShortHFModel

MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LAYERS_PATH = "model.layers"
N_PRUNE_LAYERS = 4  
BATCH_SIZE = 1

# Evaluation tasks
EVAL_TASKS = ["winogrande"]

print(f"Initializing {MODEL_NAME}...")
short_model = ShortHFModel(
    model_name=MODEL_NAME,
    layers_path=LAYERS_PATH,
    n_prune_layers=N_PRUNE_LAYERS
)

print(f"Total layers before pruning: {len(short_model.layers)}")

# Remove top N layers WITHOUT computing block influence
print("\nRemoving top N layers (no BI computation)...")
# Specify which layers to remove - top N layers means the last N layers
total_layers = len(short_model.layers)
layers_to_remove = list(range(total_layers - N_PRUNE_LAYERS, total_layers))

print(f"Removing layers: {layers_to_remove}")

# Remove layers manually
for layer_idx in sorted(layers_to_remove, reverse=True):
    del short_model.layers[layer_idx]

print(f"Remaining layers: {len(short_model.layers)}")

# Save the pruned model temporarily
print("\nSaving pruned model...")
pruned_model_path = "./pruned_tinyllama_top_layers"
short_model.model.save_pretrained(pruned_model_path)
short_model.tokenizer.save_pretrained(pruned_model_path)

# Evaluate using the saved model path directly
print("\nEvaluating pruned model with LM Evaluation Harness...")
results = lm_eval.simple_evaluate(
    model="hf",
    model_args=f"pretrained={pruned_model_path},dtype=float16",
    tasks=EVAL_TASKS,
    num_fewshot=0,
    batch_size=BATCH_SIZE,
    device="cuda"
)

print("\n" + "="*60)
print("EVALUATION RESULTS (Top N Layers Removed)")
print("="*60)
print(f"Removed layers: {layers_to_remove}")
print(f"Remaining layers: {len(short_model.layers)}")
print("\nResults:")
print(results)

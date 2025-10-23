import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import lm_eval
from lm_eval.models.huggingface import HFLM

from short_hf import ShortHFModel
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LAYERS_PATH = "model.layers"
N_PRUNE_LAYERS = 3  
MAX_SEQ_LEN = 1024
STRIDE = 256
BATCH_SIZE = 1
N_CALIBRATION_SAMPLES = 1000  

# Evaluation tasks
EVAL_TASKS = ["winogrande"]

print("Loading calibration dataset...")
data = load_dataset("pg19", split="validation", trust_remote_code=True)
data = data.select(range(min(N_CALIBRATION_SAMPLES, len(data))))


dataloader = DataLoader(
    data,
    batch_size=BATCH_SIZE,
    shuffle=True,
    generator=torch.Generator().manual_seed(42)  
)

print(f"Initializing {MODEL_NAME}...")
short_model = ShortHFModel(
    model_name=MODEL_NAME,
    layers_path=LAYERS_PATH,
    n_prune_layers=N_PRUNE_LAYERS
)

print(f"Total layers before pruning: {len(short_model.layers)}")

print("Computing block influence scores...")
for batch in tqdm(dataloader, desc="Processing calibration data"):
    prompts = batch['text']
    
    short_model.eval_importance(
        prompts=prompts,
        max_seq_len=MAX_SEQ_LEN,
        stride=STRIDE,
        max_gen_len=0,  # No generation during importance computation
        angular=False  # Set to True for angular distance method
    )


print("\nRemoving layers based on importance scores...")
removed_layers = short_model.remove_layers()
print(f"Removed layers: {removed_layers}")
print(f"Remaining layers: {len(short_model.layers)}")

# Save the pruned model temporarily
print("\nSaving pruned model...")
pruned_model_path = "./pruned_tinyllama_temp"
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
print("EVALUATION RESULTS")
print("="*60)
print(f"Removed layers: {removed_layers}")
print(f"Remaining layers: {len(short_model.layers)}")
print("\nResults:")
print(results)
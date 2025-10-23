import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from tqdm import tqdm
import lm_eval
from lm_eval.models.huggingface import HFLM

from short_hf import ShortHFModel
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
LAYERS_PATH = "model.layers"
N_PRUNE_LAYERS = 4  
MAX_SEQ_LEN = 1024
STRIDE = 256
BATCH_SIZE = 1
N_CALIBRATION_SAMPLES = 1000  

# Evaluation tasks
EVAL_TASKS = ["winogrande"]

print("Loading calibration dataset...")
# data = load_dataset("pg19", split="validation", trust_remote_code=True)
# data = data.select(range(min(N_CALIBRATION_SAMPLES, len(data))))
data = load_dataset("ag_news", split="test")
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

print("\nEvaluating pruned model with LM Evaluation Harness...")

# Wrap the pruned model for lm-eval
class PrunedHFLM(HFLM):
    """Custom wrapper for pruned model evaluation"""
    def __init__(self, pruned_model, tokenizer, **kwargs):
        # Initialize parent class attributes
        self._model = pruned_model
        self.tokenizer = tokenizer
        self._device = pruned_model.device
        self._batch_size = kwargs.get('batch_size', 1)
        self._max_length = kwargs.get('max_length', 2048)
        
    @property
    def model(self):
        return self._model
    
    @property
    def eot_token_id(self):
        return self.tokenizer.eos_token_id
    
    @property
    def max_length(self):
        return self._max_length
    
    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def device(self):
        return self._device

# Create wrapper
pruned_lm = PrunedHFLM(
    pruned_model=short_model.model,
    tokenizer=short_model.tokenizer,
    batch_size=BATCH_SIZE
)

results = lm_eval.simple_evaluate(
    model=pruned_lm,
    tasks=EVAL_TASKS,
    num_fewshot=0,
    batch_size=BATCH_SIZE,
    device="cuda"
)

print(results)
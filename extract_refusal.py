import torch
from transformers import AutoModelForCausalLM

print("Loading Instruct (Safe) weights...")
safe_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float32)

print("Loading Abliterated (Uncensored) weights...")
abl_model = AutoModelForCausalLM.from_pretrained("huihui-ai/Qwen2.5-0.5B-Instruct-abliterated-v3", torch_dtype=torch.float32)

layer_idx = 14
# Extract the MLP down-projection matrices
W_orig = safe_model.model.layers[layer_idx].mlp.down_proj.weight.detach()
W_abl = abl_model.model.layers[layer_idx].mlp.down_proj.weight.detach()

# Calculate the difference matrix
delta_W = W_orig - W_abl

# Run SVD to isolate the rank-1 refusal direction
U, S, V = torch.svd(delta_W)

# The first left singular vector is your exact Refusal Vector
refusal_vector = U[:, 0]

print(f"Refusal Vector extracted perfectly. Shape: {refusal_vector.shape}")
torch.save(refusal_vector, "data/svd_refusal_vector.pt")

# Free up RAM immediately
del safe_model, abl_model, W_orig, W_abl, delta_W
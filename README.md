# dynamic-steering — extraction utilities

Quick setup and usage notes for extracting residual-stream vectors h^{(l)} from a
HookedTransformer-wrapped model.

Files added
- [requirements.txt](requirements.txt): python packages (note: install PyTorch separately)
- [.env.template](.env.template): copy to `.env` and add `HUGGINGFACE_HUB_TOKEN`
- [setup_model.py](setup_model.py): helper to load `HookedTransformer`
- [data_prep.py](data_prep.py): `ContrastiveDataset` and `prepare_dataloader()`
- [extract_h_vectors.py](extract_h_vectors.py): demo extraction loop

Setup (conda `interp` environment)

1. Activate your environment:

```
conda activate interp
```

2. Install a matching PyTorch wheel manually (pick the one for your CUDA or CPU/MPS):

CUDA 12.1 example:

```
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Or CPU-only:

```
pip install torch torchvision torchaudio
```

3. Install the remaining Python packages:

```
pip install -r requirements.txt
```

4. Create `.env` from the template and add your Hugging Face token:

```
cp .env.template .env
# edit .env and set HUGGINGFACE_HUB_TOKEN=...
```

Run a quick smoke test

```
python setup_model.py
python extract_h_vectors.py
```

Notes
- These scripts assume a HookedTransformer-compatible model and tokenizer.
- The `extract_h_vectors.py` script inspects the run cache to find residual-stream
  hook keys and saves final-token vectors per layer to `h_vectors.pkl`.
- If loading the model requires more VRAM than available, consider using
  `device_map` or running on a host with a larger GPU.

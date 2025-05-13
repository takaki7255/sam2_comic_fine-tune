import torch, platform, subprocess, os

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("Memory (GB):", torch.cuda.get_device_properties(0).total_memory/1e9)
import torch

print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("CUDA version used by PyTorch:", torch.version.cuda)
    print(
        "GPU memory:",
        round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
        "GB"
    )

    x = torch.randn(3000, 3000, device="cuda")
    y = x @ x
    torch.cuda.synchronize()

    print("GPU computation successful:", y.device)
else:
    print("PyTorch is currently unable to access a CUDA GPU.")
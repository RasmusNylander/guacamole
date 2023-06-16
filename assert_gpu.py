import torch

def assert_gpu() -> torch.device:
    assert torch.cuda.is_available(), f"Hardware acceleration is not available"
    return torch.device("cuda")

import torch

def test_gpu():
    assert torch.cuda.is_available(), "gpu unavailable"
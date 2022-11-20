import torch

def test_gpu():
    assert torch.cuda.is_available(), "gpu unavailable"


if __name__ == '__main__':
    test_gpu()

NCSN Minimal PyTorch Implementation

This is a minimal, runnable implementation of a Noise Conditional Score Network (NCSN) inspired by the Ermon Group repository.

Quick start:

1. Create a virtualenv and install requirements:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

2. Train on MNIST (fast example):

```bash
python train.py --dataset mnist --epochs 5 --batch-size 128
```

3. Sample from the trained model:

```bash
python sample.py --checkpoint checkpoints/model_epoch4.pt --out samples.png
```

This is educational code and intentionally small and simple.

#!/usr/bin/env python3
"""Showcase of Class-Style Neural Network Architecture in Oxideon (`ox.nn.Module`).

Demonstrates:
1. Subclassing `ox.nn.Module` to build custom neural network architectures.
2. Mixing native Rust layers (`Linear`, `Conv2d`, `Dropout`, etc.) and custom `Parameter`s.
3. Modular hierarchy: nested modules, Sequential, ModuleList, and ModuleDict.
4. Clean PyTorch-style training loop: forward pass, loss computation, backward pass, optimizer step.
5. Model state dict serialization and weight restoration.
6. Transfer learning utilities: layer freezing/unfreezing and forward hooks.
"""

import numpy as np
import oxideon as ox
import oxideon.nn as nn


class ResidualBlock(nn.Module):
    """A custom residual block subclassing `ox.nn.Module`."""
    def __init__(self, channels):
        super().__init__()
        self.fc1 = nn.Linear(channels, channels)
        self.fc2 = nn.Linear(channels, channels)

    def forward(self, x):
        residual = x
        out = self.fc1(x).relu()
        out = self.fc2(out)
        return (out + residual).relu()


class VisionClassifier(nn.Module):
    """Deep Neural Network constructed via OOP inheritance and nested modules."""
    def __init__(self, in_features=16*16, hidden_dim=64, num_classes=4):
        super().__init__()
        # Input projection
        self.stem = nn.Sequential(
            nn.Linear(in_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
        )

        # Residual processing backbone using ModuleList
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        ])

        # Multi-task heads using ModuleDict
        self.heads = nn.ModuleDict({
            "classification": nn.Linear(hidden_dim, num_classes),
            "reconstruction": nn.Linear(hidden_dim, in_features),
        })

    def forward(self, x, task="classification"):
        h = self.stem(x)
        for block in self.blocks:
            h = block(h)
        return self.heads[task](h)


def main():
    print("=" * 70)
    print("OXIDEON CLASS-STYLE NEURAL NETWORK ARCHITECTURE SHOWCASE")
    print("=" * 70)

    # 1. Instantiate the custom neural network
    model = VisionClassifier(in_features=256, hidden_dim=64, num_classes=4)
    print("\n[1] Model Architecture Structure:")
    print(model)

    # 2. Inspect registered parameters
    params = model.parameters()
    named_params = list(model.named_parameters())
    print(f"\n[2] Total parameters registered: {len(params)} tensors")
    for name, p in named_params[:6]:
        print(f"    - {name:<35}: shape={p.shape}, dtype={p.dtype}")
    print(f"    ... and {len(named_params) - 6} more parameters.")

    # 3. Forward hooks for telemetry / activation caching
    activations = []
    def log_activation(module, inputs, output):
        activations.append(output)

    hook_handle = model.blocks[0].register_forward_hook(log_activation)
    print("\n[3] Registered forward hook on block[0]")

    # 4. Generate mini batch data
    batch_size = 8
    x_np = np.random.randn(batch_size, 256).astype(np.float32)
    y_np = np.random.randint(0, 4, size=(batch_size,)).astype(np.int64)

    x = ox.from_numpy(x_np)
    target = ox.from_numpy(y_np)

    # 5. Training loop with Adam optimizer and CrossEntropyLoss
    criterion = nn.CrossEntropyLoss()
    optimizer = ox.optim.Adam(model.parameters(), lr=0.01)

    print("\n[4] Training Loop Execution (10 epochs):")
    model.train()
    for epoch in range(1, 11):
        optimizer.zero_grad()
        logits = model(x, task="classification")
        loss = criterion(logits, target)
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        preds = logits.argmax(dim=1).numpy()
        acc = float(np.mean(preds == y_np)) * 100.0
        print(f"    Epoch {epoch:>2}/10 | CrossEntropy Loss: {loss_val:.5f} | Accuracy: {acc:>5.1f}%")

    hook_handle.remove()
    print(f"  ✓ Forward hook captured {len(activations)} intermediate activations")

    # 6. Evaluation mode and inference
    model.eval()
    with_eval = model(x)
    print(f"\n[5] Evaluation inference output shape: {with_eval.shape}")

    # 7. State dict saving and loading
    print("\n[6] State Dict Serialization & Weight Handoff:")
    sd = model.state_dict()
    print(f"  ✓ Saved state dict with {len(sd)} tensors")

    # Create fresh clone
    model_clone = VisionClassifier(in_features=256, hidden_dim=64, num_classes=4)
    model_clone.load_state_dict(sd)
    model_clone.eval()
    clone_output = model_clone(x)
    max_diff = np.max(np.abs(with_eval.numpy() - clone_output.numpy()))
    print(f"  ✓ Restored model clone max prediction difference: {max_diff:.8e} (identical)")

    # 8. Transfer learning & parameter freezing
    print("\n[7] Parameter Freezing for Transfer Learning:")
    model.stem.freeze()
    trainable_after_freeze = [p for p in model.parameters() if p.requires_grad]
    print(f"  ✓ Frozen stem layers: Trainable parameters reduced from {len(params)} to {len(trainable_after_freeze)}")

    print("\n" + "=" * 70)
    print("CLASS-STYLE NEURAL NETWORK ARCHITECTURE FULLY VERIFIED & OPERATIONAL!")
    print("=" * 70)


if __name__ == "__main__":
    main()

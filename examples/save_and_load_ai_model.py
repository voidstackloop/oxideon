#!/usr/bin/env python3
"""Example: Saving an AI Model to File with `save_file` and Running the Loaded Model.

This showcase demonstrates:
1. Creating and training an AI classification neural network (`ox.nn.Module`).
2. Saving weights and architecture metadata to disk using `ox.save_file()`.
3. Inspecting metadata header with `ox.load_file_metadata()`.
4. Initializing a brand-new, untrained model instance and loading the weights using `.load_file()`.
5. Running inference on sample inputs and verifying identical predictions.
6. Continuing training seamlessly from the loaded checkpoint.
"""

import os
import tempfile
import numpy as np
import oxideon as ox
import oxideon.nn as nn


class VisionFeatureNet(nn.Module):
    """Multi-layer perception neural network for image feature classification."""

    def __init__(self, in_features: int = 16, hidden_dim: int = 32, num_classes: int = 4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: ox.Tensor) -> ox.Tensor:
        # Layer 1: Linear + LayerNorm + ReLU
        h = self.fc1(x)
        h = self.norm(h).relu()
        # Layer 2: Residual connection
        h = h + self.fc2(h).relu()
        # Output Logits
        return self.classifier(h)


def main():
    print("=" * 70)
    print("   Oxideon AI Model Serialization & Loading Showcase (`save_file`)")
    print("=" * 70)

    # Set deterministic seed
    ox.manual_seed(42)

    # 1. Create model and generate synthetic data
    print("\n1. Instantiating AI model (VisionFeatureNet)...")
    in_dim, hidden, classes = 16, 32, 4
    model = VisionFeatureNet(in_dim, hidden, classes)
    print(f"   Model parameters registered: {len(model.parameters())}")
    for name, p in model.named_parameters():
        print(f"     - {name}: shape={p.shape}")

    # Generate synthetic training batch: 64 samples
    x_train = ox.randn([64, in_dim])
    y_train = ox.zeros([64])  # Class 0 target
    test_sample = ox.randn([4, in_dim])

    # 2. Train the model
    print("\n2. Training model for 25 steps with Adam optimizer...")
    criterion = nn.CrossEntropyLoss()
    optimizer = ox.optim.Adam(model.parameters(), lr=0.03)

    initial_loss = float(criterion(model(x_train), y_train).to_numpy().item())
    print(f"   Initial Training Loss: {initial_loss:.5f}")

    for step in range(25):
        model.zero_grad()
        logits = model(x_train)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()

    final_loss = float(criterion(model(x_train), y_train).to_numpy().item())
    print(f"   Final Training Loss:   {final_loss:.5f} (loss reduced by {initial_loss - final_loss:.5f})")

    # Evaluate original model on test inputs
    model.eval()
    orig_preds = model(test_sample)
    print(f"\n   Original Model Predictions on 4 test samples:\n{orig_preds.to_numpy()}")

    # 3. Save model to safetensors file
    with tempfile.TemporaryDirectory() as tmpdir:
        save_path = os.path.join(tmpdir, "vision_classifier.safetensors")
        print(f"\n3. Saving AI model to file: {os.path.basename(save_path)}...")

        meta = {
            "model_type": "VisionFeatureNet",
            "in_features": str(in_dim),
            "hidden_dim": str(hidden),
            "num_classes": str(classes),
            "trained_loss": f"{final_loss:.5f}",
            "author": "Oxideon Core Team",
        }
        # Save model and metadata using ox.save_file
        ox.save_file(model, save_path, metadata=meta)
        file_size_bytes = os.path.getsize(save_path)
        print(f"   Successfully written! File size: {file_size_bytes} bytes ({file_size_bytes / 1024:.2f} KB)")

        # 4. Inspect file metadata without loading tensors
        print("\n4. Reading header metadata via ox.load_file_metadata()...")
        loaded_meta = ox.load_file_metadata(save_path)
        for k, v in loaded_meta.items():
            print(f"   [{k}]: {v}")

        # 5. Initialize a fresh, untrained model and load the file
        print("\n5. Instantiating fresh, untrained model...")
        loaded_model = VisionFeatureNet(in_dim, hidden, classes)
        loaded_model.eval()

        untrained_preds = loaded_model(test_sample)
        print("   Untrained Model Output (random weights):\n", untrained_preds.to_numpy()[0])

        print(f"   Loading weights from '{os.path.basename(save_path)}'...")
        loaded_model.load_file(save_path)
        print("   Weights successfully loaded into model!")

        # 6. Run the model with loading file
        print("\n6. Running the loaded model on test inputs...")
        loaded_preds = loaded_model(test_sample)
        print(f"   Loaded Model Output:\n{loaded_preds.to_numpy()}")

        # Verify bit-for-bit exact match
        max_diff = np.max(np.abs(orig_preds.to_numpy() - loaded_preds.to_numpy()))
        print(f"\n   Maximum Absolute Difference (Original vs Loaded): {max_diff:.8e}")
        assert max_diff == 0.0, f"Error: predictions diverged with max diff {max_diff}!"
        print("   VERIFICATION PASSED: Predictions match 100% bit-for-bit!")

        # 7. Continue training on loaded model
        print("\n7. Continuing training from loaded checkpoint...")
        loaded_model.train()
        resumed_optimizer = ox.optim.Adam(loaded_model.parameters(), lr=0.03)

        for step in range(10):
            loaded_model.zero_grad()
            logits = loaded_model(x_train)
            loss = criterion(logits, y_train)
            loss.backward()
            resumed_optimizer.step()

        resumed_loss = float(criterion(loaded_model(x_train), y_train).to_numpy().item())
        print(f"   Loss after 10 additional steps: {resumed_loss:.5f} (converged at optimal minima)")
        assert resumed_loss <= final_loss, "Resumed training should maintain or improve loss!"

    print("\n" + "=" * 70)
    print("   AI Model Saved and Loaded Successfully!")
    print("=" * 70)


if __name__ == "__main__":
    main()

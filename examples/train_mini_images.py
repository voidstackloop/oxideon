#!/usr/bin/env python3
"""
Oxideon 1,000 Mini Images Convolutional Neural Network Training Showcase
========================================================================
Demonstrates procedural generation of 1,000 balanced mini-images (16x16)
across 4 distinct spatial waveform classes, building a 4-stage ConvNet,
training with AdamW, and evaluating test classification accuracy.
"""

import oxideon as ox

CLASS_NAMES = [
    "Horizontal Gratings (Class 0)",
    "Vertical Gratings   (Class 1)",
    "Concentric Rings    (Class 2)",
    "Diagonal Waveforms  (Class 3)",
]

def banner(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)

def main():
    banner("1. Procedural Generation of 1,000 Mini Images")
    # Generates 1000 mini images (1x16x16) balanced across 4 geometric pattern classes
    dataset = ox.MiniImageDataset.standard_1000()
    print(f"Dataset generated:       {dataset}")
    print(f"Total Images:            {len(dataset)}")
    print(f"Tensor Shape:            {dataset.images.shape}")
    print(f"Number of Classes:       {dataset.num_classes}")

    # Count class distribution
    labels = dataset.labels
    counts = [labels.count(c) for c in range(dataset.num_classes)]
    for c, name in enumerate(CLASS_NAMES):
        print(f"  {name}: {counts[c]} images ({counts[c] / len(dataset) * 100:.1f}%)")

    banner("2. Dataset Partitioning (80/20 Train / Validation Split)")
    train_set, val_set = dataset.split(train_ratio=0.8)
    print(f"Training Subsample:      {len(train_set)} images (shape: {train_set.images.shape})")
    print(f"Validation Subsample:    {len(val_set)} images (shape: {val_set.images.shape})")
    print(f"Training Mini-Batches:   {train_set.num_batches(batch_size=32)} (batch_size=32)")

    banner("3. Convolutional Neural Network (MiniImageClassifier)")
    model = ox.MiniImageClassifier.standard()
    print(f"Model Architecture:      {model}")
    print(f"Total Learnable Params:  {model.num_parameters():,}")
    print("Network Topology:")
    print("  Stage 1: Conv2d(1 -> 16, k=3, p=1) + ReLU + MaxPool2d(2x2) -> [B, 16, 8, 8]")
    print("  Stage 2: Conv2d(16 -> 32, k=3, p=1) + ReLU + MaxPool2d(2x2) -> [B, 32, 4, 4]")
    print("  Stage 3: Flatten(512) + Linear(512 -> 64) + ReLU")
    print("  Stage 4: Linear(64 -> 4) Logits Head")

    # Initial untrained evaluation
    init_loss, init_acc = model.evaluate(val_set.images, val_set.labels)
    print(f"\nBaseline Untrained Val Loss:     {init_loss:.4f}")
    print(f"Baseline Untrained Val Accuracy: {init_acc * 100.0:.2f}% (Chance level: 25.00%)")

    banner("4. Model Training (AdamW Optimization across Mini-Batches)")
    epochs = 4
    batch_size = 32
    learning_rate = 0.005

    print(f"Training on {len(train_set)} images for {epochs} epochs (batch_size={batch_size}, lr={learning_rate})...\n")
    print(f"{'Epoch':<8}{'Train Loss':<14}{'Train Acc (%)':<16}{'Val Loss':<14}{'Val Acc (%)':<14}")
    print("-" * 66)

    result = model.fit(
        train_set,
        val_dataset=val_set,
        epochs=epochs,
        batch_size=batch_size,
        lr=learning_rate,
    )

    for ep in result["history"]:
        epoch_idx = ep["epoch"]
        t_loss = ep["train_loss"]
        t_acc = ep["train_accuracy"] * 100.0
        v_loss = ep["val_loss"]
        v_acc = ep["val_accuracy"] * 100.0
        print(f"{epoch_idx:<8}{t_loss:<14.4f}{t_acc:<16.2f}{v_loss:<14.4f}{v_acc:<14.2f}")

    final_train_acc = result["final_train_accuracy"] * 100.0
    final_val_acc = result["final_val_accuracy"] * 100.0
    print("-" * 66)
    print(f"Final Training Accuracy:   {final_train_acc:.2f}%")
    print(f"Final Validation Accuracy: {final_val_acc:.2f}%")

    banner("5. Sample Visual Inference & Classification Verification")
    sample_batch, sample_labels = val_set.batch(batch_idx=0, batch_size=8)
    predictions = model.predict(sample_batch)

    print(f"{'Sample':<8}{'Ground Truth Class':<32}{'Predicted Class':<32}{'Result':<10}")
    print("-" * 82)
    correct_count = 0
    for idx, (gt, pred) in enumerate(zip(sample_labels, predictions)):
        match = gt == pred
        status = "MATCH (100%)" if match else "MISMATCH"
        if match:
            correct_count += 1
        gt_str = CLASS_NAMES[gt]
        pred_str = CLASS_NAMES[pred]
        print(f"#{idx + 1:<7}{gt_str:<32}{pred_str:<32}{status:<10}")

    print("-" * 82)
    print(f"Validation Sample Accuracy: {correct_count} / {len(sample_labels)} ({correct_count / len(sample_labels) * 100:.1f}%)")

    banner("SUCCESS: Oxideon Basic Image Model & 1,000 Mini Images Complete!")

if __name__ == "__main__":
    main()

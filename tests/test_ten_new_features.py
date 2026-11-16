"""Tests for 10 New Advanced Deep Learning Features in Oxideon:
1. ox.transforms (Computer Vision transforms & augmentations)
2. ox.Trainer (High-level model training & evaluation harness)
3. ox.callbacks (ModelCheckpoint, CSVLogger, LearningRateLogger, TerminateOnNaN)
4. Recurrent Layers (LSTMCell, GRUCell, LSTM, GRU)
5. Normalization Layers (RMSNorm, BatchNorm1d, BatchNorm2d, GroupNorm)
6. Vision Pooling & Resampling (MaxPool2d, AvgPool2d, AdaptiveAvgPool2d, Upsample)
7. NLP Tokenizers & Sequences (WordTokenizer, CharTokenizer, pad_sequence)
8. Synthetic Benchmark Datasets (make_moons, make_circles, make_blobs, make_regression)
9. Experiment Tracking (SummaryWriter)
10. Model Quantization Engine (quantize_dynamic, QuantizedLinear)
"""

import os
import shutil
import tempfile
import numpy as np
import pytest

import oxideon as ox
from oxideon import nn, optim, data, transforms, text, callbacks, utils, quantization


# ---------------------------------------------------------------------------
# Feature 1: ox.transforms
# ---------------------------------------------------------------------------

def test_feature1_transforms_pipeline():
    raw_img = np.random.randint(0, 256, (32, 32, 3), dtype=np.uint8)

    pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.CenterCrop(24),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.Resize((16, 16)),
    ])

    out = pipeline(raw_img)
    assert isinstance(out, ox.Tensor)
    assert out.shape == [3, 16, 16]

    # Test RandomCrop
    rc = transforms.RandomCrop(size=(12, 12), padding=2)
    cropped = rc(out)
    assert cropped.shape == [3, 12, 12]


# ---------------------------------------------------------------------------
# Feature 2: ox.Trainer
# ---------------------------------------------------------------------------

def test_feature2_trainer_harness():
    X, y = data.make_regression(n_samples=64, n_features=4, noise=0.01, random_state=42)
    dataset = data.TensorDataset(X, y)
    train_loader = data.DataLoader(dataset, batch_size=16, shuffle=True)
    val_loader = data.DataLoader(dataset, batch_size=16, shuffle=False)

    model = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 1),
    )
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    trainer = ox.Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        max_grad_norm=1.0,
    )

    history = trainer.fit(train_loader, val_loader=val_loader, epochs=3, verbose=False)
    assert "train_loss" in history
    assert len(history["train_loss"]) == 3
    assert "val_loss" in history
    assert len(history["val_loss"]) == 3

    eval_res = trainer.evaluate(val_loader)
    assert "loss" in eval_res
    assert eval_res["loss"] >= 0.0

    preds = trainer.predict(val_loader)
    assert preds.shape[0] == 64


# ---------------------------------------------------------------------------
# Feature 3: Callbacks Suite
# ---------------------------------------------------------------------------

def test_feature3_callbacks():
    tmp_dir = tempfile.mkdtemp()
    try:
        csv_path = os.path.join(tmp_dir, "training.csv")
        ckpt_path = os.path.join(tmp_dir, "best_model.safetensors")

        csv_cb = callbacks.CSVLogger(csv_path)
        ckpt_cb = callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", mode="min")
        lr_cb = callbacks.LearningRateLogger()
        nan_cb = callbacks.TerminateOnNaN()

        model = nn.Sequential(nn.Linear(2, 1))
        optimizer = optim.SGD(model.parameters(), lr=0.05)
        criterion = nn.MSELoss()

        X = ox.randn([20, 2])
        y = ox.randn([20, 1])
        loader = data.DataLoader(data.TensorDataset(X, y), batch_size=10)

        trainer = ox.Trainer(
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            callbacks=[csv_cb, ckpt_cb, lr_cb, nan_cb],
        )
        trainer.fit(loader, val_loader=loader, epochs=2, verbose=False)

        assert os.path.exists(csv_path)
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            assert len(lines) >= 3  # Header + 2 epochs

        assert os.path.exists(ckpt_path)
        loaded = ox.load_file(ckpt_path)
        assert any("weight" in k for k in loaded)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Feature 4: Recurrent Neural Network Layers
# ---------------------------------------------------------------------------

def test_feature4_recurrent_layers():
    # LSTMCell & GRUCell
    lstm_cell = nn.LSTMCell(input_size=4, hidden_size=8)
    gru_cell = nn.GRUCell(input_size=4, hidden_size=8)

    x_step = ox.randn([2, 4])
    h_lstm, c_lstm = lstm_cell(x_step)
    assert h_lstm.shape == [2, 8]
    assert c_lstm.shape == [2, 8]

    h_gru = gru_cell(x_step)
    assert h_gru.shape == [2, 8]

    # Full Sequence LSTM & GRU
    seq_x = ox.randn([2, 5, 4])  # [batch=2, seq_len=5, features=4]
    lstm = nn.LSTM(input_size=4, hidden_size=8, num_layers=2, batch_first=True)
    gru = nn.GRU(input_size=4, hidden_size=8, num_layers=2, batch_first=True)

    out_lstm, (hn_l, cn_l) = lstm(seq_x)
    assert out_lstm.shape == [2, 5, 8]
    assert len(hn_l) == 2

    out_gru, hn_g = gru(seq_x)
    assert out_gru.shape == [2, 5, 8]
    assert len(hn_g) == 2


# ---------------------------------------------------------------------------
# Feature 5: Normalization Layers Suite
# ---------------------------------------------------------------------------

def test_feature5_normalization_layers():
    # RMSNorm
    rms = nn.RMSNorm(dim=16)
    x = ox.randn([4, 16])
    normed = rms(x)
    assert normed.shape == [4, 16]

    # BatchNorm1d
    bn1d = nn.BatchNorm1d(num_features=8)
    x1d = ox.randn([10, 8])
    bn1d.train()
    out1d = bn1d(x1d)
    assert out1d.shape == [10, 8]

    bn1d.eval()
    out1d_eval = bn1d(x1d)
    assert out1d_eval.shape == [10, 8]

    # BatchNorm2d
    bn2d = nn.BatchNorm2d(num_features=4)
    x2d = ox.randn([2, 4, 8, 8])
    bn2d.train()
    out2d = bn2d(x2d)
    assert out2d.shape == [2, 4, 8, 8]

    # GroupNorm
    gn = nn.GroupNorm(num_groups=2, num_channels=4)
    out_gn = gn(x2d)
    assert out_gn.shape == [2, 4, 8, 8]


# ---------------------------------------------------------------------------
# Feature 6: Vision Pooling & Resampling
# ---------------------------------------------------------------------------

def test_feature6_vision_pooling_and_resampling():
    img = ox.randn([2, 3, 16, 16])

    # MaxPool2d
    mp = nn.MaxPool2d(kernel_size=2, stride=2)
    out_mp = mp(img)
    assert out_mp.shape == [2, 3, 8, 8]

    # AvgPool2d
    ap = nn.AvgPool2d(kernel_size=2, stride=2)
    out_ap = ap(img)
    assert out_ap.shape == [2, 3, 8, 8]

    # AdaptiveAvgPool2d
    adap = nn.AdaptiveAvgPool2d(output_size=(1, 1))
    out_adap = adap(img)
    assert out_adap.shape == [2, 3, 1, 1]

    # Upsample
    up = nn.Upsample(scale_factor=2)
    out_up = up(out_mp)
    assert out_up.shape == [2, 3, 16, 16]


# ---------------------------------------------------------------------------
# Feature 7: NLP Tokenizers & Sequences
# ---------------------------------------------------------------------------

def test_feature7_nlp_tokenizers_and_sequences():
    corpus = [
        "Oxideon is an advanced deep learning framework.",
        "Neural networks learn from data and gradients.",
        "Deep learning with high performance and zero overhead.",
    ]

    tokenizer = text.WordTokenizer()
    tokenizer.build_vocab(corpus, min_freq=1)
    assert tokenizer.vocab_size > 10

    encoded = tokenizer.encode("Oxideon neural networks")
    assert len(encoded) >= 4  # Includes BOS and EOS
    decoded = tokenizer.decode(encoded)
    assert "oxideon" in decoded

    # CharTokenizer
    char_tok = text.CharTokenizer()
    char_tok.build_vocab(corpus)
    c_encoded = char_tok.encode("hello")
    assert len(c_encoded) == 5
    assert char_tok.decode(c_encoded) == "hello"

    # pad_sequence
    seq1 = [1, 2, 3]
    seq2 = [4, 5]
    padded = text.pad_sequence([seq1, seq2], batch_first=True, padding_value=0)
    assert padded.shape == [2, 3]
    assert padded[1, 2].item() == 0.0


# ---------------------------------------------------------------------------
# Feature 8: Synthetic Benchmark Datasets
# ---------------------------------------------------------------------------

def test_feature8_synthetic_datasets():
    # make_moons
    X_m, y_m = data.make_moons(n_samples=50, noise=0.05, random_state=42)
    assert X_m.shape == [50, 2]
    assert y_m.shape == [50]

    # make_circles
    X_c, y_c = data.make_circles(n_samples=50, factor=0.5, noise=0.05, random_state=42)
    assert X_c.shape == [50, 2]
    assert y_c.shape == [50]

    # make_blobs
    X_b, y_b = data.make_blobs(n_samples=60, n_features=3, centers=3, random_state=42)
    assert X_b.shape == [60, 3]
    assert y_b.shape == [60]

    # make_regression
    X_r, y_r = data.make_regression(n_samples=40, n_features=5, noise=0.1, random_state=42)
    assert X_r.shape == [40, 5]
    assert y_r.shape == [40]


# ---------------------------------------------------------------------------
# Feature 9: Experiment Tracking (SummaryWriter)
# ---------------------------------------------------------------------------

def test_feature9_summary_writer():
    tmp_dir = tempfile.mkdtemp()
    try:
        with utils.SummaryWriter(log_dir=tmp_dir) as writer:
            for step in range(5):
                writer.add_scalar("loss/train", 1.0 / (step + 1), global_step=step)
                writer.add_scalars("metrics", {"acc": 0.5 + 0.1 * step, "f1": 0.4 + 0.1 * step}, global_step=step)
            writer.add_text("notes", "Training started successfully", global_step=0)

            history = writer.get_history("loss/train")
            assert len(history) == 5
            assert np.isclose(history[0][1], 1.0)

            json_path = writer.export_json()
            assert os.path.exists(json_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Feature 10: Model Quantization Engine
# ---------------------------------------------------------------------------

def test_feature10_model_quantization():
    class TwoLayerMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(8, 16)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(16, 4)

        def forward(self, x):
            return self.fc2(self.relu(self.fc1(x)))

    float_model = TwoLayerMLP()
    x = ox.randn([3, 8])
    float_out = float_model(x).to_numpy()

    # Dynamic INT8 Quantization
    q_model = quantization.quantize_dynamic(float_model, dtype="qint8")
    assert isinstance(q_model.fc1, quantization.QuantizedLinear)
    assert isinstance(q_model.fc2, quantization.QuantizedLinear)
    assert q_model.fc1.qweight.dtype == np.int8

    q_out = q_model(x).to_numpy()
    assert q_out.shape == (3, 4)
    # Output should closely match float output
    assert np.allclose(float_out, q_out, atol=0.2)

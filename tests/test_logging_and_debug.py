"""
Unit tests for Oxideon Logging, Training Process Tracking, and Debug Mode:
- Global debug state (set_debug, is_debug, enable_debug, disable_debug, debug_mode)
- Logger output formatting and log levels
- Real-time training step and epoch summary progress tracking
- Tensor statistical audit and NaN/Inf detection
- Trainer debug mode with gradient norm and shape tracking
"""
import io
import numpy as np
import pytest
import oxideon as ox
from oxideon.logging import LogLevel, OxideonLogger

def test_global_debug_toggles():
    ox.disable_debug()
    assert ox.is_debug() is False

    ox.enable_debug()
    assert ox.is_debug() is True

    ox.set_debug(False)
    assert ox.is_debug() is False

    with ox.debug_mode(True):
        assert ox.is_debug() is True
    assert ox.is_debug() is False

    with ox.debug_mode(False):
        assert ox.is_debug() is False

def test_logger_stream_and_levels():
    stream = io.StringIO()
    logger = OxideonLogger("test_logger", stream=stream)

    # Info message
    logger.info("Normal execution running", epoch=1, loss=0.45)
    content = stream.getvalue()
    assert "Normal execution running" in content
    assert "epoch=1" in content

    # Debug message when debug is disabled
    ox.disable_debug()
    logger.debug("Hidden debug detail")
    # Shouldn't be written when global level is INFO
    assert "Hidden debug detail" not in stream.getvalue()

    # Enable debug and check
    ox.enable_debug()
    logger.debug("Visible debug detail")
    assert "Visible debug detail" in stream.getvalue()
    ox.disable_debug()

def test_tensor_inspection_and_nan_check():
    x = ox.from_numpy(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))
    diag = ox.debug_tensor(x, name="layer_weights")
    assert diag["name"] == "layer_weights"
    assert diag["shape"] == [4]
    assert diag["has_nan"] is False
    assert diag["has_inf"] is False
    assert np.isclose(diag["min"], 1.0)
    assert np.isclose(diag["max"], 4.0)
    assert np.isclose(diag["mean"], 2.5)

    # Test NaN detection
    x_nan = ox.from_numpy(np.array([1.0, np.nan, 3.0], dtype=np.float32))
    assert ox.check_nan(x_nan, name="bad_tensor") is True

    # Under debug mode, check_nan raises FloatingPointError
    with ox.debug_mode(True):
        with pytest.raises(FloatingPointError):
            ox.check_nan(x_nan, name="bad_tensor")

def test_training_step_and_epoch_summary_formatting():
    stream = io.StringIO()
    logger = OxideonLogger("progress_tracker", stream=stream)

    # Log training step
    logger.log_training_step(
        epoch=1,
        step=5,
        total_steps=10,
        loss=0.3421,
        metrics={"acc": 0.94},
        samples_per_sec=128.5,
        lr=0.001,
    )
    output = stream.getvalue()
    assert "Epoch 01" in output
    assert "loss=0.3421" in output
    assert "128.5 samples/s" in output
    assert "5/10" in output

    # Log epoch summary
    logger.log_epoch_summary(
        epoch=1,
        total_epochs=10,
        train_loss=0.25,
        train_acc=0.96,
        val_loss=0.30,
        val_acc=0.94,
        elapsed_sec=1.45,
        samples_per_sec=250.0,
    )
    summary_output = stream.getvalue()
    assert "EPOCH 01/10 SUMMARY" in summary_output
    assert "Train Loss : 0.25000" in summary_output
    assert "Val Loss   : 0.30000" in summary_output

def test_trainer_with_debug_mode():
    # Synthetic dataset
    X = ox.randn([20, 4])
    y = ox.from_numpy(np.random.randint(0, 2, size=(20,)).astype(np.int64))
    dataset = ox.data.TensorDataset(X, y)
    loader = ox.data.DataLoader(dataset, batch_size=5, shuffle=False)

    model = ox.nn.Sequential(
        ox.nn.Linear(4, 8),
        ox.nn.ReLU(),
        ox.nn.Linear(8, 2),
    )
    optimizer = ox.optim.Adam(model.parameters(), lr=0.01)
    criterion = ox.nn.CrossEntropyLoss()

    # Trainer initialized with debug=True
    trainer = ox.Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        max_grad_norm=1.0,
        debug=True,
    )

    history = trainer.fit(loader, epochs=2, verbose=True, debug=True)
    assert len(history["train_loss"]) == 2
    assert history["train_loss"][-1] < 10.0

def test_trainer_with_debug_false_concise():
    X = ox.randn([10, 2])
    y = ox.from_numpy(np.zeros(10, dtype=np.int64))
    loader = ox.data.DataLoader(ox.data.TensorDataset(X, y), batch_size=5)

    model = ox.nn.Linear(2, 2)
    optimizer = ox.optim.SGD(model.parameters(), lr=0.01)
    criterion = ox.nn.CrossEntropyLoss()

    trainer = ox.Trainer(model=model, optimizer=optimizer, criterion=criterion, debug=False)
    history = trainer.fit(loader, epochs=1, verbose=False, debug=False)
    assert len(history["train_loss"]) == 1

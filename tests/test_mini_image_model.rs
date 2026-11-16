use oxideon::error::Result;
use oxideon::nn::Module;
use oxideon::tensor::Tensor;
use oxideon::vision::{MiniAutoEncoder, MiniImageClassifier, MiniImageDataset, MiniResNet};

#[test]
fn test_mini_image_dataset_generation() -> Result<()> {
    let dataset = MiniImageDataset::standard_1000()?;

    // Verify dataset size and shapes
    assert_eq!(dataset.len(), 1000);
    assert!(!dataset.is_empty());
    assert_eq!(dataset.images.shape(), &[1000, 1, 16, 16]);
    assert_eq!(dataset.labels.len(), 1000);
    assert_eq!(dataset.num_classes, 4);

    // Verify class balance
    let mut class_counts = [0usize; 4];
    for &label in &dataset.labels {
        assert!(label < 4, "Class label must be in 0..4");
        class_counts[label] += 1;
    }
    for &count in &class_counts {
        assert_eq!(
            count, 250,
            "Every class should have exactly 250 images in standard 1000 dataset"
        );
    }

    // Verify split
    let (train_set, val_set) = dataset.split(0.8)?;
    assert_eq!(train_set.len(), 800);
    assert_eq!(val_set.len(), 200);
    assert_eq!(train_set.images.shape(), &[800, 1, 16, 16]);
    assert_eq!(val_set.images.shape(), &[200, 1, 16, 16]);

    // Verify batching
    assert_eq!(train_set.num_batches(32), 25);
    let (batch_imgs, batch_lbls) = train_set.batch(0, 32)?;
    assert_eq!(batch_imgs.shape(), &[32, 1, 16, 16]);
    assert_eq!(batch_lbls.len(), 32);

    Ok(())
}

#[test]
fn test_mini_image_classifier_architecture_and_forward() -> Result<()> {
    let model = MiniImageClassifier::standard()?;

    // Check parameter count
    let params = model.parameters();
    let total_params: usize = params
        .iter()
        .map(|p| p.shape().iter().product::<usize>())
        .sum();
    assert!(total_params > 30_000 && total_params < 45_000);

    // Forward pass with batch size 4
    let input = Tensor::randn(&[4, 1, 16, 16])?;
    let logits = model.forward(&input)?;

    assert_eq!(logits.shape(), &[4, 4]);

    // Check predict
    let preds = model.predict(&input)?;
    assert_eq!(preds.len(), 4);
    for p in preds {
        assert!(p < 4);
    }

    // Check evaluate
    let targets = vec![0, 1, 2, 3];
    let (loss, acc) = model.evaluate(&input, &targets)?;
    assert!(loss > 0.0);
    assert!((0.0..=1.0).contains(&acc));

    Ok(())
}

#[test]
fn test_mini_image_classifier_training_convergence() -> Result<()> {
    let dataset = MiniImageDataset::standard_1000()?;
    let (train_set, val_set) = dataset.split(0.8)?;

    let mut model = MiniImageClassifier::standard()?;

    // Evaluate initial untrained loss
    let (init_loss, init_acc) = model.evaluate(&val_set.images, &val_set.labels)?;
    println!(
        "Initial val loss: {:.4}, acc: {:.2}%",
        init_loss,
        init_acc * 100.0
    );

    // Train for 4 epochs
    let history = model.fit(&train_set, Some(&val_set), 4, 32, 0.005)?;
    assert_eq!(history.history.len(), 4);

    let first_epoch = &history.history[0];
    let last_epoch = &history.history[3];

    println!(
        "Epoch 1 train loss: {:.4}, acc: {:.2}% | Epoch 4 train loss: {:.4}, acc: {:.2}%",
        first_epoch.train_loss,
        first_epoch.train_accuracy * 100.0,
        last_epoch.train_loss,
        last_epoch.train_accuracy * 100.0
    );

    // Assert that training loss decreased
    assert!(
        last_epoch.train_loss < first_epoch.train_loss,
        "Training loss must decrease: {} vs {}",
        last_epoch.train_loss,
        first_epoch.train_loss
    );

    // Assert that accuracy reached strong pattern recognition level (>60%)
    assert!(
        last_epoch.train_accuracy > 0.60,
        "Final train accuracy should exceed 60%, got: {}",
        last_epoch.train_accuracy
    );

    let final_val_acc = history.final_val_accuracy.unwrap();
    assert!(
        final_val_acc > 0.50,
        "Validation accuracy should exceed 50%, got: {}",
        final_val_acc
    );

    Ok(())
}

#[test]
fn test_mini_resnet_training_convergence() -> Result<()> {
    let dataset = MiniImageDataset::standard_1000()?;
    let (train_set, val_set) = dataset.split(0.8)?;

    let mut resnet = MiniResNet::standard()?;

    // Verify parameter count
    let params = resnet.parameters();
    let total_params: usize = params
        .iter()
        .map(|p| p.shape().iter().product::<usize>())
        .sum();
    assert!(total_params > 25_000 && total_params < 45_000);

    // Initial evaluation
    let (init_loss, init_acc) = resnet.evaluate(&val_set.images, &val_set.labels)?;
    assert!(init_loss > 0.0);
    assert!((0.0..=1.0).contains(&init_acc));

    // Train for 3 epochs
    let history = resnet.fit(&train_set, Some(&val_set), 3, 32, 0.005)?;
    assert_eq!(history.len(), 3);

    let first_loss = history[0].train_loss;
    let last_loss = history[2].train_loss;
    let final_acc = history[2].train_accuracy;

    assert!(last_loss < first_loss, "ResNet loss must decrease");
    assert!(final_acc > 0.60, "ResNet train accuracy must exceed 60%");

    Ok(())
}

#[test]
fn test_mini_autoencoder_reconstruction_and_interpolation() -> Result<()> {
    let dataset = MiniImageDataset::standard_1000()?;
    let mut ae = MiniAutoEncoder::standard()?;

    // Verify parameter count
    let params = ae.parameters();
    let total_params: usize = params
        .iter()
        .map(|p| p.shape().iter().product::<usize>())
        .sum();
    assert!(total_params > 25_000 && total_params < 50_000);

    // Test forward reconstruction
    let (sample_batch, _) = dataset.batch(0, 4)?;
    let reconstructed = ae.forward(&sample_batch)?;
    assert_eq!(reconstructed.shape(), &[4, 1, 16, 16]);

    // Test latent interpolation between 2 images
    let (img1, _) = dataset.batch(0, 1)?;
    let (img2, _) = dataset.batch(1, 1)?;
    let morphed_frames = ae.interpolate(&img1, &img2, 6)?;
    assert_eq!(morphed_frames.shape(), &[6, 1, 16, 16]);

    // Test generative random sampling
    let synthetic_imgs = ae.sample(4)?;
    assert_eq!(synthetic_imgs.shape(), &[4, 1, 16, 16]);

    // Train for 2 epochs on the first 100 images
    let (small_train, _) = dataset.split(0.2)?;
    let history = ae.fit(&small_train, None, 2, 32, 0.005)?;
    assert_eq!(history.len(), 2);
    assert!(
        history[1].train_loss < history[0].train_loss,
        "Reconstruction loss must decrease"
    );

    Ok(())
}

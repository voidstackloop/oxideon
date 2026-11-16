//! Vision Subsystem: Datasets, Image Models, and Convolutional Classifiers.

pub mod autoencoder;
pub mod classifier;
pub mod dataset;
pub mod resnet;
pub mod rope2d;

pub use autoencoder::MiniAutoEncoder;
pub use classifier::{EpochMetric, MiniImageClassifier, TrainingHistory};
pub use dataset::MiniImageDataset;
pub use resnet::{MiniResNet, ResidualBlock};
pub use rope2d::RotaryEmbedding2D;

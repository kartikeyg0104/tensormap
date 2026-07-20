"""Test fixtures for creating trained models for interpretability tests.

Provides utilities to create simple trained models with test datasets
for use in Week 9-10 interpretability and hyperparameter tuning tests.
"""

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
from sqlmodel import Session

from app.models.ml import ModelBasic
from app.models.training_job import TrainingJob, TrainingStatus
from app.models.training_metric import TrainingMetric


def create_test_dataset(num_samples: int = 150, num_features: int = 4, num_classes: int = 3):
    """Create a simple synthetic classification dataset (Iris-like).

    Args:
        num_samples: Number of samples to generate
        num_features: Number of input features
        num_classes: Number of output classes

    Returns:
        Tuple of (X_train, y_train, X_test, y_test, feature_names)
    """
    np.random.seed(42)

    # Generate synthetic data
    X = np.random.randn(num_samples, num_features)

    # Create class labels with some structure
    # Each class gets roughly equal samples
    y = np.zeros(num_samples, dtype=int)
    samples_per_class = num_samples // num_classes
    for i in range(num_classes):
        start_idx = i * samples_per_class
        end_idx = start_idx + samples_per_class
        y[start_idx:end_idx] = i
        # Add some class-specific bias to features
        X[start_idx:end_idx] += i * 0.5

    # Shuffle
    indices = np.random.permutation(num_samples)
    X = X[indices]
    y = y[indices]

    # Split train/test (80/20)
    split_idx = int(0.8 * num_samples)
    X_train = X[:split_idx]
    y_train = y[:split_idx]
    X_test = X[split_idx:]
    y_test = y[split_idx:]

    # Feature names
    feature_names = [f"feature_{i}" for i in range(num_features)]

    return X_train, y_train, X_test, y_test, feature_names


def create_simple_model(num_features: int = 4, num_classes: int = 3):
    """Create a simple Dense classification model.

    Args:
        num_features: Number of input features
        num_classes: Number of output classes

    Returns:
        Compiled Keras model
    """
    import tensorflow as tf

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=(num_features,)),
            tf.keras.layers.Dense(16, activation="relu"),
            tf.keras.layers.Dense(8, activation="relu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_and_save_test_model(
    export_dir: Path,
    num_features: int = 4,
    num_classes: int = 3,
    epochs: int = 10,
):
    """Train a simple model and save it to export_dir.

    Args:
        export_dir: Directory to save model.keras
        num_features: Number of input features
        num_classes: Number of output classes
        epochs: Number of training epochs

    Returns:
        Tuple of (model, X_test, y_test, feature_names, history)
    """
    # Create dataset
    X_train, y_train, X_test, y_test, feature_names = create_test_dataset(
        num_features=num_features,
        num_classes=num_classes,
    )

    # Create and train model
    model = create_simple_model(num_features, num_classes)

    history = model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        verbose=0,
    )

    # Save model
    export_dir.mkdir(parents=True, exist_ok=True)
    model_path = export_dir / "model.keras"
    model.save(str(model_path))

    return model, X_test, y_test, feature_names, history


def create_test_training_job(
    session: Session,
    export_dir: Path,
    model_name: str = "test_model",
    num_features: int = 4,
    num_classes: int = 3,
    epochs: int = 10,
):
    """Create a complete test training job with trained model and metrics.

    This is the main fixture function to use in tests. It creates:
    - A ModelBasic record
    - A TrainingJob record with COMPLETED status
    - TrainingMetric records for each epoch
    - A trained model.keras file
    - Test dataset

    Args:
        session: Database session
        export_dir: Base export directory (will create job subdirectory)
        model_name: Name for the model
        num_features: Number of input features
        num_classes: Number of output classes
        epochs: Number of training epochs

    Returns:
        Tuple of (job_id, X_test, y_test, feature_names, model)
    """
    # Create model record
    graph_ir = {
        "nodes": [
            {"node_params": {"layer_type": "input", "shape": [num_features]}},
            {"node_params": {"layer_type": "dense", "units": 16, "activation": "relu"}},
            {"node_params": {"layer_type": "dense", "units": 8, "activation": "relu"}},
            {"node_params": {"layer_type": "dense", "units": num_classes, "activation": "softmax"}},
        ]
    }

    model = ModelBasic(
        model_name=model_name,
        graph_ir=graph_ir,
    )
    session.add(model)
    session.commit()
    session.refresh(model)

    # Create training job
    job_id = str(uuid4())
    job = TrainingJob(
        id=job_id,
        model_id=model.id,
        status=TrainingStatus.COMPLETED,
        hyperparams={
            "optimizer": "adam",
            "lr": 0.001,
            "epochs": epochs,
            "batch_size": 32,
        },
        started_at=datetime.now(UTC),
        completed_at=datetime.now(UTC),
    )
    session.add(job)
    session.commit()

    # Train and save model
    job_export_dir = export_dir / job_id
    trained_model, X_test, y_test, feature_names, history = train_and_save_test_model(
        job_export_dir,
        num_features=num_features,
        num_classes=num_classes,
        epochs=epochs,
    )

    # Create training metrics from history
    for epoch in range(epochs):
        metrics = [
            TrainingMetric(
                job_id=job_id,
                epoch=epoch + 1,
                metric_name="loss",
                metric_value=float(history.history["loss"][epoch]),
            ),
            TrainingMetric(
                job_id=job_id,
                epoch=epoch + 1,
                metric_name="accuracy",
                metric_value=float(history.history["accuracy"][epoch]),
            ),
            TrainingMetric(
                job_id=job_id,
                epoch=epoch + 1,
                metric_name="val_loss",
                metric_value=float(history.history["val_loss"][epoch]),
            ),
            TrainingMetric(
                job_id=job_id,
                epoch=epoch + 1,
                metric_name="val_accuracy",
                metric_value=float(history.history["val_accuracy"][epoch]),
            ),
        ]
        for metric in metrics:
            session.add(metric)

    session.commit()

    return job_id, X_test, y_test, feature_names, trained_model

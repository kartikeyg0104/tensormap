"""Tests for model export service (SavedModel, TFLite, ONNX)."""

import importlib.util
from datetime import UTC, datetime, timedelta
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from sqlmodel import Session

from app.models.ml import ModelBasic
from app.models.training_job import TrainingJob, TrainingStatus
from app.services.model_export import (
    ONNXUnsupportedError,
    cleanup_exports,
    delete_job_exports,
    delete_model_exports,
    export_onnx,
    export_savedmodel,
    export_tflite,
    get_export_formats,
    validate_onnx_compatible,
)


def _check_tf2onnx_available():
    """Check if tf2onnx can be imported without errors."""
    try:
        import tf2onnx  # noqa: F401

        return True
    except (ImportError, AttributeError):
        return False


@pytest.fixture
def mock_keras_model():
    """Create a mock Keras model."""
    mock_model = Mock()
    mock_model.input_shape = (None, 28, 28, 1)
    mock_model.save = Mock()
    return mock_model


@pytest.fixture
def setup_export_dir(tmp_path):
    """Create a temporary export directory with model.keras."""
    job_id = str(uuid4())
    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)

    # Create a dummy model.keras file
    model_file = export_dir / "model.keras"
    model_file.write_text("dummy model content")

    return job_id, export_dir


def test_savedmodel_export_creates_zip(tmp_path, mock_keras_model):
    """SavedModel export creates a zip file."""
    job_id = str(uuid4())
    model_name = "test_model"

    # Setup
    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)
    (export_dir / "model.keras").write_text("dummy")

    # Patch EXPORTS_BASE and tensorflow
    with (
        patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"),
        patch("tensorflow.keras.models.load_model", return_value=mock_keras_model),
        patch("tensorflow.keras.models.save_model"),
    ):
        # Create savedmodel directory for zipping
        savedmodel_dir = export_dir / "savedmodel"
        savedmodel_dir.mkdir()
        (savedmodel_dir / "saved_model.pb").write_text("dummy pb")

        zip_path = export_savedmodel(job_id, model_name)

        assert zip_path.exists()
        assert zip_path.suffix == ".zip"
        assert model_name in zip_path.name


def test_tflite_export_creates_file(tmp_path, mock_keras_model):
    """TFLite export creates a .tflite file."""
    job_id = str(uuid4())
    model_name = "test_model"

    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)
    (export_dir / "model.keras").write_text("dummy")

    # Mock TFLite converter
    mock_converter = Mock()
    mock_converter.convert.return_value = b"tflite model bytes"

    # Import tensorflow to make the patch work
    import tensorflow as tf

    with (
        patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"),
        patch.object(tf.keras.models, "load_model", return_value=mock_keras_model),
        patch.object(tf.lite.TFLiteConverter, "from_keras_model", return_value=mock_converter),
    ):
        tflite_path = export_tflite(job_id, model_name)

        assert tflite_path.exists()
        assert tflite_path.suffix == ".tflite"
        assert tflite_path.read_bytes() == b"tflite model bytes"


def test_savedmodel_cached_on_second_call(tmp_path, mock_keras_model):
    """Second call to export_savedmodel returns existing file without regenerating."""
    job_id = str(uuid4())
    model_name = "test_model"

    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)
    (export_dir / "model.keras").write_text("dummy")

    # Create existing zip
    zip_path = export_dir / f"{model_name}.savedmodel.zip"
    zip_path.write_text("existing zip")

    with patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"):
        result = export_savedmodel(job_id, model_name)

        assert result == zip_path
        assert result.read_text() == "existing zip"  # Not regenerated


def test_export_fails_404_if_no_keras_file(tmp_path):
    """Export fails if model.keras doesn't exist (training not completed)."""
    job_id = str(uuid4())
    model_name = "test_model"

    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)
    # No model.keras file

    with (
        patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"),
        pytest.raises(FileNotFoundError, match="model.keras not found"),
    ):
        export_savedmodel(job_id, model_name)


def test_validate_onnx_compatible_passes_for_simple_model():
    """validate_onnx_compatible returns empty list for simple Dense model."""
    graph_ir = {
        "nodes": [
            {"node_params": {"layer_type": "input", "shape": [28, 28, 1]}},
            {"node_params": {"layer_type": "dense", "units": 10}},
        ]
    }

    mock_model = Mock()
    issues = validate_onnx_compatible(mock_model, graph_ir)

    assert issues == []


def test_validate_onnx_compatible_fails_for_rnn_var_length():
    """validate_onnx_compatible detects LSTM with variable-length input."""
    graph_ir = {
        "nodes": [
            {"node_params": {"layer_type": "input", "shape": [None, 128]}},  # Variable length
            {"node_params": {"layer_type": "lstm", "units": 64, "return_sequences": True}},
        ]
    }

    mock_model = Mock()
    issues = validate_onnx_compatible(mock_model, graph_ir)

    assert len(issues) > 0
    assert "LSTM/GRU" in issues[0]
    assert "fixed input sequence length" in issues[0]


@pytest.mark.skipif(
    not importlib.util.find_spec("tf2onnx") or not _check_tf2onnx_available(),
    reason="tf2onnx not available or dependencies incompatible",
)
def test_onnx_export_simple_dense(tmp_path, mock_keras_model):
    """ONNX export succeeds for simple Dense model."""
    job_id = str(uuid4())
    model_name = "test_model"

    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)
    (export_dir / "model.keras").write_text("dummy")

    graph_ir = {
        "nodes": [
            {"node_params": {"layer_type": "input", "shape": [28, 28]}},
            {"node_params": {"layer_type": "dense", "units": 10}},
        ]
    }

    # Mock ONNX model
    mock_onnx_model = Mock()
    mock_onnx_model.SerializeToString.return_value = b"onnx model bytes"

    with (
        patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"),
        patch("tensorflow.keras.models.load_model", return_value=mock_keras_model),
        patch("tensorflow.TensorSpec", return_value="tensor_spec"),
        patch("tf2onnx.convert.from_keras", return_value=(mock_onnx_model, None)),
    ):
        onnx_path = export_onnx(job_id, model_name, graph_ir)

        assert onnx_path.exists()
        assert onnx_path.suffix == ".onnx"


def test_onnx_export_raises_on_unsupported(tmp_path, mock_keras_model):
    """ONNX export raises ONNXUnsupportedError for incompatible architecture."""
    job_id = str(uuid4())
    model_name = "test_model"

    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)
    (export_dir / "model.keras").write_text("dummy")

    graph_ir = {
        "nodes": [
            {"node_params": {"layer_type": "input", "shape": [None, 128]}},
            {"node_params": {"layer_type": "lstm", "units": 64, "return_sequences": True}},
        ]
    }

    with (
        patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"),
        patch("tensorflow.keras.models.load_model", return_value=mock_keras_model),
        pytest.raises(ONNXUnsupportedError) as exc_info,
    ):
        export_onnx(job_id, model_name, graph_ir)

    assert len(exc_info.value.issues) > 0


@pytest.mark.skipif(
    not importlib.util.find_spec("tf2onnx") or not _check_tf2onnx_available(),
    reason="tf2onnx not available or dependencies incompatible",
)
def test_onnx_export_succeeds_with_fixed_shape(tmp_path, mock_keras_model):
    """ONNX export succeeds for LSTM with fixed input shape."""
    job_id = str(uuid4())
    model_name = "test_model"

    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)
    (export_dir / "model.keras").write_text("dummy")

    graph_ir = {
        "nodes": [
            {"node_params": {"layer_type": "input", "shape": [10, 128]}},  # Fixed length
            {"node_params": {"layer_type": "lstm", "units": 64, "return_sequences": True}},
        ]
    }

    # Mock ONNX model
    mock_onnx_model = Mock()
    mock_onnx_model.SerializeToString.return_value = b"onnx model bytes"

    with (
        patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"),
        patch("tensorflow.keras.models.load_model", return_value=mock_keras_model),
        patch("tensorflow.TensorSpec", return_value="tensor_spec"),
        patch("tf2onnx.convert.from_keras", return_value=(mock_onnx_model, None)),
    ):
        onnx_path = export_onnx(job_id, model_name, graph_ir)

        assert onnx_path.exists()


def test_get_export_formats_structure(tmp_path):
    """get_export_formats returns correct structure."""
    job_id = str(uuid4())
    model_name = "test_model"

    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)
    (export_dir / "model.keras").write_text("dummy")

    graph_ir = {"nodes": [{"node_params": {"layer_type": "dense", "units": 10}}]}

    mock_model = Mock()
    with (
        patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"),
        patch("tensorflow.keras.models.load_model", return_value=mock_model),
    ):
        formats = get_export_formats(job_id, model_name, graph_ir)

        assert "savedmodel" in formats
        assert "tflite" in formats
        assert "onnx" in formats

        # Check structure
        assert "available" in formats["savedmodel"]
        assert "size_bytes" in formats["savedmodel"]
        assert "expires_at" in formats["savedmodel"]

        # ONNX should have additional fields
        assert "onnx_supported" in formats["onnx"]


def test_cleanup_deletes_old_directories(tmp_path, db_session: Session):
    """cleanup_exports deletes export directories older than retention_days."""
    # Create old export directory
    old_job_id = str(uuid4())
    old_dir = tmp_path / "exports" / old_job_id
    old_dir.mkdir(parents=True)
    (old_dir / "model.keras").write_text("old")

    # Create recent export directory
    recent_job_id = str(uuid4())
    recent_dir = tmp_path / "exports" / recent_job_id
    recent_dir.mkdir(parents=True)
    (recent_dir / "model.keras").write_text("recent")

    # Parent model row so the training_job FK constraint is satisfied (Postgres/CI).
    db_session.add(ModelBasic(id=1, model_name="export-cleanup-model"))
    db_session.commit()

    # Create training jobs
    old_job = TrainingJob(
        id=old_job_id,
        model_id=1,
        status=TrainingStatus.COMPLETED,
        hyperparams={},
        completed_at=datetime.now(UTC) - timedelta(days=10),
    )
    recent_job = TrainingJob(
        id=recent_job_id,
        model_id=1,
        status=TrainingStatus.COMPLETED,
        hyperparams={},
        completed_at=datetime.now(UTC),
    )

    db_session.add(old_job)
    db_session.add(recent_job)
    db_session.commit()

    with patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"):
        count = cleanup_exports(retention_days=7, session=db_session)

        assert count == 1
        assert not old_dir.exists()
        assert recent_dir.exists()


def test_delete_job_exports(tmp_path):
    """delete_job_exports removes export directory for a specific job."""
    job_id = str(uuid4())
    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)
    (export_dir / "model.keras").write_text("dummy")

    with patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"):
        result = delete_job_exports(job_id)

        assert result is True
        assert not export_dir.exists()


def test_delete_model_exports_cascades(tmp_path, db_session: Session):
    """delete_model_exports removes all export directories for model's jobs."""
    model_id = 1

    # Parent model row so the training_job FK constraint is satisfied (Postgres/CI).
    db_session.add(ModelBasic(id=model_id, model_name="export-cascade-model"))
    db_session.commit()

    # Create multiple jobs for the model
    job1_id = str(uuid4())
    job2_id = str(uuid4())

    job1 = TrainingJob(id=job1_id, model_id=model_id, status=TrainingStatus.COMPLETED, hyperparams={})
    job2 = TrainingJob(id=job2_id, model_id=model_id, status=TrainingStatus.COMPLETED, hyperparams={})

    db_session.add(job1)
    db_session.add(job2)
    db_session.commit()

    # Create export directories
    job1_dir = tmp_path / "exports" / job1_id
    job1_dir.mkdir(parents=True)
    (job1_dir / "model.keras").write_text("job1")

    job2_dir = tmp_path / "exports" / job2_id
    job2_dir.mkdir(parents=True)
    (job2_dir / "model.keras").write_text("job2")

    with patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"):
        count = delete_model_exports(model_id, db_session)

        assert count == 2
        assert not job1_dir.exists()
        assert not job2_dir.exists()


def test_path_traversal_prevention(tmp_path, mock_keras_model):
    """Model names with path traversal attempts are sanitized."""
    job_id = str(uuid4())
    # Attempt path traversal with malicious model name
    malicious_name = "../../../tmp/pwned"

    export_dir = tmp_path / "exports" / job_id
    export_dir.mkdir(parents=True)
    (export_dir / "model.keras").write_text("dummy")

    with (
        patch("app.services.model_export.EXPORTS_BASE", tmp_path / "exports"),
        patch("tensorflow.keras.models.load_model", return_value=mock_keras_model),
        patch("tensorflow.keras.models.save_model"),
    ):
        # Create savedmodel directory for zipping
        savedmodel_dir = export_dir / "savedmodel"
        savedmodel_dir.mkdir()
        (savedmodel_dir / "saved_model.pb").write_text("dummy pb")

        zip_path = export_savedmodel(job_id, malicious_name)

        # Verify the file is created inside the export_dir, not outside
        assert zip_path.is_relative_to(export_dir)
        assert ".." not in str(zip_path)
        # Verify sanitized name doesn't contain path separators
        assert "/" not in zip_path.name
        assert "\\" not in zip_path.name
        # Verify the actual path traversal was prevented
        assert zip_path.parent == export_dir

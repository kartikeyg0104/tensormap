"""Tests for app/services/data_process.py.

All DB interactions use MagicMock – no real database required.
File-system interactions use pytest's tmp_path fixture.
"""

import json
import uuid
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from app.models.data import DataFile, DataProcess, ImageProperties
from app.schemas.data_process import TransformationItem
from app.services.data_process import (
    add_target_service,
    delete_one_target_by_id_service,
    get_all_targets_service,
    get_data_metrics,
    get_file_data,
    get_one_target_by_id_service,
    preprocess_data,
    update_image_properties,
)

# ── shared helpers ─────────────────────────────────────────────────────────────

SAMPLE_CSV = "sepal_length,sepal_width,species\n5.1,3.5,setosa\n6.0,2.9,virginica\n4.7,3.2,setosa\n"


def _make_csv(tmp_path: Path, file_name: str = "dataset") -> tuple:
    """Write a CSV to *tmp_path* and return a matching DataFile stub + its path."""
    csv_path = tmp_path / f"{file_name}.csv"
    csv_path.write_text(SAMPLE_CSV)
    file = DataFile(id=uuid.uuid4(), file_name=file_name, file_type="csv")
    return file, csv_path


def _db_returning(first_value) -> MagicMock:
    """Return a Session mock whose exec().first() yields *first_value*."""
    db = MagicMock()
    db.exec.return_value.first.return_value = first_value
    return db


def _db_returning_sequence(values: list) -> MagicMock:
    """Return a Session mock whose successive exec().first() calls yield *values*."""
    db = MagicMock()
    db.exec.return_value.first.side_effect = values
    return db


def _db_returning_all(all_value) -> MagicMock:
    """Return a Session mock whose exec().all() yields *all_value*."""
    db = MagicMock()
    db.exec.return_value.all.return_value = all_value
    return db


# ── add_target_service ─────────────────────────────────────────────────────────


def test_add_target_success():
    """Creates a DataProcess record and returns 201 when file exists."""
    file = DataFile(id=uuid.uuid4(), file_name="iris", file_type="csv")
    db = _db_returning(file)

    body, code = add_target_service(db, file_id=file.id, target="species")

    assert code == 201
    assert body["success"] is True
    db.add.assert_called_once()
    db.commit.assert_called_once()


def test_add_target_file_not_found():
    """Returns 400 when file_id is absent from the database."""
    db = _db_returning(None)

    body, code = add_target_service(db, file_id=uuid.uuid4(), target="label")

    assert code == 400
    assert body["success"] is False


def test_add_target_db_error():
    """Returns 500 when the database raises an unexpected exception."""
    db = MagicMock()
    db.exec.side_effect = RuntimeError("connection lost")

    body, code = add_target_service(db, file_id=uuid.uuid4(), target="label")

    assert code == 500
    assert body["success"] is False
    assert "connection lost" in body["message"]


# ── get_all_targets_service ────────────────────────────────────────────────────


def test_get_all_targets_with_records():
    """Returns a list with one entry per DataProcess that has a matching DataFile."""
    file_id = uuid.uuid4()
    proc = DataProcess(id=1, file_id=file_id, target="species")
    file = DataFile(id=file_id, file_name="iris", file_type="csv")

    db = MagicMock()
    db.exec.return_value.all.return_value = [proc]
    db.exec.return_value.first.return_value = file

    body, code = get_all_targets_service(db)

    assert code == 200
    assert body["success"] is True
    assert len(body["data"]) == 1
    record = body["data"][0]
    assert record["file_name"] == "iris"
    assert record["target_field"] == "species"


def test_get_all_targets_empty():
    """Returns an empty list when no DataProcess records exist."""
    db = _db_returning_all([])

    body, code = get_all_targets_service(db)

    assert code == 200
    assert body["data"] == []


def test_get_all_targets_skips_orphaned_process():
    """Skips DataProcess records whose DataFile no longer exists."""
    proc = DataProcess(id=2, file_id=uuid.uuid4(), target="label")

    db = MagicMock()
    db.exec.return_value.all.return_value = [proc]
    # The file lookup returns None (orphaned)
    db.exec.return_value.first.return_value = None

    body, code = get_all_targets_service(db)

    assert code == 200
    assert body["data"] == []


# ── delete_one_target_by_id_service ───────────────────────────────────────────


def test_delete_target_success():
    """Deletes the DataProcess record and returns 200."""
    file_id = uuid.uuid4()
    file = DataFile(id=file_id, file_name="iris", file_type="csv")
    proc = DataProcess(id=1, file_id=file_id, target="species")

    db = _db_returning_sequence([file, proc])

    body, code = delete_one_target_by_id_service(db, file_id=file_id)

    assert code == 200
    assert body["success"] is True
    db.delete.assert_called_once_with(proc)
    db.commit.assert_called_once()


def test_delete_target_file_not_found():
    """Returns 400 when the DataFile doesn't exist."""
    db = _db_returning(None)

    body, code = delete_one_target_by_id_service(db, file_id=uuid.uuid4())

    assert code == 400
    assert body["success"] is False


def test_delete_target_process_not_found():
    """Returns 400 when there is no DataProcess record for the file."""
    file = DataFile(id=uuid.uuid4(), file_name="iris", file_type="csv")
    db = _db_returning_sequence([file, None])

    body, code = delete_one_target_by_id_service(db, file_id=file.id)

    assert code == 400
    assert body["success"] is False


# ── get_one_target_by_id_service ──────────────────────────────────────────────


def test_get_one_target_success():
    """Returns target metadata for a known file_id."""
    file_id = uuid.uuid4()
    file = DataFile(id=file_id, file_name="iris", file_type="csv")
    proc = DataProcess(id=1, file_id=file_id, target="species")

    db = _db_returning_sequence([file, proc])

    body, code = get_one_target_by_id_service(db, file_id=file_id)

    assert code == 200
    assert body["success"] is True
    assert body["data"]["target_field"] == "species"
    assert body["data"]["file_name"] == "iris"


def test_get_one_target_file_not_found():
    """Returns 400 when the DataFile doesn't exist."""
    db = _db_returning(None)

    body, code = get_one_target_by_id_service(db, file_id=uuid.uuid4())

    assert code == 400
    assert body["success"] is False


def test_get_one_target_process_not_found():
    """Returns 400 when no target record exists for the file."""
    file = DataFile(id=uuid.uuid4(), file_name="iris", file_type="csv")
    db = _db_returning_sequence([file, None])

    body, code = get_one_target_by_id_service(db, file_id=file.id)

    assert code == 400
    assert body["success"] is False


# ── get_data_metrics ──────────────────────────────────────────────────────────


def test_get_data_metrics_success(tmp_path):
    """Returns a dict with data_types, correlation_matrix, and metric for a valid CSV."""
    file, csv_path = _make_csv(tmp_path)
    db = _db_returning(file)

    with patch("app.services.data_process._get_file_path", return_value=str(csv_path)):
        body, code = get_data_metrics(db, file_id=file.id)

    assert code == 200
    assert body["success"] is True
    data = body["data"]
    assert "data_types" in data
    assert "correlation_matrix" in data
    assert "metric" in data
    # Numeric columns should appear in the correlation matrix
    assert "sepal_length" in data["correlation_matrix"]


def test_get_data_metrics_file_not_found():
    """Returns 400 when no DataFile exists for the given id."""
    db = _db_returning(None)

    body, code = get_data_metrics(db, file_id=uuid.uuid4())

    assert code == 400
    assert body["success"] is False


# ── get_file_data ─────────────────────────────────────────────────────────────


def test_get_file_data_success(tmp_path):
    """Returns JSON-serialised rows for a valid CSV file."""
    file, csv_path = _make_csv(tmp_path)
    db = _db_returning(file)

    with patch("app.services.data_process._get_file_path", return_value=str(csv_path)):
        body, code = get_file_data(db, file_id=file.id)

    assert code == 200
    assert body["success"] is True
    rows = json.loads(body["data"])
    assert len(rows) == 3
    assert rows[0]["species"] == "setosa"


def test_get_file_data_file_not_found():
    """Returns 400 when the DataFile record is missing."""
    db = _db_returning(None)

    body, code = get_file_data(db, file_id=uuid.uuid4())

    assert code == 400
    assert body["success"] is False


# ── preprocess_data ───────────────────────────────────────────────────────────


def test_preprocess_one_hot_encoding(tmp_path):
    """One-hot-encoding a categorical column expands it into binary columns."""
    file, csv_path = _make_csv(tmp_path)
    db = _db_returning(file)

    transformations = [TransformationItem(transformation="One Hot Encoding", feature="species")]

    with (
        patch("app.services.data_process._get_file_path", return_value=str(csv_path)),
        patch("app.services.data_process.get_settings") as mock_settings,
    ):
        mock_settings.return_value.upload_folder = str(tmp_path)
        body, code = preprocess_data(db, file_id=file.id, transformations=transformations)

    assert code == 200
    assert body["success"] is True
    # The resulting file must exist on disk
    output_files = list(tmp_path.glob("*_preprocessed.csv"))
    assert len(output_files) == 1
    result_df = pd.read_csv(output_files[0])
    # Original 'species' column should be gone, replaced by dummy columns
    assert "species" not in result_df.columns
    assert any(col.startswith("species_") for col in result_df.columns)


def test_preprocess_categorical_to_numerical(tmp_path):
    """Categorical-to-numerical encodes a string column as integer codes."""
    file, csv_path = _make_csv(tmp_path)
    db = _db_returning(file)

    transformations = [TransformationItem(transformation="Categorical to Numerical", feature="species")]

    with (
        patch("app.services.data_process._get_file_path", return_value=str(csv_path)),
        patch("app.services.data_process.get_settings") as mock_settings,
    ):
        mock_settings.return_value.upload_folder = str(tmp_path)
        body, code = preprocess_data(db, file_id=file.id, transformations=transformations)

    assert code == 200
    output_files = list(tmp_path.glob("*_preprocessed.csv"))
    result_df = pd.read_csv(output_files[0])
    assert result_df["species"].dtype in (int, "int64", "int8")


def test_preprocess_drop_column(tmp_path):
    """Drop-column removes the specified column entirely."""
    file, csv_path = _make_csv(tmp_path)
    db = _db_returning(file)

    transformations = [TransformationItem(transformation="Drop Column", feature="sepal_width")]

    with (
        patch("app.services.data_process._get_file_path", return_value=str(csv_path)),
        patch("app.services.data_process.get_settings") as mock_settings,
    ):
        mock_settings.return_value.upload_folder = str(tmp_path)
        body, code = preprocess_data(db, file_id=file.id, transformations=transformations)

    assert code == 200
    output_files = list(tmp_path.glob("*_preprocessed.csv"))
    result_df = pd.read_csv(output_files[0])
    assert "sepal_width" not in result_df.columns
    assert "sepal_length" in result_df.columns


def test_preprocess_multiple_transformations(tmp_path):
    """Multiple transformations are applied in order."""
    file, csv_path = _make_csv(tmp_path)
    db = _db_returning(file)

    transformations = [
        TransformationItem(transformation="Categorical to Numerical", feature="species"),
        TransformationItem(transformation="Drop Column", feature="sepal_width"),
    ]

    with (
        patch("app.services.data_process._get_file_path", return_value=str(csv_path)),
        patch("app.services.data_process.get_settings") as mock_settings,
    ):
        mock_settings.return_value.upload_folder = str(tmp_path)
        body, code = preprocess_data(db, file_id=file.id, transformations=transformations)

    assert code == 200
    output_files = list(tmp_path.glob("*_preprocessed.csv"))
    result_df = pd.read_csv(output_files[0])
    assert "sepal_width" not in result_df.columns
    assert result_df["species"].dtype in (int, "int64", "int8")


def test_preprocess_file_not_found():
    """Returns 400 when the DataFile doesn't exist in the database."""
    db = _db_returning(None)
    transformations = [TransformationItem(transformation="Drop Column", feature="col")]

    body, code = preprocess_data(db, file_id=uuid.uuid4(), transformations=transformations)

    assert code == 400
    assert body["success"] is False


def test_preprocess_invalid_csv(tmp_path):
    """Returns 500 when the file on disk is not valid CSV."""
    bad_csv = tmp_path / "bad.csv"
    # Write a binary file that is not valid CSV
    bad_csv.write_bytes(b"\x00\x01\x02\x03" * 100)
    file = DataFile(id=uuid.uuid4(), file_name="bad", file_type="csv")
    db = _db_returning(file)

    transformations = [TransformationItem(transformation="Drop Column", feature="col")]

    with (
        patch("app.services.data_process._get_file_path", return_value=str(bad_csv)),
        patch("app.services.data_process.get_settings") as mock_settings,
    ):
        mock_settings.return_value.upload_folder = str(tmp_path)
        body, code = preprocess_data(db, file_id=file.id, transformations=transformations)

    assert code == 500
    assert body["success"] is False


# ── update_image_properties ───────────────────────────────────────────────────


def test_update_image_properties_creates_new():
    """Creates a new ImageProperties record when none exists and returns 201."""
    file_id = uuid.uuid4()
    db = _db_returning(None)

    body, code = update_image_properties(db, file_id=file_id, image_size=128, batch_size=32,
                                         color_mode="rgb", label_mode="categorical")

    assert code == 201
    assert body["success"] is True
    db.add.assert_called_once()
    db.commit.assert_called_once()


def test_update_image_properties_updates_existing():
    """Updates an existing ImageProperties record and returns 200."""
    file_id = uuid.uuid4()
    existing = ImageProperties(id=file_id, image_size=64, batch_size=16,
                               color_mode="grayscale", label_mode="int")
    db = _db_returning(existing)

    body, code = update_image_properties(db, file_id=file_id, image_size=128, batch_size=32,
                                         color_mode="rgb", label_mode="categorical")

    assert code == 200
    assert body["success"] is True
    assert existing.image_size == 128
    assert existing.batch_size == 32
    assert existing.color_mode == "rgb"
    assert existing.label_mode == "categorical"
    db.add.assert_called_once_with(existing)
    db.commit.assert_called_once()


def test_update_image_properties_db_error():
    """Returns 500 when the database raises an unexpected exception."""
    db = MagicMock()
    db.exec.side_effect = RuntimeError("db failure")

    body, code = update_image_properties(db, file_id=uuid.uuid4(), image_size=64, batch_size=16,
                                         color_mode="grayscale", label_mode="int")

    assert code == 500
    assert body["success"] is False

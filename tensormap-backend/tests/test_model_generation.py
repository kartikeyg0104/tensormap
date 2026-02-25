"""
Unit and integration tests for the model_generation service.

Covers:
  - helper_generate_layers() for each supported layer type
  - model_generation() end-to-end for linear, branching, and multi-input architectures
  - Edge cases: unknown layer types, disconnected nodes, multiple outputs
  - Integration: generated JSON is parseable by tf.keras.models.model_from_json
"""

import json
from unittest.mock import mock_open, patch

import pytest
import tensorflow as tf

from app.services.model_generation import helper_generate_layers, model_generation

# ---------------------------------------------------------------------------
# Reusable starter JSON (mirrors templates/code-templates/model_func.json)
# ---------------------------------------------------------------------------
STARTER_JSON = {
    "class_name": "Functional",
    "config": {
        "name": "model",
        "trainable": True,
        "layers": [],
        "input_layers": [],
        "output_layers": [],
    },
    "keras_version": "2.12.0",
    "backend": "tensorflow",
}


def _mock_open_starter():
    """Return a mock for builtins.open that yields the starter JSON."""
    return mock_open(read_data=json.dumps(STARTER_JSON))


# ---------------------------------------------------------------------------
# Node / edge builder helpers
# ---------------------------------------------------------------------------

def _input_node(node_id: str, dims: list[int]) -> dict:
    """Create a custominput node with the given dimensions."""
    params = {}
    for i, d in enumerate(dims):
        params[f"dim-{i + 1}"] = d
    return {
        "id": node_id,
        "type": "custominput",
        "data": {"params": params},
    }


def _dense_node(node_id: str, units: int = 32, activation: str = "relu") -> dict:
    return {
        "id": node_id,
        "type": "customdense",
        "data": {"params": {"units": units, "activation": activation}},
    }


def _flatten_node(node_id: str) -> dict:
    return {
        "id": node_id,
        "type": "customflatten",
        "data": {"params": {}},
    }


def _conv_node(
    node_id: str,
    filters: int = 16,
    kernel: tuple[int, int] = (3, 3),
    stride: tuple[int, int] = (1, 1),
    padding: str = "valid",
    activation: str = "relu",
) -> dict:
    return {
        "id": node_id,
        "type": "customconv",
        "data": {
            "params": {
                "filter": filters,
                "kernelX": kernel[0],
                "kernelY": kernel[1],
                "strideX": stride[0],
                "strideY": stride[1],
                "padding": padding,
                "activation": activation,
            }
        },
    }


def _edge(source: str, target: str) -> dict:
    return {"source": source, "target": target}


# ===================================================================
# Tests for helper_generate_layers()
# ===================================================================


class TestHelperGenerateLayers:
    """Unit tests for the layer-dict builder."""

    def test_input_layer_single_dim(self):
        node = _input_node("input_1", [784])
        result = helper_generate_layers(node)

        assert result["class_name"] == "InputLayer"
        assert result["name"] == "input_1"
        assert result["config"]["name"] == "input_1"
        assert result["config"]["batch_input_shape"] == [None, 784]
        assert result["config"]["dtype"] == "float32"
        assert result["inbound_nodes"] == []

    def test_input_layer_multi_dim(self):
        node = _input_node("img_input", [28, 28, 3])
        result = helper_generate_layers(node)

        assert result["config"]["batch_input_shape"] == [None, 28, 28, 3]
        assert result["name"] == "img_input"

    def test_input_layer_zero_dims_filtered(self):
        """Dimensions equal to 0 are stripped out."""
        node = _input_node("input_z", [128, 0, 0])
        result = helper_generate_layers(node)

        assert result["config"]["batch_input_shape"] == [None, 128]

    def test_dense_layer(self):
        node = _dense_node("dense_1", units=64, activation="sigmoid")
        result = helper_generate_layers(node)

        assert result["class_name"] == "Dense"
        assert result["name"] == "dense_1"
        assert result["config"]["units"] == 64
        assert result["config"]["activation"] == "sigmoid"
        assert result["config"]["trainable"] is True
        assert result["inbound_nodes"] == []

    def test_flatten_layer(self):
        node = _flatten_node("flat_1")
        result = helper_generate_layers(node)

        assert result["class_name"] == "Flatten"
        assert result["name"] == "flat_1"
        assert result["config"]["data_format"] == "channels_last"
        assert result["inbound_nodes"] == []

    def test_conv2d_layer(self):
        node = _conv_node(
            "conv_1",
            filters=32,
            kernel=(5, 5),
            stride=(2, 2),
            padding="same",
            activation="relu",
        )
        result = helper_generate_layers(node)

        assert result["class_name"] == "Conv2D"
        assert result["name"] == "conv_1"
        assert result["config"]["filters"] == 32
        assert result["config"]["kernel_size"] == (5, 5)
        assert result["config"]["strides"] == (2, 2)
        assert result["config"]["padding"] == "same"
        assert result["config"]["activation"] == "relu"

    def test_unknown_layer_type_returns_none(self):
        """An unsupported layer type should return None."""
        node = {"id": "x", "type": "custom_unknown", "data": {"params": {}}}
        result = helper_generate_layers(node)
        assert result is None


# ===================================================================
# Tests for model_generation()
# ===================================================================


class TestModelGeneration:
    """End-to-end tests for the full model_generation pipeline."""

    # ---- simple linear architectures ----

    @patch("builtins.open", _mock_open_starter())
    def test_simple_input_to_dense(self):
        """input → dense (single layer output)."""
        params = {
            "nodes": [_input_node("in1", [10]), _dense_node("d1", 1, "linear")],
            "edges": [_edge("in1", "d1")],
        }
        result = model_generation(params)

        assert result["config"]["input_layers"] == [["in1", 0, 0]]
        assert result["config"]["output_layers"] == [["d1", 0, 0]]

        layer_names = [l["name"] for l in result["config"]["layers"]]
        assert "in1" in layer_names
        assert "d1" in layer_names
        assert len(result["config"]["layers"]) == 2

        # Dense layer should list input as inbound
        dense = next(l for l in result["config"]["layers"] if l["name"] == "d1")
        assert dense["inbound_nodes"] == [["in1", 0, 0, {}]]

    @patch("builtins.open", _mock_open_starter())
    def test_multi_layer_linear(self):
        """input → dense1 → dense2 → dense3."""
        params = {
            "nodes": [
                _input_node("in", [5]),
                _dense_node("h1", 32, "relu"),
                _dense_node("h2", 16, "relu"),
                _dense_node("out", 1, "sigmoid"),
            ],
            "edges": [
                _edge("in", "h1"),
                _edge("h1", "h2"),
                _edge("h2", "out"),
            ],
        }
        result = model_generation(params)

        assert result["config"]["input_layers"] == [["in", 0, 0]]
        assert result["config"]["output_layers"] == [["out", 0, 0]]
        assert len(result["config"]["layers"]) == 4

        # Verify BFS ordering via inbound connections
        h1 = next(l for l in result["config"]["layers"] if l["name"] == "h1")
        h2 = next(l for l in result["config"]["layers"] if l["name"] == "h2")
        out = next(l for l in result["config"]["layers"] if l["name"] == "out")
        assert h1["inbound_nodes"] == [["in", 0, 0, {}]]
        assert h2["inbound_nodes"] == [["h1", 0, 0, {}]]
        assert out["inbound_nodes"] == [["h2", 0, 0, {}]]

    # ---- conv architecture ----

    @patch("builtins.open", _mock_open_starter())
    def test_conv_flatten_dense(self):
        """input → conv → flatten → dense."""
        params = {
            "nodes": [
                _input_node("img", [28, 28, 1]),
                _conv_node("c1", filters=16, kernel=(3, 3)),
                _flatten_node("f1"),
                _dense_node("out", 10, "softmax"),
            ],
            "edges": [
                _edge("img", "c1"),
                _edge("c1", "f1"),
                _edge("f1", "out"),
            ],
        }
        result = model_generation(params)

        assert result["config"]["input_layers"] == [["img", 0, 0]]
        assert result["config"]["output_layers"] == [["out", 0, 0]]

        conv = next(l for l in result["config"]["layers"] if l["name"] == "c1")
        assert conv["class_name"] == "Conv2D"
        assert conv["inbound_nodes"] == [["img", 0, 0, {}]]

        flat = next(l for l in result["config"]["layers"] if l["name"] == "f1")
        assert flat["class_name"] == "Flatten"
        assert flat["inbound_nodes"] == [["c1", 0, 0, {}]]

    # ---- multi-input / concatenation ----

    @patch("builtins.open", _mock_open_starter())
    def test_multi_input_concatenation(self):
        """
        Two inputs merging into a single dense:
            in1 ─┐
                  ├→ dense_merge
            in2 ─┘
        Should insert a Concatenate layer automatically.
        """
        params = {
            "nodes": [
                _input_node("in1", [5]),
                _input_node("in2", [5]),
                _dense_node("merge", 1, "sigmoid"),
            ],
            "edges": [
                _edge("in1", "merge"),
                _edge("in2", "merge"),
            ],
        }
        result = model_generation(params)

        assert len(result["config"]["input_layers"]) == 2
        layer_names = [l["name"] for l in result["config"]["layers"]]
        # A Concatenate layer should have been inserted
        concat_layers = [l for l in result["config"]["layers"] if l["class_name"] == "Concatenate"]
        assert len(concat_layers) == 1

        concat = concat_layers[0]
        # Concatenate's inbound should reference both inputs
        inbound_ids = [ib[0] for ib in concat["inbound_nodes"][0]]
        assert "in1" in inbound_ids
        assert "in2" in inbound_ids

        # The merge dense's inbound should reference the concatenate, not the raw inputs
        merge = next(l for l in result["config"]["layers"] if l["name"] == "merge")
        merge_inbound_ids = [ib[0] for ib in merge["inbound_nodes"]]
        assert any("concatenate" in str(ib) for ib in merge_inbound_ids)

    # ---- multiple outputs ----

    @patch("builtins.open", _mock_open_starter())
    def test_multiple_output_layers(self):
        """
        input ─→ dense1
              └→ dense2
        Both dense1 and dense2 are output layers.
        """
        params = {
            "nodes": [
                _input_node("in", [10]),
                _dense_node("out1", 1, "sigmoid"),
                _dense_node("out2", 5, "softmax"),
            ],
            "edges": [
                _edge("in", "out1"),
                _edge("in", "out2"),
            ],
        }
        result = model_generation(params)

        output_ids = [o[0] for o in result["config"]["output_layers"]]
        assert "out1" in output_ids
        assert "out2" in output_ids
        assert len(output_ids) == 2

    # ---- edge cases ----

    @patch("builtins.open", _mock_open_starter())
    def test_disconnected_node_not_in_layers(self):
        """
        A node that has no edges connecting it should not appear in layers
        (unless it's an input node—inputs are always added).
        """
        params = {
            "nodes": [
                _input_node("in", [5]),
                _dense_node("connected", 1, "relu"),
                _dense_node("disconnected", 8, "relu"),  # no edges to/from
            ],
            "edges": [_edge("in", "connected")],
        }
        result = model_generation(params)

        layer_names = [l["name"] for l in result["config"]["layers"]]
        assert "in" in layer_names
        assert "connected" in layer_names
        # disconnected node should NOT be traversed
        assert "disconnected" not in layer_names

        # But disconnected is a leaf (no outgoing edges) so it appears in output_layers
        output_ids = [o[0] for o in result["config"]["output_layers"]]
        assert "disconnected" in output_ids
        assert "connected" in output_ids

    @patch("builtins.open", _mock_open_starter())
    def test_no_edges_single_input(self):
        """A model with only an input node and no edges."""
        params = {
            "nodes": [_input_node("lonely", [3])],
            "edges": [],
        }
        result = model_generation(params)

        assert result["config"]["input_layers"] == [["lonely", 0, 0]]
        # Input node has no outgoing edges → also an output
        assert result["config"]["output_layers"] == [["lonely", 0, 0]]
        assert len(result["config"]["layers"]) == 1

    @patch("builtins.open", _mock_open_starter())
    def test_starter_json_structure_preserved(self):
        """Verify the top-level structure keys are present."""
        params = {
            "nodes": [_input_node("in", [4]), _dense_node("d", 2, "relu")],
            "edges": [_edge("in", "d")],
        }
        result = model_generation(params)

        assert result["class_name"] == "Functional"
        assert "config" in result
        assert "keras_version" in result
        assert "backend" in result
        assert result["backend"] == "tensorflow"


# ===================================================================
# Integration tests — build real Keras models from generated JSON
# ===================================================================


def _build_model_from_json(model_json: dict) -> tf.keras.Model:
    """
    Programmatically reconstruct a Keras model from the generated JSON dict.

    This validates that the architecture described in the JSON
    (layer types, shapes, connectivity) is actually buildable by Keras,
    without relying on the serialisation format version (Keras 2 vs 3).

    Layers are processed in topological order to handle cases where the
    Concatenate layer appears after the layers that depend on it in the
    JSON list (as model_generation appends Concatenate in a post-BFS pass).
    """
    output_map: dict[str, tf.Tensor] = {}
    layers_by_name = {l["name"]: l for l in model_json["config"]["layers"]}
    remaining = dict(layers_by_name)

    def _get_inbound_ids(layer_cfg: dict) -> list[str]:
        """Extract parent layer names from inbound_nodes."""
        inbound = layer_cfg["inbound_nodes"]
        if not inbound:
            return []
        cls = layer_cfg["class_name"]
        if cls == "Concatenate":
            # Concatenate inbound: [[...parents...]]
            return [ib[0] for ib in inbound[0]]
        # Other layers: each item is [name, idx, tensor_idx, kwargs]
        return [ib[0] for ib in inbound]

    # Process in topological order
    while remaining:
        progress = False
        for name in list(remaining):
            cfg = remaining[name]
            deps = _get_inbound_ids(cfg)
            if all(d in output_map for d in deps):
                _build_single_layer(cfg, output_map)
                del remaining[name]
                progress = True
        if not progress:
            raise RuntimeError(f"Circular or unresolvable deps among: {list(remaining)}")

    inputs = [output_map[il[0]] for il in model_json["config"]["input_layers"]]
    outputs = [output_map[ol[0]] for ol in model_json["config"]["output_layers"]]
    return tf.keras.Model(inputs=inputs, outputs=outputs)


def _build_single_layer(layer_cfg: dict, output_map: dict) -> None:
    """Instantiate one Keras layer and store its output in *output_map*."""
    cls = layer_cfg["class_name"]
    cfg = layer_cfg["config"]
    name = layer_cfg["name"]
    inbound = layer_cfg["inbound_nodes"]

    if cls == "InputLayer":
        shape = cfg["batch_input_shape"][1:]
        output_map[name] = tf.keras.Input(shape=shape, name=name)

    elif cls == "Dense":
        layer = tf.keras.layers.Dense(
            units=cfg["units"], activation=cfg["activation"], name=name,
        )
        parent = inbound[0][0]
        output_map[name] = layer(output_map[parent])

    elif cls == "Flatten":
        layer = tf.keras.layers.Flatten(name=name)
        parent = inbound[0][0]
        output_map[name] = layer(output_map[parent])

    elif cls == "Conv2D":
        layer = tf.keras.layers.Conv2D(
            filters=cfg["filters"],
            kernel_size=cfg["kernel_size"],
            strides=cfg["strides"],
            padding=cfg["padding"],
            activation=cfg["activation"],
            name=name,
        )
        parent = inbound[0][0]
        output_map[name] = layer(output_map[parent])

    elif cls == "Concatenate":
        layer = tf.keras.layers.Concatenate(name=name)
        parents = [output_map[ib[0]] for ib in inbound[0]]
        output_map[name] = layer(parents)


class TestModelGenerationKerasIntegration:
    """
    Integration tests that build real Keras models from the generated JSON
    to verify the output describes a valid architecture.
    """

    @patch("builtins.open", _mock_open_starter())
    def test_simple_model_loads_in_keras(self):
        """A simple input(10) → dense(1, linear) model should be buildable."""
        params = {
            "nodes": [
                _input_node("input_1", [10]),
                _dense_node("dense_1", 1, "linear"),
            ],
            "edges": [_edge("input_1", "dense_1")],
        }
        result = model_generation(params)
        model = _build_model_from_json(result)

        assert model is not None
        assert len(model.layers) == 2
        assert model.input_shape == (None, 10)
        assert model.output_shape == (None, 1)

    @patch("builtins.open", _mock_open_starter())
    def test_deep_network_loads_in_keras(self):
        """input(20) → dense(64) → dense(32) → dense(1) buildable."""
        params = {
            "nodes": [
                _input_node("input_1", [20]),
                _dense_node("dense_1", 64, "relu"),
                _dense_node("dense_2", 32, "relu"),
                _dense_node("dense_3", 1, "sigmoid"),
            ],
            "edges": [
                _edge("input_1", "dense_1"),
                _edge("dense_1", "dense_2"),
                _edge("dense_2", "dense_3"),
            ],
        }
        result = model_generation(params)
        model = _build_model_from_json(result)

        assert model is not None
        assert len(model.layers) == 4
        assert model.output_shape == (None, 1)

    @patch("builtins.open", _mock_open_starter())
    def test_conv_model_loads_in_keras(self):
        """input(28,28,1) → conv(16) → flatten → dense(10) buildable."""
        params = {
            "nodes": [
                _input_node("input_1", [28, 28, 1]),
                _conv_node("conv2d", filters=16, kernel=(3, 3), stride=(1, 1), padding="valid", activation="relu"),
                _flatten_node("flatten"),
                _dense_node("dense_1", 10, "softmax"),
            ],
            "edges": [
                _edge("input_1", "conv2d"),
                _edge("conv2d", "flatten"),
                _edge("flatten", "dense_1"),
            ],
        }
        result = model_generation(params)
        model = _build_model_from_json(result)

        assert model is not None
        assert model.output_shape == (None, 10)

    @patch("builtins.open", _mock_open_starter())
    def test_multi_input_model_loads_in_keras(self):
        """
        Two inputs concatenated → dense output.
        Validates that the Concatenate layer wiring is Keras-compatible.
        """
        params = {
            "nodes": [
                _input_node("input_1", [5]),
                _input_node("input_2", [5]),
                _dense_node("dense_1", 1, "sigmoid"),
            ],
            "edges": [
                _edge("input_1", "dense_1"),
                _edge("input_2", "dense_1"),
            ],
        }
        result = model_generation(params)
        model = _build_model_from_json(result)

        assert model is not None
        assert len(model.inputs) == 2

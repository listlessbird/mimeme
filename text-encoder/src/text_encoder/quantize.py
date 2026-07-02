from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
from onnxruntime.quantization import QuantType, quantize_dynamic

FP32_MATMUL_GROUPS = ("fc2", "out_proj")


def _find_token_embedding(model: onnx.ModelProto):
    initializers = {i.name: i for i in model.graph.initializer}
    for node in model.graph.node:
        if node.op_type != "Gather":
            continue
        init = initializers.get(node.input[0])
        if init is not None and len(init.dims) == 2 and init.dims[0] > 100_000:
            return node, init
    raise ValueError("token embedding Gather not found")


def convert_token_embedding_fp16(model: onnx.ModelProto) -> onnx.ModelProto:
    node, init = _find_token_embedding(model)
    table = numpy_helper.to_array(init).astype(np.float16)
    model.graph.initializer.remove(init)
    model.graph.initializer.append(numpy_helper.from_array(table, init.name + "_fp16"))

    raw_out = node.output[0]
    node.input[0] = init.name + "_fp16"
    node.output[0] = raw_out + "_fp16"
    idx = list(model.graph.node).index(node)
    model.graph.node.insert(
        idx + 1,
        helper.make_node("Cast", [raw_out + "_fp16"], [raw_out], to=TensorProto.FLOAT),
    )
    return model


def quantize_matmuls_dynamic_int8(model_path: Path, out_path: Path) -> None:
    model = onnx.load(str(model_path))
    initializers = {i.name for i in model.graph.initializer}
    targets = [
        node.name
        for node in model.graph.node
        if node.op_type == "MatMul"
        and node.input[1] in initializers
        and not any(group in node.name for group in FP32_MATMUL_GROUPS)
    ]
    quantize_dynamic(
        str(model_path),
        str(out_path),
        weight_type=QuantType.QInt8,
        per_channel=True,
        nodes_to_quantize=targets,
    )


def build_int8_hybrid(preprocessed_path: Path, out_path: Path) -> None:
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as tmp:
        dynq_path = Path(tmp.name)
    try:
        quantize_matmuls_dynamic_int8(preprocessed_path, dynq_path)
        model = convert_token_embedding_fp16(onnx.load(str(dynq_path)))
        onnx.save_model(model, str(out_path))
    finally:
        dynq_path.unlink(missing_ok=True)

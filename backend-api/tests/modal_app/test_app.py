from __future__ import annotations

import numpy as np

from mimeme.modal_app.app import _to_numpy


class _Tensor:
    def __init__(self, array: np.ndarray) -> None:
        self._array = array

    def cpu(self) -> _Tensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._array


class _Output:
    def __init__(self, field: str, tensor: _Tensor) -> None:
        setattr(self, field, tensor)


def test_gpu_half_precision_features_become_float32() -> None:
    tensor = _Tensor(np.array([[3.0, 0.0]], dtype=np.float16))

    array = _to_numpy(tensor, kind="image")

    assert array.dtype == np.float32
    assert array.tolist() == [[3.0, 0.0]]


def test_float32_features_are_passed_through_without_a_copy() -> None:
    source = np.array([[0.0, 4.0]], dtype=np.float32)

    array = _to_numpy(_Tensor(source), kind="image")

    assert array is source


def test_model_output_fields_are_cast_as_well() -> None:
    for kind, field in (("image", "image_embeds"), ("text", "text_embeds")):
        output = _Output(field, _Tensor(np.array([1.0], dtype=np.float16)))

        array = _to_numpy(output, kind=kind)

        assert array.dtype == np.float32

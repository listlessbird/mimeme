from __future__ import annotations

import hashlib

import pytest
from pydantic import SecretStr, ValidationError

from mimeme import storage


def test_object_accepts_positional_key() -> None:
    obj = storage.Object("images/ab/cd/x.jpg")
    assert obj.key == "images/ab/cd/x.jpg"
    assert str(obj) == "images/ab/cd/x.jpg"


@pytest.mark.parametrize(
    "key",
    ["", "/leading", "trailing/", "a//b", "a/../b", "a/./b", " spaced "],
)
def test_object_rejects_unsafe_keys(key: str) -> None:
    with pytest.raises(ValidationError):
        storage.Object(key)


def test_object_is_frozen_and_forbids_extra() -> None:
    obj = storage.Object("a/b")
    with pytest.raises(ValidationError):
        obj.key = "other"


def test_checksum_of_matches_hashlib() -> None:
    data = b"hello world"
    checksum = storage.Checksum.of(data)
    assert checksum.algorithm == "sha256"
    assert checksum.value == hashlib.sha256(data).hexdigest()


def test_checksum_rejects_non_hex() -> None:
    with pytest.raises(ValidationError):
        storage.Checksum(value="not-a-real-hash")


def test_config_redacts_secret() -> None:
    config = storage.Config(
        endpoint_url="http://localhost:9000",
        region="us-east-1",
        access_key="key",
        secret_key=SecretStr("super-secret"),
        bucket="mimeme-media",
    )
    assert "super-secret" not in repr(config)
    assert config.secret_key.get_secret_value() == "super-secret"

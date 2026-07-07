from typing import BinaryIO, Protocol


class ApiStorage(Protocol):
    def presign(self, key: str, expiration: int = 3600) -> str: ...
    def upload_bytes(self, data: bytes | BinaryIO, key: str, content_type: str) -> str: ...
    def delete(self, key: str) -> None: ...
    def exists(self, key: str) -> bool: ...


class ApiStorageAdapter(Protocol):
    def generate_presigned_url(self, key: str, expiration: int = 3600) -> str: ...
    def upload_bytes(self, data: bytes | BinaryIO, key: str, content_type: str) -> str: ...
    def delete(self, key: str) -> None: ...
    def exists(self, key: str) -> bool: ...


class BotoApiStorage:
    def __init__(self, storage: ApiStorageAdapter) -> None:
        self._storage = storage

    def presign(self, key: str, expiration: int = 3600) -> str:
        return self._storage.generate_presigned_url(key, expiration=expiration)

    def upload_bytes(self, data: bytes | BinaryIO, key: str, content_type: str) -> str:
        return self._storage.upload_bytes(data, key, content_type)

    def delete(self, key: str) -> None:
        self._storage.delete(key)

    def exists(self, key: str) -> bool:
        return self._storage.exists(key)

from mimeme.shared.services.storage import S3Config, StorageService


def test_storage_services_address_their_assigned_bucket() -> None:
    media = StorageService(
        S3Config(
            endpoint_url="https://r2.example.test",
            region="auto",
            access_key="media-key",
            secret_key="media-secret",
            bucket="mimeme-media",
            force_path_style=False,
        )
    )
    artifacts = StorageService(
        S3Config(
            endpoint_url="https://r2.example.test",
            region="auto",
            access_key="artifact-key",
            secret_key="artifact-secret",
            bucket="mimeme-artifacts-prod",
            force_path_style=False,
        )
    )

    assert media.bucket == "mimeme-media"
    assert artifacts.bucket == "mimeme-artifacts-prod"

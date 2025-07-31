import asyncio
import os
import uuid
from datetime import timedelta
from pathlib import Path

from google.cloud import storage

DEFAULT_SERVICE_ACCOUNT = "r8-proxy-models-input-us@replicate.iam.gserviceaccount.com"
DEFAULT_BUCKET = "replicate-proxy-models-input-us"
GOOGLE_APPLICATION_CREDENTIALS = Path(".gcp-service-account.json")


class ReplicateGCPBucket:
    def __init__(self):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(
            GOOGLE_APPLICATION_CREDENTIALS.resolve()
        )
        self.service_account = DEFAULT_SERVICE_ACCOUNT
        self.bucket_name = DEFAULT_BUCKET
        self.client = storage.Client()

    def upload_raw(self, local_path: str, blob_name: str) -> None:
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(blob_name)
        try:
            blob.upload_from_filename(local_path)
        except Exception:
            raise Exception(f"Failed to upload {local_path}")

    def get_signed_url(
        self, blob_name: str, expiration: timedelta = timedelta(minutes=15)
    ) -> str:
        bucket = self.client.bucket(self.bucket_name)
        blob = bucket.blob(blob_name)
        return blob.generate_signed_url(
            version="v4",
            method="GET",
            expiration=expiration,
            service_account_email=self.service_account,
        )

    def _upload_file_sync(
        self, file_path: Path, expiration: timedelta = timedelta(minutes=15)
    ) -> str:
        unique_id = uuid.uuid4().hex
        blob_name = f"{unique_id}_{file_path.name}"
        self.upload_raw(str(file_path), blob_name)
        return self.get_signed_url(blob_name, expiration)

    async def upload_file(
        self, file_path: Path, expiration: timedelta = timedelta(minutes=15)
    ) -> str:
        return await asyncio.to_thread(self._upload_file_sync, file_path, expiration)

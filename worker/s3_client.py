from __future__ import annotations

import os

import boto3
from botocore.client import Config

from worker.config import settings


def _client():
    return boto3.client(
        "s3",
        endpoint_url=settings.S3_ENDPOINT_URL,
        aws_access_key_id=settings.S3_ACCESS_KEY,
        aws_secret_access_key=settings.S3_SECRET_KEY,
        config=Config(signature_version="s3v4"),
    )


def download(s3_key: str, local_path: str) -> None:
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    _client().download_file(settings.S3_BUCKET, s3_key, local_path)


def upload(local_path: str, s3_key: str, content_type: str = "application/octet-stream") -> None:
    _client().upload_file(
        local_path,
        settings.S3_BUCKET,
        s3_key,
        ExtraArgs={"ContentType": content_type},
    )


def delete_prefix(prefix: str) -> int:
    client = _client()
    paginator = client.get_paginator("list_objects_v2")
    deleted = 0
    for page in paginator.paginate(Bucket=settings.S3_BUCKET, Prefix=prefix):
        objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
        if objects:
            client.delete_objects(Bucket=settings.S3_BUCKET, Delete={"Objects": objects})
            deleted += len(objects)
    return deleted

"""Azure Blob Storage utility functions."""

import os

from azure.storage.blob import BlobServiceClient


def download_invoice_from_blob(file_path: str) -> bytes:
    """
    Download an invoice file from Azure Blob Storage.

    The file_path must include the container name as its first segment,
    followed by the blob path.

    Args:
        file_path: Full path in the format '<container>/<blob_name>'
                   e.g. 'invoices/01-2026/123456.pdf'

    Returns:
        File content as bytes

    Raises:
        ValueError: If AZURE_STORAGE_CONNECTION_STRING is not set or
                    file_path format is invalid.
        Exception:  If the blob download fails.
    """
    conn_str = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    if not conn_str:
        raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable is not set")

    # Split into container (first segment) and blob name (rest)
    parts = file_path.lstrip("/").split("/", 1)
    if len(parts) < 2:
        raise ValueError(f"Invalid invoice_file_path format (expected '<container>/...') — got: {file_path}")

    container_name = parts[0]
    blob_name = parts[1]

    blob_service_client = BlobServiceClient.from_connection_string(conn_str)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)
    return blob_client.download_blob().readall()

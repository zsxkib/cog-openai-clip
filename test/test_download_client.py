"""Tests for download client functionality."""

from unittest.mock import MagicMock, patch

import httpx
import pytest

from helpers.download.client import download_file


async def mock_aiter_bytes_generator(data_chunks):
    """Helper async generator for mocking aiter_bytes."""
    for chunk in data_chunks:
        yield chunk


@pytest.fixture
def mock_httpx_client():
    """Create a mock httpx client."""
    with patch("httpx.AsyncClient") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


@pytest.fixture
def mock_response():
    """Create a mock response with streaming capability."""
    response = MagicMock()
    response.headers = {"content-length": "1024"}
    response.raise_for_status = MagicMock()
    return response


@pytest.fixture
def mock_tempfile():
    """Mock tempfile operations to avoid actual I/O."""
    with patch("tempfile.NamedTemporaryFile") as mock:
        mock_file = MagicMock()
        mock_file.name = "/tmp/test_file.jpg"
        mock_file.__enter__.return_value = mock_file
        mock.return_value = mock_file
        yield mock_file


class AsyncContextManagerMock:
    """Mock class that implements the async context manager protocol."""

    def __init__(self, response):
        self.response = response

    async def __aenter__(self):
        return self.response

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.mark.asyncio
async def test_successful_download(mock_httpx_client, mock_response, mock_tempfile):
    """Test successful file download."""
    # Mock the streaming response
    mock_response.aiter_bytes = MagicMock(
        return_value=mock_aiter_bytes_generator([b"x" * 1024])
    )
    mock_httpx_client.stream = MagicMock(
        return_value=AsyncContextManagerMock(mock_response)
    )

    # Download the file
    await download_file(
        mock_httpx_client,
        "http://example.com/file.jpg",
        "jpg",
        retry_delay=0.1,
        chunk_size=1024,
    )

    # Verify the file path and write operations
    assert mock_tempfile.write.call_count > 0
    mock_httpx_client.stream.assert_called_once_with(
        "GET", "http://example.com/file.jpg"
    )


@pytest.mark.asyncio
async def test_download_retry_on_http_error(
    mock_httpx_client, mock_response, mock_tempfile
):
    """Test that download retries on HTTP errors."""
    # Mock the streaming response to fail twice then succeed
    mock_response.aiter_bytes = MagicMock(
        return_value=mock_aiter_bytes_generator([b"x" * 1024])
    )
    mock_httpx_client.stream = MagicMock(
        side_effect=[
            httpx.HTTPError("Connection error"),
            httpx.HTTPError("Timeout"),
            AsyncContextManagerMock(mock_response),
        ]
    )

    # Download the file
    await download_file(
        mock_httpx_client,
        "http://example.com/file.jpg",
        "jpg",
        max_retries=3,
        retry_delay=0.1,
        chunk_size=1024,
    )

    # Verify the file path and write operations
    assert mock_tempfile.write.call_count > 0
    assert mock_httpx_client.stream.call_count == 3


@pytest.mark.asyncio
async def test_download_retry_on_incomplete_file(
    mock_httpx_client, mock_response, mock_tempfile
):
    """Test that download retries when file is incomplete."""
    # Each call to aiter_bytes should return a new generator
    mock_response.aiter_bytes = MagicMock(
        side_effect=[
            mock_aiter_bytes_generator([b"x" * 512]),  # Incomplete
            mock_aiter_bytes_generator([b"x" * 512]),  # Incomplete
            mock_aiter_bytes_generator([b"x" * 1024]),  # Complete
        ]
    )
    mock_httpx_client.stream = MagicMock(
        return_value=AsyncContextManagerMock(mock_response)
    )

    # Download the file
    await download_file(
        mock_httpx_client,
        "http://example.com/file.jpg",
        "jpg",
        max_retries=3,
        retry_delay=0.1,
        chunk_size=1024,
    )

    # Verify the file path and write operations
    assert (
        mock_tempfile.write.call_count > 0
    )  # Should be called for each chunk across retries
    assert mock_httpx_client.stream.call_count == 3
    assert (
        mock_response.aiter_bytes.call_count == 3
    )  # aiter_bytes is called once per successful stream attempt


@pytest.mark.asyncio
async def test_download_fails_after_max_retries(mock_httpx_client):
    """Test that download fails after maximum retries."""
    # Mock the streaming response to always fail
    mock_httpx_client.stream = MagicMock(
        side_effect=httpx.HTTPError("Connection error")
    )

    # Attempt to download the file
    with pytest.raises(
        Exception, match="Failed to download file after maximum retries"
    ):
        await download_file(
            mock_httpx_client,
            "http://example.com/file.jpg",
            "jpg",
            max_retries=2,
            retry_delay=0.1,
            chunk_size=1024,
        )

    assert mock_httpx_client.stream.call_count == 2  # Initial + 1 retry


@pytest.mark.asyncio
async def test_download_fails_on_too_small_file(mock_httpx_client, mock_response):
    """Test that download fails when file is too small."""
    # Mock the streaming response to return a small file that matches content-length
    mock_response.headers = {"content-length": "100"}
    mock_response.aiter_bytes = MagicMock(
        side_effect=lambda chunk_size=None: mock_aiter_bytes_generator([b"x" * 100])  # noqa: ARG005
    )
    mock_httpx_client.stream = MagicMock(
        return_value=AsyncContextManagerMock(mock_response)
    )

    # Attempt to download the file
    with pytest.raises(
        Exception, match="Failed to download file after maximum retries"
    ):
        await download_file(
            mock_httpx_client,
            "http://example.com/file.jpg",
            "jpg",
            retry_delay=0.1,
            chunk_size=1024,  # chunk_size > downloaded size to get one chunk
        )

    # download_file retries on any exception, so stream will be called 1 (initial) + 3 (default MAX_RETRIES - 1) = 4 times
    assert mock_httpx_client.stream.call_count == 4
    # aiter_bytes is called in each attempt due to the lambda side_effect providing a fresh iterator
    assert mock_response.aiter_bytes.call_count == 4


@pytest.mark.asyncio
async def test_download_without_content_length(
    mock_httpx_client, mock_response, mock_tempfile
):
    """Test download when content-length header is not present."""
    # Mock response without content-length header
    mock_response.headers = {}
    mock_response.aiter_bytes = MagicMock(
        return_value=mock_aiter_bytes_generator([b"x" * 1024])
    )
    mock_httpx_client.stream = MagicMock(
        return_value=AsyncContextManagerMock(mock_response)
    )

    # Download the file
    await download_file(
        mock_httpx_client,
        "http://example.com/file.jpg",
        "jpg",
        retry_delay=0.1,
        chunk_size=1024,
    )

    # Verify the file path and write operations
    assert mock_tempfile.write.call_count > 0
    assert mock_httpx_client.stream.call_count == 1

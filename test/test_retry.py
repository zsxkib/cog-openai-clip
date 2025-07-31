"""Tests for retry functionality."""

from unittest.mock import patch

import pytest

from helpers.utils.retry import (
    RetryType,
    retry,
    retry_with_capped_exponential_backoff,
    retry_with_exponential_backoff,
    retry_with_uniform_backoff,
)


@pytest.mark.asyncio
async def test_exponential_backoff():
    """Test exponential backoff retry."""
    with patch("asyncio.sleep") as mock_sleep:
        result = await retry_with_exponential_backoff(
            retry_count=0,
            max_retries=3,
            base_delay=1,
        )
        assert result == (True, 1)
        mock_sleep.assert_called_once()
        # First retry should be base_delay * 2^0 * jitter
        delay = mock_sleep.call_args[0][0]
        assert 0.8 <= delay <= 1.2  # jitter range


@pytest.mark.asyncio
async def test_uniform_backoff():
    """Test uniform backoff retry."""
    with patch("asyncio.sleep") as mock_sleep:
        result = await retry_with_uniform_backoff(
            retry_count=0,
            max_retries=3,
            base_delay=1,
        )
        assert result == (True, 1)
        mock_sleep.assert_called_once()
        # Uniform delay should be base_delay * jitter
        delay = mock_sleep.call_args[0][0]
        assert 0.8 <= delay <= 1.2  # jitter range


@pytest.mark.asyncio
async def test_capped_exponential_backoff():
    """Test capped exponential backoff retry."""
    with patch("asyncio.sleep") as mock_sleep:
        result = await retry_with_capped_exponential_backoff(
            retry_count=0,
            max_retries=3,
            base_delay=1,
            max_delay=2,
        )
        assert result == (True, 1)
        mock_sleep.assert_called_once()
        # First retry should be min(base_delay * 2^0 * jitter, max_delay)
        delay = mock_sleep.call_args[0][0]
        assert 0.8 <= delay <= 2.0  # jitter range capped by max_delay


@pytest.mark.asyncio
async def test_max_retries_exceeded():
    """Test that max retries exception is raised."""
    with pytest.raises(Exception, match="Queue is full"):
        await retry(
            retry_count=3,
            max_retries=3,
            base_delay=1,
        )


@pytest.mark.asyncio
async def test_capped_exponential_missing_max_delay():
    """Test that max_delay is required for capped exponential."""
    with pytest.raises(ValueError, match="max_delay is required"):
        await retry(
            retry_count=0,
            max_retries=3,
            base_delay=1,
            retry_type=RetryType.CAPPED_EXPONENTIAL,
        )


@pytest.mark.asyncio
async def test_custom_error_type_and_message():
    """Test custom error type and message."""
    with patch("asyncio.sleep") as mock_sleep:
        result = await retry(
            retry_count=0,
            max_retries=3,
            base_delay=1,
            error_type="Rate limit",
            error_message="Too many requests",
        )
        assert result == (True, 1)
        mock_sleep.assert_called_once()


@pytest.mark.asyncio
async def test_exponential_backoff_increases():
    """Test that exponential backoff increases with each retry."""
    with patch("asyncio.sleep") as mock_sleep:
        # First retry
        result1 = await retry(
            retry_count=0,
            max_retries=3,
            base_delay=1,
        )
        delay1 = mock_sleep.call_args[0][0]
        mock_sleep.reset_mock()

        # Second retry
        result2 = await retry(
            retry_count=1,
            max_retries=3,
            base_delay=1,
        )
        delay2 = mock_sleep.call_args[0][0]

        assert result1 == (True, 1)
        assert result2 == (True, 2)
        # Second delay should be roughly double the first (accounting for jitter)
        assert delay2 > delay1 * 1.5  # Allow for jitter variation


@pytest.mark.asyncio
async def test_capped_exponential_respects_max():
    """Test that capped exponential respects max delay."""
    with patch("asyncio.sleep") as mock_sleep:
        # Use a large retry count to ensure exponential would exceed max
        result = await retry(
            retry_count=10,
            max_retries=20,
            base_delay=1,
            retry_type=RetryType.CAPPED_EXPONENTIAL,
            max_delay=5,
        )
        assert result == (True, 11)
        delay = mock_sleep.call_args[0][0]
        assert delay <= 5.0  # Should be capped at max_delay

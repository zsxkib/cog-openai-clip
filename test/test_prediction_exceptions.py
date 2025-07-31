"""Tests for prediction exception handling."""

import sys

import pytest

from helpers.exceptions.prediction import (
    ErrorCode,
    ModelError,
    check_for_prediction_error,
    disable_exception_traceback,
)


def test_disable_exception_traceback():
    """Test that the context manager properly disables and restores traceback limits."""
    original_limit = getattr(sys, "tracebacklimit", 1000)

    with disable_exception_traceback():
        assert sys.tracebacklimit == 0

    assert sys.tracebacklimit == original_limit


def test_model_error_messages():
    """Test that ModelError provides correct user messages for different error codes."""
    # Test rate limit error
    error = ModelError(ErrorCode.RATE_LIMIT_EXCEEDED)
    assert "high demand" in str(error)
    assert ErrorCode.RATE_LIMIT_EXCEEDED.value in str(error)

    # Test service unavailable error
    error = ModelError(ErrorCode.SERVICE_UNAVAILABLE)
    assert "temporarily unavailable" in str(error)
    assert ErrorCode.SERVICE_UNAVAILABLE.value in str(error)

    # Test moderation content error
    error = ModelError(ErrorCode.MODERATION_CONTENT)
    assert "flagged as sensitive" in str(error)
    assert ErrorCode.MODERATION_CONTENT.value in str(error)

    # Test default message for other error codes
    error = ModelError(ErrorCode.INSUFFICIENT_CREDITS)
    assert "An error occurred" in str(error)
    assert ErrorCode.INSUFFICIENT_CREDITS.value in str(error)


@pytest.mark.parametrize(
    ("error_message", "expected_code"),
    [
        ("Insufficient credits in your account", ErrorCode.INSUFFICIENT_CREDITS),
        ("Your subscription plan has expired", ErrorCode.INSUFFICIENT_CREDITS),
        ("Invalid API key provided", ErrorCode.INVALID_API_KEY),
        ("Authentication failed: invalid key", ErrorCode.INVALID_API_KEY),
        ("Rate limit exceeded, please try again later", ErrorCode.RATE_LIMIT_EXCEEDED),
        ("429 Too Many Requests", ErrorCode.RATE_LIMIT_EXCEEDED),
        ("503 Service Unavailable", ErrorCode.SERVICE_UNAVAILABLE),
        ("Internal server error occurred", ErrorCode.SERVICE_UNAVAILABLE),
        ("Content flagged as NSFW", ErrorCode.MODERATION_CONTENT),
        ("Output blocked by risk control", ErrorCode.MODERATION_CONTENT),
    ],
)
def test_check_for_prediction_error_patterns(error_message, expected_code):
    """Test that check_for_prediction_error correctly identifies error patterns."""
    with pytest.raises(ModelError) as exc_info:
        check_for_prediction_error(Exception(error_message))

    assert exc_info.value.error_code == expected_code


def test_check_for_prediction_error_unknown():
    """Test that unknown errors are re-raised as is."""
    unknown_error = Exception("Some unknown error")
    with pytest.raises(Exception) as exc_info:
        check_for_prediction_error(unknown_error)

    assert str(exc_info.value) == "Some unknown error"
    assert isinstance(exc_info.value, Exception)
    assert not isinstance(exc_info.value, ModelError)

import pytest

from helpers.utils.validation import validate_url


def test_validate_url_valid():
    """Test that valid URLs pass validation."""
    # Test HTTP URLs
    validate_url("test", "http://example.com")
    validate_url("test", "http://example.com/path")
    validate_url("test", "http://example.com:8080/path")

    # Test HTTPS URLs
    validate_url("test", "https://example.com")
    validate_url("test", "https://example.com/path")
    validate_url("test", "https://example.com:8080/path")


def test_validate_url_invalid():
    """Test that invalid URLs raise ValueError."""
    # Missing protocol
    with pytest.raises(ValueError, match="must start with http:// or https://"):
        validate_url("test", "example.com")

    # Invalid format
    with pytest.raises(ValueError, match="Invalid URL format"):
        validate_url("test", "http://")

    # Invalid characters in host
    with pytest.raises(
        ValueError, match="Invalid non-printable ASCII character in URL"
    ):
        validate_url("test", "http://example.com/\x00")


def test_validate_url_empty():
    """Test that empty or None URLs are allowed."""
    validate_url("test", None)
    validate_url("test", "")

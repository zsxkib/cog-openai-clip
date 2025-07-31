"""Tests for billing metrics functionality."""

from unittest.mock import MagicMock, patch

import pytest

from helpers.billing.metrics import (
    ALL_METRICS,
    BOOL_METRICS,
    FLOAT_METRICS,
    INTEGER_METRICS,
    STRING_METRICS,
    record_billing_metric,
)


@pytest.fixture
def mock_scope():
    """Create a mock scope for testing."""
    with patch("helpers.billing.metrics.current_scope") as mock:
        scope = MagicMock()
        mock.return_value = scope
        yield scope


def test_record_integer_metric(mock_scope):
    """Test recording a valid integer metric."""
    record_billing_metric("token_input_count", 100)
    # Verify the metric is recorded
    mock_scope.record_metric.assert_called_once_with("token_input_count", 100)


def test_record_float_metric(mock_scope):
    """Test recording a valid float metric."""
    record_billing_metric("audio_output_duration_seconds", 1.5)
    mock_scope.record_metric.assert_called_once_with(
        "audio_output_duration_seconds", 1.5
    )


def test_invalid_metric_name():
    """Test that invalid metric names raise ValueError."""
    with pytest.raises(ValueError, match="Invalid metric name"):
        record_billing_metric("invalid_metric", 100)


def test_negative_value():
    """Test that negative values raise ValueError."""
    with pytest.raises(ValueError, match="Metric value must be non-negative"):
        record_billing_metric("token_input_count", -1)


def test_float_for_integer_metric():
    """Test that float values for integer metrics raise ValueError."""
    with pytest.raises(ValueError, match="requires an integer value"):
        record_billing_metric("token_input_count", 1.5)


def test_token_output_count_metric(mock_scope):
    """Test that token output count is recorded correctly."""
    record_billing_metric("token_output_count", 100)
    mock_scope.record_metric.assert_called_once_with("token_output_count", 100)


def test_metric_sets():
    """Test that metric sets are properly defined and don't overlap."""
    # Verify no overlap between different metric types
    assert not (INTEGER_METRICS & FLOAT_METRICS)
    assert not (INTEGER_METRICS & STRING_METRICS)
    assert not (INTEGER_METRICS & BOOL_METRICS)
    assert not (FLOAT_METRICS & STRING_METRICS)
    assert not (FLOAT_METRICS & BOOL_METRICS)
    assert not (STRING_METRICS & BOOL_METRICS)
    # Verify all metrics are included in ALL_METRICS
    assert (
        ALL_METRICS == INTEGER_METRICS | FLOAT_METRICS | STRING_METRICS | BOOL_METRICS
    )

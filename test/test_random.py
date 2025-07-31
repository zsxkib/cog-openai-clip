"""Tests for random utility functions."""

from helpers.utils.random import seed_or_random_seed


def test_valid_seed():
    """Test that a valid seed is returned unchanged."""
    test_seed = 12345
    result = seed_or_random_seed(test_seed)
    assert result == test_seed


def test_none_seed():
    """Test that a random seed is generated when None is provided."""
    result = seed_or_random_seed(None)
    assert isinstance(result, int)
    assert result > 0
    assert result <= 0x7FFFFFFF  # Max seed value


def test_negative_seed():
    """Test that a random seed is generated when a negative seed is provided."""
    result = seed_or_random_seed(-12345)
    assert isinstance(result, int)
    assert result > 0
    assert result <= 0x7FFFFFFF


def test_zero_seed():
    """Test that a random seed is generated when zero is provided."""
    result = seed_or_random_seed(0)
    assert isinstance(result, int)
    assert result > 0
    assert result <= 0x7FFFFFFF


def test_random_seed_uniqueness():
    """Test that different random seeds are generated for multiple calls."""
    seed1 = seed_or_random_seed(None)
    seed2 = seed_or_random_seed(None)
    assert seed1 != seed2

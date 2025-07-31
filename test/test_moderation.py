"""Tests for content moderation functionality."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from helpers.moderation.client import (
    ContentModerationError,
    OpenAIModerationClient,
)


def create_mock_categories(**kwargs):
    """Create mock categories with default False values, overridden by kwargs."""
    default_categories = {
        "harassment": False,
        "harassment_threatening": False,
        "hate": False,
        "hate_threatening": False,
        "self_harm": False,
        "self_harm_intent": False,
        "self_harm_instructions": False,
        "sexual": False,
        "sexual_minors": False,
        "violence": False,
        "violence_graphic": False,
    }
    default_categories.update(kwargs)
    return MagicMock(**default_categories)


def create_mock_result(categories):
    """Create a mock result with the given categories."""
    return MagicMock(
        categories=categories,
        model_dump=lambda: {
            "id": "mod-123",
            "categories": {
                k.replace("_", "/"): v for k, v in categories.__dict__.items()
            },
        },
    )


@pytest.fixture
def mock_openai_client():
    """Create a mock OpenAI client."""
    with patch("helpers.moderation.client.AsyncOpenAI") as mock:
        client = MagicMock()
        mock.return_value = client
        yield client


@pytest.fixture
def moderation_client(mock_openai_client):  # noqa: ARG001
    """Create a moderation client with mocked OpenAI client.

    The mock_openai_client argument is required for patching AsyncOpenAI,
    even though it's not directly used in the function body.
    """
    return OpenAIModerationClient()


@pytest.fixture
def mock_moderation_response():
    """Create a mock moderation response with all categories set to False."""
    categories = create_mock_categories()
    return MagicMock(results=[create_mock_result(categories)])


@pytest.mark.asyncio
async def test_check_content_text(
    moderation_client, mock_openai_client, mock_moderation_response
):
    """Test checking text content."""
    mock_openai_client.moderations.create = AsyncMock(
        return_value=mock_moderation_response
    )

    result = await moderation_client.check_content(texts=["test text"])

    assert result["id"] == "mod-123"
    mock_openai_client.moderations.create.assert_called_once()


@pytest.mark.asyncio
async def test_check_content_image_url(
    moderation_client, mock_openai_client, mock_moderation_response
):
    """Test checking image URL content."""
    mock_openai_client.moderations.create = AsyncMock(
        return_value=mock_moderation_response
    )

    result = await moderation_client.check_content(
        image_url="http://example.com/image.jpg"
    )

    assert result["id"] == "mod-123"
    mock_openai_client.moderations.create.assert_called_once()


@pytest.mark.asyncio
async def test_check_content_image_path(
    moderation_client, mock_openai_client, mock_moderation_response
):
    """Test checking local image content."""
    mock_openai_client.moderations.create = AsyncMock(
        return_value=mock_moderation_response
    )

    with patch("helpers.moderation.client.optimized_base64") as mock_base64:
        mock_base64.return_value = "base64_image_data"
        result = await moderation_client.check_content(image_path=Path("test.jpg"))

    assert result["id"] == "mod-123"
    mock_openai_client.moderations.create.assert_called_once()


@pytest.mark.asyncio
async def test_raise_if_flagged_clean_content(
    moderation_client, mock_openai_client, mock_moderation_response
):
    """Test that clean content doesn't raise an error."""
    mock_openai_client.moderations.create = AsyncMock(
        return_value=mock_moderation_response
    )

    await moderation_client.raise_if_flagged(
        types=["harassment", "hate"], texts=["clean text"]
    )


@pytest.mark.asyncio
async def test_raise_if_flagged_harassment(moderation_client, mock_openai_client):
    """Test that harassment content raises an error."""
    categories = create_mock_categories(harassment=True)
    response = MagicMock(results=[create_mock_result(categories)])
    mock_openai_client.moderations.create = AsyncMock(return_value=response)

    with pytest.raises(ContentModerationError, match="Content flagged for: harassment"):
        await moderation_client.raise_if_flagged(
            types=["harassment"], texts=["harassing text"]
        )


@pytest.mark.asyncio
async def test_raise_if_flagged_sexual_minors(moderation_client, mock_openai_client):
    """Test that sexual/minors content raises a specific error."""
    categories = create_mock_categories(sexual_minors=True)
    response = MagicMock(results=[create_mock_result(categories)])
    mock_openai_client.moderations.create = AsyncMock(return_value=response)

    with pytest.raises(
        ContentModerationError,
        match="Content flagged and reported for containing illegal material",
    ):
        await moderation_client.raise_if_flagged(
            types=["sexual/minors"], texts=["inappropriate text"]
        )


@pytest.mark.asyncio
async def test_raise_if_flagged_sexual_with_other(
    moderation_client, mock_openai_client
):
    """Test that sexual content with other violations raises an error."""
    categories = create_mock_categories(harassment=True, sexual=True)
    response = MagicMock(results=[create_mock_result(categories)])
    mock_openai_client.moderations.create = AsyncMock(return_value=response)

    with pytest.raises(
        ContentModerationError,
        match="Content flagged for containing sexual content with other violations",
    ):
        await moderation_client.raise_if_flagged(
            types=["sexual", "harassment"], texts=["inappropriate text"]
        )


@pytest.mark.asyncio
async def test_raise_if_flagged_timeout(moderation_client, mock_openai_client):
    """Test handling of timeout errors."""
    mock_openai_client.moderations.create = AsyncMock(
        side_effect=asyncio.TimeoutError()
    )

    await moderation_client.raise_if_flagged(types=["harassment"], texts=["test text"])
    # Should not raise an error, just print a warning


@pytest.mark.asyncio
async def test_raise_if_flagged_invalid_type(
    moderation_client, mock_openai_client, mock_moderation_response
):
    """Test handling of invalid moderation types."""
    mock_openai_client.moderations.create = AsyncMock(
        return_value=mock_moderation_response
    )

    # Should not raise an error, just print a warning
    await moderation_client.raise_if_flagged(
        types=["invalid_type"], texts=["test text"]
    )

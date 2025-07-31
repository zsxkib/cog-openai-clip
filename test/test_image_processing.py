"""Tests for image processing functionality."""

import base64
import tempfile
from unittest.mock import patch

import pytest
import requests
from PIL import Image

from helpers.images.processing import (
    clear_image_metadata,
    convert_to_supported_jpeg_mode,
    optimized_base64,
    optimized_file,
    resize_image,
    save_to_base64,
    save_to_file,
    validate_image_aspect_ratio,
)


@pytest.fixture
def test_image_rgb():
    return Image.new("RGB", (800, 600), color="red")


@pytest.fixture
def test_image_rgba():
    return Image.new("RGBA", (800, 600), color=(255, 0, 0, 128))


@pytest.fixture
def test_image_path(tmp_path, test_image_rgb):
    """Create a temporary test image file."""
    img_path = tmp_path / "test.jpg"
    test_image_rgb.save(img_path, format="JPEG")
    return img_path


@pytest.fixture
def wide_image_path(tmp_path):
    """Create a temporary wide test image file."""
    img = Image.new(
        "RGB", (1000, 380), color="blue"
    )  # 2.63:1 ratio, outside max of 2.5:1
    img_path = tmp_path / "wide.jpg"
    img.save(img_path, format="JPEG")
    return img_path


@pytest.fixture
def tall_image_path(tmp_path):
    """Create a temporary tall test image file."""
    img = Image.new(
        "RGB", (380, 1000), color="green"
    )  # 1:2.63 ratio, outside min of 1:2.5
    img_path = tmp_path / "tall.jpg"
    img.save(img_path, format="JPEG")
    return img_path


def test_validate_image_aspect_ratio_valid(test_image_path):
    """Test aspect ratio validation with valid ratios."""
    # Test with default ratios (1:2.5 to 2.5:1)
    validate_image_aspect_ratio(test_image_path)  # 800:600 = 1.33:1, should pass

    # Test with custom ratios
    validate_image_aspect_ratio(
        test_image_path, min_aspect_ratio=1.0, max_aspect_ratio=2.0
    )
    validate_image_aspect_ratio(
        test_image_path, min_aspect_ratio=1.2, max_aspect_ratio=1.5
    )


def test_validate_image_aspect_ratio_invalid(wide_image_path, tall_image_path):
    """Test aspect ratio validation with invalid ratios."""
    # Test too wide image (2.63:1)
    with pytest.raises(
        ValueError,
        match="Error processing image .* for aspect ratio validation: Image aspect ratio \\(2.63\\) is outside the allowed range \\[0.40 \\(1:2.5\\), 2.50 \\(2.5:1\\)\\]",
    ):
        validate_image_aspect_ratio(wide_image_path)

    # Test too tall image (1:2.63)
    with pytest.raises(
        ValueError,
        match="Error processing image .* for aspect ratio validation: Image aspect ratio \\(0.38\\) is outside the allowed range \\[0.40 \\(1:2.5\\), 2.50 \\(2.5:1\\)\\]",
    ):
        validate_image_aspect_ratio(tall_image_path)

    # Test with custom error message
    custom_message = "Please use a more square image"
    with pytest.raises(
        ValueError,
        match=f"Error processing image .* for aspect ratio validation: Image aspect ratio \\(2.63\\) is outside the allowed range \\[0.40 \\(1:2.5\\), 2.50 \\(2.5:1\\)\\]. {custom_message}",
    ):
        validate_image_aspect_ratio(
            wide_image_path, aspect_ratio_error_message=custom_message
        )


def test_validate_image_aspect_ratio_errors(tmp_path):
    """Test aspect ratio validation error cases."""
    # Test non-existent file
    non_existent = tmp_path / "nonexistent.jpg"
    with pytest.raises(
        ValueError,
        match=r"Error processing image .* for aspect ratio validation: .*No such file or directory",
    ):
        validate_image_aspect_ratio(non_existent)

    # Test invalid image file
    invalid_file = tmp_path / "invalid.jpg"
    invalid_file.write_text("not an image")
    with pytest.raises(ValueError, match="Error processing image"):
        validate_image_aspect_ratio(invalid_file)


def test_resize_image(test_image_path):
    """Test image resizing while maintaining aspect ratio."""
    # Test with max_dim larger than image
    img = resize_image(test_image_path, max_dim=2000)
    assert img.size == (800, 600)  # Should not resize

    # Test with max_dim smaller than image
    img = resize_image(test_image_path, max_dim=400)
    assert max(img.size) == 400
    assert img.size[0] / img.size[1] == pytest.approx(
        800 / 600
    )  # Aspect ratio preserved


def test_resize_image_with_min_dim(tmp_path):
    """Test image resizing with min_dim parameter."""
    # Create a small test image (200x200)
    small_img = Image.new("RGB", (200, 200), color="red")
    small_img_path = tmp_path / "small.jpg"
    small_img.save(small_img_path, format="JPEG")

    # Test scaling up with default min_dim (300)
    img = resize_image(small_img_path, max_dim=1024, min_dim=300)
    assert min(img.size) == 300  # Should scale up to meet minimum
    assert img.size[0] / img.size[1] == pytest.approx(1.0)  # Aspect ratio preserved

    # Test scaling up with custom min_dim
    img = resize_image(small_img_path, max_dim=1024, min_dim=400)
    assert min(img.size) == 400  # Should scale up to meet minimum
    assert img.size[0] / img.size[1] == pytest.approx(1.0)  # Aspect ratio preserved

    # Test with image already meeting min_dim (should not scale up)
    medium_img = Image.new("RGB", (400, 400), color="blue")
    medium_img_path = tmp_path / "medium.jpg"
    medium_img.save(medium_img_path, format="JPEG")

    img = resize_image(medium_img_path, max_dim=1024, min_dim=300)
    assert img.size == (400, 400)  # Should not resize

    # Test with rectangular image that needs scaling up
    wide_small_img = Image.new("RGB", (150, 100), color="green")
    wide_small_img_path = tmp_path / "wide_small.jpg"
    wide_small_img.save(wide_small_img_path, format="JPEG")

    img = resize_image(wide_small_img_path, max_dim=1024, min_dim=300)
    assert min(img.size) == 300  # Should scale up to meet minimum
    assert img.size[0] / img.size[1] == pytest.approx(1.5)  # Aspect ratio preserved


def test_resize_image_priority(tmp_path):
    """Test that max_dim takes priority over min_dim when both constraints apply."""
    # Create a very large image (2000x1500)
    large_img = Image.new("RGB", (2000, 1500), color="purple")
    large_img_path = tmp_path / "large.jpg"
    large_img.save(large_img_path, format="JPEG")

    # Test with both max_dim and min_dim constraints
    # max_dim=800 should take priority over min_dim=1000
    img = resize_image(large_img_path, max_dim=800, min_dim=1000)
    assert max(img.size) == 800  # Should respect max_dim
    assert img.size[0] / img.size[1] == pytest.approx(
        2000 / 1500
    )  # Aspect ratio preserved

    # The resulting image will be 800x600, which is less than min_dim=1000
    # This is correct behavior as max_dim takes priority


def test_resize_image_no_resize_needed(tmp_path):
    """Test that images within both min_dim and max_dim bounds are not resized."""
    # Create an image that's already within bounds (500x400)
    medium_img = Image.new("RGB", (500, 400), color="orange")
    medium_img_path = tmp_path / "medium.jpg"
    medium_img.save(medium_img_path, format="JPEG")

    # Test with bounds that include the current size
    img = resize_image(medium_img_path, max_dim=1024, min_dim=300)
    assert img.size == (500, 400)  # Should not resize

    # Test with tighter bounds that still include the current size
    img = resize_image(medium_img_path, max_dim=600, min_dim=350)
    assert img.size == (500, 400)  # Should not resize


def test_convert_to_supported_jpeg_mode(test_image_rgba):
    """Test conversion of unsupported JPEG modes."""
    # Test RGBA conversion
    converted = convert_to_supported_jpeg_mode(test_image_rgba)
    assert converted.mode == "RGB"

    # Test RGB image (should not convert)
    rgb_img = Image.new("RGB", (100, 100), color="blue")
    converted = convert_to_supported_jpeg_mode(rgb_img)
    assert converted.mode == "RGB"


def test_clear_image_metadata(test_image_rgb):
    """Test clearing image metadata."""
    test_image_rgb.info["test"] = "metadata"
    cleared = clear_image_metadata(test_image_rgb)
    assert not cleared.info


def test_clear_image_metadata_png_preserves_metadata(tmp_path):
    """Test that clear_image_metadata does not remove metadata from PNG images."""
    img = Image.new("RGB", (10, 10), color="blue")
    png_path = tmp_path / "meta.png"
    # Save with custom metadata
    from PIL import PngImagePlugin

    meta = PngImagePlugin.PngInfo()
    meta.add_text("custom", "value")
    img.save(png_path, format="PNG", pnginfo=meta)

    loaded = Image.open(png_path)
    assert "custom" in loaded.info
    cleared = clear_image_metadata(loaded)
    assert "custom" in cleared.info


def test_save_to_base64(test_image_path):
    """Test saving image to base64."""
    # Test with default parameters
    result = save_to_base64(test_image_path)
    assert result.startswith("data:image/jpeg;base64,")

    # Test raw base64
    raw_result = save_to_base64(test_image_path, raw=True)
    assert not raw_result.startswith("data:image")
    # Verify it's valid base64
    base64.b64decode(raw_result)

    # Test with different format
    png_result = save_to_base64(test_image_path, img_format="png")
    assert png_result.startswith("data:image/png;base64,")


def test_save_to_file(test_image_path, tmp_path):
    """Test saving image to file."""
    with patch("tempfile.NamedTemporaryFile") as mock_temp:
        mock_temp.return_value.__enter__.return_value.name = str(tmp_path / "test.jpg")
        result = save_to_file(test_image_path)
        assert result.suffix == ".jpg"
        assert result.parent == tmp_path


@pytest.mark.asyncio
async def test_optimized_base64(test_image_path):
    """Test async base64 conversion."""
    result = await optimized_base64(test_image_path)
    assert result.startswith("data:image/jpeg;base64,")


@pytest.mark.asyncio
async def test_optimized_file(test_image_path, tmp_path):
    """Test async file saving."""
    with patch("tempfile.NamedTemporaryFile") as mock_temp:
        mock_temp.return_value.__enter__.return_value.name = str(tmp_path / "test.jpg")
        result = await optimized_file(test_image_path)
        assert result.suffix == ".jpg"
        assert result.parent == tmp_path


def test_image_quality_and_size(test_image_path):
    """Test that image quality and size parameters are respected."""
    # Test with high quality
    high_quality = save_to_base64(test_image_path, quality=95)
    high_quality_size = len(base64.b64decode(high_quality.split(",")[1]))

    # Test with low quality
    low_quality = save_to_base64(test_image_path, quality=10)
    low_quality_size = len(base64.b64decode(low_quality.split(",")[1]))

    # High quality should result in larger file size
    assert high_quality_size > low_quality_size


def test_resize_image_heic():
    """Test resizing a HEIC image from URL."""
    url = "https://replicate.delivery/pbxt/NM8ePjxU47T9cqMHFQ7xAE0Yih72FiXShCo5G8dJzq9CLLYT/IMG_1845.HEIC"
    with tempfile.NamedTemporaryFile(suffix=".HEIC") as tmp:
        resp = requests.get(url)
        tmp.write(resp.content)
        tmp.flush()
        img = resize_image(tmp.name)
        assert isinstance(img, Image.Image)
        assert max(img.size) == 1024  # Default max_dim

        # Test base64 conversion
        b64 = save_to_base64(tmp.name)
        assert b64.startswith("data:image/jpeg;base64,")


def test_resize_image_avif():
    """Test resizing an AVIF image from URL."""
    url = "https://replicate.delivery/pbxt/NM8fUF5vdinBEBweWzwD5d94VqclSsOuYSsuqjWOiLJnLlHH/titanic.avif"
    with tempfile.NamedTemporaryFile(suffix=".avif") as tmp:
        resp = requests.get(url)
        tmp.write(resp.content)
        tmp.flush()
        img = resize_image(tmp.name)
        assert isinstance(img, Image.Image)
        assert max(img.size) == 1024  # Default max_dim

        # Test base64 conversion
        b64 = save_to_base64(tmp.name)
        assert b64.startswith("data:image/jpeg;base64,")

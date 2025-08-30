import pytest
from PIL import Image, ImageDraw

# This file makes fixtures available to all test files in this directory.

@pytest.fixture
def test_image() -> Image.Image:
    """Creates a simple 32x32 RGBA image for testing."""
    # Use a color that is not at the maximum value for brightness tests
    return Image.new('RGBA', (32, 32), (200, 50, 50, 255))

@pytest.fixture
def image_with_border() -> Image.Image:
    """Creates an image with a transparent border for cropping tests."""
    img = Image.new('RGBA', (48, 48), (0, 0, 0, 0))
    # Draw a red square in the middle
    for x in range(16, 32):
        for y in range(16, 32):
            img.putpixel((x, y), (255, 0, 0, 255))
    return img

@pytest.fixture
def circle_image() -> Image.Image:
    """Creates an image with a red circle on a blue background."""
    img = Image.new('RGBA', (100, 100), (0, 0, 255, 255))
    draw = ImageDraw.Draw(img)
    # Draw a red circle in the middle
    draw.ellipse((25, 25, 75, 75), fill=(255, 0, 0, 255))
    return img

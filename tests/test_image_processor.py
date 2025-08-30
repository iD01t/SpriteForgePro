import pytest
from PIL import Image
import numpy as np
import sys
from pathlib import Path

# Add the root directory to the Python path to allow importing the main script
sys.path.insert(0, str(Path(__file__).parent.parent))

from sprite_forge_ultimate import ImageProcessor, HAS_OPENCV

def test_pixelate(test_image):
    """Tests the pixelation effect."""
    factor = 4
    processed = ImageProcessor.pixelate(test_image, factor)
    assert processed.size == test_image.size
    # A pixelated image should have fewer unique colors than the original gradient (if one was used)
    # For a solid color, this test is simpler.
    assert processed.getpixel((0, 0)) == test_image.getpixel((0, 0))

def test_enhance(test_image):
    """Tests the image enhancement function."""
    processed = ImageProcessor.enhance(test_image, brightness=1.5, contrast=1.2)
    assert processed.size == test_image.size
    # Check if the pixel value has changed (increased due to brightness)
    original_pixel = test_image.getpixel((0, 0))
    processed_pixel = processed.getpixel((0, 0))
    assert processed_pixel[0] > original_pixel[0]

def test_apply_doom_palette(test_image):
    """Tests applying the Doom palette."""
    processed = ImageProcessor.apply_doom_palette(test_image, dither=False, preserve_transparency=True)
    assert processed.size == test_image.size
    assert processed.mode == 'RGBA'
    # Check if a pixel is mapped to a color from the palette (or close to it)
    # This is a loose check; a better one would be to check against the actual palette colors.
    processed_pixel = processed.getpixel((1, 1))
    assert processed_pixel[3] == 255 # Alpha preserved

def test_auto_crop(image_with_border):
    """Tests the auto-cropping feature."""
    assert image_with_border.size == (48, 48)
    processed = ImageProcessor.auto_crop(image_with_border)
    assert processed.size == (16, 16)

@pytest.mark.skipif(not HAS_OPENCV, reason="OpenCV is not installed, skipping background removal test.")
def test_remove_background(circle_image):
    """Tests background removal on a more complex image."""
    # We want to remove the blue background
    # GrabCut is not deterministic, so this test is still a bit fragile.
    # A better approach for perfect testability would be to mock cv2.grabCut

    processed = ImageProcessor.remove_background(circle_image)
    assert processed.size == circle_image.size

    # Check a pixel from the background (a corner) is now transparent
    background_pixel_alpha = processed.getpixel((5, 5))[3]
    assert background_pixel_alpha < 50 # Allow for some fuzzy edges, but should be mostly transparent

    # Check a pixel from the foreground (center of circle) is still opaque
    foreground_pixel_alpha = processed.getpixel((50, 50))[3]
    assert foreground_pixel_alpha > 200

def test_create_sprite_rotations(test_image):
    """Tests the sprite rotation function."""
    num_rotations = 8
    rotations = ImageProcessor.create_sprite_rotations(test_image, num_rotations=num_rotations)
    assert len(rotations) == num_rotations
    assert 'angle_0' in rotations
    assert 'angle_7' in rotations
    # Check if expanded image size is correct (for a square, it's sqrt(2) * side)
    rotated_size = rotations['angle_1'].size
    assert rotated_size[0] > test_image.size[0]

def test_emboss(test_image):
    """Tests the emboss filter."""
    processed = ImageProcessor.emboss(test_image)
    assert processed.size == test_image.size
    # Embossed images are typically grayscale and have a different look
    assert processed.mode == 'RGB'
    original_pixel = np.array(test_image.getpixel((10, 10)))
    processed_pixel = np.array(processed.getpixel((10, 10)))
    # A simple check: the pixel values should be different.
    assert not np.array_equal(original_pixel[:3], processed_pixel)

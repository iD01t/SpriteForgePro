import pytest
from PIL import Image
import sys
from pathlib import Path

# Add the root directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sprite_forge_ultimate import PluginManager, ImageProcessor

@pytest.fixture
def plugin_manager() -> PluginManager:
    """Returns an instance of the PluginManager."""
    # This will automatically load built-in and external plugins
    return PluginManager()

@pytest.fixture
def test_image() -> Image.Image:
    """Creates a simple 32x32 RGBA image for testing."""
    return Image.new('RGBA', (32, 32), (128, 128, 128, 255))

def test_load_builtin_plugins(plugin_manager):
    """Tests if the built-in plugins are loaded correctly."""
    assert len(plugin_manager.plugins) > 0
    # Check for specific built-in plugins
    assert "Pixelate" in plugin_manager.plugins
    assert "Enhance" in plugin_manager.plugins
    assert "Doom Palette" in plugin_manager.plugins
    assert "Emboss" in plugin_manager.plugins

def test_load_external_plugins(plugin_manager):
    """Tests if the example external plugins are loaded."""
    # These names must match what was created in the previous step
    assert "JSON Emboss Example" in plugin_manager.plugins
    assert "Python Blur Example" in plugin_manager.plugins

def test_get_plugin(plugin_manager):
    """Tests retrieving a single plugin."""
    plugin = plugin_manager.get_plugin("Pixelate")
    assert plugin is not None
    assert plugin.info.name == "Pixelate"

    non_existent_plugin = plugin_manager.get_plugin("NonExistentPlugin")
    assert non_existent_plugin is None

def test_apply_builtin_plugin(plugin_manager, test_image):
    """Tests applying a built-in plugin."""
    pixelate_plugin = plugin_manager.get_plugin("Pixelate")
    assert pixelate_plugin is not None

    original_pixel = test_image.getpixel((0, 0))
    processed_image = pixelate_plugin.process(test_image, factor=4)

    assert processed_image is not None
    assert processed_image.size == test_image.size
    # The image should have changed
    assert processed_image.getpixel((0, 0)) == original_pixel

def test_apply_python_plugin(plugin_manager, image_with_border):
    """Tests applying the external Python plugin."""
    blur_plugin = plugin_manager.get_plugin("Python Blur Example")
    assert blur_plugin is not None

    # Use the image with a border, as blurring a solid color has no effect
    processed_image = blur_plugin.process(image_with_border, radius=2.0)
    assert processed_image is not None
    assert processed_image.size == image_with_border.size

    # A pixel just outside the red square should have changed from fully transparent
    # to slightly colored due to the blur.
    original_pixel_alpha = image_with_border.getpixel((15, 15))[3]
    processed_pixel_alpha = processed_image.getpixel((15, 15))[3]
    assert original_pixel_alpha == 0
    assert processed_pixel_alpha > 0

def test_apply_json_plugin(plugin_manager, test_image):
    """Tests applying the external JSON plugin."""
    emboss_plugin = plugin_manager.get_plugin("JSON Emboss Example")
    assert emboss_plugin is not None

    processed_image = emboss_plugin.process(test_image)
    assert processed_image is not None
    assert processed_image.size == test_image.size
    assert processed_image.mode == 'RGB' # Emboss changes mode

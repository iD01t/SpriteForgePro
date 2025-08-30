import pytest
import sys
from pathlib import Path
from PIL import Image

# Add the root directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Must import main script first to set up dependency flags
import sprite_forge_ultimate
from sprite_forge_ultimate import GUI_AVAILABLE

# Conditionally import Qt classes
if GUI_AVAILABLE:
    from PyQt6.QtWidgets import QApplication
    from sprite_forge_ultimate import SpriteForgeUltimateWindow, ModernImageCanvas, ProjectSettings

@pytest.mark.skipif(not GUI_AVAILABLE, reason="PyQt6 is not installed, skipping GUI tests.")
def test_gui_app_instantiation():
    """
    Tests if the main application window can be created without crashing.
    This should be run with an offscreen platform plugin, e.g.:
    pytest --qt-qpa-platform offscreen
    """
    # QApplication.instance() is a safe way to get the app instance,
    # or create one if it doesn't exist. pytest-qt handles this.
    app = QApplication.instance() or QApplication(sys.argv)

    settings = ProjectSettings.load()
    window = SpriteForgeUltimateWindow(settings)

    assert window is not None
    assert window.windowTitle().startswith("Sprite Forge Ultimate")

    # Show and process events for a moment to catch any init errors
    window.show()
    app.processEvents()
    window.hide()

@pytest.mark.skipif(not GUI_AVAILABLE, reason="PyQt6 is not installed, skipping GUI tests.")
def test_canvas_instantiation_and_set_image():
    """
    Tests if the ModernImageCanvas can be created and can be given an image.
    """
    app = QApplication.instance() or QApplication(sys.argv)

    canvas = ModernImageCanvas()
    assert canvas is not None

    # Test setting a PIL image
    test_image = Image.new('RGBA', (100, 100), 'blue')
    canvas.set_image(test_image)

    assert canvas.pixmap is not None
    assert not canvas.pixmap.isNull()
    assert canvas.pixmap.width() == 100
    assert canvas.pixmap.height() == 100

    # Test clearing the image
    canvas.set_image(None)
    assert canvas.pixmap is None

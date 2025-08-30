# This is an example of a Python-based plugin for Sprite Forge Ultimate.

# In a real multi-file project, you would install the main app as a package
# and import from it. For this single-file script, we import directly.
from sprite_forge_ultimate import BasePlugin, PluginInfo
# The PIL/Pillow classes are imported at the top of the main script,
# but for clarity and standalone analysis, a plugin could also import them.
from PIL import ImageFilter

class BlurPlugin(BasePlugin):
    """An example plugin that applies a Gaussian blur."""

    def __init__(self):
        super().__init__()
        self.info = PluginInfo(
            name="Python Blur Example",
            version="1.0.1",
            author="SFU Example",
            description="Applies a Gaussian blur to the image using a Python plugin.",
            category="Examples",
            parameters={
                "radius": {
                    "type": "float",
                    "min": 0.1,
                    "max": 20.0,
                    "default": 2.0,
                    "label": "Blur Radius"
                }
            }
        )

    def process(self, image, **kwargs):
        """Applies the blur effect."""
        radius = kwargs.get('radius', 2.0)
        # We have access to ImageFilter because it's injected by the PluginManager
        return image.filter(ImageFilter.GaussianBlur(radius=radius))

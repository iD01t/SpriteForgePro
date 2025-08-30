# Sprite Forge Ultimate v3.0.0

Welcome to **Sprite Forge Ultimate**, a premium, production-ready, cross-platform desktop application for professional sprite and texture creation. It is especially powerful for developers in the Doom modding community.

This application is the result of merging the best features of `Sprite Forge Pro` and `Sprite Forge Enhanced` into a single, cohesive, and powerful tool.

![Screenshot Placeholder](https://via.placeholder.com/800x450.png?text=Sprite+Forge+Ultimate+UI)

## ✨ Features

- **Modern UI**: A sleek, professional dark theme powered by PyQt6, with dockable panels and a high-DPI aware interface.
- **Advanced Image Processing**: A powerful pipeline including:
  - AI-assisted background removal (OpenCV GrabCut).
  - High-quality Doom palette mapping with optional dithering.
  - Pixelation, enhancement (brightness, contrast, saturation, sharpness), and auto-cropping.
- **Unified Plugin System**: Extend the application's functionality with custom Python or JSON plugins. The app loads plugins from the `./plugins/` directory and a user-specific folder.
- **Headless Batch Mode**: A powerful command-line interface for automating your workflow. Process hundreds of images with a single command.
- **Comprehensive Exporting**: Native export to formats essential for game development, including `PNG`, `WAD`, `PK3`, `GIF`, `ZIP`, and combined `Sprite Sheet`.
- **Robust Project Management**:
  - Persistent project settings saved to a clean JSON file.
  - Crash-safe autosave functionality.
  - Automatic migration from older `Pro` and `Enhanced` settings formats.
- **Professional Canvas**: An advanced image canvas supporting high-zoom, smooth panning, and toggleable grids (including a pixel grid).

## 🚀 Quickstart

1.  **Install Dependencies**: Make sure you have Python 3.8+ installed. The application will attempt to install required Python packages on first run. Alternatively, you can install them manually:
    ```bash
    pip install PyQt6 Pillow numpy opencv-python
    ```

2.  **Run the Application**:
    ```bash
    python sprite_forge_ultimate.py
    ```

3.  **Open an Image**: Go to `File > Open...` or use the shortcut `Ctrl+O`.

4.  **Apply Effects**: Use the plugins in the side panel to modify your image.

5.  **Export**: Save your work in your desired format.

## ⌨️ Keyboard Shortcuts

| Shortcut | Action |
| :--- | :--- |
| `Ctrl+O` | Open Image |
| `Ctrl+S` | Save Project (Not yet implemented) |
| `Ctrl+Q` | Quit Application |
| `Spacebar` | Zoom to Fit (Not yet implemented) |
| `1` | Zoom to 100% (Not yet implemented) |
| `G` | Toggle Grid (Not yet implemented) |
| `P` | Toggle Pixel Grid (Not yet implemented) |

## 🤖 Batch Processing (CLI)

Automate your image processing tasks using the headless mode.

**Syntax:**
`python sprite_forge_ultimate.py --headless --input <path> --output <path> --apply <Operation> [params...]`

**Examples:**

- **Pixelate a single file:**
  ```bash
  python sprite_forge_ultimate.py --headless --input assets/player.png --output processed/player_pixel.png --apply Pixelate factor=8
  ```

- **Process an entire folder and apply multiple effects:**
  ```bash
  python sprite_forge_ultimate.py --headless --input assets/monsters/ --output processed/monsters/ --apply "Doom Palette" dither=true --apply Enhance brightness=1.1 contrast=1.05
  ```

## 🔌 Plugin Authoring

You can easily create your own plugins.

### JSON Plugins

For simple `ImageProcessor` operations, create a `.json` file in the `plugins/` directory.

**`plugins/example_emboss_plugin.json`**
```json
{
  "name": "Emboss",
  "version": "1.0.0",
  "description": "Applies a simple emboss effect.",
  "author": "Your Name",
  "category": "Effects",
  "operation": "emboss",
  "parameters": {}
}
```
*(Note: The "emboss" operation would need to be added to `ImageProcessor` for this specific example to work).*

### Python Plugins

For more complex logic, create a `.py` file. (Full Python plugin support to be fleshed out).

## 📜 License

This project is licensed under the MIT License.

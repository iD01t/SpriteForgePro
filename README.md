<img width="1536" height="1024" alt="banner" src="https://github.com/user-attachments/assets/b9dccaac-6850-470f-8d55-725af0de89c0" />

# Sprite Forge Ultimate

Sprite Forge Ultimate is the definitive all-in-one sprite editing, batch processing, and plugin-driven toolkit. It merges the best of **Sprite Forge Pro** and **Sprite Forge Enhanced** into one polished, production-ready application. Built with **PyQt6**, it delivers a premium workflow for artists, modders, and developers who need pixel-perfect control and modern automation.

---

## 🚀 Key Features

### Image Processing

* **Pixelate** – scalable pixel grid conversion with configurable block sizes.
* **Enhance** – brightness, contrast, saturation, and sharpness tuning.
* **Apply Doom Palette** – advanced palette mapping with optional dithering and transparency preservation.
* **Auto Crop** – intelligent trimming with threshold control.
* **Background Removal** – OpenCV GrabCut with adjustable tolerance and edge smoothing.
* **Sprite Rotations** – generate N rotation frames with pixel-perfect or smooth interpolation.

### Modern Image Canvas

* Zoom, pan, and rotate with high-DPI precision.
* **Transparency checkerboard** with customizable grid colors.
* **Grid and pixel grid overlays** toggleable on the fly.
* Optimized QImage rendering for large sprites with no performance stutter.

### Plugin System

* **Unified PluginManager** supporting:

  * Built-in plugins
  * External Python plugins
  * JSON-defined effect plugins
* Safe parameter validation, error handling, and sandboxed execution.
* Example plugins included (Emboss, Blur).

### Export & Formats

* Save as **PNG, GIF, WAD, PK3, ZIP, or sprite sheets**.
* Batch export with progress tracking and cancelation support.
* Alpha channel preserved with no corruption across formats.

### Project Management

* **ProjectSettings** with unified schema from Pro and Enhanced.
* Friendly JSON persistence and migration of older configs.
* Crash-safe autosave with adjustable intervals.
* Quick resume of last project on launch.

### CLI Batch Mode

Run headless without Qt for automation:

```bash
python sprite_forge_ultimate.py --input sprite.png --output out.png --format png --apply "Doom Palette" --dither true --rotations 8
```

Supports chaining multiple operations, full transparency preservation, and export to all formats.

### User Experience

* Dark theme with polished UI.
* Keyboard shortcuts for every major action.
* Non-blocking progress dialogs for batch tasks.
* Graceful dependency checks with actionable install prompts.
* Structured logging with rotating files + stdout streaming.

---

## 🧩 Premium Improvements Over Pro & Enhanced

* **Merged APIs** – one ImageProcessor, one ModernImageCanvas, one PluginManager, no duplication.
* **Extended Doom Palette Mapper** – dithering + transparency options not present before.
* **Unified Plugin Framework** – JSON + Python in one validated pipeline.
* **Consistent Settings** – reconciled enums and schema, with backward compatibility.
* **Safer Dependency Handling** – no silent installs during GUI sessions, user consent required.
* **Headless CLI** – runs without PyQt imports, perfect for servers and automation.
* **Full Test Suite** – pytest coverage for processing, plugins, and GUI smoke tests.
* **Build Scripts** – Windows (PyInstaller), macOS (universal2), Linux (AppImage).
* **Documentation** – README, CHANGELOG, MIGRATION\_NOTES, and examples.

---

## 📦 Installation

### Requirements

* Python 3.10+
* PyQt6
* NumPy, Pillow, OpenCV (optional), SciPy (optional)

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the app:

```bash
python sprite_forge_ultimate.py
```

---

## 🧪 Testing

Run all tests:

```bash
pytest tests/
```

---

## 📄 Documentation

* `README.md` – Quickstart and feature overview
* `MIGRATION_NOTES.md` – API changes and compatibility map
* `CHANGELOG.md` – Version history
* `plugins/` – Example plugins (JSON and Python)
* `tests/` – Unit and smoke tests
* `build_*` – Platform build scripts

---

## 🔥 Why This Is the Ultimate Version

Sprite Forge Ultimate isn’t just a merge, it’s a **refined upgrade**. Every missing feature from Enhanced is integrated, every professional feature from Pro is preserved, and new polish has been added across the stack. The result: one bulletproof app ready for indie developers and pro studios alike.



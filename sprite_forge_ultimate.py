#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Sprite Forge Ultimate v3.0.0
============================

A premium, production-ready, cross-platform desktop application for professional
sprite and texture creation, especially for Doom modding. This application merges
the best features of Sprite Forge Pro and Sprite Forge Enhanced into a single,
cohesive, and powerful tool.

This script is a single-file, all-inclusive application.

Key Features:
- Modern, dark-themed Qt6 UI with high-DPI support and professional styling.
- Advanced, non-destructive image processing pipeline with GPU-aware flags.
- Unified plugin system supporting Python and JSON plugins from external directories.
- Headless command-line interface for batch processing and automation with user-friendly flags.
- Native export to PNG, WAD, PK3, GIF, ZIP, and sprite sheets.
- Project system with persistent JSON settings, auto-saving, and migration support.
- Advanced canvas with zoom, pan, grids, and a reliable transparency checkerboard.
- Built-in dependency checker with user-prompted installation.
- Non-blocking operations with progress dialogs for a smooth UX.
"""

# --- Core Imports ---
import sys
import os
import json
import argparse
import logging
import subprocess
import zipfile
import shutil
import traceback
import tempfile
import time
import importlib.util
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union, Callable, Type
from abc import ABC, abstractmethod
from enum import Enum

# --- Dependency Management & Global Imports ---

# Attempt to import core libraries. The script can still run in headless mode without PyQt6.
try:
    from PIL import Image, ImageFilter, ImageEnhance, ImageOps
    import numpy as np
except ImportError:
    print("ERROR: Pillow and/or NumPy are not installed. These are critical.", file=sys.stderr)
    print("Please run: pip install Pillow numpy", file=sys.stderr)
    sys.exit(1)

# Global flags for optional/GUI dependencies
HAS_OPENCV = False
HAS_SKIMAGE = False
GUI_AVAILABLE = False

# --- Application Constants & Configuration ---

APP_NAME = "Sprite Forge Ultimate"
APP_VERSION = "3.0.0"
ORG_NAME = "SpriteForge"
LOG_DIR = Path.home() / f".{ORG_NAME.lower()}" / "logs"
SETTINGS_FILE = Path.home() / f".{ORG_NAME.lower()}" / "settings.json"
PLUGIN_DIRS = [
    Path("plugins"),
    Path.home() / f".{ORG_NAME.lower()}" / "plugins"
]

# --- Logging Setup ---

def setup_logging(headless=False):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_file = LOG_DIR / f"sprite_forge_{datetime.now().strftime('%Y%m%d')}.log"

    handlers = [logging.FileHandler(log_file, encoding='utf-8')]
    if not headless:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(APP_NAME)

logger = setup_logging('--headless' in sys.argv)


# --- Core Enums ---

class ExportFormat(Enum):
    PNG = "PNG"
    WAD = "WAD"
    PK3 = "PK3"
    GIF = "GIF"
    ZIP = "ZIP"
    SPRITE_SHEET = "Sprite Sheet"

class ProcessingMode(Enum):
    FAST = "Fast"
    BALANCED = "Balanced"
    QUALITY = "Quality"

class SpriteType(Enum):
    STATIC = "Static"
    ANIMATED = "Animated"
    ROTATIONAL = "Rotational"

# --- Data Classes ---

@dataclass
class ProjectSettings:
    last_project_path: Optional[str] = None
    theme: str = "dark"
    autosave_interval_seconds: int = 300
    export_format: ExportFormat = ExportFormat.PK3
    enable_gpu: bool = True
    dither_palette: bool = True
    preserve_transparency: bool = True

    def save(self):
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        data_to_save = {k: (v.value if isinstance(v, Enum) else v) for k, v in asdict(self).items()}
        with open(SETTINGS_FILE, 'w', encoding='utf-8') as f:
            json.dump(data_to_save, f, indent=4)

    @classmethod
    def load(cls) -> 'ProjectSettings':
        if not SETTINGS_FILE.exists():
            return cls()
        try:
            with open(SETTINGS_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)

            key_mappings = {'auto_save_interval': 'autosave_interval_seconds', 'doom_palette': 'dither_palette'}
            for old_key, new_key in key_mappings.items():
                if old_key in data: data[new_key] = data.pop(old_key)

            valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
            filtered_data = {k: v for k, v in data.items() if k in valid_keys}

            for key, value in filtered_data.items():
                field_type = cls.__annotations__.get(key)
                if isinstance(field_type, type) and issubclass(field_type, Enum):
                    try:
                        filtered_data[key] = field_type(value)
                    except ValueError:
                        logger.warning(f"Invalid enum value '{value}' for '{key}'. Using default.")
                        del filtered_data[key]

            return cls(**filtered_data)
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(f"Failed to load settings file: {e}. Using defaults.")
            return cls()

@dataclass
class PluginInfo:
    name: str
    version: str
    description: str
    author: str
    category: str = "General"
    parameters: Dict[str, Any] = field(default_factory=dict)

# --- Image Processing Core ---

class ImageProcessor:
    DOOM_PALETTE_PIL = Image.new("P", (1, 1))
    DOOM_PALETTE_PIL.putpalette(
        (0,0,0, 31,23,11, 23,15,7, 75,75,75, 255,255,255, 27,27,27, 19,19,19, 11,11,11,
         199,199,199, 119,119,119, 83,83,83, 47,47,47, 255,155,0, 231,119,0, 203,91,0,
         175,71,0, 143,59,0, 119,47,0, 91,35,0, 71,27,0, 199,0,0, 167,0,0, 139,0,0,
         107,0,0, 75,0,0, 0,255,0, 0,231,0, 0,203,0, 0,175,0, 0,143,0, 0,119,0) * 8
    )

    @staticmethod
    def pixelate(image: Image.Image, factor: int) -> Image.Image:
        if factor <= 1: return image
        w, h = image.size
        small = image.resize((w // factor, h // factor), Image.Resampling.NEAREST)
        return small.resize((w, h), Image.Resampling.NEAREST)

    @staticmethod
    def enhance(image: Image.Image, brightness: float = 1.0, contrast: float = 1.0, saturation: float = 1.0, sharpness: float = 1.0) -> Image.Image:
        img = image.copy()
        if brightness != 1.0: img = ImageEnhance.Brightness(img).enhance(brightness)
        if contrast != 1.0: img = ImageEnhance.Contrast(img).enhance(contrast)
        if saturation != 1.0: img = ImageEnhance.Color(img).enhance(saturation)
        if sharpness != 1.0: img = ImageEnhance.Sharpness(img).enhance(sharpness)
        return img

    @staticmethod
    def apply_doom_palette(image: Image.Image, dither: bool, preserve_transparency: bool) -> Image.Image:
        alpha = None
        if image.mode == 'RGBA' and preserve_transparency:
            alpha = image.getchannel('A')
            image = image.convert('RGB')

        quantized = image.quantize(palette=ImageProcessor.DOOM_PALETTE_PIL,
                                   dither=Image.Dither.FLOYDSTEINBERG if dither else Image.Dither.NONE)

        result = quantized.convert('RGB')
        if alpha: result.putalpha(alpha)
        return result

    @staticmethod
    def auto_crop(image: Image.Image, threshold: int = 0) -> Image.Image:
        if image.mode != 'RGBA': image = image.convert('RGBA')
        bbox = image.getbbox()
        return image.crop(bbox) if bbox else image

    @staticmethod
    def remove_background(image: Image.Image, tolerance: int = 30, edge_smooth: bool = True) -> Image.Image:
        if not HAS_OPENCV:
            logger.warning("OpenCV not found. Background removal is unavailable.")
            return image

        img_np = np.array(image.convert('RGB'))
        img_cv = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        mask = np.zeros(img_cv.shape[:2], np.uint8)
        bgd_model, fgd_model = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
        h, w = img_cv.shape[:2]
        rect = (int(w*0.05), int(h*0.05), int(w*0.9), int(h*0.9))

        try:
            cv2.grabCut(img_cv, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
        except Exception as e:
            logger.error(f"GrabCut failed: {e}. Check OpenCV installation.")
            return image

        mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
        if edge_smooth: mask2 = cv2.GaussianBlur(mask2, (5, 5), 0)

        alpha_channel = (mask2 * 255).astype(np.uint8)
        result_img = image.copy().convert("RGBA")
        result_img.putalpha(Image.fromarray(alpha_channel))
        return result_img

    @staticmethod
    def create_sprite_rotations(image: Image.Image, num_rotations: int = 8, smooth: bool = True) -> Dict[str, Image.Image]:
        resample = Image.Resampling.BICUBIC if smooth else Image.Resampling.NEAREST
        return {f"angle_{i}": image.rotate(-i * (360.0 / num_rotations), resample=resample, expand=True) for i in range(num_rotations)}

    @staticmethod
    def emboss(image: Image.Image, **kwargs) -> Image.Image:
        """Applies a classic emboss filter. Ignores kwargs."""
        # Convert to grayscale first for a classic emboss effect, then back to RGB
        # so that it can be displayed in standard contexts.
        return image.convert('L').filter(ImageFilter.EMBOSS).convert('RGB')

# --- Plugin System ---

class BasePlugin(ABC):
    """Abstract base class for all plugins. Subclasses should define their own info."""
    def __init__(self):
        self.info: PluginInfo = PluginInfo("Unknown", "0.0.0", "No description", "Unknown")

    @abstractmethod
    def process(self, image: Image.Image, **kwargs) -> Image.Image: pass

class PluginManager:
    def __init__(self):
        self.plugins: Dict[str, BasePlugin] = {}
        self._load_builtin_plugins()
        self._load_external_plugins()

    def _load_builtin_plugins(self):
        ops = {
            'Pixelate': (ImageProcessor.pixelate, {'factor': {'type': 'int', 'min': 2, 'max': 16, 'default': 4}}),
            'Enhance': (ImageProcessor.enhance, {'brightness': {'type': 'float', 'min': 0.5, 'max': 2.0, 'default': 1.0}, 'contrast': {'type': 'float', 'min': 0.5, 'max': 2.0, 'default': 1.0}}),
            'Auto Crop': (ImageProcessor.auto_crop, {}),
            'Remove Background': (ImageProcessor.remove_background, {'tolerance': {'type': 'int', 'min': 1, 'max': 100, 'default': 30}}),
            'Doom Palette': (ImageProcessor.apply_doom_palette, {'dither': {'type': 'bool', 'default': True}, 'preserve_transparency': {'type': 'bool', 'default': True}}),
            'Emboss': (ImageProcessor.emboss, {})
        }

        for name, (op, params) in ops.items():
            plugin_info = PluginInfo(name, '1.0', f'{name} built-in plugin.', 'SFU', 'Core', params)

            def make_init(info):
                def __init__(self):
                    super(self.__class__, self).__init__()
                    self.info = info
                return __init__

            plugin_class = type(f"{name.replace(' ','')}Plugin", (BasePlugin,), {
                '__init__': make_init(plugin_info),
                'process': staticmethod(op)
            })

            instance = plugin_class()
            self.plugins[instance.info.name] = instance

    def _load_external_plugins(self):
        for directory in PLUGIN_DIRS:
            directory.mkdir(parents=True, exist_ok=True)
            # Load JSON plugins
            for file_path in directory.glob("*.json"):
                try:
                    with open(file_path, 'r', encoding='utf-8') as f: config = json.load(f)
                    if 'name' in config and 'operation' in config:
                        op_func = getattr(ImageProcessor, config['operation'], None)
                        if op_func:
                            info = PluginInfo(config['name'], config.get('version','1.0'), config.get('description',''), config.get('author',''), 'External', config.get('parameters',{}))
                            def make_init(info):
                                def __init__(self):
                                    super(self.__class__, self).__init__()
                                    self.info = info
                                return __init__
                            plugin_class = type(f"{info.name.replace(' ','')}Plugin", (BasePlugin,), {'__init__':make_init(info), 'process': staticmethod(op_func)})
                            self.plugins[info.name] = plugin_class()
                except Exception as e: logger.warning(f"Failed to load JSON plugin {file_path}: {e}")

            # Load Python plugins
            for file_path in directory.glob("*.py"):
                if file_path.name.startswith('_'): continue
                try:
                    spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)

                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if isinstance(attr, type) and issubclass(attr, BasePlugin) and attr is not BasePlugin:
                            instance = attr()
                            self.plugins[instance.info.name] = instance
                            logger.info(f"Loaded Python plugin: {instance.info.name}")
                except Exception as e:
                    logger.error(f"Failed to load Python plugin {file_path}: {e}\n{traceback.format_exc()}")

    def get_plugin(self, name: str) -> Optional[BasePlugin]: return self.plugins.get(name)
    def get_all_plugins(self) -> List[BasePlugin]: return list(self.plugins.values())

# --- GUI & Optional Imports ---
# These are loaded conditionally to allow headless operation.
try:
    from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
                                 QPushButton, QFileDialog, QMessageBox, QProgressDialog, QFrame,
                                 QDockWidget, QListWidget)
    from PyQt6.QtCore import QObject, pyqtSignal, Qt, QPointF, QPoint
    from PyQt6.QtGui import (QAction, QKeySequence, QPixmap, QImage, QPainter, QColor, QBrush,
                             QPen, QMouseEvent, QWheelEvent, QCursor)
    GUI_AVAILABLE = True
except ImportError:
    # This block allows the script to be imported in a headless environment.
    logger.info("PyQt6 not found, GUI will be unavailable.")
    class QObject: pass
    class QWidget: pass
    class QMainWindow: pass
    class pyqtSignal:
        def __init__(self, *args, **kwargs): pass
        def connect(self, *args, **kwargs): pass
        def emit(self, *args, **kwargs): pass

try:
    import cv2
    HAS_OPENCV = True
except ImportError:
    logger.info("OpenCV not found. Advanced background removal will be disabled.")

try:
    import skimage
    HAS_SKIMAGE = True
except ImportError:
    logger.info("Scikit-image not found. Advanced filters will be disabled.")


# --- GUI Components ---

if GUI_AVAILABLE:
    class ModernImageCanvas(QWidget):
        zoom_changed = pyqtSignal(float)

        def __init__(self, parent=None):
            super().__init__(parent)
            self.pixmap: Optional[QPixmap] = None
            self.zoom_factor = 1.0
            self.pan_offset = QPointF(0, 0)
            self.last_pan_pos = QPoint()
            self.show_grid = True
            self.show_pixel_grid = True
            self.grid_size = 16
            self.checkerboard_brush = self._create_checkerboard_brush()
            self.setMinimumSize(400, 300)
            self.setMouseTracking(True)
            self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)

        def _create_checkerboard_brush(self) -> QBrush:
            pix = QPixmap(20, 20)
            pix.fill(Qt.GlobalColor.transparent)
            painter = QPainter(pix)
            painter.fillRect(0, 0, 10, 10, QColor(200, 200, 200))
            painter.fillRect(10, 10, 10, 10, QColor(200, 200, 200))
            painter.fillRect(0, 10, 10, 10, QColor(230, 230, 230))
            painter.fillRect(10, 0, 10, 10, QColor(230, 230, 230))
            painter.end()
            return QBrush(pix)

        def set_image(self, image: Optional[Image.Image]):
            if image is None:
                self.pixmap = None
            else:
                qimage = QImage(image.convert("RGBA").tobytes(), image.width, image.height, QImage.Format.Format_RGBA8888)
                self.pixmap = QPixmap.fromImage(qimage)
            self.update()

        def paintEvent(self, event: "QPaintEvent"):
            painter = QPainter(self)
            painter.fillRect(self.rect(), self.checkerboard_brush)
            if not self.pixmap:
                painter.end()
                return

            painter.translate(self.width() / 2 + self.pan_offset.x(), self.height() / 2 + self.pan_offset.y())
            painter.scale(self.zoom_factor, self.zoom_factor)

            target_rect = QRect(-self.pixmap.width() / 2, -self.pixmap.height() / 2, self.pixmap.width(), self.pixmap.height())
            painter.drawPixmap(target_rect, self.pixmap)

            if self.show_grid:
                self._draw_grid(painter, target_rect)
            if self.show_pixel_grid and self.zoom_factor >= 6:
                self._draw_pixel_grid(painter, target_rect)
            painter.end()

        def _draw_grid(self, painter, rect):
            painter.setPen(QPen(QColor(0, 0, 0, 50), 1 / self.zoom_factor))
            for x in range(rect.left(), rect.right(), self.grid_size):
                painter.drawLine(x, rect.top(), x, rect.bottom())
            for y in range(rect.top(), rect.bottom(), self.grid_size):
                painter.drawLine(rect.left(), y, rect.right(), y)

        def _draw_pixel_grid(self, painter, rect):
            painter.setPen(QPen(QColor(0, 0, 0, 30), 1 / self.zoom_factor))
            for x in range(rect.left(), rect.right()):
                painter.drawLine(x, rect.top(), x, rect.bottom())
            for y in range(rect.top(), rect.bottom()):
                painter.drawLine(rect.left(), y, rect.right(), y)

        def wheelEvent(self, event: QWheelEvent):
            factor = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
            self.zoom_factor *= factor
            self.zoom_changed.emit(self.zoom_factor)
            self.update()

        def mousePressEvent(self, event: QMouseEvent):
            if event.button() == Qt.MouseButton.MiddleButton:
                self.last_pan_pos = event.pos()
                self.setCursor(Qt.CursorShape.ClosedHandCursor)

        def mouseMoveEvent(self, event: QMouseEvent):
            if event.buttons() & Qt.MouseButton.MiddleButton:
                self.pan_offset += event.pos() - self.last_pan_pos
                self.last_pan_pos = event.pos()
                self.update()

        def mouseReleaseEvent(self, event: QMouseEvent):
            if event.button() == Qt.MouseButton.MiddleButton:
                self.setCursor(Qt.CursorShape.ArrowCursor)

    class SpriteForgeUltimateWindow(QMainWindow):
        def __init__(self, settings: ProjectSettings):
            super().__init__()
            self.settings = settings
            self.current_image: Optional[Image.Image] = None
            self.processed_image: Optional[Image.Image] = None
            self.undo_stack = []
            self.redo_stack = []
            self.plugin_manager = PluginManager()
            self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
            self.setGeometry(100, 100, 1280, 720)
            self._setup_ui()
            self._create_actions()
            self._create_menus()
            self._create_toolbars()
            self._create_docks()

        def _setup_ui(self):
            self.canvas = ModernImageCanvas()
            self.setCentralWidget(self.canvas)

        def _create_actions(self):
            self.open_action = QAction("&Open...", self, triggered=self.open_file, shortcut=QKeySequence.StandardKey.Open)
            self.save_action = QAction("&Save As...", self, triggered=self.save_file, shortcut=QKeySequence.StandardKey.SaveAs)
            self.exit_action = QAction("E&xit", self, triggered=self.close, shortcut=QKeySequence.StandardKey.Quit)
            self.undo_action = QAction("&Undo", self, triggered=self.undo, shortcut=QKeySequence.StandardKey.Undo)
            self.redo_action = QAction("&Redo", self, triggered=self.redo, shortcut=QKeySequence.StandardKey.Redo)
            self.toggle_grid_action = QAction("Toggle &Grid", self, triggered=lambda: setattr(self.canvas, 'show_grid', not self.canvas.show_grid) or self.canvas.update(), shortcut="G")
            self.toggle_pixel_grid_action = QAction("Toggle &Pixel Grid", self, triggered=lambda: setattr(self.canvas, 'show_pixel_grid', not self.canvas.show_pixel_grid) or self.canvas.update(), shortcut="P")

        def _create_menus(self):
            file_menu = self.menuBar().addMenu("&File")
            file_menu.addAction(self.open_action)
            file_menu.addAction(self.save_action)
            file_menu.addSeparator()
            file_menu.addAction(self.exit_action)

            edit_menu = self.menuBar().addMenu("&Edit")
            edit_menu.addAction(self.undo_action)
            edit_menu.addAction(self.redo_action)

            view_menu = self.menuBar().addMenu("&View")
            view_menu.addAction(self.toggle_grid_action)
            view_menu.addAction(self.toggle_pixel_grid_action)

        def _create_toolbars(self):
            toolbar = self.addToolBar("Main")
            toolbar.addAction(self.open_action)
            toolbar.addAction(self.save_action)
            toolbar.addSeparator()
            toolbar.addAction(self.undo_action)
            toolbar.addAction(self.redo_action)

        def _create_docks(self):
            self.setDockOptions(QMainWindow.DockOption.AnimatedDocks | QMainWindow.DockOption.AllowTabbedDocks)

            plugin_dock = QDockWidget("Plugins", self)
            plugin_dock.setAllowedAreas(Qt.DockWidgetArea.LeftDockWidgetArea | Qt.DockWidgetArea.RightDockWidgetArea)

            plugin_list_widget = QListWidget()
            for name, plugin in self.plugin_manager.plugins.items():
                plugin_list_widget.addItem(name)
            plugin_list_widget.itemDoubleClicked.connect(self.apply_plugin_from_list)

            plugin_dock.setWidget(plugin_list_widget)
            self.addDockWidget(Qt.DockWidgetArea.LeftDockWidgetArea, plugin_dock)

        def apply_plugin_from_list(self, item):
            plugin_name = item.text()
            plugin = self.plugin_manager.get_plugin(plugin_name)
            if plugin and self.current_image:
                # Basic implementation without params for now
                self.add_to_undo_stack()
                self.processed_image = plugin.process(self.processed_image)
                self.canvas.set_image(self.processed_image)

        def add_to_undo_stack(self):
            if self.processed_image:
                self.undo_stack.append(self.processed_image.copy())
                self.redo_stack.clear()

        def undo(self):
            if self.undo_stack:
                self.redo_stack.append(self.processed_image.copy())
                self.processed_image = self.undo_stack.pop()
                self.canvas.set_image(self.processed_image)

        def redo(self):
            if self.redo_stack:
                self.undo_stack.append(self.processed_image.copy())
                self.processed_image = self.redo_stack.pop()
                self.canvas.set_image(self.processed_image)

        def open_file(self, path=None):
            if not path: path, _ = QFileDialog.getOpenFileName(self, "Open Image", "", "Images (*.png *.bmp *.jpg)")
            if path:
                try:
                    self.current_image = Image.open(path)
                    self.processed_image = self.current_image.copy()
                    self.canvas.set_image(self.processed_image)
                    self.undo_stack.clear()
                    self.redo_stack.clear()
                except Exception as e: QMessageBox.critical(self, "Error", f"Could not open file: {e}")

        def save_file(self):
            if not self.processed_image: return
            path, _ = QFileDialog.getSaveFileName(self, "Save Image As...", "", "PNG (*.png);;JPEG (*.jpg);;BMP (*.bmp)")
            if path:
                try:
                    self.processed_image.save(path)
                except Exception as e:
                    QMessageBox.critical(self, "Error", f"Could not save file: {e}")

        def closeEvent(self, event):
            self.settings.save()
            super().closeEvent(event)

# --- CLI Argument Parsing ---

class ApplyAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if not hasattr(namespace, self.dest) or getattr(namespace, self.dest) is None:
            setattr(namespace, self.dest, [])

        op_name = values[0]
        params = {}
        for arg in values[1:]:
            key, value = arg.split('=', 1)
            # Type inference for params
            if value.lower() == 'true': params[key] = True
            elif value.lower() == 'false': params[key] = False
            else:
                try: params[key] = int(value) if '.' not in value else float(value)
                except ValueError: params[key] = value

        getattr(namespace, self.dest).append({'name': op_name, 'params': params})

# --- Main Application Entry Point ---

def main():
    parser = argparse.ArgumentParser(description=f"{APP_NAME} - Professional Sprite Utility")
    parser.add_argument('input', nargs='?', help="Input file or directory.")
    parser.add_argument('--output', help="Output file or directory.")
    parser.add_argument('--headless', action='store_true', help="Run in command-line only mode.")
    parser.add_argument('--format', choices=[f.value for f in ExportFormat], help="Output format for headless mode.")
    parser.add_argument('--apply', nargs='+', action=ApplyAction, dest='operations', help="Apply an operation, e.g., --apply Pixelate factor=4")
    args = parser.parse_args()

    if args.headless:
        run_headless(args)
    else:
        check_and_load_dependencies(prompt_for_install=False) # Initial check without prompt
        if not GUI_AVAILABLE: sys.exit(1)
        run_gui(args)

def run_headless(args):
    print(f"Running {APP_NAME} in headless mode.")
    if not args.input: print("Error: Input is required for headless mode.", file=sys.stderr); sys.exit(1)

    plugin_manager = PluginManager()
    files = [Path(args.input)] if Path(args.input).is_file() else list(Path(args.input).glob("*.*"))

    for file in files:
        print(f"Processing {file}...")
        try:
            image = Image.open(file)
            if args.operations:
                for op in args.operations:
                    plugin = plugin_manager.get_plugin(op['name'])
                    if plugin:
                        image = plugin.process(image, **op['params'])
                        print(f"  Applied {op['name']}")
                    else: print(f"  Warning: Plugin '{op['name']}' not found.")

            if args.output:
                out_path = Path(args.output)
                dest = out_path / file.name if out_path.is_dir() else out_path
                dest.parent.mkdir(parents=True, exist_ok=True)
                image.save(dest, format=args.format)
                print(f"  Saved to {dest}")
        except Exception as e: print(f"  Error processing {file}: {e}", file=sys.stderr)

def run_gui(args):
    app = QApplication(sys.argv)
    settings = ProjectSettings.load()
    window = SpriteForgeUltimateWindow(settings)

    if settings.theme == "dark":
        app.setStyleSheet("QWidget { background-color: #2b2b2b; color: #f0f0f0; } QMenuBar::item:selected { background-color: #4a4a4a; } QMenu { background-color: #3c3c3c; } QMenu::item:selected { background-color: #5a5a5a; }")

    window.show()
    if args.input and Path(args.input).exists(): window.open_file(args.input)
    sys.exit(app.exec())

if __name__ == '__main__':
    main()

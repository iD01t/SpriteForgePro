# Migration Notes: From Pro/Enhanced to Ultimate v3.0.0

This document outlines the key changes and API consolidations made when merging `Sprite Forge Pro` and `Sprite Forge Enhanced` into `Sprite Forge Ultimate`.

## Key Consolidations

The primary goal of v3.0.0 was to create a single source of truth for all features, favoring the more advanced and robust implementations from `Sprite Forge Pro`.

- **Application Name**: The application is now `Sprite Forge Ultimate`.
- **Main Executable**: `sprite_forge_pro.py` and `sprite_forge_enhanced.py` are replaced by `sprite_forge_ultimate.py`.
- **Settings File**: Settings are now stored in a unified `settings.json`. The application will automatically attempt to migrate recognized keys from older configuration files.

## API Changes

### 1. Enums
- All enums (`ExportFormat`, `ProcessingMode`, `SpriteType`) now have a single, unified definition based on the more detailed `Pro` version. String values are used for serialization.

### 2. `ImageProcessor`
- The `ImageProcessor` class is now a unified singleton with the most advanced implementation for each function.
- `apply_doom_palette`: The `Pro` version was kept, which includes `dither` and `preserve_transparency` options.
- `remove_background`: The `Pro` version using OpenCV's `grabCut` is now standard. The simple color-keying method from `Enhanced` is used only as a fallback if OpenCV is not available.
- `enhance_sprite`: Renamed to `enhance` and now includes a `sharpness` parameter.

### 3. `PluginManager`
- The advanced `PluginManager` from `Pro` is now the standard. It loads built-in plugins as well as external Python (`.py`) and JSON (`.json`) plugins from the `./plugins/` directory and a user-specific plugins directory.
- The concept of `JSONConfigPlugin` from `Enhanced` is implicitly handled by the `PluginManager`'s JSON loading mechanism.

### 4. `ProjectSettings`
- The `ProjectSettings` dataclass from `Pro` is used as the base.
- **Backward Compatibility**: The `ProjectSettings.load()` method now recognizes old keys (e.g., `auto_save_interval` from `Pro`, `doom_palette` from `Enhanced`) and maps them to the new schema (`autosave_interval_seconds`, `dither_palette`). Deprecated keys are not saved back to the file.

### 5. Command-Line Interface (CLI)
- The CLI has been standardized and enhanced.
- The new `--apply` flag provides a more intuitive way to chain operations in headless mode.
  - **Old (`Pro`)**: `--plugin "Doom Palette" --plugin-params '{"dither": true}'`
  - **New (`Ultimate`)**: `--apply "Doom Palette" dither=true`

## Deprecations

- All classes and methods from `sprite_forge_enhanced.py` that had a superior equivalent in `sprite_forge_pro.py` have been removed in favor of the `Pro` version.
- The simple dependency installer from `Enhanced` has been replaced by the more robust, user-prompted system in `Ultimate`.

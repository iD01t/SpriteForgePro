#!/bin/bash
# Build script for Sprite Forge Ultimate on macOS
# This script uses PyInstaller to create a .app bundle.

# --- Configuration ---
SCRIPT_NAME="sprite_forge_ultimate.py"
APP_NAME="Sprite Forge Ultimate"
ICON_FILE="assets/app.icns"
PLUGINS_DIR="plugins"

# --- Build Process ---
echo "Building ${APP_NAME} for macOS..."

# 1. Check for PyInstaller
if ! command -v pyinstaller &> /dev/null
then
    echo "PyInstaller could not be found. Please install it first:"
    echo "pip install pyinstaller"
    exit 1
fi

# 2. Check for icon file (optional)
if [ ! -f "${ICON_FILE}" ]; then
    echo "Warning: Icon file not found at ${ICON_FILE}. Building with default icon."
    ICON_OPTION=""
else
    ICON_OPTION="--icon=${ICON_FILE}"
fi

# 3. Run PyInstaller
# We use --windowed to create a .app bundle for a GUI application.
# The --add-data flag bundles our plugins directory. Note the ':' separator for macOS/Linux.
# For a universal2 binary, you might need to run on an Apple Silicon Mac with a
# universal2 Python build, or use cross-compilation tools.
pyinstaller \
    --name "${APP_NAME}" \
    --onefile \
    --windowed \
    $ICON_OPTION \
    --add-data "${PLUGINS_DIR}:${PLUGINS_DIR}" \
    "${SCRIPT_NAME}"

# 4. Check for build errors
if [ $? -ne 0 ]; then
    echo ""
    echo "Build failed with errors."
    exit 1
fi

echo ""
echo "Build successful!"
echo "Application bundle created in the 'dist' folder."
echo ""
echo "Note on Qt plugins: If the .app fails to run due to Qt plugin issues,"
echo "you may need to install 'pyinstaller-hooks-contrib' to ensure all"
echo "necessary Qt components are bundled."
echo "  pip install pyinstaller-hooks-contrib"
exit 0

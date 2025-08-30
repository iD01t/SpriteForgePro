#!/bin/bash
# Build script for Sprite Forge Ultimate on Linux
# This script uses PyInstaller and provides instructions for creating an AppImage.

# --- Configuration ---
SCRIPT_NAME="sprite_forge_ultimate.py"
APP_NAME="Sprite Forge Ultimate"
ICON_FILE="assets/app.png" # AppImage typically uses a .png
PLUGINS_DIR="plugins"
APPDIR_NAME="${APP_NAME}.AppDir"

# --- Build Process ---
echo "Building ${APP_NAME} for Linux..."

# 1. Check for PyInstaller
if ! command -v pyinstaller &> /dev/null
then
    echo "PyInstaller could not be found. Please install it first:"
    echo "pip install pyinstaller"
    exit 1
fi

# 2. Clean up previous builds
rm -rf dist build "${APPDIR_NAME}" "${APP_NAME}-x86_64.AppImage"

# 3. Run PyInstaller to create the initial bundle
# For AppImage, it's better to build a one-folder bundle, not one-file.
pyinstaller \
    --name "${APP_NAME}" \
    --noconfirm \
    --windowed \
    --add-data "${PLUGINS_DIR}:${PLUGINS_DIR}" \
    "${SCRIPT_NAME}"

# 4. Check for build errors
if [ $? -ne 0 ]; then
    echo ""
    echo "PyInstaller build failed with errors."
    exit 1
fi

echo "PyInstaller build successful. Now proceeding to AppImage creation..."

# --- AppImage Creation ---
# This part requires 'appimagetool', which you must download.

# 5. Check for appimagetool
if ! command -v appimagetool &> /dev/null
then
    echo ""
    echo "------------------------------------------------------------------------"
    echo "AppImage Creation (Manual Step Required)"
    echo ""
    echo "'appimagetool' not found. To create an AppImage, please follow these steps:"
    echo "1. Download appimagetool:"
    echo "   wget -c https://github.com/AppImage/AppImageKit/releases/download/continuous/appimagetool-x86_64.AppImage"
    echo "   chmod +x appimagetool-x86_64.AppImage"
    echo ""
    echo "2. Your PyInstaller bundle has been created in: 'dist/${APP_NAME}'"
    echo "   You can run this bundle directly to test it."
    echo ""
    echo "3. To package it as an AppImage, you would typically run:"
    echo "   ./appimagetool-x86_64.AppImage dist/${APP_NAME}"
    echo "------------------------------------------------------------------------"
    exit 0
fi

# 6. If appimagetool is found, proceed with packaging
echo "Found appimagetool. Packaging..."

# The structure for AppImage is specific. We create an AppDir.
mkdir -p "${APPDIR_NAME}/usr/bin"
mv "dist/${APP_NAME}"/* "${APPDIR_NAME}/usr/bin/"
rm -rf "dist/${APP_NAME}"

# Create a symlink for the executable
ln -s "usr/bin/${APP_NAME}" "${APPDIR_NAME}/AppRun"

# Copy icon
if [ -f "${ICON_FILE}" ]; then
    cp "${ICON_FILE}" "${APPDIR_NAME}/${APP_NAME}.png"
    ln -s "${APP_NAME}.png" "${APPDIR_NAME}/.DirIcon"
fi

# Create .desktop file
cat > "${APPDIR_NAME}/${APP_NAME}.desktop" <<EOF
[Desktop Entry]
Name=${APP_NAME}
Exec=${APP_NAME}
Icon=${APP_NAME}
Type=Application
Categories=Graphics;2DGraphics;
EOF

# 7. Run appimagetool
ARCH=x86_64 appimagetool "${APPDIR_NAME}"

if [ $? -ne 0 ]; then
    echo ""
    echo "AppImage creation failed."
    exit 1
fi

echo ""
echo "AppImage created successfully!"
echo "Find it at: ${APP_NAME}-x86_64.AppImage"
exit 0

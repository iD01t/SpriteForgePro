@echo off
rem Build script for Sprite Forge Ultimate on Windows
rem This script uses PyInstaller to create a single executable file.

set SCRIPT_NAME="sprite_forge_ultimate.py"
set APP_NAME="Sprite Forge Ultimate"
set ICON_FILE="assets/app.ico"
set PLUGINS_DIR="plugins"

echo Building %APP_NAME% for Windows...

rem Check if PyInstaller is installed
pyinstaller --version >nul 2>nul
if errorlevel 1 (
    echo PyInstaller not found. Please install it first:
    echo pip install pyinstaller
    exit /b 1
)

rem Check for icon file (optional)
if not exist %ICON_FILE% (
    echo Warning: Icon file not found at %ICON_FILE%. Building with default icon.
    set ICON_OPTION=""
) else (
    set ICON_OPTION=--icon=%ICON_FILE%
)

rem Run PyInstaller
pyinstaller ^
    --name %APP_NAME% ^
    --onefile ^
    --windowed ^
    %ICON_OPTION% ^
    --add-data "%PLUGINS_DIR%;%PLUGINS_DIR%" ^
    %SCRIPT_NAME%

if errorlevel 1 (
    echo.
    echo Build failed with errors.
    exit /b 1
)

echo.
echo Build successful!
echo Executable created in the 'dist' folder.
pause

@echo off
setlocal

echo.
echo ========================================
echo   EchoHands Model Setup
echo ========================================
echo.

echo Checking Docker...

docker --version >nul 2>&1

if errorlevel 1 (
    echo Docker is not installed or not running.
    echo Please install and start Docker Desktop first.
    pause
    exit /b 1
)

echo Docker is available.
echo.

echo Downloading EchoHands models...

docker pull shivanshu043s/echohands-models:latest

if errorlevel 1 (
    echo.
    echo Failed to download EchoHands models.
    pause
    exit /b 1
)

echo.
echo Creating temporary model container...

docker create --name echohands-model-extractor shivanshu043s/echohands-models:latest

if errorlevel 1 (
    echo Failed to create temporary container.
    pause
    exit /b 1
)

echo.
echo Extracting models...

if not exist models mkdir models

docker cp echohands-model-extractor:/models/. models

echo.
echo Cleaning up...

docker rm echohands-model-extractor

echo.
echo ========================================
echo   EchoHands models installed successfully
echo ========================================
echo.

pause
@echo off
chcp 936 >nul
pushd "%~dp0"

echo =====================================
echo Starting What-to-eat-today project...
echo =====================================

docker info >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Docker is not running. Please start Docker Desktop.
    pause
    exit /b 1
)

echo [INFO] Docker is running
echo [INFO] Current directory: %cd%

echo [INFO] Starting services...
docker compose up -d

if errorlevel 1 (
    echo [ERROR] Failed to start services.
    pause
    exit /b 1
)

echo [INFO] Services started successfully
echo [INFO] Waiting for services...
timeout /t 10 /nobreak >nul

echo [INFO] Opening browser...
start "" "http://localhost"

echo =====================================
echo Project started
echo =====================================
echo Home    : http://localhost
echo Frontend: http://localhost:3000
echo Backend : http://localhost:8000
echo Neo4j   : http://localhost:7474
echo Milvus  : http://localhost:9001
echo =====================================

popd
pause
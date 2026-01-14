@echo off
REM Start Milvus using Docker Compose on Windows

echo ========================================
echo Starting Milvus Vector Database
echo ========================================
echo.

REM Check if Docker Desktop is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ERROR: Docker Desktop is not running!
    echo Please start Docker Desktop first.
    echo.
    pause
    exit /b 1
)

echo Docker is running. Starting Milvus...
echo.

docker-compose up -d

echo.
echo Waiting for services to start (30 seconds)...
timeout /t 30 /nobreak >nul

echo.
echo Checking service status...
docker ps

echo.
echo ========================================
echo Milvus should be running on:
echo - Main service: localhost:19530
echo - Admin UI: http://localhost:9091
echo - MinIO Console: http://localhost:9001
echo ========================================
echo.

pause
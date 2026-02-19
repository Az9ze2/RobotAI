@echo off
REM Stop all services on Windows

echo ========================================
echo Stopping All Services
echo ========================================
echo.

echo Stopping Milvus containers...
docker-compose down

echo.
echo Services stopped!
echo.
echo Note: Ollama service continues running in the background.
echo To stop Ollama, close the Ollama application from system tray.
echo.

pause
@echo off
REM Start the FastAPI server on Windows

echo ========================================
echo Starting ROS2 MCP AI Brain Server
echo ========================================
echo.

REM Check if virtual environment is activated
if not defined VIRTUAL_ENV (
    echo Activating virtual environment...
    call venv\Scripts\activate.bat
)

REM Check if Milvus is running
echo Checking Milvus connection...
curl -s http://localhost:19530 >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Cannot connect to Milvus!
    echo Make sure Docker Desktop is running and Milvus is started.
    echo Run: start_milvus.bat
    echo.
    pause
)

REM Check if Ollama is running
echo Checking Ollama connection...
curl -s http://localhost:11434/api/tags >nul 2>&1
if errorlevel 1 (
    echo.
    echo WARNING: Cannot connect to Ollama!
    echo Make sure Ollama is installed and running.
    echo.
    pause
)

echo.
echo Starting FastAPI server...
echo Server will be available at: http://localhost:8000
echo Press Ctrl+C to stop
echo.

python api\main.py

pause
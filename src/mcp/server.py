
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
from loguru import logger

try:
    from mcp_agent import MCPAgent
except ImportError:
    from src.mcp.mcp_agent import MCPAgent

# Initialize App and Agent
app = FastAPI(title="Robot AI Brain - MCP Service")
agent = MCPAgent()

# --- Data Models ---

class FacePayload(BaseModel):
    student_id: Optional[str] = None
    confidence: float
    timestamp: float

class SpeechPayload(BaseModel):
    session_id: str
    text: str
    confidence: float

# --- Endpoints ---

@app.get("/")
async def health_check():
    return {"status": "running", "service": "mcp"}

@app.post("/api/v1/perception/face")
async def process_face(payload: FacePayload):
    try:
        result = agent.process_face_event(payload.model_dump())
        return result
    except Exception as e:
        logger.error(f"Error processing face event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/perception/speech")
async def process_speech(payload: SpeechPayload):
    try:
        result = agent.process_speech_input(
            payload.session_id, 
            payload.text, 
            payload.confidence
        )
        return result
    except Exception as e:
        logger.error(f"Error processing speech event: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

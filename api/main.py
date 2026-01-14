"""
FastAPI Main Application
Central API server for PC AI Brain
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict
import yaml
from pathlib import Path
from loguru import logger
import sys

# Import custom modules
sys.path.append(str(Path(__file__).parent.parent))
from vector_db.milvus_client import MilvusClient
from llm.typhoon_client import TyphoonClient
from mcp.context_builder import ContextBuilder

# Configure logging
logger.add("logs/api.log", rotation="500 MB", level="INFO")

# Load configuration
config_path = Path(__file__).parent.parent / "config" / "settings.yaml"
with open(config_path, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

# Initialize FastAPI
app = FastAPI(
    title="ROS2 MCP Robot AI Brain",
    description="Offline AI system for campus service robot",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize components
milvus_client = None
typhoon_client = None
context_builder = ContextBuilder()


# Pydantic models
class ContextUpdate(BaseModel):
    session_id: str
    student_id: Optional[str] = None
    student_name: Optional[str] = None
    location: Optional[str] = None
    environment_data: Optional[Dict] = None


class SpeechInput(BaseModel):
    session_id: str
    text: str
    confidence: float


class ChatResponse(BaseModel):
    session_id: str
    response_text: str
    intent: str
    should_navigate: bool
    navigation_goal: Optional[Dict] = None


class MemoryInsert(BaseModel):
    text: str
    memory_type: str
    student_id: Optional[str] = ""
    timestamp: Optional[int] = None


class MemorySearch(BaseModel):
    query: str
    top_k: int = 5
    memory_type: Optional[str] = None
    student_id: Optional[str] = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    global milvus_client, typhoon_client
    
    try:
        # Initialize Milvus
        milvus_client = MilvusClient(
            host=config["milvus"]["host"],
            port=config["milvus"]["port"],
            collection_name=config["milvus"]["collection_name"],
            embedding_model=config["embedding"]["model_name"]
        )
        logger.info("Milvus client initialized")
        
        # Initialize Typhoon LLM
        typhoon_client = TyphoonClient(
            api_url=config["llm"]["api_url"],
            model=config["llm"]["model"]
        )
        logger.info("Typhoon LLM client initialized")
        
        logger.success("All services started successfully")
        
    except Exception as e:
        logger.error(f"Startup failed: {e}")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    if milvus_client:
        milvus_client.close()
    logger.info("Services shut down")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "running",
        "service": "ROS2 MCP AI Brain",
        "version": "1.0.0"
    }


@app.post("/context/update")
async def update_context(ctx: ContextUpdate):
    """
    Update session context with new information
    Used by ROS2 nodes to update robot state
    """
    try:
        # Create session if doesn't exist
        if not context_builder.get_session(ctx.session_id):
            context_builder.create_session(ctx.session_id)
        
        # Update student identity
        if ctx.student_id and ctx.student_name:
            context_builder.update_student_identity(
                ctx.session_id, ctx.student_id, ctx.student_name
            )
        
        # Update location
        if ctx.location:
            context_builder.update_location(ctx.session_id, ctx.location)
        
        # Update environment
        if ctx.environment_data:
            context_builder.update_environment(ctx.session_id, ctx.environment_data)
        
        return {
            "status": "success",
            "session_id": ctx.session_id,
            "message": "Context updated"
        }
        
    except Exception as e:
        logger.error(f"Context update failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/speech/input", response_model=ChatResponse)
async def process_speech(speech: SpeechInput):
    """
    Process speech input and generate response
    Main conversation endpoint
    """
    try:
        # Check confidence threshold
        if speech.confidence < config["stt"]["confidence_threshold"]:
            return ChatResponse(
                session_id=speech.session_id,
                response_text="ขอโทษค่ะ ฉันไม่ค่อยได้ยินชัดเจน ช่วยพูดอีกครั้งได้ไหมคะ",
                intent="clarification",
                should_navigate=False
            )
        
        # Add user message to context
        context_builder.add_conversation_turn(
            speech.session_id, "user", speech.text
        )
        
        # Search relevant memories
        memories = milvus_client.search_memory(
            query=speech.text,
            top_k=config["memory"]["retrieval_top_k"]
        )
        
        # Build LLM context
        llm_context = context_builder.build_llm_context(
            speech.session_id, retrieved_memories=memories
        )
        
        # Create prompt
        system_prompt = """คุณคือหุ่นยนต์บริการในมหาวิทยาลัย ชื่อว่า "น้องบอท" 
คุณพูดภาษาไทยอย่างเป็นมิตร และช่วยเหลือนักศึกษาในการนำทาง ตอบคำถาม และบันทึกไดอารี่

ตอบคำถามอย่างกระชับและเป็นธรรมชาติ 
ถ้านักศึกษาต้องการนำทางไปยังสถานที่ ให้ระบุ intent เป็น "navigation"
ถ้าเป็นการสนทนาทั่วไป ให้ระบุ intent เป็น "conversation"

ตอบกลับในรูปแบบ JSON:
{
    "response": "คำตอบของคุณ",
    "intent": "navigation หรือ conversation",
    "location": "ชื่อสถานที่ (ถ้า intent เป็น navigation)"
}"""
        
        context_text = context_builder.format_context_as_prompt(llm_context)
        user_message = f"{context_text}\n\nนักศึกษา: {speech.text}\n\nน้องบอท:"
        
        # Get LLM response
        llm_output = typhoon_client.generate_structured(
            system_prompt=system_prompt,
            user_message=user_message,
            temperature=config["llm"]["temperature"]
        )
        
        if not llm_output:
            raise Exception("LLM returned no response")
        
        # Parse response
        response_text = llm_output.get("response", "ขอโทษค่ะ ฉันไม่เข้าใจ")
        intent = llm_output.get("intent", "conversation")
        location = llm_output.get("location")
        
        # Add assistant response to context
        context_builder.add_conversation_turn(
            speech.session_id, "assistant", response_text
        )
        
        # Prepare navigation goal if needed
        navigation_goal = None
        should_navigate = False
        
        if intent == "navigation" and location:
            should_navigate = True
            navigation_goal = {
                "target_location": location,
                "priority": "normal"
            }
        
        return ChatResponse(
            session_id=speech.session_id,
            response_text=response_text,
            intent=intent,
            should_navigate=should_navigate,
            navigation_goal=navigation_goal
        )
        
    except Exception as e:
        logger.error(f"Speech processing failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/insert")
async def insert_memory(memory: MemoryInsert):
    """Insert new memory into vector database"""
    try:
        success = milvus_client.insert_memory(
            text=memory.text,
            memory_type=memory.memory_type,
            student_id=memory.student_id,
            timestamp=memory.timestamp
        )
        
        if success:
            return {"status": "success", "message": "Memory inserted"}
        else:
            raise Exception("Memory insertion failed")
            
    except Exception as e:
        logger.error(f"Memory insert failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/memory/search")
async def search_memory(search: MemorySearch):
    """Search memories in vector database"""
    try:
        memories = milvus_client.search_memory(
            query=search.query,
            top_k=search.top_k,
            memory_type=search.memory_type,
            student_id=search.student_id
        )
        
        return {
            "status": "success",
            "count": len(memories),
            "memories": memories
        }
        
    except Exception as e:
        logger.error(f"Memory search failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """Get session context"""
    session = context_builder.get_session(session_id)
    if session:
        return {"status": "success", "session": session}
    else:
        raise HTTPException(status_code=404, detail="Session not found")


@app.delete("/session/{session_id}")
async def clear_session(session_id: str):
    """Clear session context"""
    context_builder.clear_session(session_id)
    return {"status": "success", "message": f"Session {session_id} cleared"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config["api"]["host"],
        port=config["api"]["port"],
        log_level="info"
    )
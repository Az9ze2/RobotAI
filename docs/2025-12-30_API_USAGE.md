# ROS2 Robot AI Brain - API Documentation

## Overview

The ROS2 Robot AI Brain is a FastAPI-based service that provides AI capabilities for a campus service robot. It includes:
- Thai language conversation processing (via Typhoon LLM)
- Vector-based memory storage and retrieval (via Milvus)
- Session and context management
- Navigation intent detection
- Student personalization

**Base URL:** `http://localhost:8000`

**Technology Stack:**
- **LLM:** Ollama with Typhoon 2.1 (Thai language model)
- **Vector DB:** Milvus with BAAI/bge-m3 embeddings
- **Framework:** FastAPI with Pydantic validation
- **Speech:** Whisper Small (STT), VITS (TTS)

---

## Table of Contents

1. [Authentication](#authentication)
2. [Endpoints](#endpoints)
   - [Health Check](#health-check)
   - [Context Management](#context-management)
   - [Speech Processing](#speech-processing)
   - [Memory Operations](#memory-operations)
   - [Session Management](#session-management)
3. [Data Models](#data-models)
4. [Usage Examples](#usage-examples)
5. [Error Handling](#error-handling)
6. [Best Practices](#best-practices)

---

## Authentication

Currently, the API does not require authentication. It is designed for internal use within a campus network.

---

## Endpoints

### Health Check

#### `GET /`

Check if the API server is running.

**Response:**
```json
{
  "status": "running",
  "service": "ROS2 MCP AI Brain",
  "version": "1.0.0"
}
```

**Example:**
```bash
curl http://localhost:8000/
```

---

### Context Management

#### `POST /context/update`

Update session context with student identity, location, and environment data.

**Request Body:**
```json
{
  "session_id": "string",
  "student_id": "string (optional)",
  "student_name": "string (optional)",
  "location": "string (optional)",
  "environment_data": {
    "temperature": 28.5,
    "humidity": 65,
    "time_of_day": "afternoon"
  }
}
```

**Response:**
```json
{
  "status": "success",
  "session_id": "session_123",
  "message": "Context updated"
}
```

**Example:**
```python
import requests

response = requests.post(
    "http://localhost:8000/context/update",
    json={
        "session_id": "session_001",
        "student_id": "STD12345",
        "student_name": "สมชาย ใจดี",
        "location": "อาคารเรียนรวม 1"
    }
)
```

---

### Speech Processing

#### `POST /speech/input`

Process speech input and generate a response. This is the main conversation endpoint.

**Request Body:**
```json
{
  "session_id": "string",
  "text": "string",
  "confidence": 0.95
}
```

**Response:**
```json
{
  "session_id": "session_123",
  "response_text": "สวัสดีครับ ยินดีที่ได้พบคุณ",
  "intent": "conversation",
  "should_navigate": false,
  "navigation_goal": null
}
```

**Intent Types:**
- `conversation` - General chat
- `navigation` - User wants to go somewhere
- `clarification` - Low confidence, needs repeat

**Example - Simple Conversation:**
```python
response = requests.post(
    "http://localhost:8000/speech/input",
    json={
        "session_id": "session_001",
        "text": "สวัสดีครับ",
        "confidence": 0.95
    }
)
print(response.json()["response_text"])
# Output: "สวัสดีครับ สมชาย ยินดีที่ได้พบ"
```

**Example - Navigation Request:**
```python
response = requests.post(
    "http://localhost:8000/speech/input",
    json={
        "session_id": "session_001",
        "text": "พาฉันไปห้องสมุดหน่อย",
        "confidence": 0.89
    }
)

result = response.json()
print(result["response_text"])
print(result["should_navigate"])  # True
print(result["navigation_goal"])  # {"target_location": "ห้องสมุด", "priority": "normal"}
```

**Example - Low Confidence:**
```python
response = requests.post(
    "http://localhost:8000/speech/input",
    json={
        "session_id": "session_001",
        "text": "...ห้อง...สม...",
        "confidence": 0.55  # Below threshold
    }
)

print(response.json()["response_text"])
# Output: "ขอโทษค่ะ ฉันไม่ค่อยได้ยินชัดเจน ช่วยพูดอีกครั้งได้ไหมคะ"
```

---

### Memory Operations

#### `POST /memory/insert`

Insert a new memory into the vector database.

**Request Body:**
```json
{
  "text": "string",
  "memory_type": "string",
  "student_id": "string (optional)",
  "timestamp": 1234567890
}
```

**Memory Types:**
- `student_profile` - Student preferences and characteristics
- `diary` - Conversation summaries and interactions
- `knowledge` - General campus information
- `navigation` - Location and route information

**Response:**
```json
{
  "status": "success",
  "message": "Memory inserted"
}
```

**Example:**
```python
import time

# Insert student profile
response = requests.post(
    "http://localhost:8000/memory/insert",
    json={
        "text": "นักศึกษาสมชายชอบเรียนวิชาคณิตศาสตร์และมักมาอาคารเรียนรวมทุกวันจันทร์",
        "memory_type": "student_profile",
        "student_id": "STD12345",
        "timestamp": int(time.time())
    }
)

# Insert campus knowledge
response = requests.post(
    "http://localhost:8000/memory/insert",
    json={
        "text": "ห้องสมุดตั้งอยู่ที่อาคาร 5 ชั้น 2 เปิดบริการจันทร์-ศุกร์ 08:00-20:00",
        "memory_type": "knowledge",
        "student_id": "",
        "timestamp": int(time.time())
    }
)
```

---

#### `POST /memory/search`

Search for relevant memories using semantic similarity.

**Request Body:**
```json
{
  "query": "string",
  "top_k": 5,
  "memory_type": "string (optional)",
  "student_id": "string (optional)"
}
```

**Response:**
```json
{
  "status": "success",
  "count": 2,
  "memories": [
    {
      "text": "ห้องสมุดตั้งอยู่ที่อาคาร 5 ชั้น 2...",
      "score": 0.85,
      "memory_type": "knowledge",
      "student_id": "",
      "timestamp": 1234567890
    }
  ]
}
```

**Example:**
```python
# Search general knowledge
response = requests.post(
    "http://localhost:8000/memory/search",
    json={
        "query": "ห้องสมุดอยู่ที่ไหน",
        "top_k": 3,
        "memory_type": "knowledge"
    }
)

for memory in response.json()["memories"]:
    print(f"Score: {memory['score']:.3f} - {memory['text']}")

# Search student-specific memories
response = requests.post(
    "http://localhost:8000/memory/search",
    json={
        "query": "สมชายชอบอะไร",
        "top_k": 5,
        "student_id": "STD12345"
    }
)
```

---

### Session Management

#### `GET /session/{session_id}`

Retrieve session context and conversation history.

**Response:**
```json
{
  "status": "success",
  "session": {
    "session_id": "session_123",
    "student_id": "STD12345",
    "student_name": "สมชาย ใจดี",
    "conversation_history": [
      {
        "role": "user",
        "content": "สวัสดีครับ",
        "timestamp": "2025-12-30T10:00:00"
      },
      {
        "role": "assistant",
        "content": "สวัสดีครับ ยินดีต้อนรับ",
        "timestamp": "2025-12-30T10:00:01"
      }
    ],
    "current_location": "อาคารเรียนรวม 1",
    "environment_info": {},
    "created_at": "2025-12-30T10:00:00",
    "last_updated": "2025-12-30T10:00:01"
  }
}
```

**Example:**
```python
response = requests.get("http://localhost:8000/session/session_001")
session = response.json()["session"]
print(f"Student: {session['student_name']}")
print(f"Location: {session['current_location']}")
```

---

#### `DELETE /session/{session_id}`

Clear session context and history.

**Response:**
```json
{
  "status": "success",
  "message": "Session session_123 cleared"
}
```

**Example:**
```python
response = requests.delete("http://localhost:8000/session/session_001")
```

---

## Data Models

### ContextUpdate
```python
{
  "session_id": str,           # Required
  "student_id": str,           # Optional
  "student_name": str,         # Optional
  "location": str,             # Optional
  "environment_data": dict     # Optional
}
```

### SpeechInput
```python
{
  "session_id": str,           # Required
  "text": str,                 # Required
  "confidence": float          # Required (0.0 - 1.0)
}
```

### ChatResponse
```python
{
  "session_id": str,
  "response_text": str,
  "intent": str,               # conversation | navigation | clarification
  "should_navigate": bool,
  "navigation_goal": dict      # null or {"target_location": str, "priority": str}
}
```

### MemoryInsert
```python
{
  "text": str,                 # Required (max 2000 chars)
  "memory_type": str,          # Required
  "student_id": str,           # Optional (default: "")
  "timestamp": int             # Optional (auto-generated if null)
}
```

### MemorySearch
```python
{
  "query": str,                # Required
  "top_k": int,                # Default: 5
  "memory_type": str,          # Optional filter
  "student_id": str            # Optional filter
}
```

---

## Usage Examples

### Complete Conversation Flow

```python
import requests
import time

BASE_URL = "http://localhost:8000"
session_id = f"session_{int(time.time())}"

# 1. Initialize context
requests.post(f"{BASE_URL}/context/update", json={
    "session_id": session_id,
    "student_id": "STD12345",
    "student_name": "สมชาย ใจดี",
    "location": "อาคารวิทยาศาสตร์"
})

# 2. Insert relevant knowledge
requests.post(f"{BASE_URL}/memory/insert", json={
    "text": "โรงอาหารกลางอยู่ติดอาคารเรียนรวม 2 มีอาหารหลากหลาย",
    "memory_type": "knowledge",
    "student_id": ""
})

# 3. Start conversation
response = requests.post(f"{BASE_URL}/speech/input", json={
    "session_id": session_id,
    "text": "สวัสดีครับ",
    "confidence": 0.95
})
print(response.json()["response_text"])

# 4. Ask for navigation
response = requests.post(f"{BASE_URL}/speech/input", json={
    "session_id": session_id,
    "text": "พาไปโรงอาหารหน่อย",
    "confidence": 0.89
})

result = response.json()
if result["should_navigate"]:
    print(f"Navigating to: {result['navigation_goal']['target_location']}")

# 5. Check session history
response = requests.get(f"{BASE_URL}/session/{session_id}")
history = response.json()["session"]["conversation_history"]
for turn in history:
    print(f"{turn['role']}: {turn['content']}")
```

### Building a Campus Knowledge Base

```python
import requests

knowledge_base = [
    "ห้องสมุดกลางตั้งอยู่ที่อาคาร 7 ชั้น 1-3 เปิดจันทร์-ศุกร์ 08:00-20:00",
    "สำนักงานทะเบียนอยู่ที่อาคารเรียนรวม 1 ชั้น 1 เปิด 08:30-16:30",
    "โรงพยาบาลมหาวิทยาลัยอยู่ติดประตูหลัก เปิด 24 ชั่วโมง",
    "สนามกีฬากลางมีสนามฟุตบอล บาสเกตบอล และลู่วิ่ง ฟรีสำหรับนักศึกษา",
    "ศูนย์คอมพิวเตอร์อยู่อาคาร IT ชั้น 2 มี WiFi และบริการพิมพ์"
]

for knowledge in knowledge_base:
    requests.post("http://localhost:8000/memory/insert", json={
        "text": knowledge,
        "memory_type": "knowledge",
        "student_id": "",
        "timestamp": int(time.time())
    })
    time.sleep(0.1)  # Small delay for indexing

# Query the knowledge base
response = requests.post("http://localhost:8000/memory/search", json={
    "query": "ห้องสมุดเปิดถึงกี่โมง",
    "top_k": 2,
    "memory_type": "knowledge"
})

for mem in response.json()["memories"]:
    print(f"[{mem['score']:.3f}] {mem['text']}")
```

---

## Error Handling

### HTTP Status Codes

- `200` - Success
- `404` - Resource not found (e.g., session doesn't exist)
- `422` - Validation error (invalid request body)
- `500` - Internal server error

### Error Response Format

```json
{
  "detail": "Error message describing what went wrong"
}
```

### Common Errors

**Low Confidence Input:**
- Automatically handled with clarification response
- Threshold: 0.7 (configurable in `settings.yaml`)

**Session Not Found:**
- Sessions are auto-created if they don't exist
- Use `DELETE /session/{id}` to explicitly clear

**Memory Insertion Failure:**
- Check text length (max 2000 characters)
- Ensure Milvus is running
- Check logs in `logs/api.log`

**LLM Timeout:**
- Default timeout: 30 seconds
- Check Ollama service status
- Verify model is loaded

---

## Best Practices

### 1. Session Management
- Use meaningful session IDs (e.g., `session_{timestamp}` or `session_{student_id}`)
- Update context before starting conversation
- Clear sessions when interaction ends to free memory

### 2. Memory Management
- Insert memories with descriptive text
- Use appropriate memory types for filtering
- Include student_id for personalized memories
- Regularly clean old memories (not implemented yet)

### 3. Conversation Flow
- Always check confidence scores
- Handle low confidence gracefully
- Keep conversation history under 10 turns (auto-limited)
- Update location context when robot moves

### 4. Performance
- Batch memory insertions when possible
- Use `top_k` parameter to limit search results
- Monitor Milvus collection size
- Check API logs for slow requests

### 5. Thai Language Processing
- Use proper Thai characters (UTF-8)
- Avoid mixing Thai and English unnecessarily
- Keep prompts natural and conversational
- Test with various Thai dialects and formality levels

### 6. Error Recovery
- Implement retry logic for critical operations
- Log all errors for debugging
- Provide fallback responses for LLM failures
- Monitor system health regularly

---

## Configuration

Key configuration options in `config/settings.yaml`:

```yaml
api:
  port: 8000
  debug: true

llm:
  model: "scb10x/typhoon2.1-gemma3-4b:latest"
  temperature: 0.7
  max_tokens: 512

milvus:
  host: "localhost"
  port: 19530

embedding:
  model_name: "BAAI/bge-m3"
  dimension: 1024

stt:
  confidence_threshold: 0.7

memory:
  retrieval_top_k: 5
  similarity_threshold: 0.75
```

---

## Testing

Run comprehensive tests:
```bash
python tests/test_api.py
```

Run sample demonstrations:
```bash
python tests/sample_requests.py
```

---

## Logging

Logs are stored in:
- `logs/api.log` - API server logs
- `logs/ai_brain.log` - General system logs

Log rotation: 500 MB

---

## Support and Troubleshooting

### API Server Won't Start
1. Check if port 8000 is available
2. Verify virtual environment is activated
3. Check `logs/api.log` for errors

### Milvus Connection Failed
1. Ensure Docker is running
2. Check Milvus containers: `docker ps`
3. Restart Milvus: `docker-compose restart`

### LLM Not Responding
1. Check Ollama service: `ollama list`
2. Verify model is loaded
3. Test with: `ollama run scb10x/typhoon2.1-gemma3-4b:latest`

### Memory Search Returns No Results
1. Ensure memories are inserted first
2. Check embedding model is loaded
3. Try broader search queries
4. Verify Milvus collection exists

---

## API Interactive Documentation

FastAPI provides interactive API documentation:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

Use these for testing endpoints directly in your browser.

---

## Version History

**v1.0.0** (2025-12-30)
- Initial release
- Thai language conversation support
- Vector memory storage
- Navigation intent detection
- Session management

---

## License

Internal use only - Campus Service Robot Project

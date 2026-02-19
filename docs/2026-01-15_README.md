# ROS2 Robot AI Brain

A complete offline AI system for a campus service robot with Thai language support, memory management, and intelligent conversation capabilities.

## 🎯 Project Overview

This system serves as the "brain" of a campus service robot, providing:
- **Thai Language Processing** using Typhoon LLM (via Ollama)
- **Long-term Memory** with vector embeddings (Milvus + BAAI/bge-m3)
- **Session Management** with context awareness
- **Navigation Intent Detection** for autonomous movement
- **Student Personalization** through memory retrieval

### Architecture

```
┌─────────────────────────────────────────────────────────┐
│                     PC (AI Brain)                       │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │   FastAPI    │  │    Ollama    │  │   Milvus     │ │
│  │    Server    │  │  (Typhoon)   │  │  Vector DB   │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│         │                 │                  │         │
│         └─────────────────┴──────────────────┘         │
│                         MCP                            │
└─────────────────────────────────────────────────────────┘
                         │
                    ROS2 Bridge
                         │
┌─────────────────────────────────────────────────────────┐
│              Raspberry Pi (Motor Control)               │
│                                                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │     ROS2     │  │    Motor     │  │     Face     │ │
│  │  Navigation  │  │   Control    │  │ Recognition  │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────┘
```

## 📋 Features

### Core Capabilities
- ✅ Thai language conversation with context awareness
- ✅ Vector-based semantic memory storage and retrieval
- ✅ Session management with conversation history
- ✅ Navigation intent detection and goal extraction
- ✅ Student recognition and personalization
- ✅ Low-confidence speech handling
- ✅ Real-time system health monitoring

### API Endpoints
- `GET /` - Health check
- `POST /context/update` - Update session context
- `POST /speech/input` - Process speech and generate response
- `POST /memory/insert` - Store new memories
- `POST /memory/search` - Semantic memory search
- `GET /session/{id}` - Retrieve session info
- `DELETE /session/{id}` - Clear session

## 🚀 Quick Start

### Prerequisites

1. **Python 3.10+**
2. **Docker Desktop** (for Milvus)
3. **Ollama** with Typhoon model
4. **Git** (for cloning)

### Installation

```bash
# 1. Clone the repository
cd Desktop/RobotAI

# 2. Create virtual environment
python -m venv venv
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Milvus (Docker)
docker-compose up -d

# 5. Install Ollama and download Typhoon model
# Download Ollama from https://ollama.ai
ollama pull scb10x/typhoon2.1-gemma3-4b:latest

# 6. Verify setup
python validate_setup.py
```

### Running the System

```bash
# Start the API server
python api/main.py

# In another terminal, run system monitor
python monitor.py

# Run tests
python tests/test_api.py

# Run sample demonstrations
python tests/sample_requests.py
```

## 📁 Project Structure

```
RobotAI/
├── api/
│   └── main.py                 # FastAPI application
├── config/
│   └── settings.yaml           # Configuration file
├── llm/
│   └── typhoon_client.py       # Ollama/Typhoon interface
├── mcp/
│   └── context_builder.py      # Context management
├── vector_db/
│   └── milvus_client.py        # Milvus vector operations
├── tests/
│   ├── test_api.py             # Comprehensive API tests
│   └── sample_requests.py      # Usage demonstrations
├── docs/
│   └── API_USAGE.md            # Detailed API documentation
├── logs/                       # Log files
├── docker-compose.yml          # Milvus configuration
├── requirements.txt            # Python dependencies
├── validate_setup.py           # Setup verification
├── monitor.py                  # System health monitor
└── README.md                   # This file
```

## 🔧 Configuration

Edit `config/settings.yaml`:

```yaml
api:
  host: "0.0.0.0"
  port: 8000

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

## 📖 Usage Examples

### Simple Conversation

```python
import requests

# Initialize context
requests.post("http://localhost:8000/context/update", json={
    "session_id": "session_001",
    "student_id": "STD12345",
    "student_name": "สมชาย ใจดี",
    "location": "อาคารวิทยาศาสตร์"
})

# Send message
response = requests.post("http://localhost:8000/speech/input", json={
    "session_id": "session_001",
    "text": "สวัสดีครับ",
    "confidence": 0.95
})

print(response.json()["response_text"])
# Output: "สวัสดีครับ สมชาย ยินดีที่ได้พบ"
```

### Navigation Request

```python
response = requests.post("http://localhost:8000/speech/input", json={
    "session_id": "session_001",
    "text": "พาฉันไปห้องสมุดหน่อย",
    "confidence": 0.89
})

result = response.json()
if result["should_navigate"]:
    print(f"Navigating to: {result['navigation_goal']['target_location']}")
```

### Memory Management

```python
# Insert campus knowledge
requests.post("http://localhost:8000/memory/insert", json={
    "text": "ห้องสมุดตั้งอยู่ที่อาคาร 5 ชั้น 2 เปิดจันทร์-ศุกร์ 08:00-20:00",
    "memory_type": "knowledge",
    "student_id": ""
})

# Search memories
response = requests.post("http://localhost:8000/memory/search", json={
    "query": "ห้องสมุดอยู่ที่ไหน",
    "top_k": 3,
    "memory_type": "knowledge"
})

for mem in response.json()["memories"]:
    print(f"[{mem['score']:.3f}] {mem['text']}")
```

See `docs/API_USAGE.md` for complete documentation.

## 🧪 Testing

### Run All Tests
```bash
python tests/test_api.py
```

### Run Sample Demonstrations
```bash
python tests/sample_requests.py
```

### System Health Check
```bash
python monitor.py
```

### Continuous Monitoring
```bash
python monitor.py --continuous 60
```

## 📊 Monitoring

The system includes comprehensive monitoring:

```bash
python monitor.py
```

Checks:
- ✅ API server health and response time
- ✅ Ollama LLM service and loaded models
- ✅ Docker and Milvus container status
- ✅ System resources (CPU, memory, disk)
- ✅ Memory operations performance
- ✅ End-to-end conversation flow

## 🐛 Troubleshooting

### API Server Won't Start
```bash
# Check if port 8000 is in use
netstat -ano | findstr :8000

# Check logs
cat logs/api.log
```

### Milvus Connection Failed
```bash
# Check Docker containers
docker ps

# Restart Milvus
docker-compose restart

# Check logs
docker-compose logs milvus-standalone
```

### LLM Not Responding
```bash
# Check Ollama service
ollama list

# Test model directly
ollama run scb10x/typhoon2.1-gemma3-4b:latest "สวัสดี"
```

### Memory Search Returns Nothing
1. Ensure memories are inserted first
2. Check if embedding model is loaded (logs will show download progress)
3. Verify Milvus collection exists
4. Try broader search queries

## 📈 Performance

Typical response times:
- Health check: ~2-5ms
- Context update: ~10-20ms
- Memory insert: ~100-200ms
- Memory search: ~50-150ms
- Conversation (with LLM): ~3-8 seconds

Optimize by:
- Reducing `max_tokens` in config
- Adjusting `temperature` for faster but less creative responses
- Using smaller embedding models
- Batching memory insertions

## 🔒 Security

**Note:** This system is designed for internal campus network use.

For production deployment:
- Add API authentication (JWT tokens)
- Enable HTTPS/TLS
- Implement rate limiting
- Add input sanitization
- Use environment variables for sensitive config
- Set up network firewalls

## 📚 Documentation

- [API Usage Guide](docs/API_USAGE.md) - Complete API reference
- [Test Scripts](tests/) - Example code and test cases
- [Configuration](config/settings.yaml) - System settings

## 🤝 Integration with ROS2

The PC communicates with the Raspberry Pi via ROS2 topics:

```python
# ROS2 node subscribes to speech recognition
/robot/speech_input -> PC API /speech/input

# PC publishes response and navigation goals
PC API response -> /robot/speech_output
Navigation goal -> /robot/navigation/goal

# Context updates from robot sensors
/robot/location -> PC API /context/update
/robot/face_detected -> PC API /context/update
```

## 📝 Development Status

### Completed ✅
- FastAPI server with all endpoints
- Thai language conversation (Typhoon LLM)
- Vector memory storage (Milvus)
- Session and context management
- Navigation intent detection
- Comprehensive testing suite
- System monitoring
- Full API documentation

### Future Enhancements 🚧
- Speech-to-Text integration (Whisper)
- Text-to-Speech implementation (VITS)
- PostgreSQL for structured metadata
- ROS2 bridge implementation
- Face recognition integration
- Web dashboard for monitoring
- Diary summarization
- Memory pruning/archiving

## 📄 License

Internal use only - Campus Service Robot Project

## 👥 Team

Developed for campus service robot with Thai language capabilities.

## 🙏 Acknowledgments

- **Typhoon LLM** - SCB 10X for Thai language model
- **Milvus** - Vector database
- **Ollama** - LLM inference
- **FastAPI** - Web framework
- **BAAI** - BGE-M3 embeddings

---

## 📞 Support

For issues or questions:
1. Check logs in `logs/` directory
2. Run `python monitor.py` to diagnose issues
3. Verify setup with `python validate_setup.py`
4. Review `docs/API_USAGE.md` for API details

---

**Last Updated:** 2025-12-30
**Version:** 1.0.0

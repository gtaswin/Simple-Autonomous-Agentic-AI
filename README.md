# ⚡ Autonomous Agentic AI System

**Next-Generation 4-Agent LangGraph Architecture** with cost-effective hybrid processing: **75% local transformers + 25% external LLM**.

A sophisticated autonomous AI assistant featuring **LangGraph multi-agent orchestration**, **3-tier memory architecture**, **real-time insights**, and **autonomous reasoning capabilities**.

[![Python](https://img.shields.io/badge/Python-3.13.5-blue.svg)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.5.4-green.svg)](https://langchain.com/langgraph)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![Next.js](https://img.shields.io/badge/Next.js-15.3.5-black.svg)](https://nextjs.org)
[![Redis](https://img.shields.io/badge/Redis-Stack-red.svg)](https://redis.io/docs/stack/)
[![Qdrant](https://img.shields.io/badge/Qdrant-1.14.1-blue.svg)](https://qdrant.tech)

## 🎯 **Key Features**

- **🤖 4-Agent LangGraph Architecture**: Specialized agents with LangChain integration
- **🔗 LangChain + LangGraph**: Pure LangGraph orchestration with LangChain agent implementations
- **🧠 Autonomous Intelligence**: Background thinking, pattern discovery, milestone tracking  
- **💾 3-Tier Memory System**: Session, Working, Short-term (TTL), Long-term (permanent)
- **💰 Cost-Effective**: 75% local processing + 25% external LLM = 60-80% cost reduction
- **⚡ Real-time**: WebSocket streaming with autonomous insights broadcasting
- **🔒 Privacy-First**: User-isolated memory with dedicated insight storage
- **🌐 Multi-Provider LLM**: Groq, Anthropic Claude, Google Gemini support

## 🔗 **LangChain + LangGraph Integration**

**Architecture Philosophy**: 
- **LangGraph** handles orchestration with StateGraph, conditional routing, and checkpointing
- **LangChain** implements all 4 agents with proper framework components (AgentExecutor, VectorStoreRetrieverMemory, Tools, Prompts)
- **No Hybrid Approaches**: Pure implementation without fallback methods

**Agent Framework Integration**:
- **Memory Reader**: LangChain VectorStoreRetrieverMemory with HybridMemoryRetriever
- **Memory Writer**: LangChain Tools and ChatPromptTemplate for fact extraction
- **Knowledge Agent**: LangChain AgentExecutor with ReAct pattern and Wikipedia/Wikidata tools
- **Organizer Agent**: LangChain synthesis with proper message handling and templates

## 🤖 **4-Agent Architecture**

| Agent | Processing Model | Memory Access | Primary Responsibility |
|-------|------------------|---------------|------------------------|
| **Memory Reader** | 🔄 LOCAL Transformers | Read all memory tiers | Context retrieval & summarization |
| **Memory Writer** | 🔄 LOCAL Transformers | Write all memory tiers | Fact extraction & storage |
| **Knowledge** | 🔄 LOCAL Transformers | Working memory only | External research (Wikipedia/Wikidata) |
| **Organizer** | 🌐 EXTERNAL LLM | Working + Long-term read | Response synthesis & coordination |

### **Cost Optimization Strategy**
- **75% Local Processing**: Memory Reader, Memory Writer, Knowledge agents use local transformers
- **25% External LLM**: Only Organizer agent makes API calls for complex synthesis
- **Parallel Execution**: Memory and Knowledge agents run concurrently
- **Intelligent Routing**: Complexity-based agent selection

## 🧠 **Memory & Intelligence System**

### **3-Tier Memory Architecture**
```
┌─ Session Memory (Redis Lists) ──────────────────────────────────┐
│  Purpose: Complete conversation history                         │
│  Limit: 50 conversations per user | TTL: Permanent              │
└─────────────────────────────────────────────────────────────────┘

┌─ Working Memory (Redis Lists per Agent) ────────────────────────┐
│  Purpose: Agent context & scratchpad                           │
│  Limit: 7 items per agent per user | TTL: 7 days               │
│  Pattern: working_memory:{user_name}:{agent_name}              │
└─────────────────────────────────────────────────────────────────┘

┌─ Short-term Memory (Redis Vector + TTL) ────────────────────────┐
│  Purpose: Temporary facts with semantic search                 │
│  TTL: 6 hours to 3 months (importance-based)                  │
│  Technology: Redis Stack Vector Index                          │
└─────────────────────────────────────────────────────────────────┘

┌─ Long-term Memory (Qdrant Vector) ──────────────────────────────┐
│  Purpose: Permanent important facts (importance ≥ 0.9)         │
│  TTL: Never expires | Technology: Qdrant vector database       │
│  Privacy: User-specific with mandatory filtering               │
└─────────────────────────────────────────────────────────────────┘
```

### **Autonomous Insights System**
```
┌─ Dedicated Insight Storage (Redis Hash) ────────────────────────┐
│  Purpose: Latest autonomous insights by type                   │
│  Pattern: autonomous_insights:{user_name}:{insight_type}       │
│  Overwrite: Latest insight overwrites previous of same type    │
│                                                                │
│  Insight Types:                                               │
│  • pattern_discovery     → Behavioral pattern analysis        │
│  • autonomous_thinking   → Background thought processes       │
│  • milestone_tracking    → Goal and achievement tracking      │
│  • life_event_detection  → Important life event recognition   │
│  • insight_generation    → Weekly collaborative insights      │
└─────────────────────────────────────────────────────────────────┘
```

## 🚀 **Quick Start**

### **Prerequisites**
```bash
# Required Software
- Python 3.9+
- Node.js 18+
- Docker & Docker Compose
- Git
```

### **1. Clone & Setup**
```bash
git clone https://github.com/yourusername/Autonomous-AgentAI.git
cd Autonomous-AgentAI
```

### **2. Start Databases**
```bash
# Start Redis Stack + Qdrant
docker-compose -f docker-compose.db.yml up -d

# Verify services
curl http://localhost:6379     # Redis
curl http://localhost:6333/health  # Qdrant  
curl http://localhost:8001     # RedisInsight UI
```

### **3. Configure Settings**
```bash
# Edit backend/config/settings.yaml
cp backend/config/settings.example.yaml backend/config/settings.yaml

# Required configurations:
# - Set your user name and description
# - Add API keys for external LLM providers
# - Configure assistant personality
```

### **4. Backend Setup**
```bash
cd backend
pip install -r requirements.txt
python run.py

# Backend runs on: http://localhost:8000
# API docs: http://localhost:8000/docs
```

### **5. Frontend Setup** (Optional)
```bash
cd frontend
npm install
npm run dev

# Frontend runs on: http://localhost:3000
```

### **6. Full Stack Deployment**
```bash
# Start everything with Docker
docker-compose up -d

# Services:
# Backend:  http://localhost:8000
# Frontend: http://localhost:3000  
# Redis UI: http://localhost:8001
# Qdrant:   http://localhost:6333
```

## ⚙️ **Configuration**

### **User Configuration** (`config/settings.yaml`)
```yaml
# User Profile
user:
  name: "John"  # Replace with your actual name
  description: "Software engineer living in India, interested in AI and cloud computing"

assistant:
  name: "Assistant"  # Your AI assistant's name
  description: "Multi-agent AI system with 4 specialized agents: Memory Reader (context retrieval), Memory Writer (fact extraction), Knowledge Agent (research), and Organizer Agent (synthesis). Uses hybrid architecture with 75% local processing."

# External LLM Models (Only for Organizer Agent - 25% usage)
organizer_external_models:
  - "groq/qwen/qwen3-32b"                     # Updated fast inference model
  - "anthropic/claude-3-sonnet-20240229"     # High quality
  - "gemini/gemini-1.5-pro"                  # Google's best
  - "gemini/gemini-1.5-flash"                # Fast Google model

# LangGraph Orchestration Configuration
langgraph:
  checkpoint_storage: "memory"  # Memory-based checkpointing for state persistence
  thread_timeout: 3600         # Thread timeout in seconds (1 hour)
  max_concurrent_threads: 10   # Maximum concurrent workflow threads

# API Keys (Required for external LLM)
providers:
  groq:
    api_key: "gsk_your_groq_key_here"
  anthropic:  
    api_key: "sk-ant-your_anthropic_key_here"
  openai:
    api_key: "sk-your_openai_key_here"
```

### **Database Configuration**
```yaml
databases:
  redis:
    host: "localhost"
    port: 6379
    working_memory_ttl: 604800  # 7 days
    max_working_items: 7
  qdrant:
    host: "localhost" 
    port: 6333
    collection_name: "agent_memories"
    vector_size: 384
    similarity_threshold: 0.7
```

### **Local AI Models** (Automatic Download)
```yaml
transformers:
  cache_dir: "./.models"
  models:
    memory_classifier: "distilbert-base-uncased"
    entity_extractor: "dslim/bert-base-NER"
    summarizer: "sshleifer/distilbart-cnn-6-6"
    embedder: "sentence-transformers/all-MiniLM-L6-v2"
    sentiment_analyzer: "cardiffnlp/twitter-roberta-base-sentiment-latest"
```

## 📡 **API Reference**

### **Core Chat API**
```bash
# Main conversation endpoint
POST /chat
{
  "message": "Tell me about my recent projects",
  "context": {"priority": "high"}  # Optional
}

# Response includes agent metadata and processing info
{
  "response": "Based on your memory...",
  "agent_name": "organizer_agent", 
  "processing_model": "external_llm_only",
  "metadata": {...}
}
```

### **Memory Management**
```bash
GET    /chat/history?limit=50&offset=0    # Conversation history
DELETE /memory/cleanup                    # Clear working + session memory (uses configured user)
GET    /status                           # System & memory statistics
```

### **Autonomous Operations**
```bash
POST   /autonomous/trigger               # Manual autonomous operation
GET    /autonomous/operations           # Available operation types  
GET    /autonomous/history              # Operation execution history
```

### **Autonomous Insights**
```bash
GET    /autonomous/insights              # All insights (uses configured user)
DELETE /autonomous/insights              # Clear all insights (uses configured user)

# System runs autonomous insights every hour automatically
# Insight types: pattern_discovery, autonomous_thinking, 
#                milestone_tracking, life_event_detection, insight_generation
```

### **System Monitoring**
```bash
GET    /health                          # Basic health check
GET    /status                          # Comprehensive system status
```

### **WebSocket Real-time**
```bash
WS     /stream                          # Unified real-time updates
# Messages: connection status, chat responses, autonomous insights, thinking updates
```

## 🎛️ **Working Memory Structure**

### **User Operations** (Manual Chat)
```
working_memory:YourName:memory_reader     ← User context retrieval
working_memory:YourName:knowledge_agent   ← User research context
```

### **Autonomous Operations** (Background AI)
```
working_memory:Assistant:memory_reader    ← Autonomous context
working_memory:Assistant:organizer_agent  ← Autonomous synthesis
```

### **Agent Isolation**
- Each agent maintains separate working memory per user
- 7-day TTL with activity-based extension
- Automatic cleanup and memory management
- Privacy isolation between users

## 📊 **System Benefits**

### **Cost Efficiency**
- **75% Local Processing**: Significant API cost reduction
- **25% External LLM**: Only for complex synthesis requiring advanced reasoning
- **Intelligent Routing**: Skip unnecessary external API calls
- **Multi-provider Fallback**: Cost optimization across providers

### **Performance & Scalability**
- **Parallel Agent Execution**: Concurrent memory and knowledge processing
- **Vector Search**: Sub-millisecond semantic search across memory
- **Automatic TTL**: Self-managing memory lifecycle
- **LangGraph Orchestration**: Fault-tolerant workflow management

### **Intelligence & Autonomy**
- **Autonomous Thinking**: Background analysis every hour
- **Pattern Discovery**: Behavioral pattern detection every 4 hours  
- **Weekly Insights**: Comprehensive intelligence reports
- **Real-time Streams**: Instant insight broadcasting

### **Privacy & Security**
- **User Isolation**: All data segregated by user_name
- **Local Processing**: Sensitive operations processed locally
- **Knowledge Agent Privacy**: No access to personal memory
- **Dedicated Storage**: Separate insight storage per user

## 🔧 **Development**

### **Project Structure**
```
Autonomous-AgentAI/
├── backend/                   # Python FastAPI backend
│   ├── agents/               # 4 LangChain agent implementations
│   ├── api/                  # FastAPI endpoints & WebSocket
│   ├── config/               # Settings and configuration
│   ├── core/                 # LangGraph orchestrator & transformers
│   ├── memory/               # 3-tier memory system
│   ├── tools/                # External tools (Wikipedia, Wikidata)
│   └── utils/                # Utilities and helpers
├── frontend/                 # Next.js React frontend
│   └── src/components/       # UI components
├── data/                     # Persistent data storage
│   ├── qdrant/               # Vector database files
│   └── redis/                # Redis persistence
└── docker-compose.yml       # Full deployment configuration
```

### **Adding New Agents**
```python
# 1. Create new agent in agents/
class NewLangChainAgent:
    def __init__(self, config, transformers_service):
        self.tools = self._create_tools()
    
# 2. Register with LangGraph orchestrator
# 3. Add to memory access matrix
# 4. Update configuration
```

### **Extending Memory System**
```python
# Add new memory tier
class CustomMemoryTier:
    async def store(self, user_name: str, content: str, metadata: dict)
    async def search(self, user_name: str, query: str) -> List[dict]
    async def cleanup(self, user_name: str) -> int
```

## 🚀 **Production Deployment**

### **Environment Variables**
```bash
# Production environment
export ENVIRONMENT=production
export API_HOST=0.0.0.0
export API_PORT=8000
export REDIS_URL=redis://redis-cluster:6379
export QDRANT_URL=http://qdrant-cluster:6333
```

### **Docker Production Setup**
```yaml
# docker-compose.prod.yml
services:
  backend:
    build: ./backend
    environment:
      - ENVIRONMENT=production
    deploy:
      replicas: 3
      
  redis:
    image: redis/redis-stack:latest
    deploy:
      placement:
        constraints: [node.role == manager]
        
  qdrant:
    image: qdrant/qdrant:latest
    deploy:
      placement:
        constraints: [node.role == worker]
```

### **Monitoring & Health Checks**
```bash
# Health endpoints
curl http://localhost:8000/health      # Backend health
curl http://localhost:6333/health      # Qdrant health  
redis-cli -p 6379 ping               # Redis health

# Metrics collection
GET /status  # Comprehensive system metrics
```

## 🤝 **Contributing**

1. **Fork the repository**
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Make changes and test**: Ensure all agents and memory tiers work
4. **Commit changes**: `git commit -m 'Add amazing feature'`  
5. **Push to branch**: `git push origin feature/amazing-feature`
6. **Open Pull Request**: Describe your changes and benefits

### **Development Guidelines**
- Follow existing code structure and naming conventions
- Add comprehensive tests for new agents or memory features
- Update documentation for API changes
- Ensure privacy and security compliance
- Test with multiple LLM providers

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **LangChain**: Powerful agent framework and LLM integrations
- **LangGraph**: State-based workflow orchestration  
- **FastAPI**: High-performance async Python web framework
- **Redis Stack**: In-memory database with vector search
- **Qdrant**: Vector database for permanent storage
- **Next.js**: React framework for modern web applications

---

**Built with ❤️ for intelligent personal assistance and autonomous reasoning.**

*Transform your interactions with AI through sophisticated multi-agent architecture and autonomous intelligence.*
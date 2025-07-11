# ⚡ Simple Autonomous Agentic AI System [CAUTION: Under Development]
**The Future of Proactive AI Assistance** - A sophisticated 3-agent autonomous system with continuous intelligence, life event planning, and real-time collaboration.

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![Next.js](https://img.shields.io/badge/Next.js-15-black.svg)](https://nextjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green.svg)](https://fastapi.tiangolo.com)
[![AutoGen](https://img.shields.io/badge/AutoGen-0.2.23+-purple.svg)](https://microsoft.github.io/autogen/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📋 Table of Contents
- [🌟 Overview](#-overview)
- [🧠 3-Agent Architecture](#-3-agent-architecture)
- [🗃️ 5-Layer Memory System](#️-5-layer-memory-system)
- [🔄 Autonomous Features](#-autonomous-features)
- [🚀 Quick Start](#-quick-start)
- [🔗 API Endpoints](#-api-endpoints)
- [⚙️ Configuration](#️-configuration)
- [🛠️ Project Structure](#️-project-structure)
- [🧪 Testing](#-testing)
- [🤝 Contributing](#-contributing)

## 🌟 Overview

The Autonomous Agentic AI System represents a breakthrough in AI assistant technology, featuring:

### ✨ **Key Innovations**
- **🎭 3-Agent Autonomous System**: Memory, Research, and Intelligence agents with AutoGen GroupChat coordination
- **🧠 Continuous Intelligence**: Hourly thinking cycles and weekly insight generation
- **🗃️ 5-Layer Memory Architecture**: Advanced memory management with Redis + Qdrant
- **📅 Life Event Planning**: Automated timeline tracking for pregnancy, learning, health milestones
- **⚡ Real-time Streams**: WebSocket thinking streams and autonomous insight broadcasts
- **🛡️ Privacy Protection**: Research Agent cannot access personal data

### 🎯 **System Benefits**
✅ **75% Code Reduction**: Simplified from complex manual coordination to autonomous GroupChat  
✅ **Continuous Intelligence**: Hourly thinking cycles with pattern discovery  
✅ **Life Event Planning**: Automated milestone tracking and proactive recommendations  
✅ **Privacy Protection**: Research Agent cannot access personal data  
✅ **Real-time Insights**: Live thinking streams and autonomous intelligence broadcasts  

## 🧠 3-Agent Architecture

### **Agent Specialization Matrix**

| Agent | Role | Memory Access | Key Capabilities |
|-------|------|---------------|------------------|
| **Memory Agent** | User Interface Hub | ✅ **Full Access** | Chat handling, memory management, user context |
| **Research Agent** | External Knowledge | ❌ **No Personal Data** | Web search, fact verification, current events |
| **Intelligence Agent** | Autonomous Thinking | ✅ **Full Access** | Pattern discovery, life planning, continuous reasoning |

### **AutoGen GroupChat Coordination**

```python
# Intelligent agent coordination with automatic speaker selection
self.group_chat = GroupChat(
    agents=[memory_agent, research_agent, intelligence_agent],
    speaker_selection_method="auto",
    max_round=10,
    allow_repeat_speaker=False
)
```

**Features:**
- **Smart Routing**: Determines when to use collaborative vs. direct responses
- **Context-Aware Selection**: Routes messages based on content analysis
- **Performance Tracking**: Monitors collaboration effectiveness
- **API Compatibility**: 100% backward compatibility maintained

## 🗃️ 5-Layer Memory System

### **Memory Architecture**

```
┌─────────────────┐
│ WORKING MEMORY  │ ← Redis (7-item limit, activity-based TTL)
│ (Redis)         │
└─────────────────┘
         ▲
         │
┌─────────────────┐
│ LONG-TERM       │ ← Qdrant (Vector database with semantic search)
│ MEMORY          │
│ (Qdrant)        │
│                 │
│ • EPISODIC      │ ← Personal experiences, conversations
│ • SEMANTIC      │ ← Facts, knowledge, preferences
│ • PROCEDURAL    │ ← Skills, how-to, patterns
│ • PROSPECTIVE   │ ← Goals, plans, intentions
└─────────────────┘
```

### **Memory Types & Usage**

1. **WORKING (Redis)**: Recent context, conversations, temporary data
2. **EPISODIC (Qdrant)**: Personal experiences, conversation history, life events
3. **SEMANTIC (Qdrant)**: Facts, knowledge, user preferences, insights
4. **PROCEDURAL (Qdrant)**: Skills, learned behaviors, decision patterns
5. **PROSPECTIVE (Qdrant)**: Goals, plans, future intentions, timelines

### **AI-Powered Memory Management**

- **Local Classification**: DeBERTa models for fast, private content analysis
- **Intelligent Filtering**: AI-powered importance scoring (0.6 threshold)
- **Automatic Categorization**: Content classified into appropriate memory types
- **Semantic Search**: Vector-based retrieval with similarity scoring

## 🔄 Autonomous Features

### **Continuous Intelligence Cycles**

**Hourly Autonomous Thinking:**
```python
async def autonomous_thinking_cycle(self):
    # Gather user data and patterns
    analysis_data = await self._gather_thinking_data()
    
    # Discover behavioral patterns
    patterns = await self._discover_patterns(analysis_data)
    
    # Generate actionable insights
    insights = await self._generate_insights(patterns)
    
    # Update life event timelines
    await self._update_strategic_plans(insights)
```

**Scheduled Operations:**
- **⏰ Hourly**: Autonomous thinking cycles and pattern discovery
- **📅 Daily 8 AM**: Life event milestone checks and updates
- **🌙 Daily 2 AM**: Memory consolidation and optimization
- **📊 Weekly Sunday 9 AM**: Comprehensive insight generation

### **Life Event Planning Examples**

**🤰 Pregnancy Timeline:**
```
User: "My wife is pregnant"
→ Intelligence Agent creates 40-week milestone timeline
→ Weekly reminders: prenatal appointments, tests, preparations
→ Proactive recommendations: vitamins, diet, childbirth classes
→ Adaptive planning based on progress and preferences
```

**📚 Learning Journey:**
```
User: "I want to learn AI"
→ Research Agent finds current learning resources
→ Intelligence Agent creates structured learning timeline
→ Memory Agent tracks progress and preferences
→ Weekly progress check-ins and adaptive recommendations
```

**🏥 Health Milestones:**
```
User: "Started new fitness routine"
→ Intelligence Agent tracks exercise patterns
→ Weekly progress analysis and recommendations
→ Health milestone celebrations and adjustments
→ Integration with other life goals and events
```

## 🚀 Quick Start

### **Prerequisites**
- **Python 3.12+**
- **Node.js 18+**
- **Docker & Docker Compose**

### **1. Clone & Setup**
```bash
git clone <repository-url>
cd autonomous-agent-ai
cp backend/config/settings.example.yaml backend/config/settings.yaml
```

### **2. Configure API Keys**
Edit `backend/config/settings.yaml`:

```yaml
providers:
  groq:
    api_key: "gsk_your_groq_key_here"
  openai:
    api_key: "sk_your_openai_key_here"
  anthropic:
    api_key: "sk-ant-your_anthropic_key_here"
  gemini:
    api_key: "your_gemini_key_here"

model_categories:
  fast: ["groq/qwen/qwen3-32b", "gemini/gemini-1.5-flash"]
  balanced: ["groq/qwen/qwen3-32b", "anthropic/claude-3-haiku"]
  quality: ["anthropic/claude-3-sonnet", "openai/gpt-4o"]
  premium: ["anthropic/claude-3-opus", "openai/gpt-4o"]

ai_functions:
  chat: "balanced"      # User conversations
  reasoning: "quality"  # Complex analysis
  memory: "fast"        # Memory operations
  autonomous: "premium" # Autonomous thinking cycles
```

### **3. Start Dependencies**
```bash
# Start Redis for working memory
docker run -d -p 6379:6379 --name redis redis:7-alpine

# Start Qdrant for long-term memory
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant:latest
```

### **4. Launch System**
```bash
# Start Autonomous AI System
cd backend
python run.py
```

### **5. Access Application**
- **🌐 Frontend Dashboard**: `http://localhost:3000`
- **⚡ Backend API**: `http://localhost:8000`
- **📚 API Documentation**: `http://localhost:8000/docs`

## 🔗 API Endpoints

### **Core System Endpoints**

#### **Health & Status**
- `GET /health` - Overall system health check
- `GET /agents/status` - 3-agent system status
- `GET /system/metrics` - System performance metrics

#### **Chat & Conversation**
- `POST /chat` - Main chat endpoint (routes through 3-agent orchestrator)
- `GET /chat/history` - Recent chat history

#### **Memory System**
- `GET /memory/insights` - Memory analytics and insights
- `GET /memory/search` - Search user memory with query

#### **Autonomous Intelligence**
- `POST /autonomous/thinking` - Manually trigger autonomous thinking cycle
- `GET /agent` - Basic agent system information

### **WebSocket Endpoints**

#### **Real-time Streams**
- `WS /thinking/stream` - Real-time autonomous thinking stream
- `WS /agent-stream` - Real-time agent communication stream

### **Example Usage**

```bash
# System health check
curl http://localhost:8000/health

# Chat with autonomous system
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "My name is raj and I like to learn lot in AI models", "user_id": "admin"}'

# Get memory insights
curl http://localhost:8000/memory/insights?user_id=admin

# Search memories
curl "http://localhost:8000/memory/search?query=learning&user_id=admin"

# Trigger autonomous thinking
curl -X POST http://localhost:8000/autonomous/thinking

# WebSocket connection
ws://localhost:8000/thinking/stream?user_id=admin
```

## ⚙️ Configuration

### **Category-Based Model Routing**

The system uses flexible model categories for different AI functions:

```yaml
model_categories:
  fast: ["groq/qwen/qwen3-32b", "gemini/gemini-1.5-flash"]
  balanced: ["groq/qwen/qwen3-32b", "anthropic/claude-3-haiku"]
  quality: ["anthropic/claude-3-sonnet", "openai/gpt-4o"]
  premium: ["anthropic/claude-3-opus", "openai/gpt-4o"]

ai_functions:
  chat: "balanced"      # User conversations
  reasoning: "quality"  # Complex analysis
  memory: "fast"        # Memory operations
  autonomous: "premium" # Autonomous thinking cycles
```

### **Privacy & Security Settings**

```yaml
memory:
  intelligent_filtering:
    use_ai_analysis: true
    storage_threshold: 0.6  # Importance threshold for long-term storage
  
development:
  debug_mode: false
  verbose_logging: false

research:
  tavily_api_key: "your_tavily_key_here"  # For Research Agent web search
```

## 🛠️ Project Structure

```
/
├── backend/
│   ├── agents/                    # 3-Agent implementation
│   │   ├── autonomous_memory_agent.py      # Memory & UI hub
│   │   ├── autonomous_research_agent.py    # External knowledge
│   │   └── autonomous_intelligence_agent.py # Autonomous thinking
│   ├── api/                       # FastAPI application
│   │   ├── autonomous_main.py              # Main API with AutoGen
│   │   └── websocket.py                    # Real-time communication
│   ├── core/                      # Core system components
│   │   ├── autonomous_orchestrator.py      # 3-agent coordination
│   │   ├── autonomous_scheduler.py         # Autonomous scheduling
│   │   ├── config.py                       # Configuration management
│   │   └── transformers_service.py         # Local AI processing
│   ├── memory/                    # 5-layer memory system
│   │   ├── autonomous_memory.py            # Memory orchestration
│   │   ├── redis_memory.py                 # Working memory
│   │   ├── qdrant_memory.py                # Long-term memory
│   │   └── memory_types.py                 # Shared memory types
│   ├── config/
│   │   ├── settings.yaml                   # Main configuration
│   │   └── settings.example.yaml           # Configuration template
│   └── run.py                     # Application entry point
├── frontend/
│   ├── src/
│   │   ├── app/page.tsx                    # Main chat interface
│   │   ├── components/                     # UI components
│   │   ├── hooks/                          # React hooks
│   │   ├── types/                          # TypeScript types
│   │   └── utils/                          # Utility functions
│   └── package.json
├── CLAUDE.md                      # Claude Code integration guide
└── README.md                      # This file
```

## 🧪 Testing

### **System Health Check**
```bash
# Basic system validation
curl http://localhost:8000/health

# Agent status verification
curl http://localhost:8000/agents/status
```

### **Memory System Test**
```bash
# Test memory storage and retrieval
curl -X GET "http://localhost:8000/memory/search?query=test&user_id=admin"
```

### **Autonomous Intelligence Test**
```bash
# Manually trigger thinking cycle
curl -X POST http://localhost:8000/autonomous/thinking
```

### **WebSocket Test**
```bash
# Test WebSocket connections
ws://localhost:8000/thinking/stream?user_id=admin
ws://localhost:8000/agent-stream?user_id=admin
```

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **🧪 Testing**: Add tests for new features and ensure existing tests pass
2. **📝 Documentation**: Update documentation for any new features or changes
3. **🔧 Configuration**: Follow the category-based configuration pattern
4. **🛡️ Security**: Never commit API keys or sensitive information
5. **⚡ Performance**: Consider performance implications of new features

### **Development Setup**
```bash
# Install Python dependencies
cd backend
pip install -r requirements.txt

# Install frontend dependencies
cd frontend
npm install

# Start development servers
# Backend: python run.py
# Frontend: npm run dev
```

## 📊 System Monitoring

### **Health Endpoints**
- `GET /health` - Overall system status
- `GET /agents/status` - 3-agent system status  
- `GET /system/metrics` - Performance metrics
- `GET /memory/insights` - Memory analytics

### **Real-time Monitoring**
- **WebSocket**: `/thinking/stream` - Real-time thinking processes
- **WebSocket**: `/agent-stream` - Agent communication streams

## 🔗 Additional Resources

- **[CLAUDE.md](CLAUDE.md)** - Claude Code integration and usage guide
- **[AutoGen Documentation](https://microsoft.github.io/autogen/)** - Multi-agent framework
- **[FastAPI Documentation](https://fastapi.tiangolo.com/)** - API framework
- **[Next.js Documentation](https://nextjs.org/docs)** - Frontend framework

---

**The Autonomous Agentic AI System represents the future of proactive AI assistance.** 🚀

Built with ❤️ using AutoGen, FastAPI, Next.js, Redis, and Qdrant.
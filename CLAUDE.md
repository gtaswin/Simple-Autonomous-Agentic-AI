# ⚡ Autonomous Agentic AI System - Claude Code Integration

This document provides comprehensive guidance for using and configuring the Autonomous Agentic AI System, with specific focus on Claude integration and the 3-agent architecture.

## 📋 Table of Contents
- [🏗️ System Overview](#️-system-overview)
- [⚙️ Configuration Guide](#️-configuration-guide)
- [🧠 3-Agent Architecture](#-3-agent-architecture)
- [🚀 Quick Start](#-quick-start)
- [🔄 Autonomous Features](#-autonomous-features)
- [🛡️ Security & Privacy](#️-security--privacy)
- [🧪 Testing & Validation](#-testing--validation)
- [🐛 Troubleshooting](#-troubleshooting)

## 🏗️ System Overview

The Autonomous Agentic AI System is a sophisticated 3-agent system designed for continuous autonomous reasoning, proactive intelligence, and life event planning. The system features:

### Core Architecture Components
- **⚡ Autonomous Orchestrator**: 3-agent coordination with autonomous GroupChat
- **🎭 Specialized Agents**: Memory (UI hub), Research (external knowledge), Intelligence (autonomous thinking)
- **🗃️ 5-Layer Memory System**: Working (Redis) + 4 persistent types (Qdrant) with autonomous management
- **🧠 Continuous Intelligence**: Hourly thinking cycles and weekly insight generation
- **🎯 Life Event Planning**: Pregnancy timelines, learning journeys, health milestones

### Key Features
- **Autonomous Thinking**: Continuous background analysis and pattern discovery
- **Proactive Insights**: Weekly collaborative intelligence generation
- **Life Event Planning**: Automated milestone tracking and timeline management
- **Real-time Streams**: WebSocket thinking streams and autonomous insight broadcasts
- **Privacy Protection**: Research Agent cannot access personal data

## ⚙️ Configuration Guide

### Starting the System

```bash
# Start dependencies
docker run -d -p 6379:6379 --name redis redis:7-alpine
docker run -d -p 6333:6333 --name qdrant qdrant/qdrant:latest

# Start Autonomous AI System
cd backend
python run.py
```

### Model Configuration

```yaml
# config/settings.yaml
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

providers:
  anthropic:
    api_key: "sk-ant-your_key_here"
  groq:
    api_key: "gsk_your_key_here"
  openai:
    api_key: "sk-your_key_here"
```

## 🧠 3-Agent Architecture

### Agent Roles and Capabilities

| Agent | Role | Memory Access | Capabilities |
|-------|------|---------------|-------------|
| **Memory Agent** | User Interface Hub | ✅ Full Access | Chat handling, memory management, user context |
| **Research Agent** | External Knowledge | ❌ No Personal Data | Web search, fact verification, current events |
| **Intelligence Agent** | Autonomous Thinking | ✅ Full Access | Pattern discovery, life planning, continuous reasoning |

### Memory System Integration

**Memory Types:**
- **WORKING** (Redis): Recent context, 7-item limit, activity-based TTL
- **EPISODIC** (Qdrant): Personal experiences, conversations, events
- **SEMANTIC** (Qdrant): Facts, knowledge, insights, preferences  
- **PROCEDURAL** (Qdrant): Skills, how-to, decision patterns
- **PROSPECTIVE** (Qdrant): Goals, plans, future intentions

## 🚀 Quick Start

### 1. System Health Check
```bash
curl http://localhost:8000/health
```

### 2. Test Chat Endpoint
```bash
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello", "user_id": "admin"}'
```

### 3. Check Agent Status
```bash
curl http://localhost:8000/agents/status
```

### 4. Trigger Autonomous Thinking
```bash
curl -X POST http://localhost:8000/autonomous/thinking
```

## 🔄 Autonomous Features

### Continuous Intelligence Cycles

**Hourly Thinking Cycles:**
- Pattern discovery in user interactions
- Behavioral analysis and insights
- Goal progress tracking
- Memory consolidation

**Weekly Insight Generation:**
- Multi-agent collaborative analysis
- Life event milestone updates
- Proactive recommendations
- Long-term planning adjustments

### Life Event Planning Examples

**Pregnancy Timeline:**
```
User: "My wife is pregnant"
→ Intelligence Agent creates 40-week timeline
→ Weekly milestone reminders (prenatal appointments, tests)  
→ Proactive recommendations (prenatal vitamins, diet)
→ Adaptive planning based on progress
```

**Learning Journey:**
```
User: "I want to learn AI"
→ Research Agent finds current learning resources
→ Intelligence Agent creates structured learning timeline
→ Memory Agent tracks progress and preferences
→ Weekly progress check-ins and recommendations
```

## 🛡️ Security & Privacy

### Privacy Protection
- **Research Agent**: Cannot access personal data (episodic, prospective memory)
- **Memory Agent**: Cannot access external internet
- **Intelligence Agent**: Full memory access for autonomous analysis

### Data Isolation
- Personal conversations stored only in episodic memory
- External research results stored separately in semantic memory
- User preferences and goals protected from research queries

## 🧪 Testing & Validation

### Basic System Test
```python
from api.autonomous_main import AutonomousComponentManager
from core.autonomous_orchestrator import AutonomousOrchestrator
from agents.autonomous_memory_agent import AutonomousMemoryAgent

print("✅ All autonomous components verified!")
```

### Memory System Test
```bash
# Test memory storage and retrieval
curl -X GET "http://localhost:8000/memory/search?query=test&user_id=admin"
```

### Autonomous Thinking Test
```bash
# Manually trigger thinking cycle
curl -X POST http://localhost:8000/autonomous/thinking
```

## 🐛 Troubleshooting

### Common Issues

**Dependencies Not Running:**
```bash
# Check Redis
redis-cli ping

# Check Qdrant
curl http://localhost:6333
```

**Import Errors:**
```bash
# Verify Python path
cd backend
python -c "from core.config import AssistantConfig; print('✅ Config OK')"
```

**API Errors:**
```bash
# Check logs
python run.py
# Look for initialization errors and missing API keys
```

### System Management
- **Start**: `python run.py`
- **Stop**: `Ctrl+C` or `pkill -f run.py`
- **Health Check**: `curl http://localhost:8000/health`
- **Agent Status**: `curl http://localhost:8000/agents/status`

## 📊 System Monitoring

### Health Endpoints
- `GET /health` - Overall system status
- `GET /agents/status` - 3-agent system status  
- `GET /system/metrics` - Performance metrics
- `GET /memory/insights` - Memory analytics

### Real-time Monitoring
- **WebSocket**: `/thinking/stream` - Real-time thinking processes
- **WebSocket**: `/agent-stream` - Agent communication streams

---

## 🎯 Final API Endpoints (Post-Cleanup)

After comprehensive cleanup and optimization, the system now provides these **11 core endpoints**:

### **Core API Endpoints (8 endpoints)**
```bash
GET  /health                   # System health
POST /chat                     # Main chat interface  
GET  /chat/history            # Conversation history
GET  /agents/status           # Complete agent system status
GET  /memory/insights         # Memory analytics
GET  /memory/search           # Memory search
GET  /system/metrics          # System performance
POST /autonomous/thinking     # Manual thinking trigger
```

### **WebSocket Streams (2 endpoints)**
```bash
WS /thinking/stream           # Real-time thinking
WS /agent-stream             # Agent communication  
```

### **Agent Info (1 endpoint)**
```bash
GET /agent                    # Basic system info
```

## 🎯 Key Benefits

✅ **75% Code Reduction**: Simplified from complex manual coordination to autonomous GroupChat  
✅ **Continuous Intelligence**: Hourly thinking cycles with pattern discovery  
✅ **Life Event Planning**: Automated milestone tracking and proactive recommendations  
✅ **Privacy Protection**: Research Agent cannot access personal data  
✅ **Real-time Insights**: Live thinking streams and autonomous intelligence broadcasts  

## 🔧 Development with Claude Code

### Recommended Commands
```bash
# System health verification
curl http://localhost:8000/health

# Test autonomous thinking
curl -X POST http://localhost:8000/autonomous/thinking

# Monitor agent status
curl http://localhost:8000/agents/status

# Check memory insights
curl http://localhost:8000/memory/insights?user_id=admin

# Search memories
curl "http://localhost:8000/memory/search?query=learning&user_id=admin"
```

### Configuration Validation
```bash
# Validate configuration
python -c "from core.config import AssistantConfig; config = AssistantConfig(); print('✅ All configurations valid')"

# Test memory system
python -c "from memory.autonomous_memory import AutonomousMemorySystem; print('✅ Memory system OK')"

# Test transformers service
python -c "from core.transformers_service import get_transformers_service; print('✅ TransformersService OK')"
```

### Real-time Monitoring
```bash
# WebSocket connections for live monitoring
ws://localhost:8000/thinking/stream?user_id=admin
ws://localhost:8000/agent-stream?user_id=admin
```

## 🚀 Production Deployment

### Docker Deployment
```bash
# Start all services
docker-compose up -d

# Check service health
docker-compose ps
```

### Environment Variables
```bash
# Set production API keys
export GROQ_API_KEY="gsk_your_production_key"
export OPENAI_API_KEY="sk_your_production_key"
export ANTHROPIC_API_KEY="sk-ant-your_production_key"
```

### Monitoring & Logs
```bash
# Monitor system logs
docker-compose logs -f backend

# Check memory usage
curl http://localhost:8000/system/metrics
```

---

**The Autonomous Agentic AI System represents the future of proactive AI assistance.** 🚀

**Built with ❤️ for the Claude Code community using AutoGen, FastAPI, Next.js, Redis, and Qdrant.**
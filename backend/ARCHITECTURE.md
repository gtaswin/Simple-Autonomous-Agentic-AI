# 🧠 Autonomous AgentAI System Architecture

**4-Agent Hybrid System** with LangGraph orchestration, 3-tier memory, and 75% local processing.

## 🌟 System Overview

Cost-effective autonomous AI system with 4 specialized agents:
- **75% Local Processing**: Memory Reader, Memory Writer, Knowledge agents use local transformers
- **25% External LLM**: Only Organizer agent uses external APIs for synthesis
- **3-Tier Memory**: Working, Short-term (TTL), Long-term (permanent)
- **LangGraph Orchestration**: State-based workflow with conditional routing
- **Pure LangChain Framework**: All agents use AgentExecutor/initialize_agent patterns

## 🎯 4-Agent Architecture

### **Agent Specialization & Memory Access**

| Agent | Processing | Memory Access | Key Role |
|-------|------------|---------------|----------|
| **Memory Reader** | LOCAL | Read: All memories<br>Write: Own working memory | Context retrieval & summarization |
| **Memory Writer** | LOCAL | Write: Short/Long-term<br>No working memory | Fact extraction & storage |
| **Knowledge** | LOCAL | Read/Write: Own working memory only | External research (no personal data) |
| **Organizer** | EXTERNAL LLM | Read/Write: Own working memory<br>Memory via Memory Reader | Response synthesis |

### **Memory Access Rules**
- **Memory Reader**: Reads all memory types → processes → provides context summary
- **Memory Writer**: Stateless - only writes facts to memory (no context needed)
- **Knowledge Agent**: Privacy-protected - no personal memory access
- **Organizer Agent**: Gets memory context from Memory Reader (no direct access)

### **LangChain Framework Implementation**

| Agent | Base Class | LangChain Pattern | Local LLM Wrapper |
|-------|------------|------------------|-------------------|
| **Memory Reader** | `BaseAgent` | AgentExecutor + ReAct | `LocalTransformerLLM` |
| **Memory Writer** | `StatelessBaseAgent` | AgentExecutor + ReAct | `LocalTransformerLLM` |
| **Knowledge** | `BaseAgent` | AgentExecutor + ReAct | `LocalTransformerLLM` |
| **Organizer** | `BaseAgent` | initialize_agent | External LLM via config |

**Base Agent Architecture:**
```python
# StatelessBaseAgent: No working memory access
class StatelessBaseAgent(ABC):
    # LangChain message handling, error handling, config access
    
# BaseAgent: With working memory access  
class BaseAgent(StatelessBaseAgent):
    async def get_own_working_memory()     # Read own working memory
    async def store_working_memory()       # Write own working memory
    async def clear_own_working_memory()   # Clear own working memory
```

## 💾 4-Tier Memory System

### **1. Session Memory** (Redis Lists)
```
session_memory:{user_name}
```
- Complete conversation history
- 50 conversations max (auto-trim)
- No TTL (permanent until cleanup)

### **2. Working Memory** (Redis Lists + TTL)
```
working_memory:{user_name}:{agent_name}
```
- Agent-specific context and scratchpad
- 7 items max per agent per user
- 7-day TTL with activity extension

### **3. Short-term Memory** (Redis Vector + TTL)
```
shortterm_memory:{memory_id}
```
- Semantic search with embeddings
- Importance-based TTL:
  - High (0.7-1.0): 3 months
  - Medium (0.5-0.7): 1 month  
  - Low (0.3-0.5): 1 week

### **4. Long-term Memory** (Qdrant Vector)
```
collection: agent_memories
```
- Permanent user knowledge base
- Only facts with importance ≥ 0.9
- Semantic search with user filtering

## 🔄 Agent Workflow Patterns

### **Simple Query** (75% of interactions)
```
User Input → Router → Memory Reader → Organizer → Memory Writer → Response
```

### **Complex Query** (25% of interactions) 
```
User Input → Router → Memory Reader → Knowledge Agent → Organizer → Memory Writer → Response
                              ↓              ↓
                      [Parallel Execution]
```

### **Agent Processing Details**

#### **Memory Reader Agent** (LangChain AgentExecutor + ReAct)
```python
@structured_output(AgentType.MEMORY_READER)
async def process(user_name: str, query: str) -> MemoryReaderOutput:
    # Pure LangChain AgentExecutor with memory tools:
    # 1. Tool: search_short_term_memory (Redis Vector)
    # 2. Tool: search_long_term_memory (Qdrant)
    # 3. ReAct pattern: Thought → Action → Observation → Final Answer
    # 4. LocalTransformerLLM for processing
    # 5. Return structured context summary (NOT raw memories)
```

#### **Knowledge Agent** (LangChain AgentExecutor + ReAct)
```python
@structured_output(AgentType.KNOWLEDGE_AGENT)
async def process(user_name: str, query: str) -> KnowledgeAgentOutput:
    # Pure LangChain AgentExecutor with research tools:
    # 1. Tool: WikipediaQueryRun (external research)
    # 2. Tool: local_summarize (local transformers)
    # 3. ReAct pattern for systematic research
    # 4. Store findings in own working memory
    # 5. Return structured research summary
```

#### **Organizer Agent** (LangChain initialize_agent)
```python
@structured_output(AgentType.ORGANIZER_AGENT)
async def process(user_name: str, user_input: str, 
                memory_context: Dict, knowledge_context: Dict) -> OrganizerAgentOutput:
    # Pure LangChain initialize_agent with external LLM:
    # 1. Tool: store_working_memory (context continuity)
    # 2. LangChain ChatPromptTemplate for synthesis
    # 3. External LLM via config.call_llm_with_fallback()
    # 4. Generate personalized response using provided contexts
    # 5. Store synthesis results in own working memory
```

#### **Memory Writer Agent** (LangChain AgentExecutor + ReAct)
```python
@structured_output(AgentType.MEMORY_WRITER)  
async def process(user_name: str, user_message: str, ai_response: str) -> MemoryWriterOutput:
    # Pure LangChain AgentExecutor with fact processing tools:
    # 1. Tool: extract_facts (local NER)
    # 2. Tool: classify_importance (local transformers)
    # 3. ReAct pattern for systematic fact extraction
    # 4. Store facts in appropriate memory tiers (short/long-term)
    # 5. StatelessBaseAgent - no working memory access
```

## 🧠 LangGraph Orchestration

### **Workflow Definition**
```python
workflow = StateGraph(MultiAgentState)

# Core nodes - all use pure LangChain process() methods
workflow.add_node("router", self._router_node)
workflow.add_node("memory_reader", self._memory_reader_node)     # calls agent.process()
workflow.add_node("knowledge_agent", self._knowledge_agent_node) # calls agent.process()
workflow.add_node("organizer", self._organizer_node)             # calls agent.process()
workflow.add_node("memory_writer", self._memory_writer_node)     # calls agent.process()

# Conditional routing
workflow.add_conditional_edges("router", self._should_research)
```

### **State Management**
```python
class MultiAgentState(TypedDict):
    user_name: str
    user_message: str
    memory_context: Optional[MemoryReaderOutput]      # Structured output from Memory Reader
    knowledge_context: Optional[KnowledgeAgentOutput] # Structured output from Knowledge Agent
    final_response: Optional[OrganizerAgentOutput]    # Structured output from Organizer
    memory_write_result: Optional[MemoryWriterOutput] # Structured output from Memory Writer
    agents_executed: List[str]
    complexity_score: float
```

## 💰 Cost Optimization

### **Processing Distribution**
- **Local Transformers (75%)**: $0 per operation
  - Memory Reader: Context summarization
  - Memory Writer: Fact extraction & classification  
  - Knowledge Agent: Research summarization
  
- **External LLM (25%)**: ~$0.01-0.10 per operation
  - Organizer Agent: Response synthesis only

### **Smart Routing**
```python
# Simple queries skip Knowledge Agent
complexity_score = calculate_complexity(message)
should_research = complexity_score > 0.6 or has_research_keywords(message)
```

## 🤖 Autonomous Operations

### **Background Scheduler**
```python
autonomous_operations = {
    "pattern_discovery": 3600,      # Every hour
    "milestone_tracking": 86400,    # Daily  
    "life_event_detection": 43200   # Every 12 hours
}
```

### **Autonomous Workflow**
```
Memory Reader → Organizer → Insight Storage → WebSocket Broadcast
```

## 📡 Real-time Communication

### **WebSocket Messages**
```json
// Chat Response
{
  "type": "chat_response",
  "data": {"response": "...", "processing_time": 2.3}
}

// Autonomous Insight
{
  "type": "autonomous_insight", 
  "data": {"insight_type": "pattern_discovery", "content": "..."}
}
```

## 🛠️ Technology Stack

### **Backend Core**
- **FastAPI**: REST API + WebSocket
- **LangGraph**: Multi-agent orchestration with StateGraph
- **LangChain**: Pure agent framework (AgentExecutor, initialize_agent)
- **Transformers + PyTorch**: Local AI processing (75% of operations)
- **Sentence-Transformers**: Local embeddings and similarity
- **Redis Stack**: Working/Short-term memory + vector search
- **Qdrant**: Long-term permanent vector storage

### **LangChain Components**
- **LocalTransformerLLM**: LangChain wrapper for local models
- **AgentExecutor**: ReAct pattern for Memory Reader/Writer/Knowledge
- **initialize_agent**: ZERO_SHOT_REACT for Organizer
- **Structured Outputs**: Pydantic v2 schemas with @structured_output
- **Tools**: Wikipedia, local NER, memory search, summarization

### **LLM Integration**
- **LiteLLM**: Multi-provider routing (Organizer agent only)
- **Supported**: Anthropic Claude, OpenAI GPT, Google Gemini, Groq
- **Cost Optimization**: 75% local processing, 25% external LLM

### **Frontend**
- **Next.js + TypeScript**: React framework
- **TailwindCSS**: Styling
- **WebSocket**: Real-time updates

## 🔒 Privacy & Security

### **Data Isolation**
```python
# All operations require user filtering
search_query = f"(@user_name:{user_name}) => [KNN 5 @vector $vec]"

# Qdrant user filtering  
search_filter = {"must": [{"key": "user_name", "match": {"value": user_name}}]}
```

### **Agent Access Controls**
- **Memory Reader** (`BaseAgent`): Reads all memory types + own working memory
- **Memory Writer** (`StatelessBaseAgent`): Writes to memory only (no working memory)
- **Knowledge Agent** (`BaseAgent`): Own working memory only (privacy-protected)
- **Organizer** (`BaseAgent`): Own working memory + context via Memory Reader

## 🚀 Deployment

### **Development**
```bash
# Start dependencies
docker run -d -p 6379:6379 redis/redis-stack:latest
docker run -d -p 6333:6333 qdrant/qdrant:latest

# Start backend
cd backend && python run.py

# Start frontend  
cd frontend && npm run dev
```

### **Production**
```yaml
# Docker Compose production setup
services:
  backend:
    image: autonomous-ai/backend
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
      replicas: 2
```

---

## ✅ Architecture Compliance Summary

### **Pure LangChain Implementation**
- ✅ **Memory Reader**: `LangChainMemoryReaderAgent(BaseAgent)` with AgentExecutor + ReAct
- ✅ **Memory Writer**: `LangChainMemoryWriterAgent(StatelessBaseAgent)` with AgentExecutor + ReAct  
- ✅ **Knowledge Agent**: `LangChainKnowledgeAgent(BaseAgent)` with AgentExecutor + ReAct
- ✅ **Organizer Agent**: `LangChainOrganizerAgent(BaseAgent)` with initialize_agent

### **Memory Access Validation**
- ✅ **Working Memory**: Each agent manages own working memory independently
- ✅ **Memory Writer Stateless**: No working memory access (StatelessBaseAgent)
- ✅ **Privacy Protection**: Knowledge Agent cannot access personal memories
- ✅ **Context Flow**: Memory Reader → processes all memory → provides summary to Organizer

### **Processing Distribution**
- ✅ **75% Local**: Memory Reader/Writer/Knowledge use LocalTransformerLLM
- ✅ **25% External**: Only Organizer uses external LLM via config
- ✅ **Cost Optimization**: Majority of operations run locally at $0 cost

### **LangGraph Integration**
- ✅ **StateGraph Orchestration**: Multi-agent workflow coordination
- ✅ **Structured Outputs**: Pydantic v2 schemas with type safety
- ✅ **Conditional Routing**: Smart research agent activation
- ✅ **Error Handling**: Graceful fallbacks and validation

**This architecture delivers intelligent, cost-effective, and privacy-preserving autonomous AI assistance with pure LangChain framework compliance.** 🧠🔗
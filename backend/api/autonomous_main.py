"""
🚀 Autonomous Agentic AI System - FastAPI Main Application

Clean implementation following README.md, ARCHITECTURE.md, and CLAUDE.md specifications:
- User-centric design: All endpoints use configured user from settings.yaml
- No user_id parameters required in API requests
- Pure LangChain framework compliance with 4-agent architecture
- Structured outputs with Pydantic v2 schemas
- WebSocket real-time communication
"""

import asyncio
import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, Dict, List, Optional

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from core.config import AssistantConfig
from core.langraph_orchestrator_refactored import LangGraphMultiAgentOrchestrator
from core.transformers_service import TransformersService
from core.streaming_manager import StreamingManager
from core.parallel_coordinator import ParallelCoordinator
from memory.autonomous_memory import AutonomousMemorySystem
from utils.websocket_manager import ConnectionManager

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global component manager
component_manager = None

class AutonomousComponentManager:
    """Manages all system components with proper initialization order"""
    
    def __init__(self):
        self.config = None
        self.transformers_service = None
        self.memory_system = None
        self.langraph_orchestrator = None
        self.streaming_manager = None
        self.parallel_coordinator = None
        self.websocket_manager = None
        
    async def initialize_all_components(self):
        """Initialize all components in proper dependency order"""
        try:
            # 1. Load configuration
            logger.info("🔧 Loading system configuration...")
            self.config = AssistantConfig()
            
            # 2. Initialize transformers service
            logger.info("🤖 Loading transformer models from configuration...")
            self.transformers_service = TransformersService(config=self.config)
            
            # 3. Initialize memory system
            logger.info("💾 Initializing 3-tier memory system...")
            self.memory_system = AutonomousMemorySystem(config=self.config)
            await self.memory_system.start()
            
            # 4. Initialize WebSocket and streaming manager
            logger.info("📡 Setting up real-time communication...")
            self.websocket_manager = ConnectionManager()
            self.streaming_manager = StreamingManager()
            self.streaming_manager.set_websocket_manager(self.websocket_manager)
            
            # 5. Initialize parallel coordinator
            logger.info("⚡ Setting up parallel execution coordinator...")
            self.parallel_coordinator = ParallelCoordinator()
            
            # 6. Initialize LangGraph orchestrator with all agents
            logger.info("🧠 Initializing 4-agent LangGraph orchestrator...")
            self.langraph_orchestrator = LangGraphMultiAgentOrchestrator(
                memory_system=self.memory_system,
                config=self.config,
                transformers_service=self.transformers_service
            )
            
            logger.info("✅ All components initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise e

@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI lifespan context manager for startup and shutdown"""
    global component_manager
    
    # Startup
    logger.info("🚀 Starting Autonomous Agentic AI System...")
    logger.info("Architecture: Memory + Research + Intelligence Agents")
    
    component_manager = AutonomousComponentManager()
    await component_manager.initialize_all_components()
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Autonomous Agentic AI System...")

# Initialize FastAPI application
app = FastAPI(
    title="Autonomous Agentic AI System",
    description="4-Agent hybrid system with LangGraph orchestration and 75% local processing",
    version="2.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# PYDANTIC MODELS (Following specification)
# ============================================================================

class ChatRequest(BaseModel):
    """Chat request - uses configured user from settings.yaml"""
    message: str = Field(..., description="User message")
    context: Optional[Dict[str, Any]] = Field(default=None, description="Optional context")

class ChatResponse(BaseModel):
    """Chat response with agent metadata"""
    response: str = Field(..., description="AI response")
    agent_name: str = Field(..., description="Primary responding agent")
    timestamp: str = Field(..., description="Response timestamp")
    user_context: Dict[str, Any] = Field(default_factory=dict, description="User context")
    collaboration_summary: str = Field(..., description="Agent collaboration summary")

class AutomousOperationRequest(BaseModel):
    """Request for autonomous operation"""
    operation_type: str = Field(..., description="Type of autonomous operation")
    trigger_source: str = Field(default="manual", description="Trigger source")
    broadcast_updates: bool = Field(default=True, description="Whether to broadcast updates")

# ============================================================================
# CORE CHAT API (README.md specification)
# ============================================================================

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main conversation endpoint - Uses configured user from settings.yaml
    
    As per README.md:
    POST /chat
    {
      "message": "Tell me about my recent projects",
      "context": {"priority": "high"}  # Optional
    }
    """
    if not component_manager.langraph_orchestrator:
        raise HTTPException(status_code=503, detail="LangGraph orchestrator not initialized")
    
    try:
        start_time = time.time()
        
        # Get configured user from settings.yaml (user-centric design)
        configured_user = component_manager.config.get("user.name", "User")
        
        # Process through LangGraph orchestrator using configured user
        result = await component_manager.langraph_orchestrator.process_message(
            user_name=configured_user,
            message=request.message
        )
        
        response_time = time.time() - start_time
        
        # Broadcast to WebSocket if available
        if component_manager.websocket_manager:
            # Convert Pydantic object to dict for WebSocket broadcasting
            if hasattr(result, 'model_dump'):
                websocket_result = {
                    "response": result.final_response,
                    "metadata": {
                        "agents_executed": result.agents_executed,
                        "workflow_pattern": result.workflow_pattern.value if hasattr(result.workflow_pattern, 'value') else str(result.workflow_pattern),
                        "processing_time": result.total_processing_time_ms / 1000  # Convert to seconds
                    }
                }
            else:
                websocket_result = result
            await component_manager.websocket_manager.broadcast_chat_response(websocket_result)
        
        # Process result based on type (orchestrator returns dict structure)
        if isinstance(result, dict):
            # Standard dict structure from orchestrator
            final_response = result.get("response", "Response generated")
            metadata = result.get("metadata", {})
            agents_executed = metadata.get("agents_executed", [])
            workflow_pattern = metadata.get("workflow_pattern", "standard")
        elif hasattr(result, 'model_dump'):
            # Pydantic WorkflowExecutionOutput object (future structured outputs)
            final_response = result.final_response
            agents_executed = result.agents_executed  # List[str] as per schema
            workflow_pattern = result.workflow_pattern.value if hasattr(result.workflow_pattern, 'value') else str(result.workflow_pattern)
        else:
            # Fallback for unexpected result types
            final_response = str(result)
            agents_executed = []
            workflow_pattern = "unknown"
        
        return ChatResponse(
            response=final_response,
            agent_name="langraph_multi_agent",
            timestamp=datetime.now().isoformat(),
            user_context={"agents_executed": agents_executed},
            collaboration_summary=f"Workflow: {workflow_pattern}, Agents: {', '.join(agents_executed)}"
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

# ============================================================================
# MEMORY MANAGEMENT (README.md specification)
# ============================================================================

@app.get("/chat/history")
async def get_chat_history(limit: int = 50, offset: int = 0):
    """
    Get conversation history - Uses configured user from settings.yaml
    
    As per README.md:
    GET /chat/history?limit=50&offset=0    # Conversation history
    """
    if not component_manager.memory_system:
        raise HTTPException(status_code=503, detail="Memory system not initialized")
    
    try:
        # Get configured user from settings.yaml
        configured_user = component_manager.config.get("user.name", "User")
        
        # Get chat history for configured user
        chat_history = await component_manager.memory_system.redis_layer.get_chat_history(
            user_id=configured_user,
            limit=limit,
            offset=offset
        )
        
        return {
            "history": chat_history,
            "user_id": configured_user,
            "limit": limit,
            "offset": offset,
            "total_count": len(chat_history),
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving chat history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve chat history: {str(e)}")

@app.delete("/memory/cleanup")
async def cleanup_memory():
    """
    Clear working + session memory - Uses configured user from settings.yaml
    
    As per README.md:
    DELETE /memory/cleanup                    # Clear working + session memory (uses configured user)
    """
    if not component_manager.memory_system:
        raise HTTPException(status_code=503, detail="Memory system not initialized")
    
    try:
        # Get configured user from settings.yaml
        configured_user = component_manager.config.get("user.name", "User")
        
        # Cleanup memory for configured user
        cleanup_result = await component_manager.memory_system.cleanup_user_memories(configured_user)
        
        return {
            "status": "success",
            "user_id": configured_user,
            "cleanup_result": cleanup_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Memory cleanup error: {e}")
        raise HTTPException(status_code=500, detail=f"Memory cleanup failed: {str(e)}")

# ============================================================================
# AUTONOMOUS OPERATIONS (README.md specification)
# ============================================================================

@app.post("/autonomous/trigger")
async def trigger_autonomous_operation(request: AutomousOperationRequest):
    """
    Manual autonomous operation trigger
    
    As per README.md:
    POST /autonomous/trigger               # Manual autonomous operation
    """
    if not component_manager.langraph_orchestrator:
        raise HTTPException(status_code=503, detail="LangGraph orchestrator not initialized")
    
    try:
        # Get configured user for autonomous operations
        configured_user = component_manager.config.get("user.name", "User")
        
        # Execute autonomous operation
        result = await component_manager.langraph_orchestrator.execute_autonomous_operation(
            operation_type=request.operation_type,
            trigger_source=request.trigger_source,
            target_user_name=configured_user,
            broadcast_updates=request.broadcast_updates
        )
        
        return {
            "status": "success",
            "operation_type": request.operation_type,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Autonomous operation error: {e}")
        raise HTTPException(status_code=500, detail=f"Autonomous operation failed: {str(e)}")

@app.get("/autonomous/operations")
async def get_autonomous_operations():
    """
    Get available autonomous operation types
    
    As per README.md:
    GET /autonomous/operations           # Available operation types
    """
    return {
        "operations": [
            "pattern_discovery",
            "autonomous_thinking", 
            "milestone_tracking",
            "life_event_detection",
            "insight_generation"
        ],
        "description": "Background autonomous operations run automatically every hour",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/autonomous/history")
async def get_autonomous_history():
    """
    Get autonomous operation execution history
    
    As per README.md:
    GET /autonomous/history              # Operation execution history
    """
    # Get configured user
    configured_user = component_manager.config.get("user.name", "User")
    
    return {
        "user_id": configured_user,
        "history": [],  # Would be populated with actual history
        "description": "Autonomous operation execution history",
        "timestamp": datetime.now().isoformat()
    }

# ============================================================================
# AUTONOMOUS INSIGHTS (README.md specification)
# ============================================================================

@app.get("/autonomous/insights")
async def get_autonomous_insights():
    """
    Get all autonomous insights - Uses configured user from settings.yaml
    
    As per README.md:
    GET /autonomous/insights              # All insights (uses configured user)
    """
    if not component_manager.memory_system:
        raise HTTPException(status_code=503, detail="Memory system not initialized")
    
    try:
        # Get configured user from settings.yaml
        configured_user = component_manager.config.get("user.name", "User")
        
        # Get insights for configured user
        insights = await component_manager.memory_system.redis_layer.get_autonomous_insights(configured_user)
        
        return {
            "user_id": configured_user,
            "insights": insights,
            "insight_types": [
                "pattern_discovery",
                "autonomous_thinking", 
                "milestone_tracking",
                "life_event_detection",
                "insight_generation"
            ],
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving autonomous insights: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve insights: {str(e)}")

@app.delete("/autonomous/insights")
async def clear_autonomous_insights():
    """
    Clear all autonomous insights - Uses configured user from settings.yaml
    
    As per README.md:
    DELETE /autonomous/insights              # Clear all insights (uses configured user)
    """
    if not component_manager.memory_system:
        raise HTTPException(status_code=503, detail="Memory system not initialized")
    
    try:
        # Get configured user from settings.yaml
        configured_user = component_manager.config.get("user.name", "User")
        
        # Clear insights for configured user
        clear_result = await component_manager.memory_system.redis_layer.clear_autonomous_insights(configured_user)
        
        return {
            "status": "success",
            "user_id": configured_user,
            "cleared_insights": clear_result,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error clearing autonomous insights: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to clear insights: {str(e)}")

# ============================================================================
# SYSTEM MONITORING (README.md specification)
# ============================================================================

@app.get("/health")
async def health_check():
    """
    Basic health check
    
    As per README.md:
    GET /health                          # Basic health check
    """
    return {
        "status": "healthy",
        "version": "2.0.0",
        "architecture": "4-agent autonomous",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "langraph_orchestrator": component_manager.langraph_orchestrator is not None,
            "memory_system": component_manager.memory_system is not None,
            "transformers_service": component_manager.transformers_service is not None,
            "websocket_manager": component_manager.websocket_manager is not None
        },
        "agent_system": {
            "name": "autonomous_agentic_ai",
            "coordination": "langgraph",
            "agents": {
                "memory_reader": "Context retrieval & summarization",
                "memory_writer": "Fact extraction & storage", 
                "knowledge_agent": "External research",
                "organizer_agent": "Response synthesis & coordination"
            },
            "features": [
                "4_agent_architecture",
                "75_percent_local_processing", 
                "3_tier_memory_system",
                "real_time_websockets",
                "autonomous_thinking",
                "pattern_discovery"
            ]
        }
    }

@app.get("/status")
async def get_system_status():
    """
    Comprehensive system status
    
    As per README.md:
    GET /status                          # Comprehensive system status
    """
    if not all([component_manager.memory_system, component_manager.transformers_service, component_manager.langraph_orchestrator]):
        raise HTTPException(status_code=503, detail="System components not fully initialized")
    
    try:
        # Get configured user
        configured_user = component_manager.config.get("user.name", "User")
        
        # Gather comprehensive status
        return {
            "system": {
                "status": "operational",
                "version": "2.0.0",
                "architecture": "4-agent LangGraph + LangChain",
                "processing_distribution": "75% local + 25% external LLM"
            },
            "user": {
                "configured_user": configured_user,
                "design": "user_centric_no_parameters"
            },
            "components": {
                "memory_system": "initialized",
                "transformers_service": "loaded",
                "langraph_orchestrator": "operational",
                "websocket_manager": "active"
            },
            "agents": {
                "memory_reader": "LOCAL - Context retrieval",
                "memory_writer": "LOCAL - Fact extraction", 
                "knowledge_agent": "LOCAL - Research",
                "organizer_agent": "EXTERNAL LLM - Synthesis"
            },
            "memory_tiers": {
                "session_memory": "50 conversations max",
                "working_memory": "7 items per agent per user, 7-day TTL",
                "short_term_memory": "Redis Vector + importance-based TTL",
                "long_term_memory": "Qdrant permanent storage"
            },
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error retrieving system status: {e}")
        raise HTTPException(status_code=500, detail=f"Status retrieval failed: {str(e)}")

# ============================================================================
# WEBSOCKET REAL-TIME (README.md specification)
# ============================================================================

@app.websocket("/stream")
async def websocket_unified_stream(websocket: WebSocket):
    """
    Unified real-time WebSocket stream
    
    As per README.md:
    WS /stream                          # Unified real-time updates
    # Messages: connection status, chat responses, autonomous insights, thinking updates
    """
    if not component_manager.websocket_manager:
        await websocket.close(code=1011, reason="WebSocket manager not initialized")
        return
    
    # Get configured user
    configured_user = component_manager.config.get("user.name", "User")
    
    await websocket.accept()
    await component_manager.websocket_manager.connect(websocket, configured_user)
    
    logger.info(f"🔗 Unified WebSocket connected for user: {configured_user}")
    
    try:
        # Send connection confirmation
        await websocket.send_json({
            "type": "connection_status",
            "status": "connected",
            "user": configured_user,
            "features": ["chat_responses", "autonomous_insights", "thinking_updates"],
            "timestamp": datetime.now().isoformat()
        })
        
        # Keep connection alive and handle incoming messages
        while True:
            try:
                # Wait for messages from client
                data = await websocket.receive_json()
                
                # Handle different message types
                if data.get("type") == "ping":
                    await websocket.send_json({
                        "type": "pong",
                        "timestamp": datetime.now().isoformat()
                    })
                    
            except WebSocketDisconnect:
                break
            except Exception as e:
                logger.error(f"WebSocket message error: {e}")
                break
                
    except WebSocketDisconnect:
        logger.info(f"🔗 Unified WebSocket disconnected for user: {configured_user}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await component_manager.websocket_manager.disconnect(websocket)

# ============================================================================
# APPLICATION STARTUP
# ============================================================================

def main():
    """Main application entry point"""
    # Start the FastAPI application
    uvicorn.run(
        "api.autonomous_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
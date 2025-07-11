"""
Autonomous Agentic AI FastAPI Main Application
Integrates 3-Agent autonomous architecture while preserving existing API compatibility
"""

import asyncio
import os
import uuid
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn
import json

# Initialize logger
logger = logging.getLogger(__name__)

# Import utilities
from utils.serialization import safe_json_dumps, prepare_websocket_message

# Import autonomous orchestrator and existing systems
try:
    import time
    from core.autonomous_orchestrator import AutonomousOrchestrator
    from core.config import AssistantConfig
    from memory.autonomous_memory import AutonomousMemorySystem
    from core.transformers_service import TransformersService
    
    # WebSocket manager for real-time communication
    try:
        from api.websocket import AgentStreamManager
        WEBSOCKET_AVAILABLE = True
    except ImportError:
        WEBSOCKET_AVAILABLE = False
    
except ImportError as e:
    logger.critical(f"❌ CRITICAL ERROR: Missing required autonomous system dependencies!")
    logger.critical(f"❌ Error: {e}")
    logger.critical(f"❌ Please install all requirements: pip install -r requirements.txt")
    exit(1)

# Component Manager for Autonomous Architecture
class AutonomousComponentManager:
    """Manages autonomous system component initialization"""
    
    def __init__(self):
        self.config: Optional[AssistantConfig] = None
        self.memory_system: Optional[AutonomousMemorySystem] = None
        self.transformers_service: Optional[TransformersService] = None
        self.autonomous_orchestrator: Optional[AutonomousOrchestrator] = None
        self.websocket_manager: Optional[AgentStreamManager] = None
        
        self.initialized = False
        self.debug_mode = False
        self.verbose_logging = False
        
    async def initialize_all_components(self):
        """Initialize all autonomous system components during startup"""
        if self.initialized:
            return
            
        logger.info("🚀 Starting autonomous system component initialization...")
        start_time = time.time()
        
        try:
            # 1. Configuration
            logger.info("⚙️ Loading configuration...")
            self.config = AssistantConfig()
            self.debug_mode = self.config.get("development.debug_mode", False)
            self.verbose_logging = self.config.get("development.verbose_logging", False)
            
            # 2. Memory System
            logger.info("🧠 Initializing memory system...")
            self.memory_system = AutonomousMemorySystem(config=self.config)
            await self.memory_system.start()
            
            # 3. TransformersService (if enabled)
            try:
                logger.info("🤖 Initializing transformers service...")
                self.transformers_service = TransformersService(config=self.config)
            except Exception as e:
                logger.warning(f"TransformersService initialization failed: {e}")
                self.transformers_service = None
            
            # 4. Autonomous Orchestrator
            logger.info("🎭 Initializing autonomous orchestrator...")
            tavily_api_key = self.config.get("research.tavily_api_key")
            
            self.autonomous_orchestrator = AutonomousOrchestrator(
                memory_system=self.memory_system,
                config=self.config,
                transformers_service=self.transformers_service,
                tavily_api_key=tavily_api_key
            )
            
            # 5. WebSocket Manager
            if WEBSOCKET_AVAILABLE:
                logger.info("🔌 Initializing WebSocket manager...")
                self.websocket_manager = AgentStreamManager()
            
            # Mark as initialized
            self.initialized = True
            
            initialization_time = time.time() - start_time
            logger.info(f"✅ Autonomous system components initialized in {initialization_time:.2f}s")
            
            # Start autonomous thinking cycles
            if self.autonomous_orchestrator:
                logger.info("🧠 Starting autonomous thinking scheduler...")
                await self._start_autonomous_cycles()
                
        except Exception as e:
            logger.error(f"❌ Component initialization failed: {e}")
            raise e
    
    async def _start_autonomous_cycles(self):
        """Start autonomous thinking cycles"""
        try:
            # Schedule autonomous thinking cycles (every hour)
            asyncio.create_task(self._autonomous_thinking_loop())
            logger.info("✅ Autonomous thinking cycles started")
        except Exception as e:
            logger.error(f"Failed to start autonomous cycles: {e}")
    
    async def _autonomous_thinking_loop(self):
        """Background loop for autonomous thinking"""
        while True:
            try:
                await asyncio.sleep(3600)  # Wait 1 hour
                
                if self.autonomous_orchestrator:
                    logger.info("🧠 Running autonomous thinking cycle...")
                    result = await self.autonomous_orchestrator.autonomous_thinking_cycle()
                    
                    if self.websocket_manager and result:
                        # Broadcast thinking results to connected clients
                        await self.websocket_manager.broadcast_thinking_update(result)
                        
            except Exception as e:
                logger.error(f"Error in autonomous thinking loop: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes on error

# Global component manager
component_manager = AutonomousComponentManager()

# Application lifespan management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application startup and shutdown"""
    # Startup
    logger.info("🌟 Starting Autonomous Agentic AI API...")
    await component_manager.initialize_all_components()
    
    if component_manager.initialized:
        logger.info("✅ Autonomous AI API ready for autonomous intelligence!")
    else:
        logger.error("❌ Autonomous AI API failed to initialize properly")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Autonomous AI API...")

# FastAPI application with autonomous system integration
app = FastAPI(
    title="Autonomous Agentic AI API",
    description="Advanced autonomous intelligence system with 3-agent architecture",
    version="2.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ChatRequest(BaseModel):
    message: str
    user_id: str = "admin"
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    response: str
    agent_name: str
    timestamp: str
    user_context: Optional[Dict[str, Any]] = None
    collaboration_summary: Optional[Dict[str, Any]] = None

# API Endpoints

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if component_manager.initialized else "initializing",
        "version": "2.0.0",
        "architecture": "3-agent autonomous",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "autonomous_orchestrator": component_manager.autonomous_orchestrator is not None,
            "memory_system": component_manager.memory_system is not None,
            "transformers_service": component_manager.transformers_service is not None,
            "websocket_manager": component_manager.websocket_manager is not None
        }
    }

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    """
    Main chat endpoint - preserves API compatibility
    Routes through autonomous orchestrator
    """
    if not component_manager.initialized or not component_manager.autonomous_orchestrator:
        raise HTTPException(status_code=503, detail="Autonomous system not initialized")
    
    try:
        start_time = time.time()
        
        # Process through autonomous orchestrator
        result = await component_manager.autonomous_orchestrator.handle_user_message(
            user_id=request.user_id,
            message=request.message,
            context=request.context
        )
        
        response_time = time.time() - start_time
        
        # Broadcast to WebSocket if available
        if component_manager.websocket_manager:
            await component_manager.websocket_manager.broadcast_chat_response(result)
        
        # Return standardized response
        return ChatResponse(
            response=result.get("response", ""),
            agent_name=result.get("agent_name", "autonomous_system"),
            timestamp=result.get("timestamp", datetime.now().isoformat()),
            user_context=result.get("user_context"),
            collaboration_summary=result.get("collaboration_summary")
        )
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

@app.get("/agents/status")
async def get_agents_status():
    """Get status of all autonomous agents"""
    if not component_manager.autonomous_orchestrator:
        raise HTTPException(status_code=503, detail="Autonomous system not initialized")
    
    try:
        return component_manager.autonomous_orchestrator.get_system_status()
    except Exception as e:
        logger.error(f"Error getting agent status: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/insights")
async def get_memory_insights(user_id: str = "admin"):
    """Get memory insights for user"""
    if not component_manager.memory_system:
        raise HTTPException(status_code=503, detail="Memory system not initialized")
    
    try:
        insights = await component_manager.memory_system.get_user_insights(user_id)
        return {
            "user_id": user_id,
            "insights": insights,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting memory insights: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/memory/search")
async def search_memory(query: str, user_id: str = "admin", limit: int = 10):
    """Search user memory"""
    if not component_manager.memory_system:
        raise HTTPException(status_code=503, detail="Memory system not initialized")
    
    try:
        results = await component_manager.memory_system.search_memories(user_id, query, limit=limit)
        return {
            "query": query,
            "user_id": user_id,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error searching memory: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat/history")
async def get_chat_history(user_id: str = "admin", limit: int = 20):
    """Get recent chat history"""
    if not component_manager.autonomous_orchestrator:
        raise HTTPException(status_code=503, detail="Autonomous system not initialized")
    
    try:
        history = component_manager.autonomous_orchestrator.get_recent_conversations(limit=limit)
        return {
            "user_id": user_id,
            "history": history,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/autonomous/thinking")
async def trigger_autonomous_thinking():
    """Manually trigger autonomous thinking cycle"""
    if not component_manager.autonomous_orchestrator:
        raise HTTPException(status_code=503, detail="Autonomous system not initialized")
    
    try:
        result = await component_manager.autonomous_orchestrator.autonomous_thinking_cycle()
        return result
    except Exception as e:
        logger.error(f"Error triggering autonomous thinking: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/system/metrics")
async def get_system_metrics():
    """Get system performance metrics"""
    if not component_manager.autonomous_orchestrator:
        raise HTTPException(status_code=503, detail="Autonomous system not initialized")
    
    try:
        status = component_manager.autonomous_orchestrator.get_system_status()
        return {
            "system_status": status,
            "memory_stats": await component_manager.memory_system.get_statistics() if component_manager.memory_system else {},
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting system metrics: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# WebSocket endpoints
if WEBSOCKET_AVAILABLE:
    @app.websocket("/thinking/stream")
    async def thinking_stream_websocket(websocket: WebSocket):
        """WebSocket for real-time thinking stream"""
        await websocket.accept()
        
        if component_manager.websocket_manager:
            try:
                await component_manager.websocket_manager.handle_thinking_stream(websocket)
            except WebSocketDisconnect:
                logger.info("Thinking stream WebSocket disconnected")
            except Exception as e:
                logger.error(f"Thinking stream WebSocket error: {e}")
        else:
            await websocket.close(code=1011, reason="WebSocket manager not available")
    
    @app.websocket("/agent-stream")
    async def agent_stream_websocket(websocket: WebSocket):
        """WebSocket for real-time agent communication"""
        await websocket.accept()
        
        if component_manager.websocket_manager:
            try:
                await component_manager.websocket_manager.handle_agent_stream(websocket)
            except WebSocketDisconnect:
                logger.info("Agent stream WebSocket disconnected")
            except Exception as e:
                logger.error(f"Agent stream WebSocket error: {e}")
        else:
            await websocket.close(code=1011, reason="WebSocket manager not available")

# Clean Agent Info Endpoint
@app.get("/agent")
async def get_agent_info():
    """Basic agent system information"""
    return {
        "agent_system": "autonomous_agentic_ai",
        "version": "2.0.0",
        "architecture": "3-agent_autogen_groupchat",
        "status": "active" if component_manager.initialized else "initializing",
        "agents": {
            "memory_agent": "UI hub & memory management",
            "research_agent": "External knowledge gathering",
            "intelligence_agent": "Autonomous thinking & planning"
        },
        "coordination": "autogen_groupchat",
        "features": [
            "autonomous_thinking_cycles",
            "5_layer_memory_system", 
            "real_time_websockets",
            "life_event_planning",
            "pattern_discovery"
        ],
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    uvicorn.run(
        "autonomous_main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
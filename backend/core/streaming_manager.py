"""
Streaming Manager for Real-time Agent Coordination Updates
Integrates with LangGraph orchestrator for live agent status streaming
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum

logger = logging.getLogger(__name__)


class AgentStatus(Enum):
    """Agent execution status for streaming updates"""
    WAITING = "waiting"
    ACTIVE = "active"
    COMPLETED = "completed"
    ERROR = "error"
    SKIPPED = "skipped"


class StreamEventType(Enum):
    """Types of streaming events"""
    AGENT_STATUS = "agent_status"
    WORKFLOW_START = "workflow_start"
    WORKFLOW_COMPLETE = "workflow_complete"
    PROGRESS_UPDATE = "progress_update"
    AGENT_RESULT = "agent_result"
    ERROR_UPDATE = "error_update"


@dataclass
class AgentStreamUpdate:
    """Agent status update for streaming"""
    agent_name: str
    status: AgentStatus
    progress_percentage: float
    current_activity: str
    execution_time: Optional[float] = None
    result_preview: Optional[str] = None
    error_message: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


@dataclass
class WorkflowStreamUpdate:
    """Workflow-level streaming update"""
    workflow_id: str
    user_name: str
    total_agents: int
    completed_agents: int
    current_phase: str
    overall_progress: float
    estimated_remaining_time: Optional[float] = None
    agents_status: Optional[Dict[str, AgentStatus]] = None
    timestamp: str = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()


class StreamingManager:
    """
    Centralized streaming manager for real-time agent coordination updates
    Integrates with LangGraph orchestrator and WebSocket manager
    """
    
    def __init__(self):
        self.websocket_manager = None
        self.active_workflows: Dict[str, WorkflowStreamUpdate] = {}
        self.stream_callbacks: List[Callable] = []
        self.buffer_size = 100
        self.update_buffer: List[Dict[str, Any]] = []
        self.throttle_interval = 0.1  # 100ms throttling
        self.last_update_time = 0
        self._buffer_lock = asyncio.Lock()
        
        # Agent execution order and phases for progress calculation
        self.agent_phases = {
            "router": {"order": 0, "weight": 0.1},
            "memory_reader": {"order": 1, "weight": 0.25},
            "knowledge_agent": {"order": 2, "weight": 0.35},
            "organizer": {"order": 3, "weight": 0.25},
            "memory_writer": {"order": 4, "weight": 0.05}
        }
        
        logger.info("🔄 Streaming manager initialized")
    
    def set_websocket_manager(self, websocket_manager):
        """Connect WebSocket manager for broadcasting"""
        self.websocket_manager = websocket_manager
        logger.info("📡 WebSocket manager connected to streaming manager")
    
    def add_stream_callback(self, callback: Callable):
        """Add custom callback for stream events"""
        self.stream_callbacks.append(callback)
        logger.info(f"➕ Added stream callback: {callback.__name__}")
    
    async def start_workflow_stream(
        self, 
        workflow_id: str, 
        user_name: str, 
        user_message: str,
        agents_to_execute: List[str]
    ) -> WorkflowStreamUpdate:
        """Start streaming for a new workflow"""
        
        workflow_update = WorkflowStreamUpdate(
            workflow_id=workflow_id,
            user_name=user_name,
            total_agents=len(agents_to_execute),
            completed_agents=0,
            current_phase="starting",
            overall_progress=0.0,
            agents_status={agent: AgentStatus.WAITING for agent in agents_to_execute}
        )
        
        self.active_workflows[workflow_id] = workflow_update
        
        # Broadcast workflow start
        await self._broadcast_stream_event(
            event_type=StreamEventType.WORKFLOW_START,
            data={
                "workflow_id": workflow_id,
                "user_name": user_name,
                "user_message": user_message[:100] + "..." if len(user_message) > 100 else user_message,
                "agents_planned": agents_to_execute,
                "total_agents": len(agents_to_execute),
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info(f"🚀 Started workflow stream: {workflow_id} for {user_name}")
        return workflow_update
    
    async def update_agent_status(
        self,
        workflow_id: str,
        agent_name: str,
        status: AgentStatus,
        current_activity: str = "",
        progress_percentage: float = 0.0,
        execution_time: Optional[float] = None,
        result_preview: Optional[str] = None,
        error_message: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Update agent status and broadcast streaming update"""
        
        if workflow_id not in self.active_workflows:
            logger.warning(f"⚠️ Workflow {workflow_id} not found in active workflows")
            return
        
        workflow = self.active_workflows[workflow_id]
        
        # Update agent status in workflow
        if workflow.agents_status:
            workflow.agents_status[agent_name] = status
        
        # Calculate overall progress
        overall_progress = self._calculate_overall_progress(workflow, agent_name, progress_percentage)
        workflow.overall_progress = overall_progress
        workflow.current_phase = f"{agent_name}: {current_activity}"
        
        # Update completed agents count
        if status == AgentStatus.COMPLETED:
            workflow.completed_agents += 1
        
        # Create agent update
        agent_update = AgentStreamUpdate(
            agent_name=agent_name,
            status=status,
            progress_percentage=progress_percentage,
            current_activity=current_activity,
            execution_time=execution_time,
            result_preview=result_preview,
            error_message=error_message,
            metadata=metadata
        )
        
        # Broadcast agent status update
        await self._broadcast_stream_event(
            event_type=StreamEventType.AGENT_STATUS,
            data={
                "workflow_id": workflow_id,
                "user_name": workflow.user_name,
                "agent_update": asdict(agent_update),
                "workflow_progress": {
                    "overall_progress": overall_progress,
                    "completed_agents": workflow.completed_agents,
                    "total_agents": workflow.total_agents,
                    "current_phase": workflow.current_phase
                },
                "agents_status": {k: v.value for k, v in workflow.agents_status.items()} if workflow.agents_status else {},
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info(f"📊 Updated {agent_name} status: {status.value} ({progress_percentage:.0f}%) in workflow {workflow_id}")
    
    async def complete_workflow_stream(
        self,
        workflow_id: str,
        final_response: str,
        total_execution_time: float,
        agents_executed: List[str],
        workflow_pattern: str
    ):
        """Complete workflow streaming and broadcast final results"""
        
        if workflow_id not in self.active_workflows:
            logger.warning(f"⚠️ Workflow {workflow_id} not found for completion")
            return
        
        workflow = self.active_workflows[workflow_id]
        workflow.overall_progress = 100.0
        workflow.current_phase = "completed"
        workflow.completed_agents = workflow.total_agents
        
        # Broadcast workflow completion
        await self._broadcast_stream_event(
            event_type=StreamEventType.WORKFLOW_COMPLETE,
            data={
                "workflow_id": workflow_id,
                "user_name": workflow.user_name,
                "final_response": final_response[:200] + "..." if len(final_response) > 200 else final_response,
                "total_execution_time": total_execution_time,
                "agents_executed": agents_executed,
                "workflow_pattern": workflow_pattern,
                "overall_progress": 100.0,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        # Clean up completed workflow
        del self.active_workflows[workflow_id]
        
        logger.info(f"✅ Completed workflow stream: {workflow_id} in {total_execution_time:.2f}s")
    
    async def stream_error(
        self,
        workflow_id: str,
        agent_name: str,
        error_message: str,
        error_details: Optional[Dict[str, Any]] = None
    ):
        """Stream error updates"""
        
        await self.update_agent_status(
            workflow_id=workflow_id,
            agent_name=agent_name,
            status=AgentStatus.ERROR,
            current_activity="Error occurred",
            error_message=error_message,
            metadata=error_details
        )
        
        # Also broadcast dedicated error event
        await self._broadcast_stream_event(
            event_type=StreamEventType.ERROR_UPDATE,
            data={
                "workflow_id": workflow_id,
                "agent_name": agent_name,
                "error_message": error_message,
                "error_details": error_details,
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.error(f"❌ Streamed error for {agent_name} in workflow {workflow_id}: {error_message}")
    
    def _calculate_overall_progress(
        self, 
        workflow: WorkflowStreamUpdate, 
        current_agent: str, 
        agent_progress: float
    ) -> float:
        """Calculate overall workflow progress based on agent phases and weights"""
        
        total_progress = 0.0
        
        if not workflow.agents_status:
            return 0.0
        
        for agent_name, status in workflow.agents_status.items():
            agent_config = self.agent_phases.get(agent_name, {"order": 999, "weight": 0.1})
            weight = agent_config["weight"]
            
            if status == AgentStatus.COMPLETED:
                total_progress += weight * 100.0
            elif status == AgentStatus.ACTIVE and agent_name == current_agent:
                total_progress += weight * agent_progress
            elif status == AgentStatus.ERROR:
                total_progress += weight * 100.0  # Count errors as "complete" for progress
        
        return min(total_progress, 100.0)
    
    async def _broadcast_stream_event(self, event_type: StreamEventType, data: Dict[str, Any]):
        """Broadcast streaming event via WebSocket with throttling"""
        
        current_time = datetime.now().timestamp()
        
        # Add to buffer
        async with self._buffer_lock:
            self.update_buffer.append({
                "type": "agent_stream",
                "event_type": event_type.value,
                "data": data,
                "timestamp": datetime.now().isoformat()
            })
            
            # Maintain buffer size
            if len(self.update_buffer) > self.buffer_size:
                self.update_buffer.pop(0)
        
        # Throttled broadcasting
        if current_time - self.last_update_time >= self.throttle_interval:
            await self._flush_stream_buffer()
            self.last_update_time = current_time
    
    async def _flush_stream_buffer(self):
        """Flush buffered updates to WebSocket"""
        
        if not self.websocket_manager or not self.update_buffer:
            return
        
        async with self._buffer_lock:
            # Get all buffered updates
            updates_to_send = self.update_buffer.copy()
            self.update_buffer.clear()
        
        # Send consolidated update
        if updates_to_send:
            consolidated_message = {
                "type": "agent_stream_batch",
                "updates": updates_to_send,
                "batch_size": len(updates_to_send),
                "timestamp": datetime.now().isoformat()
            }
            
            # Broadcast to all connected clients
            for user_name in list(self.websocket_manager.active_connections.keys()):
                await self.websocket_manager.send_personal_message(consolidated_message, user_name)
        
        # Execute custom callbacks
        for callback in self.stream_callbacks:
            try:
                await callback(updates_to_send)
            except Exception as e:
                logger.error(f"❌ Stream callback failed: {e}")
    
    async def force_flush_buffer(self):
        """Force flush buffer (useful for workflow completion)"""
        await self._flush_stream_buffer()
    
    async def stream_parallel_execution_start(
        self,
        workflow_id: str,
        parallel_agents: List[str],
        execution_phase: str
    ):
        """Stream parallel execution start event"""
        
        await self._broadcast_stream_event(
            event_type=StreamEventType.PROGRESS_UPDATE,
            data={
                "workflow_id": workflow_id,
                "event_subtype": "parallel_execution_start",
                "parallel_agents": parallel_agents,
                "execution_phase": execution_phase,
                "optimization_type": "concurrent_execution",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info(f"⚡ Streaming parallel execution start: {parallel_agents} in phase {execution_phase}")
    
    async def stream_parallel_completion(
        self,
        workflow_id: str,
        completed_agents: List[str],
        execution_time: float,
        speedup_factor: float
    ):
        """Stream parallel execution completion with performance metrics"""
        
        await self._broadcast_stream_event(
            event_type=StreamEventType.PROGRESS_UPDATE,
            data={
                "workflow_id": workflow_id,
                "event_subtype": "parallel_execution_complete",
                "completed_agents": completed_agents,
                "execution_time": execution_time,
                "speedup_factor": speedup_factor,
                "optimization_result": "success",
                "timestamp": datetime.now().isoformat()
            }
        )
        
        logger.info(f"✨ Parallel execution completed: {speedup_factor:.2f}x speedup in {execution_time:.2f}s")
    
    def get_active_workflows(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all active workflows"""
        return {
            workflow_id: asdict(workflow) 
            for workflow_id, workflow in self.active_workflows.items()
        }
    
    def get_streaming_stats(self) -> Dict[str, Any]:
        """Get streaming manager statistics"""
        return {
            "active_workflows": len(self.active_workflows),
            "buffer_size": len(self.update_buffer),
            "callbacks_registered": len(self.stream_callbacks),
            "websocket_connected": self.websocket_manager is not None,
            "throttle_interval": self.throttle_interval,
            "last_update": self.last_update_time,
            "agent_phases_configured": len(self.agent_phases),
            "parallel_optimizations": {
                "concurrent_memory_knowledge": True,
                "background_memory_writer": True,
                "dependency_resolution": True,
                "streaming_coordination": True
            }
        }


# Global streaming manager instance
streaming_manager = StreamingManager()
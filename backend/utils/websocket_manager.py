"""
WebSocket Connection Manager
Handles real-time WebSocket connections for the Autonomous Agentic AI System.

Provides:
- WebSocket connection management
- User-specific broadcasting
- Chat response distribution
- Autonomous insight broadcasting
- Agent streaming updates
"""

import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from fastapi import WebSocket
from utils.serialization import safe_json_dumps, prepare_websocket_message

logger = logging.getLogger(__name__)


class WebSocketConnectionManager:
    """
    Manages WebSocket connections for real-time communication.
    
    Features:
    - User-centric connection management 
    - Chat response broadcasting
    - Autonomous insight streaming
    - Agent status updates
    - Error handling and connection cleanup
    """
    
    def __init__(self):
        # Active connections: user_name -> Set[WebSocket]
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        
        # Connection metadata: websocket -> user_info
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        
        # Message statistics
        self.message_stats = {
            "total_sent": 0,
            "total_connections": 0,
            "active_users": 0,
            "last_activity": None
        }
        
        # Connection lock for thread safety
        self._connection_lock = asyncio.Lock()
        
        logger.info("🔗 WebSocket Connection Manager initialized")
    
    async def connect(self, websocket: WebSocket, user_name: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Connect a new WebSocket for a user.
        
        Args:
            websocket: FastAPI WebSocket instance
            user_name: User name from settings.yaml
            metadata: Optional connection metadata
        """
        async with self._connection_lock:
            # Initialize user connections if needed
            if user_name not in self.active_connections:
                self.active_connections[user_name] = set()
            
            # Add websocket to user's connections
            self.active_connections[user_name].add(websocket)
            
            # Store connection metadata
            self.connection_metadata[websocket] = {
                "user_name": user_name,
                "connected_at": datetime.now().isoformat(),
                "message_count": 0,
                **(metadata or {})
            }
            
            # Update statistics
            self.message_stats["total_connections"] += 1
            self.message_stats["active_users"] = len(self.active_connections)
            self.message_stats["last_activity"] = datetime.now().isoformat()
            
            logger.info(f"🔗 WebSocket connected: {user_name} (total: {len(self.active_connections[user_name])} connections)")
    
    async def disconnect(self, websocket: WebSocket):
        """
        Disconnect a WebSocket and clean up resources.
        
        Args:
            websocket: FastAPI WebSocket instance to disconnect
        """
        async with self._connection_lock:
            # Get user info before cleanup
            user_info = self.connection_metadata.get(websocket)
            user_name = user_info.get("user_name", "unknown") if user_info else "unknown"
            
            # Remove from active connections
            if user_name in self.active_connections:
                self.active_connections[user_name].discard(websocket)
                
                # Remove user entry if no more connections
                if not self.active_connections[user_name]:
                    del self.active_connections[user_name]
            
            # Remove metadata
            self.connection_metadata.pop(websocket, None)
            
            # Update statistics
            self.message_stats["active_users"] = len(self.active_connections)
            self.message_stats["last_activity"] = datetime.now().isoformat()
            
            logger.info(f"🔗 WebSocket disconnected: {user_name} (remaining users: {len(self.active_connections)})")
    
    async def send_personal_message(self, message: Dict[str, Any], user_name: str):
        """
        Send message to all connections for a specific user.
        
        Args:
            message: Message dictionary to send
            user_name: Target user name
        """
        if user_name not in self.active_connections:
            logger.debug(f"📤 No active connections for user: {user_name}")
            return
        
        # Get user's connections
        user_connections = self.active_connections[user_name].copy()
        
        if not user_connections:
            logger.debug(f"📤 No WebSocket connections for user: {user_name}")
            return
        
        # Prepare message for sending
        json_message = safe_json_dumps(message)
        
        # Send to all user's connections
        disconnected_sockets = []
        
        for websocket in user_connections:
            try:
                await websocket.send_text(json_message)
                
                # Update message count
                if websocket in self.connection_metadata:
                    self.connection_metadata[websocket]["message_count"] += 1
                
                self.message_stats["total_sent"] += 1
                
            except Exception as e:
                logger.error(f"❌ Failed to send message to {user_name}: {e}")
                disconnected_sockets.append(websocket)
        
        # Clean up disconnected sockets
        for websocket in disconnected_sockets:
            await self.disconnect(websocket)
        
        logger.debug(f"📤 Message sent to {len(user_connections) - len(disconnected_sockets)} connections for user: {user_name}")
    
    async def broadcast_to_all(self, message: Dict[str, Any]):
        """
        Broadcast message to all connected users.
        
        Args:
            message: Message dictionary to broadcast
        """
        if not self.active_connections:
            logger.debug("📡 No active connections for broadcast")
            return
        
        logger.info(f"📡 Broadcasting to {len(self.active_connections)} users")
        
        # Send to each user
        for user_name in list(self.active_connections.keys()):
            await self.send_personal_message(message, user_name)
    
    async def broadcast_chat_response(self, result: Dict[str, Any]):
        """
        Broadcast chat response to all connected clients.
        
        Args:
            result: Chat result from orchestrator
        """
        message = prepare_websocket_message(
            message_type="chat_response",
            data={
                "response": result.get("response", ""),
                "metadata": result.get("metadata", {}),
                "processing_time": result.get("processing_time", 0),
                "workflow_pattern": result.get("metadata", {}).get("workflow_pattern", "standard"),
                "agents_executed": result.get("metadata", {}).get("agents_executed", [])
            }
        )
        
        await self.broadcast_to_all(message)
        logger.info("📡 Chat response broadcasted")
    
    async def broadcast_autonomous_insight(self, insight_data: Dict[str, Any], user_name: str):
        """
        Broadcast autonomous insight to specific user.
        
        Args:
            insight_data: Insight data to broadcast
            user_name: Target user name
        """
        message = prepare_websocket_message(
            message_type="autonomous_insight",
            data=insight_data,
            user_name=user_name
        )
        
        await self.send_personal_message(message, user_name)
        logger.info(f"🧠 Autonomous insight sent to: {user_name}")
    
    async def broadcast_agent_status(self, agent_update: Dict[str, Any]):
        """
        Broadcast agent status update to all clients.
        
        Args:
            agent_update: Agent status update data
        """
        message = prepare_websocket_message(
            message_type="agent_status",
            data=agent_update
        )
        
        await self.broadcast_to_all(message)
        logger.debug(f"🤖 Agent status update broadcasted: {agent_update.get('agent_name', 'unknown')}")
    
    async def broadcast_workflow_update(self, workflow_data: Dict[str, Any]):
        """
        Broadcast workflow progress update.
        
        Args:
            workflow_data: Workflow progress data
        """
        message = prepare_websocket_message(
            message_type="workflow_update",
            data=workflow_data
        )
        
        await self.broadcast_to_all(message)
        logger.debug(f"⚡ Workflow update broadcasted: {workflow_data.get('workflow_id', 'unknown')}")
    
    def get_connection_stats(self) -> Dict[str, Any]:
        """
        Get current connection statistics.
        
        Returns:
            Dictionary with connection statistics
        """
        total_connections = sum(len(connections) for connections in self.active_connections.values())
        
        return {
            "active_users": len(self.active_connections),
            "total_active_connections": total_connections,
            "total_connections_ever": self.message_stats["total_connections"],
            "total_messages_sent": self.message_stats["total_sent"],
            "last_activity": self.message_stats["last_activity"],
            "users_connected": list(self.active_connections.keys()),
            "connections_per_user": {
                user: len(connections) 
                for user, connections in self.active_connections.items()
            }
        }
    
    def get_user_connections(self, user_name: str) -> int:
        """
        Get number of active connections for a user.
        
        Args:
            user_name: User name to check
            
        Returns:
            Number of active connections
        """
        return len(self.active_connections.get(user_name, set()))
    
    async def send_system_message(self, message_type: str, data: Any, target_user: Optional[str] = None):
        """
        Send system message to users.
        
        Args:
            message_type: Type of system message
            data: Message data
            target_user: Optional target user (broadcasts to all if None)
        """
        message = prepare_websocket_message(
            message_type=f"system_{message_type}",
            data=data
        )
        
        if target_user:
            await self.send_personal_message(message, target_user)
        else:
            await self.broadcast_to_all(message)
        
        logger.info(f"🔔 System message sent: {message_type}")


# Create a singleton instance for the application
ConnectionManager = WebSocketConnectionManager
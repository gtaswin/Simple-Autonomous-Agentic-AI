"""
WebSocket manager for real-time agent communication
"""
import asyncio
import json
from typing import Dict, List
from fastapi import WebSocket
from datetime import datetime

# Import proper JSON serialization utilities
from utils.serialization import safe_json_dumps, prepare_websocket_message


class AgentStreamManager:
    """Manages WebSocket connections for real-time agent updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        
    async def connect(self, websocket: WebSocket, user_id: str):
        """Connect a WebSocket for a user"""
        await websocket.accept()
        if user_id not in self.active_connections:
            self.active_connections[user_id] = []
        self.active_connections[user_id].append(websocket)
        # WebSocket connected
        
    async def disconnect(self, websocket: WebSocket, user_id: str):
        """Disconnect a WebSocket"""
        if user_id in self.active_connections:
            try:
                self.active_connections[user_id].remove(websocket)
                if not self.active_connections[user_id]:
                    del self.active_connections[user_id]
                print(f"ℹ️ WebSocket connection removed for user: {user_id}")
            except ValueError:
                print(f"⚠️ WebSocket already removed for user: {user_id}")
        
    async def send_personal_message(self, message: dict, user_id: str):
        """Send message to specific user"""
        if user_id in self.active_connections:
            for connection in self.active_connections[user_id]:
                try:
                    await connection.send_text(safe_json_dumps(message))
                except Exception as e:
                    print(f"❌ WebSocket send failed for {user_id}: {e}")
                    # Remove failed connection
                    try:
                        self.active_connections[user_id].remove(connection)
                    except (ValueError, KeyError):
                        pass
    async def broadcast_thought(self, thought: dict):
        """Broadcast agent thoughts to all connected users"""
        message = prepare_websocket_message("thought", thought)
        
        for user_id, connections in self.active_connections.items():
            for connection in connections:
                try:
                    await connection.send_text(safe_json_dumps(message))
                except Exception as e:
                    print(f"❌ Failed to broadcast thought to {user_id}: {e}")
                    # Remove failed connection
                    try:
                        self.active_connections[user_id].remove(connection)
                    except (ValueError, KeyError):
                        pass
    async def broadcast_decision(self, decision: dict):
        """Broadcast agent decisions to all connected users"""
        message = prepare_websocket_message("decision", decision)
        
        for user_id, connections in self.active_connections.items():
            for connection in connections:
                try:
                    await connection.send_text(safe_json_dumps(message))
                except Exception as e:
                    print(f"❌ Failed to broadcast decision to {user_id}: {e}")
                    # Remove failed connection
                    try:
                        self.active_connections[user_id].remove(connection)
                    except (ValueError, KeyError):
                        pass
    async def send_agent_status(self, status: dict, user_id: str):
        """Send agent status updates"""
        message = prepare_websocket_message("agent_status", status, user_id)
        await self.send_personal_message(message, user_id)
        
    async def broadcast_new_thought(self, thought: dict):
        """Broadcast new autonomous thoughts in real-time"""
        message = {
            "type": "new_thought",
            "data": {
                "id": thought.get("id"),
                "content": thought.get("content"),
                "type": thought.get("type"),
                "timestamp": thought.get("timestamp", datetime.now().isoformat()),
                "priority": thought.get("priority"),
                "user_id": thought.get("user_id", "admin")  # Will be updated by caller
            },
            "timestamp": datetime.now().isoformat()
        }
        
        for user_id, connections in self.active_connections.items():
            for connection in connections:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    print(f"❌ Failed to broadcast new thought to {user_id}: {e}")
                    # Remove failed connection
                    try:
                        self.active_connections[user_id].remove(connection)
                    except (ValueError, KeyError):
                        pass
    async def broadcast_agent_conversation(self, conversation: dict):
        """Broadcast multi-agent conversations in real-time"""
        message = {
            "type": "agent_conversation",
            "data": {
                "collaboration_id": conversation.get("collaboration_id"),
                "speaker": conversation.get("speaker"),
                "message": conversation.get("message"),
                "timestamp": conversation.get("timestamp", datetime.now().isoformat()),
                "agent_type": conversation.get("agent_type")
            },
            "timestamp": datetime.now().isoformat()
        }
        
        for user_id, connections in self.active_connections.items():
            for connection in connections:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    print(f"Failed to broadcast agent conversation to {user_id}: {e}")
                    
    async def broadcast_autonomous_insight(self, insight: dict):
        """Broadcast autonomous insights as they're discovered"""
        message = {
            "type": "autonomous_insight",
            "data": {
                "insight_id": insight.get("id"),
                "title": insight.get("title"),
                "content": insight.get("content"),
                "confidence": insight.get("confidence"),
                "category": insight.get("category"),
                "timestamp": insight.get("timestamp", datetime.now().isoformat())
            },
            "timestamp": datetime.now().isoformat()
        }
        
        for user_id, connections in self.active_connections.items():
            for connection in connections:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    print(f"Failed to broadcast autonomous insight to {user_id}: {e}")
    
    async def stream_memory_update(self, user_id: str, memory_event: dict):
        """Stream memory storage and classification events"""
        message = {
            "type": "memory_update",
            "data": {
                "event_type": memory_event.get("event_type"),  # "stored", "classified", "pattern_discovered"
                "memory_type": memory_event.get("memory_type"),
                "content_preview": memory_event.get("content", "")[:100],
                "importance": memory_event.get("importance", 0.5),
                "tags": memory_event.get("tags", []),
                "classification": memory_event.get("classification"),
                "timestamp": memory_event.get("timestamp", datetime.now().isoformat())
            },
            "timestamp": datetime.now().isoformat()
        }
        await self.send_personal_message(message, user_id)
    
    async def stream_expert_team_flow(self, collaboration_id: str, step: dict):
        """Stream expert team collaboration flow in real-time"""
        message = {
            "type": "expert_team_flow",
            "data": {
                "collaboration_id": collaboration_id,
                "step": step.get("step"),  # "research_start", "memory_analysis", "thinking_process", "coordination"
                "agent": step.get("agent"),
                "status": step.get("status"),  # "started", "in_progress", "completed"
                "result_preview": step.get("result", "")[:150],
                "confidence": step.get("confidence"),
                "timestamp": step.get("timestamp", datetime.now().isoformat())
            },
            "timestamp": datetime.now().isoformat()
        }
        
        for user_id, connections in self.active_connections.items():
            for connection in connections:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    print(f"Failed to stream expert team flow to {user_id}: {e}")
    
    async def stream_continuous_thinking(self, user_id: str, thinking_event: dict):
        """Stream continuous reasoning and pattern discovery"""
        message = {
            "type": "continuous_thinking",
            "data": {
                "thinking_type": thinking_event.get("type"),  # "observation", "reflection", "pattern", "insight"
                "content": thinking_event.get("content"),
                "reasoning_chain": thinking_event.get("reasoning_chain", []),
                "confidence": thinking_event.get("confidence", 0.5),
                "triggered_by": thinking_event.get("triggered_by"),
                "action_needed": thinking_event.get("action_needed", False),
                "timestamp": thinking_event.get("timestamp", datetime.now().isoformat())
            },
            "timestamp": datetime.now().isoformat()
        }
        await self.send_personal_message(message, user_id)
    
    async def stream_autonomous_decision(self, user_id: str, decision: dict):
        """Stream autonomous decision making in real-time"""
        message = {
            "type": "autonomous_decision",
            "data": {
                "decision_id": decision.get("id"),
                "decision_type": decision.get("type"),  # "proactive", "reactive", "scheduled"
                "description": decision.get("description"),
                "reasoning": decision.get("reasoning"),
                "confidence": decision.get("confidence"),
                "reversible": decision.get("reversible", True),
                "impact_level": decision.get("impact_level", "low"),
                "requires_approval": decision.get("requires_approval", False),
                "timestamp": decision.get("timestamp", datetime.now().isoformat())
            },
            "timestamp": datetime.now().isoformat()
        }
        await self.send_personal_message(message, user_id)
    async def stream_thinking_content(self, user_id: str, thinking_data: dict):
        """Stream AI thinking content for transparency"""
        
        message = prepare_websocket_message({
            "type": "thinking_content", 
            "user_id": user_id,
            "thinking": {
                "content": thinking_data.get("thinking", []),
                "agent": thinking_data.get("agent", "unknown"),
                "context": thinking_data.get("context", ""),
                "timestamp": datetime.now().isoformat(),
                "has_thinking": thinking_data.get("has_thinking", False)
            }
        })
        
        await self.send_personal_message(message, user_id)
    
    async def handle_thinking_stream(self, websocket: WebSocket):
        """Handle thinking stream WebSocket connection"""
        try:
            # Extract user_id from query params if available
            user_id = websocket.query_params.get("user_id", "admin")
            await self.connect(websocket, user_id)
            
            # Send initial connection confirmation
            await websocket.send_text(json.dumps({
                "type": "connection_established",
                "message": "Thinking stream connected",
                "timestamp": datetime.now().isoformat()
            }))
            
            # Keep connection alive and listen for messages
            while True:
                try:
                    # Wait for messages from client (e.g., requests for thinking updates)
                    data = await websocket.receive_text()
                    # Handle client requests if needed
                    # For now, just acknowledge
                    await websocket.send_text(json.dumps({
                        "type": "acknowledged",
                        "timestamp": datetime.now().isoformat()
                    }))
                except Exception as e:
                    print(f"Error in thinking stream: {e}")
                    break
                    
        except Exception as e:
            print(f"Thinking stream connection error: {e}")
        finally:
            await self.disconnect(websocket, user_id)
    
    async def handle_agent_stream(self, websocket: WebSocket):
        """Handle agent communication stream WebSocket connection"""
        try:
            # Extract user_id from query params if available
            user_id = websocket.query_params.get("user_id", "admin")
            await self.connect(websocket, user_id)
            
            # Send initial connection confirmation
            await websocket.send_text(json.dumps({
                "type": "connection_established",
                "message": "Agent stream connected",
                "timestamp": datetime.now().isoformat()
            }))
            
            # Keep connection alive and listen for messages
            while True:
                try:
                    # Wait for messages from client
                    data = await websocket.receive_text()
                    # Handle client requests if needed
                    await websocket.send_text(json.dumps({
                        "type": "acknowledged",
                        "timestamp": datetime.now().isoformat()
                    }))
                except Exception as e:
                    print(f"Error in agent stream: {e}")
                    break
                    
        except Exception as e:
            print(f"Agent stream connection error: {e}")
        finally:
            await self.disconnect(websocket, user_id)
    
    async def broadcast_chat_response(self, chat_result: dict):
        """Broadcast chat response to all connected WebSocket clients"""
        message = {
            "type": "chat_response",
            "data": {
                "response": chat_result.get("response", ""),
                "agent_name": chat_result.get("agent_name", "autonomous_system"),
                "timestamp": chat_result.get("timestamp", datetime.now().isoformat()),
                "user_context": chat_result.get("user_context"),
                "collaboration_summary": chat_result.get("collaboration_summary")
            },
            "timestamp": datetime.now().isoformat()
        }
        
        # Broadcast to all connected users
        for user_id, connections in self.active_connections.items():
            for connection in connections:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    print(f"Failed to broadcast chat response to {user_id}: {e}")
                    # Remove failed connection
                    try:
                        self.active_connections[user_id].remove(connection)
                    except (ValueError, KeyError):
                        pass
    
    async def broadcast_thinking_update(self, thinking_data: dict):
        """Broadcast autonomous thinking updates to all connected clients"""
        message = {
            "type": "thinking_update",
            "data": thinking_data,
            "timestamp": datetime.now().isoformat()
        }
        
        for user_id, connections in self.active_connections.items():
            for connection in connections:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    print(f"Failed to broadcast thinking update to {user_id}: {e}")
                    try:
                        self.active_connections[user_id].remove(connection)
                    except (ValueError, KeyError):
                        pass
    
    async def broadcast_insight_update(self, insight_data: dict):
        """Broadcast weekly insights to all connected clients"""
        message = {
            "type": "insight_update",
            "data": insight_data,
            "timestamp": datetime.now().isoformat()
        }
        
        for user_id, connections in self.active_connections.items():
            for connection in connections:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    print(f"Failed to broadcast insight update to {user_id}: {e}")
                    try:
                        self.active_connections[user_id].remove(connection)
                    except (ValueError, KeyError):
                        pass
    
    async def broadcast_milestone_update(self, milestone_data: dict):
        """Broadcast life event milestone updates to all connected clients"""
        message = {
            "type": "milestone_update",
            "data": milestone_data,
            "timestamp": datetime.now().isoformat()
        }
        
        for user_id, connections in self.active_connections.items():
            for connection in connections:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    print(f"Failed to broadcast milestone update to {user_id}: {e}")
                    try:
                        self.active_connections[user_id].remove(connection)
                    except (ValueError, KeyError):
                        pass
    
    async def broadcast_alert(self, alert_data: dict):
        """Broadcast system alerts to all connected clients"""
        message = {
            "type": "system_alert",
            "data": alert_data,
            "timestamp": datetime.now().isoformat()
        }
        
        for user_id, connections in self.active_connections.items():
            for connection in connections:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    print(f"Failed to broadcast alert to {user_id}: {e}")
                    try:
                        self.active_connections[user_id].remove(connection)
                    except (ValueError, KeyError):
                        pass
    
    async def broadcast_memory_update(self, memory_data: dict):
        """Broadcast memory system updates to all connected clients"""
        message = {
            "type": "memory_update",
            "data": memory_data,
            "timestamp": datetime.now().isoformat()
        }
        
        for user_id, connections in self.active_connections.items():
            for connection in connections:
                try:
                    await connection.send_text(json.dumps(message))
                except Exception as e:
                    print(f"Failed to broadcast memory update to {user_id}: {e}")
                    try:
                        self.active_connections[user_id].remove(connection)
                    except (ValueError, KeyError):
                        pass

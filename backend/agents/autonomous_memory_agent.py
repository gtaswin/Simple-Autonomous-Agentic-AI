"""
Autonomous Memory Agent - User Interface Hub & Memory Management
Preserves all existing memory system functionality while providing autonomous integration
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import logging

from autogen import ConversableAgent

from memory.autonomous_memory import AutonomousMemorySystem
from core.config import AssistantConfig
from core.transformers_service import TransformersService
from utils.serialization import safe_serialize

logger = logging.getLogger(__name__)

class AutonomousMemoryAgent(ConversableAgent):
    """
    Memory Agent - Primary user interface and memory management hub
    
    Responsibilities:
    - All user chat interactions
    - Memory system operations (Redis/Qdrant)
    - User context management
    - Conversation history
    - Decision coordination with other agents
    """
    
    def __init__(
        self,
        name: str = "memory_agent",
        memory_system: Optional[AutonomousMemorySystem] = None,
        config: Optional[AssistantConfig] = None,
        transformers_service: Optional[TransformersService] = None,
        **kwargs
    ):
        """Initialize Memory Agent with existing memory system integration"""
        
        # System message for AutoGen
        system_message = """You are the Memory Agent, the primary interface for user interactions and memory management.

Your core responsibilities:
1. Handle all user conversations with warmth and intelligence
2. Manage user context, preferences, and personal information
3. Store and retrieve memories using our advanced memory system
4. Coordinate with Research Agent for external knowledge when needed
5. Collaborate with Intelligence Agent for strategic planning and insights
6. Provide contextual, personalized responses based on user history

Key capabilities:
- Access to comprehensive user memory (conversations, preferences, life events)
- Semantic memory search for relevant context
- Life event tracking and timeline management
- Conversation history management
- User personalization and adaptation

Always maintain user privacy and provide helpful, contextual responses."""

        # Initialize AutoGen ConversableAgent with disabled LLM client
        # We handle LLM calls manually through our existing system
        super().__init__(
            name=name,
            system_message=system_message,
            human_input_mode="NEVER",
            code_execution_config=False,
            llm_config=False,  # Disable AutoGen LLM client initialization
            **kwargs
        )
        
        # Preserve existing memory system
        self.memory_system = memory_system
        self.config = config or AssistantConfig()
        self.transformers_service = transformers_service
        
        # Chat state management
        self.current_user_id = None
        self.conversation_context = {}
        
        # Note: Teachable capability disabled due to import issues
        # Can be re-enabled when AutoGen version supports it
        # self.teachable = None
        
        logger.info(f"AutoGen Memory Agent initialized with memory system: {memory_system is not None}")
    
    async def handle_user_message(self, user_id: str, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point for user messages
        Preserves existing API compatibility
        """
        try:
            self.current_user_id = user_id
            self.conversation_context = context or {}
            
            # Get user context from memory system
            user_context = None
            if self.memory_system:
                user_context = await self.memory_system.get_user_context(user_id, message)
            
            # Prepare message with context
            contextual_message = await self._prepare_contextual_message(user_id, message, user_context)
            
            # Generate response using AutoGen
            response = await self._generate_response(contextual_message)
            
            # Store conversation in memory
            if self.memory_system:
                await self.memory_system.store_conversation(user_id, message, response)
            
            # Return formatted response
            return {
                "response": response,
                "agent_name": self.name,
                "timestamp": datetime.now().isoformat(),
                "user_context": safe_serialize(user_context) if user_context else None,
                "memory_stored": True
            }
            
        except Exception as e:
            logger.error(f"Error in handle_user_message: {e}")
            return {
                "response": "I apologize, but I encountered an error processing your message. Please try again.",
                "agent_name": self.name,
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    async def _prepare_contextual_message(self, user_id: str, message: str, user_context: Dict[str, Any]) -> str:
        """Prepare message with user context for better responses"""
        
        if not user_context:
            return message
        
        # Build contextual prompt
        context_parts = []
        
        # User information
        if user_context.get("user_info"):
            context_parts.append(f"User background: {user_context['user_info']}")
        
        # Recent conversations
        if user_context.get("recent_conversations"):
            recent = user_context["recent_conversations"][:3]  # Last 3 conversations
            context_parts.append(f"Recent conversation context: {recent}")
        
        # Life events
        if user_context.get("life_events"):
            context_parts.append(f"Important life events: {user_context['life_events']}")
        
        # Current preferences
        if user_context.get("preferences"):
            context_parts.append(f"User preferences: {user_context['preferences']}")
        
        if context_parts:
            contextual_message = f"""
Context about the user:
{chr(10).join(context_parts)}

Current user message: {message}

Please respond with this context in mind, providing personalized and relevant assistance.
"""
            return contextual_message
        
        return message
    
    async def _generate_response(self, message: str) -> str:
        """Generate response using AutoGen capabilities"""
        
        # For now, use a simple response mechanism
        # This will be enhanced when integrated with AutoGen GroupChat
        
        # Check if we need to involve other agents
        needs_research = await self._needs_research(message)
        needs_intelligence = await self._needs_intelligence(message)
        
        if needs_research or needs_intelligence:
            # This will be handled by GroupChat coordination
            logger.info(f"Message requires collaboration: research={needs_research}, intelligence={needs_intelligence}")
        
        # Generate direct response for now
        # This will be replaced with proper AutoGen message handling
        response = await self._generate_direct_response(message)
        return response
    
    async def _needs_research(self, message: str) -> bool:
        """Determine if message needs research agent involvement"""
        
        if not self.transformers_service:
            # Simple keyword-based detection
            research_keywords = [
                "search", "find", "look up", "what is", "current", "latest", "news",
                "weather", "stock", "price", "definition", "explain", "research"
            ]
            return any(keyword in message.lower() for keyword in research_keywords)
        
        # Use transformers service for better classification
        try:
            # TransformersService methods are synchronous, not async
            classification_result = self.transformers_service.classify_intent(message)
            # classification_result is a TransformerResult object with additional_data
            return classification_result.additional_data.get("needs_external_data", False)
        except Exception as e:
            logger.warning(f"Error in intent classification: {e}")
            return False
    
    async def _needs_intelligence(self, message: str) -> bool:
        """Determine if message needs intelligence agent involvement"""
        
        if not self.transformers_service:
            # Simple keyword-based detection
            intelligence_keywords = [
                "plan", "strategy", "timeline", "schedule", "organize", "analyze",
                "insight", "recommendation", "advice", "suggest", "future", "goals"
            ]
            return any(keyword in message.lower() for keyword in intelligence_keywords)
        
        # Use transformers service for better classification
        try:
            # TransformersService methods are synchronous, not async
            classification_result = self.transformers_service.classify_intent(message)
            # classification_result is a TransformerResult object with additional_data
            return classification_result.additional_data.get("needs_planning", False)
        except Exception as e:
            logger.warning(f"Error in intent classification: {e}")
            return False
    
    async def _generate_direct_response(self, message: str) -> str:
        """Generate direct response without other agents"""
        
        # Simple response for now - will be enhanced with proper LLM integration
        if "hi" in message.lower() or "hello" in message.lower():
            user_name = ""
            if self.current_user_id and self.memory_system:
                try:
                    user_context = await self.memory_system.get_user_context(self.current_user_id, "")
                    if user_context and user_context.get("user_info"):
                        # Extract name from user info
                        user_info = user_context["user_info"]
                        if "name" in user_info.lower():
                            user_name = " " + user_info.split("name")[1].split()[0].title()
                except:
                    pass
            
            return f"Hello{user_name}! How can I help you today?"
        
        return "I understand your message. Let me help you with that."
    
    async def search_memory(self, query: str, user_id: str = None) -> List[Dict[str, Any]]:
        """Search user memory for relevant information"""
        
        if not self.memory_system:
            return []
        
        target_user = user_id or self.current_user_id
        if not target_user:
            return []
        
        try:
            results = await self.memory_system.search_memories(target_user, query)
            return results
        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return []
    
    async def store_memory(self, user_id: str, content: str, memory_type: str = "episodic") -> bool:
        """Store information in user memory"""
        
        if not self.memory_system:
            return False
        
        try:
            await self.memory_system.store_memory(user_id, content, memory_type)
            return True
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return False
    
    async def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get comprehensive user context"""
        
        if not self.memory_system:
            return {}
        
        try:
            return await self.memory_system.get_user_context(user_id, "")
        except Exception as e:
            logger.error(f"Error getting user context: {e}")
            return {}
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        
        return {
            "agent_name": self.name,
            "status": "active",
            "capabilities": [
                "user_conversation",
                "memory_management", 
                "context_retrieval",
                "life_event_tracking",
                "agent_coordination"
            ],
            "memory_system_connected": self.memory_system is not None,
            "transformers_service_connected": self.transformers_service is not None,
            "current_user": self.current_user_id,
            "timestamp": datetime.now().isoformat()
        }
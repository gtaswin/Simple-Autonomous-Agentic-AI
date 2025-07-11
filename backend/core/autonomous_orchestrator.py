"""
Autonomous Orchestrator - 3-Agent Coordination System
Replaces manual multi-agent coordination with Autonomous GroupChat
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import logging

from autogen import GroupChat, GroupChatManager, ConversableAgent

from agents.autonomous_memory_agent import AutonomousMemoryAgent
from agents.autonomous_research_agent import AutonomousResearchAgent
from agents.autonomous_intelligence_agent import AutonomousIntelligenceAgent
from memory.autonomous_memory import AutonomousMemorySystem
from core.config import AssistantConfig
from core.transformers_service import TransformersService
from utils.serialization import safe_serialize

logger = logging.getLogger(__name__)

class AutonomousOrchestrator:
    """
    Autonomous Orchestrator - Central coordination for 3-agent system
    
    Replaces the complex manual multi-agent coordination with autonomous
    built-in GroupChat capabilities while preserving all existing functionality.
    """
    
    def __init__(
        self,
        memory_system: Optional[AutonomousMemorySystem] = None,
        config: Optional[AssistantConfig] = None,
        transformers_service: Optional[TransformersService] = None,
        tavily_api_key: Optional[str] = None
    ):
        """Initialize autonomous orchestrator with 3 specialized agents"""
        
        self.memory_system = memory_system
        self.config = config or AssistantConfig()
        self.transformers_service = transformers_service
        
        # Initialize the 3 specialized agents
        # Note: llm_config=False is set inside each agent to disable AutoGen LLM client
        self.memory_agent = AutonomousMemoryAgent(
            name="memory_agent",
            memory_system=memory_system,
            config=config,
            transformers_service=transformers_service
        )
        
        self.research_agent = AutonomousResearchAgent(
            name="research_agent",
            config=config,
            tavily_api_key=tavily_api_key
        )
        
        self.intelligence_agent = AutonomousIntelligenceAgent(
            name="intelligence_agent",
            memory_system=memory_system,
            config=config,
            transformers_service=transformers_service
        )
        
        # Create Autonomous GroupChat
        self.group_chat = GroupChat(
            agents=[self.memory_agent, self.research_agent, self.intelligence_agent],
            messages=[],
            max_round=10,
            speaker_selection_method="auto",
            allow_repeat_speaker=False
        )
        
        # Custom GroupChat Manager with intelligent routing
        # Disable LLM config to prevent AutoGen API key warnings
        self.group_chat_manager = CustomGroupChatManager(
            groupchat=self.group_chat,
            llm_config=False,  # Disable AutoGen LLM client - we handle manually
            memory_system=memory_system,
            transformers_service=transformers_service
        )
        
        # Performance tracking
        self.conversation_history = []
        self.performance_metrics = {
            "total_conversations": 0,
            "agent_collaboration_count": 0,
            "direct_responses": 0,
            "average_response_time": 0.0
        }
        
        logger.info("Autonomous Orchestrator initialized with 3 agents")
    
    # Note: _get_llm_config method removed since we disabled AutoGen LLM clients
    # All LLM calls are handled through our existing configuration system
    
    async def handle_user_message(self, user_id: str, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main entry point for user messages
        Maintains API compatibility with existing system
        """
        try:
            start_time = datetime.now()
            
            # Update performance metrics
            self.performance_metrics["total_conversations"] += 1
            
            # Determine routing strategy
            routing_decision = await self._determine_routing(message, context)
            
            # Handle based on routing decision
            if routing_decision["strategy"] == "direct_memory":
                # Simple message - Memory Agent handles alone
                response = await self.memory_agent.handle_user_message(user_id, message, context)
                self.performance_metrics["direct_responses"] += 1
                
            else:
                # Complex message - Use GroupChat collaboration
                response = await self._handle_collaborative_message(user_id, message, context, routing_decision)
                self.performance_metrics["agent_collaboration_count"] += 1
            
            # Update performance metrics
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds()
            self._update_response_time_metric(response_time)
            
            # Store conversation history
            self.conversation_history.append({
                "user_id": user_id,
                "message": message,
                "response": response,
                "routing_decision": routing_decision,
                "response_time": response_time,
                "timestamp": start_time.isoformat()
            })
            
            # Keep only last 100 conversations
            if len(self.conversation_history) > 100:
                self.conversation_history = self.conversation_history[-100:]
            
            return response
            
        except Exception as e:
            logger.error(f"Error in handle_user_message: {e}")
            return {
                "response": "I apologize, but I encountered an error processing your message. Please try again.",
                "agent_name": "orchestrator",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    async def _determine_routing(self, message: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Intelligent routing decision for message handling
        Uses transformers service if available, fallback to keyword analysis
        """
        
        routing = {
            "strategy": "direct_memory",
            "needs_research": False,
            "needs_intelligence": False,
            "complexity": "simple",
            "reasoning": ""
        }
        
        try:
            # Use transformers service for classification if available
            if self.transformers_service:
                # TransformersService methods are synchronous, not async
                classification_result = self.transformers_service.classify_intent(message)
                
                # classification_result is a TransformerResult object with additional_data
                routing["needs_research"] = classification_result.additional_data.get("needs_external_data", False)
                routing["needs_intelligence"] = classification_result.additional_data.get("needs_planning", False)
                
                if routing["needs_research"] or routing["needs_intelligence"]:
                    routing["strategy"] = "collaborative"
                    routing["complexity"] = "complex"
                    routing["reasoning"] = "AI classification determined collaboration needed"
                
                return routing
        
        except Exception as e:
            logger.warning(f"Error in AI classification, using fallback: {e}")
        
        # Fallback to keyword-based routing
        message_lower = message.lower()
        
        # Research indicators
        research_keywords = [
            "search", "find", "look up", "what is", "current", "latest", "news",
            "weather", "stock", "price", "definition", "explain", "research",
            "fact", "verify", "check", "source"
        ]
        
        # Intelligence/Planning indicators  
        intelligence_keywords = [
            "plan", "strategy", "timeline", "schedule", "organize", "analyze",
            "insight", "recommendation", "advice", "suggest", "future", "goals",
            "project", "roadmap", "steps", "how to", "guide", "approach"
        ]
        
        # Life event indicators (high priority for intelligence)
        life_event_keywords = [
            "pregnant", "pregnancy", "baby", "wedding", "marriage", "career",
            "job", "move", "house", "health", "doctor", "family"
        ]
        
        # Check for research needs
        if any(keyword in message_lower for keyword in research_keywords):
            routing["needs_research"] = True
        
        # Check for intelligence needs
        if any(keyword in message_lower for keyword in intelligence_keywords):
            routing["needs_intelligence"] = True
        
        # Life events always trigger intelligence
        if any(keyword in message_lower for keyword in life_event_keywords):
            routing["needs_intelligence"] = True
            routing["complexity"] = "complex"
        
        # Determine strategy
        if routing["needs_research"] or routing["needs_intelligence"]:
            routing["strategy"] = "collaborative"
            routing["complexity"] = "complex"
            routing["reasoning"] = f"Keyword analysis: research={routing['needs_research']}, intelligence={routing['needs_intelligence']}"
        else:
            routing["reasoning"] = "Simple message, direct memory agent response"
        
        return routing
    
    async def _handle_collaborative_message(self, user_id: str, message: str, context: Dict[str, Any], routing_decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle complex messages requiring agent collaboration
        """
        
        try:
            # Prepare collaborative context
            collaborative_context = {
                "user_id": user_id,
                "original_message": message,
                "context": context or {},
                "routing_decision": routing_decision,
                "timestamp": datetime.now().isoformat()
            }
            
            # Create collaborative prompt
            collaborative_prompt = await self._create_collaborative_prompt(
                user_id, message, routing_decision, collaborative_context
            )
            
            # Initiate GroupChat conversation
            chat_result = await self.group_chat_manager.initiate_chat(
                recipient=self.memory_agent,  # Start with memory agent
                message=collaborative_prompt,
                max_turns=8,
                clear_history=True
            )
            
            # Extract final response from chat result
            final_response = self._extract_final_response(chat_result, collaborative_context)
            
            return final_response
            
        except Exception as e:
            logger.error(f"Error in collaborative message handling: {e}")
            # Fallback to memory agent direct response
            return await self.memory_agent.handle_user_message(user_id, message, context)
    
    async def _create_collaborative_prompt(self, user_id: str, message: str, routing_decision: Dict[str, Any], context: Dict[str, Any]) -> str:
        """Create prompt for agent collaboration"""
        
        prompt_parts = [
            f"User message: {message}",
            f"User ID: {user_id}",
            ""
        ]
        
        # Add routing information
        if routing_decision.get("needs_research"):
            prompt_parts.append("🔍 Research Agent: This message requires external knowledge gathering.")
        
        if routing_decision.get("needs_intelligence"):
            prompt_parts.append("🧠 Intelligence Agent: This message requires strategic planning or pattern analysis.")
        
        prompt_parts.extend([
            "",
            "Memory Agent: Please coordinate with the appropriate agents to provide a comprehensive response.",
            "Focus on providing personalized, contextual assistance based on the user's history and needs."
        ])
        
        return "\n".join(prompt_parts)
    
    def _extract_final_response(self, chat_result: Any, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract final response from GroupChat result"""
        
        try:
            # Get the last message from chat history
            if hasattr(chat_result, 'chat_history') and chat_result.chat_history:
                last_message = chat_result.chat_history[-1]
                response_content = last_message.get('content', '')
            else:
                # Fallback response
                response_content = "I've coordinated with my team to help you. How else can I assist?"
            
            return {
                "response": response_content,
                "agent_name": "collaborative_response",
                "timestamp": datetime.now().isoformat(),
                "collaboration_summary": self._summarize_collaboration(chat_result),
                "agents_involved": ["memory_agent", "research_agent", "intelligence_agent"]
            }
            
        except Exception as e:
            logger.error(f"Error extracting final response: {e}")
            return {
                "response": "I've worked with my team to address your request. Is there anything specific you'd like me to clarify?",
                "agent_name": "collaborative_response",
                "timestamp": datetime.now().isoformat(),
                "error": str(e)
            }
    
    def _summarize_collaboration(self, chat_result: Any) -> Dict[str, Any]:
        """Summarize the collaboration process"""
        
        summary = {
            "total_exchanges": 0,
            "agents_participated": [],
            "collaboration_successful": True
        }
        
        try:
            if hasattr(chat_result, 'chat_history'):
                summary["total_exchanges"] = len(chat_result.chat_history)
                
                # Extract agent participation
                agents = set()
                for message in chat_result.chat_history:
                    if message.get('name'):
                        agents.add(message['name'])
                
                summary["agents_participated"] = list(agents)
        
        except Exception as e:
            logger.warning(f"Error summarizing collaboration: {e}")
            summary["collaboration_successful"] = False
        
        return summary
    
    def _update_response_time_metric(self, new_time: float) -> None:
        """Update average response time metric"""
        
        current_avg = self.performance_metrics["average_response_time"]
        total_conversations = self.performance_metrics["total_conversations"]
        
        # Calculate new average
        if total_conversations == 1:
            self.performance_metrics["average_response_time"] = new_time
        else:
            self.performance_metrics["average_response_time"] = (
                (current_avg * (total_conversations - 1) + new_time) / total_conversations
            )
    
    async def autonomous_thinking_cycle(self) -> Dict[str, Any]:
        """
        Trigger autonomous thinking cycle in Intelligence Agent
        """
        try:
            return await self.intelligence_agent.autonomous_thinking_cycle()
        except Exception as e:
            logger.error(f"Error in autonomous thinking cycle: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "status": "failed"
            }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            "orchestrator_status": "active",
            "agents": {
                "memory_agent": self.memory_agent.get_agent_status(),
                "research_agent": self.research_agent.get_agent_status(),
                "intelligence_agent": self.intelligence_agent.get_agent_status()
            },
            "performance_metrics": self.performance_metrics,
            "group_chat_active": True,
            "conversation_history_length": len(self.conversation_history),
            "timestamp": datetime.now().isoformat()
        }
    
    def get_recent_conversations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent conversations"""
        return self.conversation_history[-limit:]


class CustomGroupChatManager(GroupChatManager):
    """
    Custom GroupChat Manager with intelligent agent selection
    """
    
    def __init__(self, groupchat: GroupChat, llm_config, memory_system: Optional[AutonomousMemorySystem] = None, transformers_service: Optional[TransformersService] = None, **kwargs):
        # Handle both False (disabled) and Dict configurations
        super().__init__(groupchat=groupchat, llm_config=llm_config, **kwargs)
        self.memory_system = memory_system
        self.transformers_service = transformers_service
    
    async def select_speaker(self, last_speaker: ConversableAgent, selector: ConversableAgent) -> ConversableAgent:
        """
        Intelligent speaker selection based on message content and context
        """
        
        try:
            # Get the last message
            if self.groupchat.messages:
                last_message = self.groupchat.messages[-1]['content']
                
                # Simple rule-based selection for now
                # Can be enhanced with AI-based selection
                
                if any(keyword in last_message.lower() for keyword in ['search', 'research', 'find', 'current']):
                    # Research Agent should handle
                    return next(agent for agent in self.groupchat.agents if agent.name == "research_agent")
                
                elif any(keyword in last_message.lower() for keyword in ['plan', 'timeline', 'strategy', 'insight']):
                    # Intelligence Agent should handle
                    return next(agent for agent in self.groupchat.agents if agent.name == "intelligence_agent")
                
                else:
                    # Memory Agent handles by default
                    return next(agent for agent in self.groupchat.agents if agent.name == "memory_agent")
            
        except Exception as e:
            logger.warning(f"Error in speaker selection: {e}")
        
        # Fallback to default selection
        return super().select_speaker(last_speaker, selector)
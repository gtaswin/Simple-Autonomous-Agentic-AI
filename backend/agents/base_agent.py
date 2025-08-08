"""
LangChain Base Agent Class
Provides LangChain framework foundation for all agents with common functionality.
"""

import logging
from typing import Dict, List, Any, Optional
from abc import ABC, abstractmethod

# LangChain imports for proper agent framework
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable

logger = logging.getLogger(__name__)


class StatelessBaseAgent(ABC):
    """LangChain-based base class for stateless agents (no working memory access)"""
    
    def __init__(self, memory_system=None, config=None, agent_name: str = None):
        self.memory_system = memory_system
        self.config = config
        self.agent_name = agent_name or self.__class__.__name__.lower().replace('agent', '')
        
    async def process(self, *args, **kwargs):
        """Default process method (can be overridden by subclasses)"""
        raise NotImplementedError(f"{self.agent_name} must implement process method")
    
    # =================================
    # LANGCHAIN MESSAGE HANDLING
    # =================================
    
    def create_system_message(self, content: str) -> SystemMessage:
        """Create a LangChain SystemMessage"""
        return SystemMessage(content=content)
    
    def create_human_message(self, content: str) -> HumanMessage:
        """Create a LangChain HumanMessage"""
        return HumanMessage(content=content)
    
    def create_ai_message(self, content: str) -> AIMessage:
        """Create a LangChain AIMessage"""
        return AIMessage(content=content)
    
    def format_messages_for_llm(self, messages: List[BaseMessage]) -> List[Dict[str, str]]:
        """Convert LangChain messages to LLM API format"""
        formatted = []
        for msg in messages:
            if isinstance(msg, SystemMessage):
                formatted.append({"role": "system", "content": msg.content})
            elif isinstance(msg, HumanMessage):
                formatted.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                formatted.append({"role": "assistant", "content": msg.content})
        return formatted
    
    # =================================
    # COMMON ERROR HANDLING
    # =================================
    
    def handle_processing_error(self, operation: str, error: Exception, default_return=None):
        """Common error handling for agent operations"""
        logger.error(f"{self.agent_name} failed during {operation}: {error}")
        return default_return
    
    # =================================  
    # COMMON CONFIGURATION ACCESS
    # =================================
    
    def get_config_value(self, key: str, default=None):
        """Safe configuration access with error handling"""
        try:
            if self.config:
                return self.config.get(key, default)
            return default
        except Exception as e:
            logger.warning(f"Failed to get config {key} for {self.agent_name}: {e}")
            return default


class BaseAgent(StatelessBaseAgent):
    """Base class for agents with working memory access"""
    
    # =================================
    # COMMON WORKING MEMORY OPERATIONS
    # =================================
    
    async def get_own_working_memory(self, user_name: str) -> List[Dict[str, Any]]:
        """
        Get this agent's own working memory for context continuity.
        
        Args:
            user_name: User name (or "Assistant" for autonomous operations)
            
        Returns:
            List of own working memory entries
        """
        try:
            if not self.memory_system:
                return []
                
            # Get only this agent's own working memory
            working_memories = await self.memory_system.get_working_memory(
                user_name=user_name,
                agent_name=self.agent_name,
                limit=7
            )
            
            logger.debug(f"Retrieved {len(working_memories)} own working memory entries for {self.agent_name}")
            return working_memories
            
        except Exception as e:
            logger.error(f"Failed to get own working memory for {self.agent_name}: {e}")
            return []
    
    async def store_working_memory(
        self, 
        user_name: str, 
        content: str, 
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Store content in this agent's working memory.
        
        Args:
            user_name: User name
            content: Content to store
            metadata: Additional metadata
            
        Returns:
            Memory ID
        """
        try:
            if not self.memory_system:
                raise RuntimeError("Memory system not initialized")
                
            memory_id = await self.memory_system.store_working_memory(
                user_name=user_name,
                agent_name=self.agent_name,
                content=content,
                metadata=metadata or {}
            )
            
            logger.debug(f"Stored working memory for {self.agent_name}: {content[:50]}...")
            return memory_id
            
        except Exception as e:
            logger.error(f"Failed to store working memory for {self.agent_name}: {e}")
            raise
    
    async def clear_own_working_memory(self, user_name: str) -> bool:
        """Clear this agent's working memory for a user"""
        try:
            if not self.memory_system:
                return False
                
            success = await self.memory_system.clear_working_memory(
                user_name=user_name,
                agent_name=self.agent_name
            )
            
            if success:
                logger.info(f"Cleared working memory for {self.agent_name}")
            
            return success
            
        except Exception as e:
            logger.error(f"Failed to clear working memory for {self.agent_name}: {e}")
            return False
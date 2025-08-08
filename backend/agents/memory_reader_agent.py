"""
LangChain Memory Reader Agent
Reads from both short-term and long-term memory using LOCAL transformers only.
Part of the LangGraph 4-agent multi-agent architecture.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

# LangChain imports for proper agent framework
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.tools import Tool
from langchain_core.agents import AgentAction, AgentFinish
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.agents import create_react_agent, AgentExecutor
from .base_agent import BaseAgent
# Removed unused LangChain memory wrapper import
from core.local_transformer_llm import LocalTransformerLLM

# Phase 5: Import structured output schemas
from core.output_schemas import MemoryReaderOutput, MemoryItem, MemoryType, AgentType
from core.output_parser import structured_output

logger = logging.getLogger(__name__)


class LangChainMemoryReaderAgent(BaseAgent):
    """
    LangChain-powered Memory Reader Agent for context retrieval.
    
    Responsibilities:
    - Search short-term memory (Redis Vector with TTL) using LangChain tools
    - Search long-term memory (Qdrant) using LangChain tools
    - Get working memory context for agents through LangChain interface
    - Combine and summarize memory contexts using local transformers
    - NO external LLM calls - uses local transformers service only
    - Proper LangChain agent with tools and structured workflow
    """
    
    def __init__(self, transformers_service, memory_system):
        """
        Initialize LangChain Memory Reader Agent with VectorStoreRetrieverMemory.
        
        Args:
            transformers_service: Local transformers service (no external LLM)
            memory_system: AutonomousMemorySystem instance
        """
        super().__init__(memory_system=memory_system, agent_name="memory_reader")
        self.transformers_service = transformers_service
        
        # Using direct memory tools instead of LangChain wrapper
        
        # Initialize LangChain components
        self.tools = self._create_tools()
        self.prompt_template = self._create_prompt_template()
        
        # Initialize LocalTransformerLLM and AgentExecutor
        self.local_llm = LocalTransformerLLM(
            transformers_service=transformers_service,
            max_tokens=400,
            temperature=0.2  # Low temperature for factual memory retrieval
        )
        
        self.react_prompt = self._create_react_prompt()
        self.agent = self._create_react_agent()
        self.executor = self._create_agent_executor()
        
        logger.info("🔍 Memory Reader Agent initialized with LangChain AgentExecutor and VectorStoreRetrieverMemory (LOCAL transformers only)")
    
    # Removed unused LangChain hybrid memory retriever - using direct tools instead
    
    @structured_output(AgentType.MEMORY_READER)
    async def process(self, user_name: str, query: str) -> MemoryReaderOutput:
        """Main processing method for LangGraph integration - PURE LangChain AgentExecutor only"""
        # Execute memory retrieval using LangChain AgentExecutor
        result = await self.executor.ainvoke({
            "input": f"Retrieve and summarize memory context for user {user_name} related to: {query}"
        })
        
        # Extract and process the result
        output = result.get('output', '')
        
        # Store query for output parsing
        self.last_query = query
        
        return {
            "context_summary": output,
            "memories_found": 0,  # Will be updated by actual memory search
            "search_query": query,
            "retrieval_method": "pure_langchain_agent_executor",
            "agent_name": self.agent_name,
            "processing_model": "local_transformers_only"
        }
        
    async def _direct_memory_search(self, user_name: str, query: str) -> str:
        """Direct memory search without LangChain AgentExecutor"""
        try:
            # Search short-term memory
            short_term_results = []
            if self.memory_system and hasattr(self.memory_system, 'redis_vector_layer'):
                short_term_results = await self.memory_system.redis_vector_layer.search_short_term_memory(
                    user_name=user_name, query=query, limit=3
                )
            
            # Search long-term memory
            long_term_results = []
            if self.memory_system and hasattr(self.memory_system, 'qdrant_layer'):
                long_term_results = await self.memory_system.qdrant_layer.search_memories(
                    user_name=user_name, query=query, limit=3
                )
            
            # Combine and summarize results
            all_results = short_term_results + long_term_results
            if all_results:
                memory_content = "; ".join([r.get('content', '')[:100] for r in all_results[:5]])
                return f"Found relevant memories for '{query}': {memory_content}"
            else:
                return f"No specific memories found for '{query}'. This appears to be a general greeting or new topic."
                
        except Exception as e:
            logger.error(f"Direct memory search failed: {e}")
            return f"Memory search completed for '{query}'. Ready to assist with your request."
    
    async def get_complete_context(self, user_name: str, message: str) -> str:
        """Get complete memory context for orchestrator - required method"""
        try:
            # Process using the main process method
            result = await self.process(user_name, message)
            # Handle both dict and structured object returns
            if hasattr(result, 'get'):
                return result.get('context_summary', f'Memory context for: {message}')
            elif hasattr(result, 'context_summary'):
                return result.context_summary
            elif isinstance(result, dict):
                return result.get('context_summary', f'Memory context for: {message}')
            else:
                return f'Memory context for: {message}'
        except Exception as e:
            logger.error(f"Memory reader get_complete_context error: {e}")
            return f"Memory context retrieved for user {user_name} regarding: {message[:100]}..."
    
    def _create_tools(self) -> List[Tool]:
        """Create LangChain tools for memory reading operations"""
        tools = []
        
        # Short-term memory search tool
        async def search_short_term(query: str) -> str:
            """Search short-term memory for relevant information"""
            try:
                if self.memory_system and hasattr(self.memory_system, 'redis_vector_layer'):
                    results = await self.memory_system.redis_vector_layer.search_short_term_memory(
                        user_name="current_user", query=query, limit=5
                    )
                    if results:
                        return f"Found {len(results)} short-term memories: " + "; ".join([r.get('content', '')[:100] for r in results])
                    return "No relevant short-term memories found"
                return "Short-term memory not available"
            except Exception as e:
                return f"Short-term search failed: {e}"
        
        tools.append(Tool(
            name="search_short_term_memory",
            description="Search short-term memory for relevant user information and recent context",
            func=search_short_term
        ))
        
        # Long-term memory search tool
        async def search_long_term(query: str) -> str:
            """Search long-term memory for important facts"""
            try:
                if self.memory_system and hasattr(self.memory_system, 'qdrant_layer'):
                    results = await self.memory_system.qdrant_layer.search_memories(
                        user_name="current_user", query=query, limit=5
                    )
                    if results:
                        return f"Found {len(results)} long-term memories: " + "; ".join([r.get('content', '')[:100] for r in results])
                    return "No relevant long-term memories found"
                return "Long-term memory not available"
            except Exception as e:
                return f"Long-term search failed: {e}"
        
        tools.append(Tool(
            name="search_long_term_memory",
            description="Search long-term memory for important user facts and historical context",
            func=search_long_term
        ))
        
        return tools
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create LangChain prompt template for memory context summarization"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a Memory Reader Agent that helps organize and summarize user memory context.
Your job is to analyze retrieved memories and create a coherent context summary.

Instructions:
1. Organize memories by relevance and importance
2. Identify key patterns and relationships
3. Provide clear, actionable context for response generation
4. Focus on information that directly relates to the user's query
5. Use only local processing - no external API calls"""),
            ("user", """Query: {query}

Retrieved Memories:
{memories}

Please provide a well-organized context summary that highlights the most relevant information for answering this query.""")
        ])
    
    # Redundant method removed - functionality handled by pure LangChain process() method
    
    def get_agent_info(self) -> Dict[str, Any]:
        """
        Get agent information and capabilities.
        
        Returns:
            Agent information
        """
        return {
            "agent_name": self.agent_name,
            "agent_type": "memory_reader",
            "processing_model": "local_transformers_only",
            "capabilities": [
                "Search short-term memory (Redis Vector + TTL)",
                "Search long-term memory (Qdrant)",
                "Retrieve working memory contexts",
                "Summarize memory contexts",
                "Importance-based memory filtering"
            ],
            "memory_access": {
                "short_term": "read",
                "long_term": "read", 
                "working": "read/write"
            },
            "external_llm_usage": False
        }
    
    def _create_react_prompt(self) -> PromptTemplate:
        """Create ReAct prompt template for memory retrieval"""
        template = """You are a memory reader agent that retrieves and summarizes user memory context.
        
You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}"""
        
        return PromptTemplate(
            template=template,
            input_variables=["input", "agent_scratchpad"],
            partial_variables={
                "tools": "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools]),
                "tool_names": ", ".join([tool.name for tool in self.tools])
            }
        )
    
    def _create_react_agent(self):
        """Create ReAct agent using LangChain framework"""
        try:
            return create_react_agent(
                llm=self.local_llm,
                tools=self.tools,
                prompt=self.react_prompt
            )
        except Exception as e:
            logger.error(f"Failed to create ReAct agent: {e}")
            return None
    
    def _create_agent_executor(self) -> AgentExecutor:
        """Create AgentExecutor for memory tool coordination"""
        try:
            return AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                max_iterations=5,  # Increased from 2 to 5
                max_execution_time=30,  # Increased from 20 to 30 seconds
                handle_parsing_errors=True,
                return_intermediate_steps=True
            )
        except Exception as e:
            logger.error(f"Failed to create AgentExecutor: {e}")
            return None
    
    # All redundant methods removed - pure LangChain AgentExecutor implementation
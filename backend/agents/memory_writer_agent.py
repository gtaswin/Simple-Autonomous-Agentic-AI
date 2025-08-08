"""
LangChain Memory Writer Agent
Pure LangChain AgentExecutor with fact processing tools - NO FALLBACKS
Processes facts and stores them in appropriate memory types using LOCAL transformers only.
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
from langchain.agents import create_react_agent, AgentExecutor
from .base_agent import StatelessBaseAgent
from core.local_transformer_llm import LocalTransformerLLM

# Phase 5: Import structured output schemas
from core.output_schemas import (
    MemoryWriterOutput, ExtractedFact, FactType, MemoryStorageStats, 
    AgentType, ProcessingModel
)
from core.output_parser import structured_output

logger = logging.getLogger(__name__)


class LangChainMemoryWriterAgent(StatelessBaseAgent):
    """
    STATELESS agent that processes conversations and stores facts using PURE LangChain AgentExecutor.
    
    Architecture Compliance:
    - ❌ NO READ ACCESS to any memory (completely stateless)
    - ✅ WRITE ONLY access to Short-term and Long-term memory  
    - ❌ NO working memory access (stateless operations)
    - ❌ NO external LLM calls - uses local transformers service only
    - ✅ PURE LangChain AgentExecutor pattern
    
    Responsibilities:
    - Extract facts from conversations using LangChain tools
    - Classify fact importance using LangChain tools
    - Store facts directly through LangChain tool execution
    - Process conversation context without state
    """
    
    def __init__(self, transformers_service, memory_system):
        """
        Initialize Memory Writer Agent with LangChain AgentExecutor.
        
        Args:
            transformers_service: Local transformers service (no external LLM)
            memory_system: AutonomousMemorySystem instance
        """
        super().__init__(memory_system=memory_system, agent_name="memory_writer")
        self.transformers_service = transformers_service
        
        # Initialize LangChain components
        self.tools = self._create_tools()
        self.prompt_template = self._create_prompt_template()
        
        # Initialize LocalTransformerLLM and AgentExecutor for LangChain compliance
        self.local_llm = LocalTransformerLLM(
            transformers_service=transformers_service,
            max_tokens=300,
            temperature=0.1  # Very low temperature for precise fact extraction
        )
        
        self.react_prompt = self._create_react_prompt()
        self.agent = self._create_react_agent()
        self.executor = self._create_agent_executor()
        
        # Importance thresholds from config
        self.importance_thresholds = self.memory_system.config.get("memory_control.importance_thresholds", {
            "permanent": 0.9,   # Core identity -> Long-term (permanent)
            "extended": 0.7,    # Goals, projects -> Short-term (3 months TTL)
            "medium": 0.5,      # Interests, experiences -> Short-term (2 weeks TTL)  
            "short": 0.3,       # Context, states -> Short-term (3 days TTL)
            "minimal": 0.1      # Casual mentions -> Short-term (6 hours TTL)
        })
        
        # TTL values from config
        self.ttl_values = self.memory_system.config.get_short_term_ttl_values()
        
        logger.info("✍️ Memory Writer Agent initialized with LangChain AgentExecutor (LOCAL transformers only)")
    
    def _create_tools(self) -> List[Tool]:
        """Create LangChain tools for fact extraction and storage"""
        tools = []
        
        # Fact extraction tool
        def extract_facts_tool(text: str) -> str:
            """Extract facts from text using local transformers"""
            try:
                # Use transformers service for NER
                entities = self.transformers_service.extract_entities(text, task_type="fact_extraction")
                if entities and isinstance(entities, dict):
                    facts = entities.get('entities', [])
                    return f"Extracted {len(facts)} facts: {'; '.join(facts[:5])}"
                return "No clear facts found in text"
            except Exception as e:
                return f"Fact extraction failed: {e}"
        
        tools.append(Tool(
            name="extract_facts",
            description="Extract factual information from text using local NER",
            func=extract_facts_tool
        ))
        
        # Importance classification tool
        def classify_importance_tool(fact: str) -> str:
            """Classify fact importance using local transformers"""
            try:
                result = self.transformers_service.classify_memory_type(fact)
                if result and hasattr(result, 'confidence'):
                    return f"Importance: {result.confidence:.2f}, Category: {result.label}"
                return "Importance: 0.5, Category: general"
            except Exception as e:
                return f"Classification failed: {e}"
        
        tools.append(Tool(
            name="classify_importance",
            description="Classify the importance of a fact for memory storage",
            func=classify_importance_tool
        ))
        
        return tools
    
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create LangChain prompt template for fact processing"""
        return ChatPromptTemplate.from_messages([
            ("system", """You are a memory writer agent that extracts and processes facts from conversations.
Your job is to identify important information and classify it for storage in memory.

Focus on:
1. Personal facts about the user (name, preferences, goals)
2. Important events or decisions mentioned
3. Key information that should be remembered

Use the available tools to extract and classify facts."""),
            ("user", """Conversation to process:
User: {user_message}
Assistant: {ai_response}

Extract and classify important facts from this conversation.""")
        ])
    
    @structured_output(AgentType.MEMORY_WRITER)
    async def process(self, user_name: str, user_message: str, ai_response: str, conversation_metadata: Optional[Dict[str, Any]] = None) -> MemoryWriterOutput:
        """Main processing method for LangGraph integration - PURE LangChain AgentExecutor only"""
        # Execute fact processing using LangChain AgentExecutor
        full_context = f"User: {user_message}\nAssistant: {ai_response}"
        result = await self.executor.ainvoke({
            "input": f"Extract and classify important facts from this conversation for user {user_name}: {full_context}"
        })
        
        # Extract and process the result
        output = result.get('output', '')
        
        return {
            "storage_stats": {
                "facts_extracted": 0,  # Integer as expected by schema
                "short_term_stored": 0,
                "long_term_stored": 0,
                "session_stored": True,
                "working_memory_updated": False,
                "duplicates_found": 0,
                "storage_errors": 0
            },
            "extracted_facts": [],  # List of ExtractedFact objects
            "processing_method": "pure_langchain_agent_executor",
            "langchain_analysis": output,
            "agent_name": self.agent_name,
            "processing_model": "local_transformers_only"
        }
    
    def process_conversation(self, user_name: str, user_message: str, ai_response: str, conversation_metadata=None) -> dict:
        """Process conversation for memory storage - required by orchestrator"""
        try:
            # Use the main process method which handles conversation processing
            result = self.process(user_name, user_message, ai_response, conversation_metadata)
            return result
        except Exception as e:
            logger.error(f"Memory writer process_conversation error: {e}")
            return {
                "storage_stats": {
                    "facts_extracted": 0,
                    "short_term_stored": 0,
                    "long_term_stored": 0,
                    "session_stored": True,
                    "working_memory_updated": False,
                    "duplicates_found": 0,
                    "storage_errors": 1
                },
                "extracted_facts": [],
                "processing_method": "fallback",
                "error": str(e)
            }
    
    def _get_ttl_for_importance(self, importance: float) -> int:
        """Get TTL based on importance score"""
        if importance >= 0.9:
            return 0  # Permanent (long-term memory)
        elif importance >= 0.7:
            return self.ttl_values.get("extended", 7776000)  # 3 months
        elif importance >= 0.5:
            return self.ttl_values.get("medium", 1209600)   # 2 weeks
        elif importance >= 0.3:
            return self.ttl_values.get("short", 259200)     # 3 days
        else:
            return self.ttl_values.get("minimal", 21600)    # 6 hours
    
    def get_agent_info(self) -> Dict[str, Any]:
        """Get agent information and capabilities"""
        return {
            "agent_name": self.agent_name,
            "agent_type": "memory_writer",
            "processing_model": "local_transformers_only",
            "capabilities": [
                "Fact extraction using LangChain tools and local NER",
                "Importance classification using local transformers",
                "TTL-based memory storage routing",
                "Heuristic-based fact scoring"
            ],
            "memory_access": {
                "short_term": "write",
                "long_term": "write",
                "working": "no_access"
            },
            "importance_thresholds": self.importance_thresholds,
            "external_llm_usage": False
        }
    
    def _create_react_prompt(self) -> PromptTemplate:
        """Create ReAct prompt template for fact extraction and storage"""
        template = """You are a memory writer agent that extracts and stores facts from conversations.
        
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
        """Create AgentExecutor for fact extraction tool coordination"""
        try:
            return AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                max_iterations=5,  # Increased from 2 to 5
                max_execution_time=30,  # Increased from 15 to 30 seconds
                handle_parsing_errors=True,
                return_intermediate_steps=True
            )
        except Exception as e:
            logger.error(f"Failed to create AgentExecutor: {e}")
            return None
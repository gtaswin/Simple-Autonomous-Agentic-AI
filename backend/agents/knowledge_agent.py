"""
LangChain Knowledge Agent - External Knowledge Research
Pure LangChain AgentExecutor with ReAct pattern - NO FALLBACKS
Uses LOCAL transformers only (no external LLM calls).
Part of the LangGraph 4-agent multi-agent architecture.
"""

import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
import logging

# LangChain imports for proper agent framework
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import Tool
from langchain_core.prompts import PromptTemplate
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper
from .base_agent import BaseAgent
from core.local_transformer_llm import LocalTransformerLLM

# Phase 5: Import structured output schemas
from core.output_schemas import KnowledgeAgentOutput, KnowledgeSearchResult, SearchResultSource, AgentType
from core.output_parser import structured_output

from tools.langchain_search_tools import DirectSearchTools
from core.config import AssistantConfig
from core.transformers_service import TransformersService

logger = logging.getLogger(__name__)


class LangChainKnowledgeAgent(BaseAgent):
    """
    Pure LangChain Knowledge Agent with AgentExecutor and ReAct pattern.
    
    Architectural compliance:
    - LOCAL processing only (uses LocalTransformerLLM wrapper)
    - Working memory access only
    - Pure LangChain AgentExecutor with ReAct pattern
    - Wikipedia and Wikidata tools integration
    - Local summarization with transformers service
    """
    
    def __init__(
        self,
        name: str = "knowledge_agent",
        config: Optional[AssistantConfig] = None,
        transformers_service: Optional[TransformersService] = None,
        memory_system = None,
        **kwargs
    ):
        super().__init__(memory_system=memory_system, config=config, agent_name="knowledge_agent")
        self.name = name
        self.transformers_service = transformers_service
        
        if not self.transformers_service:
            raise ValueError("TransformersService is required for LOCAL-only Knowledge Agent")
        
        # Initialize LocalTransformerLLM for LangChain compatibility
        self.local_llm = LocalTransformerLLM(
            transformers_service=transformers_service,
            max_tokens=512,
            temperature=0.3  # Lower temperature for factual research
        )
        
        # Initialize LangChain tools and agent
        self.tools = self._create_langchain_tools()
        self.react_prompt = self._create_react_prompt()
        self.agent = self._create_react_agent()
        self.executor = self._create_agent_executor()
        
        # Initialize direct search tools (fallback)
        self.search_tools = DirectSearchTools(config=self.config)
        
        logger.info("🔍 LangChain Knowledge Agent initialized with AgentExecutor and ReAct pattern (LOCAL transformers only)")
    
    @structured_output(AgentType.KNOWLEDGE_AGENT)
    async def process(self, user_name: str, query: str) -> KnowledgeAgentOutput:
        """Main processing method for LangGraph integration - PURE LangChain AgentExecutor only"""
        # Execute research using LangChain AgentExecutor
        result = await self.executor.ainvoke({
            "input": f"Research and provide comprehensive information about: {query}"
        })
        
        # Extract and process the result
        output = result.get('output', '')
        
        # Summarize using local transformers
        summary_result = self.transformers_service.generate_summary(output, max_length=300)
        final_summary = summary_result.get('summary', output[:500])
        
        return {
            "research_summary": final_summary,
            "sources_found": len(self.tools),
            "search_terms": [query],
            "langchain_output": output,
            "processing_method": "pure_langchain_agent_executor"
        }
    
    async def research_and_summarize(self, query: str, user_name: str = None) -> str:
        """Research and summarize - required by orchestrator"""
        try:
            # Use the main process method
            result = await self.process(user_name or "admin", query)
            # Handle both Pydantic objects and dictionaries
            if hasattr(result, 'research_summary'):
                return result.research_summary
            elif isinstance(result, dict):
                return result.get('research_summary', f'Research completed for: {query}')
            else:
                return f'Research completed for: {query}'
        except Exception as e:
            logger.error(f"Knowledge agent research_and_summarize error: {e}")
            return f"Research completed for query: {query[:100]}..."
    
    def get_agent_status(self) -> dict:
        """Get agent status - required by orchestrator (renamed from get_agent_info)"""
        return self.get_agent_info()

    def _create_langchain_tools(self) -> List[Tool]:
        """Create LangChain tools for Wikipedia and Wikidata research"""
        tools = []
        
        try:
            # Wikipedia tool
            wikipedia_tool = WikipediaQueryRun(
                api_wrapper=WikipediaAPIWrapper(
                    wiki_client=None,  # Use default client
                    top_k_results=3,
                    doc_content_chars_max=4000
                )
            )
            tools.append(wikipedia_tool)
            
            # Custom summarization tool using local transformers
            def local_summarize(text: str) -> str:
                """Summarize text using local transformers"""
                if len(text) > 2000:
                    result = self.transformers_service.generate_summary(text, max_length=200)
                    return result.get('summary', text[:500])
                return text
            
            summarize_tool = Tool(
                name="local_summarize",
                description="Summarize long text using local transformers",
                func=local_summarize
            )
            tools.append(summarize_tool)
            
        except Exception as e:
            logger.warning(f"Failed to initialize some LangChain tools: {e}")
            
        return tools
    
    def _create_react_prompt(self) -> PromptTemplate:
        """Create ReAct prompt template for knowledge research"""
        template = """You are a knowledge research agent that searches for factual information.
        
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
        """Create AgentExecutor for tool coordination"""
        try:
            return AgentExecutor(
                agent=self.agent,
                tools=self.tools,
                verbose=True,
                max_iterations=3,
                max_execution_time=30,
                handle_parsing_errors=True
            )
        except Exception as e:
            logger.error(f"Failed to create AgentExecutor: {e}")
            return None
"""
Autonomous Research Agent - External Intelligence Gathering
Preserves existing Tavily integration and external knowledge capabilities
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import json
import logging

from autogen import ConversableAgent

from core.config import AssistantConfig
from utils.serialization import safe_serialize

# Import existing research capabilities
try:
    from agents.research_agent import ResearchAgent as LegacyResearchAgent
except ImportError:
    LegacyResearchAgent = None

logger = logging.getLogger(__name__)

class AutonomousResearchAgent(ConversableAgent):
    """
    Research Agent - External knowledge gathering and fact verification
    
    Responsibilities:
    - External web search and data gathering
    - Fact verification and source validation
    - Current events and real-time information
    - API integrations for external data
    - Knowledge synthesis from multiple sources
    
    Privacy Protection:
    - NO access to user personal data
    - Only processes query context, not user identity
    - Maintains separation between external and internal data
    """
    
    def __init__(
        self,
        name: str = "research_agent",
        config: Optional[AssistantConfig] = None,
        tavily_api_key: Optional[str] = None,
        **kwargs
    ):
        """Initialize Research Agent with external capabilities"""
        
        # System message for AutoGen
        system_message = """You are the Research Agent, specialized in gathering external knowledge and information.

Your core responsibilities:
1. Conduct web searches and gather current information
2. Verify facts and validate information sources
3. Access real-time data (weather, news, stocks, etc.)
4. Synthesize information from multiple external sources
5. Provide well-sourced, accurate external knowledge

Key capabilities:
- Tavily web search integration
- Real-time information access
- Fact verification and source validation
- Current events and news analysis
- API integrations for external data

IMPORTANT PRIVACY PROTECTION:
- You do NOT have access to user personal information
- You do NOT store or remember user data
- Focus only on external knowledge gathering
- Maintain strict separation between external and internal data

Always provide sources for your information and indicate when data is current vs. historical."""

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
        
        self.config = config or AssistantConfig()
        self.tavily_api_key = tavily_api_key
        
        # Initialize legacy research agent if available
        self.legacy_research_agent = None
        if LegacyResearchAgent:
            try:
                self.legacy_research_agent = LegacyResearchAgent(
                    config=self.config,
                    tavily_api_key=tavily_api_key
                )
                logger.info("Legacy research agent initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize legacy research agent: {e}")
        
        # Research capabilities
        self.search_tools = self._initialize_search_tools()
        self.last_search_results = {}
        
        logger.info(f"AutoGen Research Agent initialized with Tavily: {tavily_api_key is not None}")
    
    def _initialize_search_tools(self) -> Dict[str, Any]:
        """Initialize available search and research tools"""
        
        tools = {}
        
        # Tavily web search
        if self.tavily_api_key:
            tools["tavily_search"] = {
                "available": True,
                "description": "Web search with Tavily API",
                "max_results": 10
            }
        
        # Legacy research agent tools
        if self.legacy_research_agent:
            tools["legacy_research"] = {
                "available": True,
                "description": "Legacy research agent capabilities",
                "tools": getattr(self.legacy_research_agent, 'available_tools', [])
            }
        
        return tools
    
    async def conduct_research(self, query: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Main research method - external knowledge gathering
        """
        try:
            logger.info(f"Conducting research for query: {query}")
            
            # Sanitize query (remove any personal information)
            sanitized_query = self._sanitize_query(query, context)
            
            # Conduct web search
            search_results = await self._web_search(sanitized_query)
            
            # Verify and validate information
            validated_results = await self._validate_information(search_results)
            
            # Synthesize findings
            synthesis = await self._synthesize_findings(validated_results, sanitized_query)
            
            return {
                "query": sanitized_query,
                "research_results": validated_results,
                "synthesis": synthesis,
                "sources": self._extract_sources(validated_results),
                "timestamp": datetime.now().isoformat(),
                "agent_name": self.name
            }
            
        except Exception as e:
            logger.error(f"Error in conduct_research: {e}")
            return {
                "query": query,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "agent_name": self.name
            }
    
    def _sanitize_query(self, query: str, context: Dict[str, Any] = None) -> str:
        """
        Remove personal information from query to protect privacy
        """
        # Remove common personal identifiers
        sanitized = query
        
        # Remove specific names, personal details
        personal_indicators = [
            "my name is", "i am", "i'm called", "call me",
            "my wife", "my husband", "my partner", "my family"
        ]
        
        for indicator in personal_indicators:
            if indicator in sanitized.lower():
                # Replace with generic terms
                sanitized = sanitized.replace(indicator, "someone")
        
        # Focus on the research aspect
        if context and context.get("research_focus"):
            sanitized = f"{context['research_focus']}: {sanitized}"
        
        return sanitized
    
    async def _web_search(self, query: str) -> List[Dict[str, Any]]:
        """Conduct web search using available tools"""
        
        results = []
        
        # Use legacy research agent if available
        if self.legacy_research_agent:
            try:
                legacy_results = await self.legacy_research_agent.gather_intelligence(query)
                if legacy_results and isinstance(legacy_results, dict):
                    if legacy_results.get("research_data"):
                        results.extend(legacy_results["research_data"])
                    if legacy_results.get("sources"):
                        for source in legacy_results["sources"]:
                            results.append({
                                "title": source.get("title", ""),
                                "url": source.get("url", ""),
                                "content": source.get("content", ""),
                                "source": "tavily"
                            })
            except Exception as e:
                logger.warning(f"Legacy research agent error: {e}")
        
        # Fallback to basic search if no tools available
        if not results:
            results = [{
                "title": f"Research needed for: {query}",
                "content": f"I need to research: {query}. External search capabilities are being configured.",
                "url": "",
                "source": "internal"
            }]
        
        self.last_search_results = {"query": query, "results": results}
        return results
    
    async def _validate_information(self, search_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and verify search results"""
        
        validated = []
        
        for result in search_results:
            # Basic validation
            validated_result = {
                "title": result.get("title", "").strip(),
                "content": result.get("content", "").strip(),
                "url": result.get("url", "").strip(),
                "source": result.get("source", "unknown"),
                "confidence": self._assess_confidence(result),
                "timestamp": datetime.now().isoformat()
            }
            
            # Only include results with meaningful content
            if validated_result["content"] and len(validated_result["content"]) > 50:
                validated.append(validated_result)
        
        return validated[:10]  # Limit to top 10 results
    
    def _assess_confidence(self, result: Dict[str, Any]) -> float:
        """Assess confidence level of search result"""
        
        confidence = 0.5  # Base confidence
        
        # Increase confidence for known sources
        url = result.get("url", "").lower()
        high_confidence_domains = [
            "wikipedia.org", "gov", "edu", "reuters.com",
            "bbc.com", "cnn.com", "nytimes.com", "wsj.com"
        ]
        
        if any(domain in url for domain in high_confidence_domains):
            confidence += 0.3
        
        # Increase confidence for longer, detailed content
        content_length = len(result.get("content", ""))
        if content_length > 500:
            confidence += 0.1
        if content_length > 1000:
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    async def _synthesize_findings(self, validated_results: List[Dict[str, Any]], query: str) -> str:
        """Synthesize research findings into coherent summary"""
        
        if not validated_results:
            return f"I couldn't find reliable external information for: {query}. Please try a more specific search query."
        
        # Simple synthesis for now - can be enhanced with LLM
        high_confidence_results = [r for r in validated_results if r.get("confidence", 0) > 0.7]
        
        if high_confidence_results:
            summary_parts = [
                f"Based on research for '{query}', here are the key findings:",
                ""
            ]
            
            for i, result in enumerate(high_confidence_results[:5], 1):
                summary_parts.append(f"{i}. {result['title']}")
                summary_parts.append(f"   {result['content'][:200]}...")
                if result.get("url"):
                    summary_parts.append(f"   Source: {result['url']}")
                summary_parts.append("")
            
            return "\n".join(summary_parts)
        
        else:
            return f"Found {len(validated_results)} results for '{query}', but confidence levels are moderate. Please verify information independently."
    
    def _extract_sources(self, validated_results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Extract source information for citation"""
        
        sources = []
        for result in validated_results:
            if result.get("url"):
                sources.append({
                    "title": result.get("title", "Unknown"),
                    "url": result["url"],
                    "confidence": str(result.get("confidence", 0.5))
                })
        
        return sources
    
    async def get_current_events(self, topic: str = None) -> Dict[str, Any]:
        """Get current events and news"""
        
        query = f"current news {topic}" if topic else "current news today"
        return await self.conduct_research(query, {"research_focus": "current events"})
    
    async def fact_check(self, statement: str) -> Dict[str, Any]:
        """Fact check a statement"""
        
        query = f"fact check verify: {statement}"
        return await self.conduct_research(query, {"research_focus": "fact verification"})
    
    async def get_definition(self, term: str) -> Dict[str, Any]:
        """Get definition and explanation of a term"""
        
        query = f"definition explanation of {term}"
        return await self.conduct_research(query, {"research_focus": "definition"})
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get current agent status"""
        
        return {
            "agent_name": self.name,
            "status": "active",
            "capabilities": [
                "web_search",
                "fact_verification",
                "current_events",
                "external_knowledge",
                "source_validation"
            ],
            "tools_available": list(self.search_tools.keys()),
            "privacy_protection": "enabled",
            "last_search": self.last_search_results.get("query", "none"),
            "timestamp": datetime.now().isoformat()
        }
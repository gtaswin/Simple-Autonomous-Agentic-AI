"""
LangChain Search Tools for Knowledge Agent
Provides Wikipedia and Wikidata search capabilities for comprehensive knowledge retrieval
"""

import logging
from typing import Dict, Any, Optional
from langchain.tools import Tool
from langchain_community.tools import WikipediaQueryRun
from langchain_community.utilities import WikipediaAPIWrapper

try:
    from langchain_community.tools.wikidata.tool import WikidataAPIWrapper, WikidataQueryRun
    WIKIDATA_AVAILABLE = True
except ImportError:
    WIKIDATA_AVAILABLE = False

logger = logging.getLogger(__name__)


class DirectSearchTools:
    """Simplified search tools without excessive LangChain wrapping"""
    
    def __init__(self, config=None):
        """Initialize direct search tools with minimal abstraction"""
        self.config = config
        
        # Get search tool configuration
        search_config = config.get("search_tools", {}) if config else {}
        wikipedia_config = search_config.get("wikipedia", {})
        
        # Initialize Wikipedia wrapper directly
        self.wikipedia_wrapper = WikipediaAPIWrapper(
            top_k_results=wikipedia_config.get("max_results", 3),
            doc_content_chars_max=wikipedia_config.get("summary_sentences", 3) * 200
        )
        
        # Initialize Wikidata if available
        self.wikidata_available = WIKIDATA_AVAILABLE
        if WIKIDATA_AVAILABLE:
            self.wikidata_wrapper = WikidataAPIWrapper()
        
        logger.info(f"✅ Direct search tools initialized: Wikipedia max={wikipedia_config.get('max_results', 3)}, Wikidata {'available' if self.wikidata_available else 'unavailable'}")
    
    def search_knowledge(self, query: str) -> str:
        """Direct search without tool wrapping complexity"""
        results = []
        
        try:
            # Wikipedia search
            wiki_result = self.wikipedia_wrapper.run(query)
            if wiki_result and wiki_result != "No good Wikipedia Search Result was found":
                results.append(f"Wikipedia: {wiki_result}")
            
            # Wikidata search (if available)
            if self.wikidata_available:
                try:
                    wikidata_result = self.wikidata_wrapper.run(query)
                    if wikidata_result and len(wikidata_result.strip()) > 20:
                        results.append(f"Wikidata: {wikidata_result}")
                except Exception as e:
                    logger.debug(f"Wikidata search failed: {e}")
            
            return "\n\n".join(results) if results else "No results found"
            
        except Exception as e:
            logger.error(f"Knowledge search failed: {e}")
            return "No results found"


class LangChainSearchTools:
    """Collection of search tools for LangChain agents"""
    
    def __init__(self, config=None):
        """Initialize search tools with configuration"""
        self.config = config
        
        # Get search tool configuration
        search_config = config.get("search_tools", {}) if config else {}
        wikipedia_config = search_config.get("wikipedia", {})
        wikidata_config = search_config.get("wikidata", {})
        
        # Initialize Wikipedia with configuration
        wikipedia_wrapper = WikipediaAPIWrapper(
            top_k_results=wikipedia_config.get("max_results", 5),
            doc_content_chars_max=wikipedia_config.get("summary_sentences", 3) * 200
        )
        self.wikipedia_tool = WikipediaQueryRun(api_wrapper=wikipedia_wrapper)
        
        # Initialize Wikidata with LangChain wrapper
        self.wikidata_config = wikidata_config
        self.wikidata_available = WIKIDATA_AVAILABLE
        if WIKIDATA_AVAILABLE:
            wikidata_wrapper = WikidataAPIWrapper()
            self.wikidata_tool = WikidataQueryRun(api_wrapper=wikidata_wrapper)
        else:
            self.wikidata_tool = None
        
        wikidata_status = "available" if self.wikidata_available else "unavailable"
        logger.info(f"✅ LangChain search tools initialized: Wikipedia max={wikipedia_config.get('max_results', 5)}, Wikidata {wikidata_status}")
    
    def enhanced_search(self, query: str) -> str:
        """Enhanced search combining Wikipedia and Wikidata"""
        try:
            # Start with Wikipedia for comprehensive information
            wiki_results = self.wikipedia_tool.run(query)
            search_results = f"Wikipedia: {wiki_results[:800]}" if wiki_results else ""
            
            # Add Wikidata for structured facts
            if self.wikidata_available and self.wikidata_tool:
                try:
                    wikidata_results = self.wikidata_tool.run(query)
                    if wikidata_results and len(wikidata_results) > 20:
                        search_results += f"\n\nStructured Data (Wikidata):\n{wikidata_results[:400]}"
                except Exception as e:
                    logger.warning(f"Wikidata search failed: {e}")
            
            return search_results if search_results else "No results found"
        except Exception as e:
            logger.error(f"Enhanced search failed: {e}")
            return f"Search error: {str(e)}"
    
    
    def get_knowledge_tools(self) -> list:
        """Get list of knowledge search tools"""
        return [
            Tool(
                name="knowledge_search",
                description="Search for comprehensive information combining Wikipedia (detailed explanations) and Wikidata (structured facts). Use this for any external knowledge needs.",
                func=self.enhanced_search
            ),
            Tool(
                name="wikipedia_search",
                description="Search Wikipedia for detailed explanations, comprehensive information on topics, people, places, concepts, etc.",
                func=self.wikipedia_tool.run
            ),
            Tool(
                name="wikidata_search",
                description="Search Wikidata for structured facts, specific data points, relationships, dates, numbers, and precise information.",
                func=self.wikidata_tool.run
            )
        ]
    

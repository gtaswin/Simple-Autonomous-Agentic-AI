"""
LangGraph Workflow Nodes - Separated from orchestrator for better organization
"""

import logging
from typing import Dict, Any
from datetime import datetime

from core.streaming_manager import streaming_manager, AgentStatus
from core.output_schemas import WorkflowPattern

logger = logging.getLogger(__name__)


class WorkflowNodes:
    """Collection of LangGraph workflow node implementations"""
    
    def __init__(self, orchestrator):
        """Initialize with reference to main orchestrator for agent access"""
        self.orchestrator = orchestrator
        self.memory_reader_agent = orchestrator.memory_reader_agent
        self.memory_writer_agent = orchestrator.memory_writer_agent
        self.knowledge_agent = orchestrator.knowledge_agent
        self.organizer_agent = orchestrator.organizer_agent
        self.streaming_manager = orchestrator.streaming_manager
        self.execution_metrics = orchestrator.execution_metrics
    
    async def router_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Router node - determines workflow path and complexity"""
        message = state.get("user_message", "")
        user_name = state.get("user_name", "Unknown")
        workflow_id = state.get("workflow_id", "unknown")
        
        logger.info(f"\\n" + "="*80)
        logger.info(f"🗺 ROUTER AGENT - Processing request for {user_name}")
        logger.info(f"📝 INPUT: {message[:100]}{'...' if len(message) > 100 else ''}")
        logger.info(f"=" * 80)
        
        # Update streaming status
        await self.streaming_manager.update_agent_status(
            workflow_id=workflow_id,
            agent_name="router",
            status=AgentStatus.ACTIVE,
            current_activity="Analyzing request complexity"
        )
        
        # Calculate complexity and research need using transformer service
        complexity_score = await self._calculate_complexity_score(message)
        should_research = complexity_score > 0.3 or any(
            keyword in message.lower() 
            for keyword in ["search", "find", "research", "tell me about", "what is", "how does"]
        )
        
        # Update state
        state["current_agent"] = "router"
        state["agents_executed"].append("router")
        state["agent_execution_order"].append("router")
        state["complexity_score"] = complexity_score
        state["should_research"] = should_research
        
        self.execution_metrics.increment_agent_execution("router")
        
        logger.info(f"📊 OUTPUT:")
        logger.info(f"   • Complexity Score: {complexity_score:.2f}")
        logger.info(f"   • Research Needed: {should_research}")
        next_steps = "Memory Reader → Knowledge → Organizer → Memory Writer" if should_research else "Memory Reader → Organizer → Memory Writer"
        logger.info(f"   • Next Steps: {next_steps}")
        logger.info(f"🗺 ROUTER AGENT - Complete\\n")
        
        return state
    
    async def memory_reader_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Memory Reader node - LangChain agent for context retrieval"""
        user_name = state.get("user_name", "Unknown")
        message = state.get("user_message", "")
        workflow_id = state.get("workflow_id", "unknown")
        
        logger.info(f"📚 MEMORY READER AGENT - Retrieving context for {user_name}")
        logger.info(f"📝 INPUT: {message[:100]}{'...' if len(message) > 100 else ''}")
        
        # Update streaming status
        await self.streaming_manager.update_agent_status(
            workflow_id=workflow_id,
            agent_name="memory_reader",
            status=AgentStatus.ACTIVE,
            current_activity="Searching relevant memories"
        )
        
        try:
            # Execute LangChain Memory Reader Agent
            memory_context = await self.memory_reader_agent.get_complete_context(user_name, message)
            
            state["memory_context"] = memory_context
            state["current_agent"] = "memory_reader"
            state["agents_executed"].append("memory_reader")
            state["agent_execution_order"].append("memory_reader")
            
            self.execution_metrics.increment_agent_execution("memory_reader")
            
            logger.info(f"📊 OUTPUT:")
            if isinstance(memory_context, dict):
                summary = memory_context.get('context_summary', 'No summary available')
                logger.info(f"   • Context: {summary[:100]}{'...' if len(summary) > 100 else ''}")
                if 'memories_found' in memory_context:
                    logger.info(f"   • Memories Found: {memory_context.get('memories_found', 0)}")
                if 'search_query' in memory_context:
                    logger.info(f"   • Search Query: {memory_context.get('search_query', 'Unknown')}")
                if 'memory_types' in memory_context:
                    logger.info(f"   • Memory Types: {memory_context.get('memory_types', [])}")
            else:
                logger.info(f"   • Raw Context: {str(memory_context)[:150]}{'...' if len(str(memory_context)) > 150 else ''}")
            
            logger.info(f"📚 MEMORY READER AGENT - Complete\\n")
            
        except Exception as e:
            logger.error(f"❌ MEMORY READER AGENT - FAILED: {e}")
            state["memory_context"] = {"context_summary": f"Memory search error: {str(e)}"}
            logger.info(f"📚 MEMORY READER AGENT - Complete (with errors)\\n")
        
        return state
    
    async def knowledge_agent_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Knowledge Agent node - LangChain agent for external research"""
        message = state.get("user_message", "")
        user_name = state.get("user_name", "Unknown")
        
        logger.info(f"🔍 KNOWLEDGE AGENT - Researching for {user_name}")
        logger.info(f"📝 INPUT: {message[:100]}{'...' if len(message) > 100 else ''}")
        
        try:
            # Execute LangChain Knowledge Agent  
            knowledge_context = await self.knowledge_agent.research_and_summarize(message, user_name)
            
            state["knowledge_context"] = knowledge_context
            state["current_agent"] = "knowledge_agent"
            state["agents_executed"].append("knowledge_agent")
            state["agent_execution_order"].append("knowledge_agent")
            
            self.execution_metrics.increment_agent_execution("knowledge_agent")
            
            logger.info(f"📊 OUTPUT:")
            if isinstance(knowledge_context, dict):
                summary = knowledge_context.get('knowledge_summary', 'No research summary available')
                logger.info(f"   • Research: {summary[:150]}{'...' if len(summary) > 150 else ''}")
                if 'sources_found' in knowledge_context:
                    logger.info(f"   • Sources Found: {knowledge_context.get('sources_found', 0)}")
                if 'search_terms' in knowledge_context:
                    logger.info(f"   • Search Terms: {knowledge_context.get('search_terms', [])}")
                if 'research_type' in knowledge_context:
                    logger.info(f"   • Research Type: {knowledge_context.get('research_type', 'general')}")
            else:
                logger.info(f"   • Raw Research: {str(knowledge_context)[:150]}{'...' if len(str(knowledge_context)) > 150 else ''}")
            
            logger.info(f"🔍 KNOWLEDGE AGENT - Complete\\n")
            
        except Exception as e:
            logger.error(f"❌ KNOWLEDGE AGENT - FAILED: {e}")
            state["knowledge_context"] = {"knowledge_summary": f"Knowledge search error: {str(e)}"}
            logger.info(f"🔍 KNOWLEDGE AGENT - Complete (with errors)\\n")
        
        return state
    
    async def organizer_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Organizer node - LangChain agent for response synthesis"""
        user_name = state.get("user_name", "Unknown")
        message = state.get("user_message", "")
        memory_context = state.get("memory_context", {})
        knowledge_context = state.get("knowledge_context", {})
        
        logger.info(f"🧠 ORGANIZER AGENT - Synthesizing response for {user_name}")
        logger.info(f"📝 INPUTS:")
        logger.info(f"   • User Message: {message[:100]}{'...' if len(message) > 100 else ''}")
        logger.info(f"   • Memory Context: {str(memory_context).replace('{','').replace('}','')[:80]}{'...' if len(str(memory_context)) > 80 else ''}")
        logger.info(f"   • Knowledge Context: {str(knowledge_context).replace('{','').replace('}','')[:80]}{'...' if len(str(knowledge_context)) > 80 else ''}")
        
        try:
            # Execute LangChain Organizer Agent
            synthesis_result = await self.organizer_agent.synthesize_response(
                user_name=user_name,
                user_input=message,
                memory_context=memory_context,
                knowledge_context=knowledge_context
            )
            
            # Handle both dict and string returns from organizer agent
            if isinstance(synthesis_result, dict):
                final_response = synthesis_result.get("response", "I couldn't generate a response.")
            elif isinstance(synthesis_result, str):
                final_response = synthesis_result
            else:
                final_response = str(synthesis_result)
            
            state["final_response"] = final_response
            state["current_agent"] = "organizer"
            state["agents_executed"].append("organizer")
            state["agent_execution_order"].append("organizer")
            
            self.execution_metrics.increment_agent_execution("organizer")
            
            logger.info(f"📊 OUTPUT:")
            logger.info(f"   • Final Response: {final_response[:200]}{'...' if len(final_response) > 200 else ''}")
            if 'reasoning' in synthesis_result:
                logger.info(f"   • Reasoning: {synthesis_result.get('reasoning', '')[:100]}{'...' if len(str(synthesis_result.get('reasoning', ''))) > 100 else ''}")
            
            logger.info(f"🧠 ORGANIZER AGENT - Complete\\n")
            
        except Exception as e:
            logger.error(f"❌ ORGANIZER AGENT - FAILED: {e}")
            final_response = f"I apologize, but I encountered an issue: {str(e)}"
            state["final_response"] = final_response
            logger.info(f"📊 OUTPUT (Error): {final_response}")
            logger.info(f"🧠 ORGANIZER AGENT - Complete (with errors)\\n")
        
        return state
    
    async def memory_writer_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Memory Writer node - LangChain agent for conversation processing"""
        user_name = state.get("user_name", "Unknown")
        user_message = state["user_message"]
        ai_response = state.get("final_response", "")
        
        logger.info(f"📝 MEMORY WRITER AGENT - Processing conversation for {user_name}")
        logger.info(f"📝 INPUTS:")
        logger.info(f"   • User Message: {user_message[:100]}{'...' if len(user_message) > 100 else ''}")
        logger.info(f"   • AI Response: {ai_response[:100]}{'...' if len(ai_response) > 100 else ''}")
        
        try:
            # Execute LangChain Memory Writer Agent
            writer_results = await self.memory_writer_agent.process_conversation(user_name, user_message, ai_response)
            
            state["current_agent"] = "memory_writer"
            state["agents_executed"].append("memory_writer")
            state["agent_execution_order"].append("memory_writer")
            
            self.execution_metrics.increment_agent_execution("memory_writer")
            
            logger.info(f"📊 OUTPUT:")
            if isinstance(writer_results, dict):
                logger.info(f"   • Facts Extracted: {writer_results.get('facts_extracted', 0)}")
                logger.info(f"   • Short-term Stored: {writer_results.get('short_term_stored', 0)}")
                logger.info(f"   • Long-term Stored: {writer_results.get('long_term_stored', 0)}")
                logger.info(f"   • Processing Status: {'Success' if writer_results.get('session_stored', False) else 'Failed'}")
            else:
                logger.info(f"   • Processing Result: {str(writer_results)[:100]}{'...' if len(str(writer_results)) > 100 else ''}")
            
            logger.info(f"📝 MEMORY WRITER AGENT - Complete")
            logger.info(f"=" * 80 + "\\n")
            
        except Exception as e:
            logger.error(f"❌ MEMORY WRITER AGENT - FAILED: {e}")
            logger.info(f"📝 MEMORY WRITER AGENT - Complete (with errors)")
            logger.info(f"=" * 80 + "\\n")
        
        return state
    
    async def _calculate_complexity_score(self, message: str) -> float:
        """Calculate message complexity for routing decisions"""
        # Simple heuristic-based complexity calculation
        complexity_factors = {
            'question_words': len([w for w in message.lower().split() if w in ['what', 'how', 'why', 'when', 'where', 'which', 'who']]) * 0.2,
            'length': min(len(message.split()) / 50, 0.3),
            'research_keywords': len([w for w in message.lower().split() if w in ['research', 'find', 'search', 'analyze', 'explain', 'compare']]) * 0.25,
            'technical_terms': len([w for w in message.lower().split() if w in ['algorithm', 'implementation', 'technical', 'system', 'architecture']]) * 0.15
        }
        
        return min(sum(complexity_factors.values()), 1.0)
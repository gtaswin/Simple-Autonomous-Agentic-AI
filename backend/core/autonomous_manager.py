"""
Autonomous Operations Manager - Separated from main orchestrator
Handles all autonomous thinking, pattern discovery, and insight generation
"""

import logging
from typing import Dict, Any, Optional
from datetime import datetime

from core.streaming_manager import streaming_manager, AgentStatus

logger = logging.getLogger(__name__)


class AutonomousManager:
    """Manages all autonomous operations separately from user interactions"""
    
    def __init__(self, orchestrator):
        """Initialize with reference to main orchestrator for agent access"""
        self.orchestrator = orchestrator
        self.memory_reader_agent = orchestrator.memory_reader_agent
        self.memory_writer_agent = orchestrator.memory_writer_agent
        self.knowledge_agent = orchestrator.knowledge_agent
        self.organizer_agent = orchestrator.organizer_agent
        self.memory_system = orchestrator.memory_system
        self.config = orchestrator.config
        self.execution_metrics = orchestrator.execution_metrics
    
    async def autonomous_router_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Router for autonomous operations"""
        operation_type = state.get("operation_type", "")
        
        logger.info(f"🤖 AUTONOMOUS ROUTER - Operation: {operation_type}")
        
        state["current_agent"] = "autonomous_router"
        state["agents_executed"].append("autonomous_router")
        state["agent_execution_order"].append("autonomous_router")
        
        return state
    
    async def autonomous_memory_analysis_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Autonomous memory analysis for pattern discovery"""
        try:
            target_user_name = state.get("target_user_name", "Unknown")
            operation_type = state.get("operation_type", "")
            
            logger.info(f"🧠 AUTONOMOUS MEMORY ANALYSIS - Analyzing {target_user_name} for {operation_type}")
            
            # Use Memory Reader Agent to analyze patterns
            memory_context = await self.memory_reader_agent.get_complete_context(
                target_user_name, 
                f"autonomous {operation_type} analysis"
            )
            
            state["memory_context"] = memory_context
            state["current_agent"] = "autonomous_memory_analysis"
            state["agents_executed"].append("autonomous_memory_analysis")
            state["agent_execution_order"].append("autonomous_memory_analysis")
            
            self.execution_metrics.increment_agent_execution("memory_reader")
            
            # Send autonomous thinking broadcast
            await streaming_manager.send_autonomous_broadcast({
                "type": "autonomous_thinking",
                "data": {
                    "operation": operation_type,
                    "target_user": target_user_name,
                    "status": "memory_analysis_complete",
                    "summary": f"Analyzed {target_user_name} for {operation_type}: {memory_context.get('context_summary', 'No summary')[:200]}..."
                }
            })
            
            logger.info(f"🧠 AUTONOMOUS MEMORY ANALYSIS - Complete")
            
        except Exception as e:
            logger.error(f"❌ AUTONOMOUS MEMORY ANALYSIS - FAILED: {e}")
            state["memory_context"] = {"context_summary": f"Autonomous memory analysis error: {str(e)}"}
        
        return state
    
    async def autonomous_research_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Autonomous research for knowledge gathering"""
        try:
            operation_type = state.get("operation_type", "")
            
            logger.info(f"🔍 AUTONOMOUS RESEARCH - Operation: {operation_type}")
            
            # Use Knowledge Agent for autonomous research
            knowledge_context = await self.knowledge_agent.research_and_summarize(
                f"autonomous {operation_type} research", target_user_name
            )
            
            state["knowledge_context"] = knowledge_context
            state["current_agent"] = "autonomous_research"
            state["agents_executed"].append("autonomous_research")
            state["agent_execution_order"].append("autonomous_research")
            
            # Since this is autonomous, use configured user name as the "AI assistant" performing research
            autonomous_user_name = state.get("user_name", "Assistant")
            
            # Send autonomous thinking broadcast
            await streaming_manager.send_autonomous_broadcast({
                "type": "autonomous_thinking",
                "data": {
                    "operation": operation_type,
                    "agent": "knowledge_research",
                    "status": "research_complete",
                    "summary": f"Autonomous research completed: {knowledge_context.get('knowledge_summary', 'No summary')[:200]}..."
                }
            })
            
            self.execution_metrics.increment_agent_execution("knowledge_agent")
            
            logger.info(f"🔍 AUTONOMOUS RESEARCH - Complete")
            
        except Exception as e:
            logger.error(f"❌ AUTONOMOUS RESEARCH - FAILED: {e}")
            state["knowledge_context"] = {"knowledge_summary": f"Autonomous research error: {str(e)}"}
        
        return state
    
    async def autonomous_synthesis_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Autonomous synthesis for insight generation"""
        try:
            target_user_name = state.get("target_user_name", self.config.raw_config.get("user", {}).get("name", "Aswin"))
            operation_type = state.get("operation_type", "")
            memory_context = state.get("memory_context", {})
            knowledge_context = state.get("knowledge_context", {})
            
            logger.info(f"⚡ AUTONOMOUS SYNTHESIS - Generating insights for {target_user_name}")
            
            # Use Organizer Agent for autonomous synthesis
            synthesis_result = await self.organizer_agent.synthesize_response(
                user_name=target_user_name,
                user_input=f"autonomous {operation_type}",
                memory_context=memory_context,
                knowledge_context=knowledge_context
            )
            
            # Store autonomous insights in dedicated insight storage
            if synthesis_result.get("response") and operation_type in ["insight_generation", "pattern_discovery", "autonomous_thinking", "milestone_tracking", "life_event_detection"]:
                try:
                    await self.memory_system.store_autonomous_insight(
                        user_name=target_user_name,
                        insight_type=operation_type,
                        insight_content=synthesis_result.get("response", ""),
                        metadata={
                            "generated_at": datetime.now().isoformat(),
                            "trigger_source": str(state.get("trigger_source", "scheduled")),
                            "agents_executed": [str(agent) for agent in state.get("agents_executed", [])],
                            "processing_time": float(state.get("processing_time", 0))
                        }
                    )
                    logger.info(f"💾 Stored autonomous insight: {operation_type}")
                except Exception as e:
                    logger.error(f"Failed to store autonomous insight: {e}")
                
                # Use Memory Writer Agent with storage constraints (no long-term for autonomous)
                try:
                    insight_content = synthesis_result.get("response", "")
                    if len(insight_content) > 50:  # Only store substantial insights
                        
                        # Create conversation context for Memory Writer Agent
                        autonomous_user_message = f"Autonomous {operation_type} analysis"
                        autonomous_ai_response = insight_content
                        
                        # Process through Memory Writer Agent (working memory only for autonomous)
                        writer_results = await self.memory_writer_agent.process_conversation(
                            target_user_name, 
                            autonomous_user_message, 
                            autonomous_ai_response
                        )
                        
                        logger.info(f"📝 Processed autonomous insight via Memory Writer:")
                        if isinstance(writer_results, dict):
                            logger.info(f"   • Facts Extracted: {writer_results.get('facts_extracted', 0)}")
                            logger.info(f"   • Short-term Stored: {writer_results.get('short_term_stored', 0)}")
                            logger.info(f"   • Long-term Stored: {writer_results.get('long_term_stored', 0)}")
                        
                        self.execution_metrics.increment_agent_execution("memory_writer")
                        
                except Exception as e:
                    logger.error(f"Failed to process autonomous discovery via Memory Writer Agent: {e}")
            
            state["final_response"] = synthesis_result.get("response", f"Autonomous {operation_type} completed")
            state["current_agent"] = "autonomous_synthesis"
            state["agents_executed"].append("autonomous_synthesis")
            state["agent_execution_order"].append("autonomous_synthesis")
            
            self.execution_metrics.increment_agent_execution("organizer")
            
            logger.info(f"⚡ AUTONOMOUS SYNTHESIS - Complete")
            
        except Exception as e:
            logger.error(f"❌ AUTONOMOUS SYNTHESIS - FAILED: {e}")
            state["final_response"] = f"Autonomous synthesis error: {str(e)}"
        
        return state
    
    async def autonomous_broadcast_node(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """Autonomous broadcast for real-time updates"""
        try:
            if state.get("broadcast_updates", False):
                operation_type = state.get("operation_type", "")
                result = state.get("final_response", "")
                user_name = state.get("user_name", "Assistant")
                
                # Broadcast autonomous completion
                await streaming_manager.send_autonomous_broadcast({
                    "type": "autonomous_completion",
                    "data": {
                        "operation": operation_type,
                        "result": result[:300] + "..." if len(result) > 300 else result,
                        "completed_at": datetime.now().isoformat(),
                        "user_name": user_name,
                        "agents_executed": state.get("agents_executed", []),
                        "processing_time": state.get("processing_time", 0),
                    }
                })
                
                # Also send insight notification for specific operations
                if operation_type in ["insight_generation", "pattern_discovery", "autonomous_thinking", "milestone_tracking", "life_event_detection"]:
                    await streaming_manager.send_autonomous_broadcast({
                        "type": "insight_generated",
                        "data": {
                            "insight_type": operation_type,
                            "insight_preview": result[:200] + "..." if len(result) > 200 else result,
                            "generated_at": datetime.now().isoformat(),
                            "agents_executed": state.get("agents_executed", []),
                            "processing_time": state.get("processing_time", 0),
                            "metadata": {
                                "trigger_source": state.get("trigger_source", "unknown"),
                                "autonomous_context": state.get("autonomous_context", {}),
                                "autonomous_parameters": state.get("autonomous_parameters", {})
                            }
                        }
                    })
                
                state["broadcast_data"] = {
                    "type": "autonomous_completion",
                    "operation": operation_type,
                    "result": result,
                    "timestamp": datetime.now().isoformat()
                }
                
                logger.info(f"📡 AUTONOMOUS BROADCAST - Sent for {operation_type}")
            
            state["current_agent"] = "autonomous_broadcast"
            state["agents_executed"].append("autonomous_broadcast")
            state["agent_execution_order"].append("autonomous_broadcast")
            
        except Exception as e:
            logger.error(f"❌ AUTONOMOUS BROADCAST - FAILED: {e}")
        
        return state
    
    def route_autonomous_operation(self, state: Dict[str, Any]) -> str:
        """Route autonomous operations based on type"""
        operation_type = state.get("operation_type", "")
        
        if operation_type in ["memory_analysis", "pattern_discovery"]:
            return "memory_analysis"
        elif operation_type in ["insight_generation", "autonomous_thinking", "milestone_tracking", "life_event_detection"]:
            return "full_analysis"
        elif operation_type == "research_only":
            return "research_only"
        else:
            return "end"
    
    def autonomous_should_research(self, state: Dict[str, Any]) -> str:
        """Determine if autonomous operation should include research"""
        if state.get("should_research", False):
            return "research"
        else:
            return "synthesize"
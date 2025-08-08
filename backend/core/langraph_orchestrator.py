"""
4-Agent LangGraph Orchestrator - Hybrid Memory Architecture
Features: Memory Reader (LOCAL), Memory Writer (LOCAL), Knowledge (LOCAL), Organizer (EXTERNAL LLM)
"""

import asyncio
from typing import Dict, List, Optional, Any, TypedDict
from datetime import datetime
from dataclasses import dataclass
import logging

# LangGraph imports for proper multi-agent orchestration
from langgraph.graph import StateGraph, END, START
from langgraph.graph.state import CompiledStateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Streaming infrastructure imports
from core.streaming_manager import streaming_manager, AgentStatus
from core.parallel_coordinator import parallel_coordinator

# Phase 5: Import structured output support
from core.output_schemas import WorkflowExecutionOutput, WorkflowPattern
from core.output_parser import workflow_output, type_conversion_service


# Update to use LangChain agent classes
from agents.memory_reader_agent import LangChainMemoryReaderAgent as MemoryReaderAgent
from agents.memory_writer_agent import LangChainMemoryWriterAgent as MemoryWriterAgent
from agents.knowledge_agent import LangChainKnowledgeAgent as KnowledgeAgent
from agents.organizer_agent import LangChainOrganizerAgent as OrganizerAgent
from memory.autonomous_memory import AutonomousMemorySystem
from core.config import AssistantConfig
from core.transformers_service import TransformersService

logger = logging.getLogger(__name__)


class MultiAgentState(TypedDict):
    """LangGraph state for 4-agent multi-agent workflow with autonomous support"""
    # Core message flow
    messages: List[Dict[str, Any]]
    user_name: str
    user_message: str
    
    # Agent contexts (processed by different agents)
    memory_context: Optional[Dict[str, Any]]
    knowledge_context: Optional[Dict[str, Any]]
    
    # Execution tracking
    current_agent: str
    agents_executed: List[str]
    next_agent: Optional[str]
    
    # Quality and routing
    should_research: bool
    complexity_score: float
    
    # Final outputs
    final_response: Optional[str]
    
    # Autonomous operation support
    operation_type: Optional[str]  # "user_input", "autonomous_thinking", "autonomous_insights", etc.
    trigger_source: Optional[str]  # "user", "scheduled", "event_driven", "threshold"
    autonomous_context: Optional[Dict[str, Any]]  # Context for autonomous operations
    autonomous_parameters: Optional[Dict[str, Any]]  # Parameters for autonomous tasks
    broadcast_updates: bool  # Whether to broadcast progress via WebSocket
    
    # Metadata
    timestamp: str
    processing_time: float
    agent_execution_order: List[str]


class LangGraphMultiAgentOrchestrator:
    """True LangGraph multi-agent orchestrator with 4 LangChain agents"""
    
    def __init__(
        self,
        memory_system: AutonomousMemorySystem,
        config: AssistantConfig,
        transformers_service: TransformersService
    ):
        self.memory_system = memory_system
        self.config = config
        self.transformers_service = transformers_service
        
        # Initialize 4 LangChain agents
        self.memory_reader_agent = MemoryReaderAgent(
            transformers_service=transformers_service,
            memory_system=memory_system
        )
        
        self.memory_writer_agent = MemoryWriterAgent(
            transformers_service=transformers_service,
            memory_system=memory_system
        )
        
        self.knowledge_agent = KnowledgeAgent(
            config=config,
            transformers_service=transformers_service
        )
        
        self.organizer_agent = OrganizerAgent(
            config=config,
            transformers_service=transformers_service,
            memory_reader_agent=self.memory_reader_agent,
            memory_writer_agent=self.memory_writer_agent
        )
        self.organizer_agent.set_memory_system(memory_system)
        
        # Build LangGraph workflows for both user and autonomous operations
        self.workflow = self._build_langgraph_workflow()
        self.autonomous_workflow = self._build_autonomous_workflow()
        self.app = self.workflow.compile(checkpointer=MemorySaver())
        self.autonomous_app = self.autonomous_workflow.compile(checkpointer=MemorySaver())
        
        # Connect streaming manager and parallel coordinator
        self.streaming_manager = streaming_manager
        self.parallel_coordinator = parallel_coordinator
        
        # Performance tracking
        self.execution_metrics = {
            "total_requests": 0,
            "agent_executions": {
                "memory_reader": 0,
                "knowledge_agent": 0,
                "organizer": 0,
                "memory_writer": 0
            },
            "workflow_patterns": {
                "simple_memory_only": 0,
                "research_enhanced": 0,
                "complex_reasoning": 0
            },
            "autonomous_operations": {
                "thinking_cycles": 0,
                "insight_generation": 0,
                "pattern_discovery": 0,
                "memory_maintenance": 0,
                "milestone_tracking": 0,
                "life_event_detection": 0
            },
            "average_processing_time": 0.0
        }
        
        logger.info("🚀 LangGraph multi-agent orchestrator initialized with 4 LangChain agents")
        logger.info("🔄 Streaming manager connected for real-time updates")
        logger.info("⚡ Parallel coordinator enabled for optimized execution")
    
    def _build_langgraph_workflow(self) -> StateGraph:
        """Build the LangGraph workflow with proper nodes and edges"""
        workflow = StateGraph(MultiAgentState)
        
        # Add nodes for each agent
        workflow.add_node("memory_reader", self._memory_reader_node)
        workflow.add_node("knowledge_agent", self._knowledge_agent_node)
        workflow.add_node("organizer", self._organizer_node)
        workflow.add_node("memory_writer", self._memory_writer_node)
        workflow.add_node("router", self._router_node)
        
        # Define workflow edges
        workflow.set_entry_point("router")
        
        # Router decides the execution path
        workflow.add_conditional_edges(
            "router",
            self._route_next_agent,
            {
                "memory_only": "memory_reader",
                "research_needed": "memory_reader",  # Still start with memory
                "end": END
            }
        )
        
        # Memory Reader can go to Organizer or trigger research
        workflow.add_conditional_edges(
            "memory_reader",
            self._should_research,
            {
                "research": "knowledge_agent",
                "organize": "organizer"
            }
        )
        
        # Knowledge Agent always goes to Organizer
        workflow.add_edge("knowledge_agent", "organizer")
        
        # Organizer can trigger memory writing or end
        workflow.add_conditional_edges(
            "organizer",
            self._should_write_memory,
            {
                "write_memory": "memory_writer",
                "end": END
            }
        )
        
        # Memory Writer always ends the workflow
        workflow.add_edge("memory_writer", END)
        
        return workflow
    
    async def _execute_parallel_workflow(
        self, 
        initial_state: MultiAgentState, 
        agents_to_execute: List[str]
    ) -> MultiAgentState:
        """Execute workflow using parallel coordinator for optimal performance"""
        
        workflow_id = initial_state.get("workflow_id", "unknown")
        user_name = initial_state.get("user_name", "Unknown")
        should_research = initial_state.get("should_research", False)
        
        # Create agent function mapping
        agent_functions = {
            "router": self._router_node,
            "memory_reader": self._memory_reader_node,
            "knowledge_agent": self._knowledge_agent_node,
            "organizer": self._organizer_node,
            "memory_writer": self._memory_writer_node
        }
        
        # Filter to only requested agents
        active_agent_functions = {
            name: func for name, func in agent_functions.items() 
            if name in agents_to_execute
        }
        
        try:
            # Execute using parallel coordinator
            final_state = await self.parallel_coordinator.execute_parallel_workflow(
                workflow_id=workflow_id,
                user_name=user_name,
                state=dict(initial_state),  # Convert TypedDict to dict
                agent_functions=active_agent_functions,
                should_research=should_research
            )
            
            # Convert back to MultiAgentState format
            result_state = initial_state.copy()
            result_state.update(final_state)
            
            return result_state
            
        except Exception as e:
            logger.error(f"❌ Parallel workflow execution failed: {e}")
            # Fallback to sequential execution
            logger.info("🔄 Falling back to sequential execution")
            config = {"configurable": {"thread_id": f"{user_name}_{datetime.now().timestamp()}"}}
            return await self.app.ainvoke(initial_state, config)
    
    def _build_autonomous_workflow(self) -> StateGraph:
        """Build the LangGraph workflow specifically for autonomous operations"""
        workflow = StateGraph(MultiAgentState)
        
        # Add nodes for autonomous operations
        workflow.add_node("autonomous_router", self._autonomous_router_node)
        workflow.add_node("autonomous_memory_analysis", self._autonomous_memory_analysis_node)
        workflow.add_node("autonomous_research", self._autonomous_research_node)
        workflow.add_node("autonomous_synthesis", self._autonomous_synthesis_node)
        workflow.add_node("autonomous_broadcast", self._autonomous_broadcast_node)
        
        # Define autonomous workflow edges
        workflow.set_entry_point("autonomous_router")
        
        # Autonomous router decides the operation path
        workflow.add_conditional_edges(
            "autonomous_router",
            self._route_autonomous_operation,
            {
                "memory_analysis": "autonomous_memory_analysis",
                "full_analysis": "autonomous_memory_analysis",  # Start with memory for full analysis
                "research_only": "autonomous_research",
                "broadcast_only": "autonomous_broadcast",
                "end": END
            }
        )
        
        # Memory analysis can go to research or synthesis
        workflow.add_conditional_edges(
            "autonomous_memory_analysis",
            self._autonomous_should_research,
            {
                "research": "autonomous_research",
                "synthesize": "autonomous_synthesis"
            }
        )
        
        # Research always goes to synthesis
        workflow.add_edge("autonomous_research", "autonomous_synthesis")
        
        # Synthesis goes directly to broadcast (no manual memory update needed)
        workflow.add_edge("autonomous_synthesis", "autonomous_broadcast")
        
        # Broadcast always ends the workflow
        workflow.add_edge("autonomous_broadcast", END)
        
        return workflow
    
    @workflow_output(capture_performance=True)
    async def process_message(self, user_name: str, message: str) -> WorkflowExecutionOutput:
        """Process message through LangGraph multi-agent workflow with streaming"""
        start_time = datetime.now()
        workflow_id = None  # Initialize early to prevent NameError
        
        try:
            self.execution_metrics["total_requests"] += 1
            
            logger.info(f"\n" + "🚀"*40)
            logger.info(f"🚀 LANGGRAPH WORKFLOW STARTED")
            logger.info(f"👤 User: {user_name}")
            logger.info(f"💬 Message: {message}")
            logger.info(f"⏰ Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"🚀"*40)
            
            # Generate workflow ID for streaming
            workflow_id = f"{user_name}_{datetime.now().timestamp()}"
            
            # Determine agents to execute (for streaming setup)
            complexity_score = await self._calculate_complexity_score(message)
            should_research = (
                complexity_score > 0.6 or
                any(keyword in message.lower() for keyword in 
                    ["latest", "current", "today", "recent", "news", "search", "find", "what is", "who is"])
            )
            
            agents_to_execute = ["router", "memory_reader"]
            if should_research:
                agents_to_execute.append("knowledge_agent")
            agents_to_execute.extend(["organizer", "memory_writer"])
            
            # Start workflow streaming
            await self.streaming_manager.start_workflow_stream(
                workflow_id=workflow_id,
                user_name=user_name,
                user_message=message,
                agents_to_execute=agents_to_execute
            )
            
            # Initialize state with streaming info
            initial_state: MultiAgentState = {
                "messages": [{"role": "user", "content": message}],
                "user_name": user_name,
                "user_message": message,
                "memory_context": None,
                "knowledge_context": None,
                "current_agent": "router",
                "agents_executed": [],
                "next_agent": None,
                "should_research": should_research,  # Use calculated value
                "complexity_score": complexity_score,  # Use calculated value
                "final_response": None,
                "operation_type": "user_input",
                "trigger_source": "user",
                "autonomous_context": None,
                "autonomous_parameters": None,
                "broadcast_updates": True,
                "workflow_id": workflow_id,
                "timestamp": datetime.now().isoformat(),
                "processing_time": 0.0,
                "agent_execution_order": []
            }
            
            # Check if parallel execution is beneficial
            use_parallel = should_research or complexity_score > 0.4
            
            if use_parallel:
                # Use parallel coordinator for optimized execution
                logger.info("⚡ Using parallel execution for improved performance")
                result = await self._execute_parallel_workflow(initial_state, agents_to_execute)
            else:
                # Use standard LangGraph workflow
                logger.info("🔄 Using sequential execution")
                config = {"configurable": {"thread_id": f"{user_name}_{datetime.now().timestamp()}"}}
                result = await self.app.ainvoke(initial_state, config)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Debug: Check what result contains
            logger.info(f"🔍 RESULT TYPE: {type(result)}")
            logger.info(f"🔍 RESULT KEYS: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            logger.info(f"🔍 HAS final_response: {'final_response' in result if isinstance(result, dict) else 'N/A'}")
            if isinstance(result, dict) and 'final_response' in result:
                logger.info(f"🔍 FINAL_RESPONSE: {result['final_response'][:100]}...")
            else:
                logger.warning(f"⚠️ final_response missing from result!")
            
            # Handle case where result might be a string instead of dict
            if isinstance(result, str):
                logger.warning(f"⚠️ Result is a string instead of dict: {result[:100]}...")
                # Create a proper dict structure
                result = {
                    "final_response": result,
                    "agents_executed": ["router", "memory_reader", "organizer", "memory_writer"],
                    "workflow_pattern": "simple_memory_only"
                }
            elif not isinstance(result, dict):
                logger.warning(f"⚠️ Result is unexpected type {type(result)}, converting to dict")
                result = {"final_response": str(result), "agents_executed": []}
            
            # Update metrics
            total_requests = self.execution_metrics["total_requests"]
            current_avg = self.execution_metrics["average_processing_time"]
            self.execution_metrics["average_processing_time"] = \
                (current_avg * (total_requests - 1) + processing_time) / total_requests
            
            # Extract values safely handling both dict and potential Pydantic objects
            def safe_get(obj, key, default=None):
                """Safely get attribute from object or dict"""
                if hasattr(obj, key):
                    return getattr(obj, key)
                elif isinstance(obj, dict):
                    return obj.get(key, default)
                else:
                    return default
            
            # Ensure result is a dictionary at this point
            if not isinstance(result, dict):
                logger.error(f"❌ Result is still not a dict after conversion: {type(result)}")
                result = {"final_response": str(result), "agents_executed": []}
            
            # Safely extract values with fallbacks
            agents_used = safe_get(result, "agents_executed", [])
            final_response = safe_get(result, "final_response", "I'm sorry, I couldn't process your request.")
            agent_execution_order = safe_get(result, "agent_execution_order", [])
            complexity_score = safe_get(result, "complexity_score", 0.0)
            memory_context = safe_get(result, "memory_context", {})
            knowledge_context = safe_get(result, "knowledge_context", {})
            
            # Determine workflow pattern
            workflow_pattern = WorkflowPattern.SIMPLE_MEMORY_ONLY
            if "knowledge_agent" in agents_used:
                if len(agents_used) > 3:
                    self.execution_metrics["workflow_patterns"]["complex_reasoning"] += 1
                    workflow_pattern = WorkflowPattern.COMPLEX_REASONING
                else:
                    self.execution_metrics["workflow_patterns"]["research_enhanced"] += 1
                    workflow_pattern = WorkflowPattern.RESEARCH_ENHANCED
            else:
                self.execution_metrics["workflow_patterns"]["simple_memory_only"] += 1
            
            # Complete workflow streaming
            await self.streaming_manager.complete_workflow_stream(
                workflow_id=workflow_id,
                final_response=final_response,
                total_execution_time=processing_time,
                agents_executed=agents_used,
                workflow_pattern=workflow_pattern.value  # Convert enum to string
            )
            
            # Log completion summary
            logger.info(f"🏁"*40)
            logger.info(f"🏁 LANGGRAPH WORKFLOW COMPLETED")
            logger.info(f"⏱️  Processing Time: {processing_time:.2f} seconds")
            logger.info(f"🤖 Agents Used: {' → '.join(agents_used)}")
            logger.info(f"📝 Final Response: {final_response[:150]}{'...' if len(final_response) > 150 else ''}")
            logger.info(f"🏁"*40 + "\n\n")
            
            # Safely build return dictionary
            return {
                "response": final_response,
                "metadata": {
                    "architecture": "langgraph_multi_agent",
                    "agents_executed": agents_used,
                    "execution_order": agent_execution_order,
                    "workflow_pattern": self._get_workflow_pattern(agents_used),
                    "processing_time": processing_time,
                    "complexity_score": complexity_score,
                    "memory_context_available": isinstance(memory_context, dict) and safe_get(memory_context, "context_summary") is not None,
                    "knowledge_context_available": isinstance(knowledge_context, dict) and safe_get(knowledge_context, "knowledge_summary") is not None
                },
                "timestamp": datetime.now().isoformat(),
                "user_name": user_name
            }
            
        except Exception as e:
            logger.error(f"❌ LangGraph workflow failed: {e}")
            return {
                "response": f"I apologize, but I encountered an issue processing your request: {str(e)}",
                "metadata": {
                    "error": str(e),
                    "architecture": "langgraph_multi_agent",
                    "processing_failed": True,
                    "processing_time": (datetime.now() - start_time).total_seconds()
                },
                "timestamp": datetime.now().isoformat(),
                "user_name": user_name
            }
    
    async def process_autonomous_operation(
        self, 
        operation_type: str,
        target_user_name: Optional[str] = None,  # The user the operation is about (for analysis)
        trigger_source: str = "scheduled",
        autonomous_context: Optional[Dict[str, Any]] = None,
        autonomous_parameters: Optional[Dict[str, Any]] = None,
        broadcast_updates: bool = True
    ) -> Dict[str, Any]:
        """Process autonomous operations through LangGraph workflow"""
        start_time = datetime.now()
        
        try:
            self.execution_metrics["total_requests"] += 1
            
            logger.info(f"🤖 Autonomous operation: {operation_type} (trigger: {trigger_source})")
            
            # Get target user from config if not provided
            if target_user_name is None:
                target_user_name = self.config.get("user.name", "Unknown")
            
            # Initialize autonomous state with dedicated autonomous user_name
            initial_state: MultiAgentState = {
                "messages": [{"role": "system", "content": f"Autonomous operation: {operation_type}"}],
                "user_name": "Assistant",  # Autonomous system uses dedicated user_name for separate memory
                "user_message": f"Autonomous operation: {operation_type}",
                "memory_context": None,
                "knowledge_context": None,
                "current_agent": "autonomous_router",
                "agents_executed": [],
                "next_agent": None,
                "should_research": False,
                "complexity_score": 0.0,
                "final_response": None,
                "operation_type": operation_type,
                "trigger_source": trigger_source,
                "target_user_name": target_user_name,
                "autonomous_context": autonomous_context or {},
                "autonomous_parameters": autonomous_parameters or {},
                "broadcast_updates": broadcast_updates,
                "target_user_name": target_user_name,  # Store the target user for analysis
                "timestamp": datetime.now().isoformat(),
                "processing_time": 0.0,
                "agent_execution_order": []
            }
            
            # Execute autonomous LangGraph workflow
            config = {"configurable": {"thread_id": f"autonomous_{operation_type}_{datetime.now().timestamp()}"}}
            
            result = await self.autonomous_app.ainvoke(initial_state, config)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update autonomous metrics
            if operation_type in self.execution_metrics["autonomous_operations"]:
                self.execution_metrics["autonomous_operations"][operation_type] += 1
            
            # Update average processing time
            total_requests = self.execution_metrics["total_requests"]
            current_avg = self.execution_metrics["average_processing_time"]
            self.execution_metrics["average_processing_time"] = \
                (current_avg * (total_requests - 1) + processing_time) / total_requests
            
            logger.info(f"✅ Autonomous operation completed in {processing_time:.2f}s: {operation_type}")
            
            # Safe get function for autonomous operations too
            def safe_get_auto(obj, key, default=None):
                """Safely get attribute from object or dict"""
                if hasattr(obj, key):
                    return getattr(obj, key)
                elif isinstance(obj, dict):
                    return obj.get(key, default)
                else:
                    return default
            
            # Send WebSocket broadcast if data is prepared
            broadcast_data = safe_get_auto(result, "broadcast_data")
            if broadcast_data and broadcast_updates:
                await self._send_autonomous_broadcast(broadcast_data)
            
            return {
                "operation_type": operation_type,
                "trigger_source": trigger_source,
                "result": safe_get_auto(result, "final_response", "Autonomous operation completed"),
                "metadata": {
                    "architecture": "langgraph_autonomous_multi_agent",
                    "agents_executed": safe_get_auto(result, "agents_executed", []),
                    "execution_order": safe_get_auto(result, "agent_execution_order", []),
                    "processing_time": processing_time,
                    "broadcast_sent": broadcast_data is not None and broadcast_updates
                },
                "timestamp": datetime.now().isoformat(),
                "user_name": "Assistant",  # Autonomous operations are always by Assistant
                "target_user_name": target_user_name  # The user being analyzed
            }
            
        except Exception as e:
            logger.error(f"❌ Autonomous operation failed: {operation_type} - {e}")
            return {
                "operation_type": operation_type,
                "trigger_source": trigger_source,
                "result": f"Autonomous operation failed: {str(e)}",
                "metadata": {
                    "error": str(e),
                    "architecture": "langgraph_autonomous_multi_agent",
                    "processing_failed": True,
                    "processing_time": (datetime.now() - start_time).total_seconds()
                },
                "timestamp": datetime.now().isoformat(),
                "user_name": "Assistant",  # Autonomous operations are always by Assistant
                "target_user_name": target_user_name  # The user being analyzed
            }
    
    # LangGraph Node Implementations
    async def _router_node(self, state: MultiAgentState) -> MultiAgentState:
        """Router node - determines execution strategy"""
        message = state.get("user_message", "")
        user_name = state.get("user_name", "Unknown")
        workflow_id = state.get("workflow_id", "unknown")
        
        logger.info(f"\n" + "="*80)
        logger.info(f"🗺 ROUTER AGENT - Processing request for {user_name}")
        logger.info(f"📝 INPUT: {message[:100]}{'...' if len(message) > 100 else ''}")
        logger.info(f"="*80)
        
        complexity_score = await self._calculate_complexity_score(message)
        
        # Determine if research is needed
        should_research = (
            complexity_score > 0.6 or
            any(keyword in message.lower() for keyword in 
                ["latest", "current", "today", "recent", "news", "search", "find", "what is", "who is"])
        )
        
        state["complexity_score"] = complexity_score
        state["should_research"] = should_research
        state["current_agent"] = "router"
        state["agents_executed"].append("router")
        state["agent_execution_order"].append("router")
        
        # Stream router completion
        await self.streaming_manager.update_agent_status(
            workflow_id=workflow_id,
            agent_name="router",
            status=AgentStatus.COMPLETED,
            current_activity="Route determined",
            progress_percentage=100.0,
            result_preview=f"Complexity: {complexity_score:.2f}, Research: {should_research}",
            metadata={"complexity_score": complexity_score, "research_needed": should_research}
        )
        
        logger.info(f"📊 OUTPUT:")
        logger.info(f"   • Complexity Score: {complexity_score:.2f}")
        logger.info(f"   • Research Needed: {should_research}")
        logger.info(f"   • Next Steps: Memory Reader → {'Knowledge Agent → ' if should_research else ''}Organizer → Memory Writer")
        logger.info(f"🗺 ROUTER AGENT - Complete\n")
        
        return state
    
    async def _memory_reader_node(self, state: MultiAgentState) -> MultiAgentState:
        """Memory Reader node - LangChain agent for context retrieval with streaming"""
        user_name = state.get("user_name", "Unknown")
        message = state.get("user_message", "")
        workflow_id = state.get("workflow_id", "unknown")
        
        # Stream memory reader start
        await self.streaming_manager.update_agent_status(
            workflow_id=workflow_id,
            agent_name="memory_reader",
            status=AgentStatus.ACTIVE,
            current_activity="Retrieving memory context",
            progress_percentage=15.0
        )
        
        logger.info(f"📚 MEMORY READER AGENT - Retrieving context for {user_name}")
        logger.info(f"📝 INPUT: {message[:100]}{'...' if len(message) > 100 else ''}")
        
        try:
            # Stream memory search
            await self.streaming_manager.update_agent_status(
                workflow_id=workflow_id,
                agent_name="memory_reader",
                status=AgentStatus.ACTIVE,
                current_activity="Searching memory systems",
                progress_percentage=60.0
            )
            
            # Execute LangChain Memory Reader Agent
            memory_context = await self.memory_reader_agent.get_complete_context(user_name, message)
            
            state["memory_context"] = memory_context
            state["current_agent"] = "memory_reader"
            state["agents_executed"].append("memory_reader")
            state["agent_execution_order"].append("memory_reader")
            self.execution_metrics["agent_executions"]["memory_reader"] += 1
            
            logger.info(f"📊 OUTPUT:")
            if isinstance(memory_context, dict):
                summary = memory_context.get('context_summary', 'No summary available')
                logger.info(f"   • Context Summary: {summary[:150]}{'...' if len(summary) > 150 else ''}")
                if 'memories_found' in memory_context:
                    logger.info(f"   • Memories Found: {memory_context.get('memories_found', 0)}")
                if 'memory_types' in memory_context:
                    logger.info(f"   • Memory Types: {memory_context.get('memory_types', [])}")
            else:
                logger.info(f"   • Raw Context: {str(memory_context)[:150]}{'...' if len(str(memory_context)) > 150 else ''}")
            
            logger.info(f"📚 MEMORY READER AGENT - Complete\n")
            
        except Exception as e:
            logger.error(f"❌ MEMORY READER AGENT - FAILED: {e}")
            state["memory_context"] = {"context_summary": f"Memory error: {str(e)}"}
            logger.info(f"📚 MEMORY READER AGENT - Complete (with errors)\n")
        
        return state
    
    async def _knowledge_agent_node(self, state: MultiAgentState) -> MultiAgentState:
        """Knowledge Agent node - LangChain agent for external research"""
        message = state.get("user_message", "")
        user_name = state.get("user_name", "Unknown")
        
        logger.info(f"🔍 KNOWLEDGE AGENT - Researching external information for {user_name}")
        logger.info(f"📝 INPUT: {message[:100]}{'...' if len(message) > 100 else ''}")
        
        try:
            # Execute LangChain Knowledge Agent
            knowledge_context = await self.knowledge_agent.research_and_summarize(message, user_name)
            
            state["knowledge_context"] = knowledge_context
            state["current_agent"] = "knowledge_agent"
            state["agents_executed"].append("knowledge_agent")
            state["agent_execution_order"].append("knowledge_agent")
            self.execution_metrics["agent_executions"]["knowledge_agent"] += 1
            
            logger.info(f"📊 OUTPUT:")
            if isinstance(knowledge_context, dict):
                summary = knowledge_context.get('knowledge_summary', 'No research summary available')
                logger.info(f"   • Research Summary: {summary[:150]}{'...' if len(summary) > 150 else ''}")
                if 'sources_found' in knowledge_context:
                    logger.info(f"   • Sources Found: {knowledge_context.get('sources_found', 0)}")
                if 'research_type' in knowledge_context:
                    logger.info(f"   • Research Type: {knowledge_context.get('research_type', 'general')}")
            else:
                logger.info(f"   • Raw Research: {str(knowledge_context)[:150]}{'...' if len(str(knowledge_context)) > 150 else ''}")
            
            logger.info(f"🔍 KNOWLEDGE AGENT - Complete\n")
            
        except Exception as e:
            logger.error(f"❌ KNOWLEDGE AGENT - FAILED: {e}")
            state["knowledge_context"] = {"knowledge_summary": f"Knowledge search error: {str(e)}"}
            logger.info(f"🔍 KNOWLEDGE AGENT - Complete (with errors)\n")
        
        return state
    
    async def _organizer_node(self, state: MultiAgentState) -> MultiAgentState:
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
            
            self.execution_metrics["agent_executions"]["organizer"] += 1
            
            logger.info(f"📊 OUTPUT:")
            logger.info(f"   • Final Response: {final_response[:200]}{'...' if len(final_response) > 200 else ''}")
            if 'reasoning' in synthesis_result:
                logger.info(f"   • Reasoning: {synthesis_result.get('reasoning', '')[:100]}{'...' if len(str(synthesis_result.get('reasoning', ''))) > 100 else ''}")
            
            logger.info(f"🧠 ORGANIZER AGENT - Complete\n")
            
        except Exception as e:
            logger.error(f"❌ ORGANIZER AGENT - FAILED: {e}")
            final_response = f"I apologize, but I encountered an issue: {str(e)}"
            state["final_response"] = final_response
            logger.info(f"📊 OUTPUT (Error): {final_response}")
            logger.info(f"🧠 ORGANIZER AGENT - Complete (with errors)\n")
        
        return state
    
    async def _memory_writer_node(self, state: MultiAgentState) -> MultiAgentState:
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
            
            self.execution_metrics["agent_executions"]["memory_writer"] += 1
            
            logger.info(f"📊 OUTPUT:")
            if isinstance(writer_results, dict):
                logger.info(f"   • Facts Extracted: {writer_results.get('facts_extracted', 0)}")
                logger.info(f"   • Short-term Stored: {writer_results.get('short_term_stored', 0)}")
                logger.info(f"   • Long-term Stored: {writer_results.get('long_term_stored', 0)}")
                logger.info(f"   • Processing Status: {'Success' if writer_results.get('session_stored', False) else 'Failed'}")
            else:
                logger.info(f"   • Processing Result: {str(writer_results)[:100]}{'...' if len(str(writer_results)) > 100 else ''}")
            
            logger.info(f"📝 MEMORY WRITER AGENT - Complete")
            logger.info(f"="*80 + "\n")
            
        except Exception as e:
            logger.error(f"❌ MEMORY WRITER AGENT - FAILED: {e}")
            logger.info(f"📝 MEMORY WRITER AGENT - Complete (with errors)")
            logger.info(f"="*80 + "\n")
        
        return state
    
    # LangGraph Routing Functions
    def _route_next_agent(self, state: MultiAgentState) -> str:
        """Route to next agent based on state"""
        if state.get("should_research", False) or state.get("complexity_score", 0.0) > 0.5:
            return "research_needed"
        elif len(state.get("messages", [])) > 0:
            return "memory_only"  # Fixed: use the correct routing key
        else:
            return "end"
    
    def _should_research(self, state: MultiAgentState) -> str:
        """Decide if Knowledge Agent should be invoked"""
        if state.get("should_research", False):
            return "research"
        return "organize"
    
    def _should_write_memory(self, state: MultiAgentState) -> str:
        """Decide if Memory Writer should be invoked"""
        # Always write memory after organizing response
        if state.get("final_response"):
            return "write_memory"
        return "end"
    
    async def _calculate_complexity_score(self, message: str) -> float:
        """Calculate message complexity for routing decisions"""
        # Simple heuristics for complexity scoring
        complexity_indicators = {
            "questions": len([w for w in message.lower().split() if w in ["what", "how", "why", "when", "where", "who"]]),
            "length": min(len(message.split()) / 50, 1.0),
            "research_keywords": len([w for w in message.lower().split() if w in ["latest", "current", "recent", "today", "news"]]),
            "technical_terms": len([w for w in message.lower().split() if len(w) > 8])
        }
        
        score = (
            complexity_indicators["questions"] * 0.3 +
            complexity_indicators["length"] * 0.2 +
            complexity_indicators["research_keywords"] * 0.3 +
            complexity_indicators["technical_terms"] * 0.2
        )
        
        return min(score, 1.0)
    
    # Autonomous LangGraph Node Implementations
    async def _autonomous_router_node(self, state: MultiAgentState) -> MultiAgentState:
        """Autonomous router node - determines autonomous operation path"""
        operation_type = state.get("operation_type", "")
        logger.info(f"🤖 Autonomous router processing: {operation_type}")
        
        state["current_agent"] = "autonomous_router"
        state["agents_executed"].append("autonomous_router")
        state["agent_execution_order"].append("autonomous_router")
        
        # Set operation-specific parameters
        if operation_type in ["autonomous_thinking", "pattern_discovery"]:
            state["should_research"] = True
        elif operation_type in ["memory_maintenance", "cleanup"]:
            state["should_research"] = False
        elif operation_type in ["insight_generation", "milestone_tracking"]:
            state["should_research"] = True
        
        logger.info(f"🤖 Autonomous router decision: {operation_type}")
        return state
    
    async def _autonomous_memory_analysis_node(self, state: MultiAgentState) -> MultiAgentState:
        """Autonomous memory analysis node - analyzes target user's memory using autonomous agent's context"""
        logger.info(f"🧠 Autonomous memory analysis...")
        
        try:
            autonomous_user_name = state["user_name"]  # Always "Assistant"
            target_user_name = state.get("target_user_name", "Unknown")  # The user being analyzed
            operation_type = state.get("operation_type", "")
            
            # Use Memory Reader Agent for autonomous analysis of target user
            if operation_type == "autonomous_thinking":
                # Analyze recent conversation patterns of target user
                memory_context = await self.memory_reader_agent.get_complete_context(
                    target_user_name, "recent patterns and insights"
                )
            elif operation_type == "milestone_tracking":
                # Analyze goal-related memories of target user
                memory_context = await self.memory_reader_agent.get_complete_context(
                    target_user_name, "goals achievements milestones progress"
                )
            elif operation_type == "pattern_discovery":
                # Analyze behavioral patterns of target user
                memory_context = await self.memory_reader_agent.get_complete_context(
                    target_user_name, "patterns behaviors preferences trends"
                )
            else:
                # General memory analysis of target user
                memory_context = await self.memory_reader_agent.get_complete_context(
                    target_user_name, "important recent context"
                )
            
            # Store analysis in autonomous agent's working memory
            await self.memory_system.store_working_memory(
                user_name=autonomous_user_name,
                agent_name="memory_reader",
                content=f"Analyzed {target_user_name} for {operation_type}: {memory_context.get('context_summary', 'No summary')[:200]}..."
            )
            
            state["memory_context"] = memory_context
            state["current_agent"] = "autonomous_memory_analysis"
            state["agents_executed"].append("autonomous_memory_analysis")
            state["agent_execution_order"].append("autonomous_memory_analysis")
            
            self.execution_metrics["agent_executions"]["memory_reader"] += 1
            
            logger.info(f"✅ Autonomous memory analysis completed")
            
        except Exception as e:
            logger.error(f"❌ Autonomous memory analysis failed: {e}")
            state["memory_context"] = {"context_summary": f"Memory analysis error: {str(e)}"}
        
        return state
    
    async def _autonomous_research_node(self, state: MultiAgentState) -> MultiAgentState:
        """Autonomous research node"""
        logger.info(f"🔍 Autonomous research...")
        
        try:
            operation_type = state.get("operation_type", "")
            
            # Use Knowledge Agent for autonomous research
            if operation_type == "insight_generation":
                research_query = "latest trends insights patterns personal development"
            elif operation_type == "milestone_tracking":
                research_query = "goal achievement strategies progress tracking methods"
            elif operation_type == "pattern_discovery":
                research_query = "behavioral pattern analysis personal insights"
            else:
                research_query = "general knowledge insights trends"
            
            autonomous_user_name = state.get("user_name", "Assistant")
            knowledge_context = await self.knowledge_agent.research_and_summarize(research_query, autonomous_user_name)
            
            state["knowledge_context"] = knowledge_context
            state["current_agent"] = "autonomous_research"
            state["agents_executed"].append("autonomous_research")
            state["agent_execution_order"].append("autonomous_research")
            
            self.execution_metrics["agent_executions"]["knowledge_agent"] += 1
            
            logger.info(f"✅ Autonomous research completed")
            
        except Exception as e:
            logger.error(f"❌ Autonomous research failed: {e}")
            state["knowledge_context"] = {"knowledge_summary": f"Autonomous research error: {str(e)}"}
        
        return state
    
    async def _autonomous_synthesis_node(self, state: MultiAgentState) -> MultiAgentState:
        """Autonomous synthesis node - synthesizes insights about target user"""
        logger.info(f"🧠 Autonomous synthesis...")
        
        try:
            autonomous_user_name = state["user_name"]  # Always "Assistant"
            target_user_name = state.get("target_user_name", self.config.raw_config.get("user", {}).get("name", "Aswin"))
            operation_type = state.get("operation_type", "")
            memory_context = state.get("memory_context", {})
            knowledge_context = state.get("knowledge_context", {})
            
            # Create autonomous synthesis prompt for target user analysis
            autonomous_prompt = f"Autonomous {operation_type} for user {target_user_name}: Analyze patterns and generate insights"
            
            # Use Organizer Agent for autonomous synthesis (uses autonomous user's working memory)
            synthesis_result = await self.organizer_agent.synthesize_response(
                user_name=autonomous_user_name,  # Use autonomous user for organizer's working memory
                user_input=autonomous_prompt,
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
                    logger.info(f"🧠 Stored autonomous insight: {operation_type} for {target_user_name}")
                except Exception as e:
                    logger.error(f"Failed to store autonomous insight: {e}")
                
                # Use Memory Writer Agent with storage constraints (no long-term for autonomous)
                try:
                    insight_content = synthesis_result.get("response", "")
                    if len(insight_content) > 50:  # Only store substantial insights
                        
                        # Create conversation context for Memory Writer Agent
                        autonomous_user_message = f"Autonomous {operation_type} analysis"
                        autonomous_ai_response = insight_content
                        
                        # Process through Memory Writer Agent with constraints to prevent long-term storage
                        writer_results = await self.memory_writer_agent.process_conversation(
                            user_name=target_user_name,
                            user_message=autonomous_user_message,
                            ai_response=autonomous_ai_response,
                            conversation_metadata={
                                "source": "autonomous_discovery",
                                "operation_type": operation_type,
                                "generated_at": datetime.now().isoformat(),
                                "autonomous_agent": True
                            },
                            storage_constraints={
                                "disable_long_term": True,        # Block long-term storage for autonomous
                                "max_importance_override": 0.8,   # Cap importance to prevent long-term
                                "force_short_term": True          # Ensure short-term storage only
                            }
                        )
                        
                        logger.info(f"📝 Memory Writer Agent processed autonomous discovery for {target_user_name}")
                        logger.info(f"   • Facts Extracted: {writer_results.get('facts_extracted', 0)}")
                        logger.info(f"   • Short-term Stored: {writer_results.get('short_term_stored', 0)}")
                        logger.info(f"   • Long-term Stored: {writer_results.get('long_term_stored', 0)}")
                        
                        # Add Memory Writer to agents executed list
                        state["agents_executed"].append("memory_writer")
                        state["agent_execution_order"].append("memory_writer")
                        self.execution_metrics["agent_executions"]["memory_writer"] += 1
                        
                except Exception as e:
                    logger.error(f"Failed to process autonomous discovery via Memory Writer Agent: {e}")
            
            state["final_response"] = synthesis_result.get("response", f"Autonomous {operation_type} completed")
            state["current_agent"] = "autonomous_synthesis"
            state["agents_executed"].append("autonomous_synthesis")
            state["agent_execution_order"].append("autonomous_synthesis")
            
            self.execution_metrics["agent_executions"]["organizer"] += 1
            
            logger.info(f"✅ Autonomous synthesis completed")
            
        except Exception as e:
            logger.error(f"❌ Autonomous synthesis failed: {e}")
            state["final_response"] = f"Autonomous synthesis failed: {str(e)}"
        
        return state
    
    # Removed _autonomous_memory_update_node - not needed
    # Memory Writer Agent is called during synthesis if needed
    # Working memory auto-trims to 7 items, short-term memory expires via TTL
    
    async def _autonomous_broadcast_node(self, state: MultiAgentState) -> MultiAgentState:
        """Autonomous broadcast node - sends updates via WebSocket using existing infrastructure"""
        logger.info(f"📡 Autonomous broadcast...")
        
        try:
            if state.get("broadcast_updates", False):
                operation_type = state.get("operation_type", "")
                result = state.get("final_response", "")
                user_name = state.get("user_name", "Assistant")
                
                # Use appropriate broadcast method based on operation type
                if operation_type in ["insight_generation", "weekly_insights"]:
                    # Use broadcast_autonomous_insight for insight-type operations
                    insight_data = {
                        "id": f"insight_{datetime.now().timestamp()}",
                        "title": f"Autonomous {operation_type.replace('_', ' ').title()}",
                        "content": result,
                        "category": operation_type,
                        "agents_executed": state.get("agents_executed", []),
                        "processing_time": state.get("processing_time", 0),
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    # Note: This will be called by the WebSocket manager when available
                    logger.info(f"📡 Prepared autonomous insight broadcast: {operation_type}")
                    state["broadcast_data"] = {"type": "insight", "data": insight_data}
                    
                else:
                    # Use broadcast_thinking_update for other autonomous operations
                    thinking_data = {
                        "type": operation_type,
                        "status": "completed",
                        "result": result,
                        "agents_executed": state.get("agents_executed", []),
                        "processing_time": state.get("processing_time", 0),
                        "operation_metadata": {
                            "trigger_source": state.get("trigger_source", "unknown"),
                            "autonomous_context": state.get("autonomous_context", {}),
                            "autonomous_parameters": state.get("autonomous_parameters", {})
                        },
                        "timestamp": datetime.now().isoformat()
                    }
                    
                    logger.info(f"📡 Prepared thinking update broadcast: {operation_type}")
                    state["broadcast_data"] = {"type": "thinking", "data": thinking_data}
            
            state["current_agent"] = "autonomous_broadcast"
            state["agents_executed"].append("autonomous_broadcast")
            state["agent_execution_order"].append("autonomous_broadcast")
            
            logger.info(f"✅ Autonomous broadcast data prepared")
            
        except Exception as e:
            logger.error(f"❌ Autonomous broadcast failed: {e}")
            state["broadcast_data"] = None
        
        return state
    
    def set_websocket_manager(self, websocket_manager):
        """Set WebSocket manager for autonomous broadcasting"""
        self.websocket_manager = websocket_manager
        logger.info("📡 WebSocket manager connected to LangGraph orchestrator")
    
    async def _send_autonomous_broadcast(self, broadcast_data: Dict[str, Any]):
        """Send autonomous broadcast using WebSocket manager"""
        if not hasattr(self, 'websocket_manager') or not self.websocket_manager:
            logger.warning("⚠️ WebSocket manager not available for autonomous broadcast")
            return
        
        try:
            broadcast_type = broadcast_data.get("type", "thinking")
            data = broadcast_data.get("data", {})
            
            if broadcast_type == "insight":
                await self.websocket_manager.broadcast_autonomous_insight(data)
                logger.info("📡 Autonomous insight broadcast sent")
            else:
                await self.websocket_manager.broadcast_thinking_update(data)
                logger.info("📡 Autonomous thinking broadcast sent")
                
        except Exception as e:
            logger.error(f"❌ Failed to send autonomous broadcast: {e}")
    
    # Autonomous Routing Functions
    def _route_autonomous_operation(self, state: MultiAgentState) -> str:
        """Route autonomous operations based on type"""
        operation_type = state.get("operation_type", "")
        
        # No memory_maintenance needed - TTL and working memory limits handle cleanup automatically
        if operation_type in ["autonomous_thinking", "pattern_discovery", "insight_generation", "milestone_tracking"]:
            return "full_analysis"
        elif operation_type in ["research_trends", "knowledge_update"]:
            return "research_only"
        elif operation_type in ["status_broadcast", "heartbeat"]:
            return "broadcast_only"
        else:
            return "memory_analysis"
    
    def _autonomous_should_research(self, state: MultiAgentState) -> str:
        """Decide if autonomous operation needs research"""
        if state.get("should_research", False):
            return "research"
        return "synthesize"
    
    # Removed _autonomous_should_update_memory - no manual memory updates needed
    # Working memory has 7-item limit with auto-trim, short-term memory has TTL
    
    def _get_workflow_pattern(self, agents_executed: List[str]) -> str:
        """Determine workflow pattern based on agents executed"""
        if "knowledge_agent" in agents_executed:
            if len(agents_executed) > 4:
                return WorkflowPattern.COMPLEX_REASONING.value
            else:
                return WorkflowPattern.RESEARCH_ENHANCED.value
        else:
            return WorkflowPattern.SIMPLE_MEMORY_ONLY.value
    
    async def process_message_simplified(self, user_name: str, message: str) -> str:
        """
        Simplified interface that returns just the response string
        """
        result = await self.process_message(user_name, message)
        # Handle WorkflowExecutionOutput properly
        if hasattr(result, 'final_response'):
            return result.final_response
        elif hasattr(result, 'model_dump'):
            result_dict = result.model_dump()
            return result_dict.get("final_response", "I'm sorry, I couldn't process your request.")
        elif isinstance(result, dict):
            return result.get("response", "I'm sorry, I couldn't process your request.")
        else:
            return str(result)
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get comprehensive LangGraph workflow status and metrics"""
        return {
            "workflow_active": True,
            "orchestrator_type": "langgraph_multi_agent",
            "architecture": "langgraph_4_agent_system",
            "agents": {
                "memory_reader": "LangChain Memory Reader Agent",
                "memory_writer": "LangChain Memory Writer Agent",
                "knowledge_agent": "LangChain Knowledge Agent",
                "organizer": "LangChain Organizer Agent"
            },
            "langgraph_features": {
                "state_graph": "enabled",
                "conditional_routing": "enabled",
                "checkpoint_persistence": "enabled",
                "node_based_execution": "enabled"
            },
            "workflow_capabilities": [
                "multi_agent_coordination",
                "intelligent_routing",
                "state_management",
                "conditional_execution",
                "checkpoint_recovery"
            ],
            "execution_metrics": self.execution_metrics,
            "memory_system_connected": self.memory_system is not None,
            "transformers_service_connected": self.transformers_service is not None,
            "timestamp": datetime.now().isoformat()
        }
    
    async def get_agent_statuses(self) -> Dict[str, Any]:
        """Get detailed status of all agents"""
        return {
            "memory_reader": self.memory_reader_agent.get_agent_info(),
            "memory_writer": self.memory_writer_agent.get_agent_info(),
            "knowledge": self.knowledge_agent.get_agent_status(),
            "organizer": self.organizer_agent.get_agent_info(),
            "orchestrator_metrics": self.execution_metrics
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get comprehensive LangGraph orchestrator metrics"""
        total_agent_executions = sum(self.execution_metrics["agent_executions"].values())
        
        return {
            "architecture": "langgraph_multi_agent",
            "total_requests": self.execution_metrics["total_requests"],
            "agent_executions": self.execution_metrics["agent_executions"],
            "workflow_patterns": self.execution_metrics["workflow_patterns"],
            "performance": {
                "average_processing_time": round(self.execution_metrics["average_processing_time"], 3),
                "total_agent_executions": total_agent_executions
            },
            "langgraph_features": {
                "state_management": "enabled",
                "conditional_routing": "enabled",
                "checkpoint_support": "enabled",
                "parallel_execution": "optimized"
            }
        }
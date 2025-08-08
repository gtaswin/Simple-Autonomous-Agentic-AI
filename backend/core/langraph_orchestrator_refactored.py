"""
Refactored LangGraph Orchestrator - Clean, focused, and modular
Reduced from 1280 lines to ~400 lines by extracting responsibilities
"""

import logging
from typing import Dict, List, Optional, Any, TypedDict
from datetime import datetime

# LangGraph imports
from langgraph.graph import StateGraph, END, START
from langgraph.checkpoint.memory import MemorySaver

# Structured output support
from core.output_schemas import WorkflowExecutionOutput, WorkflowPattern
from core.output_parser import workflow_output

# Modular components
from core.workflow_nodes import WorkflowNodes
from core.autonomous_manager import AutonomousManager
from core.workflow_utils import WorkflowMetrics, safe_get, get_workflow_pattern
from core.parallel_coordinator import parallel_coordinator
from core.streaming_manager import streaming_manager

# Agents and core services
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
    operation_type: Optional[str]
    trigger_source: Optional[str]
    autonomous_context: Optional[Dict[str, Any]]
    autonomous_parameters: Optional[Dict[str, Any]]
    broadcast_updates: bool
    
    # Metadata
    timestamp: str
    processing_time: float
    agent_execution_order: List[str]


class LangGraphMultiAgentOrchestrator:
    """Clean, modular LangGraph orchestrator with separated responsibilities"""
    
    def __init__(
        self,
        memory_system: AutonomousMemorySystem,
        config: AssistantConfig,
        transformers_service: TransformersService
    ):
        self.memory_system = memory_system
        self.config = config
        self.transformers_service = transformers_service
        
        # Initialize coordinators first (needed by modular components)
        self.parallel_coordinator = parallel_coordinator
        self.streaming_manager = streaming_manager
        self.execution_metrics = WorkflowMetrics()
        
        # Initialize 4 LangChain agents
        self._initialize_agents()
        
        # Initialize modular components (after streaming_manager is available)
        self.workflow_nodes = WorkflowNodes(self)
        self.autonomous_manager = AutonomousManager(self)
        
        # Build LangGraph workflows
        self.app = self._build_user_workflow()
        self.autonomous_app = self._build_autonomous_workflow()
        
        logger.info("✅ LangGraph Multi-Agent Orchestrator initialized (refactored)")
    
    def _initialize_agents(self):
        """Initialize the 4 LangChain agents"""
        self.memory_reader_agent = MemoryReaderAgent(
            transformers_service=self.transformers_service,
            memory_system=self.memory_system
        )
        
        self.memory_writer_agent = MemoryWriterAgent(
            transformers_service=self.transformers_service,
            memory_system=self.memory_system
        )
        
        self.knowledge_agent = KnowledgeAgent(
            config=self.config,
            transformers_service=self.transformers_service
        )
        
        self.organizer_agent = OrganizerAgent(
            config=self.config,
            transformers_service=self.transformers_service,
            memory_reader_agent=self.memory_reader_agent,
            memory_writer_agent=self.memory_writer_agent,
            memory_system=self.memory_system
        )
    
    def _build_user_workflow(self) -> StateGraph:
        """Build the main user interaction workflow"""
        workflow = StateGraph(MultiAgentState)
        
        # Add nodes using modular workflow nodes
        workflow.add_node("router", self.workflow_nodes.router_node)
        workflow.add_node("memory_reader", self.workflow_nodes.memory_reader_node)
        workflow.add_node("knowledge_agent", self.workflow_nodes.knowledge_agent_node)
        workflow.add_node("organizer", self.workflow_nodes.organizer_node)
        workflow.add_node("memory_writer", self.workflow_nodes.memory_writer_node)
        
        # Define workflow edges
        workflow.set_entry_point("router")
        
        # Router decides the path
        workflow.add_conditional_edges(
            "router",
            self._determine_next_agent_from_router,
            {
                "knowledge_agent": "knowledge_agent",
                "memory_reader": "memory_reader"
            }
        )
        
        # Knowledge agent always goes to memory reader
        workflow.add_edge("knowledge_agent", "memory_reader")
        
        # Memory reader always goes to organizer
        workflow.add_edge("memory_reader", "organizer")
        
        # Organizer always goes to memory writer
        workflow.add_edge("organizer", "memory_writer")
        
        # Memory writer ends the workflow
        workflow.add_edge("memory_writer", END)
        
        return workflow.compile(checkpointer=MemorySaver())
    
    def _build_autonomous_workflow(self) -> StateGraph:
        """Build the autonomous operations workflow"""
        workflow = StateGraph(MultiAgentState)
        
        # Add autonomous nodes using modular autonomous manager
        workflow.add_node("autonomous_router", self.autonomous_manager.autonomous_router_node)
        workflow.add_node("autonomous_memory_analysis", self.autonomous_manager.autonomous_memory_analysis_node)
        workflow.add_node("autonomous_research", self.autonomous_manager.autonomous_research_node)
        workflow.add_node("autonomous_synthesis", self.autonomous_manager.autonomous_synthesis_node)
        workflow.add_node("autonomous_broadcast", self.autonomous_manager.autonomous_broadcast_node)
        
        # Define autonomous workflow edges
        workflow.set_entry_point("autonomous_router")
        
        # Router decides the operation path
        workflow.add_conditional_edges(
            "autonomous_router",
            self.autonomous_manager.route_autonomous_operation,
            {
                "memory_analysis": "autonomous_memory_analysis",
                "full_analysis": "autonomous_memory_analysis",
                "research_only": "autonomous_research",
                "broadcast_only": "autonomous_broadcast",
                "end": END
            }
        )
        
        # Memory analysis can go to research or synthesis
        workflow.add_conditional_edges(
            "autonomous_memory_analysis",
            self.autonomous_manager.autonomous_should_research,
            {
                "research": "autonomous_research",
                "synthesize": "autonomous_synthesis"
            }
        )
        
        # Research always goes to synthesis
        workflow.add_edge("autonomous_research", "autonomous_synthesis")
        
        # Synthesis goes to broadcast
        workflow.add_edge("autonomous_synthesis", "autonomous_broadcast")
        
        # Broadcast ends the workflow
        workflow.add_edge("autonomous_broadcast", END)
        
        return workflow.compile(checkpointer=MemorySaver())
    
    @workflow_output(capture_performance=True)
    async def process_message(self, user_name: str, message: str) -> WorkflowExecutionOutput:
        """Process user message through LangGraph multi-agent workflow"""
        start_time = datetime.now()
        workflow_id = f"{user_name}_{datetime.now().timestamp()}"
        
        try:
            self.execution_metrics.metrics["total_requests"] += 1
            
            logger.info(f"\\n" + "🚀"*40)
            logger.info(f"🚀 LANGGRAPH WORKFLOW STARTED")
            logger.info(f"👤 User: {user_name}")
            logger.info(f"💬 Message: {message}")
            logger.info(f"⏰ Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            logger.info(f"🚀"*40)
            
            # Initialize workflow streaming with correct parameters
            await self.streaming_manager.start_workflow_stream(
                workflow_id=workflow_id,
                user_name=user_name,
                user_message=message,
                agents_to_execute=["router", "memory_reader", "organizer", "memory_writer"]  # Default agent flow
            )
            
            # Prepare initial state
            initial_state = {
                "messages": [{"role": "user", "content": message}],
                "user_name": user_name,
                "user_message": message,
                "memory_context": None,
                "knowledge_context": None,
                "current_agent": "router",
                "agents_executed": [],
                "next_agent": None,
                "should_research": False,
                "complexity_score": 0.0,
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
            
            # Execute workflow
            config = {"configurable": {"thread_id": f"{user_name}_{datetime.now().timestamp()}"}}
            result = await self.app.ainvoke(initial_state, config)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Extract values safely
            agents_used = safe_get(result, "agents_executed", [])
            final_response = safe_get(result, "final_response", "I'm sorry, I couldn't process your request.")
            agent_execution_order = safe_get(result, "agent_execution_order", [])
            complexity_score = safe_get(result, "complexity_score", 0.0)
            
            # Update metrics
            self.execution_metrics.update_processing_time(processing_time)
            
            # Determine workflow pattern
            workflow_pattern = get_workflow_pattern(agents_used)
            self.execution_metrics.increment_workflow_pattern(workflow_pattern)
            
            # Complete workflow streaming
            await self.streaming_manager.complete_workflow_stream(
                workflow_id=workflow_id,
                final_response=final_response,
                total_execution_time=processing_time,
                agents_executed=agents_used,
                workflow_pattern=workflow_pattern
            )
            
            # Log completion
            logger.info(f"🏁 WORKFLOW COMPLETED in {processing_time:.2f}s")
            logger.info(f"🤖 Agents: {' → '.join(agents_used)}")
            logger.info(f"📝 Response: {final_response[:150]}{'...' if len(final_response) > 150 else ''}")
            
            # Return structured response for decorator processing
            return {
                "response": final_response,
                "metadata": {
                    "architecture": "langgraph_multi_agent",
                    "agents_executed": agents_used,
                    "execution_order": agent_execution_order,
                    "workflow_pattern": workflow_pattern,
                    "processing_time": processing_time,
                    "complexity_score": complexity_score,
                    "memory_context_available": isinstance(safe_get(result, "memory_context"), dict),
                    "knowledge_context_available": isinstance(safe_get(result, "knowledge_context"), dict)
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
    
    async def execute_autonomous_operation(
        self,
        operation_type: str,
        trigger_source: str = "manual",
        target_user_name: str = None,
        broadcast_updates: bool = True
    ) -> Dict[str, Any]:
        """Execute autonomous operation through dedicated autonomous workflow"""
        start_time = datetime.now()
        
        try:
            # Get configured user if not specified
            if not target_user_name:
                target_user_name = self.config.get("user.name", "Unknown")
            
            logger.info(f"🤖 Starting autonomous operation: {operation_type} for {target_user_name}")
            
            # Prepare autonomous state
            autonomous_state = {
                "messages": [],
                "user_name": "Assistant",  # Autonomous operations are by Assistant
                "user_message": f"autonomous {operation_type}",
                "memory_context": None,
                "knowledge_context": None,
                "current_agent": "autonomous_router",
                "agents_executed": [],
                "next_agent": None,
                "should_research": operation_type in ["insight_generation", "autonomous_thinking"],
                "complexity_score": 0.8,  # Autonomous operations are typically complex
                "final_response": None,
                "operation_type": operation_type,
                "trigger_source": trigger_source,
                "target_user_name": target_user_name,
                "autonomous_context": {"autonomous": True},
                "autonomous_parameters": {"operation": operation_type},
                "broadcast_updates": broadcast_updates,
                "timestamp": datetime.now().isoformat(),
                "processing_time": 0.0,
                "agent_execution_order": []
            }
            
            # Execute autonomous workflow
            config = {"configurable": {"thread_id": f"autonomous_{datetime.now().timestamp()}"}}
            result = await self.autonomous_app.ainvoke(autonomous_state, config)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            self.execution_metrics.update_processing_time(processing_time)
            
            logger.info(f"✅ Autonomous operation completed in {processing_time:.2f}s: {operation_type}")
            
            return {
                "operation_type": operation_type,
                "trigger_source": trigger_source,
                "result": safe_get(result, "final_response", "Autonomous operation completed"),
                "metadata": {
                    "architecture": "langgraph_autonomous_multi_agent",
                    "agents_executed": safe_get(result, "agents_executed", []),
                    "execution_order": safe_get(result, "agent_execution_order", []),
                    "processing_time": processing_time,
                    "broadcast_sent": safe_get(result, "broadcast_data") is not None and broadcast_updates
                },
                "timestamp": datetime.now().isoformat(),
                "user_name": "Assistant",
                "target_user_name": target_user_name
            }
            
        except Exception as e:
            logger.error(f"❌ Autonomous operation failed: {operation_type} - {e}")
            return {
                "operation_type": operation_type,
                "trigger_source": trigger_source,
                "result": f"Autonomous operation failed: {str(e)}",
                "metadata": {
                    "error": str(e),
                    "processing_failed": True,
                    "processing_time": (datetime.now() - start_time).total_seconds()
                },
                "timestamp": datetime.now().isoformat(),
                "user_name": "Assistant",
                "target_user_name": target_user_name or "Unknown"
            }
    
    def _determine_next_agent_from_router(self, state: MultiAgentState) -> str:
        """Determine next agent from router based on complexity and research needs"""
        if state.get("should_research", False) or state.get("complexity_score", 0.0) > 0.5:
            return "knowledge_agent"
        else:
            return "memory_reader"
    
    def get_workflow_status(self) -> Dict[str, Any]:
        """Get comprehensive workflow status and metrics"""
        return {
            "workflow_active": True,
            "orchestrator_type": "langgraph_multi_agent_refactored",
            "architecture": "modular_4_agent_system",
            "components": {
                "workflow_nodes": "Separated user workflow nodes",
                "autonomous_manager": "Separated autonomous operations",
                "workflow_utils": "Shared utilities and metrics"
            },
            "metrics": self.execution_metrics.get_status(),
            "agents": {
                "memory_reader": "LangChain Memory Reader Agent",
                "memory_writer": "LangChain Memory Writer Agent", 
                "knowledge_agent": "LangChain Knowledge Agent",
                "organizer": "LangChain Organizer Agent"
            }
        }
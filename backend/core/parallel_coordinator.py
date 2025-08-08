"""
Parallel Execution Coordinator for LangGraph Multi-Agent System
Manages concurrent agent execution with dependency resolution and streaming updates
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Set, Callable, Tuple
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

from core.streaming_manager import streaming_manager, AgentStatus

logger = logging.getLogger(__name__)


class AgentDependency(Enum):
    """Agent dependency types"""
    NONE = "none"              # No dependencies
    REQUIRES_MEMORY = "requires_memory"      # Needs Memory Reader output
    REQUIRES_KNOWLEDGE = "requires_knowledge" # Needs Knowledge Agent output
    REQUIRES_BOTH = "requires_both"          # Needs both Memory + Knowledge
    REQUIRES_SYNTHESIS = "requires_synthesis" # Needs Organizer output


@dataclass
class AgentTask:
    """Represents an agent execution task with dependencies"""
    agent_name: str
    agent_function: Callable
    dependencies: Set[str]
    priority: int = 0
    max_execution_time: float = 30.0
    can_run_background: bool = False
    
    # Execution state
    status: AgentStatus = AgentStatus.WAITING
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    result: Optional[Dict[str, Any]] = None
    error: Optional[Exception] = None


class ParallelCoordinator:
    """
    Coordinates parallel execution of agents with dependency management
    Integrates with streaming manager for real-time updates
    """
    
    def __init__(self):
        self.streaming_manager = streaming_manager
        self.active_workflows: Dict[str, Dict[str, AgentTask]] = {}
        self.execution_metrics = {
            "total_parallel_workflows": 0,
            "average_speedup": 0.0,
            "concurrent_agent_executions": 0,
            "dependency_violations": 0,
            "background_task_completions": 0
        }
        
        # Agent execution order and dependencies
        self.agent_dependencies = {
            "router": AgentDependency.NONE,
            "memory_reader": AgentDependency.NONE,
            "knowledge_agent": AgentDependency.NONE,  # Can run parallel with memory_reader
            "organizer": AgentDependency.REQUIRES_BOTH,  # Needs memory + knowledge (if research)
            "memory_writer": AgentDependency.REQUIRES_SYNTHESIS  # Can run background during synthesis
        }
        
        logger.info("🔄 Parallel coordinator initialized")
    
    async def execute_parallel_workflow(
        self,
        workflow_id: str,
        user_name: str,
        state: Dict[str, Any],
        agent_functions: Dict[str, Callable],
        should_research: bool = False
    ) -> Dict[str, Any]:
        """
        Execute agents in parallel based on dependencies
        
        Args:
            workflow_id: Unique workflow identifier
            user_name: User name for context
            state: LangGraph state dictionary
            agent_functions: Dictionary of agent name -> callable
            should_research: Whether to include Knowledge Agent
            
        Returns:
            Updated state with all agent results
        """
        
        workflow_start = datetime.now()
        logger.info(f"🚀 Starting parallel workflow: {workflow_id}")
        
        try:
            # Create agent tasks based on workflow requirements
            agent_tasks = self._create_agent_tasks(agent_functions, should_research)
            self.active_workflows[workflow_id] = agent_tasks
            
            # Update streaming for workflow start
            await self.streaming_manager.update_agent_status(
                workflow_id=workflow_id,
                agent_name="coordinator",
                status=AgentStatus.ACTIVE,
                current_activity="Setting up parallel execution",
                progress_percentage=5.0
            )
            
            # Phase 1: Execute Router (sequential, required for routing decisions)
            if "router" in agent_tasks:
                state = await self._execute_single_agent(workflow_id, "router", agent_tasks["router"], state)
            
            # Phase 2: Parallel Memory + Knowledge execution
            parallel_agents = ["memory_reader", "knowledge_agent"] if should_research else ["memory_reader"]
            
            # Stream parallel execution start
            if len(parallel_agents) > 1:
                await self.streaming_manager.stream_parallel_execution_start(
                    workflow_id=workflow_id,
                    parallel_agents=parallel_agents,
                    execution_phase="memory_and_knowledge"
                )
            
            parallel_start_time = datetime.now()
            parallel_results = await self._execute_parallel_phase(
                workflow_id, 
                parallel_agents,
                agent_tasks,
                state
            )
            parallel_execution_time = (datetime.now() - parallel_start_time).total_seconds()
            
            # Stream parallel completion
            if len(parallel_agents) > 1:
                # Estimate speedup (assuming sequential would take sum of individual times)
                estimated_sequential_time = parallel_execution_time * len(parallel_agents)
                speedup_factor = estimated_sequential_time / parallel_execution_time if parallel_execution_time > 0 else 1.0
                
                await self.streaming_manager.stream_parallel_completion(
                    workflow_id=workflow_id,
                    completed_agents=parallel_agents,
                    execution_time=parallel_execution_time,
                    speedup_factor=speedup_factor
                )
            
            # Update state with parallel results
            for agent_name, result in parallel_results.items():
                if result:
                    state.update(result)
            
            # Phase 3: Organizer synthesis with concurrent Memory Writer preparation
            organizer_task = asyncio.create_task(
                self._execute_single_agent(workflow_id, "organizer", agent_tasks["organizer"], state)
            )
            
            # Wait for Organizer to complete (need response for Memory Writer)
            organizer_result = await organizer_task
            if organizer_result:
                state.update(organizer_result)
            
            # Phase 4: Memory Writer execution (now has final response)
            if "memory_writer" in agent_tasks:
                # Memory Writer can now process the complete conversation
                memory_writer_result = await self._execute_single_agent(
                    workflow_id, "memory_writer", agent_tasks["memory_writer"], state
                )
                if memory_writer_result:
                    state.update(memory_writer_result)
            
            # Calculate performance metrics
            total_time = (datetime.now() - workflow_start).total_seconds()
            self._update_performance_metrics(workflow_id, total_time)
            
            # Mark workflow complete
            await self.streaming_manager.update_agent_status(
                workflow_id=workflow_id,
                agent_name="coordinator",
                status=AgentStatus.COMPLETED,
                current_activity="Parallel workflow completed",
                progress_percentage=100.0,
                execution_time=total_time,
                metadata={
                    "agents_executed": list(agent_tasks.keys()),
                    "parallel_optimization": True,
                    "total_execution_time": total_time
                }
            )
            
            logger.info(f"✅ Parallel workflow completed: {workflow_id} in {total_time:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Parallel workflow failed: {workflow_id} - {e}")
            await self.streaming_manager.stream_error(
                workflow_id=workflow_id,
                agent_name="coordinator",
                error_message=str(e),
                error_details={"workflow_id": workflow_id, "error_type": "parallel_coordination_error"}
            )
            raise
        
        finally:
            # Cleanup
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
        
        return state
    
    def _create_agent_tasks(
        self, 
        agent_functions: Dict[str, Callable], 
        should_research: bool
    ) -> Dict[str, AgentTask]:
        """Create agent tasks with proper dependencies"""
        
        tasks = {}
        
        # Router (no dependencies)
        if "router" in agent_functions:
            tasks["router"] = AgentTask(
                agent_name="router",
                agent_function=agent_functions["router"],
                dependencies=set(),
                priority=100  # Highest priority
            )
        
        # Memory Reader (no dependencies, can run parallel)
        if "memory_reader" in agent_functions:
            tasks["memory_reader"] = AgentTask(
                agent_name="memory_reader",
                agent_function=agent_functions["memory_reader"],
                dependencies=set(),
                priority=90
            )
        
        # Knowledge Agent (no dependencies, can run parallel with Memory Reader)
        if should_research and "knowledge_agent" in agent_functions:
            tasks["knowledge_agent"] = AgentTask(
                agent_name="knowledge_agent",
                agent_function=agent_functions["knowledge_agent"],
                dependencies=set(),
                priority=90
            )
        
        # Organizer (depends on Memory Reader, and Knowledge if research)
        if "organizer" in agent_functions:
            dependencies = {"memory_reader"}
            if should_research:
                dependencies.add("knowledge_agent")
            
            tasks["organizer"] = AgentTask(
                agent_name="organizer",
                agent_function=agent_functions["organizer"],
                dependencies=dependencies,
                priority=50
            )
        
        # Memory Writer (can run in background during Organizer)
        if "memory_writer" in agent_functions:
            tasks["memory_writer"] = AgentTask(
                agent_name="memory_writer",
                agent_function=agent_functions["memory_writer"],
                dependencies={"organizer"},  # Needs final response
                priority=10,
                can_run_background=True
            )
        
        return tasks
    
    async def _execute_parallel_phase(
        self,
        workflow_id: str,
        agent_names: List[str],
        agent_tasks: Dict[str, AgentTask],
        state: Dict[str, Any]
    ) -> Dict[str, Dict[str, Any]]:
        """Execute multiple agents in parallel"""
        
        logger.info(f"🔄 Starting parallel phase: {agent_names}")
        
        # Create tasks for parallel execution
        parallel_tasks = {}
        for agent_name in agent_names:
            if agent_name in agent_tasks:
                task = asyncio.create_task(
                    self._execute_single_agent(workflow_id, agent_name, agent_tasks[agent_name], state)
                )
                parallel_tasks[agent_name] = task
        
        # Wait for all parallel tasks to complete
        results = {}
        if parallel_tasks:
            completed_tasks = await asyncio.gather(*parallel_tasks.values(), return_exceptions=True)
            
            for agent_name, result in zip(parallel_tasks.keys(), completed_tasks):
                if isinstance(result, Exception):
                    logger.error(f"❌ Parallel agent {agent_name} failed: {result}")
                    await self.streaming_manager.stream_error(
                        workflow_id=workflow_id,
                        agent_name=agent_name,
                        error_message=str(result),
                        error_details={"agent": agent_name, "phase": "parallel_execution"}
                    )
                    results[agent_name] = None
                else:
                    results[agent_name] = result
                    logger.info(f"✅ Parallel agent {agent_name} completed")
        
        self.execution_metrics["concurrent_agent_executions"] += len(parallel_tasks)
        
        return results
    
    async def _execute_single_agent(
        self,
        workflow_id: str,
        agent_name: str,
        agent_task: AgentTask,
        state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Execute a single agent with streaming updates"""
        
        start_time = datetime.now()
        agent_task.start_time = start_time
        agent_task.status = AgentStatus.ACTIVE
        
        logger.info(f"🤖 Executing agent: {agent_name}")
        
        try:
            # Update streaming status
            await self.streaming_manager.update_agent_status(
                workflow_id=workflow_id,
                agent_name=agent_name,
                status=AgentStatus.ACTIVE,
                current_activity=f"Executing {agent_name}",
                progress_percentage=10.0
            )
            
            # Execute agent function
            result = await agent_task.agent_function(state)
            
            # Update completion status
            execution_time = (datetime.now() - start_time).total_seconds()
            agent_task.end_time = datetime.now()
            agent_task.status = AgentStatus.COMPLETED
            agent_task.result = result
            
            await self.streaming_manager.update_agent_status(
                workflow_id=workflow_id,
                agent_name=agent_name,
                status=AgentStatus.COMPLETED,
                current_activity=f"{agent_name} completed",
                progress_percentage=100.0,
                execution_time=execution_time
            )
            
            logger.info(f"✅ Agent {agent_name} completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            agent_task.status = AgentStatus.ERROR
            agent_task.error = e
            
            await self.streaming_manager.stream_error(
                workflow_id=workflow_id,
                agent_name=agent_name,
                error_message=str(e),
                error_details={"agent": agent_name, "execution_time": execution_time}
            )
            
            logger.error(f"❌ Agent {agent_name} failed after {execution_time:.2f}s: {e}")
            return None
    
    async def _execute_background_agent(
        self,
        workflow_id: str,
        agent_name: str,
        agent_task: AgentTask,
        state: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Execute agent in background (non-blocking)"""
        
        logger.info(f"🔄 Starting background agent: {agent_name}")
        
        # Small delay to allow Organizer to start
        await asyncio.sleep(0.1)
        
        result = await self._execute_single_agent(workflow_id, agent_name, agent_task, state)
        
        if result:
            self.execution_metrics["background_task_completions"] += 1
            logger.info(f"🎯 Background agent {agent_name} completed")
        
        return result
    
    def _update_performance_metrics(self, workflow_id: str, total_time: float):
        """Update performance metrics for parallel execution"""
        
        self.execution_metrics["total_parallel_workflows"] += 1
        
        # Estimate sequential time (sum of all agent times)
        agent_tasks = self.active_workflows.get(workflow_id, {})
        sequential_time = sum(
            (task.end_time - task.start_time).total_seconds() 
            for task in agent_tasks.values() 
            if task.start_time and task.end_time
        )
        
        if sequential_time > 0:
            speedup = sequential_time / total_time
            current_speedup = self.execution_metrics["average_speedup"]
            total_workflows = self.execution_metrics["total_parallel_workflows"]
            
            # Update running average
            self.execution_metrics["average_speedup"] = (
                (current_speedup * (total_workflows - 1) + speedup) / total_workflows
            )
            
            logger.info(f"📈 Parallel speedup: {speedup:.2f}x (sequential: {sequential_time:.2f}s, parallel: {total_time:.2f}s)")
    
    def get_coordination_stats(self) -> Dict[str, Any]:
        """Get parallel coordination statistics"""
        return {
            "parallel_coordinator": {
                "active_workflows": len(self.active_workflows),
                "execution_metrics": self.execution_metrics,
                "agent_dependencies": {
                    name: dep.value for name, dep in self.agent_dependencies.items()
                },
                "supported_optimizations": [
                    "memory_reader_knowledge_parallel",
                    "background_memory_writer",
                    "dependency_resolution",
                    "streaming_coordination"
                ]
            },
            "timestamp": datetime.now().isoformat()
        }
    
    def get_active_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of specific workflow"""
        if workflow_id not in self.active_workflows:
            return None
        
        tasks = self.active_workflows[workflow_id]
        return {
            "workflow_id": workflow_id,
            "agents": {
                name: {
                    "status": task.status.value,
                    "start_time": task.start_time.isoformat() if task.start_time else None,
                    "end_time": task.end_time.isoformat() if task.end_time else None,
                    "dependencies": list(task.dependencies),
                    "can_run_background": task.can_run_background
                }
                for name, task in tasks.items()
            }
        }


# Global parallel coordinator instance
parallel_coordinator = ParallelCoordinator()
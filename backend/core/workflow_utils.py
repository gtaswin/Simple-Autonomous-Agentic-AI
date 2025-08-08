"""
Workflow Utilities - Shared helper functions for LangGraph orchestration
"""

import logging
from typing import Dict, Any, Union
from core.output_schemas import WorkflowPattern

logger = logging.getLogger(__name__)


def safe_get(obj: Union[Dict, Any], key: str, default=None):
    """Safely get attribute from object or dict - handles both Pydantic and dict objects"""
    if hasattr(obj, key):
        return getattr(obj, key)
    elif isinstance(obj, dict):
        return obj.get(key, default)
    else:
        return default


def get_workflow_pattern(agents_executed: list) -> str:
    """Determine workflow pattern based on agents executed"""
    if "knowledge_agent" in agents_executed:
        if len(agents_executed) > 4:
            return WorkflowPattern.COMPLEX_REASONING.value
        else:
            return WorkflowPattern.RESEARCH_ENHANCED.value
    else:
        return WorkflowPattern.SIMPLE_MEMORY_ONLY.value


def determine_next_agent_from_router(state: Dict[str, Any]) -> str:
    """Determine next agent from router based on complexity and research needs"""
    if state.get("should_research", False) or state.get("complexity_score", 0.0) > 0.5:
        return "knowledge_agent"
    elif len(state.get("messages", [])) > 0:
        return "memory_reader"
    else:
        return "memory_reader"


def should_include_knowledge_agent(state: Dict[str, Any]) -> str:
    """Determine if knowledge agent should be included in workflow"""
    if state.get("should_research", False):
        return "knowledge_agent"
    else:
        return "memory_reader"


def check_workflow_completion(state: Dict[str, Any]) -> str:
    """Check if workflow is complete and ready for memory writing"""
    if state.get("final_response"):
        return "memory_writer"
    else:
        return "organizer"


class WorkflowMetrics:
    """Centralized workflow metrics tracking"""
    
    def __init__(self):
        self.metrics = {
            "total_requests": 0,
            "average_processing_time": 0.0,
            "agent_executions": {
                "router": 0,
                "memory_reader": 0,
                "memory_writer": 0,
                "knowledge_agent": 0,
                "organizer": 0
            },
            "workflow_patterns": {
                "simple_memory_only": 0,
                "research_enhanced": 0,
                "complex_reasoning": 0
            }
        }
    
    def update_processing_time(self, processing_time: float):
        """Update average processing time"""
        total_requests = self.metrics["total_requests"]
        current_avg = self.metrics["average_processing_time"]
        self.metrics["average_processing_time"] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )
    
    def increment_agent_execution(self, agent_name: str):
        """Increment agent execution count"""
        if agent_name in self.metrics["agent_executions"]:
            self.metrics["agent_executions"][agent_name] += 1
    
    def increment_workflow_pattern(self, pattern: str):
        """Increment workflow pattern count"""
        if pattern in self.metrics["workflow_patterns"]:
            self.metrics["workflow_patterns"][pattern] += 1
    
    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive metrics status"""
        return {
            "total_workflows_executed": self.metrics["total_requests"],
            "average_processing_time_seconds": self.metrics["average_processing_time"],
            "agent_utilization": self.metrics["agent_executions"],
            "workflow_distribution": self.metrics["workflow_patterns"],
            "system_efficiency": {
                "requests_per_minute": 0,  # Would need time tracking
                "agent_success_rate": 0.95  # Would need error tracking
            }
        }
"""
Core module exports for Autonomous AI Agent

This module provides the essential components for the autonomous agent system:
- Agent state management and data structures
- Configuration management
- Event-driven architecture
- Expert team interface for multi-agent collaboration
- Performance optimization layer
"""

# Core agent state and entities
from .agent_state import (
    AgentState, 
    AgentMode, 
    ThoughtType,
    Thought, 
    Decision, 
    Goal, 
    create_initial_state
)

# Configuration management
from .config import AssistantConfig

# Event-driven architecture
from .event_bus import (
    Event,
    EventBus,
    EventTypes,
    publish_event, 
    subscribe_to_event,
    global_event_bus
)

# Expert team interface (removed - replaced by AutoGen GroupChat)

# Performance optimization (moved to legacy backup)

# Model routing (legacy - use performance layer instead)
from .models import QueryContext, ModelResponse

__all__ = [
    # Agent state
    "AgentState", "AgentMode", "ThoughtType", "Thought", "Decision", "Goal", "create_initial_state",
    
    # Configuration
    "AssistantConfig",
    
    # Event system
    "Event", "EventBus", "EventTypes", "publish_event", "subscribe_to_event", "global_event_bus",
    
    # Expert team (removed - replaced by AutoGen GroupChat)
    
    # Performance (moved to legacy backup)
    
    # Models (legacy)
    "QueryContext", "ModelResponse"
]
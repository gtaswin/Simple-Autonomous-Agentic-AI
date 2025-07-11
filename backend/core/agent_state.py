from typing import Dict, List, Any, Optional, TypedDict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import asyncio


class AgentMode(str, Enum):
    IDLE = "idle"
    THINKING = "thinking"
    ACTING = "acting"
    COLLABORATING = "collaborating"
    LEARNING = "learning"


class ThoughtType(str, Enum):
    OBSERVATION = "observation"
    REFLECTION = "reflection"
    PLANNING = "planning"
    INSIGHT = "insight"
    DECISION = "decision"


@dataclass
class Thought:
    id: str
    agent_name: str
    content: str
    thought_type: ThoughtType
    timestamp: datetime
    importance: float = 0.5
    tags: List[str] = field(default_factory=list)
    associations: List[str] = field(default_factory=list)
    confidence: float = 0.8
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "agent_name": self.agent_name,
            "content": self.content,
            "thought_type": self.thought_type.value,
            "timestamp": self.timestamp.isoformat(),
            "importance": self.importance,
            "tags": self.tags,
            "associations": self.associations,
            "confidence": self.confidence
        }


@dataclass
class Decision:
    id: str
    action: str
    reasoning: str
    confidence: float
    reversible: bool
    timestamp: datetime
    agent_name: str
    category: str
    executed: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "action": self.action,
            "reasoning": self.reasoning,
            "confidence": self.confidence,
            "reversible": self.reversible,
            "timestamp": self.timestamp.isoformat(),
            "agent_name": self.agent_name,
            "category": self.category,
            "executed": self.executed
        }


@dataclass
class Goal:
    id: str
    title: str
    description: str
    priority: int
    status: str = "active"
    progress: float = 0.0
    deadline: Optional[datetime] = None
    subtasks: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority,
            "status": self.status,
            "progress": self.progress,
            "deadline": self.deadline.isoformat() if self.deadline else None,
            "subtasks": self.subtasks,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class AgentState(TypedDict):
    """LangGraph state for autonomous agent"""
    
    # Core state
    mode: AgentMode
    current_focus: str
    cognitive_load: float
    
    # Memory and context
    recent_thoughts: List[Thought]
    active_goals: List[Goal]
    user_context: Dict[str, Any]
    
    # Decision making
    pending_decisions: List[Decision]
    recent_decisions: List[Decision]
    
    # Collaboration
    agent_communications: List[Dict[str, Any]]
    
    # Background processes
    background_tasks: List[str]
    
    # Metrics
    thoughts_today: int
    decisions_today: int
    patterns_found: int
    cost_today: float
    
    # User interaction
    last_user_message: Optional[str]
    last_user_timestamp: Optional[datetime]
    proactive_suggestions: List[str]
    conversation_history: List[Dict[str, str]]  # Track recent conversation
    session_id: str  # Unique session identifier
    
    # Learning
    insights: List[Dict[str, Any]]
    preference_updates: List[Dict[str, Any]]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert state to dictionary for serialization"""
        return {
            "mode": self.mode.value if isinstance(self.mode, AgentMode) else self.mode,
            "current_focus": self.current_focus,
            "cognitive_load": self.cognitive_load,
            "recent_thoughts": [t.to_dict() if hasattr(t, 'to_dict') else t for t in self.recent_thoughts],
            "active_goals": [g.to_dict() if hasattr(g, 'to_dict') else g for g in self.active_goals],
            "user_context": self.user_context,
            "pending_decisions": [d.to_dict() if hasattr(d, 'to_dict') else d for d in self.pending_decisions],
            "recent_decisions": [d.to_dict() if hasattr(d, 'to_dict') else d for d in self.recent_decisions],
            "agent_communications": self.agent_communications,
            "background_tasks": self.background_tasks,
            "thoughts_today": self.thoughts_today,
            "decisions_today": self.decisions_today,
            "patterns_found": self.patterns_found,
            "cost_today": self.cost_today,
            "last_user_message": self.last_user_message,
            "last_user_timestamp": self.last_user_timestamp.isoformat() if self.last_user_timestamp else None,
            "proactive_suggestions": self.proactive_suggestions,
            "conversation_history": self.conversation_history,
            "session_id": self.session_id,
            "insights": self.insights,
            "preference_updates": self.preference_updates
        }


def create_initial_state(user_id: str) -> AgentState:
    """Create initial agent state for a user"""
    return AgentState(
        mode=AgentMode.IDLE,
        current_focus="initialization",
        cognitive_load=0.1,
        recent_thoughts=[],
        active_goals=[],
        user_context={"user_id": user_id},
        pending_decisions=[],
        recent_decisions=[],
        agent_communications=[],
        background_tasks=[],
        thoughts_today=0,
        decisions_today=0,
        patterns_found=0,
        cost_today=0.0,
        last_user_message=None,
        last_user_timestamp=None,
        proactive_suggestions=[],
        conversation_history=[],
        session_id=f"session_{user_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        insights=[],
        preference_updates=[]
    )
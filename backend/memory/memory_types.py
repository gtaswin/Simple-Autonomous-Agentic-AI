"""
Memory Types and Data Classes

Shared types used across the memory system to avoid circular imports.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum


class MemoryType(str, Enum):
    """5-layer memory architecture for autonomous agents (as per README)"""
    WORKING = "working"         # Redis: Recent context, 7-item limit, activity-based TTL
    EPISODIC = "episodic"      # Qdrant: Personal experiences, conversations, events
    SEMANTIC = "semantic"      # Qdrant: Facts, knowledge, insights, preferences
    PROCEDURAL = "procedural"  # Qdrant: Skills, how-to, decision patterns
    PROSPECTIVE = "prospective" # Qdrant: Goals, plans, future intentions


class MemoryPriority(str, Enum):
    """Priority levels for memory storage and retrieval"""
    CRITICAL = "critical"      # Must be remembered (personal info, critical decisions)
    HIGH = "high"             # Important (goals, preferences, key insights)
    MEDIUM = "medium"         # Moderately important (useful patterns, context)
    LOW = "low"               # Background information (general knowledge)
    EPHEMERAL = "ephemeral"   # Temporary information


class ConsolidationState(str, Enum):
    """Memory consolidation lifecycle"""
    FRESH = "fresh"
    CONSOLIDATING = "consolidating"
    CONSOLIDATED = "consolidated"
    REINFORCED = "reinforced"
    DECAYING = "decaying"
    ARCHIVED = "archived"


@dataclass
class MemoryTrace:
    """
    Individual memory trace - the fundamental unit of memory storage.
    
    Represents a single piece of information stored in the agent's memory,
    whether it's a conversation, fact, skill, goal, or insight.
    """
    id: str
    content: str
    memory_type: MemoryType
    priority: MemoryPriority
    user_id: str
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    consolidation_state: ConsolidationState = ConsolidationState.FRESH
    embedding: Optional[List[float]] = None
    tags: Set[str] = field(default_factory=set)
    entities: Set[str] = field(default_factory=set)
    concepts: Set[str] = field(default_factory=set)
    emotions: Dict[str, float] = field(default_factory=dict)
    related_memories: Set[str] = field(default_factory=set)
    causal_links: Dict[str, str] = field(default_factory=dict)
    temporal_links: List[str] = field(default_factory=list)
    importance_score: float = 0.5
    novelty_score: float = 0.5
    certainty_score: float = 0.8
    source_reliability: float = 0.8
    rehearsal_count: int = 0
    interference_resistance: float = 0.5
    retrieval_strength: float = 1.0
    storage_strength: float = 1.0
    
    # Legacy compatibility fields
    emotional_valence: float = 0.0
    associations: List[str] = field(default_factory=list)
    source: str = "user_interaction"
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure tags is a set and handle mutable defaults"""
        if isinstance(self.tags, list):
            self.tags = set(self.tags)
        if self.tags is None:
            self.tags = set()
        if self.associations is None:
            self.associations = []
        if self.context is None:
            self.context = {}
    
    def update_access(self):
        """Update access tracking"""
        self.last_accessed = datetime.now()
        self.access_count += 1
        self.retrieval_strength = min(1.0, self.retrieval_strength + 0.1)
    
    def decay(self, decay_rate: float = 0.01):
        """Apply memory decay"""
        if self.priority != MemoryPriority.CRITICAL:
            self.retrieval_strength = max(0.1, self.retrieval_strength - decay_rate)
            if self.retrieval_strength < 0.3 and self.consolidation_state != ConsolidationState.ARCHIVED:
                self.consolidation_state = ConsolidationState.DECAYING
    
    def is_expired(self, ttl_hours: int = 168) -> bool:
        """Check if memory has expired (default 7 days)"""
        if self.priority == MemoryPriority.CRITICAL:
            return False  # Critical memories never expire
        
        expiry_time = self.last_accessed + timedelta(hours=ttl_hours)
        return datetime.now() > expiry_time
    
    def get_age_hours(self) -> float:
        """Get age of memory in hours"""
        return (datetime.now() - self.created_at).total_seconds() / 3600
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "priority": self.priority.value,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "importance_score": self.importance_score,
            "novelty_score": self.novelty_score,
            "emotional_valence": self.emotional_valence,
            "tags": list(self.tags),
            "associations": self.associations,
            "source": self.source,
            "context": self.context,
            "embedding": self.embedding
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MemoryTrace":
        """Create from dictionary"""
        return cls(
            id=data["id"],
            content=data["content"],
            memory_type=MemoryType(data["memory_type"]),
            priority=MemoryPriority(data["priority"]),
            user_id=data["user_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data.get("access_count", 0),
            importance_score=data.get("importance_score", 0.5),
            novelty_score=data.get("novelty_score", 0.5),
            emotional_valence=data.get("emotional_valence", 0.0),
            tags=set(data.get("tags", [])),
            associations=data.get("associations", []),
            source=data.get("source", "user_interaction"),
            context=data.get("context", {}),
            embedding=data.get("embedding")
        )


@dataclass 
class MemoryContext:
    """Context information for memory operations"""
    user_id: str
    session_id: Optional[str] = None
    conversation_id: Optional[str] = None
    interaction_type: str = "chat"  # Added back for compatibility
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "user_interaction"
    metadata: Dict[str, Any] = field(default_factory=dict)
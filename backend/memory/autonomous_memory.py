"""
Clean 3-Tier Memory System
Simple, focused implementation without bloat

Architecture:
- SESSION Memory: Chat history (Redis, no TTL, 50 conversations)
- WORKING Memory: Agent context (Redis, 7 items per agent per user)  
- LONG_TERM Memory: Important facts (Qdrant, importance-based)
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import uuid

# Import serialization utilities
from utils.serialization import safe_json_dumps

# Import layers after defining types to avoid circular imports

logger = logging.getLogger(__name__)


# =================================
# MEMORY TYPES AND DATA CLASSES
# =================================

class MemoryType(str, Enum):
    """5-tier memory architecture for autonomous agents"""
    SESSION = "session"        # Redis: Chat history, no TTL, 50 conversations max
    WORKING = "working"        # Redis: Agent context, 7 items per agent per user
    SHORT_TERM = "short_term"  # Redis Vector: User-specific temporary facts with TTL
    LONG_TERM_PERSONAL = "long_term_personal"    # Qdrant: User-specific important facts
    LONG_TERM_SHARED = "long_term_shared"        # Qdrant: Anonymized patterns, no user_name
    GENERAL_KNOWLEDGE = "general_knowledge"      # Qdrant: Universal facts, no user_name


class MemoryPriority(str, Enum):
    """Priority levels for memory storage and retrieval"""
    CRITICAL = "critical"      # Must be remembered (personal info, critical decisions)
    HIGH = "high"             # Important (goals, preferences, key insights)
    MEDIUM = "medium"         # Moderately important (useful patterns, context)
    LOW = "low"               # Background information (general knowledge)
    EPHEMERAL = "ephemeral"   # Temporary information

class MemoryScope(str, Enum):
    """Scope of memory - determines user access and privacy"""
    PERSONAL = "personal"      # User-specific private data (requires user_name)
    SHARED_PATTERN = "shared_pattern"  # Anonymized patterns (no user_name, for learning)
    UNIVERSAL = "universal"    # General knowledge (no user_name, public facts)


@dataclass
class MemoryTrace:
    """
    Enhanced memory trace for 5-tier hybrid system
    Supports both user-specific and shared memory storage
    """
    id: str
    content: str
    memory_type: MemoryType
    priority: MemoryPriority
    user_name: Optional[str]  # None for shared patterns and general knowledge
    created_at: datetime
    last_accessed: datetime
    memory_scope: MemoryScope = MemoryScope.PERSONAL  # Determines access pattern
    
    # Essential metadata
    access_count: int = 0
    importance_score: float = 0.5
    embedding: Optional[List[float]] = None
    tags: Set[str] = field(default_factory=set)
    
    # Compatibility fields for Qdrant storage
    concepts: Set[str] = field(default_factory=set)
    entities: Set[str] = field(default_factory=set)
    
    # Legacy compatibility fields (minimal)
    context: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Ensure tags, concepts, and entities are sets"""
        if isinstance(self.tags, list):
            self.tags = set(self.tags)
        if self.tags is None:
            self.tags = set()
            
        if isinstance(self.concepts, list):
            self.concepts = set(self.concepts)
        if self.concepts is None:
            self.concepts = set()
            
        if isinstance(self.entities, list):
            self.entities = set(self.entities)
        if self.entities is None:
            self.entities = set()
        if self.context is None:
            self.context = {}
    
    def update_access(self):
        """Update access tracking"""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            "id": self.id,
            "content": self.content,
            "memory_type": self.memory_type.value,
            "priority": self.priority.value,
            "user_name": self.user_name,
            "created_at": self.created_at.isoformat(),
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "importance_score": self.importance_score,
            "tags": list(self.tags),
            "concepts": list(self.concepts),
            "entities": list(self.entities),
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
            user_name=data["user_name"],
            created_at=datetime.fromisoformat(data["created_at"]),
            last_accessed=datetime.fromisoformat(data["last_accessed"]),
            access_count=data.get("access_count", 0),
            importance_score=data.get("importance_score", 0.5),
            tags=set(data.get("tags", [])),
            concepts=set(data.get("concepts", [])),
            entities=set(data.get("entities", [])),
            context=data.get("context", {}),
            embedding=data.get("embedding")
        )


class AutonomousMemorySystem:
    """Simple 3-Tier Memory System - No Bloat, Just Core Functionality"""
    
    def __init__(self, config=None, transformers_service=None):
        self.config = config
        self.transformers_service = transformers_service  # For universal working memory compression
        self.redis_layer: Optional[Any] = None  # RedisMemoryLayer (SESSION + WORKING)
        self.redis_vector_layer: Optional[Any] = None  # RedisVectorMemoryLayer (SHORT_TERM with TTL)
        self.qdrant_layer: Optional[Any] = None  # QdrantMemoryLayer (LONG_TERM)
        self.started = False
        
        logger.info("🧠 Hybrid 3-Tier Memory System initialized")
    
    async def start(self):
        """Initialize memory system components"""
        if self.started:
            return
        
        try:
            # Import here to avoid circular imports
            from .redis_memory import RedisMemoryLayer
            from .redis_vector_memory import RedisVectorMemoryLayer
            from .qdrant_memory import QdrantMemoryLayer
            import redis
            
            # Initialize Redis layer (for SESSION and WORKING memory)
            self.redis_layer = RedisMemoryLayer(self.config)
            await self.redis_layer.connect()
            
            # Initialize Redis Vector layer (for SHORT_TERM memory with TTL)
            redis_client = redis.Redis(
                host=self.config.get('databases.redis.host', 'localhost'),
                port=self.config.get('databases.redis.port', 6379),
                db=self.config.get('databases.redis.db', 0),
                decode_responses=False  # Vector operations need bytes
            )
            
            self.redis_vector_layer = RedisVectorMemoryLayer(
                redis_client=redis_client,
                embedding_model=self.config.get('transformers.models.embedder', 'sentence-transformers/all-MiniLM-L6-v2'),
                vector_dimensions=self.config.get('redis_vector.vector_dimensions', 384),
                similarity_threshold=self.config.get('redis_vector.similarity_threshold', 0.7),
                default_ttl=self.config.get_memory_ttl("short_term", "default"),
                index_name=self.config.get('redis_vector.short_term_index', 'shortterm_memory_idx'),
                config=self.config  # Pass config for unified TTL system
            )
            await self.redis_vector_layer.connect()
            
            # Initialize Qdrant layer (for LONG_TERM memory) 
            self.qdrant_layer = QdrantMemoryLayer(self.config)
            await self.qdrant_layer.connect()
            
            self.started = True
            logger.info("✅ Hybrid 3-Tier Memory System started successfully")
            
        except Exception as e:
            logger.error(f"❌ Memory system startup failed: {e}")
            raise
    
    async def stop(self):
        """Cleanup memory system"""
        try:
            if self.redis_layer:
                await self.redis_layer.disconnect()
            if self.redis_vector_layer:
                # Redis vector layer cleanup (connection will be handled by redis_layer)
                pass
            if self.qdrant_layer:
                await self.qdrant_layer.cleanup()
            self.started = False
            logger.info("🛑 Memory system stopped")
        except Exception as e:
            logger.error(f"Error stopping memory system: {e}")
    
    # =================================
    # SESSION MEMORY (Chat History)
    # =================================
    
    async def store_session_memory(self, user_name: str, user_message: str, ai_response: str, metadata: Dict[str, Any] = None) -> str:
        """Store chat conversation in session memory"""
        if not self.redis_layer:
            raise RuntimeError("Redis layer not initialized")
        
        session_key = f"session_memory:{user_name}"
        conversation_data = {
            "user_message": user_message,
            "ai_response": ai_response,
            "timestamp": datetime.now().isoformat(),
            "metadata": metadata or {}
        }
        
        return await self.redis_layer.store_session_conversation(
            key=session_key,
            conversation=conversation_data,
            max_conversations=50
        )
    
    async def get_session_memory(self, user_name: str, limit: int = 50) -> List[Dict]:
        """Get chat history from session memory"""
        if not self.redis_layer:
            return []
        
        session_key = f"session_memory:{user_name}"
        return await self.redis_layer.get_session_conversations(session_key, limit=limit)
    
    async def clear_session_memory(self, user_name: str) -> bool:
        """Clear all session memory for user"""
        if not self.redis_layer:
            return False
        
        session_key = f"session_memory:{user_name}"
        return await self.redis_layer.clear_session_memory(session_key)
    
    # =================================
    # WORKING MEMORY (Agent Context)
    # =================================
    
    async def get_working_memory(self, user_name: str, agent_name: str, limit: int = 7) -> List[Dict]:
        """Get working memory for specific agent and user"""
        if not self.redis_layer:
            return []
        
        key = f"working_memory:{user_name}:{agent_name}"
        return await self.redis_layer.get_working_memory_by_key(key, limit=limit)
    
    async def store_working_memory(self, user_name: str, agent_name: str, content: str, metadata: Dict[str, Any] = None) -> str:
        """Store item in agent's working memory with universal compression"""
        if not self.redis_layer:
            raise RuntimeError("Redis layer not initialized")
        
        # Apply universal compression if content is large
        compressed_content = await self._compress_working_memory_if_needed(content, agent_name)
        
        key = f"working_memory:{user_name}:{agent_name}"
        return await self.redis_layer.store_working_memory_by_key(
            key=key,
            content=compressed_content,
            metadata=metadata or {}
        )
    
    async def clear_working_memory(self, user_name: str, agent_name: str) -> bool:
        """Clear working memory for specific agent and user"""
        if not self.redis_layer:
            return False
        
        key = f"working_memory:{user_name}:{agent_name}"
        return await self.redis_layer.clear_working_memory_by_key(key)
    
    # =================================
    # SHORT_TERM MEMORY (Redis Vector with TTL)
    # =================================
    
    async def _handle_deduplication_before_storage(
        self, 
        user_name: str, 
        content: str, 
        importance_score: float
    ) -> Dict[str, Any]:
        """Common deduplication logic for all storage methods"""
        duplicate_check = await self.check_for_duplicate_fact(
            user_name=user_name,
            content=content,
            importance_score=importance_score
        )
        
        if duplicate_check["is_duplicate"]:
            logger.info(f"Duplicate fact detected (similarity: {duplicate_check['similarity_score']:.3f}) in {duplicate_check['memory_tier']}")
            
            # Update existing memory instead of storing new
            update_success = await self.update_existing_memory(
                user_name=user_name,
                existing_memory=duplicate_check["existing_memory"],
                new_content=content,
                new_importance=importance_score,
                memory_tier=duplicate_check["memory_tier"]
            )
            
            if update_success:
                logger.info(f"Updated existing memory instead of storing duplicate")
                return {
                    "should_store": False,
                    "memory_id": duplicate_check["existing_memory"].get("memory_id") or duplicate_check["existing_memory"].get("id", "")
                }
            else:
                logger.warning("Failed to update existing memory, proceeding with storage")
        
        return {"should_store": True, "memory_id": None}
    
    async def store_short_term_memory(
        self, 
        user_name: str, 
        content: str, 
        importance_score: float = 0.5,
        ttl: int = None,  # Use config-based TTL
        metadata: Dict[str, Any] = None
    ) -> str:
        """Store content in short-term memory with automatic TTL expiration and deduplication"""
        if not self.redis_vector_layer:
            raise RuntimeError("Redis Vector layer not initialized")
        
        # Handle deduplication
        dedup_result = await self._handle_deduplication_before_storage(user_name, content, importance_score)
        if not dedup_result["should_store"]:
            return dedup_result["memory_id"]
        
        # Use provided TTL (Memory Writer Agent should always provide TTL)  
        ttl = ttl or self.config.get_memory_ttl("short_term", "default")
        
        return await self.redis_vector_layer.store_short_term_memory(
            user_name=user_name,
            content=content,
            importance_score=importance_score,
            ttl=ttl,
            metadata=metadata or {}
        )
    
    async def search_short_term_memory(
        self, 
        user_name: str, 
        query: str, 
        limit: int = None,
        min_importance: float = 0.0
    ) -> List[Dict[str, Any]]:
        """Search short-term memory using vector similarity"""
        if not self.redis_vector_layer:
            return []
        
        # Use config-based limit if not specified
        actual_limit = limit or self.config.get_memory_limit("search_results")
        
        return await self.redis_vector_layer.search_short_term_memory(
            user_name=user_name,
            query=query,
            limit=actual_limit,
            min_importance=min_importance
        )
    
    async def get_user_short_term_memories(
        self, 
        user_name: str, 
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """Get all short-term memories for a user"""
        if not self.redis_vector_layer:
            return []
        
        return await self.redis_vector_layer.get_user_short_term_memories(
            user_name=user_name,
            limit=limit
        )
    
    async def cleanup_user_short_term_memory(self, user_name: str) -> int:
        """Clean up all short-term memories for a user"""
        if not self.redis_vector_layer:
            return 0
        
        return await self.redis_vector_layer.cleanup_user_short_term_memory(user_name)
    
    # =================================
    # LONG_TERM MEMORY (Important Facts)
    # =================================
    
    async def store_long_term(self, user_name: str, content: str, importance_score: float = 0.5) -> str:
        """Store important facts in long-term memory with deduplication"""
        if not self.qdrant_layer:
            raise RuntimeError("Qdrant layer not initialized")
        
        # Handle deduplication
        dedup_result = await self._handle_deduplication_before_storage(user_name, content, importance_score)
        if not dedup_result["should_store"]:
            return dedup_result["memory_id"]
        
        # Create memory trace
        memory_trace = MemoryTrace(
            id=str(uuid.uuid4()),
            content=content,
            memory_type=MemoryType.LONG_TERM_PERSONAL,
            priority=self._importance_to_priority(importance_score),
            user_name=user_name,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            importance_score=importance_score,
            context={"importance_score": importance_score, "source": "organizer_agent"}
        )
        
        # Store in Qdrant
        success = await self.qdrant_layer.store_memory(memory_trace)
        return memory_trace.id if success else ""
    
    async def get_long_term_memories(self, user_name: str, query: str, limit: int = 10, min_importance: float = 0.3) -> List[Dict]:
        """Get important facts from long-term memory"""
        if not self.qdrant_layer:
            return []
        
        # Search by query and user_name
        memories = await self.qdrant_layer.search_memories(
            query=query,
            user_name=user_name,
            memory_types=[MemoryType.LONG_TERM_PERSONAL],
            limit=limit
        )
        
        # Filter by importance if specified
        if min_importance > 0:
            filtered_memories = []
            for memory in memories:
                if hasattr(memory, 'importance_score') and memory.importance_score >= min_importance:
                    filtered_memories.append({
                        "content": memory.content,
                        "importance_score": memory.importance_score,
                        "created_at": memory.created_at.isoformat() if hasattr(memory, 'created_at') else "",
                        "context": getattr(memory, 'context', {})
                    })
                elif isinstance(memory, dict) and memory.get("importance_score", 0) >= min_importance:
                    filtered_memories.append(memory)
            return filtered_memories
        
        # Return all memories if no importance filter
        result = []
        for memory in memories:
            if hasattr(memory, 'content'):
                result.append({
                    "content": memory.content,
                    "importance_score": getattr(memory, 'importance_score', 0.5),
                    "created_at": memory.created_at.isoformat() if hasattr(memory, 'created_at') else "",
                    "context": getattr(memory, 'context', {})
                })
            elif isinstance(memory, dict):
                result.append(memory)
        
        return result
    
    async def search_long_term_memory(self, query: str, user_name: str = "Aswin", limit: int = 10) -> List[Dict]:
        """Search long-term memory - alias for get_long_term_memories for agent compatibility"""
        return await self.get_long_term_memories(
            user_name=user_name,
            query=query,
            limit=limit,
            min_importance=0.3
        )
    
    # =================================
    # MEMORY DEDUPLICATION SYSTEM
    # =================================
    
    async def check_for_duplicate_fact(
        self, 
        user_name: str, 
        content: str, 
        importance_score: float = 0.5,
        similarity_threshold: float = None
    ) -> Dict[str, Any]:
        """
        Check if similar fact already exists across memory tiers.
        
        Returns:
            Dict with 'is_duplicate', 'existing_memory', 'similarity_score', 'memory_tier'
        """
        if not self.config or not self.config.get("memory_control.deduplication.enabled", True):
            return {"is_duplicate": False, "existing_memory": None, "similarity_score": 0.0, "memory_tier": None}
        
        # Get similarity threshold from config or use provided
        if similarity_threshold is None:
            similarity_threshold = self.config.get("memory_control.deduplication.default_threshold", 0.85)
        
        try:
            # Check cross-tier deduplication if enabled
            cross_tier_enabled = self.config.get("memory_control.deduplication.cross_tier_deduplication", True)
            
            highest_similarity = 0.0
            duplicate_memory = None
            memory_tier = None
            
            # 1. Check short-term memory first (most recent)
            if self.redis_vector_layer:
                short_term_results = await self.search_short_term_memory(
                    user_name=user_name,
                    query=content,
                    limit=5,
                    min_importance=0.0  # Check all importance levels
                )
                
                for memory in short_term_results:
                    # Calculate similarity using embeddings if available
                    similarity = await self._calculate_content_similarity(content, memory.get("content", ""))
                    
                    if similarity > highest_similarity and similarity >= similarity_threshold:
                        highest_similarity = similarity
                        duplicate_memory = memory
                        memory_tier = "short_term"
            
            # 2. Check long-term memory if cross-tier enabled or no short-term duplicate
            if cross_tier_enabled and self.qdrant_layer and (duplicate_memory is None or highest_similarity < 0.95):
                long_term_results = await self.get_long_term_memories(
                    user_name=user_name,
                    query=content,
                    limit=5,
                    min_importance=0.0
                )
                
                for memory in long_term_results:
                    similarity = await self._calculate_content_similarity(content, memory.get("content", ""))
                    
                    if similarity > highest_similarity and similarity >= similarity_threshold:
                        highest_similarity = similarity
                        duplicate_memory = memory
                        memory_tier = "long_term"
            
            return {
                "is_duplicate": duplicate_memory is not None,
                "existing_memory": duplicate_memory,
                "similarity_score": highest_similarity,
                "memory_tier": memory_tier
            }
            
        except Exception as e:
            logger.error(f"Error checking for duplicate fact: {e}")
            return {"is_duplicate": False, "existing_memory": None, "similarity_score": 0.0, "memory_tier": None}
    
    async def _calculate_content_similarity(self, content1: str, content2: str) -> float:
        """Calculate semantic similarity between two content strings using embeddings"""
        try:
            if not content1 or not content2 or content1.strip() == content2.strip():
                return 1.0 if content1.strip() == content2.strip() else 0.0
            
            # Use transformers service for embedding generation if available
            if self.transformers_service:
                embeddings = await self.transformers_service.generate_embeddings([content1, content2])
                if embeddings and len(embeddings) == 2:
                    return self._cosine_similarity(embeddings[0], embeddings[1])
            
            # Fallback to Redis Vector layer embedding model
            if self.redis_vector_layer and hasattr(self.redis_vector_layer, 'embedding_model'):
                import numpy as np
                embedding1 = self.redis_vector_layer.embedding_model.encode([content1])[0]
                embedding2 = self.redis_vector_layer.embedding_model.encode([content2])[0]
                return self._cosine_similarity(embedding1.tolist(), embedding2.tolist())
            
            # Final fallback - simple text similarity
            return self._simple_text_similarity(content1, content2)
            
        except Exception as e:
            logger.warning(f"Error calculating similarity: {e}")
            return self._simple_text_similarity(content1, content2)
    
    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        try:
            import numpy as np
            
            # Convert to numpy arrays
            v1 = np.array(vec1)
            v2 = np.array(vec2)
            
            # Calculate cosine similarity
            dot_product = np.dot(v1, v2)
            norms = np.linalg.norm(v1) * np.linalg.norm(v2)
            
            if norms == 0:
                return 0.0
            
            similarity = dot_product / norms
            return float(similarity)
            
        except Exception as e:
            logger.warning(f"Error in cosine similarity calculation: {e}")
            return 0.0
    
    def _simple_text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity fallback using character overlap"""
        try:
            # Normalize texts
            t1 = text1.lower().strip()
            t2 = text2.lower().strip()
            
            if t1 == t2:
                return 1.0
            
            # Calculate Jaccard similarity using character n-grams
            def get_ngrams(text: str, n: int = 3) -> set:
                return set([text[i:i+n] for i in range(len(text) - n + 1)])
            
            ngrams1 = get_ngrams(t1)
            ngrams2 = get_ngrams(t2)
            
            if not ngrams1 and not ngrams2:
                return 1.0
            if not ngrams1 or not ngrams2:
                return 0.0
            
            intersection = ngrams1.intersection(ngrams2)
            union = ngrams1.union(ngrams2)
            
            return len(intersection) / len(union) if union else 0.0
            
        except Exception:
            return 0.0
    
    async def update_existing_memory(
        self, 
        user_name: str, 
        existing_memory: Dict[str, Any], 
        new_content: str, 
        new_importance: float,
        memory_tier: str
    ) -> bool:
        """Update existing memory with new information and merge metadata"""
        try:
            update_config = self.config.get("memory_control.deduplication.update_on_duplicate", {})
            if not update_config.get("enabled", True):
                return False
            
            # Determine final importance score
            merge_importance = update_config.get("merge_importance", "max")
            existing_importance = existing_memory.get("importance_score", 0.5)
            
            if merge_importance == "max":
                final_importance = max(new_importance, existing_importance)
            elif merge_importance == "avg":
                final_importance = (new_importance + existing_importance) / 2
            else:  # "latest"
                final_importance = new_importance
            
            # Update based on memory tier
            if memory_tier == "short_term" and self.redis_vector_layer:
                # Update short-term memory with proper TTL calculation
                memory_id = existing_memory.get("memory_id") or existing_memory.get("id")
                if memory_id:
                    # For deduplication updates, use reasonable default TTL
                    # This should ideally be done by Memory Writer Agent, not here
                    calculated_ttl = self.config.get_memory_ttl("short_term", "default")
                    
                    try:
                        success = await self.redis_vector_layer.update_memory(
                            memory_id=memory_id,
                            user_name=user_name,
                            new_content=new_content,
                            new_importance=final_importance,
                            refresh_ttl=update_config.get("refresh_ttl", True),
                            new_ttl=calculated_ttl
                        )
                    except Exception as e:
                        logger.error(f"Redis vector layer update failed: {e}")
                        success = False
                    if success:
                        logger.info(f"Updated short-term memory {memory_id} with importance {final_importance}")
                        return True
            
            elif memory_tier == "long_term" and self.qdrant_layer:
                # For Qdrant, we need to delete and re-insert since it doesn't support updates
                memory_id = existing_memory.get("id")
                if memory_id:
                    # Create updated memory trace
                    memory_trace = MemoryTrace(
                        id=memory_id,
                        content=new_content,
                        memory_type=MemoryType.LONG_TERM_PERSONAL,
                        priority=self._importance_to_priority(final_importance),
                        user_name=user_name,
                        created_at=datetime.now(),
                        last_accessed=datetime.now(),
                        importance_score=final_importance,
                        context={"importance_score": final_importance, "source": "deduplication_update"}
                    )
                    
                    # Delete old and store new
                    await self.qdrant_layer.delete_memory(memory_id)
                    success = await self.qdrant_layer.store_memory(memory_trace)
                    if success:
                        logger.info(f"Updated long-term memory {memory_id} with importance {final_importance}")
                        return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error updating existing memory: {e}")
            return False
    
    # =============================================
    # UNIVERSAL WORKING MEMORY COMPRESSION
    # =============================================
    
    async def _compress_working_memory_if_needed(self, content: str, agent_name: str) -> str:
        """Apply intelligent compression for working memory items > 100 chars"""
        
        # Get compression settings from config
        compression_config = self.config.get("memory_control.working_memory_compression", {}) if self.config else {}
        enabled = compression_config.get("enabled", True)
        compression_threshold = compression_config.get("compression_threshold", 100)
        target_length = compression_config.get("target_length", 100)
        use_transformers = compression_config.get("use_transformers", True)
        
        # Skip compression if disabled
        if not enabled:
            return content
        
        # Skip compression if content is already short
        if len(content) <= compression_threshold:
            return content
        
        # Skip compression if transformers service not available or disabled
        if not use_transformers or not self.transformers_service:
            logger.debug(f"Transformers service not available/disabled, using truncation for {agent_name}")
            return self._fallback_truncation(content, target_length)
        
        try:
            # Use local transformers for intelligent summarization
            result = await self.transformers_service.summarize(
                text=content,
                max_length=25,  # ~25 words ≈ 100 chars
                task_type=f"working_memory_{agent_name}"
            )
            
            if result and isinstance(result, dict) and "summary" in result:
                compressed = result["summary"]
                
                # Ensure we stay within target length
                if len(compressed) > target_length:
                    compressed = compressed[:target_length-3] + "..."
                
                compression_ratio = len(content) / len(compressed) if len(compressed) > 0 else 1
                logger.debug(f"Compressed working memory for {agent_name}: {len(content)} → {len(compressed)} chars ({compression_ratio:.1f}x)")
                
                return compressed
            
        except Exception as e:
            logger.warning(f"Working memory compression failed for {agent_name}: {e}")
        
        # Fallback to intelligent truncation
        return self._fallback_truncation(content, target_length)
    
    def _fallback_truncation(self, content: str, target_length: int) -> str:
        """Intelligent truncation preserving key information"""
        if len(content) <= target_length:
            return content
        
        # Try to truncate at sentence boundary
        truncated = content[:target_length-3]
        last_period = truncated.rfind('.')
        last_space = truncated.rfind(' ')
        
        # Use sentence boundary if available and reasonable
        if last_period > target_length * 0.7:
            return content[:last_period+1]
        elif last_space > target_length * 0.8:
            return content[:last_space] + "..."
        else:
            return content[:target_length-3] + "..."
    
    # Legacy cleanup methods removed - Redis Stack handles TTL natively
    
    # =================================
    # USER CLEANUP SYSTEM  
    # =================================
    
    async def cleanup_user_memories(self, user_name: str) -> Dict[str, Any]:
        """Clean up session memory, working memory, and short-term memory for specific user"""
        results = {
            "session_cleared": False,
            "working_memory_cleared": {},
            "short_term_cleaned": 0,
            "long_term_preserved": True
        }
        
        # Clear session memory
        results["session_cleared"] = await self.clear_session_memory(user_name)
        
        # Clear working memory for all agents (4-agent system)
        agents = ["memory_reader", "memory_writer", "knowledge_agent", "organizer_agent"]
        for agent in agents:
            success = await self.clear_working_memory(user_name, agent)
            results["working_memory_cleared"][agent] = success
        
        # Clean up short-term memory (TTL will handle automatic expiration, but this provides immediate cleanup)
        results["short_term_cleaned"] = await self.cleanup_user_short_term_memory(user_name)
        
        logger.info(f"🧹 User cleanup completed for: {user_name}")
        return results
    
    # =================================
    # AUTONOMOUS INSIGHTS STORAGE
    # =================================
    
    async def store_autonomous_insight(
        self, 
        user_name: str, 
        insight_type: str, 
        insight_content: str, 
        metadata: Dict[str, Any] = None
    ) -> str:
        """
        Store latest autonomous insight, overwriting previous one of same type.
        
        Args:
            user_name: Target user name (e.g., "Aswin")
            insight_type: Type of insight ("pattern_discovery", "weekly_insights", "milestone_tracking", etc.)
            insight_content: The actual insight content
            metadata: Additional metadata about the insight
        
        Returns:
            Insight ID
        """
        if not self.redis_layer:
            raise RuntimeError("Redis layer not initialized")
        
        try:
            insight_id = f"{user_name}_{insight_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            insight_key = f"autonomous_insights:{user_name}:{insight_type}"
            
            insight_data = {
                "insight_id": insight_id,
                "user_name": user_name,
                "insight_type": insight_type,
                "content": insight_content,
                "created_at": datetime.now().isoformat(),
                "metadata": metadata or {}
            }
            
            # Store latest insight (overwrites previous)
            await self.redis_layer.redis.hset(
                insight_key,
                mapping={
                    "insight_id": insight_id,
                    "user_name": user_name,
                    "insight_type": insight_type,
                    "content": insight_content,
                    "created_at": insight_data["created_at"],
                    "metadata": safe_json_dumps(insight_data["metadata"])
                }
            )
            
            # Add to user's insight index
            user_insights_key = f"user_insights:{user_name}"
            await self.redis_layer.redis.sadd(user_insights_key, insight_type)
            
            logger.info(f"🧠 Stored autonomous insight: {insight_type} for {user_name}")
            return insight_id
            
        except Exception as e:
            logger.error(f"Failed to store autonomous insight: {e}")
            raise
    
    async def get_user_autonomous_insights(self, user_name: str) -> List[Dict[str, Any]]:
        """
        Get all latest autonomous insights for a user.
        
        Args:
            user_name: User name
            
        Returns:
            List of latest insights by type
        """
        if not self.redis_layer:
            return []
        
        try:
            # Get all insight types for user
            user_insights_key = f"user_insights:{user_name}"
            insight_types = await self.redis_layer.redis.smembers(user_insights_key)
            
            insights = []
            for insight_type in insight_types:
                insight_type_str = insight_type  # Already decoded due to decode_responses=True
                insight_key = f"autonomous_insights:{user_name}:{insight_type_str}"
                
                insight_data = await self.redis_layer.redis.hgetall(insight_key)
                if insight_data:
                    try:
                        import json
                        metadata = json.loads(insight_data.get('metadata', '{}'))
                    except:
                        metadata = {}
                    
                    insight = {
                        "insight_id": insight_data.get('insight_id', ''),
                        "user_name": insight_data.get('user_name', ''),
                        "insight_type": insight_data.get('insight_type', ''),
                        "content": insight_data.get('content', ''),
                        "created_at": insight_data.get('created_at', ''),
                        "metadata": metadata
                    }
                    insights.append(insight)
            
            # Sort by creation time (newest first)
            insights.sort(key=lambda x: x.get('created_at', ''), reverse=True)
            return insights
            
        except Exception as e:
            logger.error(f"Failed to get autonomous insights for {user_name}: {e}")
            return []
    
    async def get_autonomous_insight_by_type(self, user_name: str, insight_type: str) -> Optional[Dict[str, Any]]:
        """
        Get specific autonomous insight by type.
        
        Args:
            user_name: User name
            insight_type: Type of insight to retrieve
            
        Returns:
            Latest insight of the specified type or None
        """
        if not self.redis_layer:
            return None
        
        try:
            insight_key = f"autonomous_insights:{user_name}:{insight_type}"
            insight_data = await self.redis_layer.redis.hgetall(insight_key)
            
            if not insight_data:
                return None
            
            try:
                import json
                metadata = json.loads(insight_data.get('metadata', '{}'))
            except:
                metadata = {}
            
            return {
                "insight_id": insight_data.get('insight_id', ''),
                "user_name": insight_data.get('user_name', ''),
                "insight_type": insight_data.get('insight_type', ''),
                "content": insight_data.get('content', ''),
                "created_at": insight_data.get('created_at', ''),
                "metadata": metadata
            }
            
        except Exception as e:
            logger.error(f"Failed to get insight {insight_type} for {user_name}: {e}")
            return None
    
    async def clear_user_autonomous_insights(self, user_name: str) -> bool:
        """Clear all autonomous insights for a user."""
        if not self.redis_layer:
            return False
        
        try:
            # Get all insight types
            user_insights_key = f"user_insights:{user_name}"
            insight_types = await self.redis_layer.redis.smembers(user_insights_key)
            
            # Delete each insight
            for insight_type in insight_types:
                insight_type_str = insight_type.decode() if isinstance(insight_type, bytes) else insight_type
                insight_key = f"autonomous_insights:{user_name}:{insight_type_str}"
                await self.redis_layer.redis.delete(insight_key)
            
            # Clear the index
            await self.redis_layer.redis.delete(user_insights_key)
            
            logger.info(f"🧹 Cleared autonomous insights for: {user_name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to clear insights for {user_name}: {e}")
            return False
    
    # =================================
    # UTILITY METHODS
    # =================================
    
    def _importance_to_priority(self, importance_score: float) -> MemoryPriority:
        """Convert importance score to priority level"""
        if importance_score >= 0.8:
            return MemoryPriority.CRITICAL
        elif importance_score >= 0.6:
            return MemoryPriority.HIGH
        elif importance_score >= 0.4:
            return MemoryPriority.MEDIUM
        else:
            return MemoryPriority.LOW
    
    async def get_memory_summary(self) -> Dict[str, Any]:
        """Get basic memory statistics"""
        try:
            stats = {
                "system_type": "hybrid-3-tier",
                "redis_connected": self.redis_layer is not None and self.redis_layer.redis is not None,
                "redis_vector_connected": self.redis_vector_layer is not None,
                "qdrant_connected": self.qdrant_layer is not None,
                "started": self.started,
                "architecture": "SESSION (Redis) + WORKING (Redis) + SHORT_TERM (Redis Vector + TTL) + LONG_TERM (Qdrant)"
            }
            
            # Get short-term memory statistics if available
            if self.redis_vector_layer:
                vector_stats = await self.redis_vector_layer.get_memory_stats()
                stats["short_term_stats"] = vector_stats
            
            return stats
        except Exception as e:
            logger.error(f"Error getting memory summary: {e}")
            return {"error": str(e)}
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get system statistics"""
        return await self.get_memory_summary()
    
    # =================================
    # SYSTEM MAINTENANCE & HEALTH CHECK
    # =================================
    
    def health_check(self) -> Dict[str, Any]:
        """Simple health check"""
        return {
            "status": "healthy" if self.started else "stopped",
            "redis_layer": self.redis_layer is not None,
            "redis_vector_layer": self.redis_vector_layer is not None,
            "qdrant_layer": self.qdrant_layer is not None,
            "architecture": "hybrid-3-tier-with-ttl"
        }
    
    # startup_memory_reclaim method removed - unnecessary with TTL and auto-trimming
    # Redis TTL handles expiration automatically
    # Working memory auto-trims to 7 items per agent per user
    # Session memory auto-trims to 50 conversations per user
    # Long-term memory is permanent by design
    
    
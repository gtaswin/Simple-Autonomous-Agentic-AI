"""
Advanced Memory Architecture for Autonomous Agents

Implements a multi-layered memory system inspired by human cognitive architecture:
- Working Memory: Temporary, immediate processing
- Episodic Memory: Personal experiences and events
- Semantic Memory: Facts, concepts, knowledge
- Procedural Memory: Skills, habits, learned behaviors
- Prospective Memory: Future intentions and plans
"""

import asyncio
import uuid
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import hashlib
import numpy as np
from collections import defaultdict, deque

# Initialize logger
logger = logging.getLogger(__name__)

# Import shared types and memory layers
from .memory_types import MemoryTrace, MemoryType, MemoryPriority, MemoryContext, ConsolidationState
from .redis_memory import RedisMemoryLayer, WorkingMemoryItem
from .qdrant_memory import QdrantMemoryLayer, QdrantMemoryItem

try:
    import litellm
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

# Import TransformersService for local AI processing
try:
    from core.transformers_service import get_transformers_service
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False


@dataclass
class MemoryCluster:
    """A cluster of related memories forming a concept or episode"""
    id: str
    name: str
    memory_ids: Set[str]
    cluster_type: str
    created_at: datetime
    importance: float
    activation_level: float = 0.0

    def activate(self, strength: float = 0.1):
        self.activation_level = min(1.0, self.activation_level + strength)


class MemoryStorage:
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.memories: Dict[str, MemoryTrace] = {}
        self.clusters: Dict[str, MemoryCluster] = {}
        self.type_index: Dict[MemoryType, Set[str]] = defaultdict(set)
        self.tag_index: Dict[str, Set[str]] = defaultdict(set)
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)
        self.temporal_index: Dict[str, List[str]] = defaultdict(list)
        self.working_memory: deque = deque(maxlen=7)

    def store_memory(self, memory: MemoryTrace):
        self.memories[memory.id] = memory
        self._update_indexes(memory)
        if memory.memory_type == MemoryType.WORKING:
            self.working_memory.append(memory.id)

    def _update_indexes(self, memory: MemoryTrace):
        self.type_index[memory.memory_type].add(memory.id)
        for tag in memory.tags:
            self.tag_index[tag].add(memory.id)
        for entity in memory.entities:
            self.entity_index[entity].add(memory.id)
        date_key = memory.created_at.strftime("%Y-%m-%d")
        self.temporal_index[date_key].append(memory.id)


class MemoryRetriever:
    def __init__(self, storage: MemoryStorage, embedding_model):
        self.storage = storage
        self.embedding_model = embedding_model

    async def retrieve(self, query: str, memory_types: List[MemoryType] = None, limit: int = 10, min_relevance: float = 0.3) -> List[MemoryTrace]:
        if not memory_types:
            memory_types = [MemoryType.EPISODIC, MemoryType.SEMANTIC]
        
        query_embedding = self.embedding_model.encode(query).tolist() if self.embedding_model else None
        
        candidates = []
        for memory_type in memory_types:
            for memory_id in self.storage.type_index[memory_type]:
                memory = self.storage.memories[memory_id]
                relevance = await self._calculate_relevance(memory, query, query_embedding)
                if relevance >= min_relevance:
                    candidates.append((memory, relevance))
        
        candidates.sort(key=lambda x: (x[1], x[0].last_accessed), reverse=True)
        
        retrieved_memories = []
        for memory, _ in candidates[:limit]:
            memory.update_access()
            retrieved_memories.append(memory)
            await self._activate_related_memories(memory.id)
            
        return retrieved_memories

    async def _calculate_relevance(self, memory: MemoryTrace, query: str, query_embedding: List[float]) -> float:
        relevance = 0.0
        if memory.embedding and query_embedding:
            semantic_sim = np.dot(memory.embedding, query_embedding) / (np.linalg.norm(memory.embedding) * np.linalg.norm(query_embedding))
            relevance += semantic_sim * 0.4
        
        query_words = set(query.lower().split())
        memory_words = set(memory.content.lower().split())
        keyword_sim = len(query_words & memory_words) / len(query_words | memory_words)
        relevance += keyword_sim * 0.3
        
        days_old = (datetime.now() - memory.last_accessed).days
        recency_score = max(0, 1 - days_old / 30)
        relevance += recency_score * 0.2
        
        relevance += memory.importance_score * 0.1
        return min(1.0, relevance)

    async def _activate_related_memories(self, memory_id: str):
        memory = self.storage.memories.get(memory_id)
        if not memory:
            return
        
        for related_id in memory.related_memories:
            if related_memory := self.storage.memories.get(related_id):
                related_memory.retrieval_strength = min(1.0, related_memory.retrieval_strength + 0.1)


class MemoryConsolidator:
    def __init__(self, storage: MemoryStorage):
        self.storage = storage

    async def consolidate_memories(self):
        print("🧠 Starting memory consolidation...")
        fresh_memories = [m for m in self.storage.memories.values() if m.consolidation_state == ConsolidationState.FRESH]
        for memory in fresh_memories:
            await self._consolidate_single_memory(memory)
        await self._create_memory_clusters()
        print(f"✅ Consolidated {len(fresh_memories)} memories")

    async def _consolidate_single_memory(self, memory: MemoryTrace):
        memory.consolidation_state = ConsolidationState.CONSOLIDATING
        if memory.priority in [MemoryPriority.CRITICAL, MemoryPriority.HIGH]:
            memory.storage_strength = min(1.0, memory.storage_strength + 0.2)
            memory.interference_resistance = min(1.0, memory.interference_resistance + 0.1)
        memory.consolidation_state = ConsolidationState.CONSOLIDATED
        memory.rehearsal_count += 1

    async def _create_memory_clusters(self):
        entity_groups = defaultdict(set)
        concept_groups = defaultdict(set)
        for memory_id, memory in self.storage.memories.items():
            for entity in memory.entities:
                entity_groups[entity].add(memory_id)
            for concept in memory.concepts:
                concept_groups[concept].add(memory_id)
        
        for entity, memory_ids in entity_groups.items():
            if len(memory_ids) >= 3:
                cluster_id = f"entity_{entity}_{uuid.uuid4().hex[:8]}"
                self.storage.clusters[cluster_id] = MemoryCluster(
                    id=cluster_id, name=f"Memories about {entity}", memory_ids=memory_ids,
                    cluster_type="entity", created_at=datetime.now(),
                    importance=sum(self.storage.memories[mid].importance_score for mid in memory_ids) / len(memory_ids)
                )


class PatternAnalyzer:
    def __init__(self, storage: MemoryStorage):
        self.storage = storage
        self.learning_patterns: Dict[str, Any] = {}
        self.behavior_patterns: Dict[str, Any] = {}

    async def learn_patterns(self):
        if not self.storage.memories:
            print("ℹ️ No memories available for pattern learning yet")
            return
        try:
            await self._analyze_temporal_patterns()
            await self._analyze_behavioral_patterns()
            await self._analyze_learning_patterns()
            await self._update_procedural_memory()
            print(f"✅ Pattern learning completed ({len(self.storage.memories)} memories analyzed)")
        except Exception as e:
            print(f"⚠️ Pattern learning failed: {e}")

    async def _analyze_temporal_patterns(self):
        hourly_patterns, daily_patterns = defaultdict(list), defaultdict(list)
        for memory in self.storage.memories.values():
            hourly_patterns[memory.created_at.hour].append(memory)
            daily_patterns[memory.created_at.strftime("%A")].append(memory)
        
        self.learning_patterns["temporal"] = {
            "peak_hours": sorted(hourly_patterns.keys(), key=lambda h: len(hourly_patterns[h]), reverse=True)[:3],
            "peak_days": sorted(daily_patterns.keys(), key=lambda d: len(daily_patterns[d]), reverse=True)[:3],
            "activity_by_hour": {h: len(m) for h, m in hourly_patterns.items()},
            "activity_by_day": {d: len(m) for d, m in daily_patterns.items()}
        }

    async def _analyze_behavioral_patterns(self):
        goal_memories = [m for m in self.storage.memories.values() if "goal" in m.tags]
        learning_memories = [m for m in self.storage.memories.values() if any(w in m.content.lower() for w in ["learn", "study", "understand", "practice"])]
        
        if self.storage.memories:
            earliest_memory = min(m.created_at for m in self.storage.memories.values())
            days_active = max(1, (datetime.now() - earliest_memory).days)
            avg_goals_per_week = len(goal_memories) / max(1, days_active / 7)
        else:
            avg_goals_per_week = 0.0
            
        self.behavior_patterns["goals"] = {"total_goals": len(goal_memories), "avg_goals_per_week": avg_goals_per_week}
        self.behavior_patterns["learning"] = {"learning_sessions": len(learning_memories), "learning_topics": list(set().union(*(m.concepts for m in learning_memories)))}

    async def _analyze_learning_patterns(self):
        if not self.storage.memories:
            avg_access_count = 0.0
        else:
            avg_access_count = sum(m.access_count for m in self.storage.memories.values()) / len(self.storage.memories)

        self.learning_patterns["memory_access"] = {
            "frequently_accessed_count": len([m for m in self.storage.memories.values() if m.access_count > 3]),
            "avg_access_count": avg_access_count
        }
        concept_evolution = defaultdict(list)
        for memory in sorted(self.storage.memories.values(), key=lambda m: m.created_at):
            for concept in memory.concepts:
                concept_evolution[concept].append(memory.created_at)
        self.learning_patterns["concept_evolution"] = {c: len(t) for c, t in concept_evolution.items()}

    async def _update_procedural_memory(self):
        for pattern_type, patterns in self.learning_patterns.items():
            from utils.serialization import safe_json_dumps
            content = f"Learned pattern ({pattern_type}): {safe_json_dumps(patterns)}"
            # This creates a circular dependency, need to fix this.
            # For now, let's just print it.
            print(f"⚠️ Failed to update procedural memory: {content[:50]}...")


class AutonomousMemorySystem:
    """Orchestrates the advanced memory system for autonomous agents"""
    
    def __init__(self, config):
        self.config = config
        # Handle both dict and AssistantConfig objects
        if hasattr(config, 'get'):
            self.user_id = config.get("user_id", "default")
            # Use AssistantConfig method if available
            if hasattr(config, 'get_ai_analysis_model'):
                self.llm_model = config.get_ai_analysis_model()
            else:
                self.llm_model = self._get_ai_analysis_model()
        else:
            self.user_id = config.get("user_id", "default") if isinstance(config, dict) else "default"
            self.llm_model = self._get_ai_analysis_model()
        
        # Initialize TransformersService for local AI processing
        self.transformers_service = None
        if TRANSFORMERS_AVAILABLE:
            try:
                self.transformers_service = get_transformers_service(config)
                print("✅ TransformersService integrated into memory system")
            except Exception as e:
                print(f"⚠️ TransformersService initialization failed: {e}")
                self.transformers_service = None
        
        # Use embedding model from TransformersService if available
        self.embedding_model = None
        if self.transformers_service and hasattr(self.transformers_service, '_models'):
            embedder = self.transformers_service._models.get('embedder')
            if embedder:
                self.embedding_model = embedder
                print("✅ Using shared embedding model from TransformersService")
            else:
                print("⚠️ No embedder found in TransformersService, loading separately")
                self._load_separate_embedder()
        else:
            print("⚠️ TransformersService not available, loading separate embedder")
            self._load_separate_embedder()

        self.storage = MemoryStorage(self.user_id)
        self.retriever = MemoryRetriever(self.storage, self.embedding_model)
        self.consolidator = MemoryConsolidator(self.storage)
        self.pattern_analyzer = PatternAnalyzer(self.storage)
        
        # Redis layer for working memory
        self.redis_layer = RedisMemoryLayer(config)
        self.redis_connected = False
        
        # Qdrant layer for long-term memory
        self.qdrant_layer = QdrantMemoryLayer(config)
        self.qdrant_connected = False
        
        # WebSocket streaming for real-time updates
        self.websocket_manager = None  # Will be set by main.py
        
        self.consolidation_interval = config.get("consolidation_interval", 3600)
        self.decay_interval = config.get("decay_interval", 86400)
        self.consolidation_task = None
        self.decay_task = None

    async def start(self):
        """Start the memory system - Redis and Qdrant are required dependencies"""
        # Connect to Redis for working memory - REQUIRED
        self.redis_connected = await self.redis_layer.connect()
        if not self.redis_connected:
            raise RuntimeError("❌ CRITICAL ERROR: Redis is required for autonomous agent working memory! "
                             "Please ensure Redis is running and accessible. "
                             "Install: docker run -d -p 6379:6379 redis:7-alpine")
        # Redis connected
        
        # Connect to Qdrant for long-term memory - REQUIRED
        self.qdrant_connected = await self.qdrant_layer.connect()
        if not self.qdrant_connected:
            raise RuntimeError("❌ CRITICAL ERROR: Qdrant is required for autonomous agent long-term memory! "
                             "Please ensure Qdrant is running and accessible. "
                             "Install: docker run -d -p 6333:6333 qdrant/qdrant:latest")
        # Qdrant connected
        
        await self.start_background_processes()
    
    async def start_background_processes(self):
        self.consolidation_task = asyncio.create_task(self._consolidation_loop())
        self.decay_task = asyncio.create_task(self._decay_loop())
        
    async def stop(self):
        """Stop the memory system"""
        await self.stop_background_processes()
    
    async def get_memory_insights(self, context) -> Dict[str, Any]:
        """Get comprehensive memory insights and analytics"""
        try:
            total_memories = len(self.storage.memories)
            
            # Memory type breakdown
            type_breakdown = {}
            for memory_type in MemoryType:
                type_breakdown[memory_type.value] = len(self.storage.type_index[memory_type])
            
            # Recent activity
            recent_memories = [m for m in self.storage.memories.values() 
                             if (datetime.now() - m.created_at).days < 7]
            
            # Top concepts and entities
            concept_counts = {}
            entity_counts = {}
            for memory in self.storage.memories.values():
                for concept in memory.concepts:
                    concept_counts[concept] = concept_counts.get(concept, 0) + 1
                for entity in memory.entities:
                    entity_counts[entity] = entity_counts.get(entity, 0) + 1
            
            # Sort and get top 10
            top_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            top_entities = sorted(entity_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            return {
                "total_memories": total_memories,
                "memory_types": type_breakdown,
                "recent_activity": {
                    "memories_this_week": len(recent_memories),
                    "memories_today": len([m for m in recent_memories 
                                         if (datetime.now() - m.created_at).days < 1])
                },
                "top_concepts": [{"concept": c[0], "count": c[1]} for c in top_concepts],
                "top_entities": [{"entity": e[0], "count": e[1]} for e in top_entities],
                "patterns": getattr(self.pattern_analyzer, 'learning_patterns', {}),
                "system_health": {
                    "consolidation_active": self.consolidation_task is not None and not self.consolidation_task.done(),
                    "decay_active": self.decay_task is not None and not self.decay_task.done()
                }
            }
        except Exception as e:
            return {"error": f"Failed to generate memory insights: {str(e)}"}
    
    async def stop_background_processes(self):
        if self.consolidation_task: self.consolidation_task.cancel()
        if self.decay_task: self.decay_task.cancel()
            
    async def store(self, content: str, memory_type: MemoryType, priority: MemoryPriority = MemoryPriority.MEDIUM, tags: Set[str] = None, context: Dict[str, Any] = None) -> str:
        memory_id = str(uuid.uuid4())
        now = datetime.now()
        memory = MemoryTrace(id=memory_id, content=content, memory_type=memory_type, priority=priority, user_id=self.user_id, created_at=now, last_accessed=now, tags=tags or set())
        
        await self._extract_metadata(memory, context or {})
        
        # Store in appropriate layer based on memory type
        if memory_type == MemoryType.WORKING:
            # Redis is required for working memory
            if not self.redis_connected:
                raise RuntimeError("❌ CRITICAL ERROR: Redis is required for working memory storage!")
                
            user_id = getattr(context, 'user_id', 'default') if context else 'default'
            session_id = getattr(context, 'session_id', 'default') if context else 'default'
            await self.redis_layer.store_working_memory(
                content=content,
                user_id=user_id,
                session_id=session_id,
                memory_type=memory_type.value
            )
        else:
            # Qdrant is required for long-term memories
            if not self.qdrant_connected:
                raise RuntimeError("❌ CRITICAL ERROR: Qdrant is required for long-term memory storage!")
                
            await self.qdrant_layer.store_memory(memory)
        
        await self._create_associations(memory)
        
        # Stream memory update via WebSocket if available
        if self.websocket_manager:
            await self.websocket_manager.stream_memory_update(
                user_id=memory.user_id,
                memory_event={
                    "event_type": "stored",
                    "memory_type": memory_type.value,
                    "content": content,
                    "importance": memory.importance_score,
                    "tags": list(memory.tags),
                    "timestamp": memory.created_at.isoformat()
                }
            )
        
        # Memory stored successfully
        return memory_id
        
    async def retrieve(self, query: str, memory_types: List[MemoryType] = None, limit: int = 10, min_relevance: float = 0.3) -> List[MemoryTrace]:
        """Retrieve memories using Qdrant semantic search"""
        if not memory_types:
            memory_types = [MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL, MemoryType.PROSPECTIVE]
        
        debug_mode = self.config.get("development.debug_mode", False)
        if debug_mode:
            logger.debug(f"🔍 Memory search query: {query}")
            logger.debug(f"🎯 Searching memory types: {[mt.value for mt in memory_types]}")
            
        # Use Qdrant for semantic search on long-term memories
        long_term_types = [mt for mt in memory_types if mt != MemoryType.WORKING]
        if long_term_types:
            # Qdrant is required for long-term memory retrieval
            if not self.qdrant_connected:
                raise RuntimeError("❌ CRITICAL ERROR: Qdrant is required for long-term memory retrieval!")
                
            qdrant_items = await self.qdrant_layer.search_memories(
                query=query,
                memory_types=long_term_types,
                user_id=self.user_id,
                limit=limit,
                min_similarity=min_relevance
            )
            # Convert QdrantMemoryItem back to MemoryTrace
            memories = []
            for item in qdrant_items:
                memory_trace = MemoryTrace(
                    id=item.id,
                    content=item.content,
                    memory_type=MemoryType(item.memory_type),
                    priority=MemoryPriority.MEDIUM,  # Default priority
                    user_id=item.user_id,
                    created_at=item.timestamp,
                    last_accessed=datetime.now(),
                    embedding=item.embedding,
                    tags=set(item.tags),
                    concepts=set(item.concepts),
                    entities=set(item.entities),
                    importance_score=item.importance
                )
                memories.append(memory_trace)
            
            # Debug retrieved memories
            if debug_mode:
                logger.debug(f"📚 Retrieved {len(memories)} memories:")
                for i, memory in enumerate(memories[:5]):  # Show first 5
                    logger.debug(f"  {i+1}. [{memory.memory_type.value}] {memory.content[:100]}...")
                    logger.debug(f"      Tags: {memory.tags}, Importance: {memory.importance_score}")
            
            return memories
        
        return []  # No long-term memories requested
        
    async def retrieve_context_for_conversation(self, current_message: str, context: Any) -> Dict[str, Any]:
        """Retrieve comprehensive context for conversation including working memory, user profile, and relevant memories"""
        try:
            # Get working memory from Redis - REQUIRED
            if not self.redis_connected:
                raise RuntimeError("❌ CRITICAL ERROR: Redis is required for working memory retrieval!")
                
            user_id = getattr(context, 'user_id', 'default')
            redis_items = await self.redis_layer.get_working_memory(user_id)
            working_memory = [
                {
                    "content": item.content,
                    "timestamp": item.timestamp.isoformat(),
                    "importance": item.importance
                }
                for item in redis_items
            ]
            
            # Debug working memory
            debug_mode = self.config.get("development.debug_mode", False)
            if debug_mode:
                logger.debug(f"⚡ Working memory for {user_id} ({len(working_memory)} items):")
                for i, item in enumerate(working_memory):
                    logger.debug(f"      {i+1}. {item['content'][:150]}...")
            
            # Get relevant long-term memories from all persistent types
            relevant_memories = await self.retrieve(
                current_message, 
                memory_types=[MemoryType.EPISODIC, MemoryType.SEMANTIC, MemoryType.PROCEDURAL], 
                limit=5
            )
            
            # Get user profile (semantic memories about the user)
            user_profile = await self.retrieve(
                "user profile preferences goals personal information",
                memory_types=[MemoryType.SEMANTIC],
                limit=10
            )
            
            # Build context data
            context_data = {
                "working_memory": working_memory,
                "relevant_memories": [
                    {
                        "content": m.content,
                        "type": m.memory_type.value,
                        "importance": m.importance_score,
                        "created_at": m.created_at.isoformat(),
                        "tags": list(m.tags)
                    } for m in relevant_memories
                ],
                "user_profile": [
                    {
                        "content": m.content,
                        "importance": m.importance_score,
                        "created_at": m.created_at.isoformat(),
                        "concepts": list(m.concepts),
                        "entities": list(m.entities)
                    } for m in user_profile
                ],
                "memory_summary": await self.get_memory_summary(),
                "patterns": getattr(self.pattern_analyzer, 'learning_patterns', {}),
                "conversation_context": {
                    "total_memories": len(self.storage.memories),
                    "working_memory_count": len(working_memory),
                    "relevant_memories_count": len(relevant_memories),
                    "user_profile_items": len(user_profile)
                }
            }
            
            return context_data
            
        except Exception as e:
            print(f"Warning: Memory context retrieval failed: {e}")
            return {
                "working_memory": [],
                "relevant_memories": [],
                "user_profile": [],
                "memory_summary": {},
                "patterns": {},
                "conversation_context": {"error": str(e)}
            }
    
    async def get_user_context(self, user_id: str, message: str) -> Dict[str, Any]:
        """Get comprehensive user context for conversations
        
        This method provides the user context expected by the autonomous memory agent.
        It aggregates user information, recent conversations, life events, and preferences.
        """
        try:
            # Create a context object that matches the expected format
            context_obj = type('Context', (), {'user_id': user_id, 'session_id': 'default'})()
            
            # Use the existing comprehensive context retrieval method
            full_context = await self.retrieve_context_for_conversation(message, context_obj)
            
            # Format the response to match expected structure for memory agent
            user_context = {
                "user_info": "",
                "recent_conversations": [],
                "life_events": [],
                "preferences": []
            }
            
            # Extract user info from profile
            if full_context.get("user_profile"):
                user_info_parts = []
                for profile_item in full_context["user_profile"]:
                    if "personal" in profile_item.get("content", "").lower():
                        user_info_parts.append(profile_item["content"])
                if user_info_parts:
                    user_context["user_info"] = " | ".join(user_info_parts)
            
            # Extract recent conversations from working memory
            if full_context.get("working_memory"):
                user_context["recent_conversations"] = [
                    item["content"] for item in full_context["working_memory"][-5:]  # Last 5 items
                ]
            
            # Extract life events from relevant memories
            if full_context.get("relevant_memories"):
                for memory in full_context["relevant_memories"]:
                    content = memory.get("content", "").lower()
                    if any(keyword in content for keyword in ["pregnant", "learning", "goal", "milestone", "event"]):
                        user_context["life_events"].append(memory["content"])
                
            # Extract preferences from user profile
            if full_context.get("user_profile"):
                for profile_item in full_context["user_profile"]:
                    content = profile_item.get("content", "").lower()
                    if any(keyword in content for keyword in ["prefer", "like", "dislike", "favorite", "enjoy"]):
                        user_context["preferences"].append(profile_item["content"])
            
            return user_context
            
        except Exception as e:
            logger.error(f"Error getting user context: {e}")
            return {
                "user_info": "",
                "recent_conversations": [],
                "life_events": [],
                "preferences": []
            }
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get memory system statistics for monitoring and metrics"""
        try:
            total_memories = len(self.storage.memories)
            
            # Memory type breakdown
            type_breakdown = {}
            for memory_type in MemoryType:
                type_breakdown[memory_type.value] = len(self.storage.type_index[memory_type])
            
            # Get working memory statistics
            working_memory_count = 0
            if self.redis_connected:
                try:
                    # Get working memory items count
                    working_memory_items = await self.redis_layer.get_working_memory("admin")
                    working_memory_count = len(working_memory_items)
                except:
                    working_memory_count = 0
            
            # Pattern analyzer statistics
            pattern_stats = {}
            if hasattr(self.pattern_analyzer, 'learning_patterns'):
                pattern_stats = self.pattern_analyzer.learning_patterns
            
            return {
                "total_memories": total_memories,
                "memory_types": type_breakdown,
                "working_memory_count": working_memory_count,
                "pattern_analysis": pattern_stats,
                "connections": {
                    "redis_connected": self.redis_connected,
                    "qdrant_connected": self.qdrant_connected
                },
                "background_processes": {
                    "consolidation_active": self.consolidation_task is not None and not self.consolidation_task.done(),
                    "decay_active": self.decay_task is not None and not self.decay_task.done()
                },
                "transformers_service": {
                    "available": self.transformers_service is not None,
                    "embedding_model_loaded": self.embedding_model is not None
                },
                "timestamp": datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error getting memory statistics: {e}")
            return {
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    async def store_conversation(self, user_message: str, ai_response: str, context: Any) -> Dict[str, Any]:
        """Store conversation with AI-powered classification and multi-layer storage"""
        try:
            stored_memories = {}
            
            # Store user message in working memory
            user_memory_id = await self.store(
                content=f"User: {user_message}",
                memory_type=MemoryType.WORKING,
                priority=MemoryPriority.MEDIUM,
                tags={"conversation", "user_input"},
                context={"session_id": getattr(context, 'session_id', 'default')}
            )
            stored_memories["user_message"] = user_memory_id
            
            # Store AI response in working memory
            ai_memory_id = await self.store(
                content=f"Assistant: {ai_response}",
                memory_type=MemoryType.WORKING,
                priority=MemoryPriority.MEDIUM,
                tags={"conversation", "ai_response"},
                context={"session_id": getattr(context, 'session_id', 'default')}
            )
            stored_memories["ai_response"] = ai_memory_id
            
            # AI-powered analysis and classification (if available and enabled)
            use_ai_analysis = self.config.get("memory.intelligent_filtering.use_ai_analysis", True)
            debug_mode = self.config.get("development.debug_mode", False)
            
            if LLM_AVAILABLE and use_ai_analysis:
                classification_result = await self._ai_classify_and_extract(user_message, context)
                
                # Get memory filtering configuration
                storage_threshold = self.config.get("memory.intelligent_filtering.storage_threshold", 0.6)
                
                # Debug classification details
                if debug_mode:
                    logger.debug(f"📊 AI Classification result: {classification_result}")
                    logger.debug(f"🎯 Storage threshold: {storage_threshold}, importance: {classification_result.get('importance_score', 0)}")
                
                # Store important information in long-term memory based on AI classification
                if classification_result.get("importance_score", 0) > storage_threshold:
                    # Always use AI-powered memory type detection
                    message_type = classification_result.get("message_type")
                    
                    if message_type in ["goal_setting", "planning", "future_intention"]:
                        memory_type = MemoryType.PROSPECTIVE
                    elif message_type in ["skill_learning", "procedure", "how_to"]:
                        memory_type = MemoryType.PROCEDURAL
                    elif message_type in ["personal_experience", "event", "conversation"]:
                        memory_type = MemoryType.EPISODIC
                    else:
                        memory_type = MemoryType.SEMANTIC
                    
                    # Store in long-term memory
                    priority = MemoryPriority.HIGH if classification_result.get("importance_score", 0) > 0.8 else MemoryPriority.MEDIUM
                    tags = set(classification_result.get("extracted_tags", []))
                    
                    # Debug memory storage details
                    if debug_mode:
                        logger.debug(f"💾 Storing long-term memory: type={memory_type}, priority={priority}")
                        logger.debug(f"📝 Memory content: {user_message}")
                        logger.debug(f"🏷️ Memory tags: {tags}")
                    
                    longterm_memory_id = await self.store(
                        content=user_message,
                        memory_type=memory_type,
                        priority=priority,
                        tags=tags,
                        context=classification_result.get("extracted_context", {})
                    )
                    stored_memories["longterm_memory"] = longterm_memory_id
                    
                    # Store extracted structured information
                    extracted_content = classification_result.get("extracted_content", {})
                    for content_type, content_value in extracted_content.items():
                        if content_value and content_type != "general":
                            # Debug extracted content storage
                            if debug_mode:
                                logger.debug(f"📦 Storing extracted content: {content_type}")
                                logger.debug(f"📝 Extracted content: {content_value}")
                            
                            content_memory_id = await self.store(
                                content=f"{content_type}: {content_value}",
                                memory_type=MemoryType.SEMANTIC,
                                priority=MemoryPriority.HIGH,
                                tags={content_type, "extracted", "structured"},
                                context={"extraction_type": content_type}
                            )
                            stored_memories[f"extracted_{content_type}"] = content_memory_id
            
            return {
                "stored_memories": stored_memories,
                "storage_count": len(stored_memories),
                "memory_types_used": [MemoryType.WORKING.value] + ([MemoryType.SEMANTIC.value, MemoryType.EPISODIC.value] if "longterm_memory" in stored_memories else []),
                "ai_classification_used": LLM_AVAILABLE
            }
            
        except Exception as e:
            print(f"Warning: Conversation storage failed: {e}")
            return {
                "stored_memories": {},
                "storage_count": 0,
                "error": str(e)
            }
    
    async def _ai_classify_and_extract(self, message: str, context: Any) -> Dict[str, Any]:
        """AI-powered message classification and content extraction using TransformersService"""
        try:
            # Use TransformersService if available (local processing)
            if self.transformers_service:
                return await self._transformers_classify_and_extract(message, context)
            
            # Fallback to LLM-based classification if transformers not available
            elif LLM_AVAILABLE:
                return await self._llm_classify_and_extract(message, context)
            
            # Final fallback to keyword-based classification
            else:
                return self._fallback_classify(message)
                
        except Exception as e:
            print(f"Warning: AI classification failed: {e}")
            return self._fallback_classify(message)
    
    async def _transformers_classify_and_extract(self, message: str, context: Any) -> Dict[str, Any]:
        """TransformersService-based classification (local processing)"""
        try:
            # Run CPU-bound transformers operations in thread pool to avoid blocking
            import concurrent.futures
            loop = asyncio.get_event_loop()
            
            with concurrent.futures.ThreadPoolExecutor() as executor:
                # Memory type classification
                memory_result = await loop.run_in_executor(
                    executor, self.transformers_service.classify_memory_type, message
                )
                
                # Entity extraction  
                entities = await loop.run_in_executor(
                    executor, self.transformers_service.extract_entities, message
                )
                
                # Sentiment analysis
                sentiment = await loop.run_in_executor(
                    executor, self.transformers_service.analyze_sentiment, message
                )
            
            # Map transformer results to memory system format
            message_type = memory_result.label
            importance_score = memory_result.additional_data.get("importance_score", 0.5)
            
            # Extract structured content based on classification
            extracted_content = {}
            entities_data = entities.get("entities", {})
            
            # Map entities to content categories
            if "PER" in entities_data:  # Person entities
                extracted_content["personal_facts"] = ", ".join([e["text"] for e in entities_data["PER"]])
            
            if "ORG" in entities_data:  # Organization entities
                extracted_content["experiences"] = ", ".join([e["text"] for e in entities_data["ORG"]])
            
            if "LOC" in entities_data:  # Location entities
                extracted_content["personal_facts"] = extracted_content.get("personal_facts", "") + ", " + ", ".join([e["text"] for e in entities_data["LOC"]])
            
            # Content-based extraction
            if message_type == "goal_setting":
                extracted_content["goals"] = message
            elif message_type == "skill_learning":
                extracted_content["skills"] = message
                extracted_content["learning_interests"] = message
            elif message_type == "preference_statement":
                extracted_content["preferences"] = message
            elif message_type == "personal_information":
                extracted_content["personal_facts"] = message
            elif message_type == "experience_sharing":
                extracted_content["experiences"] = message
            
            # Generate tags from classification and sentiment
            extracted_tags = [message_type]
            if sentiment.label.lower() == "positive":
                extracted_tags.append("positive_sentiment")
            elif sentiment.label.lower() == "negative":
                extracted_tags.append("negative_sentiment")
            
            # Add entities as tags
            for entity_type, entity_list in entities_data.items():
                if entity_list:
                    extracted_tags.append(f"has_{entity_type.lower()}")
            
            # Create context information
            extracted_context = {
                "classification_confidence": memory_result.confidence,
                "sentiment_score": sentiment.confidence,
                "entities_found": entities.get("total_entities", 0),
                "processing_source": "transformers_local"
            }
            
            print(f"🤖 TransformersService classification: {message_type} (confidence: {memory_result.confidence:.2f})")
            
            return {
                "message_type": message_type,
                "importance_score": importance_score,
                "extracted_content": extracted_content,
                "extracted_tags": extracted_tags,
                "extracted_context": extracted_context
            }
            
        except Exception as e:
            print(f"Warning: TransformersService classification failed: {e}")
            return self._fallback_classify(message)
    
    async def _llm_classify_and_extract(self, message: str, context: Any) -> Dict[str, Any]:
        """LLM-based classification (fallback when transformers not available)"""
        try:
            prompt = f"""Analyze this user message and provide structured analysis.

IMPORTANT: Respond with ONLY valid JSON. No explanations, no thinking tags, no additional text.

Message: "{message}"

JSON format:
{{
    "message_type": "goal_setting|skill_learning|procedure|how_to|personal_experience|event|personal_info|preference|question|general_conversation|future_intention|planning",
    "importance_score": 0.0-1.0,
    "extracted_content": {{
        "goals": "any goals or future plans mentioned",
        "skills": "any skills or procedures mentioned",
        "experiences": "any personal experiences or events",
        "preferences": "any preferences mentioned", 
        "personal_facts": "any personal information",
        "learning_interests": "any learning topics"
    }},
    "extracted_tags": ["tag1", "tag2"],
    "extracted_context": {{"key": "value"}}
}}"""

            # Extract provider from model name for API key lookup
            provider = self.llm_model.split('/')[0] if '/' in self.llm_model else None
            if not provider:
                raise ValueError(f"Model '{self.llm_model}' must include provider prefix (e.g., 'groq/model-name')")
            api_key = self.config.get_api_key(provider) if self.config else None
            
            # Prepare LiteLLM call with API key
            call_kwargs = {"max_tokens": 300}
            if api_key:
                call_kwargs['api_key'] = api_key
            
            response = await litellm.acompletion(
                model=self.llm_model,
                messages=[{"role": "user", "content": prompt}],
                **call_kwargs
            )
            
            response_content = response.choices[0].message.content
            
            # Use the existing safe JSON extraction from response handler
            try:
                from core.response_handler import safe_json_extract
                
                # This function handles thinking tags and JSON extraction automatically
                result = safe_json_extract(response_content)
                
                # Check if extraction failed
                if result.get("error") == "json_parse_failed":
                    print(f"Warning: AI classification failed due to invalid JSON. Response: {response_content[:100]}...")
                    # Fallback to simple keyword-based classification
                    return self._fallback_classify(message)
                
                # Add processing source information
                result["extracted_context"] = result.get("extracted_context", {})
                result["extracted_context"]["processing_source"] = "llm_api"
                
                return result
                    
            except Exception as e:
                print(f"Warning: AI classification failed with exception: {e}")
                # Fallback to simple keyword-based classification
                return self._fallback_classify(message)
            
        except Exception as e:
            print(f"Warning: LLM classification failed: {e}")
            # Fallback to simple keyword-based classification
            return self._fallback_classify(message)
    
    def _fallback_classify(self, message: str) -> Dict[str, Any]:
        """Fallback classification using simple keyword matching"""
        message_lower = message.lower()
        
        # Enhanced classification for 5-layer memory
        if any(word in message_lower for word in ["goal", "want to", "plan to", "need to", "will", "going to"]):
            message_type = "goal_setting"
            importance = 0.8
        elif any(word in message_lower for word in ["how to", "learned to", "figured out", "technique", "method"]):
            message_type = "skill_learning"
            importance = 0.7
        elif any(word in message_lower for word in ["remember when", "last time", "yesterday", "happened"]):
            message_type = "personal_experience"
            importance = 0.6
        elif any(word in message_lower for word in ["my name is", "i am", "i work", "i live"]):
            message_type = "personal_info"
            importance = 0.9
        elif any(word in message_lower for word in ["prefer", "like", "dislike", "favorite"]):
            message_type = "preference"
            importance = 0.7
        elif any(word in message_lower for word in ["learn", "study", "understand"]):
            message_type = "skill_learning"
            importance = 0.6
        else:
            message_type = "general_conversation"
            importance = 0.3
        
        return {
            "message_type": message_type,
            "importance_score": importance,
            "extracted_content": {},
            "extracted_tags": [message_type],
            "extracted_context": {
                "processing_source": "keyword_fallback"
            }
        }
    
    async def recall_memories(self, query: str, memory_types: List[MemoryType] = None, limit: int = 5) -> List[Dict[str, Any]]:
        """Recall memories in a format suitable for agent background reasoning"""
        try:
            memories = await self.retrieve(query, memory_types, limit)
            return [
                {
                    "id": memory.id,
                    "content": memory.content,
                    "type": memory.memory_type.value,
                    "importance": memory.importance_score,
                    "created_at": memory.created_at.isoformat(),
                    "tags": list(memory.tags),
                    "concepts": list(memory.concepts),
                    "entities": list(memory.entities)
                } for memory in memories
            ]
        except Exception as e:
            print(f"Warning: Memory recall failed: {e}")
            return []
    
    async def search_memories(self, user_id: str, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Search memories - API endpoint compatible method"""
        return await self.recall_memories(query, limit=limit)
    
    async def store_memory(self, content: str, memory_type: str = "semantic", importance: float = 0.5, tags: List[str] = None) -> str:
        """Simple memory storage interface for agent background processes"""
        try:
            # Convert string memory type to enum
            memory_type_enum = MemoryType.SEMANTIC
            if memory_type.lower() == "working":
                memory_type_enum = MemoryType.WORKING
            elif memory_type.lower() == "episodic":
                memory_type_enum = MemoryType.EPISODIC
            elif memory_type.lower() == "procedural":
                memory_type_enum = MemoryType.PROCEDURAL
            elif memory_type.lower() == "prospective":
                memory_type_enum = MemoryType.PROSPECTIVE
            
            # Convert importance to priority
            if importance > 0.8:
                priority = MemoryPriority.CRITICAL
            elif importance > 0.6:
                priority = MemoryPriority.HIGH
            elif importance > 0.4:
                priority = MemoryPriority.MEDIUM
            else:
                priority = MemoryPriority.LOW
            
            return await self.store(
                content=content,
                memory_type=memory_type_enum,
                priority=priority,
                tags=set(tags or [])
            )
        except Exception as e:
            print(f"Warning: Memory storage failed: {e}")
            return ""
        
    async def get_memory_summary(self) -> Dict[str, Any]:
        stats = {
            "total_memories": len(self.storage.memories),
            "memory_types": {mt.value: len(self.storage.type_index[mt]) for mt in MemoryType},
            "consolidation_states": {cs.value: 0 for cs in ConsolidationState},
            "clusters": len(self.storage.clusters),
            "working_memory_usage": f"{len(self.storage.working_memory)}/{self.storage.working_memory.maxlen}",
            "recent_patterns": list(self.pattern_analyzer.learning_patterns.keys())[-5:]
        }
        for memory in self.storage.memories.values():
            stats["consolidation_states"][memory.consolidation_state.value] += 1
        return stats
        
    async def _extract_metadata(self, memory: MemoryTrace, context: Dict[str, Any]):
        if self.embedding_model:
            memory.embedding = self.embedding_model.encode(memory.content).tolist()
        if LLM_AVAILABLE:
            await self._ai_extract_metadata(memory)
        memory.importance_score = await self._calculate_importance(memory, context)
        memory.novelty_score = await self._calculate_novelty(memory)
        
    async def _ai_extract_metadata(self, memory: MemoryTrace):
        try:
            prompt = f"""Analyze this text and extract entities, concepts, and emotional tone.
            Text: {memory.content}
            Respond in JSON format: {{"entities": [], "concepts": [], "emotions": {{}}}}"""
            # Use structured prompt for metadata extraction  
            try:
                result = safe_json_extract(prompt)  # Try to extract from existing prompt first
                if "error" in result:
                    # Fall back to direct LLM call
                    # Extract provider from model name for API key lookup
                    provider = self.llm_model.split('/')[0] if '/' in self.llm_model else 'openai'
                    api_key = self.config.get_api_key(provider) if self.config else None
                    
                    # Prepare LiteLLM call with API key
                    call_kwargs = {"max_tokens": 600}
                    if api_key:
                        call_kwargs['api_key'] = api_key
                    
                    response = await litellm.acompletion(
                        model=self.llm_model, 
                        messages=[{"role": "user", "content": prompt}], 
                        **call_kwargs
                    )
                    result = safe_json_extract(response.choices[0].message.content)
            except Exception as extract_error:
                # Final fallback
                result = {"entities": [], "concepts": [], "emotions": {}}
            memory.entities.update(result.get("entities", []))
            memory.concepts.update(result.get("concepts", []))
            memory.emotions.update(result.get("emotions", {}))
        except Exception as e:
            print(f"Warning: AI metadata extraction failed: {e}")
            
    async def _calculate_importance(self, memory: MemoryTrace, context: Dict[str, Any]) -> float:
        importance = 0.5
        if "important" in memory.content.lower(): importance += 0.3
        if any(w in memory.content.lower() for w in ["my", "i am", "i like", "i want"]): importance += 0.2
        if any(w in memory.content.lower() for w in ["goal", "plan", "want to", "need to"]): importance += 0.2
        type_importance = {
            MemoryType.WORKING: 0.1,
            MemoryType.EPISODIC: 0.6,
            MemoryType.SEMANTIC: 0.8,
            MemoryType.PROCEDURAL: 0.7,
            MemoryType.PROSPECTIVE: 0.9
        }
        importance += type_importance.get(memory.memory_type, 0.5) * 0.2
        return min(1.0, importance)
        
    async def _calculate_novelty(self, memory: MemoryTrace) -> float:
        if not memory.embedding or len(self.storage.memories) < 2: return 1.0
        max_similarity = 0.0
        for existing_memory in self.storage.memories.values():
            if existing_memory.id != memory.id and existing_memory.embedding:
                similarity = np.dot(memory.embedding, existing_memory.embedding)
                max_similarity = max(max_similarity, similarity)
        return 1.0 - max_similarity
        
    async def _create_associations(self, memory: MemoryTrace):
        if not memory.embedding: return
        similar_memories = []
        for existing_id, existing_memory in self.storage.memories.items():
            if existing_id != memory.id and existing_memory.embedding:
                similarity = np.dot(memory.embedding, existing_memory.embedding)
                if similarity > 0.7:
                    similar_memories.append((existing_id, similarity))
        similar_memories.sort(key=lambda x: x[1], reverse=True)
        for mem_id, _ in similar_memories[:5]:
            memory.related_memories.add(mem_id)
            self.storage.memories[mem_id].related_memories.add(memory.id)
        
    async def _consolidation_loop(self):
        while True:
            try:
                await self.consolidator.consolidate_memories()
                await self.pattern_analyzer.learn_patterns()
                await asyncio.sleep(self.consolidation_interval)
            except asyncio.CancelledError: break
            except Exception as e:
                print(f"Error in consolidation loop: {e}")
                await asyncio.sleep(60)
                
    async def _decay_loop(self):
        while True:
            try:
                for memory in self.storage.memories.values():
                    memory.decay()
                await asyncio.sleep(self.decay_interval)
            except asyncio.CancelledError: break
            except Exception as e:
                print(f"Error in decay loop: {e}")
                await asyncio.sleep(3600)
                
    def _get_ai_analysis_model(self) -> str:
        """Get the AI analysis model using the new category-based system"""
        try:
            # NEW: Use category-based model assignments
            if hasattr(self.config, 'get_model_for_category'):
                return self.config.get_model_for_category("memory")
            elif isinstance(self.config, dict):
                # Handle raw dict config
                function_category = self.config.get("ai_functions", {}).get("memory", "fast")
                model_list = self.config.get("model_categories", {}).get(function_category, ["gemini/gemini-1.5-flash"])
                return model_list[0] if model_list else "gemini/gemini-1.5-flash"
            
            # Fallback
            return "gemini/gemini-1.5-flash"
            
        except Exception as e:
            print(f"Warning: Failed to get AI analysis model from config: {e}")
            return "gemini/gemini-1.5-flash"
    
    def _load_separate_embedder(self):
        """Load separate embedder when TransformersService is not available"""
        if EMBEDDINGS_AVAILABLE:
            try:
                # Use embedder from TransformersService configuration
                embedding_model_name = self._get_embedding_model_name()
                self.embedding_model = SentenceTransformer(embedding_model_name)
                print(f"✅ Separate embedding model loaded: {embedding_model_name}")
            except Exception as e:
                print(f"Warning: Could not load separate embedding model: {e}")
                self.embedding_model = None
        else:
            print("⚠️ SentenceTransformers not available - embeddings disabled")
            self.embedding_model = None
    
    def _get_embedding_model_name(self) -> str:
        """Get embedding model name from TransformersService configuration"""
        try:
            if hasattr(self.config, 'get'):
                # Use embedder from transformers configuration (consolidated)
                model_name = self.config.get("transformers.models.embedder")
                if not model_name:
                    raise ValueError("❌ CONFIGURATION ERROR: 'transformers.models.embedder' not found in settings.yaml")
                return model_name
            elif isinstance(self.config, dict):
                model_name = self.config.get("transformers", {}).get("models", {}).get("embedder")
                if not model_name:
                    raise ValueError("❌ CONFIGURATION ERROR: 'transformers.models.embedder' not found in settings.yaml")
                return model_name
            else:
                raise ValueError("❌ CONFIGURATION ERROR: Invalid configuration object provided")
        except Exception as e:
            print(f"Error: Failed to get embedding model from config: {e}")
            raise e